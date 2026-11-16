import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor tailored for MiniGrid visual observations (7x7x3).
    Follows the architecture from the paper:
      - Conv2d(n_channels, 16, kernel_size=3, padding=1) + ReLU
      - Conv2d(16, 32, kernel_size=3, padding=1) + ReLU
      - Flatten
      - Linear(n_flatten, features_dim) + ReLU
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample_tensor = th.as_tensor(observation_space.sample()[None]).float()
            permuted_sample = sample_tensor.permute(0, 3, 1, 2)
            n_flatten = self.cnn(permuted_sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # MiniGrid observations are (Batch, Height, Width, Channels) of uint8 categorical IDs.
        # Ensure float32 representation without dividing by 255.
        if observations.dtype != th.float32:
            observations = observations.float()
        # If an external wrapper pre-normalized to [0, 1], restore categorical integer scale
        if observations.max() <= 1.0 and observations.max() > 0:
            observations = observations * 255.0

        # Permute from (Batch, Height, Width, Channels) to PyTorch CNN standard: (Batch, Channels, Height, Width)
        permuted_observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(permuted_observations))


class MinigridSemanticSlotExtractor(BaseFeaturesExtractor):
    """
    Lightweight Semantic Slot Extractor with Relational Match Comparator for MiniGrid-PutNear.
    Keeps policy parameter size comparable to standard vision-only CNN (~158k params):
      1. Image CNN: Conv2d(3, 16) -> ReLU -> Conv2d(16, 32) -> ReLU -> Flatten (1568) -> Linear(1568, 96) -> ReLU.
      2. Slot Embeddings:
         - Color Embedding: nn.Embedding(8, 8)
         - Type Embedding: nn.Embedding(12, 8)
         - Carrying Embedding: nn.Embedding(2, 8)
      3. Relational Match Comparator (8-dim relational vector):
         - is_facing_move: 1-bit float indicator whether cell (3, 5) matches (move_color, move_type)
         - is_facing_target: 1-bit float indicator whether cell (3, 5) matches (target_color, target_type)
         - is_facing_wrong: 1-bit float indicator whether cell (3, 5) is an interactive object (key/ball/box) that is NOT the current goal
         - is_facing_wall: 1-bit float indicator whether cell (3, 5) is a wall
         - is_facing_empty: 1-bit float indicator whether cell (3, 5) is empty / floor
         - is_valid_drop: 1-bit float indicator: carrying move object + front cell empty + target object in 8-nbr of front cell (ready to drop!)
         - is_target_adjacent: 1-bit float indicator: carrying move object + target object in 5-nbr of agent (reached goal vicinity!)
         - is_carrying: 1-bit float indicator: 1.0 if agent holds move object in inventory, 0.0 if empty-handed
      4. Slot Linear: Linear(40 + 8, 24) -> ReLU.
      5. Fusion: Concatenate visual features (96-d) + slot features (24-d) + direct comparator bits (8-d) = 128-d output!
         Provides the policy action head with a direct linear skip connection to relational truth.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        img_space = observation_space["image"]
        # MiniGrid image observations always consist of 3 channels: [type, color, state]
        n_input_channels = 3

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            dummy_img = th.zeros((1, n_input_channels, 7, 7))
            n_flatten = self.cnn(dummy_img).shape[1]

        self.comp_dim = 8
        slot_dim = 24
        vis_dim = max(16, features_dim - slot_dim - self.comp_dim)
        self.vis_linear = nn.Sequential(
            nn.Linear(n_flatten, vis_dim),
            nn.ReLU()
        )

        self.color_embed = nn.Embedding(8, 8)
        self.type_embed = nn.Embedding(12, 8)

        # Check if 1-bit is_carrying flag is present in mission space (shape (5,))
        mission_dim = observation_space["mission"].shape[0] if "mission" in observation_space.spaces else 4
        self.has_carrying_slot = (mission_dim >= 5)
        if self.has_carrying_slot:
            self.carrying_embed = nn.Embedding(2, 8)
            in_slot_dim = 40 + self.comp_dim  # 4 * 8 + 8 + 8 = 48
        else:
            in_slot_dim = 32 + self.comp_dim  # 4 * 8 + 8 = 40

        self.slot_linear = nn.Sequential(
            nn.Linear(in_slot_dim, slot_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        raw_img = observations["image"].float()
        if raw_img.max() <= 1.0 and raw_img.max() > 0:
            raw_img = raw_img * 255.0

        is_channel_last = (raw_img.shape[-1] == 3)

        # Helper to query categorical type and color at egocentric coordinate (x, y)
        def get_cell(x: int, y: int):
            if raw_img.ndim == 4:
                if is_channel_last:  # (Batch, 7, 7, 3)
                    t = raw_img[:, x, y, 0].round().long()
                    c = raw_img[:, x, y, 1].round().long()
                else:  # (Batch, 3, 7, 7)
                    t = raw_img[:, 0, x, y].round().long()
                    c = raw_img[:, 1, x, y].round().long()
            elif raw_img.ndim == 3:
                if is_channel_last:  # (7, 7, 3)
                    t = raw_img[x, y, 0].round().long().unsqueeze(0)
                    c = raw_img[x, y, 1].round().long().unsqueeze(0)
                else:  # (3, 7, 7)
                    t = raw_img[0, x, y].round().long().unsqueeze(0)
                    c = raw_img[1, x, y].round().long().unsqueeze(0)
            else:
                raise ValueError(f"Unexpected image shape {raw_img.shape}")
            return t, c

        # Extract front cell (3, 5) categorical type and color from egocentric view
        front_type, front_color = get_cell(3, 5)

        # Permute for standard PyTorch Conv2d: (Batch, Channels, Height, Width)
        if raw_img.ndim == 4 and is_channel_last:
            img = raw_img.permute(0, 3, 1, 2)
        elif raw_img.ndim == 3 and is_channel_last:
            img = raw_img.permute(2, 0, 1).unsqueeze(0)
        elif raw_img.ndim == 3 and not is_channel_last:
            img = raw_img.unsqueeze(0)
        else:
            img = raw_img
        v = self.vis_linear(self.cnn(img))

        slots = observations["mission"].long()
        if slots.ndim == 1:
            slots = slots.unsqueeze(0)

        m_c = self.color_embed(slots[:, 0])
        m_t = self.type_embed(slots[:, 1])
        t_c = self.color_embed(slots[:, 2])
        t_t = self.type_embed(slots[:, 3])

        target_color = slots[:, 2]
        target_type = slots[:, 3]

        # Extract carrying status (0 = empty-handed, 1 = carrying move object)
        if self.has_carrying_slot and slots.shape[1] >= 5:
            is_carrying = slots[:, 4].clamp(0, 1)
        else:
            is_carrying = th.zeros_like(slots[:, 0])
        is_carrying_bool = (is_carrying > 0)

        # Compute Front-Cell Semantic Match Comparator features
        is_facing_move = ((front_color == slots[:, 0]) & (front_type == slots[:, 1])).float().unsqueeze(1)
        is_facing_target = ((front_color == slots[:, 2]) & (front_type == slots[:, 3])).float().unsqueeze(1)
        # Interactive objects in MiniGrid: key=5, ball=6, box=7
        is_interactive = ((front_type >= 5) & (front_type <= 7)).unsqueeze(1)

        # Phase-aware wrong object indicator:
        # - Phase 1 (empty hands): both distractor and target object are wrong to pick up
        # - Phase 2 (carrying move object): ONLY distractor is wrong; target object is the goal!
        is_carrying_col = is_carrying_bool.unsqueeze(1)
        wrong_p1 = is_interactive & (is_facing_move == 0)
        wrong_p2 = is_interactive & (is_facing_target == 0) & (is_facing_move == 0)
        is_facing_wrong = th.where(is_carrying_col, wrong_p2, wrong_p1).float()

        is_facing_wall = (front_type == 2).float().unsqueeze(1)
        is_facing_empty = ((front_type == 1) | (front_type == 3)).float().unsqueeze(1)

        # Helper to check if Target Object is present at cell (x, y)
        def is_target_at(x: int, y: int) -> th.Tensor:
            t, c = get_cell(x, y)
            return (t == target_type) & (c == target_color)

        # 7 candidate neighbors around front cell (3, 5) for valid drop adjacency
        f_nbrs = [
            is_target_at(2, 4), is_target_at(3, 4), is_target_at(4, 4),
            is_target_at(2, 5),                     is_target_at(4, 5),
            is_target_at(2, 6),                     is_target_at(4, 6),
        ]
        target_near_front = f_nbrs[0]
        for n in f_nbrs[1:]:
            target_near_front = target_near_front | n

        front_empty_bool = (front_type == 1) | (front_type == 3)
        is_valid_drop = (is_carrying_bool & front_empty_bool & target_near_front).float().unsqueeze(1)

        # 5 candidate neighbors around agent (3, 6) for arrival adjacency
        a_nbrs = [
            is_target_at(2, 5), is_target_at(3, 5), is_target_at(4, 5),
            is_target_at(2, 6),                     is_target_at(4, 6),
        ]
        target_near_agent = a_nbrs[0]
        for n in a_nbrs[1:]:
            target_near_agent = target_near_agent | n

        is_target_adjacent = (is_carrying_bool & target_near_agent).float().unsqueeze(1)

        comp = th.cat([
            is_facing_move,
            is_facing_target,
            is_facing_wrong,
            is_facing_wall,
            is_facing_empty,
            is_valid_drop,
            is_target_adjacent,
            is_carrying_col.float()
        ], dim=1)

        if self.has_carrying_slot and slots.shape[1] >= 5:
            c_emb = self.carrying_embed(is_carrying)
            slot_cat = th.cat([m_c, m_t, t_c, t_t, c_emb, comp], dim=1)
        else:
            slot_cat = th.cat([m_c, m_t, t_c, t_t, comp], dim=1)

        s = self.slot_linear(slot_cat)
        # Direct bypass connection: concatenate visual + slot + direct comparator bits
        return th.cat([v, s, comp], dim=1)


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) block for MiniGrid.
    Applies affine channel-wise modulation to convolutional feature maps based on language embedding:
        h = (1.0 + gamma) * conv(x) + beta
    Initialized to identity (gamma=0, beta=0) for smooth RL start.
    """
    def __init__(self, in_channels: int, out_channels: int, text_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.film_gen = nn.Linear(text_dim, 2 * out_channels)
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)

    def forward(self, x: th.Tensor, text_emb: th.Tensor) -> th.Tensor:
        h = self.conv(x)
        film = self.film_gen(text_emb)
        gamma, beta = th.chunk(film, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = (1.0 + gamma) * h + beta
        return F.relu(h)


class MinigridMultimodalFiLMExtractor(BaseFeaturesExtractor):
    """
    State-of-the-Art Multimodal Feature Extractor for MiniGrid instruction following.
    Combines:
      1. Bidirectional GRU text encoder over tokenized mission strings.
      2. 2-layer CNN with FiLM (Feature-wise Linear Modulation) conditioning.
      3. Gated Residual Multimodal Fusion between visual and linguistic streams.
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
        vocab_size: int = 20,
        embed_dim: int = 32,
        text_dim: int = 64
    ):
        super().__init__(observation_space, features_dim)
        img_space = observation_space["image"]
        n_input_channels = img_space.shape[2]

        # 1. Linguistic Encoder
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, text_dim // 2, batch_first=True, bidirectional=True)

        # 2. FiLM Visual Backbone
        self.block1 = FiLMBlock(n_input_channels, 32, text_dim)
        self.block2 = FiLMBlock(32, 64, text_dim)

        # Dynamically determine flattened dimension
        with th.no_grad():
            dummy_img = th.zeros((1, n_input_channels, img_space.shape[0], img_space.shape[1]))
            dummy_text = th.zeros((1, text_dim))
            dummy_h = self.block2(self.block1(dummy_img, dummy_text), dummy_text)
            n_flatten = dummy_h.flatten(1).shape[1]

        self.vis_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, 128),
            nn.ReLU()
        )

        # 3. Gated Residual Multimodal Fusion
        self.proj_text = nn.Linear(text_dim, 128)
        self.gate = nn.Sequential(
            nn.Linear(128 + text_dim, 128),
            nn.Sigmoid()
        )
        self.head = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        # 1. Encode Mission text
        text = observations["mission"].long()
        emb = self.emb(text)
        _, h_n = self.gru(emb)
        # Bidirectional GRU output: shape (batch, text_dim)
        text_emb = th.cat([h_n[0], h_n[1]], dim=1)

        # 2. Encode Visual grid with FiLM modulation
        img = observations["image"].float()
        if img.max() <= 1.0 and img.max() > 0:
            img = img * 255.0
        # Permute (Batch, H, W, C) -> (Batch, C, H, W)
        img = img.permute(0, 3, 1, 2)
        h1 = self.block1(img, text_emb)
        h2 = self.block2(h1, text_emb)
        v = self.vis_fc(h2)

        # 3. Gated Multimodal Fusion
        t_proj = self.proj_text(text_emb)
        g = self.gate(th.cat([v, text_emb], dim=1))
        fused = g * v + (1.0 - g) * t_proj
        return self.head(fused)


class MinigridMultimodalConcatExtractor(BaseFeaturesExtractor):
    """
    Multimodal Concatenation Feature Extractor (Baseline ablation).
    Extracts visual features via standard CNN and linguistic features via GRU,
    then concatenates them into an MLP head.
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
        vocab_size: int = 20,
        embed_dim: int = 32,
        text_dim: int = 64
    ):
        super().__init__(observation_space, features_dim)
        img_space = observation_space["image"]
        n_input_channels = img_space.shape[2]

        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, text_dim // 2, batch_first=True, bidirectional=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            dummy_img = th.zeros((1, n_input_channels, img_space.shape[0], img_space.shape[1]))
            n_flatten = self.cnn(dummy_img).shape[1]

        self.vis_fc = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(128 + text_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        text = observations["mission"].long()
        emb = self.emb(text)
        _, h_n = self.gru(emb)
        text_emb = th.cat([h_n[0], h_n[1]], dim=1)

        img = observations["image"].float()
        if img.max() <= 1.0 and img.max() > 0:
            img = img * 255.0
        img = img.permute(0, 3, 1, 2)
        v = self.vis_fc(self.cnn(img))

        concat = th.cat([v, text_emb], dim=1)
        return self.head(concat)
