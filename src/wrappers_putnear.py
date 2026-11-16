import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
import numpy as np
from typing import Dict, Any, Optional, Tuple, Set

from src.rm_model import ActionLevelRewardModel
from src.prompt_utils_putnear import (
    extract_putnear_state_and_action_desc,
    get_putnear_objects
)


class PutNearTokenizer:
    """
    Tokenizer for PutNear mission strings:
    e.g. "put the red ball near the blue box"
    Fixed length: 8 tokens.
    """
    VOCAB = {
        "<pad>": 0,
        "<unk>": 1,
        "put": 2,
        "the": 3,
        "near": 4,
        "next": 5,
        "to": 6,
        "red": 7,
        "green": 8,
        "blue": 9,
        "purple": 10,
        "yellow": 11,
        "grey": 12,
        "key": 13,
        "ball": 14,
        "box": 15,
    }
    VOCAB_SIZE = 16
    MAX_LEN = 8

    @classmethod
    def encode(cls, text: str, max_len: int = 8) -> np.ndarray:
        tokens = text.lower().strip().split()
        ids = [cls.VOCAB.get(w, cls.VOCAB["<unk>"]) for w in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [cls.VOCAB["<pad>"]] * (max_len - len(ids))
        return np.array(ids, dtype=np.int64)

    @classmethod
    def decode(cls, ids: np.ndarray) -> str:
        inv_vocab = {v: k for k, v in cls.VOCAB.items()}
        words = [inv_vocab.get(int(i), "<unk>") for i in ids if i != 0]
        return " ".join(words)


class PutNearMultimodalWrapper(gym.ObservationWrapper):
    """
    Multimodal observation wrapper for MiniGrid-PutNear.
    Transforms dictionary observations from:
        {'image': (7, 7, 3), 'direction': int, 'mission': str}
    into a standard Gymnasium Dict space suitable for SB3 MultiInputPolicy:
      - mode='slots' (default, lightweight ~150k params):
        {'image': Box(0, 255, (7, 7, 3), uint8), 'mission': Box(0, 15, (4,), int64)}
        where mission holds [move_color_id, move_type_id, target_color_id, target_type_id]
        aligned with MiniGrid's internal constants.
      - mode='tokens' (BabyAI FiLM format, ~500k params):
        {'image': Box(0, 255, (7, 7, 3), uint8), 'mission': Box(0, VOCAB_SIZE, (max_len,), int64)}
    """
    def __init__(self, env: gym.Env, mode: str = "slots", max_len: int = 8):
        super().__init__(env)
        self.mode = mode
        self.max_len = max_len
        self.tokenizer = PutNearTokenizer

        img_space = env.observation_space["image"]
        if self.mode == "slots":
            mission_space = gym.spaces.Box(
                low=0,
                high=15,
                shape=(5,),
                dtype=np.int64
            )
        else:
            mission_space = gym.spaces.Box(
                low=0,
                high=self.tokenizer.VOCAB_SIZE,
                shape=(self.max_len,),
                dtype=np.int64
            )

        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0,
                high=255,
                shape=img_space.shape,
                dtype=img_space.dtype
            ),
            "mission": mission_space
        })

    def observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        mission_str = obs.get("mission", "")
        if self.mode == "slots":
            # 1-bit is_carrying flag: 1 if agent is currently carrying an object, 0 otherwise
            is_carrying = 1 if getattr(self.env.unwrapped, "carrying", None) is not None else 0
            if isinstance(mission_str, np.ndarray):
                if mission_str.shape == (5,):
                    mission_data = mission_str.copy()
                    mission_data[4] = is_carrying
                elif mission_str.shape == (4,):
                    mission_data = np.append(mission_str, is_carrying).astype(np.int64)
                else:
                    mission_data = np.zeros((5,), dtype=np.int64)
            else:
                words = str(mission_str).lower().strip().split()
                if len(words) >= 8:
                    from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
                    m_color, m_type = words[2], words[3]
                    t_color, t_type = words[6], words[7]
                    mission_data = np.array([
                        COLOR_TO_IDX.get(m_color, 0),
                        OBJECT_TO_IDX.get(m_type, 0),
                        COLOR_TO_IDX.get(t_color, 0),
                        OBJECT_TO_IDX.get(t_type, 0),
                        is_carrying
                    ], dtype=np.int64)
                else:
                    mission_data = np.zeros((5,), dtype=np.int64)
        else:
            if isinstance(mission_str, np.ndarray) and mission_str.shape == (self.max_len,):
                mission_data = mission_str
            else:
                mission_data = self.tokenizer.encode(str(mission_str), max_len=self.max_len)

        return {
            "image": obs["image"],
            "mission": mission_data
        }


class PutNearDebertaRewardWrapper(gym.Wrapper):
    """
    Action-Level Reward Model wrapper for MiniGrid-PutNear.
    Translates relational state and discrete action into natural language descriptions,
    queries the fine-tuned DeBERTa model, and shapes the reward signal with calibrated
    milestones, directional guidance, and anti-farming penalties.
    """
    def __init__(
        self,
        env: gym.Env,
        reward_model: ActionLevelRewardModel,
        step_penalty: float = -0.02,
        collision_penalty: float = -0.25,
        ineffective_penalty: float = -0.25,
        anti_farming_penalty: float = -2.50,
        wrong_pickup_penalty: float = -3.50,
        active_rejection_bonus: float = 0.05,
        max_stall_steps: int = 0,
        selective_eval: bool = False,
        early_term_on_penalty: bool = False,
        include_env_oracle: bool = False,
    ):
        super().__init__(env)
        self.rm = reward_model
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.ineffective_penalty = ineffective_penalty
        self.anti_farming_penalty = anti_farming_penalty
        self.wrong_pickup_penalty = wrong_pickup_penalty
        self.active_rejection_bonus = active_rejection_bonus
        self.max_stall_steps = max_stall_steps
        self.selective_eval = selective_eval
        self.early_term_on_penalty = early_term_on_penalty
        self.include_env_oracle = include_env_oracle

        # Milestone tracking per episode to prevent farming loops
        self.has_picked_up_move_obj = False
        self.has_placed_target = False

        # In-place stall, anti-spin, and turn reversal tracking
        self.stall_steps = 0
        self.consecutive_turns = 0
        self.last_turn_action: Optional[int] = None

    def get_text_description(self, action: int) -> str:
        state_desc, _, _, _ = extract_putnear_state_and_action_desc(self.env, int(action))
        return state_desc

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.has_picked_up_move_obj = False
        self.has_placed_target = False
        self.stall_steps = 0
        self.consecutive_turns = 0
        self.last_turn_action = None
        return self.env.reset(**kwargs)

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        u = self.env.unwrapped
        prev_pos = tuple(u.agent_pos)
        agent_dir = int(u.agent_dir)
        dir_deltas = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        dx, dy = dir_deltas[agent_dir]
        pre_fwd_cell = u.grid.get(prev_pos[0] + dx, prev_pos[1] + dy)
        pre_carrying = u.carrying

        objs = get_putnear_objects(self.env)
        move_color = objs['move_color']
        move_type = objs['move_type']

        was_facing_wrong_obj = False
        if pre_carrying is None and pre_fwd_cell is not None:
            c_type = getattr(pre_fwd_cell, 'type', '')
            c_color = getattr(pre_fwd_cell, 'color', '')
            if c_type in ['key', 'ball', 'box'] and not (c_type == move_type and c_color == move_color):
                was_facing_wrong_obj = True

        state_text = self.get_text_description(action)

        obs, original_reward, terminated, truncated, info = self.env.step(action)

        curr_pos = tuple(u.agent_pos)
        has_moved = (curr_pos != prev_pos)

        # Track displacement, consecutive turns, and immediate reversals
        is_turn_reversal = False
        if has_moved:
            self.stall_steps = 0
            self.consecutive_turns = 0
            self.last_turn_action = None
        else:
            self.stall_steps += 1
            if action in [0, 1]:
                self.consecutive_turns += 1
                if self.last_turn_action is not None and action != self.last_turn_action:
                    is_turn_reversal = True
                self.last_turn_action = action
            else:
                self.consecutive_turns = 0
                self.last_turn_action = None

        current_deberta_score = 0.0
        needs_evaluation = not self.selective_eval

        if self.selective_eval:
            if action == 2:
                collided = (curr_pos == prev_pos and not terminated)
                needs_evaluation = collided
            elif action in [3, 4]:
                needs_evaluation = True
            elif action in [0, 1] and was_facing_wrong_obj:
                needs_evaluation = True

        if needs_evaluation:
            try:
                current_deberta_score = self.rm.predict_reward(state_text)
            except Exception:
                current_deberta_score = 0.0

        total_reward = 0.0

        # Milestone Tier (Positive Rewards: 2.0 for pickup, 10.0 for goal placement)
        if current_deberta_score >= 1.0:
            reward_class = round(current_deberta_score)
            if reward_class == 2:
                if not self.has_picked_up_move_obj:
                    self.has_picked_up_move_obj = True
                    total_reward = current_deberta_score
                    self.stall_steps = 0
                else:
                    total_reward = self.step_penalty
            elif reward_class == 10:
                if not self.has_placed_target:
                    self.has_placed_target = True
                    total_reward = current_deberta_score
                    self.stall_steps = 0
                else:
                    total_reward = self.step_penalty

        # Severe Wrong Object Pickup Tier (DeBERTa predicts <= -3.0 on pickup or wrong interaction)
        elif current_deberta_score <= -3.0 and action == 3:
            total_reward = self.wrong_pickup_penalty
            if self.early_term_on_penalty:
                terminated = True

        # Wall/Obstacle Collision (action == 2 and agent did not move, or DeBERTa collision)
        elif (action == 2 and not has_moved and not (terminated and original_reward > 0)) or (current_deberta_score <= -3.0 and action == 2):
            if pre_fwd_cell is not None and getattr(pre_fwd_cell, 'type', '') in ['key', 'ball', 'box']:
                total_reward = -1.00  # Severe object collision penalty: strongly discourages ramming into objects!
            else:
                total_reward = self.collision_penalty  # Standard wall collision penalty (-0.25)
            if self.early_term_on_penalty:
                terminated = True

        # Any other severe penalty <= -3.0 (fallback)
        elif current_deberta_score <= -3.0:
            total_reward = self.wrong_pickup_penalty
            if self.early_term_on_penalty:
                terminated = True

        # Anti-Farming Penalties Tier (DeBERTa predicts -2.5 for premature drop away from target)
        elif current_deberta_score <= -2.0:
            total_reward = self.anti_farming_penalty  # Strict anti-farming penalty (-2.50 to negate +2.0 milestone)
            if self.early_term_on_penalty:
                terminated = True

        # Severe Empty-Handed Drop Penalty: dropping with empty hands is a completely invalid in-place no-op
        elif action == 4 and pre_carrying is None:
            total_reward = -1.00

        # Immediate In-Place Turn Reversal Penalty: turn_left immediately followed by turn_right (or vice versa) in same cell
        elif action in [0, 1] and is_turn_reversal:
            total_reward = -0.50

        # Ineffective Actions Tier (Toggle / Done actions unused in PutNear)
        elif action in [5, 6]:
            total_reward = self.ineffective_penalty

        # Ineffective In-Place Action Tier (-0.10 for empty pickup / drop)
        elif -0.12 <= current_deberta_score <= -0.08 and action in [3, 4]:
            total_reward = self.ineffective_penalty

        # Anti-Spinning In-Place Turning Penalty: 3rd+ consecutive turn in same cell (permits legitimate 180° turnaround)
        elif action in [0, 1] and self.consecutive_turns >= 3:
            total_reward = self.ineffective_penalty

        # Active Rejection Bonus: Agent turned away from wrong object when empty-handed
        elif action in [0, 1] and was_facing_wrong_obj:
            total_reward = self.active_rejection_bonus

        # Directional Shaping (-0.05 for progress, -0.15 for regressive)
        elif needs_evaluation:
            total_reward = current_deberta_score

        # Default standard motion / single turning step penalty
        else:
            total_reward = self.step_penalty

        # Stagnation Truncation (if enabled)
        if self.max_stall_steps > 0 and self.stall_steps >= self.max_stall_steps and not terminated:
            truncated = True
            info["TimeLimit.truncated"] = True
            info["stagnation_truncated"] = True

        if self.include_env_oracle:
            total_reward += (original_reward * 10.0)

        info["deberta_score"] = current_deberta_score
        info["deberta_reward"] = total_reward
        info["stall_steps"] = self.stall_steps

        return obs, total_reward, terminated, truncated, info


class PutNearRewardShapingWrapper(gym.Wrapper):
    """
    Heuristic / Oracle Reward Shaping baseline for PutNear:
    Provides hardcoded milestone rewards without a neural network:
      - Pick up move object: +2.0
      - Drop near target object: +10.0
      - Step penalty: -0.02
      - Wall collision penalty: -0.25

    When enable_anti_farming is False (standard baseline mode without safeguards):
      - No semantic negative data augmentation (no -3.50 distractor pickup penalty, no +0.05 rejection bonus)
      - No anti-farming safeguards (no -2.50 drop penalty, unlatches pickup on drop, allowing cyclical farming)
      - No special penalties for object ramming, turn reversal, or empty drops
    """
    def __init__(
        self,
        env: gym.Env,
        pickup_reward: float = 2.0,
        goal_reward: float = 10.0,
        step_penalty: float = -0.02,
        collision_penalty: float = -0.25,
        anti_farming_penalty: float = -2.50,
        wrong_pickup_penalty: float = -3.50,
        active_rejection_bonus: float = 0.05,
        ineffective_penalty: float = -0.25,
        max_stall_steps: int = 0,
        enable_anti_farming: bool = False,
    ):
        super().__init__(env)
        self.pickup_reward = pickup_reward
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.enable_anti_farming = enable_anti_farming

        # If anti-farming is disabled, neutralize all semantic negative penalties and bonuses
        if not enable_anti_farming:
            self.anti_farming_penalty = step_penalty
            self.wrong_pickup_penalty = step_penalty
            self.active_rejection_bonus = step_penalty
            self.ineffective_penalty = step_penalty
        else:
            self.anti_farming_penalty = anti_farming_penalty
            self.wrong_pickup_penalty = wrong_pickup_penalty
            self.active_rejection_bonus = active_rejection_bonus
            self.ineffective_penalty = ineffective_penalty

        self.max_stall_steps = max_stall_steps

        self.has_picked_up = False
        self.has_dropped_target = False
        self.stall_steps = 0
        self.consecutive_turns = 0
        self.last_turn_action: Optional[int] = None

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.has_picked_up = False
        self.has_dropped_target = False
        self.stall_steps = 0
        self.consecutive_turns = 0
        self.last_turn_action = None
        return self.env.reset(**kwargs)

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        u = self.env.unwrapped
        prev_pos = tuple(u.agent_pos)
        agent_dir = int(u.agent_dir)
        dir_deltas = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        dx, dy = dir_deltas[agent_dir]
        pre_fwd_cell = u.grid.get(prev_pos[0] + dx, prev_pos[1] + dy)
        pre_carrying = u.carrying

        objs = get_putnear_objects(self.env)
        move_color = objs['move_color']
        move_type = objs['move_type']
        target_pos = objs['target_pos']

        was_facing_wrong_obj = False
        if pre_carrying is None and pre_fwd_cell is not None:
            c_type = getattr(pre_fwd_cell, 'type', '')
            c_color = getattr(pre_fwd_cell, 'color', '')
            if c_type in ['key', 'ball', 'box'] and not (c_type == move_type and c_color == move_color):
                was_facing_wrong_obj = True

        obs, reward, terminated, truncated, info = self.env.step(action)

        curr_pos = tuple(u.agent_pos)
        has_moved = (curr_pos != prev_pos)

        # Track displacement, consecutive turns, and immediate reversals
        is_turn_reversal = False
        if has_moved:
            self.stall_steps = 0
            self.consecutive_turns = 0
            self.last_turn_action = None
        else:
            self.stall_steps += 1
            if action in [0, 1]:
                self.consecutive_turns += 1
                if self.last_turn_action is not None and action != self.last_turn_action:
                    is_turn_reversal = True
                self.last_turn_action = action
            else:
                self.consecutive_turns = 0
                self.last_turn_action = None

        total_reward = self.step_penalty

        # Wall / obstacle collision
        if action == 2 and not has_moved and not (terminated and reward > 0):
            if self.enable_anti_farming and pre_fwd_cell is not None and getattr(pre_fwd_cell, 'type', '') in ['key', 'ball', 'box']:
                total_reward = -1.00  # Strict object collision penalty only if anti-farming / safeguards enabled
            else:
                total_reward = self.collision_penalty  # Standard wall collision (-0.25)

        # Immediate In-Place Turn Reversal Penalty
        elif self.enable_anti_farming and action in [0, 1] and is_turn_reversal:
            total_reward = -0.50

        # Consecutive turns (punish 3+ consecutive turns in same cell; allows 180° turnaround)
        elif self.enable_anti_farming and action in [0, 1] and self.consecutive_turns >= 3:
            total_reward = self.ineffective_penalty

        # Active rejection of wrong objects (only if anti-farming / shaping safeguards enabled)
        elif self.enable_anti_farming and action in [0, 1] and was_facing_wrong_obj:
            total_reward = self.active_rejection_bonus

        # Unused actions (toggle, done)
        elif self.enable_anti_farming and action in [5, 6]:
            total_reward = self.ineffective_penalty

        # Pickup (action == 3)
        elif action == 3:
            now_carrying = u.carrying
            if now_carrying and not pre_carrying:
                if now_carrying.type == move_type and now_carrying.color == move_color:
                    if not self.has_picked_up or not self.enable_anti_farming:
                        self.has_picked_up = True
                        total_reward = self.pickup_reward
                    else:
                        total_reward = self.step_penalty
                else:
                    # Picked up wrong / distractor object
                    total_reward = self.wrong_pickup_penalty
            else:
                total_reward = self.ineffective_penalty if self.enable_anti_farming else self.step_penalty

        # Drop (action == 4)
        elif action == 4:
            if pre_carrying:
                # Check if drop position is near target
                dir_deltas = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
                dx, dy = dir_deltas[u.agent_dir]
                ox, oy = prev_pos[0] + dx, prev_pos[1] + dy
                chebyshev = max(abs(ox - target_pos[0]), abs(oy - target_pos[1]))

                if chebyshev <= 1:
                    if not self.has_dropped_target:
                        self.has_dropped_target = True
                        total_reward = self.goal_reward
                    else:
                        total_reward = self.step_penalty
                else:
                    if self.enable_anti_farming:
                        total_reward = self.anti_farming_penalty
                    else:
                        # Unlatch move object pickup so the agent can pick it up again for +2.0 (enabling the reward farming loop)
                        self.has_picked_up = False
                        total_reward = self.step_penalty
            else:
                total_reward = -1.00 if self.enable_anti_farming else self.step_penalty

        if self.max_stall_steps > 0 and self.stall_steps >= self.max_stall_steps and not terminated:
            truncated = True
            info["TimeLimit.truncated"] = True

        return obs, total_reward, terminated, truncated, info

