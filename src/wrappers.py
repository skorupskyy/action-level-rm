import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
import numpy as np
from typing import Dict, Any, Optional, Tuple, Set

from src.rm_model import ActionLevelRewardModel
from src.prompt_utils import extract_env_state_and_action_desc


class DebertaRewardWrapper(gym.Wrapper):
    """
    Action-Level Reward Model wrapper.
    Converts symbolic state and discrete action into natural language descriptions,
    queries the fine-tuned DeBERTa model, and shapes the reward signal with a step penalty.
    """
    def __init__(
        self,
        env: gym.Env,
        reward_model: ActionLevelRewardModel,
        step_penalty: float = -0.02,
        collision_penalty: float = -0.25,
        ineffective_penalty: float = -0.25,
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
        self.max_stall_steps = max_stall_steps
        self.selective_eval = selective_eval
        self.early_term_on_penalty = early_term_on_penalty
        self.include_env_oracle = include_env_oracle

        self.action_map = {
            0: 'turn left',
            1: 'turn right',
            2: 'move forward',
            3: 'pick up',
            4: 'drop',
            5: 'toggle or open',
            6: 'done'
        }
        self.dir_map = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

        # Milestone tracking per episode to prevent farming loops
        self.has_picked_up_key = False
        self.has_opened_door = False
        self.has_reached_goal = False

        # In-place stall & anti-spin tracking
        self.stall_steps = 0
        self.consecutive_turns = 0

    def get_text_description(self, action: int) -> str:
        state_desc, _, _, _ = extract_env_state_and_action_desc(self.env, int(action))
        return state_desc

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.has_picked_up_key = False
        self.has_opened_door = False
        self.has_reached_goal = False
        self.stall_steps = 0
        self.consecutive_turns = 0
        return self.env.reset(**kwargs)

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        prev_pos = tuple(self.env.unwrapped.agent_pos)
        state_text = self.get_text_description(action)

        obs, original_reward, terminated, truncated, info = self.env.step(action)

        curr_pos = tuple(self.env.unwrapped.agent_pos)
        has_moved = (curr_pos != prev_pos)

        # Track displacement and consecutive turns
        if has_moved:
            self.stall_steps = 0
            self.consecutive_turns = 0
        else:
            self.stall_steps += 1
            if action in [0, 1]:
                self.consecutive_turns += 1
            else:
                self.consecutive_turns = 0

        current_deberta_score = 0.0
        needs_evaluation = not self.selective_eval

        if self.selective_eval:
            # Action 2 is forward: check if collided (agent didn't move) or reached goal
            if action == 2:
                collided = (curr_pos == prev_pos and not terminated)
                reached_goal = (terminated and original_reward > 0)
                if collided or reached_goal:
                    needs_evaluation = True
            elif action in [3, 4, 5]:
                # Object manipulation actions: pick up, drop, toggle
                needs_evaluation = True

        if needs_evaluation:
            try:
                current_deberta_score = self.rm.predict_reward(state_text)
            except Exception:
                current_deberta_score = 0.0

        total_reward = 0.0

        # Milestone Tier (Positive Rewards: 2.0, 3.0, 10.0)
        if current_deberta_score >= 1.0:
            reward_class = round(current_deberta_score)
            if reward_class == 2:
                if not self.has_picked_up_key:
                    self.has_picked_up_key = True
                    total_reward = current_deberta_score
                    self.stall_steps = 0
                else:
                    total_reward = self.step_penalty  # Already collected, standard step penalty
            elif reward_class == 3:
                if not self.has_opened_door:
                    self.has_opened_door = True
                    total_reward = current_deberta_score
                    self.stall_steps = 0
                else:
                    total_reward = self.step_penalty
            elif reward_class == 10:
                if not self.has_reached_goal:
                    self.has_reached_goal = True
                    total_reward = current_deberta_score
                    self.stall_steps = 0
                else:
                    total_reward = self.step_penalty

        # Severe Wall/Obstacle Collision Tier (DeBERTa predicts -3.5 or physics detected collision)
        elif current_deberta_score <= -3.0 or (action == 2 and not has_moved and not (terminated and original_reward > 0)):
            total_reward = self.collision_penalty
            if self.early_term_on_penalty:
                terminated = True

        # Anti-Farming Penalties Tier (DeBERTa predicts -2.5 for drop key or close door)
        elif current_deberta_score <= -2.0:
            total_reward = current_deberta_score  # -2.50 to strictly punish farming loops
            if self.early_term_on_penalty:
                terminated = True

        # Ineffective Done Action Tier (Action 6 when not at goal)
        elif action == 6 and not (terminated and original_reward > 0):
            total_reward = self.ineffective_penalty

        # Ineffective In-Place Action Tier (DeBERTa predicts -0.10 for empty pickup / drop / toggle)
        elif -0.12 <= current_deberta_score <= -0.08 and action in [3, 4, 5]:
            total_reward = self.ineffective_penalty

        # Anti-Spinning In-Place Turning Penalty: 2nd+ consecutive turn in same cell
        elif action in [0, 1] and self.consecutive_turns >= 2:
            total_reward = self.ineffective_penalty

        # Directional Shaping (DeBERTa predicts -0.05 for progress, -0.15 for regressive)
        elif needs_evaluation:
            total_reward = current_deberta_score

        # Default standard motion / single turning step penalty
        else:
            total_reward = self.step_penalty

        # Stagnation Truncation: Cut off episode if agent stagnates in the same cell (if enabled)
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


class IntrinsicMotivationWrapper(gym.Wrapper):
    """
    Count-based exploration bonus wrapper:
      r_t = r_env + beta / sqrt(N(s, a))
    """
    def __init__(self, env: gym.Env, beta: float = 0.05, count_type: str = "state_action"):
        super().__init__(env)
        self.beta = beta
        self.count_type = count_type
        self.visit_counts: Dict[Any, int] = {}

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        return self.env.reset(**kwargs)

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        unwrapped = self.env.unwrapped
        if self.count_type == "state_action":
            state_key = (
                tuple(unwrapped.agent_pos),
                unwrapped.agent_dir,
                bool(unwrapped.carrying),
                int(action)
            )
        else:
            state_key = tuple(unwrapped.agent_pos)

        count = self.visit_counts.get(state_key, 0) + 1
        self.visit_counts[state_key] = count

        intrinsic_reward = self.beta / np.sqrt(count)
        total_reward = reward + intrinsic_reward

        info["intrinsic_reward"] = intrinsic_reward
        info["visit_count"] = count

        return obs, total_reward, terminated, truncated, info


class RewardShapingWrapper(gym.Wrapper):
    """
    Heuristic / Oracle Reward Shaping baseline:
    Provides hardcoded milestone rewards without a neural network:
      - Pick up key: +2.0
      - Unlock/open door: +3.0
      - Reach goal: +10.0
      - Step penalty: -0.02
      - Wall collision penalty: -0.75
      - Anti-farming penalty (drop key): -2.5
      - Ineffective penalty (invalid no-ops / spinning): -0.25
      - Stagnation truncation: truncates episode after max_stall_steps in same cell
    """
    def __init__(
        self,
        env: gym.Env,
        key_pickup_reward: float = 2.0,
        door_open_reward: float = 3.0,
        goal_reward: float = 10.0,
        step_penalty: float = -0.02,
        collision_penalty: float = -0.25,
        anti_farming_penalty: float = -2.5,
        ineffective_penalty: float = -0.25,
        max_stall_steps: int = 0,
        enable_anti_farming: bool = True,
    ):
        super().__init__(env)
        self.key_pickup_reward = key_pickup_reward
        self.door_open_reward = door_open_reward
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.anti_farming_penalty = anti_farming_penalty if enable_anti_farming else 0.0
        self.ineffective_penalty = ineffective_penalty
        self.max_stall_steps = max_stall_steps
        self.enable_anti_farming = enable_anti_farming

        self.has_picked_up_key = False
        self.has_opened_door = False
        self.has_reached_goal = False

        # In-place stall & anti-spin tracking
        self.stall_steps = 0
        self.consecutive_turns = 0

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self.has_picked_up_key = False
        self.has_opened_door = False
        self.has_reached_goal = False
        self.stall_steps = 0
        self.consecutive_turns = 0
        return self.env.reset(**kwargs)

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        prev_pos = tuple(self.env.unwrapped.agent_pos)
        was_carrying = bool(self.env.unwrapped.carrying)

        obs, original_reward, terminated, truncated, info = self.env.step(action)

        curr_pos = tuple(self.env.unwrapped.agent_pos)
        has_moved = (curr_pos != prev_pos)

        # Track displacement and consecutive turns
        if has_moved:
            self.stall_steps = 0
            self.consecutive_turns = 0
        else:
            self.stall_steps += 1
            if action in [0, 1]:
                self.consecutive_turns += 1
            else:
                self.consecutive_turns = 0

        shaping_reward = 0.0
        now_carrying = bool(self.env.unwrapped.carrying)

        # 1. Key pickup milestone
        if now_carrying and not was_carrying and (not self.has_picked_up_key or not self.enable_anti_farming):
            self.has_picked_up_key = True
            shaping_reward += self.key_pickup_reward
            self.stall_steps = 0

        # 2. Key drop (anti-farming check)
        elif was_carrying and not now_carrying:
            if not self.enable_anti_farming:
                # Unlatch key pickup so agent can pick it up again for +2.0 (enabling the farming loop)
                self.has_picked_up_key = False
            shaping_reward += self.anti_farming_penalty

        # 3. Door toggle milestone
        else:
            door_is_open = False
            for i in range(self.env.unwrapped.grid.width):
                for j in range(self.env.unwrapped.grid.height):
                    cell = self.env.unwrapped.grid.get(i, j)
                    if cell and cell.type == "door" and cell.is_open:
                        door_is_open = True
                        break

            if door_is_open and (not self.has_opened_door or not self.enable_anti_farming):
                self.has_opened_door = True
                shaping_reward += self.door_open_reward
                self.stall_steps = 0
            elif not door_is_open and not self.enable_anti_farming:
                # Door closed without anti-farming: unlatch door milestone so agent can reopen it for +3.0
                self.has_opened_door = False

            # 4. Wall collision penalty
            elif action == 2 and curr_pos == prev_pos and not terminated:
                shaping_reward += self.collision_penalty

            # 5. Goal milestone
            elif terminated and original_reward > 0 and not self.has_reached_goal:
                self.has_reached_goal = True
                shaping_reward += self.goal_reward
                self.stall_steps = 0

            # 6. Ineffective done action (Action 6 when not at goal)
            elif action == 6 and not (terminated and original_reward > 0):
                shaping_reward += self.ineffective_penalty

            # 7. Ineffective in-place manipulation (pickup, drop, toggle with no effect)
            elif action in [3, 4, 5]:
                shaping_reward += self.ineffective_penalty

            # 8. Anti-spinning turning penalty (2nd+ consecutive turn in same cell)
            elif action in [0, 1] and self.consecutive_turns >= 2:
                shaping_reward += self.ineffective_penalty

        total_reward = shaping_reward if shaping_reward != 0.0 else self.step_penalty

        # Stagnation Truncation: Cut off episode if agent stagnates in the same cell (if enabled)
        if self.max_stall_steps > 0 and self.stall_steps >= self.max_stall_steps and not terminated:
            truncated = True
            info["TimeLimit.truncated"] = True
            info["stagnation_truncated"] = True

        info["stall_steps"] = self.stall_steps

        return obs, total_reward, terminated, truncated, info
