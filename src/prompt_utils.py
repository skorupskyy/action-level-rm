"""
Utility functions for generating grounded natural language prompt representations
and action effects for MiniGrid-DoorKey-8x8-v0.

Ensures 100% parity between offline dataset generation and online RL reward model queries.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import gymnasium as gym


def extract_env_state_and_action_desc(env: gym.Env, action: int) -> Tuple[str, int, float, str]:
    """
    Extracts a fully specified, grounded natural language prompt and reward for a given action.

    Returns:
        (text_description, action_id, reward_value, category_name)
    """
    u = env.unwrapped
    ax, ay = int(u.agent_pos[0]), int(u.agent_pos[1])
    agent_dir = int(u.agent_dir)
    dir_names = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
    dir_deltas = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
    facing_str = dir_names[agent_dir]
    dx, dy = dir_deltas[agent_dir]

    door_pos = None
    door_state = 'closed'
    door_cell = None
    key_pos = None
    goal_pos = (int(u.grid.width - 2), int(u.grid.height - 2))

    for i in range(u.grid.width):
        for j in range(u.grid.height):
            c = u.grid.get(i, j)
            if c is not None:
                if c.type == 'door':
                    door_pos = (int(i), int(j))
                    door_cell = c
                    if c.is_open:
                        door_state = 'open'
                    elif c.is_locked:
                        door_state = 'locked'
                    else:
                        door_state = 'unlocked and closed'
                elif c.type == 'key':
                    key_pos = (int(i), int(j))

    split_x = door_pos[0] if door_pos else 4
    if ax < split_x:
        room_str = 'Room 1 (left)'
    elif ax > split_x:
        room_str = 'Room 2 (right)'
    else:
        room_str = 'on Doorway'

    is_carrying_key = bool(u.carrying and u.carrying.type == 'key')
    carrying_str = 'key' if is_carrying_key else 'nothing'
    key_loc_str = 'carried in inventory' if is_carrying_key else f'at {key_pos}'

    fwd_x, fwd_y = ax + dx, ay + dy
    fwd_cell = u.grid.get(fwd_x, fwd_y)
    if fwd_cell is None:
        fwd_str = 'empty space'
    elif fwd_cell.type == 'door':
        fwd_str = f'{door_state} door'
    else:
        fwd_str = fwd_cell.type

    # Active target in the sequential task
    if door_state == 'open':
        active_target_name = 'Goal'
        target_pos = goal_pos
    elif is_carrying_key:
        active_target_name = 'Door'
        target_pos = door_pos
    else:
        active_target_name = 'Key'
        target_pos = key_pos if key_pos else (split_x - 1, ay)

    act_names = {
        0: 'turn left', 1: 'turn right', 2: 'move forward',
        3: 'pick up', 4: 'drop', 5: 'toggle or open', 6: 'done'
    }
    act_str = act_names.get(action, str(action))

    effect_str = ''
    reward_val = -0.10
    category_str = 'neutral_step'

    curr_dist = abs(ax - target_pos[0]) + abs(ay - target_pos[1])

    if action == 2:  # move forward
        is_obstacle = (fwd_cell is not None and not getattr(fwd_cell, 'can_overlap', lambda: False)())
        if is_obstacle:
            effect_str = 'collides with obstacle'
            reward_val = -3.50
            category_str = 'wall_collision'
        elif (fwd_x, fwd_y) == goal_pos:
            effect_str = 'reaches Goal'
            reward_val = 10.0
            category_str = 'goal_reached'
        else:
            new_dist = abs(fwd_x - target_pos[0]) + abs(fwd_y - target_pos[1])
            if new_dist < curr_dist:
                effect_str = f'moves closer to target {active_target_name}'
                reward_val = -0.05
                category_str = 'progress_step'
            elif new_dist > curr_dist:
                effect_str = f'moves away from target {active_target_name}'
                reward_val = -0.15
                category_str = 'regressive_step'
            else:
                effect_str = 'lateral move'
                reward_val = -0.10
                category_str = 'neutral_step'

    elif action in (0, 1):  # turn left / turn right
        new_dir = (agent_dir - 1) % 4 if action == 0 else (agent_dir + 1) % 4
        new_dx, new_dy = dir_deltas[new_dir]
        target_dx = np.sign(target_pos[0] - ax)
        target_dy = np.sign(target_pos[1] - ay)

        points_toward = (new_dx == target_dx and target_dx != 0) or (new_dy == target_dy and target_dy != 0)
        old_points_toward = (dx == target_dx and target_dx != 0) or (dy == target_dy and target_dy != 0)

        if points_toward and not old_points_toward:
            effect_str = f'turns toward target {active_target_name}'
            reward_val = -0.05
            category_str = 'progress_turn'
        elif not points_toward and old_points_toward:
            effect_str = f'turns away from target {active_target_name}'
            reward_val = -0.15
            category_str = 'regressive_turn'
        else:
            effect_str = 'turns orientation'
            reward_val = -0.10
            category_str = 'neutral_turn'

    elif action == 3:  # pick up
        if fwd_cell and fwd_cell.type == 'key' and not is_carrying_key:
            effect_str = 'picks up target Key'
            reward_val = 2.0
            category_str = 'key_pickup'
        else:
            effect_str = 'ineffective pick up'
            reward_val = -0.10
            category_str = 'ineffective_action'

    elif action == 5:  # toggle or open
        if fwd_cell and fwd_cell.type == 'door':
            if fwd_cell.is_locked and is_carrying_key:
                effect_str = 'unlocks and opens door'
                reward_val = 3.0
                category_str = 'door_opened'
            elif fwd_cell.is_open:
                effect_str = 'closes open door'
                reward_val = -2.50
                category_str = 'anti_farming_door'
            else:
                effect_str = 'cannot open door without key'
                reward_val = -0.10
                category_str = 'ineffective_action'
        else:
            effect_str = 'ineffective toggle'
            reward_val = -0.10
            category_str = 'ineffective_action'

    elif action == 4:  # drop
        if is_carrying_key:
            effect_str = 'drops essential Key'
            reward_val = -2.50
            category_str = 'anti_farming_drop'
        else:
            effect_str = 'ineffective drop'
            reward_val = -0.10
            category_str = 'ineffective_action'

    text_desc = (
        f'Agent at ({ax}, {ay}) in {room_str}, facing {facing_str}. '
        f'Forward vision: {fwd_str}. Inventory: {carrying_str}. '
        f'Dividing wall at x={split_x}. Door at {door_pos} is {door_state}. '
        f'Key: {key_loc_str}. Goal at {goal_pos}. '
        f'Active target: {active_target_name} at {target_pos}. '
        f'Action taken: {act_str} ({effect_str})'
    )
    return text_desc, action, reward_val, category_str
