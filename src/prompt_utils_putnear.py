"""
Utility functions for generating grounded natural language prompt representations
and action effects for MiniGrid-PutNear-8x8-N3-v0 (and PutNear family).

Ensures 100% parity between offline dataset generation and online RL reward model queries.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import gymnasium as gym


def get_putnear_objects(env: gym.Env) -> Dict[str, Any]:
    """
    Extracts the move object, target object, and distractor object from PutNearEnv.
    """
    u = env.unwrapped
    move_color = getattr(u, 'moveColor', '')
    move_type = getattr(u, 'move_type', '')
    target_color = getattr(u, 'target_color', '')
    target_type = getattr(u, 'target_type', '')
    target_pos = tuple(int(x) for x in u.target_pos)

    is_carrying_move = bool(
        u.carrying and u.carrying.type == move_type and u.carrying.color == move_color
    )

    move_pos = None
    distractor_info = None

    for i in range(u.grid.width):
        for j in range(u.grid.height):
            c = u.grid.get(i, j)
            if c is not None and c.type != 'wall':
                if not is_carrying_move and c.type == move_type and c.color == move_color:
                    move_pos = (int(i), int(j))
                elif c.type == target_type and c.color == target_color:
                    # Target object position
                    pass
                else:
                    distractor_info = {
                        'type': c.type,
                        'color': c.color,
                        'pos': (int(i), int(j)),
                        'cell': c
                    }

    # Fallback for move_pos if not found (e.g. from env.move_pos)
    if move_pos is None and not is_carrying_move and hasattr(u, 'move_pos'):
        move_pos = tuple(int(x) for x in u.move_pos)

    return {
        'move_color': move_color,
        'move_type': move_type,
        'move_pos': move_pos,
        'is_carrying_move': is_carrying_move,
        'target_color': target_color,
        'target_type': target_type,
        'target_pos': target_pos,
        'distractor': distractor_info,
        'mission': getattr(u, 'mission', f"put the {move_color} {move_type} near the {target_color} {target_type}")
    }


def extract_putnear_state_and_action_desc(env: gym.Env, action: int) -> Tuple[str, int, float, str]:
    """
    Extracts a fully specified, grounded natural language prompt and calibrated reward for PutNear.

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

    objs = get_putnear_objects(env)
    move_color = objs['move_color']
    move_type = objs['move_type']
    move_pos = objs['move_pos']
    is_carrying_move = objs['is_carrying_move']
    target_color = objs['target_color']
    target_type = objs['target_type']
    target_pos = objs['target_pos']
    distractor = objs['distractor']
    mission_str = objs['mission']

    dist_str = f"{distractor['color']} {distractor['type']} at {distractor['pos']}" if distractor else "none"
    dist_pos = distractor['pos'] if distractor else None

    # Carrying status
    if is_carrying_move:
        carrying_str = f"{move_color} {move_type}"
        move_loc_str = "carried in inventory"
    elif u.carrying is not None:
        carrying_str = f"{u.carrying.color} {u.carrying.type}"
        move_loc_str = f"at {move_pos}"
    else:
        carrying_str = "nothing"
        move_loc_str = f"at {move_pos}"

    # Forward tile inspection
    fwd_x, fwd_y = ax + dx, ay + dy
    fwd_cell = u.grid.get(fwd_x, fwd_y)
    if fwd_cell is None:
        fwd_str = 'empty space'
    else:
        fwd_color = getattr(fwd_cell, 'color', '')
        fwd_str = f"{fwd_color} {fwd_cell.type}".strip()

    # Active target logic:
    # Phase 1 (Empty hands): Navigate to Move Object to pick it up
    # Phase 2 (Carrying Move Object): Navigate to Target Object to drop adjacent to it
    if is_carrying_move:
        active_target_name = f"Target {target_color} {target_type}"
        active_target_pos = target_pos
    else:
        active_target_name = f"Move {move_color} {move_type}"
        active_target_pos = move_pos if move_pos else target_pos

    act_names = {
        0: 'turn left', 1: 'turn right', 2: 'move forward',
        3: 'pick up', 4: 'drop', 5: 'toggle', 6: 'done'
    }
    act_str = act_names.get(action, str(action))

    effect_str = ''
    reward_val = -0.10
    category_str = 'neutral_step'

    # Current distance to active target (Manhattan metric)
    curr_dist = abs(ax - active_target_pos[0]) + abs(ay - active_target_pos[1])

    # -------------------------------------------------------------------------
    # ACTION 2: MOVE FORWARD
    # -------------------------------------------------------------------------
    if action == 2:
        is_obstacle = (fwd_cell is not None and not getattr(fwd_cell, 'can_overlap', lambda: False)())
        if is_obstacle:
            effect_str = f"collides with {fwd_str}"
            reward_val = -3.50
            category_str = 'wall_collision'
        else:
            new_dist = abs(fwd_x - active_target_pos[0]) + abs(fwd_y - active_target_pos[1])
            # Check if moving closer to distractor when empty handed
            approaching_distractor = False
            if not is_carrying_move and dist_pos:
                dist_d_old = abs(ax - dist_pos[0]) + abs(ay - dist_pos[1])
                dist_d_new = abs(fwd_x - dist_pos[0]) + abs(fwd_y - dist_pos[1])
                if dist_d_new < dist_d_old and new_dist >= curr_dist:
                    approaching_distractor = True

            if new_dist < curr_dist:
                effect_str = f"moves closer to {active_target_name}"
                reward_val = -0.05
                category_str = 'progress_step'
            elif approaching_distractor or new_dist > curr_dist:
                if approaching_distractor:
                    effect_str = f"distracted toward {dist_str}"
                else:
                    effect_str = f"moves away from {active_target_name}"
                reward_val = -0.15
                category_str = 'regressive_step'
            else:
                effect_str = "lateral move"
                reward_val = -0.10
                category_str = 'neutral_step'

    # -------------------------------------------------------------------------
    # ACTIONS 0, 1: TURN LEFT / TURN RIGHT
    # -------------------------------------------------------------------------
    elif action in (0, 1):
        new_dir = (agent_dir - 1) % 4 if action == 0 else (agent_dir + 1) % 4
        new_dx, new_dy = dir_deltas[new_dir]
        target_dx = np.sign(active_target_pos[0] - ax)
        target_dy = np.sign(active_target_pos[1] - ay)

        points_toward = (new_dx == target_dx and target_dx != 0) or (new_dy == target_dy and target_dy != 0)
        old_points_toward = (dx == target_dx and target_dx != 0) or (dy == target_dy and target_dy != 0)

        # Check if agent was directly facing a wrong interactive object while empty-handed
        is_facing_wrong_obj = False
        if not is_carrying_move and u.carrying is None and fwd_cell is not None:
            c_type = getattr(fwd_cell, 'type', '')
            c_color = getattr(fwd_cell, 'color', '')
            if c_type in ['key', 'ball', 'box'] and not (c_type == move_type and c_color == move_color):
                is_facing_wrong_obj = True

        if is_facing_wrong_obj:
            effect_str = f"actively turns away from wrong object ({fwd_str}) to continue searching"
            reward_val = 0.05
            category_str = 'active_rejection'
        elif points_toward and not old_points_toward:
            effect_str = f"turns toward {active_target_name}"
            reward_val = -0.05
            category_str = 'progress_turn'
        elif not points_toward and old_points_toward:
            effect_str = f"turns away from {active_target_name}"
            reward_val = -0.15
            category_str = 'regressive_turn'
        else:
            effect_str = "turns orientation"
            reward_val = -0.10
            category_str = 'neutral_turn'

    # -------------------------------------------------------------------------
    # ACTION 3: PICK UP
    # -------------------------------------------------------------------------
    elif action == 3:
        if u.carrying is not None:
            effect_str = "cannot pick up with hands full"
            reward_val = -0.10
            category_str = 'ineffective_action'
        elif fwd_cell is not None:
            # Check which object is in front
            if fwd_cell.type == move_type and fwd_cell.color == move_color:
                effect_str = f"picks up target move object ({move_color} {move_type})"
                reward_val = 2.0
                category_str = 'pickup_move_obj'
            elif distractor and fwd_cell.type == distractor['type'] and fwd_cell.color == distractor['color']:
                effect_str = f"picks up wrong distractor object ({distractor['color']} {distractor['type']})"
                reward_val = -3.50
                category_str = 'wrong_object_pickup'
            elif fwd_cell.type == target_type and fwd_cell.color == target_color:
                effect_str = f"picks up wrong target object ({target_color} {target_type})"
                reward_val = -3.50
                category_str = 'wrong_object_pickup'
            else:
                effect_str = "ineffective pick up"
                reward_val = -0.10
                category_str = 'ineffective_action'
        else:
            effect_str = "ineffective pick up (empty tile)"
            reward_val = -0.10
            category_str = 'ineffective_action'

    # -------------------------------------------------------------------------
    # ACTION 4: DROP
    # -------------------------------------------------------------------------
    elif action == 4:
        if not is_carrying_move:
            effect_str = "ineffective drop (not carrying move object)"
            reward_val = -0.10
            category_str = 'ineffective_action'
        else:
            # Front tile must be empty for a physical drop to succeed
            if fwd_cell is not None:
                effect_str = "cannot drop onto occupied tile"
                reward_val = -0.10
                category_str = 'ineffective_action'
            else:
                # Chebyshev distance <= 1 to target position: abs(ox - tx) <= 1 and abs(oy - ty) <= 1
                chebyshev_dist = max(abs(fwd_x - target_pos[0]), abs(fwd_y - target_pos[1]))
                if chebyshev_dist <= 1:
                    effect_str = f"successfully drops near target {target_color} {target_type}"
                    reward_val = 10.0
                    category_str = 'goal_reached'
                else:
                    effect_str = f"premature drop away from target (chebyshev_dist={chebyshev_dist})"
                    reward_val = -2.50
                    category_str = 'anti_farming_drop'

    # -------------------------------------------------------------------------
    # ACTIONS 5, 6: TOGGLE / DONE (UNUSED IN PUTNEAR)
    # -------------------------------------------------------------------------
    else:
        effect_str = f"ineffective action ({act_str})"
        reward_val = -0.10
        category_str = 'ineffective_action'

    text_desc = (
        f"Agent at ({ax}, {ay}) facing {facing_str}. "
        f"Forward vision: {fwd_str}. Inventory: {carrying_str}. "
        f"Mission: {mission_str}. "
        f"Move object: {move_color} {move_type} ({move_loc_str}). "
        f"Target object: {target_color} {target_type} at {target_pos}. "
        f"Distractor: {dist_str}. "
        f"Active target: {active_target_name} at {active_target_pos}. "
        f"Distance to active target: {curr_dist}. "
        f"Action taken: {act_str} ({effect_str})"
    )
    return text_desc, action, reward_val, category_str

