from enum import IntEnum
import jax.numpy as jnp


def create_helper_actions_enum(num_boxes, no_freeze):
    # Create dictionary of enum values
    enum_dict = {}
    
    # Add box actions
    for box_idx in range(num_boxes):
        base_idx = box_idx * 4  # Each box has 4 actions
        enum_dict[f'BOX_{box_idx}_RIGHT'] = base_idx
        enum_dict[f'BOX_{box_idx}_DOWN'] = base_idx + 1
        enum_dict[f'BOX_{box_idx}_LEFT'] = base_idx + 2
        enum_dict[f'BOX_{box_idx}_UP'] = base_idx + 3
    
    # Add special actions at the end
    enum_dict['DO_NOTHING'] = num_boxes * 4
    if not no_freeze:
        enum_dict['FREEZE_AGENT_1'] = num_boxes * 4 + 1
    
    HelperActions = IntEnum('HelperActions', enum_dict)
    
    return HelperActions


def create_helper_action_directions(num_boxes):
    # Basic movement directions
    right = [0, 1]
    down = [1, 0]
    left = [0, -1]
    up = [-1, 0]
    no_move = [0, 0]
    
    # List to store all directions
    directions = []
    
    # Add directions for each box
    for _ in range(num_boxes):
        directions.extend([right, down, left, up])
    
    # Add special actions
    directions.extend([
        no_move,  # DO_NOTHING
        no_move   # FREEZE_AGENT_1
    ])
    
    return jnp.array(directions)
