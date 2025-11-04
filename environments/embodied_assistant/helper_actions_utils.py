from enum import IntEnum
import jax.numpy as jnp


def create_helper_actions_enum(no_freeze, no_pull_helper_action):
    # Create dictionary of enum values
    enum_dict = {}

    # Add its own movement actions
    enum_dict["RIGHT"] = 0
    enum_dict["DOWN"] = 1
    enum_dict["LEFT"] = 2
    enum_dict["UP"] = 3

    # Add push actions
    enum_dict[f"PUSH_BOX_RIGHT"] = 4
    enum_dict[f"PUSH_BOX_DOWN"] = 5
    enum_dict[f"PUSH_BOX_LEFT"] = 6
    enum_dict[f"PUSH_BOX_UP"] = 7

    next_action_id = 8

    # Add pull actions only if pulling is enabled
    if not no_pull_helper_action:
        enum_dict[f"PULL_BOX_LEFT_FROM_RIGHT"] = next_action_id
        enum_dict[f"PULL_BOX_UP_FROM_DOWN"] = next_action_id + 1
        enum_dict[f"PULL_BOX_RIGHT_FROM_LEFT"] = next_action_id + 2
        enum_dict[f"PULL_BOX_DOWN_FROM_UP"] = next_action_id + 3
        next_action_id += 4

    # Add special actions at the end
    enum_dict["DO_NOTHING"] = next_action_id
    next_action_id += 1

    if not no_freeze:
        enum_dict["FREEZE_AGENT_1"] = next_action_id

    HelperActions = IntEnum("HelperActions", enum_dict)

    return HelperActions


def create_helper_action_directions(no_pull_helper_action, no_freeze):
    # Basic movement directions
    right = [0, 1]
    down = [1, 0]
    left = [0, -1]
    up = [-1, 0]
    no_move = [0, 0]

    # List to store all directions
    directions = []

    # Add directions for its own movement (actions 0-3)
    directions.extend([right, down, left, up])

    # Add directions for pushing (actions 4-7)
    directions.extend([right, down, left, up])

    # Add directions for pulling only if pulling is enabled (actions 8-11)
    if not no_pull_helper_action:
        directions.extend([left, up, right, down])

    # Add special actions
    directions.append(no_move)  # DO_NOTHING
    if not no_freeze:
        directions.append(no_move)  # FREEZE_AGENT_1

    return jnp.array(directions)
