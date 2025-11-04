from .multiagent_gridworld import State
import chex
import jax.numpy as jnp


def convert_state_to_grid_observation(
    state: State, height: int, width: int
) -> chex.Array:
    """
    Assumes state is not batched!
    Grid observation for agents
    Convert state to a grid-based observation of shape (height, width, channels).

    Channels:
    0: Agent 0 position
    1: Agent 1 position (if exists)
    2: Goals for Agent 0
    3: Goals for Agent 1 (if exists)
    4: Key for Agent 0
    5: Key for Agent 1 (if exists)
    6: Boxes
    7: Traps
    8: Walls
    9: Helper position
    10: Freeze timer (constant across grid)
    11: Has Keys for Agent 0
    12: Has Keys for Agent 1 (if exists)
    """
    # Assume unbatched input: state.agent_pos has shape (num_agents, 2)
    print(f"state.agent_pos shape: {state.agent_pos.shape}")
    num_agents = state.agent_pos.shape[0]
    num_goals = state.goal_pos.shape[0]
    num_keys = state.key_pos.shape[0]

    agent_to_goal_mapping = jnp.arange(num_goals)
    # NOTE: assumes that number of goals is less than or equal to number of agents
    if len(agent_to_goal_mapping) < num_agents:
        agent_to_goal_mapping = jnp.concatenate(
            (agent_to_goal_mapping, agent_to_goal_mapping)
        )

    key_pos = state.key_pos
    if num_keys == 1:
        key_pos = jnp.repeat(key_pos, num_agents, axis=0)

    # Determine number of channels needed
    channels = (
        5 + num_agents * 4
    )  # base + num_agents number of channels for position and goal position and key position and has key

    print(f"num_agents: {num_agents}")
    print(f"num channels: {channels}")

    # Initialize grid observation - no batch dimension
    grid_obs = jnp.zeros((height, width, channels))

    # Channel index
    ch = 0

    # Agent positions (one channel per agent)
    for agent_idx in range(num_agents):
        agent_pos = state.agent_pos[agent_idx, :]  # (2,)
        agent_grid = jnp.zeros((height, width))

        # Place agent on grid
        y, x = agent_pos[0], agent_pos[1]
        agent_grid = agent_grid.at[y, x].set(1.0)
        grid_obs = grid_obs.at[:, :, ch].set(agent_grid)
        ch += 1

    # Goal positions (one channel per agent's goals)
    for agent_idx in range(num_agents):
        goal_idx = agent_to_goal_mapping[agent_idx]
        goal_pos = state.goal_pos[goal_idx, :]  # (2,)

        goal_grid = jnp.zeros((height, width))
        y, x = goal_pos[0], goal_pos[1]
        goal_grid = goal_grid.at[y, x].set(1.0)
        grid_obs = grid_obs.at[:, :, ch].set(goal_grid)
        ch += 1

    # Key positions (one channel per agent's key)
    for agent_idx in range(num_agents):
        agent_key_pos = key_pos[agent_idx, :]  # (2,)
        key_grid = jnp.zeros((height, width))
        y, x = agent_key_pos[0], agent_key_pos[1]
        key_grid = key_grid.at[y, x].set(1.0)
        grid_obs = grid_obs.at[:, :, ch].set(key_grid)
        ch += 1

    # Box positions
    box_grid = jnp.zeros((height, width))
    num_boxes = state.box_pos.shape[0]
    for box_idx in range(num_boxes):
        box_pos = state.box_pos[box_idx, :]  # (2,)
        y, x = box_pos[0], box_pos[1]
        box_grid = box_grid.at[y, x].set(1.0)
    grid_obs = grid_obs.at[:, :, ch].set(box_grid)
    ch += 1

    # Trap positions
    trap_grid = jnp.zeros((height, width))
    num_traps = state.trap_pos.shape[0]
    for trap_idx in range(num_traps):
        trap_pos = state.trap_pos[trap_idx, :]  # (2,)
        y, x = trap_pos[0], trap_pos[1]
        trap_grid = trap_grid.at[y, x].set(1.0)
    grid_obs = grid_obs.at[:, :, ch].set(trap_grid)
    ch += 1

    # Wall positions
    wall_grid = jnp.zeros((height, width))
    num_walls = state.wall_pos.shape[0]
    for wall_idx in range(num_walls):
        wall_pos = state.wall_pos[wall_idx, :]  # (2,)
        y, x = wall_pos[0], wall_pos[1]
        wall_grid = wall_grid.at[y, x].set(1.0)
    grid_obs = grid_obs.at[:, :, ch].set(wall_grid)
    ch += 1

    # Helper position
    helper_grid = jnp.zeros((height, width))
    helper_pos = state.helper_pos[0, :]  # (2,) - assuming helper_pos shape is (1, 2)
    y, x = helper_pos[0], helper_pos[1]
    helper_grid = helper_grid.at[y, x].set(1.0)
    grid_obs = grid_obs.at[:, :, ch].set(helper_grid)
    ch += 1

    # Freeze timer (constant across the grid)
    # state.freeze_timer is a scalar
    freeze_timer_grid = jnp.ones((height, width)) * state.freeze_timer
    grid_obs = grid_obs.at[:, :, ch].set(freeze_timer_grid)
    ch += 1

    # has keys (one channel per agents has key)
    for agent_idx in range(num_agents):
        has_key_grid = jnp.ones((height, width)) * state.has_key[agent_idx]
        grid_obs = grid_obs.at[:, :, ch].set(has_key_grid)
        ch += 1

    return grid_obs


def convert_state_to_partial_grid_observation(
    state: State, height: int, width: int
) -> chex.Array:
    """
    Assumes that the state is not batched!
    Grid observation for assistant (does not include the goal channel)
    Convert state to a grid-based observation of shape (height, width, channels).

    Channels:
    0: Agent 0 position
    1: Agent 1 position (if exists)
    2: Key for Agent 0
    3: Key for Agent 1 (if exists)
    4: Boxes
    5: Traps
    6: Walls
    7: Helper position
    8: Freeze timer (constant across grid)
    9: Has Keys for Agent 0
    10: Has Keys for Agent 1 (if exists)
    """
    # Assume unbatched input: state.agent_pos has shape (num_agents, 2)
    num_agents = state.agent_pos.shape[0]
    num_keys = state.key_pos.shape[0]

    key_pos = state.key_pos
    if num_keys == 1:
        key_pos = jnp.repeat(key_pos, num_agents, axis=0)

    # Determine number of channels needed
    channels = 5 + num_agents * 3

    # Initialize grid observation - no batch dimension
    grid_obs = jnp.zeros((height, width, channels))

    # Channel index
    ch = 0

    # Agent positions (one channel per agent)
    for agent_idx in range(num_agents):
        agent_pos = state.agent_pos[agent_idx, :]  # (2,)
        agent_grid = jnp.zeros((height, width))
        
        # Place agent on grid
        y, x = agent_pos[0], agent_pos[1]
        agent_grid = agent_grid.at[y, x].set(1.0)
        grid_obs = grid_obs.at[:, :, ch].set(agent_grid)
        ch += 1

    # Key positions (one channel per agent's key)
    for agent_idx in range(num_agents):
        agent_key_pos = key_pos[agent_idx, :]  # (2,)
        key_grid = jnp.zeros((height, width))
        y, x = agent_key_pos[0], agent_key_pos[1]
        key_grid = key_grid.at[y, x].set(1.0)
        grid_obs = grid_obs.at[:, :, ch].set(key_grid)
        ch += 1

    # Box positions
    box_grid = jnp.zeros((height, width))
    num_boxes = state.box_pos.shape[0]
    for box_idx in range(num_boxes):
        box_pos = state.box_pos[box_idx, :]  # (2,)
        y, x = box_pos[0], box_pos[1]
        box_grid = box_grid.at[y, x].set(1.0)
    grid_obs = grid_obs.at[:, :, ch].set(box_grid)
    ch += 1

    # Trap positions
    trap_grid = jnp.zeros((height, width))
    num_traps = state.trap_pos.shape[0]
    for trap_idx in range(num_traps):
        trap_pos = state.trap_pos[trap_idx, :]  # (2,)
        y, x = trap_pos[0], trap_pos[1]
        trap_grid = trap_grid.at[y, x].set(1.0)
    grid_obs = grid_obs.at[:, :, ch].set(trap_grid)
    ch += 1

    # Wall positions
    wall_grid = jnp.zeros((height, width))
    num_walls = state.wall_pos.shape[0]
    for wall_idx in range(num_walls):
        wall_pos = state.wall_pos[wall_idx, :]  # (2,)
        y, x = wall_pos[0], wall_pos[1]
        wall_grid = wall_grid.at[y, x].set(1.0)
    grid_obs = grid_obs.at[:, :, ch].set(wall_grid)
    ch += 1

    # Helper position
    helper_grid = jnp.zeros((height, width))
    helper_pos = state.helper_pos[0, :]  # (2,) - assuming helper_pos shape is (1, 2)
    y, x = helper_pos[0], helper_pos[1]
    helper_grid = helper_grid.at[y, x].set(1.0)
    grid_obs = grid_obs.at[:, :, ch].set(helper_grid)
    ch += 1

    # Freeze timer (constant across the grid)
    # state.freeze_timer is a scalar
    freeze_timer_grid = jnp.ones((height, width)) * state.freeze_timer
    grid_obs = grid_obs.at[:, :, ch].set(freeze_timer_grid)
    ch += 1

    # has keys (one channel per agents has key)
    for agent_idx in range(num_agents):
        has_key_grid = jnp.ones((height, width)) * state.has_key[agent_idx]
        grid_obs = grid_obs.at[:, :, ch].set(has_key_grid)
        ch += 1

    return grid_obs


def convert_state_to_flattened_vector(state: State) -> chex.Array:
    """This assumes that the state is batched"""
    batch_size = state.agent_pos.shape[0]
    num_agents = state.agent_pos.shape[-2]
    num_boxes = state.box_pos.shape[-2]
    num_traps = state.trap_pos.shape[-2]
    num_walls = state.wall_pos.shape[-2]
    num_keys = state.key_pos.shape[-2]
    # Reshape the last two dimensions into a single vector
    agent_pos_flat = jnp.reshape(state.agent_pos, (batch_size, num_agents * 2))
    box_pos_flat = jnp.reshape(state.box_pos, (batch_size, num_boxes * 2))
    trap_pos_flat = jnp.reshape(state.trap_pos, (batch_size, num_traps * 2))
    wall_pos_flat = jnp.reshape(state.wall_pos, (batch_size, num_walls * 2))
    helper_pos_flat = jnp.reshape(state.helper_pos, (batch_size, 2))
    freeze_timer_flat = jnp.reshape(state.freeze_timer, (batch_size, 1))
    key_pos_flat = jnp.reshape(state.key_pos, (batch_size, num_keys * 2))
    has_key_flat = jnp.reshape(state.has_key, (batch_size, 2))

    # Concatenate along the last dimension
    return jnp.concatenate(
        [
            agent_pos_flat,
            box_pos_flat,
            trap_pos_flat,
            wall_pos_flat,
            helper_pos_flat,
            freeze_timer_flat,
            key_pos_flat,
            has_key_flat,
        ],
        axis=-1,
    )
