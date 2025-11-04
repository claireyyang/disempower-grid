import jax
import jax.numpy as jnp
from .multiagent_gridworld import (
    MultiAgentGridWorld,
    State,
)


def estimate_discrete_choice(
    env: MultiAgentGridWorld,
    initial_state: State,
    n_trajectories: int = 20,
    horizon: int = 3,
    agent_idx: int = 0,  # leader agent index
):
    """
    Calculate discrete choice for the leader agent.
    Returns the number of unique states reachable within horizon steps.
    """
    
    # Create batch of initial states for parallel simulation
    batch_state = State(
        agent_pos=jnp.repeat(initial_state.agent_pos[None], n_trajectories, axis=0),
        goal_pos=jnp.repeat(initial_state.goal_pos[None], n_trajectories, axis=0),
        box_pos=jnp.repeat(initial_state.box_pos[None], n_trajectories, axis=0),
        trap_pos=jnp.repeat(initial_state.trap_pos[None], n_trajectories, axis=0),
        wall_pos=jnp.repeat(initial_state.wall_pos[None], n_trajectories, axis=0),
        helper_pos=jnp.repeat(initial_state.helper_pos[None], n_trajectories, axis=0),
        key_pos=jnp.repeat(initial_state.key_pos[None], n_trajectories, axis=0),
        time=jnp.repeat(initial_state.time, n_trajectories),
        terminal=jnp.repeat(initial_state.terminal, n_trajectories),
        freeze_timer=jnp.repeat(initial_state.freeze_timer, n_trajectories),
        has_key=jnp.repeat(initial_state.has_key[None], n_trajectories, axis=0)
    )
    
    # Simulate n-step trajectories with random agent actions
    def step_trajectory(carry, _):
        current_state, key = carry
        
        # Generate random actions for all agents
        key_splits = jax.random.split(key, env.num_agents + 2)
        key = key_splits[0]
        
        actions = {}
        for i, agent_id in enumerate(env.agents):
            actions[agent_id] = jax.random.randint(
                key_splits[i + 1], (n_trajectories,), 0, len(env.action_set)
            )
        
        # Helper does nothing during simulation
        helper_actions = jnp.full(
            (n_trajectories,), env.HelperActions.DO_NOTHING.value, dtype=jnp.int32
        )
        
        # Step environment for all trajectories
        _, next_state, _, _, _ = jax.vmap(
            lambda s, a_dict, h: env.step_env(key_splits[-1], s, a_dict, h)
        )(current_state, actions, helper_actions)
        
        return (next_state, key), next_state
    
    # Run n-step simulation
    key = jax.random.PRNGKey(0)
    initial_carry = (batch_state, key)
    (final_state, _), _ = jax.lax.scan(
        step_trajectory, initial_carry, None, length=horizon
    )
    
    # Extract final positions of the leader agent
    leader_final_positions = final_state.agent_pos[:, agent_idx, :]  # [n_trajectories, 2]
    
    # Count unique positions (discrete choice)
    # Convert 2D positions to 1D indices for easier uniqueness counting
    position_indices = (
        leader_final_positions[:, 0] * env.width + leader_final_positions[:, 1]
    )
    
    # Count unique positions
    unique_positions = jnp.unique(position_indices, size=env.height * env.width, fill_value=-1)
    # Count non-fill values
    discrete_choice = jnp.sum(unique_positions >= 0)
    
    return discrete_choice

def estimate_entropic_choice(
    env: MultiAgentGridWorld,
    initial_state: State,
    n_trajectories: int = 20,
    horizon: int = 3,
    agent_idx: int = 0,  # leader agent index
):
    """
    Calculate entropic choice for the leader agent.
    Returns the entropy of the probability distribution of the leader agent's final positions.
    """
    
    # Create batch of initial states for parallel simulation
    batch_state = State(
        agent_pos=jnp.repeat(initial_state.agent_pos[None], n_trajectories, axis=0),
        goal_pos=jnp.repeat(initial_state.goal_pos[None], n_trajectories, axis=0),
        box_pos=jnp.repeat(initial_state.box_pos[None], n_trajectories, axis=0),
        trap_pos=jnp.repeat(initial_state.trap_pos[None], n_trajectories, axis=0),
        wall_pos=jnp.repeat(initial_state.wall_pos[None], n_trajectories, axis=0),
        helper_pos=jnp.repeat(initial_state.helper_pos[None], n_trajectories, axis=0),
        key_pos=jnp.repeat(initial_state.key_pos[None], n_trajectories, axis=0),
        time=jnp.repeat(initial_state.time, n_trajectories),
        terminal=jnp.repeat(initial_state.terminal, n_trajectories),
        freeze_timer=jnp.repeat(initial_state.freeze_timer, n_trajectories),
        has_key=jnp.repeat(initial_state.has_key[None], n_trajectories, axis=0)
    )
    
    # Simulate n-step trajectories with random agent actions
    def step_trajectory(carry, _):
        current_state, key = carry
        
        # Generate random actions for all agents
        key_splits = jax.random.split(key, env.num_agents + 2)
        key = key_splits[0]
        
        actions = {}
        for i, agent_id in enumerate(env.agents):
            actions[agent_id] = jax.random.randint(
                key_splits[i + 1], (n_trajectories,), 0, len(env.action_set)
            )
        
        # Helper does nothing during simulation
        helper_actions = jnp.full(
            (n_trajectories,), env.HelperActions.DO_NOTHING.value, dtype=jnp.int32
        )
        
        # Step environment for all trajectories
        _, next_state, _, _, _ = jax.vmap(
            lambda s, a_dict, h: env.step_env(key_splits[-1], s, a_dict, h)
        )(current_state, actions, helper_actions)
        
        return (next_state, key), next_state
    
    # Run n-step simulation
    key = jax.random.PRNGKey(0)
    initial_carry = (batch_state, key)
    (final_state, _), _ = jax.lax.scan(
        step_trajectory, initial_carry, None, length=horizon
    )
    
    # Extract final positions of the leader agent
    leader_final_positions = final_state.agent_pos[:, agent_idx, :]  # [n_trajectories, 2]
    position_indices = (
        leader_final_positions[:, 0] * env.width + leader_final_positions[:, 1]
    )
    
    # Estimate probability distribution from samples
    total_states = env.height * env.width
    state_counts = jnp.bincount(position_indices, length=total_states)
    state_probs = state_counts / n_trajectories
    
    # Calculate entropy: H = -sum(p * log(p)) for p > 0
    entropic_choice = -jnp.sum(
        jnp.where(state_probs > 0, 
                 state_probs * jnp.log2(state_probs), 
                 0.0)
    )
    
    return entropic_choice


def estimate_immediate_choice(
    env: MultiAgentGridWorld,
    initial_state: State,
    agent_idx: int = 0,
):
    """
    Estimate immediate choice assuming uniform random policy over feasible actions.
    This provides a state-dependent reward signal for the helper.
    If we assume uniform random policy without any constraints, the immediate choice is the entropy of the uniform distribution over valid actions.
    Which means there would be no signal.
    
    The intuition: helpers should create states where the leader has more valid options.

    NOTE: This still doesn't work. Policy estimator is needed, but that's a very strong assumption. Will omit.
    """
    
    agent_pos = initial_state.agent_pos[agent_idx]
    
    def is_action_valid(action_idx):
        action = env.action_set[action_idx]
        next_pos = agent_pos + env.action_to_dir[action]
        
        # Check all constraints
        in_bounds = jnp.all((next_pos >= 0) & (next_pos < jnp.array([env.height, env.width])))
        
        no_wall_collision = ~jnp.any(jnp.all(next_pos == initial_state.wall_pos, axis=-1))
        no_box_collision = ~jnp.any(jnp.all(next_pos == initial_state.box_pos, axis=-1))
        
        # Could add more constraints like trap avoidance, agent collision, etc.
        no_trap_collision = ~jnp.any(jnp.all(next_pos == initial_state.trap_pos, axis=-1))
        no_agent_collision = ~jnp.any(jnp.all(next_pos == initial_state.agent_pos[agent_idx+1], axis=-1))
        
        return in_bounds & no_wall_collision & no_box_collision & no_trap_collision & no_agent_collision
    
    # Check validity of all actions
    action_validity = jax.vmap(is_action_valid)(jnp.arange(len(env.action_set)))
    num_valid_actions = jnp.sum(action_validity)
    
    # Ensure at least one valid action (STAY should always be valid)
    num_valid_actions = jnp.maximum(num_valid_actions, 1.0)
    
    # Immediate choice = entropy of uniform distribution over valid actions
    immediate_choice = jnp.log2(num_valid_actions)
    
    return immediate_choice