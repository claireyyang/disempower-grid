from .multiagent_gridworld import MultiAgentGridWorld
import jax
import jax.numpy as jnp


def reset_envs(master_key, env: MultiAgentGridWorld, num_trajectories: int):
    split_keys = jax.random.split(master_key, num_trajectories)
    return jax.vmap(lambda env_key: env.reset(env_key)[1])(split_keys)


def reset_envs_specific_positions(
    master_key, env: MultiAgentGridWorld, specific_position_reset: dict, num_trajectories: int
):
    split_keys = jax.random.split(master_key, num_trajectories)
    agent_pos = jnp.array(specific_position_reset["agent_pos"], dtype=jnp.int32)
    goal_pos = jnp.array(specific_position_reset["goal_pos"], dtype=jnp.int32)
    box_pos = jnp.array(specific_position_reset["box_pos"], dtype=jnp.int32)
    helper_pos = jnp.array(specific_position_reset["helper_pos"], dtype=jnp.int32)
    key_pos = jnp.array(specific_position_reset["key_pos"], dtype=jnp.int32)
    trap_pos = jnp.array(specific_position_reset["trap_pos"], dtype=jnp.int32)
    wall_pos = jnp.array(specific_position_reset["wall_pos"], dtype=jnp.int32)
    return jax.vmap(
        lambda env_key: env.reset_specific_positions(
            env_key, agent_pos, goal_pos, box_pos, trap_pos, wall_pos, helper_pos, key_pos
        )[1]
    )(split_keys)


def step_envs(master_key, states, actions, helper_actions, env: MultiAgentGridWorld, num_trajectories: int):
    split_keys = jax.random.split(master_key, num_trajectories)
    print(f"In step_envs: States: {states}")
    print(f"In step_envs: Actions: {actions}")
    print(f"In step_envs: Helper actions: {helper_actions}")
    print(f"In step_envs: num_trajectories: {num_trajectories}")
    return jax.vmap(
        lambda env_key, state, action, helper_action: env.step_env(
            env_key, state, action, helper_action
        )
    )(split_keys, states, actions, helper_actions)
