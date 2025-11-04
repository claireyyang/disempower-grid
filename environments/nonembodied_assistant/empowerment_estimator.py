import jax
import jax.numpy as jnp
from .nonembodied_multiagent_gridworld import (
    NonEmbodiedMultiAgentGridWorld,
    State,
)
from functools import partial


@partial(jax.jit, static_argnums=(2, 3))
def estimate_empowerment_variance_proxy(
    env: NonEmbodiedMultiAgentGridWorld,
    initial_state: State,
    n_trajectories: int = 20,
    horizon: int = 3,
):
    """
    Estimate empowerment using diversity of reachable states after random action sequences.
    Now compatible with both single states and batched states from vmap.

    Args:
        env: NonEmbodiedMultiAgentGridWorld instance
        initial_state: State object (can be a single state or batched)
        policy: Helper policy
        n_trajectories: Number of random trajectories to sample (N in the algorithm)
        horizon: Length of each trajectory (T in the algorithm)
    """

    key = jax.random.PRNGKey(0)
    # Create batch of n_trajectories copies of the initial state
    batch_state = State(
        agent_pos=jnp.repeat(
            initial_state.agent_pos[None], n_trajectories, axis=0
        ),
        goal_pos=jnp.repeat(
            initial_state.goal_pos[None], n_trajectories, axis=0
        ),
        box_pos=jnp.repeat(initial_state.box_pos[None], n_trajectories, axis=0),
        trap_pos=jnp.repeat(
            initial_state.trap_pos[None], n_trajectories, axis=0
        ),
        wall_pos=jnp.repeat(
            initial_state.wall_pos[None], n_trajectories, axis=0
        ),
        time=jnp.repeat(initial_state.time, n_trajectories),
        terminal=jnp.repeat(initial_state.terminal, n_trajectories),
        freeze_timer=jnp.repeat(initial_state.freeze_timer, n_trajectories)
    )

    # Vectorized step function for all trajectories
    def step_trajectory(carry, _):
        current_state, key = carry

        # Generate random actions for all trajectories at once
        key_splits = jax.random.split(key, env.num_agents + 2)
        key = key_splits[0]

        # Sample actions for agents across all trajectories
        actions = {}
        for i, agent_id in enumerate(env.agents):
            actions[agent_id] = jax.random.randint(
                key_splits[i + 1], (n_trajectories,), 0, len(env.action_set)
            )

        # Get helper actions for all trajectories
        key, subkey = jax.random.split(key)
        helper_actions = jnp.full(
            (n_trajectories,), env.HelperActions.DO_NOTHING.value, dtype=jnp.int32
        )
        if helper_actions.ndim == 2:
            helper_actions = helper_actions.reshape(-1)

        # Step environment for all trajectories in parallel
        _, next_state, _, _, _ = jax.vmap(
            lambda s, a_dict, h: env.step_env(key, s, a_dict, h)
        )(current_state, actions, helper_actions)

        return (next_state, key), next_state

    # Run all trajectories in parallel using scan
    initial_carry = (batch_state, key)
    (final_state, _), _ = jax.lax.scan(
        step_trajectory, initial_carry, None, length=horizon
    )

    # Calculate empowerment for each agent using final positions
    empowerment_by_agent = []
    for agent_idx in range(env.num_agents):
        # Get final positions for this agent across all trajectories
        agent_positions = final_state.agent_pos[
            :, agent_idx, :
        ]  # Shape: [n_trajectories, 2]
        # Calculate variance across trajectories
        emp = jnp.var(agent_positions, axis=0).sum()
        empowerment_by_agent.append(emp)
    return jnp.array(empowerment_by_agent)


@partial(jax.jit, static_argnums=(0, 2, 3))
def estimate_empowerment_monte_carlo(
    env: NonEmbodiedMultiAgentGridWorld,
    initial_state: State,
    n_trajectories: int = 10,
    horizon: int = 3,
):
    key = jax.random.PRNGKey(0)
    empowerment_by_agent = []

    for agent_idx, agent_id in enumerate(env.agents):
        # Initialize grid to collect final positions for each action
        all_final_positions = []

        # For each possible first action
        for action in range(len(env.action_set)):
            # Create batch state
            batch_state = State(
                agent_pos=jnp.repeat(
                    initial_state.agent_pos[None], n_trajectories, axis=0
                ),
                goal_pos=jnp.repeat(
                    initial_state.goal_pos[None], n_trajectories, axis=0
                ),
                box_pos=jnp.repeat(initial_state.box_pos[None], n_trajectories, axis=0),
                trap_pos=jnp.repeat(
                    initial_state.trap_pos[None], n_trajectories, axis=0
                ),
                wall_pos=jnp.repeat(
                    initial_state.wall_pos[None], n_trajectories, axis=0
                ),
                time=jnp.repeat(initial_state.time, n_trajectories),
                terminal=jnp.repeat(initial_state.terminal, n_trajectories),
                freeze_timer=jnp.repeat(initial_state.freeze_timer, n_trajectories),
            )

            # Prepare first step actions
            first_step_actions = {}
            for i, aid in enumerate(env.agents):
                if aid == agent_id:
                    first_step_actions[aid] = jnp.full((n_trajectories,), action)
                else:
                    # NOTE: currently the other agent moves randomly - don't want to assume that the agents' policies are known to the assistant for calculation
                    key, subkey = jax.random.split(key)
                    first_step_actions[aid] = jax.random.randint(
                        subkey, (n_trajectories,), 0, len(env.action_set)
                    )
            key, subkey = jax.random.split(key)
            helper_actions = jnp.full(
                (n_trajectories,), env.HelperActions.DO_NOTHING.value, dtype=jnp.int32
            )
            if helper_actions.ndim == 2:
                helper_actions = helper_actions.reshape(-1)

            # Step environment for first action
            key, step_key = jax.random.split(key)
            _, next_state, _, _, _ = jax.vmap(
                lambda s, a_dict, h: env.step_env(step_key, s, a_dict, h)
            )(batch_state, first_step_actions, helper_actions)

            # Define the step function for scan
            def step_trajectory(carry, _):
                current_state, key = carry

                # NOTE: currently the agents move randomly - don't want to assume that the agents' policies are known to the assistant for calculation
                random_actions = {}
                key_splits = jax.random.split(key, env.num_agents + 2)
                key = key_splits[0]

                for i, aid in enumerate(env.agents):
                    random_actions[aid] = jax.random.randint(
                        key_splits[i + 1], (n_trajectories,), 0, len(env.action_set)
                    )

                key, subkey = jax.random.split(key)
                helper_actions = jnp.full(
                    (n_trajectories,),
                    env.HelperActions.DO_NOTHING.value,
                    dtype=jnp.int32,
                )
                if helper_actions.ndim == 2:
                    helper_actions = helper_actions.reshape(-1)

                # Step environment
                _, next_state, _, _, _ = jax.vmap(
                    lambda s, a_dict, h: env.step_env(key_splits[-1], s, a_dict, h)
                )(current_state, random_actions, helper_actions)

                return (next_state, key), next_state

            # Run remaining K-1 steps using scan
            initial_carry = (next_state, key)
            (final_state, _), _ = jax.lax.scan(
                step_trajectory, initial_carry, None, length=horizon - 1
            )

            # Extract final positions for this agent
            final_positions = final_state.agent_pos[:, agent_idx]
            all_final_positions.append(final_positions)

        # Stack all positions by action
        all_final_positions = jnp.stack(
            all_final_positions
        )  # [num_actions, n_trajectories, 2]

        # Compute empirical distributions using histograms
        # Create one-hot encodings of positions
        position_indices = (
            all_final_positions[..., 0] * env.width + all_final_positions[..., 1]
        )
        position_one_hot = jax.nn.one_hot(position_indices, env.height * env.width)

        # Compute p(s+|s,a) for each action
        p_s_plus_given_s_a = jnp.mean(
            position_one_hot, axis=1
        )  # [num_actions, height*width]
        p_s_plus_given_s_a = p_s_plus_given_s_a.reshape(
            len(env.action_set), env.height, env.width
        )

        # Compute p(s+|s) by marginalizing over actions
        p_s_plus_given_s = jnp.mean(p_s_plus_given_s_a, axis=0)  # [height, width]

        # Compute mutual information
        mi_terms = jnp.where(
            (p_s_plus_given_s_a > 0) & (p_s_plus_given_s > 0)[None, ...],
            p_s_plus_given_s_a
            * jnp.log2(p_s_plus_given_s_a / p_s_plus_given_s[None, ...]),
            0.0,
        )

        # Weight by uniform action probability and sum
        p_a_given_s = 1.0 / len(env.action_set)
        mi = p_a_given_s * jnp.sum(mi_terms)

        empowerment_by_agent.append(mi)

    return empowerment_by_agent
