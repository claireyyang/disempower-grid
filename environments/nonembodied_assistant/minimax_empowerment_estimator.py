import jax
import jax.numpy as jnp
from .nonembodied_multiagent_gridworld import NonEmbodiedMultiAgentGridWorld, State
from .empowerment_estimator import estimate_empowerment_monte_carlo
from functools import partial


@partial(jax.jit, static_argnums=(0,))
def estimate_minimax_regret_empowerment(
    env: NonEmbodiedMultiAgentGridWorld,
    state: State,
    n_trajectories: int = 10,
    horizon: int = 3,
):
    """
    Estimate the negative minimax regret over both agents' empowerment.

    This function evaluates all possible helper actions and computes:
    1. For each action, estimate empowerment for both agents
    2. For each agent, compute regret = max_empowerment - current_empowerment
    3. For each action, find maximum regret across agents
    4. Return negative of minimum maximum regret (so maximizing this minimizes regret)

    Args:
        env: MultiAgentGridWorld instance
        state: Current state
        n_trajectories: Number of trajectories for empowerment estimation
        horizon: Horizon for empowerment estimation

    Returns:
        Negative minimax regret value (higher is better for training)
    """
    key = jax.random.PRNGKey(0)

    def evaluate_action_empowerment(action):
        """Evaluate empowerment after taking a helper action."""
        # Create dummy agent actions (random)
        agent_actions = {}
        action_key, _ = jax.random.split(key)
        random_actions = jax.random.randint(
            action_key, shape=(len(env.agents),), minval=0, maxval=len(env.action_set)
        )
        for i, agent_id in enumerate(env.agents):
            agent_actions[agent_id] = random_actions[i]

        # Step environment with this helper action
        step_key, _ = jax.random.split(action_key)
        _, next_state, _, _, _ = env.step_env(
            step_key, state, agent_actions, action
        )

        # Estimate empowerment for both agents
        empowerment = estimate_empowerment_monte_carlo(env, next_state, n_trajectories, horizon)
        return jnp.array(empowerment)  # [num_agents]

    # Evaluate empowerment for all possible helper actions
    helper_actions = jnp.arange(len(env.HelperActions))
    empowerment_values = jax.vmap(evaluate_action_empowerment)(helper_actions)
    # empowerment_values shape: [num_actions, num_agents]

    # Compute minimax regret
    # For each agent, find the best possible empowerment across all actions
    max_empowerment_per_agent = jnp.max(empowerment_values, axis=0)  # [num_agents]

    # For each action, compute regret for each agent
    regret_per_action_agent = max_empowerment_per_agent[None, :] - empowerment_values  # [num_actions, num_agents]

    # For each action, find the maximum regret across agents
    max_regret_per_action = jnp.max(regret_per_action_agent, axis=1)  # [num_actions]

    # Find the minimum of maximum regrets (the minimax regret value)
    minimax_regret = jnp.min(max_regret_per_action)

    # Return negative regret (so that maximizing this objective minimizes regret)
    return -minimax_regret