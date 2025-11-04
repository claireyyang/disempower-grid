"""
Train embodied assistant environments.
"""

import distrax
from networks.actor_critic_agents import ActorCriticAgent
from networks.actor_critic_assistant import ActorCriticAssistant
from environments.embodied_assistant.environment import reset_envs_specific_positions
from environments.embodied_assistant.environment import step_envs
from utils.file_utils import load_specific_positions_from_file
from environments.embodied_assistant.observation import (
    convert_state_to_grid_observation,
    convert_state_to_partial_grid_observation,
    convert_state_to_flattened_vector,
)
from environments.embodied_assistant.multiagent_gridworld import (
    MultiAgentGridWorld,
    EnvMode,
    Actions,
)
from utils.helper_objectives_enum import HelperObjective
import jax
import jax.numpy as jnp
import wandb
import chex
from typing import Tuple, NamedTuple, Sequence, Any
from utils.argparse_utils import create_parser
import json
import os
from datetime import datetime
import optax
from flax import nnx
from environments.embodied_assistant.empowerment_estimator import (
    estimate_empowerment_monte_carlo,
    estimate_empowerment_variance_proxy,
)
from environments.embodied_assistant.choice_estimator import (
    estimate_discrete_choice,
    estimate_entropic_choice,
)
from environments.embodied_assistant.minimax_empowerment_estimator import (
    estimate_minimax_regret_empowerment,
)
import pickle
import orbax.checkpoint as ocp
from environments.embodied_assistant.viz import (
    create_frame_data,
    create_trajectory_viewer_px,
)
import csv
import pdb
from utils.random_helper_policy import RandomPolicy


class Transition(NamedTuple):
    """Transition tuple for PPO rollouts."""

    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    """Convert agent dict to batched array."""
    x = jnp.stack([x[a] for a in agent_list])
    # Check if this is a grid observation (has spatial dimensions)
    if (
        len(x.shape) > 2
    ):  # Grid observation: (num_agents, num_envs, height, width, channels)
        # Reshape to (num_actors, height, width, channels) preserving spatial structure
        return x.reshape((num_actors,) + x.shape[2:])
    else:  # Vector observation
        return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """Convert batched array back to agent dict."""
    if len(x.shape) > 2:  # Grid observation: preserve spatial dimensions
        # x has shape (num_actors, height, width, channels) or similar
        # Reshape to (num_agents, num_envs, height, width, channels)
        x = x.reshape((len(agent_list), num_envs) + x.shape[1:])
    else:  # Vector observation
        x = x.reshape((len(agent_list), num_envs, -1))
    return {a: x[i].squeeze() for i, a in enumerate(agent_list)}


def create_ppo_config():
    """Create PPO configuration."""
    return {
        "NUM_ENVS": args.num_trajectories,
        "NUM_STEPS": args.max_steps,
        "TOTAL_TIMESTEPS": args.epochs * args.max_steps * args.num_trajectories,
        "NUM_UPDATES": args.epochs,
        "NUM_MINIBATCHES": 4,
        "UPDATE_EPOCHS": 4,
        "LR": 3e-4,
        "CLIP_EPS": 0.2,
        "GAE_LAMBDA": 0.95,
        "GAMMA": 0.99,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.1,  # increased from 0.01 to 0.05 to 0.1
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
    }


def make_ppo_train_phase1(
    envs: MultiAgentGridWorld, helper_policy: RandomPolicy, wandb_id: str
):
    """Create PPO training function for phase 1 (train agents with random helper)."""
    config = create_ppo_config()
    config["NUM_ACTORS"] = envs.num_agents * config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train_phase1(rng, specific_position_reset=None):

        # Create observation from state
        def state_to_obs(state):
            # convert_state_to_grid_observation now returns (height, width, channels)
            return convert_state_to_grid_observation(state, envs.height, envs.width)

        # INIT SEPARATE NETWORKS - but combine them into a single model for training
        print("STARTING TRAIN PHASE 1")
        rng, _rng = jax.random.split(rng)

        # Create separate networks for each agent
        agent_networks = {}
        network_rngs = jax.random.split(_rng, envs.num_agents)
        for i in range(envs.num_agents):
            agent_networks[f"agent_{i}"] = ActorCriticAgent(
                len(Actions),
                envs.width,
                envs.height,
                num_agents=envs.num_agents,
                activation=config["ACTIVATION"],
                rngs=nnx.Rngs(network_rngs[i]),
            )

        # Create a combined network class that holds all agent networks
        class MultiAgentNetwork(nnx.Module):
            def __init__(self, agent_networks):
                self.agent_networks = agent_networks
                self.num_agents = len(agent_networks)

            def __call__(self, obs_batch, agent_ids):
                # obs_batch shape: (batch_size, height, width, channels)
                # agent_ids shape: (batch_size,)
                batch_size = obs_batch.shape[0]

                # Process each observation individually and route to correct network
                def process_single_obs(obs_and_id):
                    obs, agent_id = obs_and_id
                    obs = obs[None, ...]  # Add batch dimension

                    # Use conditional logic to route to correct network
                    def agent_0_forward(obs):
                        return self.agent_networks["agent_0"](obs)

                    def agent_1_forward(obs):
                        return self.agent_networks["agent_1"](obs)

                    # Route based on agent ID
                    if self.num_agents == 2:
                        pi, value = jax.lax.cond(
                            agent_id == 0, agent_0_forward, agent_1_forward, obs
                        )
                    else:
                        # Single agent case
                        pi, value = agent_0_forward(obs)

                    return pi.logits[0], value[0]  # Remove batch dimension

                # Vectorize over the batch
                logits, values = jax.vmap(process_single_obs)((obs_batch, agent_ids))

                combined_pi = distrax.Categorical(logits=logits)
                return combined_pi, values

        network = MultiAgentNetwork(agent_networks)

        # Get observation space size from grid observation
        test_state = reset_envs_specific_positions(
            _rng, envs, specific_position_reset, 1
        )
        test_obs = jax.vmap(state_to_obs)(test_state)
        print(f"test_obs shape: {test_obs.shape}")

        # Initialize network with dummy input and warmup
        init_x = jnp.tile(test_obs, (2, 1, 1, 1))  # Shape: (2, height, width, channels)
        test_agent_ids = jnp.array([0, 1])  # Shape: (2,)
        _ = network(init_x, test_agent_ids)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        optimizer = nnx.Optimizer(network, tx)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        if specific_position_reset:
            env_state = reset_envs_specific_positions(
                _rng, envs, specific_position_reset, config["NUM_ENVS"]
            )

        obsv = jax.vmap(state_to_obs)(
            env_state
        )  # This will be (NUM_ENVS, height, width, channels)
        print(f"obsv shape: {obsv.shape}")
        obsv = {f"agent_{i}": obsv for i in range(envs.num_agents)}

        # Split optimizer for JAX compatibility
        optimizer_def, optimizer_state = nnx.split(optimizer)

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            # Reset environment for new epoch
            optimizer_state, env_state, last_obs, helper_policy_state, rng = (
                runner_state
            )
            rng, reset_rng = jax.random.split(rng)
            if specific_position_reset:
                env_state = reset_envs_specific_positions(
                    reset_rng, envs, specific_position_reset, config["NUM_ENVS"]
                )
            obsv = jax.vmap(state_to_obs)(env_state)
            obsv = {f"agent_{i}": obsv for i in range(envs.num_agents)}
            runner_state = (optimizer_state, env_state, obsv, helper_policy_state, rng)

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                optimizer_state, env_state, last_obs, helper_policy_state, rng = (
                    runner_state
                )
                print(f"ENV STATE: {env_state}")
                # Reconstruct optimizer for this step
                optimizer = nnx.merge(optimizer_def, optimizer_state)

                # SELECT ACTION using PPO network with agent IDs
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, envs.agents, config["NUM_ACTORS"])
                print(f"OBS BATCH: {obs_batch}")

                # Create agent IDs for the batch
                # First half are agent 0, second half are agent 1
                agent_ids = jnp.concatenate(
                    [
                        jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32),  # Agent 0
                        jnp.ones(config["NUM_ENVS"], dtype=jnp.int32),  # Agent 1
                    ]
                )
                print(f"Agent IDs: {agent_ids}")

                pi, value = optimizer.model(obs_batch, agent_ids)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, envs.agents, config["NUM_ENVS"], envs.num_agents
                )
                print(f"ENV ACT: {env_act}")
                # Get helper action (random policy)
                vectorized_states = convert_state_to_flattened_vector(env_state)
                helper_action, helper_action_infos = helper_policy_state.next_action(
                    vectorized_states
                )
                print(f"HELPER ACTION: {helper_action}")

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                next_obs, env_state, reward, done, info = step_envs(
                    _rng, env_state, env_act, helper_action, envs, config["NUM_ENVS"]
                )

                # Convert next_obs
                next_obsv = jax.vmap(state_to_obs)(env_state)
                next_obsv = {f"agent_{i}": next_obsv for i in range(envs.num_agents)}

                print(f"IN _ENV_STEP: done: {done}")
                print(f"IN _ENV_STEP: reward: {reward}")
                # first convert reward into a dictionary for batchify
                reward_dict = {
                    f"agent_{i}": reward[:, i] for i in range(envs.num_agents)
                }
                print(f"IN _ENV_STEP: reward_dict: {reward_dict}")

                # Add helper actions to info for logging
                # Repeat helper_action to match the batch size for agents
                helper_action_expanded = jnp.repeat(
                    helper_action, envs.num_agents, axis=0
                )
                info["helper_actions"] = helper_action_expanded

                transition = Transition(
                    batchify(done, envs.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward_dict, envs.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info if isinstance(info, dict) else {},
                )

                # Split optimizer state back for scan
                _, updated_optimizer_state = nnx.split(optimizer)
                runner_state = (
                    updated_optimizer_state,
                    env_state,
                    next_obsv,
                    helper_policy_state,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            print(f"TRAJ BATCH: {traj_batch}")

            # CALCULATE ADVANTAGE
            optimizer_state, env_state, last_obs, helper_policy_state, rng = (
                runner_state
            )
            # Reconstruct optimizer for advantage calculation
            optimizer = nnx.merge(optimizer_def, optimizer_state)
            last_obs_batch = batchify(last_obs, envs.agents, config["NUM_ACTORS"])
            # Create agent IDs for the batch
            last_agent_ids = jnp.concatenate(
                [
                    jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32),  # Agent 0
                    jnp.ones(config["NUM_ENVS"], dtype=jnp.int32),  # Agent 1
                ]
            )
            _, last_val = optimizer.model(last_obs_batch, last_agent_ids)

            # Calculate episode metrics
            episode_returns = traj_batch.reward.sum(axis=0)  # Sum over timesteps
            episode_lengths = config["NUM_STEPS"] * jnp.ones_like(episode_returns)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                @nnx.scan
                def _update_minibatch(optimizer, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(model, traj_batch, gae, targets):
                        # RERUN NETWORK
                        # Reconstruct agent IDs for the batch
                        batch_size = traj_batch.obs.shape[0]
                        agent_ids_for_loss = jnp.concatenate(
                            [
                                jnp.zeros(batch_size // 2, dtype=jnp.int32),  # Agent 0
                                jnp.ones(batch_size // 2, dtype=jnp.int32),  # Agent 1
                            ]
                        )
                        pi, value = model(traj_batch.obs, agent_ids_for_loss)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        optimizer.model, traj_batch, advantages, targets
                    )
                    optimizer.update(grads)
                    return optimizer, total_loss

                optimizer_state, traj_batch, advantages, targets, rng = update_state
                # Reconstruct optimizer for updates
                optimizer = nnx.merge(optimizer_def, optimizer_state)
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                optimizer, total_loss = _update_minibatch(optimizer, minibatches)
                # Split optimizer back to state for scan
                _, updated_optimizer_state = nnx.split(optimizer)
                update_state = (
                    updated_optimizer_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (optimizer_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            optimizer_state = update_state[0]
            # Reconstruct optimizer for final step
            optimizer = nnx.merge(optimizer_def, optimizer_state)
            metric = {
                "episode_returns": episode_returns,
                "episode_lengths": episode_lengths,
            }
            if hasattr(traj_batch, "info") and traj_batch.info:
                metric.update(traj_batch.info)
            rng = update_state[-1]

            # Collect metrics for logging outside scan
            # Extract loss components from loss_info (shape: [UPDATE_EPOCHS, NUM_MINIBATCHES])
            total_losses, (value_losses, actor_losses, entropies) = loss_info

            # Collect per-agent returns data
            agent_returns_data = {}
            for i in range(envs.num_agents):
                agent_returns = episode_returns[
                    i :: envs.num_agents
                ]  # Extract returns for this agent
                agent_returns_data[f"agent_{i}_returns"] = agent_returns

            # Collect all data needed for logging
            logging_metrics = {
                "episode_returns": episode_returns,
                "episode_lengths": episode_lengths,
                "total_losses": total_losses,
                "value_losses": value_losses,
                "actor_losses": actor_losses,
                "entropies": entropies,
                "advantages": advantages,
                "agent_returns_data": agent_returns_data,
                "agent_actions": traj_batch.action,
                "helper_actions": traj_batch.info.get(
                    "helper_actions",
                    jnp.zeros((config["NUM_STEPS"], config["NUM_ENVS"])),
                ),  # Extract helper actions from info
            }

            print(f"IN _UPDATE_STEP: logging_metrics: {logging_metrics}")

            runner_state = (
                optimizer_state,
                env_state,
                last_obs,
                helper_policy_state,
                rng,
            )
            return runner_state, (metric, logging_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (optimizer_state, env_state, obsv, helper_policy, _rng)
        runner_state, outputs = jax.lax.scan(
            _update_step,
            runner_state,
            jnp.arange(config["NUM_UPDATES"]),
            config["NUM_UPDATES"],
        )

        # Unpack outputs from scan
        metric, logging_metrics_all = outputs

        # Log Phase 1 metrics outside scan
        for update_idx in range(config["NUM_UPDATES"]):
            # Extract metrics for this epoch
            epoch_logging_metrics = jax.tree_util.tree_map(
                lambda x: x[update_idx], logging_metrics_all
            )

            # Create combined metrics dictionary for this epoch
            epoch_metrics = {
                "phase1/episode_returns_mean": float(
                    jnp.mean(epoch_logging_metrics["episode_returns"])
                ),
                "phase1/episode_returns_std": float(
                    jnp.std(epoch_logging_metrics["episode_returns"])
                ),
                "phase1/episode_lengths_mean": float(
                    jnp.mean(epoch_logging_metrics["episode_lengths"])
                ),
                "phase1/total_loss": float(
                    jnp.mean(epoch_logging_metrics["total_losses"])
                ),
                "phase1/value_loss": float(
                    jnp.mean(epoch_logging_metrics["value_losses"])
                ),
                "phase1/actor_loss": float(
                    jnp.mean(epoch_logging_metrics["actor_losses"])
                ),
                "phase1/entropy": float(jnp.mean(epoch_logging_metrics["entropies"])),
                "phase1/advantages_mean": float(
                    jnp.mean(epoch_logging_metrics["advantages"])
                ),
                "phase1/advantages_std": float(
                    jnp.std(epoch_logging_metrics["advantages"])
                ),
                "phase1/epoch": update_idx,
                "phase1/episode_returns_min": float(
                    jnp.min(epoch_logging_metrics["episode_returns"])
                ),
                "phase1/episode_returns_max": float(
                    jnp.max(epoch_logging_metrics["episode_returns"])
                ),
                "phase1/episode_returns_median": float(
                    jnp.median(epoch_logging_metrics["episode_returns"])
                ),
                "phase1/episode_returns_q25": float(
                    jnp.percentile(epoch_logging_metrics["episode_returns"], 25)
                ),
                "phase1/episode_returns_q75": float(
                    jnp.percentile(epoch_logging_metrics["episode_returns"], 75)
                ),
            }

            # Add per-agent metrics
            for i in range(envs.num_agents):
                agent_returns = epoch_logging_metrics["agent_returns_data"][
                    f"agent_{i}_returns"
                ]
                epoch_metrics.update(
                    {
                        f"phase1/agent_{i}_returns_mean": float(
                            jnp.mean(agent_returns)
                        ),
                        f"phase1/agent_{i}_returns_std": float(jnp.std(agent_returns)),
                        f"phase1/agent_{i}_success_rate": float(
                            jnp.mean(agent_returns > 0)
                        ),
                    }
                )

            # Add action logging for Phase 1
            if update_idx < config["NUM_UPDATES"]:  # Only log for valid indices
                # Log agent actions
                agent_actions = epoch_logging_metrics.get("agent_actions", None)
                if agent_actions is not None:
                    # Unbatchify agent actions to get per-agent means
                    agent_actions_unbatched = unbatchify(
                        agent_actions, envs.agents, config["NUM_ENVS"], envs.num_agents
                    )
                    for i, agent_id in enumerate(envs.agents):
                        if agent_id in agent_actions_unbatched:
                            epoch_metrics[f"phase1/actions/{agent_id}_mean"] = float(
                                jnp.mean(agent_actions_unbatched[agent_id])
                            )

                # Log helper actions
                helper_actions = epoch_logging_metrics.get("helper_actions", None)
                if helper_actions is not None:
                    epoch_metrics["phase1/actions/helper_mean"] = float(
                        jnp.mean(helper_actions)
                    )

                    # Log all metrics for this epoch

                wandb.log(epoch_metrics, step=update_idx)

            # visualize one of the trajectories here
            if update_idx == 1 or update_idx == 249:
                # Recreate the trajectory by replaying actions

                # Reset environment to get initial state (same as training)
                rng_replay, _ = jax.random.split(rng)
                if specific_position_reset:
                    initial_state = reset_envs_specific_positions(
                        rng_replay, envs, specific_position_reset, 1
                    )

                # Extract actions for first trajectory
                helper_actions_traj = epoch_logging_metrics.get(
                    "helper_actions", jnp.zeros((config["NUM_STEPS"],))
                )[:, 0]
                agent_actions = epoch_logging_metrics.get("agent_actions", None)

                if agent_actions is not None:
                    # Extract actions for first trajectory (environment 0)
                    # Actions are batched as: [agent0_env0, agent0_env1, ..., agent1_env0, agent1_env1, ...]
                    # pdb.set_trace()
                    print(f"!!! agent_actions shape: {agent_actions.shape}")
                    print(f"NUM_ENVS: {config['NUM_ENVS']}")
                    agent_0_actions_traj = agent_actions[
                        :, 0
                    ]  # Agent 0, first trajectory
                    agent_1_actions_traj = agent_actions[
                        :, config["NUM_ENVS"]
                    ]  # Agent 1, first trajectory
                    print(f"Agent 0 actions (first 5): {agent_0_actions_traj[:5]}")
                    print(f"Agent 1 actions (first 5): {agent_1_actions_traj[:5]}")

                    agent_actions_dict = {
                        "agent_0": [agent_0_actions_traj],
                        "agent_1": [agent_1_actions_traj],
                    }

                    # Replay the episode step by step and collect rewards
                    def replay_step(state, actions_at_step):
                        agent_0_action, agent_1_action, helper_action = actions_at_step

                        # Reconstruct agent actions in the format expected by step_envs
                        env_act = {
                            "agent_0": jnp.array(
                                [agent_0_action]
                            ),  # Wrap in array for single env
                            "agent_1": jnp.array([agent_1_action]),
                        }

                        # Step the environment
                        rng_step, _ = jax.random.split(rng_replay)
                        _, next_state, reward, _, _ = step_envs(
                            rng_step,
                            state,
                            env_act,
                            jnp.array([helper_action]),
                            envs,
                            1,
                        )

                        return next_state, (
                            state,
                            reward,
                        )  # Return state and reward for collection

                    # Prepare action sequences
                    action_sequence = jnp.stack(
                        [
                            agent_0_actions_traj,
                            agent_1_actions_traj,
                            helper_actions_traj,
                        ],
                        axis=1,
                    )  # Shape: (NUM_STEPS, 3)

                    # Replay all steps
                    final_state, (states_trajectory, rewards_trajectory) = jax.lax.scan(
                        replay_step, initial_state, action_sequence
                    )

                    # Extract rewards for visualization (squeeze out env dimension)
                    episode_rewards = jnp.squeeze(
                        rewards_trajectory, axis=1
                    )  # Shape: (NUM_STEPS, num_agents)

                    # Create frames from the replayed states
                    states_trajectory_squeezed = jax.tree_util.tree_map(
                        lambda x: jnp.squeeze(x, axis=1), states_trajectory
                    )
                    frames = jax.vmap(lambda s: create_frame_data(s, envs), in_axes=0)(
                        states_trajectory_squeezed
                    )

                    fig = create_trajectory_viewer_px(
                        todays_date + "_" + wandb_id,
                        frames,
                        helper_actions_traj,
                        agent_actions_dict,
                        envs.no_freeze,
                        envs.no_pull_helper_action,
                        agent_rewards=episode_rewards,  # Pass the actual rewards
                    )
                    fig.write_html(
                        f"train_data/{todays_date}/{wandb_id}/phase_1_{wandb_id}_epoch_{update_idx}_testing_episode.html",
                        auto_play=False,
                    )

        # Reconstruct final optimizer for return
        final_optimizer_state = runner_state[0]
        final_optimizer = nnx.merge(optimizer_def, final_optimizer_state)
        final_runner_state = (
            final_optimizer,
            runner_state[1],
            runner_state[2],
            runner_state[3],
            runner_state[4],
        )
        return {"runner_state": final_runner_state, "metrics": metric}

    return train_phase1


def make_ppo_train_phase2(
    envs: MultiAgentGridWorld,
    frozen_agent_graph_def,
    frozen_agent_state,
    helper_objective: HelperObjective,
    num_epochs_phase1: int,
    num_epochs_phase2: int,
    helper_idx: int,
    wandb_id: str,
):
    """Create PPO training function for phase 2 (train helper with frozen agents)."""
    config = create_ppo_config()
    config["NUM_ACTORS"] = 1 * config["NUM_ENVS"]  # Only helper is learning
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train_phase2(rng, specific_position_reset=None):
        rng, helper_rng, agent_rng = jax.random.split(rng, 3)
        helper_network = ActorCriticAssistant(
            len(envs.HelperActions),
            envs.width,
            envs.height,
            activation=config["ACTIVATION"],
            num_agents=envs.num_agents,
            rngs=nnx.Rngs(helper_rng),
        )
        # Reconstruct frozen agent network from graph_def and state
        agent_network = nnx.merge(frozen_agent_graph_def, frozen_agent_state)

        # Create observation from state
        def state_to_partial_obs(state):
            # convert_state_to_partial_grid_observation now returns (height, width, channels)
            return convert_state_to_partial_grid_observation(
                state, envs.height, envs.width
            )

        # Create observation from state
        def state_to_obs(state):
            # convert_state_to_grid_observation now returns (height, width, channels)
            return convert_state_to_grid_observation(state, envs.height, envs.width)

        # Get observation space size from grid observation
        test_state = reset_envs_specific_positions(
            rng, envs, specific_position_reset, 1
        )  # Get a sample state
        test_obs = jax.vmap(state_to_partial_obs)(test_state)
        init_x = jnp.zeros(test_obs.shape)

        print(f"Assistant:init_x shape: {init_x.shape}")

        # Initialize helper network with dummy input
        _ = helper_network(init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        helper_optimizer = nnx.Optimizer(helper_network, tx)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        if specific_position_reset:
            env_state = reset_envs_specific_positions(
                _rng, envs, specific_position_reset, config["NUM_ENVS"]
            )

        agent_obsv = jax.vmap(state_to_obs)(
            env_state
        )  # This will be (NUM_ENVS, height, width, channels)
        agent_obsv = {f"agent_{i}": agent_obsv for i in range(envs.num_agents)}
        helper_obsv = jax.vmap(state_to_partial_obs)(
            env_state
        )  # This will be (NUM_ENVS, height, width, channels)

        # Split helper optimizer for JAX compatibility
        helper_optimizer_def, helper_optimizer_state = nnx.split(helper_optimizer)

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            # Reset environment for new epoch
            helper_optimizer_state, env_state, last_agent_obs, last_helper_obs, rng = (
                runner_state
            )
            rng, reset_rng = jax.random.split(rng)
            if specific_position_reset:
                env_state = reset_envs_specific_positions(
                    reset_rng, envs, specific_position_reset, config["NUM_ENVS"]
                )
            agent_obsv = jax.vmap(state_to_obs)(env_state)
            agent_obsv = {f"agent_{i}": agent_obsv for i in range(envs.num_agents)}
            helper_obsv = jax.vmap(state_to_partial_obs)(env_state)
            runner_state = (
                helper_optimizer_state,
                env_state,
                agent_obsv,
                helper_obsv,
                rng,
            )

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    helper_optimizer_state,
                    env_state,
                    last_agent_obs,
                    last_helper_obs,
                    rng,
                ) = runner_state
                # Reconstruct helper optimizer for this step
                helper_optimizer = nnx.merge(
                    helper_optimizer_def, helper_optimizer_state
                )

                # SELECT ACTIONS
                rng, _rng = jax.random.split(rng)

                # Get agent actions from frozen network (deterministic)
                # agent_obs_dict = {
                #     f"agent_{i}": last_agent_obs for i in range(envs.num_agents)
                # }
                agent_obs_batch = batchify(
                    last_agent_obs, envs.agents, envs.num_agents * config["NUM_ENVS"]
                )
                print(f"agent_obs_batch: {agent_obs_batch}")
                # Create agent IDs for the frozen agent network
                agent_ids_phase2 = jnp.concatenate(
                    [
                        jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32),  # Agent 0
                        jnp.ones(config["NUM_ENVS"], dtype=jnp.int32),  # Agent 1
                    ]
                )
                print(f"Phase 2 agent_ids_phase2: {agent_ids_phase2}")
                agent_pi, _ = agent_network(agent_obs_batch, agent_ids_phase2)
                # Use stochastic sampling like in phase 1, not deterministic mode
                agent_actions = agent_pi.sample(seed=_rng)
                print(f"Phase 2 agent_actions shape: {agent_actions.shape}")
                print(f"Phase 2 agent_actions[0]: {agent_actions[0]}")
                print(
                    f"Phase 2 agent_actions[{config['NUM_ENVS']}]: {agent_actions[config['NUM_ENVS']]}"
                )
                print(
                    f"Phase 2 actions identical? {agent_actions[0] == agent_actions[config['NUM_ENVS']]}"
                )
                env_act = unbatchify(
                    agent_actions, envs.agents, config["NUM_ENVS"], envs.num_agents
                )

                # Get helper action from trainable network or random action
                if helper_objective == HelperObjective.RANDOM:
                    # Use random actions instead of trained network
                    helper_action = jax.random.randint(
                        _rng, (config["NUM_ENVS"],), 0, len(envs.HelperActions)
                    )
                    helper_value = jnp.zeros((config["NUM_ENVS"],))
                    helper_log_prob = jnp.zeros((config["NUM_ENVS"],))

                if helper_objective == HelperObjective.NO_OP:
                    # Always use DO_NOTHING action
                    helper_action = jnp.full(
                        (config["NUM_ENVS"],),
                        envs.HelperActions.DO_NOTHING.value,
                        dtype=jnp.int32,
                    )
                    helper_value = jnp.zeros((config["NUM_ENVS"],))
                    helper_log_prob = jnp.zeros((config["NUM_ENVS"],))
                else:
                    helper_pi, helper_value = helper_optimizer.model(last_helper_obs)
                    helper_action = helper_pi.sample(seed=_rng)
                    helper_log_prob = helper_pi.log_prob(helper_action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                next_obs, env_state, reward, done, info = step_envs(
                    _rng, env_state, env_act, helper_action, envs, config["NUM_ENVS"]
                )

                nnx.display(envs)

                nnx.display(env_state)

                # Convert next_obs
                next_agent_obsv = jax.vmap(state_to_obs)(env_state)
                next_agent_obsv = {
                    f"agent_{i}": next_agent_obsv for i in range(envs.num_agents)
                }

                next_helper_obsv = jax.vmap(state_to_partial_obs)(env_state)

                # Calculate helper reward (could be based on agent empowerment or success)
                if helper_objective == HelperObjective.MONTE_CARLO_EMPOWERMENT:
                    # Create a vmapped version that handles batched states
                    vmapped_estimate_empowerment = jax.vmap(
                        estimate_empowerment_monte_carlo,
                        in_axes=(
                            None,
                            0,
                        ),  # vmap over initial_state (axis 0), env stays same
                        out_axes=0,
                    )
                    empowerment_estimates = vmapped_estimate_empowerment(
                        envs, env_state
                    )
                    helper_reward = empowerment_estimates[0]
                    # the helper only cares about the agent at index zero
                    agent_0_empowerment = helper_reward
                    agent_1_empowerment = empowerment_estimates[1]
                    print(f"helper_reward: {helper_reward}")
                elif helper_objective == HelperObjective.AVE_PROXY:
                    # Create a vmapped version that handles batched states
                    vmapped_estimate_ave_proxy = jax.vmap(
                        estimate_empowerment_variance_proxy,
                        in_axes=(
                            None,
                            0,
                        ),  # vmap over initial_state (axis 0), env stays same
                        out_axes=0,
                    )
                    ave_estimates = vmapped_estimate_ave_proxy(envs, env_state)
                    helper_reward = ave_estimates[
                        :, 0
                    ]  # the helper only cares about the agent at index zero
                    print(f"helper_reward AVE: {helper_reward}")
                elif helper_objective == HelperObjective.DISCRETE_CHOICE:
                    # Create a vmapped version that handles batched states
                    vmapped_estimate_discrete_choice = jax.vmap(
                        estimate_discrete_choice,
                        in_axes=(
                            None,
                            0,
                        ),  # vmap over initial_state (axis 0), env stays same
                        out_axes=0,
                    )
                    discrete_choice_estimates = vmapped_estimate_discrete_choice(
                        envs, env_state
                    )
                    helper_reward = (
                        discrete_choice_estimates  # Use discrete choice as reward
                    )
                elif helper_objective == HelperObjective.ENTROPIC_CHOICE:
                    # Create a vmapped version that handles batched states
                    vmapped_estimate_entropic_choice = jax.vmap(
                        estimate_entropic_choice,
                        in_axes=(
                            None,
                            0,
                        ),  # vmap over initial_state (axis 0), env stays same
                        out_axes=0,
                    )
                    entropic_choice_estimates = vmapped_estimate_entropic_choice(
                        envs, env_state
                    )
                    helper_reward = (
                        entropic_choice_estimates  # Use entropic choice as reward
                    )
                elif helper_objective == HelperObjective.IMMEDIATE_CHOICE:
                    raise Exception(
                        "Immediate choice is not implemented due to dependency on policy estimator"
                    )
                    # # Create a vmapped version that handles batched states
                    # vmapped_estimate_immediate_choice = jax.vmap(
                    #     estimate_immediate_choice,
                    #     in_axes=(None, 0),  # vmap over initial_state (axis 0), env stays same
                    #     out_axes=0,
                    # )
                    # immediate_choice_estimates = vmapped_estimate_immediate_choice(envs, env_state)
                    # helper_reward = immediate_choice_estimates  # Use immediate choice as reward
                elif (
                    helper_objective == HelperObjective.RANDOM
                    or helper_objective == HelperObjective.NO_OP
                ):
                    # Random helper doesn't need rewards for training, but set to zero for consistency
                    helper_reward = jnp.zeros((config["NUM_ENVS"],))
                elif helper_objective == HelperObjective.JOINT_EMPOWERMENT:
                    # Create a vmapped version that handles batched states
                    vmapped_estimate_joint_empowerment = jax.vmap(
                        estimate_empowerment_monte_carlo,
                        in_axes=(
                            None,
                            0,
                        ),  # vmap over initial_state (axis 0), env stays same
                        out_axes=0,
                    )
                    joint_empowerment_estimates = vmapped_estimate_joint_empowerment(
                        envs, env_state
                    )
                    # pdb.set_trace()
                    helper_reward = (
                        joint_empowerment_estimates[0] + joint_empowerment_estimates[1]
                    )
                    print(f"helper_reward JOINT_EMPOWERMENT: {helper_reward}")
                elif helper_objective == HelperObjective.MINIMAX_REGRET_EMPOWERMENT:
                    # Create a vmapped version that handles batched states
                    vmapped_estimate_minimax_regret = jax.vmap(
                        estimate_minimax_regret_empowerment,
                        in_axes=(
                            None,
                            0,
                        ),  # vmap over initial_state (axis 0), env stays same
                        out_axes=0,
                    )
                    minimax_regret_estimates = vmapped_estimate_minimax_regret(
                        envs, env_state
                    )
                    helper_reward = (
                        minimax_regret_estimates  # Use minimax regret as reward
                    )
                    print(f"helper_reward MINIMAX_REGRET_EMPOWERMENT: {helper_reward}")

                if helper_objective != HelperObjective.MONTE_CARLO_EMPOWERMENT:
                    # Create a vmapped version that handles batched states
                    vmapped_estimate_empowerment = jax.vmap(
                        estimate_empowerment_monte_carlo,
                        in_axes=(
                            None,
                            0,
                        ),  # vmap over initial_state (axis 0), env stays same
                        out_axes=0,
                    )
                    empowerment_estimates = vmapped_estimate_empowerment(
                        envs, env_state
                    )
                    agent_0_empowerment = empowerment_estimates[0]
                    agent_1_empowerment = empowerment_estimates[1]

                # Pass the agent rewards through Transition in info as well
                print(f"phase 2 reward: {reward}")
                info["agent_rewards"] = reward
                info["agent_actions"] = env_act  # Add agent actions to info
                info["agent_0_empowerment"] = (
                    agent_0_empowerment  # Add agent 0 empowerment for logging
                )
                info["agent_1_empowerment"] = (
                    agent_1_empowerment  # Add agent 1 empowerment for logging
                )

                transition = Transition(
                    batchify(done, envs.agents, config["NUM_ACTORS"]).squeeze(),
                    helper_action,
                    helper_value,
                    helper_reward,
                    helper_log_prob,
                    last_helper_obs,
                    info if isinstance(info, dict) else {},
                )
                # Split helper optimizer state back for scan
                _, updated_helper_optimizer_state = nnx.split(helper_optimizer)
                runner_state = (
                    updated_helper_optimizer_state,
                    env_state,
                    next_agent_obsv,
                    next_helper_obsv,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            helper_optimizer_state, env_state, last_agent_obs, last_helper_obs, rng = (
                runner_state
            )
            # Reconstruct helper optimizer for advantage calculation
            helper_optimizer = nnx.merge(helper_optimizer_def, helper_optimizer_state)
            _, last_val = helper_optimizer.model(last_helper_obs)

            # Calculate episode metrics
            print(f"traj_batch: {traj_batch}")
            episode_returns = traj_batch.reward.sum(axis=0)  # Sum over timesteps
            episode_critic_predictions = traj_batch.value.sum(
                axis=0
            )  # Sum over timesteps
            episode_lengths = config["NUM_STEPS"] * jnp.ones_like(episode_returns)

            agent_episode_returns = traj_batch.info["agent_rewards"].sum(axis=0)
            agent_0_episode_returns = agent_episode_returns[:, 0]
            agent_1_episode_returns = agent_episode_returns[:, 1]

            # Calculate summed empowerment for agent 1 over each episode
            agent_0_episode_empowerment = traj_batch.info["agent_0_empowerment"].sum(
                axis=0
            )
            agent_1_episode_empowerment = traj_batch.info["agent_1_empowerment"].sum(
                axis=0
            )

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    # we only care about the done for the agent at index zero naively
                    done = done[:, 0]
                    print(f"reward: {reward}")
                    print(f"next_value: {next_value}")
                    print(f"done: {done}")
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE HELPER NETWORK
            def _update_epoch(update_state, unused):
                @nnx.scan
                def _update_minibatch(helper_optimizer, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(model, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = model(traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        helper_optimizer.model, traj_batch, advantages, targets
                    )
                    helper_optimizer.update(grads)
                    return helper_optimizer, total_loss

                helper_optimizer_state, traj_batch, advantages, targets, rng = (
                    update_state
                )
                # Reconstruct helper optimizer for updates
                helper_optimizer = nnx.merge(
                    helper_optimizer_def, helper_optimizer_state
                )
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                helper_optimizer, total_loss = _update_minibatch(
                    helper_optimizer, minibatches
                )
                # Split helper optimizer back to state for scan
                _, updated_helper_optimizer_state = nnx.split(helper_optimizer)
                update_state = (
                    updated_helper_optimizer_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                helper_optimizer_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            helper_optimizer_state = update_state[0]
            # Reconstruct helper optimizer for final step
            helper_optimizer = nnx.merge(helper_optimizer_def, helper_optimizer_state)
            metric = {
                "episode_returns": episode_returns,
                "episode_lengths": episode_lengths,
            }
            if hasattr(traj_batch, "info") and traj_batch.info:
                metric.update(traj_batch.info)
            rng = update_state[-1]

            # Collect metrics for logging outside scan
            # Extract loss components from loss_info (shape: [UPDATE_EPOCHS, NUM_MINIBATCHES])
            total_losses, (value_losses, actor_losses, entropies) = loss_info

            # Collect all data needed for logging
            logging_metrics = {
                "episode_returns": episode_returns,
                "episode_critic_predictions": episode_critic_predictions,
                "episode_lengths": episode_lengths,
                "total_losses": total_losses,
                "value_losses": value_losses,
                "actor_losses": actor_losses,
                "entropies": entropies,
                "advantages": advantages,
                "helper_rewards": traj_batch.reward,
                "agent_0_episode_returns": agent_0_episode_returns,
                "agent_1_episode_returns": agent_1_episode_returns,
                "agent_0_episode_empowerment": agent_0_episode_empowerment,
                "agent_1_episode_empowerment": agent_1_episode_empowerment,
                "helper_actions": traj_batch.action,  # Helper actions from trajectory
                "agent_actions": traj_batch.info.get(
                    "agent_actions",
                    jnp.zeros(
                        (config["NUM_STEPS"], config["NUM_ENVS"], envs.num_agents)
                    ),
                ),  # Extract from info
            }

            runner_state = (
                helper_optimizer_state,
                env_state,
                last_agent_obs,
                last_helper_obs,
                rng,
            )  # should include last_agent_obs even though it's not used to keep shapes consistent
            return runner_state, (metric, logging_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (
            helper_optimizer_state,
            env_state,
            agent_obsv,
            helper_obsv,
            _rng,
        )
        runner_state, outputs = jax.lax.scan(
            _update_step,
            runner_state,
            jnp.arange(config["NUM_UPDATES"]),
            config["NUM_UPDATES"],
        )

        # Unpack outputs from scan
        metric, logging_metrics_all = outputs

        # Extract helper name from wandb_id for logging
        helper_name = str(helper_objective).lower()

        # Create CSV file for logging phase 2 data
        csv_dir = f"train_data/{datetime.now().strftime('%Y%m%d')}/{wandb_id}"
        os.makedirs(csv_dir, exist_ok=True)
        csv_filename = f"{csv_dir}/{wandb_id}_phase_2_data.csv"

        # Initialize CSV file with headers
        csv_headers = [
            "wandbid",
            "helper_objective",
            "epoch",
            "helper_mean_episode_returns",
            "helper_mean_episode_critic_predictions",
            "agent_0_mean_episode_returns",
            "agent_1_mean_episode_returns",
            "agent_0_mean_episode_empowerment",
            "agent_1_mean_episode_empowerment",
        ]
        csv_file_exists = os.path.exists(csv_filename)

        # Log Phase 2 metrics outside scan
        for update_idx in range(config["NUM_UPDATES"]):
            # Extract metrics for this epoch
            epoch_logging_metrics = jax.tree_util.tree_map(
                lambda x: x[update_idx], logging_metrics_all
            )

            # Calculate global step (continue from where Phase 1 left off)
            global_step = (
                num_epochs_phase1 + num_epochs_phase2 * helper_idx + update_idx
            )

            # Create combined metrics dictionary for this epoch with helper-specific prefix
            epoch_metrics = {
                f"phase2_{helper_name}/episode_returns_mean": float(
                    jnp.mean(epoch_logging_metrics["episode_returns"])
                ),
                f"phase2_{helper_name}/episode_returns_std": float(
                    jnp.std(epoch_logging_metrics["episode_returns"])
                ),
                f"phase2_{helper_name}/episode_lengths_mean": float(
                    jnp.mean(epoch_logging_metrics["episode_lengths"])
                ),
                f"phase2_{helper_name}/total_loss": float(
                    jnp.mean(epoch_logging_metrics["total_losses"])
                ),
                f"phase2_{helper_name}/value_loss": float(
                    jnp.mean(epoch_logging_metrics["value_losses"])
                ),
                f"phase2_{helper_name}/actor_loss": float(
                    jnp.mean(epoch_logging_metrics["actor_losses"])
                ),
                f"phase2_{helper_name}/entropy": float(
                    jnp.mean(epoch_logging_metrics["entropies"])
                ),
                f"phase2_{helper_name}/advantages_mean": float(
                    jnp.mean(epoch_logging_metrics["advantages"])
                ),
                f"phase2_{helper_name}/advantages_std": float(
                    jnp.std(epoch_logging_metrics["advantages"])
                ),
                f"phase2_{helper_name}/helper_rewards_mean": float(
                    jnp.mean(epoch_logging_metrics["helper_rewards"])
                ),
                f"phase2_{helper_name}/helper_rewards_std": float(
                    jnp.std(epoch_logging_metrics["helper_rewards"])
                ),
                f"phase2_{helper_name}/agent_0_episode_returns_mean": float(
                    jnp.mean(epoch_logging_metrics["agent_0_episode_returns"])
                ),
                f"phase2_{helper_name}/agent_0_episode_returns_std": float(
                    jnp.std(epoch_logging_metrics["agent_0_episode_returns"])
                ),
                f"phase2_{helper_name}/agent_1_episode_returns_mean": float(
                    jnp.mean(epoch_logging_metrics["agent_1_episode_returns"])
                ),
                f"phase2_{helper_name}/agent_1_episode_returns_std": float(
                    jnp.std(epoch_logging_metrics["agent_1_episode_returns"])
                ),
                f"phase2_{helper_name}/agent_1_episode_empowerment_mean": float(
                    jnp.mean(epoch_logging_metrics["agent_1_episode_empowerment"])
                ),
                f"phase2_{helper_name}/agent_1_episode_empowerment_std": float(
                    jnp.std(epoch_logging_metrics["agent_1_episode_empowerment"])
                ),
                f"phase2_{helper_name}/episode_critic_predictions_mean": float(
                    jnp.mean(epoch_logging_metrics["episode_critic_predictions"])
                ),
                f"phase2_{helper_name}/episode_critic_predictions_std": float(
                    jnp.std(epoch_logging_metrics["episode_critic_predictions"])
                ),
                f"phase2_{helper_name}/agent_0_episode_empowerment_mean": float(
                    jnp.mean(epoch_logging_metrics["agent_0_episode_empowerment"])
                ),
                f"phase2_{helper_name}/agent_0_episode_empowerment_std": float(
                    jnp.std(epoch_logging_metrics["agent_0_episode_empowerment"])
                ),
                f"phase2_{helper_name}/epoch": update_idx,
                "training/global_epoch": global_step,
            }

            # Add action logging for Phase 2 with helper-specific prefix
            # Log helper actions from trajectory batch
            if "helper_actions" in epoch_logging_metrics:
                helper_actions = epoch_logging_metrics["helper_actions"]
                epoch_metrics[f"phase2_{helper_name}/actions/helper_mean"] = float(
                    jnp.mean(helper_actions)
                )

            # Log agent actions (these are deterministic from frozen policy)
            if "agent_actions" in epoch_logging_metrics:
                agent_actions = epoch_logging_metrics["agent_actions"]
                for i, agent_id in enumerate(envs.agents):
                    if agent_id in agent_actions:
                        epoch_metrics[
                            f"phase2_{helper_name}/actions/{agent_id}_mean"
                        ] = float(jnp.mean(agent_actions[agent_id]))

            # Log all metrics for this epoch using global step
            wandb.log(epoch_metrics, step=global_step)

            # Write CSV data for this epoch
            csv_data = {
                "wandbid": wandb_id,
                "helper_objective": helper_name,
                "epoch": update_idx,
                "helper_mean_episode_returns": float(
                    jnp.mean(epoch_logging_metrics["episode_returns"])
                ),
                "helper_mean_episode_critic_predictions": float(
                    jnp.mean(epoch_logging_metrics["episode_critic_predictions"])
                ),
                "agent_0_mean_episode_returns": float(
                    jnp.mean(epoch_logging_metrics["agent_0_episode_returns"])
                ),
                "agent_1_mean_episode_returns": float(
                    jnp.mean(epoch_logging_metrics["agent_1_episode_returns"])
                ),
                "agent_0_mean_episode_empowerment": float(
                    jnp.mean(epoch_logging_metrics["agent_0_episode_empowerment"])
                ),
                "agent_1_mean_episode_empowerment": float(
                    jnp.mean(epoch_logging_metrics["agent_1_episode_empowerment"])
                ),
            }

            # Write to CSV file
            with open(csv_filename, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
                # Write header only if file is new or empty
                if not csv_file_exists:
                    writer.writeheader()
                    csv_file_exists = True  # Set flag to prevent writing header again
                writer.writerow(csv_data)

            # visualize one of the trajectories here
            if update_idx == 1 or update_idx == 249:
                # Recreate the trajectory by replaying actions

                # Reset environment to get initial state (same as training)
                rng_replay, _ = jax.random.split(rng)
                if specific_position_reset:
                    initial_state = reset_envs_specific_positions(
                        rng_replay, envs, specific_position_reset, 1
                    )

                # Extract actions for first trajectory
                helper_actions_traj = epoch_logging_metrics.get(
                    "helper_actions", jnp.zeros((config["NUM_STEPS"],))
                )[:, 0]
                agent_actions = epoch_logging_metrics.get("agent_actions", None)

                if agent_actions is not None:
                    agent_0_actions_traj = agent_actions["agent_0"][
                        :, 0
                    ]  # Agent 0, first trajectory
                    agent_1_actions_traj = agent_actions["agent_1"][
                        :, 0
                    ]  # Agent 1, first trajectory

                    agent_actions_dict = {
                        "agent_0": [agent_0_actions_traj],
                        "agent_1": [agent_1_actions_traj],
                    }

                    # Replay the episode step by step and collect rewards
                    def replay_step(state, actions_at_step):
                        agent_0_action, agent_1_action, helper_action = actions_at_step

                        # Reconstruct agent actions in the format expected by step_envs
                        env_act = {
                            "agent_0": jnp.array(
                                [agent_0_action]
                            ),  # Wrap in array for single env
                            "agent_1": jnp.array([agent_1_action]),
                        }

                        # Step the environment
                        rng_step, _ = jax.random.split(rng_replay)
                        _, next_state, reward, _, _ = step_envs(
                            rng_step,
                            state,
                            env_act,
                            jnp.array([helper_action]),
                            envs,
                            1,
                        )

                        return next_state, (
                            state,
                            reward,
                        )  # Return state and reward for collection

                    # Prepare action sequences
                    action_sequence = jnp.stack(
                        [
                            agent_0_actions_traj,
                            agent_1_actions_traj,
                            helper_actions_traj,
                        ],
                        axis=1,
                    )  # Shape: (NUM_STEPS, 3)

                    # Replay all steps
                    final_state, (states_trajectory, rewards_trajectory) = jax.lax.scan(
                        replay_step, initial_state, action_sequence
                    )

                    # Extract rewards for visualization (squeeze out env dimension)
                    episode_rewards = jnp.squeeze(
                        rewards_trajectory, axis=1
                    )  # Shape: (NUM_STEPS, num_agents)

                    # Create frames from the replayed states
                    states_trajectory_squeezed = jax.tree_util.tree_map(
                        lambda x: jnp.squeeze(x, axis=1), states_trajectory
                    )
                    frames = jax.vmap(lambda s: create_frame_data(s, envs), in_axes=0)(
                        states_trajectory_squeezed
                    )

                    fig = create_trajectory_viewer_px(
                        todays_date + "_" + wandb_id,
                        frames,
                        helper_actions_traj,
                        agent_actions_dict,
                        envs.no_freeze,
                        envs.no_pull_helper_action,
                        agent_rewards=episode_rewards,  # Pass the actual rewards
                    )
                    fig.write_html(
                        f"train_data/{todays_date}/{wandb_id}/phase_2_{wandb_id}_{helper_name}_epoch_{update_idx}_testing_episode.html",
                        auto_play=False,
                    )

        # Reconstruct final helper optimizer for return
        final_helper_optimizer_state = runner_state[0]
        final_helper_optimizer = nnx.merge(
            helper_optimizer_def, final_helper_optimizer_state
        )
        final_runner_state = (
            final_helper_optimizer,
            runner_state[1],
            runner_state[2],
            runner_state[3],
        )
        return {"runner_state": final_runner_state, "metrics": metric}

    return train_phase2


def train(
    key: chex.PRNGKey,
    envs: MultiAgentGridWorld,
    num_epochs_phase1: int,
    num_epochs_phase2: int,
    helper_objectives: list[HelperObjective],
    specific_positions_file: str = None,
    save_checkpoints: bool = False,  # default to false for now
) -> Tuple[Any, dict, dict]:
    """
    Multi-helper training:
    Phase 1: Train agent policies with random helper (once)
    Phase 2: Train multiple helper policies with frozen agent policies (multiple runs)
    """

    print(
        f"Training with PPO: Phase 1 ({num_epochs_phase1} epochs), Phase 2 ({num_epochs_phase2} epochs x {len(helper_objectives)} helpers)"
    )

    wandb.define_metric("evaluation/success", summary="mean")
    wandb.define_metric("evaluation/*", step_metric="epoch")
    wandb.define_metric("phase1/*", step_metric="epoch")

    # Define metrics for each helper objective
    for helper_objective in helper_objectives:
        helper_name = str(helper_objective).lower()
        wandb.define_metric(f"phase2_{helper_name}/*")

    todays_date = datetime.now().strftime("%Y%m%d")

    specific_positions = (
        load_specific_positions_from_file(specific_positions_file)
        if specific_positions_file
        else None
    )

    # Phase 1: Train agent policies with random helper (ONLY ONCE)
    print("Starting Phase 1: Training agent policies with random helper...")

    # Create a random helper (random policy)
    helper_action_dim = len(envs.HelperActions)
    key, helper_key = jax.random.split(key)
    random_helper = RandomPolicy(helper_key, a_dim=helper_action_dim)

    # Train agent policies
    train_phase1 = make_ppo_train_phase1(envs, random_helper, wandbid)
    key, phase1_key = jax.random.split(key)

    # Update config for phase 1
    config = create_ppo_config()
    config["NUM_UPDATES"] = num_epochs_phase1

    phase1_result = train_phase1(phase1_key, specific_positions)
    phase1_optimizer = phase1_result["runner_state"][0]  # Extract optimizer
    # Get model state for phase 2
    graph_def, trained_agent_state = nnx.split(phase1_optimizer.model)
    phase1_metrics = phase1_result["metrics"]

    # Log Phase 1 completion metrics
    wandb.log(
        {
            "training/phase1_complete": True,
            "training/phase1_final_returns_mean": jnp.mean(
                phase1_metrics["episode_returns"][-1]
            ),
            "training/phase1_final_returns_std": jnp.std(
                phase1_metrics["episode_returns"][-1]
            ),
        }
    )

    # Save agent policies (once)
    if save_checkpoints:
        save_path = f"train_data/{todays_date}/{wandbid}/agents/checkpoints"
        save_path = os.path.abspath(save_path)
        os.makedirs(save_path, exist_ok=True)
        path = ocp.test_utils.erase_and_create_empty(save_path)

        checkpointer = ocp.StandardCheckpointer()
        filename = f"{wandbid}_trained_agent_state"
        checkpointer.save(path / filename, trained_agent_state)

        # Save constructor arguments
        agent_config = {
            "action_dim": len(Actions),
            "grid_width": envs.width,
            "grid_height": envs.height,
            "num_agents": envs.num_agents,
            "activation": config["ACTIVATION"],
        }
        with open(f"{save_path}/{wandbid}_trained_agent_config.pkl", "wb") as f:
            pickle.dump(agent_config, f)

        # # Save metrics for reference
        # with open(f"{save_dir}/{wandbid}_agent_metrics.pkl", "wb") as f:
        #     pickle.dump(phase1_metrics, f)

        print(f"Agent policy saved to {save_path}")
    print(f"Phase 1 complete. Trained agent policies for {num_epochs_phase1} epochs.")

    # Phase 2: Train multiple helper policies with frozen agent policies
    print(
        "Starting Phase 2: Training multiple helper policies with frozen agent policies..."
    )

    trained_helper_states = {}
    all_phase2_metrics = {}
    helper_idx = 0

    for helper_objective in helper_objectives:
        print(f"\nTraining helper with objective: {helper_objective}")

        # Create unique identifier for this helper
        helper_name = str(helper_objective).lower()

        # Create Phase 2 training function for this specific objective
        train_phase2 = make_ppo_train_phase2(
            envs,
            graph_def,
            trained_agent_state,
            helper_objective,
            num_epochs_phase1,
            num_epochs_phase2,
            helper_idx,
            wandbid,
        )
        key, phase2_key = jax.random.split(key)

        # Update config for phase 2
        config["NUM_UPDATES"] = num_epochs_phase2

        phase2_result = train_phase2(phase2_key, specific_positions)
        phase2_helper_optimizer = phase2_result["runner_state"][
            0
        ]  # Extract helper optimizer
        # Get helper model state
        helper_graph_def, trained_helper_state = nnx.split(
            phase2_helper_optimizer.model
        )
        phase2_metrics = phase2_result["metrics"]

        # Store results for this helper
        trained_helper_states[helper_name] = trained_helper_state
        all_phase2_metrics[helper_name] = phase2_metrics

        # Log Phase 2 completion metrics for this helper
        wandb.log(
            {
                f"training/phase2_{helper_name}_complete": True,
                f"training/phase2_{helper_name}_final_returns_mean": jnp.mean(
                    phase2_metrics["episode_returns"][-1]
                ),
                f"training/phase2_{helper_name}_final_returns_std": jnp.std(
                    phase2_metrics["episode_returns"][-1]
                ),
                f"training/helper_{helper_name}_final_performance": jnp.mean(
                    phase2_metrics["episode_returns"][-1]
                ),
            }
        )

        # Create directory for saving this specific helper
        if save_checkpoints:
            save_dir = f"train_data/{todays_date}/{wandbid}/{helper_name}/checkpoints"
            save_dir = os.path.abspath(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_path_helper = ocp.test_utils.erase_and_create_empty(save_dir)

            filename = f"{wandbid}_{helper_name}_trained_helper_state"
            checkpointer.save(
                directory=save_path_helper / filename, state=trained_helper_state
            )
            checkpointer.wait_until_finished()

            # Save constructor arguments for this helper
            helper_config = {
                "action_dim": len(Actions),
                "grid_width": envs.width,
                "grid_height": envs.height,
                "activation": config["ACTIVATION"],
                "helper_objective": str(helper_objective),
            }
            with open(
                f"{save_dir}/{wandbid}_{helper_name}_trained_helper_config.pkl", "wb"
            ) as f:
                pickle.dump(helper_config, f)

            # # Save metrics for reference
            # with open(f"{save_dir}/{wandbid}_helper_metrics.pkl", "wb") as f:
            #     pickle.dump(phase2_metrics, f)

            # # Save optimizer state if you need to resume training later
            # with open(f"{save_dir}/{wandbid}_helper_optimizer.pkl", "wb") as f:
            #     pickle.dump(phase2_helper_optimizer, f)

            print(f"Agent policy {helper_name} saved to {save_dir}")
        helper_idx += 1

    print(
        f"Phase 2 complete. Trained {helper_objectives} helper policies for {num_epochs_phase2} epochs each."
    )

    # Log final metrics
    wandb.log({"training/phase1_epochs": num_epochs_phase1})
    wandb.log({"training/phase2_epochs": num_epochs_phase2})
    wandb.log(
        {
            "training/total_epochs": num_epochs_phase1
            + len(helper_objectives) * num_epochs_phase2
        }
    )
    wandb.log({"training/num_helpers_trained": len(helper_objectives)})

    # Combine metrics for plotting
    training_metrics = {
        "phase1": phase1_metrics,
        "phase2": all_phase2_metrics,
    }

    return (graph_def, trained_agent_state), trained_helper_states, training_metrics


if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    if args.env_mode == "independent":
        env_mode = EnvMode.INDEPENDENT
    elif args.env_mode == "cooperative":
        env_mode = EnvMode.COOPERATIVE
    elif args.env_mode == "competitive":
        env_mode = EnvMode.COMPETITIVE
    else:
        raise Exception(f"Env mode {args.env_mode} is invalid")

    # Parse helper objectives (either single or multiple)
    helper_objectives = []
    for obj_str in args.helper_objective:
        if obj_str == "monte_carlo_emp":
            helper_objectives.append(HelperObjective.MONTE_CARLO_EMPOWERMENT)
        elif obj_str == "ave":
            helper_objectives.append(HelperObjective.AVE_PROXY)
        # elif obj_str == "immediate_choice":
        #     helper_objectives.append(HelperObjective.IMMEDIATE_CHOICE)
        elif obj_str == "discrete_choice":
            helper_objectives.append(HelperObjective.DISCRETE_CHOICE)
        elif obj_str == "entropic_choice":
            helper_objectives.append(HelperObjective.ENTROPIC_CHOICE)
        elif obj_str == "random":
            helper_objectives.append(HelperObjective.RANDOM)
        elif obj_str == "no_op":
            helper_objectives.append(HelperObjective.NO_OP)
        elif obj_str == "joint_empowerment":
            helper_objectives.append(HelperObjective.JOINT_EMPOWERMENT)
        elif obj_str == "minimax_regret_empowerment":
            helper_objectives.append(HelperObjective.MINIMAX_REGRET_EMPOWERMENT)
        else:
            raise Exception(f"Helper objective {obj_str} is invalid")

    boxes_can_cover_goal = args.boxes_can_cover_goal
    num_traps = args.num_traps
    num_boxes = args.num_boxes
    width = args.grid_width
    height = args.grid_height
    num_goals = args.num_goals
    num_agents = args.num_agents
    num_walls = args.num_walls
    num_keys = args.num_keys
    if num_agents != 1 and num_agents != 2:
        raise Exception(f"Number of agents {num_agents} is invalid - must be 1 or 2")

    specific_positions_file = args.specific_positions_file
    no_freeze = args.no_freeze
    no_pull_helper_action = args.no_pull

    # initialize the environments
    env = MultiAgentGridWorld(
        height=height,
        width=width,
        max_steps=args.max_steps,
        num_agents=num_agents,
        env_mode=env_mode,
        boxes_can_cover_goal=boxes_can_cover_goal,
        random_reset_helper_pos=False,
        num_traps=num_traps,
        num_boxes=num_boxes,
        num_goals=num_goals,
        num_walls=num_walls,
        no_freeze=no_freeze,
        no_pull_helper_action=no_pull_helper_action,
        num_keys=num_keys,
    )

    # initialize policy
    key = jax.random.PRNGKey(args.seed)
    key, rng = jax.random.split(key)

    # wandb initialization
    config = vars(args)
    config["env_name"] = "multiagent_gridworld"
    config["env_mode"] = env_mode
    config["boxes_can_cover_goal"] = boxes_can_cover_goal
    config["helper_objectives"] = [str(obj) for obj in helper_objectives]
    wandbid = wandb.util.generate_id(4)
    print(f"!!! Wandb ID: {wandbid}")
    todays_date = datetime.now().strftime("%Y%m%d")
    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    wandb.init(
        project="multiagent_empowerment",
        config=config,
        id=f"multiagent_gridworld_{wandbid}_{timestamp}",
    )

    os.makedirs(f"train_data/{todays_date}/{wandbid}", exist_ok=True)
    # save the args to a text file in case I want to inspect
    with open(f"train_data/{todays_date}/{wandbid}/{wandbid}_args.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # save the environment json to a json file
    with open(f"train_data/{todays_date}/{wandbid}/{wandbid}_env.json", "w") as f:
        data = load_specific_positions_from_file(specific_positions_file)
        json.dump(data, f, indent=2)

    trained_agent_data, trained_helper_states, training_metrics = train(
        key, env, args.epochs, args.epochs, helper_objectives, specific_positions_file
    )
