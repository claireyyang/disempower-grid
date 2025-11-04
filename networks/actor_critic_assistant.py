import distrax
import numpy as np
import jax.numpy as jnp
import jax
from flax import nnx


class ActorCriticAssistant(nnx.Module):
    """PPO Actor-Critic network for assistant policy with grid observations."""

    def __init__(self, action_dim: int, grid_width: int, grid_height: int, activation: str = "tanh", num_agents: int = 2, *, rngs: nnx.Rngs, has_keys: bool = True):
        self.action_dim = action_dim
        self.activation = activation
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_agents = num_agents

        # Calculate number of observation channels for assistant (no goal channels)
        # 5 base channels (boxes, traps, walls, helper, freeze_timer) + 3 * num_agents (agent_pos, key_pos, has_key)
        # if theres keys
        if has_keys:
            observation_channels = 5 + num_agents * 3

        # 4 base channels (boxes, traps, walls, freeze_timer) + 1 * num_agents (agent_pos only)
        # if no keys
        else:
            observation_channels = 4 + num_agents

        # Calculate flattened input size for dense layers
        self.input_size = grid_width * grid_height * observation_channels

        # Actor layers
        self.actor_dense1 = nnx.Linear(
            in_features=self.input_size,
            out_features=64,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(np.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
        )
        self.actor_dense2 = nnx.Linear(
            in_features=64,
            out_features=64,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(np.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
        )
        self.actor_out = nnx.Linear(
            in_features=64,
            out_features=action_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(0.01),
            bias_init=nnx.initializers.constant(0.0),
        )

        # Critic layers
        self.critic_dense1 = nnx.Linear(
            in_features=self.input_size,
            out_features=64,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(np.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
        )
        self.critic_dense2 = nnx.Linear(
            in_features=64,
            out_features=64,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(np.sqrt(2)),
            bias_init=nnx.initializers.constant(0.0),
        )
        self.critic_out = nnx.Linear(
            in_features=64,
            out_features=1,
            rngs=rngs,
            kernel_init=nnx.initializers.orthogonal(1.0),
            bias_init=nnx.initializers.constant(0.0),
        )

    def __call__(self, x):
        activation_fn = jax.nn.relu if self.activation == "relu" else jax.nn.tanh

        if len(x.shape) == 3:  # (height, width, channels) - single observation
            x = jnp.expand_dims(x, 0)  # (1, height, width, channels)

        batch_size = x.shape[0]

        # Flatten spatial features: (batch, height, width, channels) -> (batch, height*width*channels)
        x = x.reshape((batch_size, -1))

        # Actor network
        actor_mean = self.actor_dense1(x)
        actor_mean = activation_fn(actor_mean)
        actor_mean = self.actor_dense2(actor_mean)
        actor_mean = activation_fn(actor_mean)
        actor_mean = self.actor_out(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic network
        critic = self.critic_dense1(x)
        critic = activation_fn(critic)
        critic = self.critic_dense2(critic)
        critic = activation_fn(critic)
        critic = self.critic_out(critic)

        return pi, jnp.squeeze(critic, axis=-1)
