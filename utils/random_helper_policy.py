import jax
from simple_pytree import Pytree, static_field
import jax.numpy as jnp


class RandomPolicy(Pytree, mutable=True):
    a_dim = static_field()  # Note: is the action space of the helper agent

    def __init__(self, key, a_dim):
        self.key = key
        self.a_dim = a_dim  # num of actions the helper can take
        self.num_steps = 0

    def next_action(self, state):
        # Get batch size from the first dimension of any state field
        batch_size = state.shape[0]

        # Split key for each batch item
        self.key, key = jax.random.split(self.key)
        split_keys = jax.random.split(key, batch_size)

        # Generate random actions for each state in the batch
        actions = jax.vmap(lambda k: jax.random.randint(k, (1,), 0, self.a_dim))(
            split_keys
        )

        # Squeeze out the unnecessary dimension from the randint shape
        actions = jnp.squeeze(actions)

        return actions, {"policy/action": actions}

    def observe(self, vectorized_states, helper_action, agent_action):
        self.num_steps += 1
        return self
