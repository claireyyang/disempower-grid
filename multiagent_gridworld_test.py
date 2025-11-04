import chex
import jax
from absl.testing import absltest
import jax.numpy as jnp
from environments.embodied_assistant.multiagent_gridworld import (
    MultiAgentGridWorld,
    Actions,
    EnvMode,
)
from environments.embodied_assistant.empowerment_estimator import (
    estimate_empowerment_variance_proxy,
    estimate_empowerment_monte_carlo,
)


class MultiAgentGridWorldTest(chex.TestCase):

    def test_should_pass(self):
        for i in (1, 2):
            with self.subTest(i=i):
                self.assertEqual(i, i)
                print("woohoo it passes")

    def test_agent_collides_with_box(self):
        """
        Tests that agent should not be able to move downwards, since it collides with the box (which is moved after the agent)
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.DOWN}
        next_helper_action = env.HelperActions.DO_NOTHING

        expected_agent_0_pos = [0, 1]

        self.assertEqual(
            env.step_env(new_key, next_state, next_action, next_helper_action)[1]
            .agent_pos[0]
            .tolist(),
            expected_agent_0_pos,
        )

    def test_agent_collides_with_agent(self):
        """
        Tests that the agent collides with another agent, and neither move as a result
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_0_position = [0, 1]
        expected_agent_1_position = [0, 3]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_position)
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_position)

    def test_agents_cannot_switch_positions(self):
        """
        Tests that the agents cannot switch positions
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.agent_pos = jnp.array([[0, 2], [0, 3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_0_pos = [0, 2]
        expected_agent_1_pos = [0, 3]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_pos)
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

    def test_agent_collides_with_wall(self):
        """
        Tests that the agent collides with a wall, and does not move
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.agent_pos = jnp.array([[2, 1], [0, 3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_0_position = [2, 1]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_position)

    def test_agent_cannot_move_out_of_bounds(self):
        """
        Tests that the agent cannot move out of bounds
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.UP, "agent_1": Actions.UP}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_0_position = [0, 1]
        expected_agent_1_position = [0, 3]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_position)
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_position)

    def test_agent_collides_with_box_and_then_other_agent_tries_to_move_into_same_position(
        self,
    ):
        """
        Test that the agent will move as expected if one agent collides with a box and then the other is trying to move into the same position
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.RIGHT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.RIGHT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_0_position = [0, 2]
        expected_agent_1_position = [0, 3]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_position)
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_position)

    def test_agent_tries_to_move_into_same_position_in_same_goal_in_independent_mode(
        self,
    ):
        """
        Test that the agent will move as expected if one agent collides with a box and then the other is trying to move into the same position
        But they're actually in the same goal, so it's okay
        """
        env = MultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.box_pos = next_state.box_pos.at[3].set(jnp.array([2, 4]))

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.DOWN}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.RIGHT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        # agent 0 enters the goal while agent 1 is in the goal and collides with the box at [2,4]
        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.DOWN}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_0_position = [1, 4]
        expected_agent_1_position = [1, 4]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_position)
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_position)

    def test_helper_can_push_box(self):
        """
        Tests that the helper can push the box to an empty space successfully
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.helper_pos = jnp.array([[1, 0]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.PUSH_BOX_RIGHT

        expected_box_1_pos = [1, 2]
        expected_helper_pos = [1, 1]

        next_state = env.step_env(new_key, next_state, next_action, next_helper_action)[
            1
        ]

        self.assertEqual(
            next_state.box_pos[1].tolist(),
            expected_box_1_pos,
        )

        self.assertEqual(
            next_state.helper_pos[0].tolist(),
            expected_helper_pos,
        )

    def test_helper_can_pull_box(self):
        """
        Tests that the helper can pull the box to an empty space successfully
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.helper_pos = jnp.array([[2, 3]])
        print("box 3 pos: ", next_state.box_pos[3])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.PULL_BOX_DOWN_FROM_UP

        expected_box_3_pos = [2, 3]
        expected_helper_pos = [3, 3]

        next_state = env.step_env(new_key, next_state, next_action, next_helper_action)[
            1
        ]

        self.assertEqual(
            next_state.helper_pos[0].tolist(),
            expected_helper_pos,
        )

        self.assertEqual(next_state.box_pos[3].tolist(), expected_box_3_pos)

    def test_helper_cannot_pull_box_if_helper_collides_with_another_box(self):
        """
        Tests that the helper cannot pull a box if the helper would collide with another box after the pull
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        # Set up: helper at [1, 3], box to pull at [0, 3], another box at [2, 3]
        next_state.helper_pos = jnp.array([[1, 3]])
        next_state.box_pos = next_state.box_pos.at[2].set(
            jnp.array([0, 3])
        )  # box to pull
        next_state.box_pos = next_state.box_pos.at[3].set(
            jnp.array([2, 3])
        )  # blocking box

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.PULL_BOX_DOWN_FROM_UP

        next_state = env.step_env(new_key, next_state, next_action, next_helper_action)[
            1
        ]

        # Box should not move, helper should not move
        expected_box_2_pos = [0, 3]
        expected_helper_pos = [[1, 3]]
        self.assertEqual(next_state.box_pos[2].tolist(), expected_box_2_pos)
        self.assertEqual(next_state.helper_pos.tolist(), expected_helper_pos)

    def test_helper_cannot_pull_box_if_helper_collides_with_wall(self):
        """
        Tests that the helper cannot pull a box if the helper would collide with a wall after the pull
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        # Set up: helper at [2, 1], box to pull at [1, 1], wall at [3, 1]
        next_state.helper_pos = jnp.array([[2, 1]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.PULL_BOX_DOWN_FROM_UP

        next_state = env.step_env(new_key, next_state, next_action, next_helper_action)[
            1
        ]

        # Box should not move, helper should not move (would collide with wall at [3,1])
        expected_box_1_pos = [1, 1]
        expected_helper_pos = [[2, 1]]
        self.assertEqual(next_state.box_pos[1].tolist(), expected_box_1_pos)
        self.assertEqual(next_state.helper_pos.tolist(), expected_helper_pos)

    def test_helper_cannot_pull_box_if_helper_collides_with_agent(self):
        """
        Tests that the helper cannot pull a box if the helper would collide with an agent after the pull
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        # Set up: helper at [1, 2], box to pull at [0, 2], agent at [2, 2]
        next_state.helper_pos = jnp.array([[1, 2]])
        next_state.box_pos = next_state.box_pos.at[0].set(
            jnp.array([0, 2])
        )  # box to pull
        next_state.agent_pos = next_state.agent_pos.at[0].set(
            jnp.array([2, 2])
        )  # blocking agent

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.PULL_BOX_DOWN_FROM_UP

        next_state = env.step_env(new_key, next_state, next_action, next_helper_action)[
            1
        ]

        # Box should not move, helper should not move
        expected_box_0_pos = [0, 2]
        expected_helper_pos = [[1, 2]]
        self.assertEqual(next_state.agent_pos[0].tolist(), [2, 2])
        self.assertEqual(next_state.helper_pos.tolist(), expected_helper_pos)
        self.assertEqual(next_state.box_pos[0].tolist(), expected_box_0_pos)

    def test_helper_cannot_pull_box_if_helper_goes_out_of_bounds(self):
        """
        Tests that the helper cannot pull a box if the helper would move out of bounds after the pull
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        # Set up: helper at [4, 4], box to pull at [4, 3] (helper would move to [4, 5] which is out of bounds)
        next_state.helper_pos = jnp.array([[4, 4]])
        next_state.box_pos = next_state.box_pos.at[0].set(
            jnp.array([4, 3])
        )  # box to pull

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.PULL_BOX_RIGHT_FROM_LEFT

        next_state = env.step_env(new_key, next_state, next_action, next_helper_action)[
            1
        ]

        # Box should not move, helper should not move (would go out of bounds)
        expected_box_0_pos = [4, 3]
        expected_helper_pos = [[4, 4]]
        self.assertEqual(next_state.helper_pos.tolist(), expected_helper_pos)
        self.assertEqual(next_state.box_pos[0].tolist(), expected_box_0_pos)

    def test_helper_can_move(self):
        """
        Tests that the helper can move to an empty space successfully
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.helper_pos = jnp.array([[2, 2]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.RIGHT

        next_state = env.step_env(new_key, next_state, next_action, next_helper_action)[
            1
        ]

        expected_helper_pos = [[2, 3]]
        self.assertEqual(next_state.helper_pos.tolist(), expected_helper_pos)

    def test_helper_collides_with_box(self):
        """
        Tests that the helper cannot move if it would collide with a box
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.helper_pos = jnp.array([[2, 2]])
        next_state.box_pos = next_state.box_pos.at[0].set(
            jnp.array([2, 3])
        )  # box blocking movement

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.RIGHT

        next_state = env.step_env(new_key, next_state, next_action, next_helper_action)[
            1
        ]

        # Helper should not move
        expected_helper_pos = [[2, 2]]
        self.assertEqual(next_state.helper_pos.tolist(), expected_helper_pos)

    def test_helper_collides_with_wall(self):
        """
        Tests that the helper cannot move if it would collide with a wall
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.helper_pos = jnp.array([[3, 0]])  # next to wall at [3, 1]

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.RIGHT

        next_state = env.step_env(new_key, next_state, next_action, next_helper_action)[
            1
        ]

        # Helper should not move (wall at [3, 1])
        expected_helper_pos = [[3, 0]]
        self.assertEqual(next_state.helper_pos.tolist(), expected_helper_pos)

    def test_helper_collides_with_agent(self):
        """
        Tests that the helper cannot move if it would collide with an agent
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.helper_pos = jnp.array([[0, 2]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.LEFT

        next_state = env.step_env(new_key, next_state, next_action, next_helper_action)[
            1
        ]

        # Helper should not move
        expected_helper_pos = [[0, 2]]
        self.assertEqual(next_state.helper_pos.tolist(), expected_helper_pos)

    def test_helper_cannot_move_out_of_bounds(self):
        """
        Tests that the helper cannot move out of bounds
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.helper_pos = jnp.array([[4, 4]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DOWN

        next_state = env.step_env(new_key, next_state, next_action, next_helper_action)[
            1
        ]

        # Helper should not move (would go out of bounds)
        expected_helper_pos = [[4, 4]]
        self.assertEqual(next_state.helper_pos.tolist(), expected_helper_pos)

    def test_box_collides_with_box(self):
        """
        Tests that the box should not be able to move, since it collides with another box
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.box_pos = next_state.box_pos.at[0].set(jnp.array([1, 0]))
        next_state.helper_pos = jnp.array([[1, 2]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.PUSH_BOX_LEFT

        expected_box_1_pos = [1, 1]
        self.assertEqual(next_state.box_pos[1].tolist(), expected_box_1_pos)

        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_box_1_pos = [1, 1]
        self.assertEqual(next_state.box_pos[1].tolist(), expected_box_1_pos)

    def test_box_collides_with_wall(self):
        """
        Tests that the box collides with a wall, and does not move
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.box_pos = next_state.box_pos.at[1].set(jnp.array([2, 1]))
        next_state.helper_pos = jnp.array([[1, 1]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.PUSH_BOX_DOWN

        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        expected_box_1_pos = [2, 1]
        self.assertEqual(next_state.box_pos[1].tolist(), expected_box_1_pos)

    def test_box_cannot_move_out_of_bounds(self):
        """
        Tests that the box cannot move out of bounds
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.helper_pos = jnp.array([[1, 2]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.PUSH_BOX_LEFT

        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        expected_box_1_pos = [1, 0]
        self.assertEqual(next_state.box_pos[1].tolist(), expected_box_1_pos)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.PUSH_BOX_LEFT

        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        expected_box_1_pos = [1, 0]
        self.assertEqual(next_state.box_pos[1].tolist(), expected_box_1_pos)

    def test_box_collides_with_agent(self):
        """
        Tests that the box does not move if it means it collides with an agent
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.helper_pos = jnp.array([[2, 1]])

        new_key = jax.random.split(key, 1)
        next_action = {
            "agent_0": Actions.STAY,
            "agent_1": Actions.STAY,
        }
        next_helper_action = env.HelperActions.PUSH_BOX_UP
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_box_1_pos = [1, 1]
        self.assertEqual(next_state.box_pos[1].tolist(), expected_box_1_pos)

    def test_freeze_agent_1_action(self):
        """
        Tests that the freeze agent 1 helper action works as expected (undoes agent 1's action)
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.FREEZE_AGENT_1
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_1_pos = [0, 3]
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

    def test_freeze_agent_1_means_agent_0_can_move_if_there_would_be_collision_otherwise(
        self,
    ):
        """
        Tests that agent 0 can move if agent 1 is frozen and there would be a collision otherwise if agent 1 was not frozen
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.FREEZE_AGENT_1
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_0_pos = [0, 2]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_pos)

    def test_freeze_agent_1_freezes_agent_1_for_2_timesteps(self):
        """
        Tests that agent 1 is frozen for 2 timesteps when the freeze action is selected
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.FREEZE_AGENT_1
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_1_pos = [0, 3]
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_1_pos = [0, 2]
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

    def test_freeze_agent_1_freezes_agent_1_for_2_timesteps_and_empowerment_is_zero(
        self,
    ):
        """
        Tests that agent 1 is frozen for 2 timesteps when the freeze action is selected and its empowerment is zero for those two timesteps
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.FREEZE_AGENT_1
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        empowerment_before = estimate_empowerment_variance_proxy(
            env, next_state, horizon=2
        )
        print("EMPOWERMENT DIVERSITY STEP 1 (Agent was frozen):", empowerment_before)

        empowerment_before = estimate_empowerment_monte_carlo(
            env, next_state, horizon=2
        )
        print("EMPOWERMENT DIRECT STEP 1 (Agent was frozen):", empowerment_before)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        empowerment_after = estimate_empowerment_variance_proxy(
            env, next_state, horizon=2
        )
        print(
            "EMPOWERMENT DIVERSITY STEP 2 (No action, but should still be frozen):",
            empowerment_after,
        )

        empowerment_after = estimate_empowerment_monte_carlo(env, next_state, horizon=2)
        print(
            "EMPOWERMENT DIRECT STEP 2 (No action, but should still be frozen):",
            empowerment_after,
        )

        expected_agent_1_pos = [0, 3]
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        empowerment_after = estimate_empowerment_variance_proxy(
            env, next_state, horizon=2
        )
        print(
            "EMPOWERMENT DIVERSITY STEP 3 (Agent 1 should be unfrozen):",
            empowerment_after,
        )

        empowerment_after = estimate_empowerment_monte_carlo(env, next_state, horizon=2)
        print(
            "EMPOWERMENT DIRECT STEP 3 (Agent 1 should be unfrozen):", empowerment_after
        )

        expected_agent_1_pos = [0, 2]
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

    def test_refreeze_agent_1(self):
        """
        Tests that agent 1 can be re-frozen before the freeze timer runs out (2 step freeze timer)
        """
        env = MultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.FREEZE_AGENT_1
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.FREEZE_AGENT_1
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_1_pos = [0, 3]
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_1_pos = [0, 2]
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

    def test_agent_1_gets_reward_in_independent_mode(self):
        """
        Tests that agents get the expected amount of reward.
        """
        env = MultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT, debug=True)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.FREEZE_AGENT_1
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_reward = 1

        self.assertListEqual(
            next_state.agent_pos[0].tolist(), next_state.goal_pos[0].tolist()
        )
        self.assertListEqual(
            next_state.agent_pos[1].tolist(), next_state.goal_pos[1].tolist()
        )

        self.assertEqual(rewards[0], expected_reward)
        self.assertEqual(rewards[1], expected_reward)

    def test_agent_1_does_not_get_reward_if_it_was_frozen_in_independent_mode(self):
        """
        Tests that agent 1 does not get reward if its frozen on the turn it was about to get into the goal
        """
        env = MultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.DOWN}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.RIGHT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.RIGHT}
        next_helper_action = env.HelperActions.FREEZE_AGENT_1
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_reward = 0

        self.assertEqual(rewards[0], expected_reward)
        self.assertEqual(rewards[1], expected_reward)

    def test_both_agents_can_get_reward_going_to_same_goal_in_independent_mode(self):
        """
        Tests that agents can get reward from going to the same goal in independent mode if there is only one goal.
        """
        env = MultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT, debug=True, num_goals=1)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.goal_pos = jnp.array([[0, 3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.goal_pos = jnp.array([[0, 3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.goal_pos = jnp.array([[0, 3]])

        expected_agent_pos = [0, 3]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_pos)
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_pos)

        expected_agent_0_reward = 1
        expected_agent_1_reward = 1

        self.assertListEqual(env.agent_to_goal_mapping.tolist(), [0, 0])
        self.assertListEqual(next_state.goal_pos.tolist(), [[0, 3]])

        self.assertEqual(rewards[0], expected_agent_0_reward)
        self.assertEqual(rewards[1], expected_agent_1_reward)

    def test_agents_get_reward_in_coop_mode(self):
        """
        Tests that both agents get reward in cooperative mode if both make it to their assigned goal
        """
        env = MultiAgentGridWorld(env_mode=EnvMode.COOPERATIVE, debug=True)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_0_pos = [0, 1]
        expected_agent_1_pos = [0, 3]

        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_pos)
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

        self.assertListEqual(next_state.goal_pos.tolist(), [[0, 1], [0, 3]])
        self.assertListEqual(env.agent_to_goal_mapping.tolist(), [0, 1])

        expected_reward = 1

        self.assertEqual(rewards[0], expected_reward)
        self.assertEqual(rewards[1], expected_reward)

    def test_agents_do_not_get_reward_in_coop_mode_if_not_to_assigned_goals(self):
        """
        Tests that both agents get reward in cooperative mode if both make it to their assigned goal
        """
        env = MultiAgentGridWorld(env_mode=EnvMode.COOPERATIVE, debug=True)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.goal_pos = jnp.array([[0, 3], [0, 1]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.goal_pos = jnp.array([[0, 3], [0, 1]])

        expected_agent_0_pos = [0, 1]
        expected_agent_1_pos = [0, 3]

        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_pos)
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

        self.assertListEqual(next_state.goal_pos.tolist(), [[0, 3], [0, 1]])
        self.assertListEqual(env.agent_to_goal_mapping.tolist(), [0, 1])

        expected_reward = 0

        self.assertEqual(rewards[0], expected_reward)
        self.assertEqual(rewards[1], expected_reward)

    def test_agents_do_not_get_reward_in_coop_mode_if_only_one_makes_it(self):
        """
        Tests that neither agent gets a reward in coop mode if only one makes it
        """
        env = MultiAgentGridWorld(env_mode=EnvMode.COOPERATIVE, debug=True)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_0_pos = [0, 1]
        expected_agent_1_pos = [0, 2]

        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_pos)
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

        expected_reward = 0

        self.assertEqual(rewards[0], expected_reward)
        self.assertEqual(rewards[1], expected_reward)

    def test_box_can_cover_goal_if_flagged_true(self):
        """
        Tests that a box can be pushed to cover the goal if the flag is set to true
        """
        env = MultiAgentGridWorld(
            env_mode=EnvMode.COOPERATIVE, boxes_can_cover_goal=True
        )
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.box_pos = next_state.box_pos.at[3].set(jnp.array([2, 4]))
        next_state.helper_pos = jnp.array([[3, 4]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.LEFT}
        next_helper_action = (
            env.HelperActions.PUSH_BOX_UP
        )  # tries to push this box onto goal
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_box_2_pos = [1, 4]

        self.assertEqual(next_state.box_pos[3].tolist(), expected_box_2_pos)

    def test_box_cannot_cover_goal_if_flagged_false(self):
        """
        Tests that a box cannot be pushed to cover the goal if the flag is set to false
        """
        env = MultiAgentGridWorld(
            env_mode=EnvMode.COOPERATIVE, boxes_can_cover_goal=False
        )
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.box_pos = next_state.box_pos.at[3].set(jnp.array([2, 4]))
        next_state.helper_pos = jnp.array([[3, 4]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.LEFT}
        next_helper_action = (
            env.HelperActions.PUSH_BOX_UP
        )  # tries to push this box onto goal
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_box_2_pos = [2, 4]

        self.assertEqual(next_state.box_pos[3].tolist(), expected_box_2_pos)

    def test_agent_cannot_move_if_in_trap(self):
        """
        Tests that an agent cannot move if it's in a trap
        """
        env = MultiAgentGridWorld(
            env_mode=EnvMode.INDEPENDENT, boxes_can_cover_goal=False, num_traps=2
        )
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.trap_pos = jnp.array([[2, 2], [3, 3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2, 2], [3, 3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2, 2], [3, 3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2, 2], [3, 3]])
        expected_agent_0_pos = [2, 2]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_pos)
        self.assertEqual(next_state.trap_pos[0].tolist(), expected_agent_0_pos)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2, 2], [3, 3]])

        # agent 0 should not move because it's in a trap
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_pos)

    def test_agent_gets_consequence_if_in_trap(self):
        """
        Tests that an agent gets penalized if it's in a trap
        """
        env = MultiAgentGridWorld(
            env_mode=EnvMode.INDEPENDENT, boxes_can_cover_goal=False, num_traps=2
        )
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.trap_pos = jnp.array([[2, 2], [3, 3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2, 2], [3, 3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2, 2], [3, 3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2, 2], [3, 3]])
        expected_agent_0_pos = [2, 2]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_pos)
        self.assertEqual(next_state.trap_pos[0].tolist(), expected_agent_0_pos)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2, 2], [3, 3]])

        expected_reward = -1
        self.assertEqual(rewards[0], expected_reward)

    def test_independent_mode_with_one_goal(self):
        env = MultiAgentGridWorld(
            env_mode=EnvMode.INDEPENDENT, boxes_can_cover_goal=False, num_goals=1
        )
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        self.assertEqual(next_state.goal_pos.shape, (1, 2))
        next_state.goal_pos = jnp.array([[0, 2]])
        print("(1) NEXT STATE FOR INDEP MODE WITH ONE GOAL:", next_state)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        print("(2) NEXT STATE FOR INDEP MODE WITH ONE GOAL:", next_state)
        expected_reward = 1
        print("REWARDS FOR INDEP MODE WITH ONE GOAL:", rewards)
        self.assertEqual(rewards[1], expected_reward)

        # new_key = jax.random.split(key, 1)
        # next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.STAY}
        # next_helper_action = env.HelperActions.DO_NOTHING
        # obs, next_state, rewards, dones, _ = env.step_env(
        #     new_key, next_state, next_action, next_helper_action
        # )
        # self.assertEqual(rewards[1], expected_reward)
        # self.assertEqual(rewards[0], expected_reward)
        # self.assertEqual(
        #     next_state.agent_pos[0].tolist(), next_state.goal_pos[0].tolist()
        # )

    def test_single_agent_mode(self):
        env = MultiAgentGridWorld(
            env_mode=EnvMode.INDEPENDENT, boxes_can_cover_goal=False, num_agents=1
        )
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        self.assertEqual(next_state.agent_pos.shape, (1, 2))

    def test_competitive_mode(self):
        env = MultiAgentGridWorld(
            env_mode=EnvMode.COMPETITIVE,
            boxes_can_cover_goal=False,
            num_agents=2,
            num_goals=1,
        )
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.goal_pos = jnp.array([[0, 2]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        self.assertEqual(rewards[0], 0)
        self.assertEqual(rewards[1], 1)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.STAY}
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        self.assertEqual(rewards[0], 0)
        self.assertEqual(rewards[1], 1)

    def test_agent_to_goal_mapping_with_more_agents_than_goals(self):
        env = MultiAgentGridWorld(
            env_mode=EnvMode.INDEPENDENT,
            boxes_can_cover_goal=False,
            num_agents=2,
            num_goals=1,
        )
        self.assertListEqual(env.agent_to_goal_mapping.tolist(), [0, 0])

    def test_agent_to_goal_mapping_with_equal_agents_to_goals(self):
        env = MultiAgentGridWorld(
            env_mode=EnvMode.INDEPENDENT,
            boxes_can_cover_goal=False,
            num_agents=2,
            num_goals=2,
        )
        self.assertListEqual(env.agent_to_goal_mapping.tolist(), [0, 1])

    def test_agent_gets_reward_for_entering_goal_with_key(self):
        """
        Tests that an agent gets reward for entering the goal with the key
        """
        env = MultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT, num_goals=1)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.goal_pos = jnp.array([[0, 2]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        self.assertEqual(rewards[1], 1)
        self.assertEqual(rewards[0], 0)


    def test_agent_does_not_get_reward_for_entering_goal_without_key(self):
        env = MultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT, num_goals=1)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.key_pos = jnp.array([[3, 0]]) # imposisble to get to
        next_state.goal_pos = jnp.array([[0, 2]])
        next_state.has_key = jnp.array([False, False])
        
        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        self.assertEqual(next_state.has_key[1], False)
        self.assertEqual(next_state.agent_pos[1].tolist(), [0, 2])
        self.assertEqual(next_state.goal_pos[0].tolist(), [0, 2])
        self.assertEqual(rewards[1], 0)
        self.assertEqual(rewards[0], 0)

    def test_agent_picks_up_key_if_at_key_position(self):
        """
        Tests that an agent picks up the key if it's at the key position
        """
        env = MultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT, num_goals=1)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.key_pos = jnp.array([[0, 2]])
        next_state.has_key = jnp.array([False, False])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        self.assertEqual(next_state.has_key[1], True)
        self.assertEqual(next_state.has_key[0], False)

if __name__ == "__main__":
    absltest.main()
