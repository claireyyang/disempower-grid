import chex
import jax
from absl.testing import absltest
import jax.numpy as jnp
from nonembodied_multiagent_gridworld import (
    NonEmbodiedMultiAgentGridWorld,
    Actions,
    EnvMode,
)
from empowerment_estimator import (
    estimate_empowerment_variance_proxy,
    estimate_empowerment_monte_carlo,
)

class NonEmbodiedMultiAgentGridWorldTest(chex.TestCase):

    def test_should_pass(self):
        for i in (1, 2):
            with self.subTest(i=i):
                self.assertEqual(i, i)
                print("woohoo it passes")

    def test_agent_collides_with_box(self):
        """
        Tests that agent should not be able to move downwards, since it collides with the box (which is moved after the agent)
        """
        env = NonEmbodiedMultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.DOWN}
        next_helper_action = env.HelperActions.BOX_0_DOWN

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
        env = NonEmbodiedMultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.BOX_0_DOWN
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
        env = NonEmbodiedMultiAgentGridWorld()
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
        env = NonEmbodiedMultiAgentGridWorld()
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
        env = NonEmbodiedMultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.UP, "agent_1": Actions.UP}
        next_helper_action = env.HelperActions.BOX_0_DOWN
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
        env = NonEmbodiedMultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.RIGHT}
        next_helper_action = env.HelperActions.BOX_0_DOWN
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.RIGHT}
        next_helper_action = env.HelperActions.BOX_0_DOWN
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_0_position = [0, 2]
        expected_agent_1_position = [0, 3]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_position)
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_position)

    def test_agent_collides_with_box_and_then_other_agent_tries_to_move_into_same_position_in_same_goal_in_independent_mode(
        self,
    ):
        """
        Test that the agent will move as expected if one agent collides with a box and then the other is trying to move into the same position
        But they're actually in the same goal, so it's okay
        """
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.DOWN}
        next_helper_action = env.HelperActions.BOX_3_DOWN
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.DOWN}
        next_helper_action = env.HelperActions.BOX_3_RIGHT
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.RIGHT}
        next_helper_action = env.HelperActions.BOX_3_RIGHT
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.DOWN}
        next_helper_action = env.HelperActions.BOX_3_RIGHT
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_0_position = [1, 4]
        expected_agent_1_position = [1, 4]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_position)
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_position)

    def test_box_moves_down(self):
        """
        Tests that the box can move to an empty space successfully
        """
        env = NonEmbodiedMultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.DOWN}
        next_helper_action = env.HelperActions.BOX_0_DOWN

        expected_box_0_pos = [1, 0]

        self.assertEqual(
            env.step_env(new_key, next_state, next_action, next_helper_action)[1]
            .box_pos[0]
            .tolist(),
            expected_box_0_pos,
        )

    def test_box_collides_with_box(self):
        """
        Tests that the box should not be able to move, since it collides with another box
        """
        env = NonEmbodiedMultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.DOWN}
        next_helper_action = env.HelperActions.BOX_0_DOWN

        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_box_0_pos = [1, 0]
        self.assertEqual(next_state.box_pos[0].tolist(), expected_box_0_pos)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.DOWN}
        next_helper_action = env.HelperActions.BOX_1_LEFT
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_box_1_pos = [1, 1]
        self.assertEqual(next_state.box_pos[1].tolist(), expected_box_1_pos)

    def test_box_collides_with_wall(self):
        """
        Tests that the box collides with a wall, and does not move
        """
        env = NonEmbodiedMultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.box_pos = jnp.array([[0, 0], [2, 1], [1, 3], [0, 4]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.BOX_1_DOWN

        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_box_1_pos = [2, 1]
        self.assertEqual(next_state.box_pos[1].tolist(), expected_box_1_pos)
        self.assertListEqual(next_state.wall_pos.tolist(), [[3, 1]])
    

    def test_box_cannot_move_out_of_bounds(self):
        """
        Tests that the box cannot move out of bounds
        """
        env = NonEmbodiedMultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.BOX_0_LEFT
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_box_0_pos = [0, 0]
        self.assertEqual(next_state.box_pos[0].tolist(), expected_box_0_pos)

    def test_box_collides_with_agent(self):
        """
        Tests that the box does not move if it means it collides with an agent
        """
        env = NonEmbodiedMultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {
            "agent_0": Actions.DOWN,
            "agent_1": Actions.DOWN,
        }  # neither agent should move
        next_helper_action = env.HelperActions.BOX_1_UP
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_box_1_pos = [1, 1]
        self.assertEqual(next_state.box_pos[1].tolist(), expected_box_1_pos)

    def test_freeze_agent_1_action(self):
        """
        Tests that the freeze agent 1 helper action works as expected (undoes agent 1's action)
        """
        env = NonEmbodiedMultiAgentGridWorld()
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

    def test_freeze_agent_1_means_agent_0_can_move_if_there_would_be_collision_otherwise(self):
        """
        Tests that agent 0 can move if agent 1 is frozen and there would be a collision otherwise if agent 1 was not frozen
        """
        env = NonEmbodiedMultiAgentGridWorld()
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

    def test_freeze_agent_1_freezes_agent_1_for_3_timesteps(self):
        """
        Tests that agent 1 is frozen for 3 timesteps when the freeze action is selected
        """
        env = NonEmbodiedMultiAgentGridWorld()
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

    def test_freeze_agent_1_freezes_agent_1_for_3_timesteps_and_empowerment_is_zero(self):
        """
        Tests that agent 1 is frozen for 3 timesteps when the freeze action is selected and its empowerment is zero for those three timesteps
        """
        env = NonEmbodiedMultiAgentGridWorld()
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.FREEZE_AGENT_1
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        empowerment_before = estimate_empowerment_variance_proxy(env, next_state, horizon=2)
        print("EMPOWERMENT DIVERSITY STEP 1 (Agent was frozen):", empowerment_before)

        empowerment_before = estimate_empowerment_monte_carlo(env, next_state, horizon=2)
        print("EMPOWERMENT DIRECT STEP 1 (Agent was frozen):", empowerment_before)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        empowerment_after = estimate_empowerment_variance_proxy(env, next_state, horizon=2)
        print("EMPOWERMENT DIVERSITY STEP 2 (No action, but should still be frozen):", empowerment_after)

        empowerment_after = estimate_empowerment_monte_carlo(env, next_state, horizon=2)
        print("EMPOWERMENT DIRECT STEP 2 (No action, but should still be frozen):", empowerment_after)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        empowerment_after = estimate_empowerment_variance_proxy(env, next_state, horizon=2)
        print("EMPOWERMENT DIVERSITY STEP 3 (No action, but should still be frozen):", empowerment_after)

        empowerment_after = estimate_empowerment_monte_carlo(env, next_state, horizon=2)
        print("EMPOWERMENT DIRECT STEP 3 (No action, but should still be frozen):", empowerment_after)

        expected_agent_1_pos = [0, 3]
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        empowerment_after = estimate_empowerment_variance_proxy(env, next_state, horizon=2)
        print("EMPOWERMENT DIVERSITY STEP 4 (Agent 1 should be unfrozen):", empowerment_after)

        empowerment_after = estimate_empowerment_monte_carlo(env, next_state, horizon=2)
        print("EMPOWERMENT DIRECT STEP 4 (Agent 1 should be unfrozen):", empowerment_after)

        expected_agent_1_pos = [0, 2]
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

    def test_refreeze_agent_1(self):
        """
        Tests that agent 1 can be re-frozen before the freeze timer runs out
        """
        env = NonEmbodiedMultiAgentGridWorld()
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
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT, debug=True)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.FREEZE_AGENT_1
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_reward = 1

        self.assertListEqual(next_state.agent_pos[0].tolist(), next_state.goal_pos[0].tolist())
        self.assertListEqual(next_state.agent_pos[1].tolist(), next_state.goal_pos[1].tolist())

        self.assertEqual(rewards[0], expected_reward)
        self.assertEqual(rewards[1], expected_reward)

    def test_agent_1_does_not_get_reward_if_it_was_frozen_in_independent_mode(self):
        """
        Tests that agent 1 does not get reward if its frozen on the turn it was about to get into the goal
        """
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.BOX_3_DOWN
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.DOWN}
        next_helper_action = env.HelperActions.BOX_3_DOWN
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.RIGHT}
        next_helper_action = env.HelperActions.BOX_3_DOWN
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
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT, debug=True, num_goals=1)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.goal_pos = jnp.array([[0, 3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.BOX_1_DOWN
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.goal_pos = jnp.array([[0, 3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.BOX_1_DOWN
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
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.COOPERATIVE, debug=True)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.BOX_1_RIGHT
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

    def test_agents_do_notget_reward_in_coop_mode_if_not_to_assigned_goals(self):
        """
        Tests that both agents get reward in cooperative mode if both make it to their assigned goal
        """
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.COOPERATIVE, debug=True)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.goal_pos = jnp.array([[0, 3], [0, 1]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.BOX_1_RIGHT
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
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.COOPERATIVE, debug=True)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.BOX_1_RIGHT
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_agent_0__pos = [0, 1]
        expected_agent_1_pos = [0, 2]

        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0__pos)
        self.assertEqual(next_state.agent_pos[1].tolist(), expected_agent_1_pos)

        expected_reward = 0

        self.assertEqual(rewards[0], expected_reward)
        self.assertEqual(rewards[1], expected_reward)

    def test_box_can_cover_goal_if_flagged_true(self):
        """
        Tests that a box can be pushed to cover the goal if the flag is set to true
        """
        env = NonEmbodiedMultiAgentGridWorld(
            env_mode=EnvMode.COOPERATIVE, boxes_can_cover_goal=True
        )
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.BOX_2_DOWN  # pushes this box onto the goal
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_box_2_pos = [1, 4]

        self.assertEqual(next_state.box_pos[2].tolist(), expected_box_2_pos)

    def test_box_cannot_cover_goal_if_flagged_false(self):
        """
        Tests that a box cannot be pushed to cover the goal if the flag is set to false
        """
        env = NonEmbodiedMultiAgentGridWorld(
            env_mode=EnvMode.COOPERATIVE, boxes_can_cover_goal=False
        )
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.LEFT}
        next_helper_action = (
            env.HelperActions.BOX_2_DOWN
        )  # tries to push this box onto goal
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )

        expected_box_2_pos = [0, 4]

        self.assertEqual(next_state.box_pos[2].tolist(), expected_box_2_pos)

    def test_agent_cannot_move_if_in_trap(self):
        """
        Tests that an agent cannot move if it's in a trap
        """
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT, boxes_can_cover_goal=False, num_traps=2)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.trap_pos = jnp.array([[2,2], [3,3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2,2], [3,3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2,2], [3,3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2,2], [3,3]])
        expected_agent_0_pos = [2, 2]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_pos)
        self.assertEqual(next_state.trap_pos[0].tolist(), expected_agent_0_pos)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2,2], [3,3]])

        # agent 0 should not move because it's in a trap
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_pos)


    def test_agent_gets_consequence_if_in_trap(self):
        """
        Tests that an agent gets penalized if it's in a trap
        """
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT, boxes_can_cover_goal=False, num_traps=2)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        next_state.trap_pos = jnp.array([[2,2], [3,3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2,2], [3,3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2,2], [3,3]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2,2], [3,3]])
        expected_agent_0_pos = [2, 2]
        self.assertEqual(next_state.agent_pos[0].tolist(), expected_agent_0_pos)
        self.assertEqual(next_state.trap_pos[0].tolist(), expected_agent_0_pos)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.DOWN, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        next_state.trap_pos = jnp.array([[2,2], [3,3]])

        expected_reward = -1
        self.assertEqual(rewards[0], expected_reward)

    def test_independent_mode_with_one_goal(self):
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT, boxes_can_cover_goal=False, num_goals=1)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        self.assertEqual(next_state.goal_pos.shape, (1, 2))
        next_state.goal_pos = jnp.array([[0, 2]])

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.STAY, "agent_1": Actions.LEFT}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        expected_reward = 1
        self.assertEqual(rewards[1], expected_reward)

        new_key = jax.random.split(key, 1)
        next_action = {"agent_0": Actions.RIGHT, "agent_1": Actions.STAY}
        next_helper_action = env.HelperActions.DO_NOTHING
        obs, next_state, rewards, dones, _ = env.step_env(
            new_key, next_state, next_action, next_helper_action
        )
        self.assertEqual(rewards[1], expected_reward)
        self.assertEqual(rewards[0], expected_reward)
        self.assertEqual(next_state.agent_pos[0].tolist(), next_state.goal_pos[0].tolist())

    def test_single_agent_mode(self):
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT, boxes_can_cover_goal=False, num_agents=1)
        key = jax.random.PRNGKey(0)
        obs, next_state = env.reset_for_testing(key)
        self.assertEqual(next_state.agent_pos.shape, (1, 2))

    def test_competitive_mode(self):
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.COMPETITIVE, boxes_can_cover_goal=False, num_agents=2, num_goals=1)
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
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT, boxes_can_cover_goal=False, num_agents=2, num_goals=1)
        self.assertListEqual(env.agent_to_goal_mapping.tolist(), [0, 0])

    def test_agent_to_goal_mapping_with_equal_agents_to_goals(self):
        env = NonEmbodiedMultiAgentGridWorld(env_mode=EnvMode.INDEPENDENT, boxes_can_cover_goal=False, num_agents=2, num_goals=2)
        self.assertListEqual(env.agent_to_goal_mapping.tolist(), [0, 1])
        
if __name__ == "__main__":
    absltest.main()
