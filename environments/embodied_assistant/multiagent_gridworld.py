from enum import IntEnum

import jax
import jax.numpy as jnp
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from typing import Tuple, Dict
import chex
from functools import partial

from .helper_actions_utils import create_helper_actions_enum
from .helper_actions_utils import create_helper_action_directions


class Actions(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3
    STAY = 4


class EnvMode(IntEnum):
    COOPERATIVE = 0
    INDEPENDENT = 1
    COMPETITIVE = 2


@chex.dataclass
class State:
    agent_pos: chex.Array  # (2, 2) array for both agents' positions
    goal_pos: chex.Array  # (num_goals, 2) array for both goals' positions
    box_pos: chex.Array  # (num_boxes, 2) array for boxes' positions
    trap_pos: chex.Array  # (num_traps, 2) array for traps' positions
    wall_pos: chex.Array  # (num_walls, 2) array for walls' positions
    helper_pos: chex.Array  # (1, 2) array for helper agent's position
    key_pos: chex.Array # (num_keys, 2) array for keys' positions
    time: int  # increments +1 for the agents' step and helper agent's step bundled together
    terminal: bool
    freeze_timer: (
        chex.Array
    )  # counter for how many timesteps agent 1 remains frozen - (1)
    has_key: chex.Array # (2) array for whether each agent has the key


@jax.tree_util.register_pytree_node_class
class MultiAgentGridWorld(MultiAgentEnv):
    """heightxwidth gridworld with num_boxes boxes, 1 or 2 agents, and 1 or 2 goals and num_traps traps and num_walls walls"""

    def __init__(
        self,
        height: int = 5,
        width: int = 5,
        max_steps: int = 50,
        num_agents: int = 2,
        random_reset_helper_pos: bool = False,
        debug: bool = False,
        env_mode: EnvMode = EnvMode.INDEPENDENT,
        boxes_can_cover_goal: bool = True,
        num_traps: int = 1,
        num_boxes: int = 4,
        num_goals: int = 2,
        num_walls: int = 1,
        num_keys: int = 1,
        no_freeze: bool = False,
        no_pull_helper_action: bool = False,  # default to being able to push and pull
    ):
        self.env_mode = env_mode
        self.num_agents = num_agents
        assert num_agents == 1 or num_agents == 2, "num_agents must be 1 or 2"
        self.num_traps = num_traps
        self.num_boxes = num_boxes
        self.num_goals = num_goals
        self.num_walls = num_walls
        self.num_keys = num_keys
        if self.num_keys != 1 and self.num_keys != 2:
            raise Exception(f"num_keys must be 1 or 2, but got {self.num_keys}")
        self.no_freeze = no_freeze
        self.no_pull_helper_action = no_pull_helper_action
        self.agent_to_goal_mapping = jnp.arange(
            num_goals
        )  # NOTE: assumes that number of goals is less than or equal to number of agents
        if len(self.agent_to_goal_mapping) < num_agents:
            self.agent_to_goal_mapping = jnp.concatenate(
                (self.agent_to_goal_mapping, self.agent_to_goal_mapping)
            )
        if self.env_mode == EnvMode.COOPERATIVE:
            assert num_goals == 2, "Cooperative mode must have exactly 2 goals"
            assert num_agents == 2, "Cooperative mode must have exactly 2 agents"
        elif self.env_mode == EnvMode.COMPETITIVE:
            assert num_agents == 2, "Competitive mode must have exactly 2 agents"
            assert num_goals == 1, "Competitive mode must have exactly 1 goal"
        self.height = height
        self.width = width
        super().__init__(num_agents=self.num_agents)
        self.max_steps = max_steps
        self.random_reset_helper_pos = random_reset_helper_pos
        self.debug = debug
        self.agents_can_go_to_same_goal = (
            True if self.num_goals == 1 and self.env_mode != EnvMode.COMPETITIVE else False
        )
        self.boxes_can_cover_goal = boxes_can_cover_goal
        self.action_set = jnp.array([x.value for x in Actions])
        if self.num_agents == 2:
            self.agents = ["agent_0", "agent_1"]
        else:
            self.agents = ["agent_0"]
        self.HelperActions = create_helper_actions_enum(
            no_freeze, no_pull_helper_action
        )
        self.helper_action_set = jnp.array([x.value for x in self.HelperActions])

        # Movement vectors for each action
        self.action_to_dir = jnp.array(
            [
                [0, 1],  # right
                [1, 0],  # down
                [0, -1],  # left
                [-1, 0],  # up
                [0, 0],  # stay
            ]
        )

        self.helper_action_to_dir = create_helper_action_directions(
            no_pull_helper_action, no_freeze
        )

        self.all_pos = jnp.array(
            [[x, y] for x in range(self.height) for y in range(self.width)]
        )

    def tree_flatten(self):
        values = ()
        aux = (
            self.height,
            self.width,
            self.max_steps,
            self.num_agents,
            self.random_reset_helper_pos,
            self.debug,
            self.env_mode,
            self.boxes_can_cover_goal,
            self.num_traps,
            self.num_boxes,
            self.num_goals,
            self.num_walls,
            self.num_keys,
            self.no_freeze,
            self.no_pull_helper_action,
        )
        return values, aux

    @classmethod
    def tree_unflatten(cls, aux: Tuple, values: Tuple):
        (
            height,
            width,
            max_steps,
            num_agents,
            random_reset_helper_pos,
            debug,
            env_mode,
            boxes_can_cover_goal,
            num_traps,
            num_boxes,
            num_goals,
            num_walls,
            num_keys,
            no_freeze,
            no_pull_helper_action,
        ) = aux
        obj = cls(
            height,
            width,
            max_steps,
            num_agents,
            random_reset_helper_pos,
            debug,
            env_mode,
            boxes_can_cover_goal,
            num_traps,
            num_boxes,
            num_goals,
            num_walls,
            num_keys,
            no_freeze,
            no_pull_helper_action,
        )
        return obj

    @partial(jax.jit, static_argnums=[0])
    def reset_for_testing(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Reset environment state for unit tests."""
        key1, key2 = jax.random.split(key)

        _, state = self.reset(key1)

        og_locations = jnp.array(
            [[0, 1], [0, 3], [1, 4], [4, 0], [3, 2]]
        )  # NOTE: if I change this again, then the tests need to be updated...

        state.agent_pos = og_locations[: self.num_agents]

        state.key_pos = state.agent_pos # NOTE: keys are placed on top of agents so don't have to change tests

        if self.debug:
            # sets agents on top of goals for debugging
            state.goal_pos = og_locations[: self.num_agents]
        else:
            state.goal_pos = og_locations[
                self.num_agents : self.num_agents + self.num_goals
            ]

        state.trap_pos = og_locations[
            self.num_agents
            + self.num_goals : self.num_agents
            + self.num_goals
            + self.num_traps
        ]

        state.wall_pos = jnp.array([[3, 1]])

        state.box_pos = jnp.array([[0, 0], [1, 1], [0, 4], [1, 3]])  # 4 boxes

        state.helper_pos = jnp.array([[2, 0]])

        state.has_key = jnp.array([True, True])

        self.state = state
        obs = self.get_obs(state)

        return obs, state

    @partial(jax.jit, static_argnums=[0])
    def reset_specific_positions(
        self,
        key: chex.PRNGKey,
        agent_pos: chex.Array,
        goal_pos: chex.Array,
        box_pos: chex.Array,
        trap_pos: chex.Array,
        wall_pos: chex.Array,
        helper_pos: chex.Array,
        key_pos: chex.Array,
    ) -> Tuple[chex.Array, State]:
        """Reset environment state with specific positions for certain scenarios to be tested"""
        key1, key2 = jax.random.split(key)

        assert agent_pos.shape == (
            self.num_agents,
            2,
        ), f"agent_pos.shape: {agent_pos.shape} != ({self.num_agents}, 2)"
        assert goal_pos.shape == (
            self.num_goals,
            2,
        ), f"goal_pos.shape: {goal_pos.shape} != ({self.num_goals}, 2)"
        assert box_pos.shape == (
            self.num_boxes,
            2,
        ), f"box_pos.shape: {box_pos.shape} != ({self.num_boxes}, 2)"
        assert trap_pos.shape == (
            self.num_traps,
            2,
        ), f"trap_pos.shape: {trap_pos.shape} != ({self.num_traps}, 2)"
        assert wall_pos.shape == (
            self.num_walls,
            2,
        ), f"wall_pos.shape: {wall_pos.shape} != ({self.num_walls}, 2)"
        assert helper_pos.shape == (
            1,
            2,
        ), f"helper_pos.shape: {helper_pos.shape} != (1,2)"
        assert key_pos.shape == (
            self.num_keys,
            2,
        ), f"key_pos.shape: {key_pos.shape} != ({self.num_keys}, 2)"

        if self.random_reset_helper_pos:
            # randomly reset the helper to be positioned anywhere that is free on the grid

            # Create set of all occupied positions
            occupied_positions = jnp.concatenate(
                [
                    agent_pos,  # agents
                    goal_pos,  # goals
                    box_pos,  # boxes
                    trap_pos,  # traps
                    wall_pos,  # walls
                    key_pos,  # keys
                ],
                axis=0,
            )

            # Find all free positions
            def is_position_free(pos):
                # Check if position is not occupied by any object
                distances = jnp.sum(jnp.abs(occupied_positions - pos[None, :]), axis=1)
                return jnp.all(
                    distances > 0
                )  # All distances > 0 means position is free

            # Get all possible positions on the grid
            all_positions = jnp.array(
                [[x, y] for x in range(self.height) for y in range(self.width)]
            )

            # Filter to only free positions using where instead of boolean indexing
            free_mask = jax.vmap(is_position_free)(all_positions)
            num_free = jnp.sum(free_mask)

            # Use jnp.where to get indices of free positions, then select from those
            free_indices = jnp.where(free_mask, size=len(all_positions), fill_value=0)[
                0
            ]

            # Randomly select a free position for helper
            key1, helper_key = jax.random.split(key1)

            # Handle case where no free positions exist (shouldn't happen in practice)
            helper_idx = jax.random.randint(helper_key, (), 0, jnp.maximum(num_free, 1))
            helper_idx = jnp.clip(helper_idx, 0, num_free - 1)

            # Get the actual position index and then the position
            selected_pos_idx = free_indices[helper_idx]
            selected_position = all_positions[selected_pos_idx]
            helper_pos = selected_position[None, :]  # Add batch dimension to make (1, 2)

        state = State(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            box_pos=box_pos,
            trap_pos=trap_pos,
            wall_pos=wall_pos,
            helper_pos=helper_pos,
            key_pos=key_pos,
            time=0,
            terminal=False,
            freeze_timer=jnp.array(0),
            has_key=jnp.array([False, False]),
        )

        self.state = state
        obs = self.get_obs(state)

        return obs, state

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Reset environment state with random placement of agents, goals, boxes, traps, and walls."""
        key1, key2 = jax.random.split(key)

        # Total number of objects cannot exceed 9/10 of the grid
        assert (
            self.num_agents
            + self.num_boxes
            + self.num_traps
            + self.num_goals
            + self.num_walls
            <= (self.width * self.height) * 9 / 10
        )

        # Randomly place agents and goals
        # indices = jax.random.permutation(key1, len(self.all_pos))
        # rand_agent_pos = self.all_pos[indices[:self.num_agents]]
        # rand_goal_pos = self.all_pos[indices[self.num_agents:self.num_agents + self.num_goals]]
        # rand_box_pos = self.all_pos[indices[self.num_agents + self.num_goals:self.num_agents + self.num_goals + self.num_boxes]]
        # rand_trap_pos = self.all_pos[indices[self.num_agents + self.num_goals + self.num_boxes:self.num_agents + self.num_goals + self.num_boxes + self.num_traps]]
        # rand_wall_pos = self.all_pos[indices[self.num_agents + self.num_goals + self.num_boxes + self.num_traps:self.num_agents + self.num_goals + self.num_boxes + self.num_traps + self.num_walls]]
        # state = State(
        #     agent_pos=rand_agent_pos,
        #     goal_pos=rand_goal_pos,
        #     box_pos=rand_box_pos,
        #     trap_pos=rand_trap_pos,
        #     wall_pos=rand_wall_pos,
        #     time=0,
        #     terminal=False,
        #     freeze_timer=0
        # )

        # Define corner agent positions as a JAX array
        agent_pos_corners = jnp.array(
            [
                [0, 0],
                [0, self.width - 1],
                [self.height - 1, 0],
                [self.height - 1, self.width - 1],
            ]
        )

        # Define corner box positions as a JAX array
        box_pos_corners = jnp.array(
            [
                [[0, 1], [1, 0]],  # corner 0
                [[0, self.width - 2], [1, self.width - 1]],  # corner 1
                [[self.height - 2, 0], [self.height - 1, 1]],  # corner 2
                [
                    [self.height - 2, self.width - 1],
                    [self.height - 1, self.width - 2],
                ],  # corner 3
            ]
        )

        # Select a random corner
        n = 4  # For example to pick two different corners
        indices = jax.random.permutation(key1, n)[:4]
        corner, corner2, corner3, corner4 = (
            indices[0],
            indices[1],
            indices[2],
            indices[3],
        )

        all_other_positions = jnp.array(
            [
                [x, y]
                for x in range(1, self.height - 1)
                for y in range(1, self.width - 1)
            ]
        )  # NOTE: the traps and extra boxes and walls are not in any of the corner or corner adjacent positions
        # Use random permutation for other positions
        indices = jax.random.permutation(key1, len(all_other_positions))

        agent_pos = jnp.zeros((self.num_agents, 2), dtype=jnp.int32)
        agent_pos = agent_pos.at[0].set(agent_pos_corners[corner])  # First agent
        agent_pos = jnp.where(
            self.num_agents == 2,
            agent_pos.at[1].set(all_other_positions[indices[:1]].reshape(-1)),
            agent_pos,
        )

        goal_pos = jnp.zeros((self.num_goals, 2), dtype=jnp.int32)
        goal_pos = goal_pos.at[0].set(agent_pos_corners[corner3])
        goal_pos = jnp.where(
            self.num_goals == 2,
            goal_pos.at[1].set(agent_pos_corners[corner4]),
            goal_pos,
        )
        trap_pos = all_other_positions[indices[1 : self.num_traps + 1]]
        wall_pos = all_other_positions[
            indices[self.num_traps + 1 : self.num_traps + self.num_walls]
        ]
        # First set the corner box positions (first 2 boxes)
        box_pos_only_corners = box_pos_corners[corner]
        box_pos = jnp.zeros((self.num_boxes, 2), dtype=jnp.int32)
        box_pos = box_pos.at[:2].set(box_pos_only_corners)

        # Then set the remaining boxes (if any)
        if self.num_boxes > 2:  # Only if we need more than 2 boxes
            remaining_box_pos = all_other_positions[
                indices[self.num_traps + 1 : self.num_traps + self.num_boxes - 1]
            ]
            box_pos = box_pos.at[2:].set(remaining_box_pos)

        # TODO maybe: implement box can cover goal (to start), as well as box can cover trap??

        assert (
            len(box_pos) == self.num_boxes
        ), f"len(box_pos): {len(box_pos)} != self.num_boxes: {self.num_boxes}"

        # Place helper in a random free position
        helper_pos = all_other_positions[indices[self.num_traps + self.num_boxes - 1]][
            None, :
        ]

        # Place keys in random free positions
        if self.num_keys == 1:
            # Single key for both agents
            key_pos = all_other_positions[indices[self.num_traps + self.num_boxes + self.num_walls - 1]][
                None, :
            ]
        else:
            # Multiple keys - place them in different positions
            key_positions = []
            for i in range(self.num_keys):
                key_positions.append(
                    all_other_positions[indices[self.num_traps + self.num_boxes + self.num_walls - 1 + i]]
                )
            key_pos = jnp.array(key_positions)

        state = State(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            box_pos=box_pos,
            trap_pos=trap_pos,
            wall_pos=wall_pos,
            helper_pos=helper_pos,
            key_pos=key_pos,
            time=0,
            terminal=False,
            freeze_timer=0,
            has_key=jnp.array([False, False]),
        )

        self.state = state
        obs = self.get_obs(state)
        return obs, state

    @partial(jax.jit, static_argnums=[0, 4])
    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        helper_action: chex.Array,
    ) -> Tuple[chex.Array, State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""

        acts = jnp.array([actions[agent] for agent in self.agents])

        agents_next_state = self.step_agents(key, state, acts, helper_action)

        next_state = self.step_helper(agents_next_state, helper_action)

        rewards, agent_dones = self.calculate_agent_rewards_dones(next_state)

        next_state = next_state.replace(
            freeze_timer=jnp.maximum(next_state.freeze_timer - 1, 0)
        )
        next_state = next_state.replace(time=state.time + 1)

        is_game_done = self.is_terminal(next_state)
        next_state = next_state.replace(terminal=is_game_done)

        obs = self.get_obs(next_state)

        dones = {f"agent_{i}": agent_dones[i] for i in range(self.num_agents)}
        dones["__all__"] = is_game_done

        return obs, next_state, rewards, dones, {}

    @partial(jax.jit, static_argnums=[0])
    def step_agents(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: chex.Array,
        helper_action: chex.Array,
    ) -> State:
        """Update agent positions."""

        # Check if agent 1 is still frozen from previous timesteps or if the helper action is to freeze agent 1
        if not self.no_freeze:
            actions = jax.lax.cond(
                jnp.logical_or(
                    helper_action == self.HelperActions.FREEZE_AGENT_1.value,
                    jnp.any(state.freeze_timer > 0),
                ),
                lambda actions: actions.at[1].set(Actions.STAY.value),
                lambda actions: actions,
                actions,
            )

        print(f"In step_agents: actions.shape: {actions.shape}")
        print(f"In step_agents: next_pos.shape before update: {state.agent_pos.shape}")
        next_pos = state.agent_pos + self.action_to_dir[actions]
        print(f"In step_agents: next_pos.shape after update: {next_pos.shape}")

        # Bound positions to grid
        next_pos = jnp.clip(next_pos, 0, jnp.array([self.height - 1, self.width - 1]))

        # Check if positions would collide with any boxes
        box_collisions = jnp.any(
            jnp.all(next_pos[:, None, :] == state.box_pos[None, :, :], axis=-1), axis=-1
        )

        # Prevent updates for agents colliding with boxes (independently for each agent)
        next_pos = jnp.where(box_collisions[:, None], state.agent_pos, next_pos)

        # Check if positions would collide with any walls
        print(f"next_pos.shape: {next_pos.shape}")
        print(f"box_pos.shape: {state.box_pos.shape}")
        print(f"state.wall_pos.shape: {state.wall_pos.shape}")
        wall_collisions = jnp.any(
            jnp.all(next_pos[:, None, :] == state.wall_pos[None, :, :], axis=-1),
            axis=-1,
        )
        # Prevent updates for agents colliding with walls (independently for each agent)
        next_pos = jnp.where(wall_collisions[:, None], state.agent_pos, next_pos)

        # Check if positions would collide with helper agent
        helper_collisions = jnp.all(next_pos == state.helper_pos, axis=-1)
        # Prevent updates for agents colliding with helper agent (independently for each agent)
        next_pos = jnp.where(helper_collisions[:, None], state.agent_pos, next_pos)

        if self.num_agents == 2:

            agents_switching = jnp.logical_and(
                jnp.all(
                    next_pos[0] == state.agent_pos[1]
                ),  # Agent 0 moving to Agent 1's position
                jnp.all(
                    next_pos[1] == state.agent_pos[0]
                ),  # Agent 1 moving to Agent 0's position
            )

            # Check if positions would collide with each other
            agent_collision = jnp.all(next_pos[0] == next_pos[1])

            agent_collision = jnp.logical_or(agent_collision, agents_switching)

            # Only allow collision if both agents are on goals (and same goal is allowed)
            if self.agents_can_go_to_same_goal:

                def on_goal(x, y):
                    return jnp.all(x == y)

                agent0_goal = jax.vmap(on_goal, in_axes=(None, 0))(
                    next_pos[0], state.goal_pos
                )
                agent1_goal = jax.vmap(on_goal, in_axes=(None, 0))(
                    next_pos[1], state.goal_pos
                )
                both_on_goals = jnp.logical_and(
                    jnp.any(agent0_goal), jnp.any(agent1_goal)
                )
                on_same_goal = jnp.any(jnp.logical_and(agent0_goal, agent1_goal))
                both_on_goals_on_same_goal = jnp.logical_and(
                    both_on_goals, on_same_goal
                )
                # Only disable collision if they're both on goals (and same goal is allowed)
                agent_collision = jnp.where(
                    both_on_goals_on_same_goal, False, agent_collision
                )

            # Create collision mask and prevent updates for colliding agents
            collision_mask = jnp.where(
                agent_collision,  # If there's an agent-agent collision,
                jnp.array([True, True]),  # Block both agents
                jnp.array([False, False]),  # Otherwise, allow movement
            )

            # If there's an agent-agent collision, then block both agents (reverts to original positions)
            next_pos = jnp.where(collision_mask[:, None], state.agent_pos, next_pos)

        # Check if any agents are in traps because now they can't move (always on original position)
        trap_collisions = jnp.any(
            jnp.all(state.agent_pos[:, None, :] == state.trap_pos[None, :, :], axis=-1),
            axis=-1,
        )
        # If any agent is in a trap, then block all movement
        next_pos = jnp.where(trap_collisions[:, None], state.agent_pos, next_pos)

        # Agents can now enter goals without keys - reward logic will handle the key requirement

        # Key pickup logic - vectorized for JAX compatibility
        has_key = state.has_key
        
        # Check if agents are at key positions using vectorized operations
        # Shape: (num_agents, num_keys) - True if agent i is at key j position
        agents_at_keys = jnp.all(
            next_pos[:, None, :] == state.key_pos[None, :, :], axis=-1
        )
        
        # Key assignment logic - JAX-compatible without if/else
        # For single key (num_keys=1): all agents can pick up the key
        # For multiple keys: agent i can only pick up key i (diagonal assignment)
        agent_indices = jnp.arange(self.num_agents)[:, None]  # (num_agents, 1)
        key_indices = jnp.arange(self.num_keys)[None, :]      # (1, num_keys)
        
        # Create diagonal assignment matrix
        diagonal_assignment = (agent_indices == key_indices)
        
        # For single key case, all agents can pick it up (all True for that key)
        single_key_assignment = jnp.ones((self.num_agents, self.num_keys), dtype=bool)
        
        # Use jax.lax.cond or conditional logic based on num_keys
        # If num_keys == 1, use single_key_assignment, else use diagonal_assignment
        can_pick_up = jnp.where(
            self.num_keys == 1,
            single_key_assignment,
            diagonal_assignment
        )
        
        # Agent picks up key if at position and can pick it up
        # Shape: (num_agents, num_keys)
        should_pick_up = jnp.logical_and(agents_at_keys, can_pick_up)
        
        # Update has_key for each agent - any key pickup makes has_key True
        # Shape: (num_agents,)
        new_key_pickups = jnp.any(should_pick_up, axis=1)
        has_key = jnp.logical_or(has_key, new_key_pickups)

        return state.replace(agent_pos=next_pos, has_key=has_key)

    @partial(jax.jit, static_argnums=[0, 2])
    def step_helper(
        self,
        state: State,
        action: chex.Array,
    ) -> State:
        """
        Apply an action to the state based on the helper action.
        Now supports embodied helper agent that can move and push/pull adjacent boxes.
        """

        def freeze_agent_1(state: State) -> State:
            """Set the freeze timer to 2 timesteps when the freeze action is selected."""
            # Always use the same shape - if original is scalar, convert to scalar, if array use array
            new_timer = jnp.asarray(2)
            if hasattr(state.freeze_timer, "shape") and state.freeze_timer.shape:
                new_timer = jnp.asarray([2])  # Make it an array with shape (1,)
            return state.replace(freeze_timer=new_timer)

        def do_nothing(state: State) -> State:
            return state

        def move_helper(state: State, direction: chex.Array) -> State:
            """Move the helper agent in the given direction with collision detection."""
            next_pos = state.helper_pos[0] + direction

            # Bound positions to grid
            next_pos = jnp.clip(
                next_pos, 0, jnp.array([self.height - 1, self.width - 1])
            )

            # Check collision with agents
            agent_collision = jnp.any(
                jnp.all(next_pos[None, :] == state.agent_pos, axis=-1)
            )

            # Check collision with boxes
            box_collision = jnp.any(
                jnp.all(next_pos[None, :] == state.box_pos, axis=-1)
            )

            # Check collision with walls
            wall_collision = jnp.any(
                jnp.all(next_pos[None, :] == state.wall_pos, axis=-1)
            )

            # If any collision, don't move
            collision = jnp.logical_or(
                jnp.logical_or(agent_collision, box_collision), wall_collision
            )
            final_pos = jnp.where(collision, state.helper_pos[0], next_pos)

            return state.replace(helper_pos=final_pos[None, :])

        def push_pull_box(state: State, action: int) -> State:
            """Push or pull a box if helper is adjacent to it."""
            direction = self.helper_action_to_dir[action]

            # Determine if this is a pull action (actions 8-11 if pull is enabled)
            is_pull_action = jnp.logical_and(
                action >= 8, action < 8 + (4 if not self.no_pull_helper_action else 0)
            )

            # For push: look in direction of movement, for pull: look opposite direction
            box_target_pos = jax.lax.cond(
                is_pull_action,
                lambda: state.helper_pos[0] - direction,  # Pull: box is behind helper
                lambda: state.helper_pos[0]
                + direction,  # Push: box is in front of helper
            )
            box_new_pos = box_target_pos + direction

            # Check if there's a box at the target position
            box_indices = jnp.arange(self.num_boxes)
            box_at_target = jnp.all(state.box_pos == box_target_pos[None, :], axis=-1)
            target_box_exists = jnp.any(box_at_target)
            target_box_idx = jnp.where(box_at_target, box_indices, -1).max()

            # If no adjacent box found, return unchanged state
            def no_box_adjacent(state: State) -> State:
                return state

            def move_adjacent_box(state: State) -> State:
                # Clip box new position to grid
                clipped_box_new_pos = jnp.clip(
                    box_new_pos, 0, jnp.array([self.height - 1, self.width - 1])
                )

                # Check collisions for the new box position
                # Check collision with other boxes (excluding the box being moved)
                other_boxes_mask = jnp.arange(self.num_boxes) != target_box_idx
                other_boxes_pos = jnp.where(
                    other_boxes_mask[:, None], state.box_pos, jnp.array([-999, -999])
                )
                box_box_collision = jnp.any(
                    jnp.all(clipped_box_new_pos[None, :] == other_boxes_pos, axis=-1)
                )

                # Check collision with agents
                box_agent_collision = jnp.any(
                    jnp.all(clipped_box_new_pos[None, :] == state.agent_pos, axis=-1)
                )

                # Check collision with helper (only for push, not pull)
                # For pull operations, box is supposed to move to helper's current position
                box_helper_collision = jnp.logical_and(
                    jnp.logical_not(is_pull_action),
                    jnp.all(clipped_box_new_pos == state.helper_pos[0]),
                )

                # Check collision with walls
                box_wall_collision = jnp.any(
                    jnp.all(clipped_box_new_pos[None, :] == state.wall_pos, axis=-1)
                )

                # Check if box would be on goal (only matters if boxes_can_cover_goal is False)
                box_on_goal = jnp.any(
                    jnp.all(clipped_box_new_pos[None, :] == state.goal_pos, axis=-1)
                )

                # Combine all collision checks
                any_collision = jnp.logical_or(
                    jnp.logical_or(
                        jnp.logical_or(box_box_collision, box_agent_collision),
                        jnp.logical_or(box_helper_collision, box_wall_collision),
                    ),
                    jnp.logical_not(self.boxes_can_cover_goal) & box_on_goal,
                )

                # If collision, don't move the box
                final_box_pos = jnp.where(
                    any_collision,
                    state.box_pos.at[target_box_idx].get(),
                    clipped_box_new_pos,
                )

                # Calculate new helper position
                # For both push and pull, helper moves in the action direction
                new_helper_pos = state.helper_pos[0] + direction
                clipped_helper_pos = jnp.clip(
                    new_helper_pos, 0, jnp.array([self.height - 1, self.width - 1])
                )

                # Check if helper would go out of bounds (before clipping)
                helper_out_of_bounds = jnp.logical_or(
                    jnp.logical_or(
                        new_helper_pos[0] < 0, new_helper_pos[0] >= self.height
                    ),
                    jnp.logical_or(
                        new_helper_pos[1] < 0, new_helper_pos[1] >= self.width
                    ),
                )

                # Check helper collision with agents, walls, and other boxes
                helper_agent_collision = jnp.any(
                    jnp.all(clipped_helper_pos[None, :] == state.agent_pos, axis=-1)
                )
                helper_wall_collision = jnp.any(
                    jnp.all(clipped_helper_pos[None, :] == state.wall_pos, axis=-1)
                )
                # For pull actions, don't check collision with the box being pulled
                # For push actions, check collision with all boxes including the moved one
                helper_box_collision = jax.lax.cond(
                    is_pull_action,
                    # Pull: check collision with other boxes (not the one being pulled)
                    lambda: jnp.any(
                        jnp.all(clipped_helper_pos[None, :] == other_boxes_pos, axis=-1)
                    ),
                    # Push: check collision with all boxes including the moved one
                    lambda: jnp.any(
                        jnp.all(clipped_helper_pos[None, :] == final_box_pos, axis=-1)
                    ),
                )

                helper_cannot_move = jnp.logical_or(
                    jnp.logical_or(
                        jnp.logical_or(helper_agent_collision, helper_wall_collision),
                        helper_box_collision,
                    ),
                    helper_out_of_bounds,
                )

                # For pull actions: if helper can't move, box shouldn't move either
                # For push actions: box moves independently, helper moves if it can
                final_box_pos = jax.lax.cond(
                    is_pull_action,
                    # Pull: only move box if helper can move and box has no collision
                    lambda: jnp.where(
                        jnp.logical_or(any_collision, helper_cannot_move),
                        state.box_pos.at[target_box_idx].get(),
                        clipped_box_new_pos,
                    ),
                    # Push: box moves if it has no collision, regardless of helper
                    lambda: final_box_pos,
                )

                # Check if box actually moved after final collision checks
                box_actually_moved = jnp.logical_not(
                    jnp.all(final_box_pos == state.box_pos.at[target_box_idx].get())
                )

                # Only move helper if box moved and helper won't collide
                final_helper_pos = jnp.where(
                    jnp.logical_and(
                        box_actually_moved, jnp.logical_not(helper_cannot_move)
                    ),
                    clipped_helper_pos,
                    state.helper_pos[0],
                )
                new_box_pos = state.box_pos.at[target_box_idx].set(final_box_pos)
                return state.replace(
                    box_pos=new_box_pos, helper_pos=final_helper_pos[None, :]
                )

            return jax.lax.cond(
                target_box_exists, move_adjacent_box, no_box_adjacent, state
            )

        def return_state(state: State) -> State:
            return state

        if not self.no_freeze:
            state = jax.lax.cond(
                action == self.HelperActions.FREEZE_AGENT_1.value,
                lambda state: freeze_agent_1(state),
                return_state,
                state,
            )
        state = jax.lax.cond(
            action == self.HelperActions.DO_NOTHING.value,
            do_nothing,
            return_state,
            state,
        )
        state = jax.lax.cond(
            action < 4,  # NOTE: this assumes that actions 0-3 are the helper movement
            lambda state: move_helper(state, self.helper_action_to_dir[action]),
            return_state,
            state,
        )

        # Check if action is push (4-7) or pull (8-11 if enabled)
        max_push_pull_action = 7 + (4 if not self.no_pull_helper_action else 0)
        is_push_or_pull = jnp.logical_and(action >= 4, action <= max_push_pull_action)

        state = jax.lax.cond(
            is_push_or_pull,
            lambda state: push_pull_box(state, action),
            return_state,
            state,
        )

        return state

    def rollout_random_agent_action_ave(
        self, key: chex.PRNGKey, state: chex.Array
    ) -> chex.Array:
        """
        Rollout random agent actions and update the state.
        Only used in ave.py for calculating the empowerment via sampled rollouts.
        TODO: needs to be updated...
        """
        # First, convert state to State dataclass
        state = state.astype(jnp.int32)

        # Extract agent positions based on number of agents
        agent_positions = []
        for i in range(self.num_agents):
            start_idx = i * 2
            agent_positions.append(state[..., start_idx : start_idx + 2])

        agent_pos = jnp.stack(agent_positions, axis=-2)
        # agent_pos shape: (50, 20, num_agents, 2) where last two dims are [agent_idx, (row, col)]

        # Calculate starting index for boxes (after agents)
        box_start_idx = 2 * self.num_agents

        # Calculate starting index for traps (after agents and boxes)
        trap_start_idx = box_start_idx + 2 * self.num_boxes

        # Calculate starting index for walls (after agents, boxes, and traps)
        wall_start_idx = trap_start_idx + 2 * self.num_traps

        helper_pos_start_idx = wall_start_idx + 2 * self.num_walls

        freeze_timer_start_idx = helper_pos_start_idx + 2

        key_pos_start_idx = freeze_timer_start_idx + 1

        has_key_start_idx = key_pos_start_idx + 2 * self.num_keys

        # Extract box positions
        box_pos = jnp.stack(
            [
                state[..., box_start_idx:trap_start_idx:2],  # box rows
                state[..., box_start_idx + 1 : trap_start_idx : 2],  # box cols
            ],
            axis=-1,
        )
        # box_pos shape: (50, 20, num_boxes, 2) where last two dims are [box_idx, (row, col)]

        # Extract trap positions
        trap_pos = jnp.stack(
            [
                state[..., trap_start_idx::2],  # trap rows
                state[..., trap_start_idx + 1 :: 2],  # trap cols
            ],
            axis=-1,
        )
        # trap_pos shape: (50, 20, num_traps, 2) where last two dims are [trap_idx, (row, col)]

        # Extract wall positions
        wall_pos = jnp.stack(
            [
                state[..., wall_start_idx::2],  # wall rows
                state[..., wall_start_idx + 1 :: 2],  # wall cols
            ],
            axis=-1,
        )
        # wall_pos shape: (50, 20, num_walls, 2) where last two dims are [wall_idx, (row, col)]

        # Extract wall positions
        helper_pos = jnp.stack(
            [
                state[..., helper_pos_start_idx::2],  # wall rows
                state[..., helper_pos_start_idx + 1 :: 2],  # wall cols
            ],
            axis=-1,
        )
        # helper_pos shape: (50, 20, 1, 2) where last two dims are [1, (row, col)]

        # Extract freeze timer
        freeze_timer = state[..., freeze_timer_start_idx:freeze_timer_start_idx+1]
        
        # Extract key positions
        key_pos = state[..., key_pos_start_idx:key_pos_start_idx+2*self.num_keys:2]
        
        # Extract has key
        has_key = state[..., has_key_start_idx:has_key_start_idx+2]


        key, subkey = jax.random.split(key)
        # Generate random actions for all agents
        actions = jax.random.randint(
            subkey,
            shape=state.shape[:-1] + (self.num_agents,),
            minval=0,
            maxval=len(Actions),
        )

        # Vectorize over the batch dimensions while handling the State structure
        def vectorized_step(agent_pos, box_pos, trap_pos, actions):
            temp_state = State(
                agent_pos=agent_pos,
                goal_pos=jnp.array(
                    []
                ),  # Empty as we don't need goals for this calculation
                box_pos=box_pos,
                trap_pos=trap_pos,
                wall_pos=wall_pos,
                helper_pos=helper_pos,
                key_pos=key_pos,
                time=0,
                terminal=False,
                freeze_timer=freeze_timer,
                has_key=has_key
            )
            updated_state = self.step_agents(subkey, temp_state, actions)
            return updated_state.agent_pos

        # Vectorize over both batch dimensions (50, 20) using in_axes
        updated_agent_pos = jax.vmap(
            jax.vmap(
                vectorized_step, in_axes=(0, 0, 0, 0)
            ),  # inner vmap handles the 20 dimension
            in_axes=(0, 0, 0, 0),  # outer vmap handles the 50 dimension
        )(agent_pos, box_pos, trap_pos, actions)

        # Reconstruct the state vector format
        reconstructed_parts = []

        # Add each agent's position
        for i in range(self.num_agents):
            reconstructed_parts.append(updated_agent_pos[..., i, :])

        # Add box and trap positions (unchanged)
        reconstructed_parts.append(state[..., box_start_idx:])

        updated_state = jnp.concatenate(reconstructed_parts, axis=-1)

        return updated_state

    @partial(jax.jit, static_argnums=[0])
    def calculate_agent_rewards_dones(
        self, state: State
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """Calculates the reward based on whether the agents are on the goal and key mechanics"""

        # Modified reward calculation
        def on_position(x, y):
            return jnp.all(x == y)

        # Check which goal each agent is on (if any)
        # Handle any number of agents (1 or 2)
        agent_on_goal = []
        agent_on_trap = []
        trap_consequences = []

        for i in range(self.num_agents):
            # Check goals for this agent
            agent_i_goal = on_position(
                state.agent_pos[i], state.goal_pos[self.agent_to_goal_mapping[i]]
            )
            agent_on_goal.append(agent_i_goal)

            # Check traps for this agent
            agent_i_on_trap = jax.vmap(on_position, in_axes=(None, 0))(
                state.agent_pos[i], state.trap_pos
            )
            agent_on_trap.append(agent_i_on_trap)

            # Calculate trap consequence
            trap_consequences.append(-1 * jnp.float32(jnp.any(agent_i_on_trap)))

        # Cooperative mode reward calculation
        if self.env_mode == EnvMode.COOPERATIVE:
            both_on_goals = jnp.logical_and(
                jnp.any(agent_on_goal[0]), jnp.any(agent_on_goal[1])
            )
            # Both agents must have keys to get goal reward
            both_have_keys = jnp.logical_and(state.has_key[0], state.has_key[1])
            goal_reward = jnp.float32(jnp.logical_and(both_on_goals, both_have_keys))
            done = jnp.logical_and(both_on_goals, both_have_keys)
            
            rewards = jnp.array([
                goal_reward + trap_consequences[0], 
                goal_reward + trap_consequences[1]
            ])
            dones = jnp.array([done, done])
            return rewards, dones

        elif (
            self.env_mode == EnvMode.INDEPENDENT or self.env_mode == EnvMode.COMPETITIVE
        ):
            agent0_on_a_goal = jnp.any(agent_on_goal[0])
            # Agent 0 must have key to get goal reward
            agent0_goal_reward = jnp.float32(jnp.logical_and(agent0_on_a_goal, state.has_key[0]))

            agent1_on_a_goal = (
                jnp.any(agent_on_goal[1]) if self.num_agents > 1 else False
            )
            # Agent 1 must have key to get goal reward
            agent1_goal_reward = (
                jnp.float32(jnp.logical_and(agent1_on_a_goal, state.has_key[1]))
                if self.num_agents > 1
                else jnp.float32(0.0)
            )
            agent1_trap = (
                trap_consequences[1] if self.num_agents > 1 else jnp.float32(0.0)
            )
            
            rewards = jnp.array([
                agent0_goal_reward + trap_consequences[0], 
                agent1_goal_reward + agent1_trap
            ])
            # Agents are done when they reach goal AND have key
            agent0_done = jnp.logical_and(agent0_on_a_goal, state.has_key[0])
            agent1_done = jnp.logical_and(agent1_on_a_goal, state.has_key[1]) if self.num_agents > 1 else False
            dones = jnp.array([agent0_done, agent1_done])
            return rewards, dones

    @partial(jax.jit, static_argnums=[0])
    def get_agent_obs(self, state: State) -> Dict[str, chex.Array]:
        """Convert state into agent observations.
        Not really used...doesn't have trap or wall observations"""
        # 5x5x6 observation: agent 0 position, agent 1 position, goal positions, box positions, key positions, current agent indicator
        obs = jnp.zeros((self.height, self.width, 6))

        # Set agent positions (channels 0 and 1)
        obs = obs.at[state.agent_pos[0, 0], state.agent_pos[0, 1], 0].set(1)
        obs = obs.at[state.agent_pos[1, 0], state.agent_pos[1, 1], 1].set(1)

        # Set goal positions (channel 2)
        obs = obs.at[state.goal_pos[:, 0], state.goal_pos[:, 1], 2].set(1)

        # Set box positions (channel 3)
        obs = obs.at[state.box_pos[:, 0], state.box_pos[:, 1], 3].set(1)

        # Set key positions (channel 4)
        obs = obs.at[state.key_pos[:, 0], state.key_pos[:, 1], 4].set(1)

        # Set current agent indicator (channel 5)
        obs_0 = obs.at[state.agent_pos[0, 0], state.agent_pos[0, 1], 5].set(
            1
        )  # Agent 0's perspective
        obs_1 = obs.at[state.agent_pos[1, 0], state.agent_pos[1, 1], 5].set(
            1
        )  # Agent 1's perspective

        # Flatten the observation and append has_key state
        obs_0_flat = obs_0.reshape(-1)
        obs_1_flat = obs_1.reshape(-1)
        
        # Append has_key information to observations
        obs_0_with_keys = jnp.concatenate([obs_0_flat, state.has_key])
        obs_1_with_keys = jnp.concatenate([obs_1_flat, state.has_key])

        return {"agent_0": obs_0_with_keys, "agent_1": obs_1_with_keys}

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> chex.Array:
        """Convert state into full grid observation for helper.
        Not really used... currently doesn't have trap or wall observations"""
        # 5x5x5 observation: agent 0 position, agent 1 position, goal positions, box positions, key positions
        obs = jnp.zeros((self.height, self.width, 5))

        # Set agent positions (channels 0 and 1)
        obs = obs.at[state.agent_pos[0, 0], state.agent_pos[0, 1], 0].set(1)
        obs = obs.at[state.agent_pos[1, 0], state.agent_pos[1, 1], 1].set(1)

        # Set goal positions (channel 2)
        obs = obs.at[state.goal_pos[:, 0], state.goal_pos[:, 1], 2].set(1)

        # Set box positions (channel 3)
        obs = obs.at[state.box_pos[:, 0], state.box_pos[:, 1], 3].set(1)

        # Set key positions (channel 4)
        obs = obs.at[state.key_pos[:, 0], state.key_pos[:, 1], 4].set(1)

        return obs

    @partial(jax.jit, static_argnums=[0])
    def is_terminal(self, state: State) -> bool:
        """Check if episode is done."""
        return state.time >= self.max_steps

    @property
    def name(self) -> str:
        """Environment name."""
        return "Multiagent Gridworld"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, agent_id: str = "") -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, agent_id: str = "") -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (self.height * self.width * self.num_boxes,))


if __name__ == "__main__":
    env = MultiAgentGridWorld()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    print("These are the obs: ", obs)
    print("This is the state: ", state)

    new_key = jax.random.split(key, 1)
    next_action = {
        "agent_0": Actions.DOWN,
        "agent_1": Actions.DOWN,
    }  # agent 0 shouldn't move
    print("These are the next actions: ", next_action)
    next_helper_action = env.HelperActions.BOX_0_DOWN
    print("This is the next helper action: ", next_helper_action)
    obs, next_state, rewards, dones, _ = env.step_env(
        new_key, state, next_action, next_helper_action
    )
    print("This is the next obs: ", obs)
    print("This is next state: ", next_state)

    new_key = jax.random.split(key, 1)
    next_action = {
        "agent_0": Actions.DOWN,
        "agent_1": Actions.DOWN,
    }  # agent 0 should move
    print("These are the next actions: ", next_action)
    next_helper_action = (
        env.HelperActions.BOX_0_DOWN
    )  # NOTE: this shouldn't move again technically (collides with box 3)
    print("This is the next helper action: ", next_helper_action)
    obs, next_state, rewards, dones, _ = env.step_env(
        new_key, next_state, next_action, next_helper_action
    )
    print("This is the next obs: ", obs)
    print("This is next state: ", next_state)

    new_key = jax.random.split(key, 1)
    next_action = {
        "agent_0": Actions.DOWN,
        "agent_1": Actions.DOWN,
    }  # agent 0 shouldn't move
    print("These are the next actions: ", next_action)
    next_helper_action = env.HelperActions.BOX_2_DOWN
    print("This is the next helper action: ", next_helper_action)
    obs, next_state, rewards, dones, _ = env.step_env(
        new_key, next_state, next_action, next_helper_action
    )
    print("This is the next obs: ", obs)
    print("This is next state: ", next_state)
