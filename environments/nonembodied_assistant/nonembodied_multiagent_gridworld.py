from collections import OrderedDict
from enum import IntEnum

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from typing import Tuple, Dict
import chex
from functools import partial
import pdb
from .helper_actions_utils import create_helper_actions_enum, create_helper_action_directions


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
    time: int  # increments +1 for the agents' step and helper agent's step bundled together
    terminal: bool
    freeze_timer: chex.Array  # counter for how many timesteps agent 1 remains frozen - (1)


def convert_state_to_single_vector(state: State) -> chex.Array:
    batch_size = state.agent_pos.shape[0]
    num_agents = state.agent_pos.shape[-2]
    num_boxes = state.box_pos.shape[-2]
    num_traps = state.trap_pos.shape[-2]
    num_walls = state.wall_pos.shape[-2]
    # Reshape the last two dimensions into a single vector
    agent_pos_flat = jnp.reshape(state.agent_pos, (batch_size, num_agents * 2))
    box_pos_flat = jnp.reshape(state.box_pos, (batch_size, num_boxes * 2))
    trap_pos_flat = jnp.reshape(state.trap_pos, (batch_size, num_traps * 2))
    wall_pos_flat = jnp.reshape(state.wall_pos, (batch_size, num_walls * 2))
    freeze_timer_flat = jnp.reshape(state.freeze_timer, (batch_size, 1))
    # Concatenate along the last dimension
    return jnp.concatenate([agent_pos_flat, box_pos_flat, trap_pos_flat, wall_pos_flat, freeze_timer_flat], axis=-1)


@jax.tree_util.register_pytree_node_class
class NonEmbodiedMultiAgentGridWorld(MultiAgentEnv):
    """heightxwidth gridworld with num_boxes boxes, 1 or 2 agents, and 1 or 2 goals and num_traps traps and num_walls walls"""

    def __init__(
        self,
        height: int = 5,
        width: int = 5,
        max_steps: int = 50,
        num_agents: int = 2,
        random_reset: bool = False,
        debug: bool = False,
        env_mode: EnvMode = EnvMode.INDEPENDENT,
        boxes_can_cover_goal: bool = True,
        num_traps: int = 1,
        num_boxes: int = 4,
        num_goals: int = 2,
        num_walls: int = 1,
        no_freeze: bool = False
    ):
        self.env_mode = env_mode
        self.num_agents = num_agents
        assert num_agents == 1 or num_agents == 2, "num_agents must be 1 or 2"
        self.num_traps = num_traps
        self.num_boxes = num_boxes
        self.num_goals = num_goals
        self.num_walls = num_walls
        self.no_freeze = no_freeze
        self.agent_to_goal_mapping = jnp.arange(num_goals) # NOTE: assumes that number of goals is less than or equal to number of agents
        if len(self.agent_to_goal_mapping) < num_agents:
            self.agent_to_goal_mapping = jnp.concatenate((self.agent_to_goal_mapping, self.agent_to_goal_mapping))
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
        self.random_reset = random_reset
        self.debug = debug
        # In competitive mode, agents cannot share the same goal even though there's only 1 goal
        # In other modes with 1 goal, agents can share it
        self.agents_can_go_to_same_goal = (
            True if self.num_goals == 1 and self.env_mode != EnvMode.COMPETITIVE else False
        )
        self.boxes_can_cover_goal = boxes_can_cover_goal
        self.action_set = jnp.array([x.value for x in Actions])
        if self.num_agents == 2:
            self.agents = ["agent_0", "agent_1"]
        else:
            self.agents = ["agent_0"]
        self.HelperActions = create_helper_actions_enum(num_boxes, no_freeze)
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

        self.helper_action_to_dir = create_helper_action_directions(num_boxes)

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
            self.random_reset,
            self.debug,
            self.env_mode,
            self.boxes_can_cover_goal,
            self.num_traps,
            self.num_boxes,
            self.num_goals,
            self.num_walls,
            self.no_freeze
        )
        return values, aux

    @classmethod
    def tree_unflatten(cls, aux: Tuple, values: Tuple):
        height, width, max_steps, num_agents, random_reset, debug, env_mode, boxes_can_cover_goal, num_traps, num_boxes, num_goals, num_walls, no_freeze = aux
        obj = cls(
            height, width, max_steps, num_agents, random_reset, debug, env_mode, boxes_can_cover_goal, num_traps, num_boxes, num_goals, num_walls, no_freeze
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

        state.agent_pos = og_locations[:self.num_agents]

        if self.debug:
            # sets agents on top of goals for debugging
            state.goal_pos = og_locations[:self.num_agents]
        else:
            state.goal_pos = og_locations[self.num_agents:self.num_agents + self.num_goals]

        state.trap_pos = og_locations[self.num_agents + self.num_goals:self.num_agents + self.num_goals + self.num_traps]

        state.wall_pos = jnp.array([[3,1]])

        state.box_pos = jnp.array([[0, 0], [1, 1], [0, 4], [1, 3]])  # 4 boxes

        self.state = state
        obs = self.get_obs(state)

        return obs, state
    
    @partial(jax.jit, static_argnums=[0])
    def reset_specific_positions(self, key: chex.PRNGKey, agent_pos: chex.Array, goal_pos: chex.Array, box_pos: chex.Array, trap_pos: chex.Array, wall_pos: chex.Array) -> Tuple[chex.Array, State]:
        """Reset environment state with specific positions for certain scenarios to be tested"""
        key1, key2 = jax.random.split(key)

        assert agent_pos.shape == (self.num_agents, 2), f"agent_pos.shape: {agent_pos.shape} != ({self.num_agents}, 2)"
        assert goal_pos.shape == (self.num_goals, 2), f"goal_pos.shape: {goal_pos.shape} != ({self.num_goals}, 2)"
        assert box_pos.shape == (self.num_boxes, 2), f"box_pos.shape: {box_pos.shape} != ({self.num_boxes}, 2)"
        assert trap_pos.shape == (self.num_traps, 2), f"trap_pos.shape: {trap_pos.shape} != ({self.num_traps}, 2)"
        assert wall_pos.shape == (self.num_walls, 2), f"wall_pos.shape: {wall_pos.shape} != ({self.num_walls}, 2)"

        state = State(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            box_pos=box_pos,
            trap_pos=trap_pos,
            wall_pos=wall_pos,
            time=0,
            terminal=False,
            freeze_timer=jnp.array(0),
        )
    
        self.state = state
        obs = self.get_obs(state)

        return obs, state

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Reset environment state with random placement of agents, goals, boxes, traps, and walls."""
        key1, key2 = jax.random.split(key)
        
        # Total number of objects cannot exceed 9/10 of the grid
        assert self.num_agents + self.num_boxes + self.num_traps + self.num_goals + self.num_walls <= (self.width * self.height) * 9/10

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
        agent_pos_corners = jnp.array([
            [0, 0],
            [0, self.width-1],
            [self.height-1, 0],
            [self.height-1, self.width-1]
        ])
        
        # Define corner box positions as a JAX array
        box_pos_corners = jnp.array([
            [[0, 1], [1, 0]],  # corner 0
            [[0, self.width-2], [1, self.width-1]],  # corner 1
            [[self.height-2, 0], [self.height-1, 1]],  # corner 2 
            [[self.height-2, self.width-1], [self.height-1, self.width-2]]  # corner 3
        ])

        # Select a random corner
        n = 4  # For example to pick two different corners
        indices = jax.random.permutation(key1, n)[:4]
        corner, _, corner3, corner4 = indices[0], indices[1], indices[2], indices[3]

        all_other_positions = jnp.array([
        [x, y] for x in range(1, self.height - 1) for y in range(1, self.width - 1)
        ]) # NOTE: the traps and extra boxes and walls are not in any of the corner or corner adjacent positions
        # Use random permutation for other positions
        indices = jax.random.permutation(key1, len(all_other_positions))
        
        agent_pos = jnp.zeros((self.num_agents, 2), dtype=jnp.int32)
        agent_pos = agent_pos.at[0].set(agent_pos_corners[corner])  # First agent
        agent_pos = jnp.where(
            self.num_agents == 2,
            agent_pos.at[1].set(all_other_positions[indices[:1]].reshape(-1)),
            agent_pos
        )
    
        goal_pos = jnp.zeros((self.num_goals, 2), dtype=jnp.int32)
        goal_pos = goal_pos.at[0].set(agent_pos_corners[corner3])
        goal_pos = jnp.where(
            self.num_goals == 2,
            goal_pos.at[1].set(agent_pos_corners[corner4]),
            goal_pos
        )
        trap_pos = all_other_positions[indices[1:self.num_traps+1]]
        wall_pos = all_other_positions[indices[self.num_traps+1:self.num_traps+self.num_walls]]
        # First set the corner box positions (first 2 boxes)
        box_pos_only_corners = box_pos_corners[corner]
        box_pos = jnp.zeros((self.num_boxes, 2), dtype=jnp.int32)
        box_pos = box_pos.at[:2].set(box_pos_only_corners)

        # Then set the remaining boxes (if any)
        if self.num_boxes > 2:  # Only if we need more than 2 boxes
            remaining_box_pos = all_other_positions[indices[self.num_traps+1:self.num_traps+self.num_boxes-1]]
            box_pos = box_pos.at[2:].set(remaining_box_pos)

        # TODO maybe: implement box can cover goal (to start), as well as box can cover trap??

        assert len(box_pos) == self.num_boxes, f"len(box_pos): {len(box_pos)} != self.num_boxes: {self.num_boxes}"

        state = State(
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            box_pos=box_pos,
            trap_pos=trap_pos,
            wall_pos=wall_pos,
            time=0,
            terminal=False,
            freeze_timer=0,
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

        next_state = next_state.replace(freeze_timer=jnp.maximum(next_state.freeze_timer - 1, 0))
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
                jnp.logical_or(helper_action == self.HelperActions.FREEZE_AGENT_1.value, jnp.any(state.freeze_timer > 0)),
                lambda actions: actions.at[1].set(Actions.STAY.value),
                lambda actions: actions,
                actions
            )

        next_pos = state.agent_pos + self.action_to_dir[actions]

        # Bound positions to grid
        next_pos = jnp.clip(next_pos, 0, jnp.array([self.height - 1, self.width - 1]))

        # Check if positions would collide with any boxes
        box_collisions = jnp.any(
            jnp.all(next_pos[:, None, :] == state.box_pos[None, :, :], axis=-1), axis=-1
        )

        # Prevent updates for agents colliding with boxes (independently for each agent)
        next_pos = jnp.where(box_collisions[:, None], state.agent_pos, next_pos)

        # Check if positions would collide with any walls
        wall_collisions = jnp.any(
            jnp.all(next_pos[:, None, :] == state.wall_pos[None, :, :], axis=-1), axis=-1
        )
        # Prevent updates for agents colliding with walls (independently for each agent)
        next_pos = jnp.where(wall_collisions[:, None], state.agent_pos, next_pos)

        if self.num_agents == 2:

            agents_switching = jnp.logical_and(
                jnp.all(next_pos[0] == state.agent_pos[1]),  # Agent 0 moving to Agent 1's position
                jnp.all(next_pos[1] == state.agent_pos[0])   # Agent 1 moving to Agent 0's position
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
                both_on_goals = jnp.logical_and(jnp.any(agent0_goal), jnp.any(agent1_goal))
                on_same_goal = jnp.any(jnp.logical_and(agent0_goal, agent1_goal))
                both_on_goals_on_same_goal = jnp.logical_and(both_on_goals, on_same_goal)
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
            jnp.all(state.agent_pos[:, None, :] == state.trap_pos[None, :, :], axis=-1), axis=-1
        )
        # If any agent is in a trap, then block all movement
        next_pos = jnp.where(trap_collisions[:, None], state.agent_pos, next_pos)

        return state.replace(agent_pos=next_pos)

    @partial(jax.jit, static_argnums=[0, 2])
    def step_helper(
        self,
        state: State,
        action: chex.Array,
    ) -> State:
        """
        Apply an action to the state based on the helper action.
        """

        def freeze_agent_1(state: State) -> State:
            """Set the freeze timer to 5 timesteps when the freeze action is selected."""
            # Always use the same shape - if original is scalar, convert to scalar, if array use array
            new_timer = jnp.asarray(4)
            if hasattr(state.freeze_timer, 'shape') and len(state.freeze_timer.shape) > 0:
                new_timer = jnp.asarray([4])  # Make it an array with shape (1,)
            return state.replace(freeze_timer=new_timer)

        def do_nothing(state: State) -> State:
            return state

        def move_box(state: State, box_idx: chex.Array, action: int) -> State:
            direction = self.helper_action_to_dir[action]
            previous_box_pos_at_idx = state.box_pos.at[box_idx].get()
            updated_box_pos_at_idx = previous_box_pos_at_idx + direction

            # Clip positions to grid
            updated_box_pos_at_idx = jnp.clip(
                updated_box_pos_at_idx, 0, jnp.array([self.height - 1, self.width - 1])
            )

            # check for collision between boxes and boxes
            box_box_collisions = jnp.any(
                jnp.all(updated_box_pos_at_idx[None, :] == state.box_pos, axis=-1)
            )

            # check for collisions between boxes and agents
            box_agent_collisions = jnp.any(
                jnp.all(updated_box_pos_at_idx[None, :] == state.agent_pos, axis=-1)
            )

            # check for "collision" between boxes and goals
            box_on_goal = jnp.any(
                jnp.all(updated_box_pos_at_idx[None, :] == state.goal_pos, axis=-1)
            )

            # check for "collision" between boxes and walls
            box_on_wall = jnp.any(
                jnp.all(updated_box_pos_at_idx[None, :] == state.wall_pos, axis=-1)
            )

            # Combine collision checks
            box_collides_with_other_box_or_agent_or_goal = jnp.logical_or(
                jnp.logical_or(box_box_collisions, box_agent_collisions),
                jnp.logical_not(self.boxes_can_cover_goal) & box_on_goal,
            )

            box_collides_with_other_box_or_agent_or_goal_or_wall = jnp.logical_or(
                box_collides_with_other_box_or_agent_or_goal,
                box_on_wall,
            )

            def update_box_pos(state: State) -> State:
                new_box_pos = state.box_pos
                new_box_pos = new_box_pos.at[box_idx].set(updated_box_pos_at_idx)
                state = state.replace(box_pos=new_box_pos)
                return state

            def do_not_update_box_pos(state: State) -> State:
                return state

            final_state = jax.lax.cond(
                box_collides_with_other_box_or_agent_or_goal_or_wall,
                do_not_update_box_pos,
                update_box_pos,
                state,
            )

            return final_state

        def return_state(state: State) -> State:
            return state

        box_idx = jnp.clip(action // 4, 0, None)

        if not self.no_freeze:
            state = jax.lax.cond(
                action == self.HelperActions.FREEZE_AGENT_1.value,
                lambda state: freeze_agent_1(state),
                return_state,
                state,
            )
        state = jax.lax.cond(
            action == self.HelperActions.DO_NOTHING.value, do_nothing, return_state, state
        )
        state = jax.lax.cond(
            action < self.num_boxes * 4,  # NOTE: this assumes there are only four possible actions per box
            lambda state: move_box(state, box_idx, action),
            return_state,
            state,
        )
        return state


    @partial(jax.jit, static_argnums=[0])
    def calculate_agent_rewards_dones(
        self, state: State
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """Calculates the reward based on whether the agents are on the goal"""

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
            both_on_goals = jnp.logical_and(jnp.any(agent_on_goal[0]), jnp.any(agent_on_goal[1]))
            goal_reward = jnp.float32(both_on_goals)
            done = both_on_goals

            rewards = jnp.array([
                goal_reward + trap_consequences[0],
                goal_reward + trap_consequences[1]
            ])
            dones = jnp.array([
                done,
                done
            ])
            return rewards, dones

        elif self.env_mode == EnvMode.INDEPENDENT or self.env_mode == EnvMode.COMPETITIVE:
            agent0_on_a_goal = jnp.any(agent_on_goal[0])
            agent0_goal_reward = jnp.float32(agent0_on_a_goal)

            agent1_on_a_goal = jnp.any(agent_on_goal[1]) if self.num_agents > 1 else False
            agent1_goal_reward = (
                jnp.float32(agent1_on_a_goal)
                if self.num_agents > 1
                else jnp.float32(0.0)
            )
            agent1_trap = trap_consequences[1] if self.num_agents > 1 else jnp.float32(0.0)

            rewards = jnp.array([
                agent0_goal_reward + trap_consequences[0],
                agent1_goal_reward + agent1_trap
            ])
            # Agents are done when they reach goal
            agent0_done = agent0_on_a_goal
            agent1_done = agent1_on_a_goal if self.num_agents > 1 else False
            dones = jnp.array([
                agent0_done,
                agent1_done
            ])
            return rewards, dones

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> chex.Array:
        """Convert state into full grid observation for helper.
        Not really used... currently doesn't have trap or wall or key observations"""
        # 5x5x4 observation: agent 0 position, agent 1 position, goal positions, box positions
        obs = jnp.zeros((self.height, self.width, 5))

        # Set agent positions (channels 0 and 1)
        obs = obs.at[state.agent_pos[0, 0], state.agent_pos[0, 1], 0].set(1)
        obs = obs.at[state.agent_pos[1, 0], state.agent_pos[1, 1], 1].set(1)

        # Set goal positions (channel 2)
        obs = obs.at[state.goal_pos[:, 0], state.goal_pos[:, 1], 2].set(1)

        # Set box positions (channel 3)
        obs = obs.at[state.box_pos[:, 0], state.box_pos[:, 1], 3].set(1)

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
    env = NonEmbodiedMultiAgentGridWorld()
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
