import jax.numpy as jnp
import chex
from jax import lax
import plotly.express as px
import pandas as pd
from .multiagent_gridworld import (
    Actions,
    MultiAgentGridWorld,
    State
)
from environments.embodied_assistant.helper_actions_utils import create_helper_actions_enum


def render(state: State, env: MultiAgentGridWorld, cell_size: int = 100) -> chex.Array:
    """
    Creates image array to be saved in WandB, scaled up by cell_sizepixels
    """
    goal_color = jnp.array([255, 255, 10]).astype(jnp.uint8)  # yellow
    agent_colors = jnp.array(
        [
            jnp.array([255, 0, 0]).astype(jnp.uint8),  # red
            jnp.array([255, 0, 255]).astype(jnp.uint8),  # pink
        ]
    )
    box_color = jnp.array([100, 0, 215]).astype(jnp.uint8)  # purple
    trap_color = jnp.array([100, 100, 100]).astype(jnp.uint8)  # grey
    wall_color = jnp.array([46, 46, 46]).astype(jnp.uint8)  # dark grey
    helper_color = jnp.array([0, 215, 39])  # green
    key_color = jnp.array([255, 255, 0]).astype(jnp.uint8)  # yellow

    width_scaled = env.width * cell_size
    height_scaled = env.height * cell_size

    # Initialize larger array with background color
    arr = jnp.zeros((3, height_scaled, width_scaled), dtype=jnp.uint8) + 50

    # Helper function to fill a cell with color using dynamic_update_slice
    def fill_cell(arr, row, col, color):
        y_start = row * cell_size
        x_start = col * cell_size
        color_block = jnp.broadcast_to(color[:, None, None], (3, cell_size, cell_size))
        return lax.dynamic_update_slice(arr, color_block, (0, y_start, x_start))

    # Draw boxes first
    box_pos = state.box_pos
    for box in box_pos:
        arr = fill_cell(arr, box[0], box[1], box_color)

    # Draw goals over boxes
    goals_pos = state.goal_pos
    for i in range(0, len(goals_pos)):
        goal_row, goal_col = goals_pos[i]
        arr = fill_cell(arr, goal_row, goal_col, goal_color)

    keys_pos = state.key_pos
    for i in range(0, len(keys_pos)):
        key_row, key_col = keys_pos[i]
        arr = fill_cell(arr, key_row, key_col, key_color)

    # Draw agents
    agent_pos_list = state.agent_pos
    for i in range(0, len(agent_pos_list)):
        agent_row, agent_col = agent_pos_list[i]
        arr = fill_cell(arr, agent_row, agent_col, agent_colors[i])

    # Draw helper agent
    helper_pos = state.helper_pos
    arr = fill_cell(arr, helper_pos[0], helper_pos[1], helper_color)

    # Draw traps
    trap_pos = state.trap_pos
    for i in range(0, len(trap_pos)):
        trap_row, trap_col = trap_pos[i]
        arr = fill_cell(arr, trap_row, trap_col, trap_color)

    # Draw walls last
    wall_pos = state.wall_pos
    for i in range(0, len(wall_pos)):
        wall_row, wall_col = wall_pos[i]
        arr = fill_cell(arr, wall_row, wall_col, wall_color)

    return arr


def create_frame_data(state: State, env: MultiAgentGridWorld):
    return {
        "agents": state.agent_pos,
        "goals": state.goal_pos,
        "boxes": state.box_pos,
        "traps": state.trap_pos,
        "walls": state.wall_pos,
        "helper": state.helper_pos,
        "keys": state.key_pos,
        "has_key": state.has_key,
        "width": env.width,
        "height": env.height,
    }


def create_trajectory_viewer_px(
    name,
    frame_data,
    helper_actions,
    agent_actions_dict,
    no_freeze,
    no_pull_helper_action,
    agent_rewards=None,  # Add this parameter: shape (num_timesteps, num_agents)
    size=800,
):
    frames_list = []
    agent_actions = [agent_actions_dict["agent_0"][0]]
    if "agent_1" in agent_actions_dict:
        agent_actions.append(agent_actions_dict["agent_1"][0])

    HelperActionsEnum = create_helper_actions_enum(no_freeze, no_pull_helper_action)

    grid_width = frame_data["width"].astype(int)
    grid_width = grid_width[0].item()
    grid_height = frame_data["height"].astype(int)
    grid_height = grid_height[0].item()

    for frame_idx in range(len(frame_data["agents"])):

        for agent_idx in range(len(frame_data["agents"][frame_idx])):
            frames_list.append(
                {
                    "frame": frame_idx,
                    "x": frame_data["agents"][frame_idx][agent_idx, 1] + 0.5,
                    "y": frame_data["agents"][frame_idx][agent_idx, 0] + 0.5,
                    "text": f"A{agent_idx} Action: {Actions(agent_actions[agent_idx][frame_idx - 1]).name if frame_idx > 0 else ''}",
                    "type": "Agent",
                    "symbol": "diamond" if agent_idx == 0 else "circle",
                    "name": f"A{agent_idx}",
                    "color": "blue",
                }
            )

        # Add reward text annotations if provided
        if agent_rewards is not None:
            for agent_idx in range(agent_rewards.shape[1]):  # num_agents
                cumulative_reward = float(jnp.sum(agent_rewards[:frame_idx+1, agent_idx]))
                frames_list.append({
                    "frame": frame_idx,
                    "x": 1,  # Position in top-left corner
                    "y": 1 + agent_idx * 0.4,  # Offset for each agent
                    "type": f"Agent{agent_idx}_Reward",
                    "symbol": "circle",
                    "name": f"Agent{agent_idx}_Reward",
                    "text": f"A{agent_idx} Reward: {cumulative_reward:.1f}",
                    "color": "white",
                    "size": 0.1,  # Small invisible marker
                })

        for goal_idx in range(len(frame_data["goals"][frame_idx])):
            frames_list.append(
                {
                    "frame": frame_idx,
                    "x": frame_data["goals"][frame_idx][goal_idx, 1] + 0.5,
                    "y": frame_data["goals"][frame_idx][goal_idx, 0] + 0.5,
                    "type": "Goal",
                    "symbol": "star",
                    "name": f"Goal{goal_idx}",
                    "text": f"Goal{goal_idx}",
                    "color": "gold",
                }
            )

        for key_idx in range(len(frame_data["keys"][frame_idx])):
            frames_list.append(
                {
                    "frame": frame_idx,
                    "x": frame_data["keys"][frame_idx][key_idx, 1] + 0.5,
                    "y": frame_data["keys"][frame_idx][key_idx, 0] + 0.5,
                    "type": "Key",
                    "symbol": "arrow-wide",
                    "name": f"Key{key_idx}",
                    "text": f"Key{key_idx}",
                    "color": "yellow",
                }
            )

        for box_idx in range(len(frame_data["boxes"][frame_idx])):
            frames_list.append(
                {
                    "frame": frame_idx,
                    "x": frame_data["boxes"][frame_idx][box_idx, 1] + 0.5,
                    "y": frame_data["boxes"][frame_idx][box_idx, 0] + 0.5,
                    "type": "Box",
                    "symbol": "square",
                    "name": f"Box{box_idx}",
                    "text": f"Box{box_idx}",
                    "color": "brown",
                }
            )

        frames_list.append(
            {
                "frame": frame_idx,
                "x": frame_data["helper"][frame_idx][0][1] + 0.5,
                "y": frame_data["helper"][frame_idx][0][0] + 0.5,
                "type": "Helper",
                "symbol": "circle",
                "name": "Helper",
                "text": "Helper",
                "color": "green",
            }
        )

        for trap_idx in range(len(frame_data["traps"][frame_idx])):
            frames_list.append(
                {
                    "frame": frame_idx,
                    "x": frame_data["traps"][frame_idx][trap_idx, 1] + 0.5,
                    "y": frame_data["traps"][frame_idx][trap_idx, 0] + 0.5,
                    "type": "Trap",
                    "symbol": "cross",
                    "name": f"Trap{trap_idx}",
                    "text": f"Trap{trap_idx}",
                    "color": "red",
                }
            )

        for wall_idx in range(len(frame_data["walls"][frame_idx])):
            frames_list.append(
                {
                    "frame": frame_idx,
                    "x": frame_data["walls"][frame_idx][wall_idx, 1] + 0.5,
                    "y": frame_data["walls"][frame_idx][wall_idx, 0] + 0.5,
                    "type": "Wall",
                    "symbol": "square",
                    "name": f"Wall{wall_idx}",
                    "text": f"Wall{wall_idx}",
                    "color": "black",
                }
            )

        frames_list.append(
            {
                "frame": frame_idx,
                "x": 0.5,
                "y": grid_height + 0.1,
                "type": "Helper Action",
                "symbol": "circle",
                "name": "Helper Action",
                "text": f"Helper Action: {HelperActionsEnum(helper_actions[frame_idx-1]).name if frame_idx > 0 else ''}",
                "color": "white",
            }
        )

    df = pd.DataFrame(frames_list)

    # Update color map to include reward types
    color_map = {
        "Agent": "blue",
        "Goal": "gold",
        "Box": "brown",
        "Trap": "red",
        "Wall": "black",
        "Helper Action": "white",
        "Helper": "green",
        "Key": "yellow",
    }
    
    # Add reward colors
    if agent_rewards is not None:
        for agent_idx in range(agent_rewards.shape[1]):
            color_map[f"Agent{agent_idx}_Reward"] = "white"

    fig = px.scatter(
        df,
        x="x",
        y="y",
        animation_frame="frame",
        color="type",
        symbol="name",
        text="text",
        title=f"{name} GridWorld Trajectory",
        color_discrete_map=color_map,
        symbol_map={
            "A0": "diamond",
            "A1": "circle" if len(frame_data["agents"][frame_idx]) > 1 else "",
            **{f"Goal{i}": "star" for i in range(len(frame_data["goals"][frame_idx]))},
            **{f"Box{i}": "square" for i in range(len(frame_data["boxes"][frame_idx]))},
            **{f"Trap{i}": "cross" for i in range(len(frame_data["traps"][frame_idx]))},
            **{
                f"Wall{i}": "square" for i in range(len(frame_data["walls"][frame_idx]))
            },
            **{"Helper": "circle"},
            # Add reward symbols
            **{f"Agent{i}_Reward": "circle" for i in range(agent_rewards.shape[1]) if agent_rewards is not None},
            **{f"Key{i}": "arrow-wide" for i in range(len(frame_data["keys"][frame_idx]))},
        },
        width=size,
        height=size,
        range_x=[0, grid_width],
        range_y=[grid_height, 0],
    )

    fig.update_xaxes(
        showgrid=True,
        tickmode="array",
        ticktext=[str(i) for i in range(grid_width)],
        tickvals=list(range(grid_width)),
    )

    fig.update_yaxes(
        showgrid=True,
        tickmode="array",
        ticktext=[str(i) for i in range(grid_height)],
        tickvals=list(range(grid_height)),
    )

    fig.update_traces(marker=dict(size=30), selector=dict(type="scatter"))

    fig.update_traces(textposition="top center")

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 2000

    return fig
