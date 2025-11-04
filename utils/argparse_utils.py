import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description="Multi-Agent Gridworld for Training/Testing Empowerment"
    )

    # For the gridworld env
    parser.add_argument(
        "--env_mode",
        type=str,
        default="independent",
        help="independent or cooperative or competitive",
    )
    parser.add_argument(
        "--boxes_can_cover_goal",
        action="store_true",
        help="Include this arg to make it so that boxes can cover the goal in the env",
    )
    parser.add_argument(
        "--num_traps", type=int, default=1, help="Number of traps in the env"
    )
    parser.add_argument("--grid_height", type=int, default=5, help="Height of grid")
    parser.add_argument("--grid_width", type=int, default=5, help="Width of grid")
    parser.add_argument(
        "--num_agents", type=int, default=2, help="Number of agents in scene"
    )  # can only be one or two
    parser.add_argument(
        "--num_boxes", type=int, default=4, help="Number of boxes in scene"
    )
    parser.add_argument(
        "--num_goals", type=int, default=2, help="Number of goals in scene"
    )  # can only be one or two
    parser.add_argument(
        "--num_walls", type=int, default=1, help="Number of wall blocks in scene"
    )
    parser.add_argument(
        "--num_keys", type=int, default=1, help="Number of keys in scene"
    )

    # for the training run
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for random number generator"
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=100,
        help="Number of environments/sampled trajectories to run in parallel",
    )
    parser.add_argument(
        "--specific_positions_file",
        type=str,
        default=None,
        help="File containing specific positions for the env",
    )
    parser.add_argument("--name", default=None, help="Name of the run")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum number of steps in an episode",
    )

    parser.add_argument(
        "--helper_policy",
        type=str,
        default="random",
        help="random, ave, esr, no_op, freeze_agent_1, or maximize_emp or maximize_joint_emp for running train_grid_parallel.py",
    )

    parser.add_argument(
        "--helper_objective",
        action="append",
        default=None,
        help="monte_carlo_emp, ave, random, no_op, discrete_choice, entropic_choice, joint_empowermen, minimax_regret_empowerment for running train.py or train_pipeline.py. ",
    )

    parser.add_argument(
        "--no_freeze",
        action="store_true",
        help="The helper does not have the ability to freeze agent 1",
    )

    parser.add_argument(
        "--no_pull",
        action="store_true",
        help="The helper does not have the ability to pull the boxes (push only, like sokoban)",
    )

    parser.add_argument(
        "--test_mode", action="store_true", help="Run in test mode, no training at all - for use in train_grid_parallel.py"
    )
    parser.add_argument(
        "--train_mode", action="store_true", help="Run in train mode, no testing at all - for use in train_grid_parallel.py"
    )

    parser.add_argument("--wandbid", type=str, default=None, help="Wandb ID for test.py")
    parser.add_argument("--date", type=str, default=None, help="Date for test.py")

    parser.add_argument("--num_envs_to_generate", type=int, default=10, help="Number of environments to generate for train_pipeline.py")

    # DEPRECATED MOSTLY
    # for the empowerment/AvE baselines
    parser.add_argument(
        "--policy_lr", default=1e-3, type=float, help="Policy learning rate"
    )
    parser.add_argument(
        "--repr_lr", default=1e-4, type=float, help="Representation learning rate"
    )
    parser.add_argument("--dual_lr", type=float, default=0.1, help="Dual learning rate")

    parser.add_argument(
        "--repr_dim", default=32, type=int, help="Representation dimension"
    )
    parser.add_argument("--hidden_dim", default=100, type=int, help="Hidden dimension")
    parser.add_argument("--buffer_size", default=15_000, type=int, help="Buffer size")
    parser.add_argument(
        "--repr_buffer_size", default=15_000, type=int, help="Contrastive buffer size"
    )

    parser.add_argument(
        "--tau", default=0.005, type=float, help="Target network update rate for policy"
    )
    parser.add_argument(
        "--gamma", default=0.9, type=float, help="Discount factor for buffer"
    )
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")

    parser.add_argument(
        "--reward",
        type=str,
        default="dot",
        choices=["dot", "norm", "diff"],
        help="Reward function",
    )
    parser.add_argument(
        "--precision",
        default=1.0,
        type=float,
        help="Initial boltzmann constant for policy",
    )
    parser.add_argument(
        "--noise", default=0.2, type=float, help="Noise in human action selection"
    )

    parser.add_argument(
        "--update_repr_freq",
        default=100,
        type=int,
        help="Update frequency for representation",
    )
    parser.add_argument(
        "--update_policy_freq",
        default=100,
        type=int,
        help="Update frequency for policy",
    )
    parser.add_argument(
        "--update_dual_freq", type=int, default=100, help="Update frequency for dual"
    )

    parser.add_argument(
        "--target_entropy", type=float, default=0.6, help="Target entropy for dual"
    )
    parser.add_argument(
        "--cache_reward", action="store_true", help="Don't recompute reward"
    )

    parser.add_argument("--phi_norm", action="store_true", help="Normalize phi")
    parser.add_argument("--psi_norm", action="store_true", help="Normalize psi")
    parser.add_argument(
        "--psi_reg", default=0.0, type=float, help="Regularization on psi"
    )
    parser.add_argument(
        "--sample_from_target", action="store_true", help="Sample from target policy"
    )

    parser.add_argument(
        "--emp_rollout_len",
        type=int,
        default=5,
        help="Length of empowerment rollout for AVE baseline",
    )
    parser.add_argument(
        "--emp_num_rollouts",
        type=int,
        default=20,
        help="Number of empowerment rollouts for AVE baseline",
    )
    parser.add_argument(
        "--smart_features",
        action="store_true",
        help="Use smart features for AVE baseline",
    )
    parser.add_argument(
        "--render_freq", type=int, default=1000, help="Frequency of rendering"
    )
    return parser
