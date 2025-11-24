# Disempower-Grid

Disempower-Grid is a set of test suites containing fully parameterized diverse multi-agent gridworld environments and evaluations to benchmark how a goal-agnostic assistive agent can disempower another agent in the process of empowering its target agent. Environments in Disempower-Grid are flexible: there are multiple flags for changing the action space of the assistant (freeze actions, pushing only), the number of obstacles, goals, sizes of the grids, and even whether the assistant is embodied or non-embodied. For more details, check out the Examples and Details sections.

This code is for the accompanying paper "When Empowerment Disempowers." If you use our code, please cite our paper, found in the Citation section below.

## Setup
The code was run on Ubuntu 22.04 (jammy) with an NVIDIA GeForce RTX 4090 and CUDA version 12.2. To set up the code on your machine, use the following:

1. Clone this repository using `git clone {HTTPS link}`
2. Navigate to repo: `cd multiagent_empowerment`
3. Install the dependencies: `conda env create --file=environment.yml`.
4. Activate the conda environment: `conda activate multiagent_empowerment`

## Example and Details
Once you have set up the environment, you can try running the training script for a specified environment using the following command: 
```
python3 train.py --grid_height 5 --grid_width 4 --num_boxes 4 --num_traps 1 --num_goals 1 --num_walls 2 --max_steps 50 --num_agents 2 --num_keys 1 --env_mode independent --helper_objective ave --helper_objective monte_carlo_emp --helper_objective random --helper_objective no_op --helper_objective discrete_choice --helper_objective entropic_choice --specific_positions_file embodied_moving_boxes_1.json --epochs 250 --no_freeze --no_pull
```

If you'd like to run the environments specified in our paper, you can do so by copying the training command from the `push_pull_embodied.json`, `push_only_embodied.json`, `freeze_nonembodied.json`, `move_nonembodied.json` files.

Each flag corresponds to some feature of the environment, the action space/objective of the assistant agent, and parameters for training the assistant agent. The full list of flags can be found in `utils/argparse_utils.py`. We briefly describe the most relevant below.

## Environments

The environments are split into two different types for implementation ease: one in which the assistant is embodied vs. non-embodied. Each of these types correspond to a separate training script (`train.py` for embodied and `train_nonembodied.py` for the non-embodied).

Beyond that, we describe the different flags in the command:

| Flag | Value | Description |
|------|-------|-------------|
| `--grid_height` | int | Height of the grid environment |
| `--grid_width` | int | Width of the grid environment |
| `--num_boxes` | int (at least 1) | Number of boxes in the environment |
| `--num_traps` | 1 | Number of traps in the environment |
| `--num_goals` | int (1-2) | Number of goals in the environment |
| `--num_walls` | int | Number of walls in the environment |
| `--max_steps` | int (default=50) | Maximum number of steps per episode |
| `--num_agents` | 2 | Number of human agents in the environment |
| `--num_keys` | int (1-2) | Number of keys in the environment |
| `--env_mode` | string (independent, cooperative, competitive) | Environment mode setting |
| `--helper_objective` | ave | Helper objective: Assistance via Empowerment Proxy |
| `--helper_objective` | monte_carlo_emp | Helper objective: Monte Carlo Empowerment |
| `--helper_objective` | random | Helper objective: Random baseline |
| `--helper_objective` | no_op | Helper objective: Noop baseline |
| `--helper_objective` | discrete_choice | Helper objective: Discrete choice |
| `--helper_objective` | entropic_choice | Helper objective: Entropic choice |
| `--specific_positions_file` | string | JSON file specifying entity positions |
| `--epochs` | int (default=250) | Number of training epochs |
| `--no_freeze` | flag | Disable assistant freezing bystander (boolean flag) |
| `--no_pull` | flag | Disable assistant pulling boxes (boolean flag) |

**Note:** The `--helper_objective` flag can appear multiple times with different values, so that the helper is trained on each of these objectives in the specified environment.

## Objectives
There are four goal-agnostic objectives currently implemented for Disempower-Grid:
- Monte Carlo Empowerment (measured in bits)
- Assistance via Empowerment - Du et al. 2020 (measured as variance)
- Discrete Choice - Franzmeyer et al. 2021 (measured as number of states)
- Entropic Choice - Franzmeyer et al. 2021 (measured in bits)

## Training Procedure
There are two phases to training - first, the user and bystander agents are trained on the environment. Then, their policies are frozen, and the helper is trained using the specified objective, with the user and bystander acting according to their frozen policies in the environment.

All the outputs from training go into the `train_data` directory, uniquely identified by the wandb ID of the training run. Each training run in one environment is given a unique wandb ID, over all helper objectives specified. The `train_data` directory includes a CSV with the aggregated average metrics of reward for each agent and each helper objective in the training run. In this CSV, the assistant is denoted as `helper`, user as `agent_0`, and bystander as `agent_1`. The directory also includes the training checkpoints (if saved) for each helper objective and HTML debug visualizations for select epochs during training. 

To replicate the results for specific tasks from the paper, run: `./run_environment_multiple_times.sh <environment_json_file>` to run the environment 5 times. The json files are:
- Push/Pull: `push_pull_embodied.json`
- Push Only: `push_only_embodied.json`
- Move Any: `move_nonembodied.json`
- Freeze: `freeze_nonembodied.json`

### Training over varied goal/key positions
To train over variations of the embodied environment (with keys, goals, and an embodied assistant), use the `run_all_variations.sh` script with a specified environment JSON. This will generate the permutations of the environment and run training over them on the specified helper objectives.

## Evaluation
There are two evals currently implemented. (1) Evaluations across helper objectives in one environment: average percent change in empowerment for the user and bystander throughout the helper's training. (2) Evaluations across multiple environment variations of goal positioning: the average percent change in empowerment, and the proportion of environments in which the user's empowerment increased while the bystander's empowerment decreased throughout assistant training.

To use the evals, move the CSV files from the training runs you are interested in evaluating into the `evals` directory. Then, run either `evals_single_env.py` or `evals_variations.py` depending on the type of evaluation you are interested in conducting.

In the paper, evaluations are also visualized as a line graph, which you can create using your favorite plotting library and the data in the CSVs. 

## BibTeX Citation
```
@article{yang2025empowerment,
  title={When Empowerment Disempowers},
  author={Yang, Claire and Cakmak, Maya and Kleiman-Weiner, Max},
  journal={arXiv preprint arXiv:2511.04177},
  year={2025}
}
```

