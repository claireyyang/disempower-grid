# Environments Details

The non-embodied assistant and embodied assistant are stored as two separate environments in the `/environments` directory. In all environments, there are assistant, user, and bystander agents. The assistive agent has an objective to increase the influence/choice of user (agent 0). 

## Embodied Assistant Environment
 The user and bystander (agent 1) must each go to the position of the key before entering the goal in order to get any reward. The assistant is embodied in this case.

All agents move concurrently in the same step. The user and bystander action space is `{left, right, up, down, stay}`. They cannot move through boxes or walls. They also collide with each other, unlesss they are both about to enter the goal.

The assistant's action space consists of the following actions (when both push and pull are available to it):
`{push box left, push box up, push box right, push box down, pull box left, pull box up, pull box right, pull box down, left, right, up, down, stay}`

## Non-Embodied Assistant Environment
The user and bystander (agent 1) get reward from entering the goal. The assistant is non-embodied.

All agents move concurrently in the same step. The user and bystander action space is `{left, right, up, down, stay}`. They cannot move through boxes or walls. They also collide with each other, unlesss they are both about to enter the goal.

The assistant's action space consists of the following actions (when freeze is available to it). Assume there are n>1 boxes:
`{move_box_1_left, move_box_1_right, move_box_1_up, move_box_1_down, ..., move_box_n_left, move_box_n_right, move_box_n_up, move_box_n_down, left, right, up, down, stay, freeze_bystander}`

## Modes
There are three modes implemented.
- Independent mode: this is the main mode (general-sum payoffs). Each agent gets their own individual reward (1) if they make it into the goal (they can go to the same goal). 
- Cooperative mode: (joint payoffs). Both agents must be in a goal at the same time (and they must be at different goals).
- Competitive mode: (zero-sum payoffs). There is only one goal. Once an agent enters the goal, the other agent cannot enter unless the other agent leaves.

In all modes, the game only terminates when the `num_steps` steps have passed.