from enum import Enum


class HelperObjective(Enum):
    MONTE_CARLO_EMPOWERMENT = 1
    AVE_PROXY = 2
    RANDOM = 3
    NO_OP = 4
    DISCRETE_CHOICE = 5
    ENTROPIC_CHOICE = 6
    IMMEDIATE_CHOICE = 7 # not implemented
    JOINT_EMPOWERMENT = 8
    MINIMAX_REGRET_EMPOWERMENT = 9 # implementation is a bit sus...

    def __str__(self):
        return self.name.lower()
