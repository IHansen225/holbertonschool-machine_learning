#!/usr/bin/env python3

import numpy as np


def q_init(env):
    """
        Initializes the Q-table
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
