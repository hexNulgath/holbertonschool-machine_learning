#!/usr/bin/env python3
"""Initialize the Q-learning exercises package."""
import numpy as np


def q_init(env):
    """Initialize the Q-learning environment.

    Args:
        env: is the FrozenLakeEnv instance

    Returns:
        Q-table as a numpy.ndarray of zeros
    """
    states = env.observation_space.n
    actions = env.action_space.n
    return np.zeros((states, actions))
