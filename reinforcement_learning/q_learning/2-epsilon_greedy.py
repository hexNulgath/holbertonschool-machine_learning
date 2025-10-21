#!/usr/bin/env python3
"""
epsilon_greedy.py
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    uses epsilon-greedy to determine the next action

    Parameters:
    Q : numpy.ndarray
        A 2D array containing the Q-table
    state : int
        The current state
    epsilon : float
        The epsilon to use for the calculation
    Returns: int
        The next action index
    """
    p = np.random.uniform()
    if p < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state, :])
