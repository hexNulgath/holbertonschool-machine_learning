#!/usr/bin/env python3
"""SARSA(λ) Algorithm."""
import numpy as np
import gymnasium as gym


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    performs SARSA(λ):

    Args:
        env: the environment instance
        Q: a numpy.ndarray of shape (s,a) containing the Q table
        lambtha: the eligibility trace factor
        episodes: the total number of episodes to train over
        max_steps: the maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate
        epsilon: the initial threshold for epsilon greedy
        min_epsilon: the minimum value that epsilon should decay to
        epsilon_decay: the decay rate for updating epsilon between episodes

    Returns: 
        Q: the updated Q table
    """