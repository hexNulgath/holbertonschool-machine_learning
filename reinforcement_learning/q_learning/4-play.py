#!/usr/bin/env python3
"""
Plays an episode using the trained Q-table
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode using the Q-table

    Args:
        env: is the OpenAI gym environment instance
        Q: is a numpy.ndarray of shape (s,a) containing the Q-table
            s: number of states in the environment
            a: number of actions in the environment
        max_steps: is the maximum number of steps in the episode

    Returns:
        The total reward for the episode
    """
    state = env.reset()[0]
    terminated = False
    truncated = False
    total_reward = 0
    actions = ["Left", "Down", "Right", "Up"]
    states = []

    for _ in range(max_steps):
        if terminated or truncated:
            break

        action = np.argmax(Q[state, :])
        states.append(env.render())
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
    states.append(env.render())

    return total_reward, states
