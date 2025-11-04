#!/usr/bin/env python3
"""TD(λ) Algorithm."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    performs the TD(λ) algorithm:

    Args:
        env: The environment instance.
        V: a numpy.ndarray of shape (s,) containing the value estimates.
        policy: A a function that takes in a state and returns the
            next action to take.
        lambtha: The eligibility trace factor.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.

    Returns:
        V: the updated value estimate
    """
    for ep in range(episodes):
        state, _ = env.reset()
        E = np.zeros(V.shape)

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, ter, trunc, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            E[state] += 1

            V += alpha * delta * E
            E *= gamma * lambtha

            state = next_state
            if ter or trunc:
                break
    return V
