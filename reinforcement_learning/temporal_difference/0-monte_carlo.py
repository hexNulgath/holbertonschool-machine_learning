#!/usr/bin/env python3
"""Monte Carlo."""
import numpy as np


def instance(env, policy, max_steps=100):
    """Generate an episode following the given policy."""
    state, _ = env.reset()
    episode = []

    for _ in range(max_steps):
        action = policy(state)
        next_state, reward, ter, trunc, _ = env.step(action)
        episode.append((state, reward))
        state = next_state
        if ter or trunc:
            break
    return episode


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Monte Carlo algorithm to estimate state-value
    function V under a given policy.

    Args:
        env: The environment instance.
        V: a numpy.ndarray of shape (s,) containing the value estimates.
        policy: A a function that takes in a state and returns the
        next action to take.
        episodes: the total number of episodes to train over.
        max_steps: the maximum number of steps per episode.
        alpha: the learning rate.
        gamma: the discount factor.

    Returns:
        V: the updated value function.
    """
    for ep in range(episodes):
        episode = instance(env, policy, max_steps)
        G = 0
        episode = np.array(episode, dtype=int)

        # Work backwards through the episode (first-visit)
        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = reward + gamma * G
            if state not in episode[:ep, 0]:
                V[state] += alpha * (G - V[state])
    return V
