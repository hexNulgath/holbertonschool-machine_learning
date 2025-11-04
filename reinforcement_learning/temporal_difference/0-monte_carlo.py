#!/usr/bin/env python3
"""Monte Carlo."""
import numpy as np
import gymnasium as gym


def instance(env, policy, max_steps=100):
    """
    Generates an episode following the given policy.

    Args:
        env: The environment instance.
        policy: A function that takes in a state and returns the
            next action to take.
        max_steps: The maximum number of steps per episode.

    Returns:
        episode: A list of (state, reward, action) tuples.
        T: The last timestep index.
    """
    state, _ = env.reset()
    episode = []

    for t in range(max_steps):
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, reward))

        if terminated or truncated:
            break
        state = next_state

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
        visited_states = set()

        # Work backwards through the episode (first-visit)
        for state, reward in reversed(episode):
            G = gamma * G + reward
            if state not in visited_states:
                visited_states.add(state)
                V[state] += alpha * (G - V[state])
    return V
