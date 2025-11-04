#!/usr/bin/env python3
"""SARSA(λ) Algorithm."""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
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
    initial_epsilon = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        E = np.zeros(Q.shape)

        # Choose initial action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        for step in range(max_steps):
            # Take action, observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Choose next action
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            # Calculate TD error
            delta = reward + (
                gamma * Q[next_state, next_action]) - Q[state, action]

            # Update eligibility trace
            E[state, action] += 1

            # Update Q-values and eligibility traces
            Q += alpha * delta * E
            E *= gamma * lambtha

            # Move to next state and action
            state, action = next_state, next_action

            if terminated or truncated:
                break

        # Decay epsilon (multiplicative decay)
        epsilon = max(
            min_epsilon, initial_epsilon * np.exp(-epsilon_decay * ep))

    return Q
