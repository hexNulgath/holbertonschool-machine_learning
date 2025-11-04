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
    for ep in range(episodes):
        state, _ = env.reset()
        E = np.zeros(Q.shape)

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        for step in range(max_steps):
            next_state, reward, ter, trunc, _ = env.step(action)
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            E[state, action] += 1

            Q += alpha * delta * E
            E *= gamma * lambtha

            state, action = next_state, next_action
            if ter or trunc:
                break
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
