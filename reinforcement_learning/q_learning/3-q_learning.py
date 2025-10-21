#!/usr/bin/env python3
"""
"""
import numpy as np
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers import TransformReward
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    performs Q-learning

    Args:
        env: is the OpenAI gym environment instance
        Q: is a numpy.ndarray of shape (s,a) containing the Q-table
            s: number of states in the environment
            a: number of actions in the environment
        episodes: is the total number of episodes to train over
        max_steps: is the maximum number of steps per episode
        alpha: is the learning rate
        gamma: is the discount factor
        epsilon: is the initial threshold for epsilon greedy
        min_epsilon: is the minimum value that epsilon should decay to
        epsilon_decay: is the decay rate for updating epsilon between episodes
    """
    total_rewards = []
    env = TimeLimit(env, max_episode_steps=max_steps)

    for episode in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0

        while not terminated and not truncated:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated and reward != 1.0:
                reward = -1.0

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

            episode_reward += reward
            state = next_state

        epsilon = max(min_epsilon, epsilon - epsilon_decay)
        total_rewards.append(episode_reward)

    return Q, total_rewards
