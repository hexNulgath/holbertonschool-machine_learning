#!/usr/bin/env python3
"""Implement the training"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    implements a full training.

    env: initial environment
    nb_episodes: number of episodes used for training
    alpha: the learning rate
    gamma: the discount factor
    Return: all values of the score
    """
    scores = []
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)

    for episode in range(nb_episodes):
        state, _ = env.reset()
        done = False
        returns, grads, rewards = [], [], []

        while not done:
            # Choose action and gradient
            action, grad = policy_gradient(state, weight)
            grads.append(grad)
            # Step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rewards.append(reward)

            state = next_state

        # Compute discounted returns G_t for each time step
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # Update weights using the full episode
        for t in range(len(grads)):
            weight += alpha * returns[t] * grads[t]

        total_reward = sum(rewards)
        scores.append(total_reward)
        print(f"Episode: {episode} Score: {total_reward:.1f}")
        if show_result and (episode + 1) % 100 == 0:
            env.render()

    return scores
