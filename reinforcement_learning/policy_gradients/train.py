#!/usr/bin/env python3
"""Implement the training"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
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
        states, grads, rewards = [], [], []

        while not done:
            # Choose action and gradient
            action, grad = policy_gradient(state, weight)
            # Step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            grads.append(grad)
            rewards.append(reward)

            state = next_state

        # Compute discounted returns G_t for each time step
        G = np.zeros_like(rewards, dtype=np.float64)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + gamma * running_add
            G[t] = running_add

        # Update weights using all episode data for MC
        for g, grad in zip(G, grads):
            weight += alpha * grad * g

        total_reward = sum(rewards)
        scores.append(total_reward)
        print(f"Episode: {episode} Score: {total_reward:.1f}")

    return scores
