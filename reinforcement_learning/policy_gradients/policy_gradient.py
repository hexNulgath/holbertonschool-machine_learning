#!/usr/bin/env python3
"""policy gradient"""
import numpy as np


def policy(matrix, weight):
    """
    computes the policy with a weight of a matrix
    """
    z = matrix @ weight
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp)


def policy_gradient(state, weight):
    """
    computes the Monte-Carlo policy gradient
    based on a state and a weight matrix.
    state: matrix representing the current observation of the environment
    weight: matrix of random weight
    Return: the action and the gradient
    """
    prob = policy(state, weight)
    action = np.random.choice(len(prob), p=prob)

    grad = np.zeros_like(weight)
    grad[:, action] = state
    for a in range(weight.shape[1]):
        grad[:, a] -= prob[a] * state

    return action, grad
