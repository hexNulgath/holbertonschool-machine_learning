#!/usr/bin/env python3
"""5-backward.py"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    performs the backward algorithm for a hidden markov model
    """
    T = Observation.shape[0]
    N = Transition.shape[0]

    # Initialize the backward path probabilities
    B = np.zeros((N, T))
    B[:, T - 1] = 1

    # Compute backward probabilities
    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(
                B[:, t + 1] * Transition[i] * Emission[:, Observation[t + 1]])

    # Compute the total probability of the observations
    P = np.sum(Initial.flatten() * Emission[:, Observation[0]] * B[:, 0])

    return P, B
