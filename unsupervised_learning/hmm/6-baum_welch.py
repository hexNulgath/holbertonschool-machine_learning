#!/usr/bin/env python3
"""6-baum_welch"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    performs the Baum-Welch algorithm for a hidden markov model
    """
    M = Transition.shape[0]
    T = Observations.shape[0]

    for _ in range(iterations):
        # Forward pass with scaling
        _, alpha = forward(Observations, Emission, Transition, Initial)

        # Backward pass
        _, beta = backward(Observations, Emission, Transition, Initial)

        # Compute gamma (state probabilities)
        gamma = alpha * beta
        gamma_sum = np.sum(gamma, axis=0, keepdims=True)
        gamma /= gamma_sum

        # Compute xi (transition probabilities)
        Xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            for i in range(M):
                for j in range(M):
                    Xi[i, j, t] = ((
                        alpha[i, t] * Transition[i, j]) * (
                            Emission[j, Observations[t + 1]] * beta[j, t + 1]))
            # Normalize Xi for each time step
            xi_sum = np.sum(Xi[:, :, t])
            Xi[:, :, t] /= xi_sum

        # Update initial probabilities
        Initial = gamma[:, 0].reshape(-1, 1)

        # Update transition matrix
        trans_num = np.sum(Xi, axis=2)
        trans_den = np.sum(gamma[:, :-1], axis=1, keepdims=True)
        Transition = trans_num / trans_den

        # Update emission matrix
        for k in range(Emission.shape[1]):
            mask = (Observations == k).astype(int)
            emission_num = np.sum(gamma * mask, axis=1)
            emission_den = np.sum(gamma, axis=1)
            Emission[:, k] = emission_num / emission_den

    return Transition, Emission


def forward(Observation, Emission, Transition, Initial):
    """
    performs the forward algorithm for a hidden markov model
    """
    if (not isinstance(Observation, np.ndarray) or
            Observation.ndim != 1 or
            not isinstance(Emission, np.ndarray) or
            Emission.ndim != 2 or
            not isinstance(Transition, np.ndarray) or
            Transition.ndim != 2 or
            not isinstance(Initial, np.ndarray) or
            Initial.ndim != 2 or
            Initial.shape[1] != 1):
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    if (N != Transition.shape[0] or N != Transition.shape[1] or
            N != Initial.shape[0] or Initial.shape[1] != 1):
        return None, None

    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(
                F[:, t - 1] * Transition[:, j]) * Emission[j, Observation[t]]

    P = np.sum(F[:, -1])
    return P, F


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
