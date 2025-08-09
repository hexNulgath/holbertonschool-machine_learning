#!/usr/bin/env python3
"""6-baum_welch"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for training HMM parameters
    """
    N = Transition.shape[0]
    M = Emission.shape[1]
    T = len(Observations)

    for iteration in range(iterations):
        # Forward algorithm
        _, alpha = forward(Observations, Emission, Transition, Initial)

        # Backward algorithm
        _, beta = backward(Observations, Emission, Transition, Initial)

        # Compute gamma (state probabilities)
        gamma = alpha * beta
        # Normalize gamma for each time step
        for t in range(T):
            gamma_sum = np.sum(gamma[:, t])
            if gamma_sum > 0:
                gamma[:, t] /= gamma_sum

        # Compute xi (transition probabilities)
        xi = np.zeros((N, N, T-1))
        for t in range(T-1):
            xi_sum = 0
            for i in range(N):
                for j in range(N):
                    xi[i, j, t] = (alpha[i, t] * Transition[i, j] *
                                   Emission[j, Observations[t+1]] *
                                   beta[j, t+1])
                    xi_sum += xi[i, j, t]

            # Normalize xi for time t
            if xi_sum > 0:
                xi[:, :, t] /= xi_sum

        # Re-estimate transition matrix
        for i in range(N):
            gamma_sum = np.sum(gamma[i, :-1])
            if gamma_sum > 0:
                for j in range(N):
                    Transition[i, j] = np.sum(xi[i, j, :]) / gamma_sum

        # Re-estimate emission matrix
        for i in range(N):
            gamma_sum = np.sum(gamma[i, :])
            if gamma_sum > 0:
                for k in range(M):
                    obs_sum = 0
                    for t in range(T):
                        if Observations[t] == k:
                            obs_sum += gamma[i, t]
                    Emission[i, k] = obs_sum / gamma_sum

    return Transition, Emission


def forward(Observation, Emission, Transition, Initial):
    """
    Forward algorithm
    """
    N = Emission.shape[0]
    T = len(Observation)

    alpha = np.zeros((N, T))

    # Initialization
    alpha[:, 0] = Initial.flatten() * Emission[:, Observation[0]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            alpha[j, t] = np.sum(
                alpha[:, t-1] * Transition[:, j]) * Emission[j, Observation[t]]

    # Total probability
    likelihood = np.sum(alpha[:, T-1])

    return likelihood, alpha


def backward(Observation, Emission, Transition, Initial):
    """
    Backward algorithm
    """
    T = len(Observation)
    N = Transition.shape[0]

    # Initialize backward probabilities
    beta = np.zeros((N, T))
    beta[:, T - 1] = 1

    # Recursion
    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[i, t] = np.sum(
                beta[:, t + 1] * Transition[i, :] *
                Emission[:, Observation[t + 1]])

    # Total probability
    P = np.sum(Initial.flatten() * Emission[:, Observation[0]] * beta[:, 0])

    return P, beta
