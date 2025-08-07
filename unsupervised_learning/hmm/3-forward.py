#!/usr/bin/env python3
import numpy as np


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
