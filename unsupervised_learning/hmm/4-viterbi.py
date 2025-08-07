#!/usr/bin/env python3
"""4-viterbi.py"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    calculates the most likely sequence of hidden
    states for a hidden markov model:
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

    V = np.zeros((N, T))
    path = np.zeros((N, T), dtype=int)

    V[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            prob = V[:, t - 1] * Transition[:, j]
            path[j, t] = np.argmax(prob)
            V[j, t] = np.max(prob) * Emission[j, Observation[t]]
    P = np.max(V[:, -1])
    path_index = np.argmax(V[:, -1])
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = path_index
    for t in range(T - 2, -1, -1):
        path_index = path[path_index, t + 1]
        best_path[t] = path_index
    best_path = best_path.tolist()
    return best_path, P
