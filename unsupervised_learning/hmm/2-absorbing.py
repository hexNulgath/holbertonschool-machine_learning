#!/usr/bin/env python3
"""2-absorbing.py"""
import numpy as np


def absorbing(P):
    """
    determines if a markov chain is absorbing
    """
    # Input validation
    if not isinstance(P, np.ndarray):
        return False
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return False

    absorbing_states = np.where(np.isclose(P.diagonal(), 1))[0]
    if len(absorbing_states) == 0:
        return False

    # Check if all non-absorbing states can reach an absorbing state
    transient_states = [i for i in range(
        P.shape[0]) if i not in absorbing_states]
    if not transient_states:
        return True

    # Check if all transient states can reach an absorbing state
    # Using BFS or matrix powers (simplified here)
    Q = P[transient_states, :][:, transient_states]
    R = P[transient_states, :][:, absorbing_states]

    # Compute fundamental matrix N = (I - Q)^{-1}
    try:
        N = np.linalg.inv(np.eye(len(Q)) - Q)
    except np.linalg.LinAlgError:
        return False

    # If any row of N @ R is all zeros, some transient states can't be absorbed
    absorption_possible = np.allclose((N @ R).sum(axis=1), 1)
    return absorption_possible
