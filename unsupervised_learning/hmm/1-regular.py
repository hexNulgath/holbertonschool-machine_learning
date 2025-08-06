#!/usr/bin/env python3
"""1-reguar.py"""
import numpy as np


def regular(P):
    """
    determines the steady state probabilities
    of a regular markov chain
    """
    if not isinstance(P, np.ndarray):
        return None
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return None
    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return None
    
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    steady_state_index = np.argmax(np.isclose(eigenvalues, 1))
    pi = eigenvectors[:, steady_state_index].real
    pi /= pi.sum()

    if not np.all(pi > 0):
        return None

    return pi
