#!/usr/bin/env python3
"""0-markov_chain.py"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    determines the probability of a markov chain
    being in a particular state after a specified
    number of iterations:
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 2 or s.shape[0] != 1:
        return None
    if not isinstance(t, int) or t < 0:
        return None
    if P.shape[0] != P.shape[1] or P.shape[0] != s.shape[1]:
        return None

    for i in range(t):
        s = np.dot(s, P)

    return s
