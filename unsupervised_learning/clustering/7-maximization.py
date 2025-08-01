#!/usr/bin/env python3
"""7-maximization.py"""
import numpy as np


def maximization(X, g):
    """
    calculates the maximization
    step in the EM algorithm for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    pi = np.sum(g, axis=1) / n
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        g_i = g[i, :].reshape(-1, 1)
        m[i] = np.sum(g_i * X, axis=0) / np.sum(g_i)
        diff = X - m[i]
        S[i] = (g_i * diff).T @ diff / np.sum(g_i)

    return pi, m, S
