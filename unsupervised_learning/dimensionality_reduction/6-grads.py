#!/usr/bin/env python3
"""
6-grads.py
"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    calculates the gradients of Y
    """
    n , d = Y.shape
    Q, num = Q_affinities(Y)
    dY = np.zeros_like(Y)
    diff = Y[:, None] - Y  # Pairwise differences
    for i in range(n):
        coeff = (P[i] - Q[i]) * (1 / (1 + compute_low_dim_distances(Y)[i]))
        dY[i] = 4 * np.sum(diff[i] * coeff[:, None], axis=0)

    return dY, Q


def compute_low_dim_distances(Y):
    """Squared Euclidean distances between all pairs in Y."""
    sum_Y = np.sum(Y**2, axis=1)
    dists = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T + sum_Y, 0)
    return np.maximum(dists, 0)