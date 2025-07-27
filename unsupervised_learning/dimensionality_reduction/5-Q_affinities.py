#!/usr/bin/env python3
"""
5-Q_affinities.py
"""
import numpy as np


def Q_affinities(Y):
    """
    calculates the Q affinities:
    """
    dists = compute_low_dim_distances(Y)
    inv_dists = 1 / (1 + dists)  # Student-t kernel (1 degree of freedom)
    np.fill_diagonal(inv_dists, 0)  # Set q_{ii} = 0
    Q = inv_dists / np.sum(inv_dists)  # Normalize
    num = q_affinity_numerator(Y)
    return Q, num


def compute_low_dim_distances(Y):
    """
    Compute pairwise squared Euclidean distances in low-dimensional space.
    """
    sum_Y = np.sum(Y**2, axis=1)
    dists = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T + sum_Y, 0)
    return np.maximum(dists, 0)


def q_affinity_numerator(Y):
    """Compute the numerator of Q affinities for all pairs."""
    dists = np.sum((Y[:, None] - Y) ** 2, axis=-1)
    dists = 1 / (1 + dists)
    np.fill_diagonal(dists, 0)
    return dists
