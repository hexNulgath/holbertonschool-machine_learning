#!/usr/bin/env python3
"""
6-grads.py
"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y
    Args:
        Y: Low-dimensional embeddings (n_samples, n_components)
        P: High-dimensional affinities (n_samples, n_samples)
    Returns:
        dY: Gradient of KL divergence w.r.t. Y
        Q: Low-dimensional affinities
    """
    n, d = Y.shape
    Q, num = Q_affinities(Y)
    dY = np.zeros_like(Y)

    # Compute squared Euclidean distances efficiently
    dists = compute_low_dim_distances(Y)

    # Compute all pairwise differences (n, n, d)
    diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]

    # Compute coefficients (n, n)
    coeff = (P - Q) * (1 / (1 + dists))

    # Vectorized gradient calculation
    dY = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)

    return dY, Q


def compute_low_dim_distances(Y):
    """Squared Euclidean distances between all pairs in Y."""
    sum_Y = np.sum(Y**2, axis=1)
    dists = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
    return np.maximum(dists, 0)
