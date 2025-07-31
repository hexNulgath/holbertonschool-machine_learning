#!/usr/bin/env python3
"""1-kmeans"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids using multivariate uniform distribution
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    min_x = np.min(X, axis=0)
    max_x = np.max(X, axis=0)
    return np.random.uniform(low=min_x, high=max_x, size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialize centroids
    C = initialize(X, k)
    if C is None:
        return None, None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    clss = np.zeros(X.shape[0], dtype=int)

    for _ in range(iterations):
        # Assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        new_clss = np.argmin(distances, axis=1)

        # Update centroids
        for j in range(k):
            mask = (new_clss == j)
            if np.any(mask):
                C[j] = X[mask].mean(axis=0)
            else:
                # Reinitialize empty cluster (2nd use of np.random.uniform)
                C[j] = np.random.uniform(
                    low=min_vals, high=max_vals, size=X.shape[1])

        # Check convergence
        if np.array_equal(new_clss, clss):
            break
        clss = new_clss

    return C, clss
