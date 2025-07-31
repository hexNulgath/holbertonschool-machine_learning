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
        # Calculate distances from centroids
        # Vectorized distance calculation instead of broadcasting
        X_vectors = np.repeat(X[:, np.newaxis], k, axis=1)
        X_vectors = np.reshape(X_vectors, (X.shape[0], k, X.shape[1]))
        C_vectors = np.tile(C[np.newaxis, :], (X.shape[0], 1, 1))
        C_vectors = np.reshape(C_vectors, (X.shape[0], k, X.shape[1]))
        # Calculate Euclidean distances
        distances = np.linalg.norm(X_vectors - C_vectors, axis=2)
        new_clss = np.argmin(distances, axis=1)
        old_C = C.copy()
        # Update centroids
        for j in range(k):
            mask = (new_clss == j)
            if np.any(mask):
                C[j] = X[mask].mean(axis=0)
            else:
                # Reinitialize empty cluster (2nd use of np.random.uniform)
                C[j] = np.random.uniform(
                    low=min_vals, high=max_vals, size=X.shape[1])

        if np.all(C == old_C):
            return C, clss
        clss = new_clss

    return C, clss
