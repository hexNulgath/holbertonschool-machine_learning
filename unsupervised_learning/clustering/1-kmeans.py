#!/usr/bin/env python3
"""1-kmeans"""
import numpy as np


def initialize(X, k):
    """
    initializes cluster centroids for K-means
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    min_x = np.min(X, axis=0)
    max_x = np.max(X, axis=0)

    centroids = np.random.uniform(low=min_x, high=max_x, size=(k, X.shape[1]))
    return centroids


def kmeans(X, k, iterations=1000):
    """
    performs K-means on a dataset
    """
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
    clss = np.zeros(X.shape[0])
    for i in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)
        new_C = C.copy()
        for j in range(k):
            if np.any(clss == j):
                new_C[j] = X[clss == j].mean(axis=0)
            else:
                new_C[j] = initialize(X, 1)
        if np.allclose(C, new_C):
            break
        C = new_C

    return C, clss
