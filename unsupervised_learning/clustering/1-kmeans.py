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
    C = initialize(X, k)
    clss = np.zeros(X.shape[0])
    for i in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, np.newaxis] - C, axis=2), axis=1)
        C_old = C.copy()
        for j in range(k):
            if np.any(clss == j):
                C[j] = X[clss == j].mean(axis=0)
            else:
                C[j] = initialize(X, 1)
        if np.all(C_old == C):
            return C, clss

    return C, clss
