#!/usr/bin/env python3
"""
0-initialize.py
"""
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
