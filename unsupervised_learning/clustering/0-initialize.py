#!/usr/bin/env python3
"""
0-initialize.py
"""
import numpy as np


def initialize(X, k):
    """
    initializes cluster centroids for K-means
    """
    min_x = np.min(X, axis=0)
    max_x = np.max(X, axis=0)
    min_y = np.min(X, axis=1)
    max_y = np.max(X, axis=1)
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i] = np.random.uniform(low=min_x, high=max_x)
    return centroids
