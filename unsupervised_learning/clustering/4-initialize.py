#!/usr/bin/env python3
"""4-initialize"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    initializes variables for a Gaussian Mixture Model
    """
    if not isinstance(X, np.ndarray):
        return None, None, None
    if len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    n, d = X.shape
    pi = np.full((k,), 1 / k)
    m, clss = kmeans(X, k)
    S = np.tile(np.eye(d), (k, 1, 1))
    return pi, m, S
