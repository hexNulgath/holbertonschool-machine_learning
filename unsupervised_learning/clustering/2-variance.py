#!/usr/bin/env python3
"""2-variance"""
import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster
    variance for a data set
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    variance = np.sum(np.min(
        np.linalg.norm(X[:, np.newaxis] - C, axis=2) ** 2, axis=1))
    return variance
