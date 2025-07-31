#!/usr/bin/env python3
"""2-variance"""
import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster
    variance for a data set
    """
    variance = np.sum(np.min(
        np.linalg.norm(X[:, np.newaxis] - C, axis=2) ** 2, axis=1))
    return variance
