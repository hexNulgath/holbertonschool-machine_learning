#!/usr/bin/env python3
"""
1-normalize.py
"""
import numpy as np


def normalize(X, m, s):
    """
    normalizes (standardizes) a matrix:
    X is the numpy.ndarray of shape (d, nx) to normalize
        d is the number of data points
        nx is the number of features
    m is a numpy.ndarray of shape (nx,)
        contains the mean of all features of X
    s is a numpy.ndarray of shape (nx,)
        contains the standard deviation of all features of X
    Returns: The normalized X matrix
    """
    normalized_X = np.ndarray(X.shape)
    for row in range(X.shape[0]):
        normalized_X[row] = ((X[row] - m) / s)
    return normalized_X
