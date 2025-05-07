#!/usr/bin/env python3
"""
1-normalize.py
"""
import tensorflow as tf
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
        normalized_X[row] = tf.nn.batch_normalization(
            X[row], mean=m, variance=s**2,
            offset=None, scale=None,
            variance_epsilon=1e-8)
    return normalized_X
