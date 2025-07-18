#!/usr/bin/env python3
"""
calculates the mean and covariance of a data set
"""
import numpy as np


def mean_cov(X):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
    Returns: mean, cov:
        mean is a numpy.ndarray of shape (1, d)
            containing the mean of the data set
        cov is a numpy.ndarray of shape (d, d)
            containing the covariance matrix of the data set
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n = X.shape[0]
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    deviations = X - mean
    cov = np.dot(deviations.T, deviations) / (n - 1)
    return mean, cov
