#!/usr/bin/env python3
"""2-shuffle_data.py"""
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way:

    X is the first numpy.ndarray of shape (m, nx) to shuffle
        m is the number of data points
        nx is the number of features in X
    Y is the second numpy.ndarray of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y
    Returns: the shuffled X and Y matrices
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
