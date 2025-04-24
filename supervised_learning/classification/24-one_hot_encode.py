#!/usr/bin/env python3
"""
converts a numeric label vector into a one-hot matrix
"""


def one_hot_encode(Y, classes):
    """
    Y: a 1D numpy array of shape (m,)
        containing the numeric class labels
        m: the number of examples
    classes:  maximum number of classes found in Y
    Returns: a one-hot encoded numpy array of shape (classes, m)
    """
    import numpy as np
    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except AttributeError:
        return None
