#!/usr/bin/env python3
"""
decode a one hot matrix into a vector labels
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    one_hot np.ndarray shape(classes, m)
        classes: maximum number of clases
        m: number of examples
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    # Check if it's a valid one-hot matrix
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None

    # Each column should have exactly one 1
    if not np.all(one_hot.sum(axis=0) == 1):
        return None

    try:
        # use argmax to get index of class
        vector = np.argmax(one_hot, axis=0)
        return vector
    except Exception:
        return None
