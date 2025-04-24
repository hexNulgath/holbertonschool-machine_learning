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
    if type(one_hot) is not np.ndarray:
        return None
    try:
        # use argmax to get index of class
        vector = np.array(np.argmax(one_hot, axis=0))
        return vector
    except Exception:
        return None
