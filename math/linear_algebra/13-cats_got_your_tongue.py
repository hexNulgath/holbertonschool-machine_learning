#!/usr/bin/env python3
"""
13-cats_got_your_tongue.py
Concatenate two numpy arrays along a specified axis.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy arrays along a specified axis.
    """
    return np.concatenate((mat1, mat2), axis=axis)
