#!/usr/bin/env python3
"""
10-ill_use_my_scale.py
"""


def np_shape(matrix):
    """
    Function to return the shape of a matrix
    """
    if not isinstance(matrix, list):
        return None
    if len(matrix) == 0:
        return (0,)
    if isinstance(matrix[0], list):
        return (len(matrix),) + np_shape(matrix[0])
    return (len(matrix),)
