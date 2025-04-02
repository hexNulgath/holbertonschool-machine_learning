#!/usr/bin/env python3
""""
This module provides a function to calculate the shape of a matrix.
"""


def matrix_shape(matrix):
    """
    Returns the shape of a matrix as a list of dimensions.
    """
    shape = []
    curr = matrix

    while isinstance(curr, list):
        shape.append(len(curr))
        if len(curr) == 0:
            break
        curr = curr[0]

    return shape
