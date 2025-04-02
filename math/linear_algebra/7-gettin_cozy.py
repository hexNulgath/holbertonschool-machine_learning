#!/usr/bin/env python3
"""
Module that contains a function that concatenates
two matrices along a specified axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specified axis.
    """
    catmat = []

    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        catmat = mat1 + mat2

    elif axis == 1:
        for i in range(len(mat1)):
            catmat.append(mat1[i] + mat2[i])

    else:
        return None

    return catmat
