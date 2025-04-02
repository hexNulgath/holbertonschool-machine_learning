#!/usr/bin/env python3
"""
5-across_the_planes.py
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)
    return result
