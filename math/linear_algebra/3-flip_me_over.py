#!/usr/bin/env python3
"""
This script demonstrates how to transpose a matrix.
"""


def matrix_transpose(matrix):
    """
    Transpose a matrix.
    Args:
        matrix (list of list of int): The matrix to transpose.
    """

    transposed = []
    for i in range(len(matrix[0])):
        new_row = []
        for j in range(len(matrix)):
            new_row.append(matrix[j][i])
        transposed.append(new_row)
    return transposed
