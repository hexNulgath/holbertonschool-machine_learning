#!/usr/bin/env python3
"""
This module contains a function to calculate the minor of a matrix.
"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """
    Calculates the minor of a matrix.
    """
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    if n == 1:
        return [[1]]
    minors = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            submatrix = [row[:j] + row[j+1:] for row in matrix[:i] + matrix[i+1:]]
            if len(submatrix) == 1:
                minor_row.append(submatrix[0][0])
            elif len(submatrix) == 2:
                minor_row.append(submatrix[0][0] * submatrix[1][1] - submatrix[0][1] * submatrix[1][0])
            else:
                minor_row.append(determinant(matrix))
        minors.append(minor_row)
    return minors
