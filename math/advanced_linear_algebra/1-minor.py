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
            # Create a submatrix by excluding the i-th row and j-th column
            submatrix = [
                [matrix[x][y] for y in range(n) if y != j]
                for x in range(n) if x != i
            ]
            minor_row.append(determinant(submatrix))
        minors.append(minor_row)
    return minors
