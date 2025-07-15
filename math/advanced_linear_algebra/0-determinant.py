#!/usr/bin/env python3


def determinant(matrix):
    """
    calculates the determinant of a matrix
    """
    if matrix == [[]]:
        return 1
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for i in range(n):
        submatrix = [row[:i] + row[i+1:] for row in matrix[1:]]
        det += (-1) ** (i) * matrix[0][i] * determinant(submatrix)

    return det
