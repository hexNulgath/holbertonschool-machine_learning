#!/usr/bin/env python3
"""
2-cofactor.py
"""
determinant = __import__('0-determinant').determinant


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix.

    Args:
        matrix (list of lists): Input square matrix.

    Returns:
        list of lists: Cofactor matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square or is empty.
    """
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    # Handle 1x1 matrix case
    if n == 1:
        return [[1]]

    cofactors = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            # Create submatrix by excluding row i and column j
            submatrix = [
                [matrix[x][y] for y in range(n) if y != j]
                for x in range(n) if x != i
            ]
            # Compute the minor and apply sign (-1)^(i+j)
            minor = determinant(submatrix)
            cofactor_row.append(minor * (-1) ** (i + j))
        cofactors.append(cofactor_row)

    return cofactors
