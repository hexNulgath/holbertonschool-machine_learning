#!/usr/bin/env python3
"""
4-inverse.py
"""
determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """
    calculates the inverse of a matrix
    """
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Compute the determinant
    det = determinant(matrix)
    if det == 0:
        return None

    # Compute the adjugate
    adj = adjugate(matrix)

    # Compute the inverse
    inv = [[adj[i][j] / det for j in range(len(adj))] for i in range(len(adj))]
    return inv
