#!/usr/bin/env python3
"""
3-adjugate.py
"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """
    calculates the adjugate matrix of a matrix
    """
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Compute the cofactor matrix
    cofactors = cofactor(matrix)

    # Transpose the cofactor matrix to get the adjugate
    adjugate = [list(row) for row in zip(*cofactors)]
    return adjugate
