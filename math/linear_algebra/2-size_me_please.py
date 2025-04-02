#!/usr/bin/env python3

def matrix_shape(matrix):
    """
    Returns the shape of a matrix as a tuple (rows, columns).
    """
    rows = len(matrix)
    columns = len(matrix[0]) if isinstance(matrix[0], list) else 1
    if isinstance(matrix[0][0], list):
        depth = len(matrix[0][0])
    else:
        return [rows, columns]
    return [rows, columns, depth]
