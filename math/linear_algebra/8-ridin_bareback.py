#!/usr/bin/env python3
"""
performs matrix multiplication
"""

def mat_mul(mat1, mat2):
    """
    Multiplies two matrices.
    """
    if len(mat1[0]) != len(mat2):
        raise None
    new_mat = []
    for i in range (len(mat1)):
        new_row = []
        for j in range(len(mat2[0])):
            new_value = 0
            for k in range(len(mat2)):
                new_value += (mat1[i][k] * mat2[k][j])
            new_row.append(new_value)
        new_mat.append(new_row)
    return new_mat
