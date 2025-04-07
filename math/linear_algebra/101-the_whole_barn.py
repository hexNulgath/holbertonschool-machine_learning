#!/usr/bin/env python3
"""
0-add_matrices.py
Adds two matrices element-wise.
"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices of the same dimension element-wise.

    Args:
        mat1: First matrix (can be 1D, 2D, or 3D list of numbers)
        mat2: Second matrix (must match dimension and shape of mat1)

    Returns:
        The resulting matrix, or None if:
        - Matrices have different dimensions
        - Elements are not numbers
        - Matrices have incompatible shapes
    """
    # Base case for scalar values (though not technically matrices)
    if isinstance(mat1, (int, float)) and isinstance(mat2, (int, float)):
        return mat1 + mat2

    # Check if both are lists
    if not (isinstance(mat1, list) and isinstance(mat2, list)):
        return None

    # Check outer dimension match
    if len(mat1) != len(mat2):
        return None

    # Recursive case for nested lists
    if isinstance(mat1[0], list):
        # Check if all sublists have the same length (consistent shape)
        if any(len(sub1) != len(mat2[0]) for
               sub1 in mat1) or any(len(sub2) != len(mat1[0]) for
                                    sub2 in mat2):
            return None
        # Recursively add sub-matrices
        result = []
        for sub1, sub2 in zip(mat1, mat2):
            sub_result = add_matrices(sub1, sub2)
            if sub_result is None:
                return None
            result.append(sub_result)
        return result
    else:
        # 1D case - element-wise addition
        if len(mat1) != len(mat2):
            return None
        # Check all elements are numbers
        if not all(isinstance(x, (int, float)) and
                   isinstance(y, (int, float)) for x, y in zip(mat1, mat2)):
            return None
        return [x + y for x, y in zip(mat1, mat2)]
