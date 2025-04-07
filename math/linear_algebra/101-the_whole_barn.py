#!/usr/bin/env python3
"""
0-add_matrices.py
Adds two matrices element-wise.
This function takes two matrices (which can be 1D, 2D, or 3D lists of numbers)
and returns a new matrix that is the element-wise sum of the two matrices.
The function supports matrices of the same dimension and shape.
The function handles the following cases:
- If the matrices have different dimensions, it returns None.
- If the matrices have incompatible shapes, it returns None.
- If the elements are not numbers, it returns None.
"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices of the same dimension element-wise.

    Args:
        mat1: First matrix (can be 1D, 2D, or 3D list of numbers)
        mat2: Second matrix (must match dimension of mat1)

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
    try:
        if isinstance(mat1[0], list):
            if len(mat1[0]) != len(mat2[0]):
                return None
            # For 2D or higher dimensions, recursively add sub-matrices
            return [add_matrices(sub1, sub2) for sub1, sub2 in zip(mat1, mat2)]
        else:
            # 1D case - element-wise addition
            if len(mat1) != len(mat2):
                return None
            return [a + b for a, b in zip(mat1, mat2)]
    except TypeError:
        # Handle case where elements aren't numbers
        return None
    except IndexError:
        # Handle dimension mismatches in sub-matrices
        return None
