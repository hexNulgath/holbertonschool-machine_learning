#!/usr/bin/env python3
"""
Concatenates two matrices along a specified axis.
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along the specified axis.

    Args:
        mat1: First matrix (list of lists)
        mat2: Second matrix (must match dimensions except along axis)
        axis: Axis along which to concatenate (0: rows, 1: columns, 2: depth)

    Returns:
        The concatenated matrix, or None if shapes are incompatible
    """
    def get_shape(matrix):
        """Helper to get matrix shape as a list of dimensions"""
        shape = []
        while isinstance(matrix, list):
            shape.append(len(matrix))
            matrix = matrix[0] if matrix else None
        return shape

    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    # Check all dimensions except the concatenation axis match
    if len(shape1) != len(shape2):
        return None
    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None

    # Perform concatenation
    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    elif axis == 2:
        return [[col1 + col2 for col1, col2 in zip(row1, row2)]
                for row1, row2 in zip(mat1, mat2)]
    elif axis == 3:
        return [[[number1 + number2 for number1, number2 in zip(col1, col2)]
                for col1, col2 in zip(row1, row2)]
                for row1, row2 in zip(mat1, mat2)]
    else:
        return None  # Invalid axis
