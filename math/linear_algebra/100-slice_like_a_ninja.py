#!/usr/bin/env python3
"""
0-slice_like_a_ninja.py
Slicing a matrix along the specified axes.
This function takes a matrix and a dictionary of axes as input,
and returns a new matrix that is sliced along the specified axes.
The function supports slicing along the first three axes (0, 1, and 2).
"""


def np_slice(matrix, axes=None):
    """
    Slices a matrix along specified axes using NumPy's slicing capabilities.

    Args:
        matrix: A numpy ndarray to be sliced
        axes: Dictionary mapping dimensions to:
              - slice objects
              - tuples of (stop,)
              - tuples of (start, stop)
              - tuples of (start, stop, step)
              Examples:
                  {0: (5,)}        # equivalent to :5
                  {1: (2, 5)}      # equivalent to 2:5
                  {2: (1, 5, 2)}    # equivalent to 1:5:2

    Returns:
        A new sliced numpy ndarray

    Raises:
        TypeError: If input is not a numpy array
        ValueError: If slice specification is invalid
    """
    if axes is None:
        axes = {}

    slices = []
    for dim in range(matrix.ndim):
        if dim in axes:
            dim_spec = axes[dim]

            # Handle slice objects directly
            if isinstance(dim_spec, slice):
                slices.append(dim_spec)
                continue

            # Handle tuple specifications
            try:
                if len(dim_spec) == 1:  # (stop,)
                    slices.append(slice(None, dim_spec[0]))
                elif len(dim_spec) == 2:  # (start, stop)
                    slices.append(slice(dim_spec[0], dim_spec[1]))
                elif len(dim_spec) == 3:  # (start, stop, step)
                    slices.append(slice(dim_spec[0], dim_spec[1], dim_spec[2]))
                else:
                    raise ValueError("Slice must have 1, 2, or 3 elements")
            except TypeError:
                raise TypeError("Axis must be a slice object or a sequence")
        else:
            slices.append(slice(None))

    return matrix[tuple(slices)]
