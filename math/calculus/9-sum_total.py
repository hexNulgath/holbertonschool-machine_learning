#!/usr/bin/env python3
"""
Calculate the sum of squares of the first n natural numbers.
"""


def summation_i_squared(n):
    """
    Args:
    n (int): The upper limit of the summation.
    Returns:
    int: The sum of squares from 1 to n.
    """
    if n < 1:
        return None
    return sum(i**2 for i in range(1, n + 1))
