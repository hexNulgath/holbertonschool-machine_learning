#!/usr/bin/env python3
"""
Calculate the sum of squares of the first n natural numbers.
"""


def summation_i_squared(n):
    """
    Computes the sum of squares from 1 to n using the mathematical formula.
    Args:
        n (int): Upper limit of the summation.
    Returns:
        int: Sum of squares, or None if n < 1.
    """
    if n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
