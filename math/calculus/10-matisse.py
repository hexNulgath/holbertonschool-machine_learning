#!/usr/bin/env python3
"""
10-matisse.py
Calculus module for polynomial operations.
"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Args:
        poly (list): Coefficients of the polynomial
        in decreasing order of degree.

    Returns:
        list: Coefficients of the derivative polynomial.
    """
    if poly == [0]:
        return [0]
    if poly == []:
        return None
    if type(poly) != list:
        return None
    if len(poly) == 1:
        return [0]
    if len(poly) == 2:
        return [poly[0]]
    return [i * coeff for i, coeff in enumerate(poly)][1:]
