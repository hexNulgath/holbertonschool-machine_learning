#!/usr/bin/env python3
"""
5-definiteness.py
"""
import numpy as np


def definiteness(matrix):
    """
    calculates the definiteness of a matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if not matrix.ndim == 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.allclose(matrix, matrix.T):
        return None
    eigenvalues = np.linalg.eigvals(matrix)
    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues >= 0) and np.any(eigenvalues == 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues <= 0) and np.any(eigenvalues == 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
