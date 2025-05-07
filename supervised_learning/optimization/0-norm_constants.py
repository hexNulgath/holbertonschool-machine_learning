#!/usr/bin/env python3
"""0-norm_constants.py"""
import numpy as np


def normalization_constants(X):
    """Calculates the normalization constants of a matrix

    Args:
        X (m, nx): The matrix to normalize

    Returns:
        tuple: The mean and standard deviation of the matrix
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return mean, std
