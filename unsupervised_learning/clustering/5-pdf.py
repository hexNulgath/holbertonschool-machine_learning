#!/usr/bin/env python3
"""5-pdf"""
import numpy as np


def pdf(X, m, S):
    """
    calculates the probability density
    function of a Gaussian distribution
    1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or m.shape != (X.shape[1],):
        return None
    if not isinstance(S, np.ndarray) or S.shape != (X.shape[1], X.shape[1]):
        return None

    X = np.asarray(X)
    m = np.asarray(m)
    S = np.asarray(S)

    n, d = X.shape
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    norm_const = 1.0 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))

    diff = X - m
    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)

    p = norm_const * np.exp(exponent)

    p = np.maximum(p, 1e-300)

    return p
