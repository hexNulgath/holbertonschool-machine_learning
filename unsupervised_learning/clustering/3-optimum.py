#!/usr/bin/env python3
"""3-optimum"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    tests for the optimum number of clusters by variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if kmin >= kmax:
        return None, None
    results = []
    total_vars = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        current_var = variance(X, C)
        total_vars.append(current_var)
    for var in total_vars:
        d_vars.append(np.sqrt((var - total_vars[0])**2))

    return results, d_vars
