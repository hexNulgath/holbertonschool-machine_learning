#!/usr/bin/env python3
"""3-optimum"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    tests for the optimum number of clusters by variance
    """
    if kmin < 1 or (kmax is not None and kmax < kmin):
        return None, None
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if iterations < 1:
        return None, None
    results = []
    d_vars = []

    # Determine the maximum k to test
    if kmax is None:
        kmax = min(X.shape[0], 12)

    # Test each k in range
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        current_var = variance(X, C)
        d_vars.append(current_var)

    return results, d_vars
