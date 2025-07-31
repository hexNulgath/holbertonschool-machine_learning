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
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    # Ensure we analyze at least 2 different cluster sizes
    if kmax is None:
        kmax = 12

    results = []
    variances = []

    # Test each cluster size
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        var = variance(X, C)
        variances.append(var)

    # Calculate differences from smallest cluster size variance
    if not variances:
        return None, None

    base_variance = variances[0]
    d_vars = [base_variance - var for var in variances]

    return results, d_vars
