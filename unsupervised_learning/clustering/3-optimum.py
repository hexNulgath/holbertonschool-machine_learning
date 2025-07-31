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

    results = []
    variances = []

    # Set initial k
    k = kmin

    while True:
        # Perform k-means clustering
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None

        results.append((C, clss))
        current_var = variance(X, C)
        variances.append(current_var)

        # Calculate variance reduction
        if k > kmin:
            prev_var = variances[-2]
            reduction = (prev_var - current_var) / prev_var

            # Stopping conditions
            if kmax is not None and k >= kmax:
                break
            if kmax is None and (reduction < 0.01 or k >= X.shape[0]):
                break

        elif kmax is None and kmin == 1:  # Ensure we test at least k=1 and k=2
            pass
        elif kmax is not None and k >= kmax:
            break

        k += 1

    # Calculate differences from smallest cluster size variance
    base_variance = variances[0]
    d_vars = [base_variance - var for var in variances]

    return results, d_vars
