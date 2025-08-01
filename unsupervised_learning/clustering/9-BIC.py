#!/usr/bin/env python3
"""9-bic.py"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    finds the best number of clusters for a GMM
    using the Bayesian Information Criterion
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None
    n, d = X.shape
    best_k = None
    best_result = None
    best_BIC = None
    L = np.zeros(kmax - kmin + 1)
    b = np.array([])
    for i in range(kmin, kmax + 1):
        pi, m, S, g, li = expectation_maximization(
            X, i, iterations, tol, verbose)
        if pi is None or m is None or S is None or g is None or li is None:
            return None, None, None, None
        p = i * (d + d * (d + 1) / 2)
        BIC = p * np.log(n) - 2 * li
        if best_BIC is None or BIC < best_BIC:
            best_k = i
            best_result = (pi, m, S)
            best_BIC = BIC
        L[i - kmin] = li
        b = np.append(b, BIC)
    return best_k, best_result, L, b
