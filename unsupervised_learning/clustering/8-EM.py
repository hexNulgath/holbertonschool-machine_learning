#!/usr/bin/env python3
"""8-EM.py"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    performs the expectation maximization for a GMM
    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
        pi is a numpy.ndarray of shape (k,)
            containing the priors for each cluster
        m is a numpy.ndarray of shape (k, d)
            containing the centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d)
            containing the covariance matrices for each cluster
        g is a numpy.ndarray of shape (k, n)
            containing the probabilities for each data point in each cluster
        l is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    n, d = X.shape
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    prev_l = None

    for i in range(iterations + 1):
        g, li = expectation(X, pi, m, S)
        if g is None or li is None:
            return None, None, None, None, None
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {li:.5f}")
        if i > 0 and prev_l is not None and abs(li - prev_l) < tol:
            if verbose:
                print(f"Log Likelihood after {i} iterations: {li:.5f}")
            return pi, m, S, g, li

        prev_l = li

        if i < iterations:
            pi, m, S = maximization(X, g)
            if pi is None or m is None or S is None:
                return None, None, None, None, None

    return pi, m, S, g, li
