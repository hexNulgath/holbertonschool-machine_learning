#!/usr/bin/env python3
"""6-expectation"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    calculates the expectation step
    in the EM algorithm for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return (None, None)
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return (None, None)
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return (None, None)
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return (None, None)
    n, d = X.shape
    k = pi.shape[0]
    if (k != S.shape[0] or d != m.shape[1] or d != S.shape[1]
            or d != S.shape[2]):
        return (None, None)
    if not np.all(np.isclose(np.sum(pi), 1)):
        return (None, None)
    pdf_k = np.zeros((k, n))
    for i in range(k):
        pdf_k[i] = pi[i] * pdf(X, m[i], S[i])

    # Compute the log-likelihood (sum over all data points)
    li = np.sum(np.log(np.sum(pdf_k, axis=0)))
    # Compute responsibilities (normalize by the sum over all Gaussians)
    g = pdf_k / np.sum(pdf_k, axis=0)
    return g, li
