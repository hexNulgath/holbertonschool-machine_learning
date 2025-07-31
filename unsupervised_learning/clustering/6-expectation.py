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
        return None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None
    n, d = X.shape
    k = pi.shape[0]

    pdf_k = np.zeros((n, k))
    for i in range(k):
        pdf_k[:, i] = pi[i] * pdf(X, m[i], S[i])

    # Compute the log-likelihood (sum over all data points)
    li = np.sum(np.log(np.sum(pdf_k, axis=1)))
    pdf_k = np.maximum(pdf_k, 1e-300)
    # Compute responsibilities (normalize by the sum over all Gaussians)
    g = pdf_k / np.sum(pdf_k, axis=1, keepdims=True)
    return g.transpose(), li
