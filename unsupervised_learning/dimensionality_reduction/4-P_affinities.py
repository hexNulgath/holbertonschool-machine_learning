#!/usr/bin/env python3
"""
4-P_affinities.py
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set

        X: numpy.ndarray of shape (n, d) containing the dataset
        tol: maximum tolerance for Shannon entropy difference
        perplexity: desired perplexity for all distributions

    Returns:
        P: numpy.ndarray of shape (n, n) containing symmetric P affinities
    """
    # Initialize distance matrix, P matrix, betas, and entropy
    D, P, betas, _ = P_init(X, perplexity)
    n = X.shape[0]
    target_H = np.log2(perplexity)  # Convert perplexity to target entropy
    for i in range(n):
        # Calculate Shannon entropy for the current row
        H, P[i, 1:] = HP(D[i, 1:], betas[i])

        # Adjust beta until the entropy is within the tolerance
        for _ in range(100):
            if np.abs(H - target_H) > tol:
                if H < target_H:
                    betas[i] *= 0.5  # Decrease beta
                else:
                    betas[i] *= 2.0  # Increase beta

                # Recalculate P affinities and entropy
                P[i, 1:] = np.exp(-D[i, 1:] * betas[i])
                P[i, 1:] /= np.sum(P[i, 1:])  # Normalize to sum to 1
                H, _ = HP(D[i, 1:], betas[i])
        # Set P[i, 0] to 0 for symmetry
        P[i, 0] = 0.0
    # Make P symmetric
    P = (P + P.T) / 2
    # Set diagonal to 0
    np.fill_diagonal(P, 0.0)
    # Normalize P to sum to 1
    P /= np.sum(P)

    return P
