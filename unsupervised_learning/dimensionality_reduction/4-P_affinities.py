#!/usr/bin/env python3
"""
Improved 4-P_affinities.py
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set for t-SNE
    """
    D, P, betas, H = P_init(X, perplexity)
    n = X.shape[0]

    # Convert betas to 1D array if it's 2D
    if betas.ndim == 2:
        betas = betas.flatten()

    for i in range(n):
        # Get distances for current point (excluding self)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Di = D[i, mask]

        # Perform binary search for this point
        beta_i = binary_search_perplexity(Di, betas[i], H, tol)

        # Calculate probabilities
        _, thisP = HP(Di, beta_i)

        # Insert probabilities back into P matrix
        P[i, mask] = thisP

    # Make P symmetric
    P = (P + P.T) / (2 * n)  # Normalize properly

    # Ensure diagonal is zero and apply minimal numerical floor
    P = np.maximum(P, np.finfo(P.dtype).eps)
    np.fill_diagonal(P, 0)

    # Final normalization to ensure sum = 1
    P = P / np.sum(P)

    return P


def binary_search_perplexity(D, beta, target_H, tol=1e-5, max_iter=100):
    """Binary search to find proper beta value that gives target entropy"""
    betamin = -np.inf
    betamax = np.inf

    H, _ = HP(D, beta)
    Hdiff = H - target_H
    tries = 0

    while np.abs(Hdiff) > tol and tries < max_iter:
        if Hdiff > 0:  # Entropy too high -> need higher beta
            betamin = beta
            if betamax == np.inf:
                beta *= 2
            else:
                beta = (beta + betamax) / 2
        else:  # Entropy too low -> need lower beta
            betamax = beta
            if betamin == -np.inf:
                beta /= 2
            else:
                beta = (beta + betamin) / 2

        # Handle extreme cases
        if beta > 1e100:
            return 1e100
        if beta < -1e100:
            return -1e100

        H, _ = HP(D, beta)
        Hdiff = H - target_H
        tries += 1

    return beta
