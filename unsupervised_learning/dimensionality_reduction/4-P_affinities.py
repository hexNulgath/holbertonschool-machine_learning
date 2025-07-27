#!/usr/bin/env python3
"""
4-P_affinities.py
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set for t-SNE

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        tol: maximum tolerance for Shannon entropy difference
        perplexity: desired perplexity for all distributions

    Returns:
        P: numpy.ndarray of shape (n, n) containing symmetric P affinities
    """
    # Initialize distance matrix, P matrix, betas, and entropy
    D, P, beta, _ = P_init(X, perplexity)
    n = X.shape[0]
    log_perplexity = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):
        # Compute all pairwise affinities except i-i
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]

        # Binary search for sigma that results in desired perplexity
        H, thisP = binary_search_perplexity(Di, beta[i], log_perplexity, tol)

        # Fill in the row (except diagonal)
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Symmetrize P and normalize
    P = (P + P.T) / (2 * n)

    # Ensure diagonal is zero and avoid numerical issues
    np.fill_diagonal(P, 0)

    # Normalize again to ensure sum is 1
    P = P / np.sum(P)

    return P


def binary_search_perplexity(D, beta, log_perplexity, tol=1e-5, max_iter=50):
    """Helper function for binary search to find proper beta values"""
    def Hbeta(D, beta):
        P = np.exp(-D * beta)
        sumP = np.sum(P)
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P

    # Initialize search parameters
    betamin = -np.inf
    betamax = np.inf
    H, P = Hbeta(D, beta)
    Hdiff = H - log_perplexity
    tries = 0

    while np.abs(Hdiff) > tol and tries < max_iter:
        if Hdiff > 0:
            betamin = beta.copy()
            if betamax == np.inf:
                beta *= 2
            else:
                beta = (beta + betamax) / 2
        else:
            betamax = beta.copy()
            if betamin == -np.inf:
                beta /= 2
            else:
                beta = (beta + betamin) / 2

        H, P = Hbeta(D, beta)
        Hdiff = H - log_perplexity
        tries += 1

    return H, P
