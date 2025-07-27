#!/usr/bin/env python3
"""2-P_init.py"""
import numpy as np


def P_init(X, perplexity):
    """
    initializes all variables required to calculate the P affinities in t-SNE
    """
    n, d = X.shape
    betas = np.ones((n, 1))
    D = np.zeros((n, n))
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = np.sum((X[i] - X[j]) ** 2)

    H = np.log2(perplexity)
    return D, P, betas, H
