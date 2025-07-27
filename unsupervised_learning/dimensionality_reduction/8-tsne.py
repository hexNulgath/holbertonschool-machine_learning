#!/usr/bin/env python3
"""
8-tsne.py
"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    performs a t-SNE transformation
    """
    intermediate = pca(X, idims)
    P = P_affinities(intermediate, perplexity)
    Y = np.random.randn(X.shape[0], ndims)
    Y_update = 0
    for i in range(100):
        dY, Q = grads(Y, P * 4)
        momentum = 0.5 if i < 20 else 0.8
        Y_update = momentum * Y_update - lr * dY
        Y += Y_update
        Y -= np.mean(Y, axis=0)
    Y_update = 0
    for i in range(iterations - 100):
        dY, Q = grads(Y, P)
        Y_update = 0.8 * Y_update - lr * dY
        Y += Y_update
        Y -= np.mean(Y, axis=0)
        if i % 100 == 0 or i == 0:
            print(f"Cost at iteration {i + 100}: {cost(P, Q)}")
    return Y
