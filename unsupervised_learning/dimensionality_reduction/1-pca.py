#!/usr/bin/env python3
"""1-pca"""
import numpy as np


def pca(X, ndim):
    """
    performs PCA on a dataset
    """
    X_centered = X - np.mean(X, axis=0)
    SVD = np.linalg.svd(X_centered, full_matrices=False)
    U = SVD[0]
    S = SVD[1]
    T = np.dot(U, np.diag(S))
    if ndim < T.shape[1]:
        T = T[:, :ndim]
    T[T == -0.0] = 0.0
    return T
