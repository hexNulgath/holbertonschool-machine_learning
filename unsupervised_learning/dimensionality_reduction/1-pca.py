#!/usr/bin/env python3
"""1-pca"""
import numpy as np


def pca(X, ndim):
    """
    performs PCA on a dataset
    """
    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    T = np.dot(X_centered, Vt.T)
    if ndim < T.shape[1]:
        T = T[:, :ndim]
    return T
