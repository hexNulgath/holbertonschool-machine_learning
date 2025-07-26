#!/usr/bin/env python3
"""0-pca"""
import numpy as np


def pca(X, var=0.95):
    """
    performs PCA on a dataset
    """
    SVD = np.linalg.svd(X, full_matrices=False)
    U = SVD[0]
    S = SVD[1]
    Vt = SVD[2]
    total_variance = np.sum(S)
    for i in range(len(S)):
        variance = np.sum(S[:i + 1])
        if variance / total_variance >= var:
            return Vt[:i + 1].T
    return Vt.T
