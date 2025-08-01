#!/usr/bin/env python3
"""10-kmeans.py"""
import sklearn.cluster
import numpy as np


def kmeans(X, k):
    """
    performs K-means on a dataset
    """
    if not isinstance(X, (list, np.ndarray)) or len(X) == 0:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None

    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
