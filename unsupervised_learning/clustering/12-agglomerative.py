#!/usr/bin/env python3
"""12-agglomerative.py"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    performs agglomerative clustering on a dataset
    """ 
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    plt.figure(figsize=(10, 7))
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist, above_threshold_color='gray')
    plt.title(f'Agglomerative Clustering (Distance Threshold: {dist})')
    plt.xlabel('Data points')
    plt.ylabel('Distance')
    plt.axhline(y=dist, color='r', linestyle='--')
    plt.show()

    return clss
