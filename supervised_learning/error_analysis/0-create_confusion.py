#!/usr/bin/env python3
"""0-create_confusion.py"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix

    Args:
        labels (list): one-hot numpy.ndarray of shape (m, classes)
        containing the correct labels for each data point
            m is the number of data points
            classes is the number of classes
        logits (numpy.ndarray): one-hot numpy.ndarray of shape
        (m, classes) containing the predicted labels

    Returns:
        numpy.ndarray: confusion matrix
    """
    # Find the maximum class index in either labels or logits
    size = max(np.max(labels), np.max(logits)) + 1
    size = int(size)
    matrix = np.zeros((size, size), dtype=int)

    for i in range(size):
        for j in range(size):
            mask = labels == i
            matrix[i, j] = (logits[mask] == j).sum()

    return matrix
