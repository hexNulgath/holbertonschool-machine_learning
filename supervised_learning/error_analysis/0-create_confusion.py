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
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)
    size = labels.shape[1]
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            matrix[i, j] += np.sum(
                (true_classes == i) & (pred_classes == j)
            )

    return matrix
