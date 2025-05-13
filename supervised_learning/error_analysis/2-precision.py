#!/usr/bin/env python3
"""2-precision.py"""
import numpy as np


def precision(confusion):
    """
    calculates the precision for each class in a confusion matrix:

    confusion numpy.ndarray(classes, classes) row indices represent the
    correct labels and column indices represent the predicted labels
        classes is the number of classes
    Returns: numpy.ndarray of shape (classes,) containing the
        precision of each class
    """
    classes = confusion.shape[0]
    precision = np.zeros(classes)
    for i in range(classes):
        true_positives = confusion[i, i]
        total_positives = np.sum(confusion[:, i])
        precision[i] = true_positives / total_positives
    return precision
