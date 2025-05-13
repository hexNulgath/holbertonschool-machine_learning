#!/usr/bin/env python3
"""1-sensitivity.py"""
import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix:
    confusion numpy.ndarray of shape (classes, classes): row indices represent
    the correct labels and column indices represent the predicted labels
    classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the sensitivity
    of each class
    """
    classes = confusion.shape[0]
    sensitivity = np.zeros(classes)
    for i in range(classes):
        true_positives = confusion[i, i]
        total = np.sum(confusion[i, :])
        sensitivity[i] = true_positives / total
    return sensitivity
