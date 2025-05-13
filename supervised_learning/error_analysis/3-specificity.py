#!/usr/bin/env python3
"""3-specificity.py"""
import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a confusion matrix:
    confusion numpy.ndarray(classes, classes) where row indices represent
    the correct labels and column indices represent the predicted labels
        classes:the number of classes
    Returns: numpy.ndarray(classes,) containing the specificity of each class
    """
    classes = confusion.shape[0]
    specificity = np.zeros(classes)
    for i in range(classes):
        tp = confusion[i, i]
        # Predicted as class i, but not truly class i
        fp = np.sum(confusion[:, i]) - tp
        # Truly class i, but predicted as another class
        fn = np.sum(confusion[i, :]) - tp
        tn = np.sum(confusion) - tp - fp - fn
        total_negatives = tn + fp

        specificity[i] = tn / total_negatives
    return specificity
