#!/usr/bin/env python3
"""4-f1_score.py"""
import numpy as np


def f1_score(confusion):
    """
    calculates the F1 score of a confusion matrix:
    confusion: numpy.ndarray(classes, classes) where row indices
    represent the correct labels and column indices represent the
    predicted labels
        classes is the number of classes
    Returns: a numpy.ndarray(classes,) containing the F1 score of each class
    """
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision
    precision = precision(confusion)
    sensitivity = sensitivity(confusion)
    f1 = (2 * (precision * sensitivity)) / (precision + sensitivity)
    return f1
