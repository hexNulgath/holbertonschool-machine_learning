#!/usr/bin/env python3
"""
one-hot encoding
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    one-hot encoding
    Args:
        labels: a numpy.ndarray with shape (m,) containing the integer
            labels for the classification task
        classes: the maximum number of classes; if classes is None, use the
            maximum value in labels + 1
    Returns:
        a one-hot encoding of labels with shape (m, classes)
    """
    if classes is None:
        return K.backend.one_hot(labels, labels.max() + 1).numpy()
    return K.bakend.one_hot(labels, classes).numpy()
