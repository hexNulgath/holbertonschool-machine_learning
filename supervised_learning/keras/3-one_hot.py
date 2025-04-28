#!/usr/bin/env python3
"""
one-hot encoding
"""
import tensorflow as tf


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
        return tf.one_hot(labels, labels.max() + 1)
    return tf.one_hot(labels, classes)
