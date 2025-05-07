#!/usr/bin/env python3
"""3-mini_batch.py"""
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    creates mini-batches to be used for training a neural
    network using mini-batch gradient descent:

    X is a numpy.ndarray of shape (m, nx) representing input data
        m is the number of data points
        nx is the number of features in X
    Y is a numpy.ndarray of shape (m, ny) representing the labels
        m is the same number of data points as in X
        ny is the number of classes for classification tasks.
    batch_size is the number of data points in a batch
    Returns: list of mini-batches containing tuples (X_batch, Y_batch)
    """
    m = X.shape[0]
    batch_list = []
    full_size = m // batch_size
    X, Y = shuffle_data(X, Y)
    for i in range(0, full_size):
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        Y_batch = Y[i * batch_size:(i + 1) * batch_size]
        batch_list.append((X_batch, Y_batch))
    if m % batch_size != 0:
        X_batch = X[full_size * batch_size:]
        Y_batch = Y[full_size * batch_size:]
        batch_list.append((X_batch, Y_batch))

    return batch_list
