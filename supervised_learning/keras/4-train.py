#!/usr/bin/env python3
"""
trains a model using mini-batch gradient descent
"""
import tensorflow.keras as keras


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Args:
        network: The model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape
            (m, classes) containing the labels
        batch_size: size of the batch for mini-batch gradient descent
        epochs: number of passes through data
        verbose: bool determining if training output should be printed
        shuffle: bool determining whether to shuffle batches each epoch

    Returns:
        History object generated during training
    """
    network.fit(
        data, labels, batch_size=batch_size,
        epochs=epochs, verbose=verbose,
        shuffle=shuffle)
    return network.history
