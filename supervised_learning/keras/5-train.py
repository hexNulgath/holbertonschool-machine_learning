#!/usr/bin/env python3
"""
trains a model using mini-batch gradient descent
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
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
        validation_data: is the data to validate the model with, if not None

    Returns:
        History object generated during training
    """
    if validation_data is None:
        validation_data = (data, labels)

    network.fit(
        data, labels, batch_size=batch_size,
        epochs=epochs, verbose=verbose,
        shuffle=shuffle, validation_data=validation_data)
    return network.history
