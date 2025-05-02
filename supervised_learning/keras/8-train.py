#!/usr/bin/env python3
"""
trains a model using mini-batch gradient descent
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
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
        early_stopping: is a boolean that indicates whether early
            stopping should be used
            early stopping should only be performed if validation_data exists
            early stopping should be based on validation loss
        patience: is the patience used for early stopping
        learning_rate_decay: is a boolean that indicates whether learning rate
            decay should be used
            learning rate decay should only be performed if
            validation_data exists
            the decay should be performed using inverse time decay
            the learning rate should decay in a stepwise fashion
            after each epoch
            each time the learning rate updates, Keras should print a message
        alpha: is the initial learning rate
        decay_rate: is the decay rate
        save_best: is a boolean indicating whether to save
            the model after each epoch if it is the best
            a model is considered the best if its validation
            loss is the lowest that the model has obtained
        filepath: is the file path where the model should be saved

    Returns:
        History object generated during training
    """
    callbaacks = []
    if learning_rate_decay and validation_data is not None:
        callbaacks.append(
            K.callbacks.LearningRateScheduler(
                lambda epoch: alpha / (1 + decay_rate * epoch),
                verbose=1
            )
        )
    if early_stopping and validation_data is not None:
        callbaacks.append(
            K.callbacks.EarlyStopping(
                patience=patience,
            )
        )
    if save_best and validation_data is not None:
        callbaacks.append(
            K.callbacks.ModelCheckpoint(
                filepath=filepath,
                save_best_only=True
            )
        )
    if validation_data is None:
        history = network.fit(
            data, labels, batch_size=batch_size,
            epochs=epochs, verbose=verbose,
            shuffle=shuffle)
    else:
        history = network.fit(
            data, labels, batch_size=batch_size,
            epochs=epochs, verbose=verbose,
            shuffle=shuffle, validation_data=validation_data,
            callbacks=callbaacks)

    return history
