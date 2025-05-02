#!/usr/bin/env python3
"""
Functions to save and load a network
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    saves an entire model:
        network is the model to save
        filename is the path of the file that the
        model should be saved to
        Returns: None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    filename is the path of the file that the
    model should be loaded from
    Returns: the loaded model
    """
    return K.models.load_model(filename)
