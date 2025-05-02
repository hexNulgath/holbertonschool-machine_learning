#!/usr/bin/env python3
"""
Module for saving and loading model configurations.

This module provides functions to save a Keras model's configuration
to a JSON file and to load a model from a JSON configuration file.
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model’s configuration in JSON format:
    network is the model whose configuration should be saved
    filename is the path of the file that the configuration
    should be saved to
    Returns: None
    """
    json_config = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_config)


def load_config(filename):
    """
    loads a model with a specific configuration:
    filename is the path of the file containing the model’s
    configuration in JSON format
    Returns: the loaded model
    """
    f = open(filename)
    model = K.models.model_from_json(f.read())
    f.close()
    return model
