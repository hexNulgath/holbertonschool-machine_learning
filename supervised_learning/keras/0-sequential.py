#!/usr/bin/env python3
"""
    This function builds a neural network with the Keras Sequential API.
"""
import tensorflow as tf
from keras import layers as k_layers


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library:
    nx is the number of input features to the network
    layers is a list containing the number of nodes
in each layer of the network
    activations is a list containing the activation functions
used for each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    You are not allowed to use the Input class
    Returns: the keras model
    """
    model = tf.keras.Sequential()

    # Add first layer with input shape
    model.add(k_layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=tf.keras.regularizers.l2(lambtha),
        input_shape=(nx,),
    ))
    model.add(k_layers.Dropout(1 - keep_prob))

    # Add remaining layers
    for i in range(1, len(layers)):
        model.add(k_layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=tf.keras.regularizers.l2(lambtha),
        ))
        if i < len(layers) - 1:
            model.add(k_layers.Dropout(1 - keep_prob))

    return model
