#!/usr/bin/env python3
import tensorflow.keras as K
"""
Builds a neural network with the Keras Sequential API.
"""


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras Sequential API.

    Args:
        nx (int): Number of input features.
        layers (list): List of integers representing
            the number of nodes in each layer.
        activations (list): List of strings representing
            the activation functions for each layer.
        lambtha (float): L2 regularization parameter.
        keep_prob (float): Dropout probability.
    """
    # input layer
    inputs = K.Input(shape=(nx,))
    # dont add dropout to the input layer
    x = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha),
    )(inputs)
    # add dropout to the rest of the layers
    # for i in range(1, len(layers)):
    for i in range(1, len(layers)):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha),
        )(x)
    # build the model
    model = K.Model(inputs=inputs, outputs=x)
    return model
