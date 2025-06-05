#!/usr/bin/env python3
"""dense_block module"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    that builds a dense block as described in Densely
    Connected Convolutional Networks:

    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block
    You should use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    All convolutions should be preceded by Batch Normalization and
    a rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the Dense
    Block and the number of filters within the
    concatenated outputs, respectively
    """
    concat = X
    for _ in range(layers):
        # Bottleneck layer
        X = K.layers.BatchNormalization(axis=3)(concat)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(
            4 * growth_rate, (1, 1), padding='same',
            kernel_initializer=K.initializers.he_normal(seed=0))(X)
        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(
            growth_rate, (3, 3), padding='same',
            kernel_initializer=K.initializers.he_normal(seed=0))(X)
        concat = K.layers.Concatenate(axis=3)([concat, X])

    return concat, nb_filters + layers * growth_rate
