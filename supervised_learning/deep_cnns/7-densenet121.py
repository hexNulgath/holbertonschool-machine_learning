#!/usr/bin/env python3
"""densenet121.py"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds the DenseNet-121 architecture as described
    in Densely Connected Convolutional Networks:

    growth_rate is the growth rate
    compression is the compression factor
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    Returns: the keras model
    """
    inputs = K.Input(shape=(224, 224, 3))

    X = K.layers.BatchNormalization(axis=3)(inputs)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0))(X)
    X = K.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), padding='same')(X)

    # Dense Block 1
    X, nb_filters = dense_block(X, 64, growth_rate, 6)

    # Transition Layer 1
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    # Transition Layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    # Transition Layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    X = K.layers.AveragePooling2D(pool_size=(7, 7))(X)

    # Fully Connected Layer
    outputs = K.layers.Dense(
        1000, activation='softmax',
        kernel_initializer=K.initializers.he_normal(seed=0))(X)

    model = K.Model(inputs=inputs, outputs=outputs)
    return model
