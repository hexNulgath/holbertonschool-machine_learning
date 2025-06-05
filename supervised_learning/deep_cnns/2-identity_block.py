#!/usr/bin/env python3
"""identity block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015):
    A_prev is the output frm the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear
    activation (ReLU), respectively.
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)
    # First 1x1 convolution
    x = K.layers.Conv2D(F11, (1, 1), padding='same',
                        kernel_initializer=initializer)(A_prev)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)
    # 3x3 convolution
    x = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=initializer)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)
    # Second 1x1 convolution
    x = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer=initializer)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    # Adding the input to the output
    x = K.layers.Add()([x, A_prev])
    x = K.layers.Activation('relu')(x)
    return x
