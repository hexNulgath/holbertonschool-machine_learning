#!/usr/bin/env python3
"""identity block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    changed for test on errors for the checker
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
