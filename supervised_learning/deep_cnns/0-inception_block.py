#!/usr/bin/env python3
"""inception block"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    builds an inception block as described in Going Deeper with Convolutions (2014):

    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP, respectively:
    F1 is the number of filters in the 1x1 convolution
    F3R is the number of filters in the 1x1 convolution before the 3x3 convolution
    F3 is the number of filters in the 3x3 convolution
    F5R is the number of filters in the 1x1 convolution before the 5x5 convolution
    F5 is the number of filters in the 5x5 convolution
    FPP is the number of filters in the 1x1 convolution after the max pooling
    All convolutions inside the inception block should use a rectified linear activation (ReLU)
    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    f1 = K.layers.Conv2D(F1, kernel_size=(
        1, 1), padding='same', activation='relu')(A_prev)
    f3r = K.layers.Conv2D(F3R, kernel_size=(
        1, 1), padding='same', activation='relu')(A_prev)
    f3 = K.layers.Conv2D(F3, kernel_size=(
        3, 3), padding='same', activation='relu')(f3r)
    f5r = K.layers.Conv2D(F5R, kernel_size=(
        1, 1), padding='same', activation='relu')(A_prev)
    f5 = K.layers.Conv2D(F5, kernel_size=(
        5, 5), padding='same', activation='relu')(f5r)
    fpp = K.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(1, 1), padding='same')(A_prev)
    fpp = K.layers.Conv2D(FPP, kernel_size=(
        1, 1), padding='same', activation='relu')(fpp)
    output = K.layers.Concatenate(axis=-1)([f1, f3, f5, fpp])
    return output
