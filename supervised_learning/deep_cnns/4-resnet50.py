#!/usr/bin/env python3
"""resnet50"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015):

    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be followed
    by batch normalization along the channels axis and a rectified
    linear activation (ReLU), respectively.
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero

    Returns: the keras model
    """
    input_1 = K.Input(shape=(224, 224, 3))

    # Stage 1
    X = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0))(input_1)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Stage 2
    X = projection_block(X, [64, 64, 256], s=1)
    for _ in range(2):
        X = identity_block(X, [64, 64, 256])

    # Stage 3
    X = projection_block(X, [128, 128, 512])
    for _ in range(3):
        X = identity_block(X, [128, 128, 512])

    # Stage 4
    X = projection_block(X, [256, 256, 1024])
    for _ in range(5):
        X = identity_block(X, [256, 256, 1024])

    # Stage 5
    X = projection_block(X, [512, 512, 2048])
    for _ in range(2):
        X = identity_block(X, [512, 512, 2048])

    # Output layer
    X = K.layers.AveragePooling2D((7, 7))(X)
    X = K.layers.Dense(1000,
                       activation='softmax',
                       kernel_initializer=K.initializers.he_normal(seed=0))(X)

    model = K.Model(inputs=input_1, outputs=X)

    return model
