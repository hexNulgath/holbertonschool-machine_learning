#!/usr/bin/env python3
"""lenet5.py"""
from tensorflow import keras as K


def lenet5(X):
    """
    builds a modified version of the LeNet-5 architecture using keras:

    X is a K.Input of shape (m, 28, 28, 1)
    containing the input images for the network
        m is the number of images

    Returns: a K.Model compiled to use Adam optimization
            (with default hyperparameters) and accuracy metrics
    """
    init = K.initializers.he_normal(seed=0)
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(
        5, 5), activation='relu', padding='same', kernel_initializer=init)(X)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(
        filters=16, kernel_size=(
            5, 5), activation='relu', padding='valid', kernel_initializer=init
    )(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    flatten = K.layers.Flatten()(pool2)
    dense1 = K.layers.Dense(
        units=120, kernel_initializer=init, activation='relu')(flatten)
    dense2 = K.layers.Dense(
        units=84, kernel_initializer=init, activation='relu')(dense1)
    dense3 = K.layers.Dense(
        units=10, kernel_initializer=init, activation='softmax')(dense2)
    model = K.Model(inputs=X, outputs=dense3)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
