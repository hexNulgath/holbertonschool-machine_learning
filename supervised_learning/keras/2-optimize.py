#!/usr/bin/env python3
"""
2-optimize.py
Module that contains a function to optimize a
model using Adam optimization algorithm
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Function that optimizes the model using Adam optimization algorithm
    Args:
        network: model to optimize
        alpha: learning rate
        beta1: first adam optimization parameter
        beta2: second adam optimization parameter

    """
    # Create the optimizer with the required parameters
    optimizer = K.optimizers.Adam(learning_rate=alpha,
                                  beta_1=beta1,
                                  beta_2=beta2)

    # Compile the model with the optimizer
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return None
