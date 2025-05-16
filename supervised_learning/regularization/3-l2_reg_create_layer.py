#!/usr/bin/env python3
"""
3-l2_reg_create_layer.py
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    creates a neural network layer in tensorFlow that includes L2 regularization:

    prev is a tensor containing the output of the previous layer
    n is the number of nodes the new layer should contain
    activation is the activation function that should be used on the layer
    lambtha is the L2 regularization parameter
    Returns: the output of the new layer
    """
    kernel_regularizer = tf.keras.regularizers.l2(lambtha)
    layer = tf.keras.layers.Dense(
        n, activation=activation, kernel_regularizer=kernel_regularizer)(prev)
    return layer
