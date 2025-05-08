#!/usr/bin/env python3
"""14-batch_norm.py"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow:

    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used on the output of the layer

    Returns: a tensor of the activated output for the layer
    """
    # Initialize the dense layer with no activation
    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'),
    )(prev)

    # Apply batch normalization to the dense layer output
    batch_norm_layer = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        gamma_initializer='ones',
        beta_initializer='zeros')(dense_layer)

    return tf.keras.layers.Activation(activation)(batch_norm_layer)
