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
    layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'),
        name="layer")(prev)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]))
    beta = tf.Variable(tf.constant(0.0, shape=[n]))
    mean, variance = tf.nn.moments(layer, axes=[0])
    bn = tf.nn.batch_normalization(layer, mean=mean, variance=variance,
                                   offset=beta, scale=gamma,
                                   variance_epsilon=1e-7)
    return activation(bn)
