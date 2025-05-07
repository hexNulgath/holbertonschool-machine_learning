#!/usr/bin/env python3
"""6-momentum.py"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    sets up the gradient descent with momentum
    optimization algorithm in TensorFlow:

    alpha is the learning rate.
    beta1 is the momentum weight.
    Returns: optimizer
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha,
                                   momentum=beta1)
