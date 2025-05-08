#!/usr/bin/env python3
"""12-learning_rate_decay.py"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    creates a learning rate decay operation in tensorflow
    using inverse time decay:

    alpha: the original learning rate
    decay_rate: the weight used to determine the rate at which alpha will decay
    decay_step: the number of passes of gradient descent that should occur
        before alpha is decayed further
    Returns: the learning rate decay operation
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)
