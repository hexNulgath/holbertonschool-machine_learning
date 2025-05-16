#!/usr/bin/env python3
"""2-l2_reg_cost.py"""
import numpy as np
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    calculates the cost of a neural network with L2 regularization:

    cost: tensor containing the cost of the network without L2 regularization
    model: Keras model that includes layers with L2 regularization
    Returns: tensor containing the total cost for each layer of the network,
        accounting for L2 regularization
    """
    total_loss = cost + tf.add_n(model.losses)

    return total_loss
