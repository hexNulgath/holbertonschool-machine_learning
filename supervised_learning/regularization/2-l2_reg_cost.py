#!/usr/bin/env python3
"""2-l2_reg_cost.py"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    calculates the cost of a neural network with L2 regularization:

    cost: tensor containing the cost of the network without L2 regularization
    model: Keras model that includes layers with L2 regularization
    Returns: tensor containing the total cost for each layer of the network,
        accounting for L2 regularization
    """
    l2_costs = []

    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.InputLayer):
            # Get the L2 regularization loss for this layer
            layer_l2 = tf.reduce_sum(layer.losses) + cost
            l2_costs.append(layer_l2)

    # Return as tensor
    return tf.convert_to_tensor(l2_costs)
