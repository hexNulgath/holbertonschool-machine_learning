#!/usr/bin/env python3
"""
0-l2_reg_cost.py
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    calculates the cost of a neural network with L2 regularization:

    cost: cost of the network without L2 regularization
    lambtha: regularization parameter
    weights: dictionary of the weights and biases
            (numpy.ndarrays) of the neural network
    L: number of layers in the neural network
    m: number of data points used
    Returns: cost of the network accounting for L2 regularization
    """
    l2_cost = cost + (lambtha / (2 * m)) * sum(np.linalg.norm(
        weights['W' + str(i)], ord='fro') ** 2 for i in range(1, L + 1))
    return l2_cost
