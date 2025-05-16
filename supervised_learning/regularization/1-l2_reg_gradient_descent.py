#!/usr/bin/env python3
"""
1-l2_reg_gradient_descent.py
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network using gradient
    descent with L2 regularization:

    Y(classes, m) that contains the correct labels for the data
        classes: the number of classes
        m: the number of data points
    weights: a dictionary of the weights and biases of the neural network
    cache: a dictionary of the outputs of each layer of the neural network
    alpha: the learning rate
    lambtha: the L2 regularization parameter
    L: the number of layers of the network
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    dW = (1 / m) * (dZ @ cache['A' + str(L - 1)].T) + \
        (lambtha / m) * weights['W' + str(L)]
    db = (1 / m) * dZ.sum(axis=1, keepdims=True)
    weights['W' + str(L)] -= alpha * dW
    weights['b' + str(L)] -= alpha * db

    for i in range(L - 1, 0, -1):
        dZ = (weights['W' + str(i + 1)].T @ dZ) * \
            (cache['A' + str(i)] * (1 - cache['A' + str(i)]))
        dW = (1 / m) * (dZ @ cache['A' + str(i - 1)].T) + \
            (lambtha / m) * weights['W' + str(i)]
        db = (1 / m) * dZ.sum(axis=1, keepdims=True)
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
    return weights
