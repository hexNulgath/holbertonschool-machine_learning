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
    grads = {}
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        # Compute gradients for current layer
        grads['dW' + str(i)] = (dZ @ cache['A' + str(i - 1)].T) / \
            m + (lambtha / m) * weights['W' + str(i)]
        grads['db' + str(i)] = dZ.sum(axis=1, keepdims=True) / m
        if i > 1:  # Propagate gradient through tanh
            dA_prev = weights[f"W{i}"].T @ dZ
            dZ = dA_prev * (1 - cache['A' + str(i - 1)] ** 2)

    # Update weights and biases (in-place)
    for i in range(1, L + 1):
        weights[f"W{i}"] -= alpha * grads[f"dW{i}"]
        weights[f"b{i}"] -= alpha * grads[f"db{i}"]

    return weights
