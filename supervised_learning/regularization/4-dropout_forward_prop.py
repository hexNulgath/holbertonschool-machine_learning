#!/usr/bin/env python3
"""Forward propagation with dropout in a neural network"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout:

    X numpy.ndarray of shape (nx, m) containing the input data for the network
        nx is the number of input features
        m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    L the number of layers in the network
    keep_prob is the probability that a node will be kept
    Returns: a dictionary containing the outputs of each layer and
    the dropout mask used on each layer
    """
    cache = {'A0': X}
    for layer in range(1, L + 1):
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]
        Z = np.matmul(W, A_prev) + b
        if layer == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        else:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A *= D
            A /= keep_prob
            cache['D' + str(layer)] = D.astype(np.int8)
        cache['A' + str(layer)] = A
    return cache
