#!/usr/bin/env python3
"""
Gradient descent with dropout
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights of a neural network with Dropout regularization
    using gradient descent:

    Y: one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
        classes: number of classes
        m: number of data points
    weights: dictionary of the weights and biases of the neural network
    cache: dictionary of the outputs and dropout masks of each layer of the
    neural network
    alpha: learning rate
    keep_prob: probability that a node will be kept
    L: number of layers of the network
    """
    m = Y.shape[1]  # Number of examples
    dZ = {}  # To store gradients of each layer

    # Backpropagation through the network
    for i in range(L, 0, -1):
        if i == L:
            # Output layer gradient (assuming softmax activation)
            dZ[str(L)] = cache['A' + str(L)] - Y
        else:
            # Hidden layers gradient (assuming tanh activation)
            dA = np.matmul(weights['W' + str(i+1)].T, dZ[str(i+1)])
            # Apply dropout mask and scale by keep_prob
            if 'D' + str(i) in cache:
                dA *= cache['D' + str(i)]
                # Inverted dropout scaling
                dA /= keep_prob
            # tanh derivative
            dZ[str(i)] = dA * (1 - np.power(cache['A' + str(i)], 2))

        # Compute weight and bias gradients
        A_prev = cache['A' + str(i-1)] if i > 1 else cache['A0']
        dW = np.matmul(dZ[str(i)], A_prev.T) / m
        db = np.sum(dZ[str(i)], axis=1, keepdims=True) / m

        # Update weights
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
