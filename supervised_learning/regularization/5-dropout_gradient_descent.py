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
    A_output = cache['A' + str(L)]
    dZ = A_output - Y  # Softmax derivative (cross-entropy loss)
    grads = {}

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i-1)]
        grads['dW' + str(i)] = np.dot(dZ, A_prev.T) / m
        grads['db' + str(i)] = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:  # Not input layer
            dA = np.matmul(weights['W' + str(i)].T, dZ)
            dA *= cache['D' + str(i-1)]  # Apply mask
            dA /= keep_prob  # Scale to maintain expected value
            dZ = dA * (1 - np.power(A_prev, 2))  # Tanh derivative (1 - tanh²)

        # Update weights (learning rate applied here)
        weights['W' + str(i)] -= alpha * grads['dW' + str(i)]
        weights['b' + str(i)] -= alpha * grads['db' + str(i)]
