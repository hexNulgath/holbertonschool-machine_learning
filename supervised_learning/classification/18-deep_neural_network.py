#!/usr/bin/env python3
"""
deep neural network performing binary classification
"""
import numpy as np


class DeepNeuralNetwork():
    """
    nx is the number of input features
    layers is a list representing the number of
    nodes in each layer of the network
    """
    def __init__(self, nx, layers):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            #   if first layer use inputs else use node in hidden layer
            if i == 0:
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
        getter for L
        """
        return self.__L

    @property
    def cache(self):
        """
        getter for cache
        """
        return self.__cache

    @property
    def weights(self):
        """
        getter for weights
        """
        return self.__weights

    @staticmethod
    def activation(z):
        """
        calculates the sigmoid activation function
        """
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neural network
        x shape (nx, m) input data
        m is the number of examples
        nx is the number of input features
        """
        self.__cache['A0'] = X
        for i in range(self.L):
            # get current layer weights and biases
            W = self.weights['W' + str(i + 1)]
            b = self.weights['b' + str(i + 1)]
            # get input to the neurons
            A_prev = self.cache['A' + str(i)]
            # calculate the activation of the neurons
            Z = np.dot(W, A_prev) + b
            # apply the activation function
            A = self.activation(Z)
            self.__cache['A' + str(i + 1)] = A
        return A, self.cache
