#!/usr/bin/env python3
"""
0-neuron.py
"""
import numpy as np


class NeuralNetwork:
    """
    Neural Network class
    """

    def __init__(self, nx, nodes):
        """
        Initialize the neural network
        :param nx: number of input features
        :param nodes: number of nodes in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

        def __init__(self, nx, nodes):
            """
            Initialize the neural network
            :param nx: number of input features
            :param nodes: number of nodes in the hidden layer
            """
            self.nx = nx
            self.nodes = nodes

    @property
    def W1(self):
        """
        Getter for W1
        :return: W1
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter for b1
        :return: b1
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter for A1
        :return: A1
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter for W2
        :return: W2
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter for b2
        :return: b2
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter for A2
        :return: A2
        """
        return self.__A2

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid activation function
        Args:
            z (np.ndarray): input to the activation function
        Returns:
            np.ndarray: activated output
        """
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        :param X: shape(nx, m) input data
            nx: number of input features
            m: number of examples
        :return: A1, A2
        """
        z = np.dot(self.W1, X) + self.b1
        self.__A1 = self.sigmoid(z)
        z = np.dot(self.W2, self.A1) + self.b2
        self.__A2 = self.sigmoid(z)
        return self.A1, self.A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        :param Y: shape(1, m) correct labels
            m: number of examples
        :param A: shape(1, m) activated output
        :return: cost
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
