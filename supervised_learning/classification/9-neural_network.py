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
