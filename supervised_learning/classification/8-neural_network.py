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
        if nx < 0:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.normal(0, 1, (nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(0, 1, (1, nodes))
        self.b2 = 0
        self.A2 = 0

        def __init__(self, nx, nodes):
            """
            Initialize the neural network
            :param nx: number of input features
            :param nodes: number of nodes in the hidden layer
            """
            if not isinstance(nx, int):
                raise TypeError("nx must be an integer")
            if nx < 0:
                raise ValueError("nx must be a positive integer")
            if not isinstance(nodes, int):
                raise TypeError("nodes must be an integer")
            if nodes < 1:
                raise ValueError("nodes must be a positive integer")
            self.nx = nx
            self.nodes = nodes
