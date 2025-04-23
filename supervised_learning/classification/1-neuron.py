#!/usr/bin/env python3
"""
0-neuron.py
Defines a class Neuron that defines a single
neuron performing binary classification
"""
import numpy as np


class Neuron():
    """
    Class Neuron that defines a single
    neuron performing binary classification
    """

    def __init__(self, nx):
        """Constructor for the Neuron class

        Args:
            nx (int): number of input features to the neuron

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector

        Returns:
            np.ndarray: weights vector for the neuron
        """
        return self.__W

    @property
    def b(self):
        """Getter for the bias

        Returns:
            int: bias for the neuron
        """
        return self.__b

    @property
    def A(self):
        """Getter for the activated output

        Returns:
            int: activated output of the neuron
        """
        return self.__A
