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
        Calculates the forward propagation of the neuron
        Args:
            X (np.ndarray): shape (nx, m)
                nx: number of input features to the neuron
                m: number of examples
        Returns:
            the output of the neuron after activation
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = self.sigmoid(Z)
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Args:
            Y (np.ndarray): shape (1, m) contains the correct labels
                            for the input data
            A (np.ndarray): shape (1, m) contains the activated
                            output of the neuron for each example
        Returns:
            float: cost of the model
            AKA how wrong is the prediction
        """
        m = Y.shape[1]
        total_cost = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = - (1 / m) * total_cost
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        Args:
            X (np.ndarray): shape (nx, m)
                nx: number of input features to the neuron
                m: number of examples
            Y (np.ndarray): shape (1, m) contains the correct labels
                            for the input data
        Returns:
            np.ndarray: the neuron’s prediction labels for
                        each example in X labelled 1 if the output
                        of the neuron is >= 0.5 and 0 otherwise
            float: total cost of the network
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        Args:
            X (np.ndarray): shape (nx, m)
                nx: number of input features to the neuron
                m: number of examples
            Y (np.ndarray): shape (1, m) contains the correct labels
                            for the input data
            A (np.ndarray): shape (1, m) contains the activated
                            output of the neuron for each example
            alpha (float): learning rate
        """
        m = Y.shape[1]
        # gradient of the cost with respect to A
        dz = A - Y
        # gradient of the cost with respect to W
        dw = np.squeeze((1 / m) * np.dot(X, dz.T))
        # gradient of the cost with respect to b
        db = np.squeeze((1 / m) * np.sum(dz))
        # update the weights and bias
        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron
        Args:
            X (np.ndarray): shape (nx, m)
                nx: number of input features to the neuron
                m: number of examples
            Y (np.ndarray): shape (1, m) contains the correct labels
                            for the input data
            iterations (int): number of iterations to train over
            alpha (float): learning rate
        Returns:
            np.ndarray: the neuron’s prediction labels for
                        each example in X labelled 1 if the output
                        of the neuron is >= 0.5 and 0 otherwise
            float: total cost of the network after training
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, self.cost(Y, A)
