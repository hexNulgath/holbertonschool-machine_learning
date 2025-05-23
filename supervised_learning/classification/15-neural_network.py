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

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        :param X: shape(nx, m) input data
            nx: number of input features
            m: number of examples
        :param Y: shape(1, m) correct labels
            m: number of examples
        :return: prediction, cost
            prediction: predicted labels
            cost: cost of the model
        """
        self.forward_prop(X)
        prediction = np.where(self.A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        :param X: shape(nx, m) input data
            nx: number of input features
            m: number of examples
        :param Y: shape(1, m) correct labels
            m: number of examples
        :param A1: shape(nodes, m) activated output of the hidden layer
        :param A2: shape(1, m) predicted output of the output layer
        :param alpha: learning rate
        """
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.matmul(self.W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        # Update weight and bias
        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neural network
        :param X: shape(nx, m) input data
            nx: number of input features
            m: number of examples
        :param Y: shape(1, m) correct labels
            m: number of examples
        :param iterations: number of iterations to train the model
        :param alpha: learning rate
        :param verbose: whether to print progress during training
        :param graph: whether to plot the cost graph
        :param step: interval for printing progress
        :return: evaluation of the training data after training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            cumulative_cost = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            if verbose and i % step == 0:
                cost = self.cost(Y, self.A2)
                if graph:
                    cumulative_cost.append(cost)
                print(f"Cost after {i} iterations: {cost}")
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
        if graph:
            self.graph_cost(cumulative_cost, step)
        if verbose and i % step != 0:
            cost = self.cost(Y, self.A2)
            print(f"Cost after {i} iterations: {cost}")
        return self.evaluate(X, Y)

    @staticmethod
    def graph_cost(cost, step):
        """
        Prints the cost at each step
        :param cost: cost of the model
        :param step: interval for printing progress
        """
        import matplotlib.pyplot as plt
        plt.plot(np.arange(0, len(cost) * step, step), cost)
        plt.title("Training Cost")
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.show()
