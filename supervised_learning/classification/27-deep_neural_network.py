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
    def activation(z, formula='sigmoid'):
        """
        calculates the sigmoid activation function
        """
        if formula == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif formula == 'softmax':
            e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            return e_z / np.sum(e_z, axis=0, keepdims=True)

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neural network
        x shape (nx, m) input data
        m is the number of examples
        nx is the number of input features
        """
        self.__cache = {'A0': X}

        for i in range(1, self.L):
            # get current layer weights and biases
            W = self.weights['W' + str(i)]
            b = self.weights['b' + str(i)]
            # get input to the neurons
            A_prev = self.cache['A' + str(i - 1)]
            # calculate the activation of the neurons
            Z = np.dot(W, A_prev) + b
            # apply the activation function
            A = self.activation(Z)
            self.__cache['A' + str(i)] = A
        W_last = self.weights['W' + str(self.L)]
        b_last = self.weights['b' + str(self.L)]
        A_prev = self.cache['A' + str(self.L - 1)]
        Z = np.dot(W_last, A_prev) + b_last
        A = self.activation(Z, 'softmax')
        self.__cache['A' + str(self.L)] = A
        return A, self.cache

    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression
        Y one-hot numpy.ndarray of shape (classes, m)
            that contains the correct labels for the input data
        A one-hot numpy.ndarray of shape (classes, m)
            containing the activated output of the neuron for each example
        """
        # Calculate binary cross-entropy loss for each class and sum
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        evaluates the neural network's predictions
        X numpy.ndarray with shape (nx, m)
            that contains the input data
        Y one-hot numpy.ndarray of shape (classes, m)
            that contains the correct labels for the input data
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.argmax(A, axis=0).reshape(1, -1)
        prediction = np.where(prediction > 1, 1, 0)
        prediction = prediction.flatten()
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        calculates one pass of gradient descent on the neural network
        Y numpy.ndarray with shape (1, m)
            that contains the correct labels for the input data
        cache is a dictionary containing all the
            intermediary values of the network
        alpha is the learning rate
        """
        m = Y.shape[1]
        # get the last layer activation
        A = cache['A' + str(self.L)]
        # get the diferences between the activation and the labels
        dZ = A - Y
        # loop through the layers in reverse order
        for i in range(self.L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            W = self.weights['W' + str(i)]
            # calculate the gradients to weights and biases
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ = np.dot(W.T, dZ) * (A_prev * (1 - A_prev))
            self.weights['W' + str(i)] -= alpha * dW
            self.weights['b' + str(i)] -= alpha * db

    @staticmethod
    def plot_graph(cost, step):
        """
        plots the cost of the model
        cost is a list of the costs of the model
        step is the period of time
            between each graph point and printed information
        """
        import matplotlib.pyplot as plt
        steps = np.arange(0, len(cost) * step, step)
        plt.plot(steps, cost)
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.title('Training Cost')
        plt.show()

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        trains the deep neural network
        X numpy.ndarray with shape (nx, m)
            that contains the input data
            nx is the number of input features
            m is the number of examples
        Y numpy.ndarray with shape (1, m)
            that contains the correct labels for the input data
        iterations is the number of iterations to train
        alpha is the learning rate
        verbose is a boolean that determines
            if the training information should be printed
        graph is a boolean that determines
            if the training information should be plotted
        step is the period of time
            between each graph point and printed information
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        Cost_list = []
        for i in range(iterations):
            self.forward_prop(X)
            if i % step == 0 and verbose:
                cost = self.cost(Y, self.cache['A' + str(self.L)])
                if graph:
                    Cost_list.append(cost)
                print("Cost after {} iterations: {}".format(i, cost))
            self.gradient_descent(Y, self.cache, alpha)
        cost = self.cost(Y, self.cache['A' + str(self.L)])
        if verbose:
            print("Cost after {} iterations: {}".format(iterations, cost))
        if graph:
            Cost_list.append(cost)
            self.plot_graph(Cost_list, step)
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        saves the instance of DeepNeuralNetwork to a file
        filename is the file to save the instance to
        """
        import pickle
        if filename[-4:] != '.pkl':
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(filename):
        """
        loads a DeepNeuralNetwork instance from a file
        filename is the file to load the instance from
        """
        import pickle
        try:
            with open(filename, '+rb') as f:
                return pickle.load(f)
        except Exception:
            return None
