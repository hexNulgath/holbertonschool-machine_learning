#!/usr/bin/env python3
"""
GRU Cell
"""
import numpy as np


class GRUCell:
    """
    Class that represents a GRU cell
    """

    def __init__(self, i, h, o):
        """
        Constructor method

        Args:
            i: The dimensionality of the data
            h: The dimensionality of the hidden state
            o: The dimensionality of the outputs
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))
        self.h = h
        self.i = i
        self.o = o

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function

        Args:
            x: numpy.ndarray of shape (m, n) to calculate the sigmoid function
                m: number of examples
                n: number of classes

        Returns:
            A numpy.ndarray of shape (m, n) containing the sigmoid
            for each example
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        Softmax activation function

        Args:
            x: numpy.ndarray of shape (m, n) to calculate the softmax function
                m: number of examples
                n: number of classes

        Returns:
            A numpy.ndarray of shape (m, n) containing the softmax
            for each example
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the
            previous hidden state
                m: number of examples
                h: dimensionality of the hidden state
            x_t: numpy.ndarray of shape (m, i) that contains the data
            input for the cell
                m: number of examples
                i: dimensionality of the data

        Returns:
            h_next: The next hidden state
            y: The output of the cell
        """
        m = x_t.shape[0]

        # Concatenate h_prev and x_t
        concat_hx = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z_t = self.sigmoid(np.dot(concat_hx, self.Wz) + self.bz)

        # Reset gate
        r_t = self.sigmoid(np.dot(concat_hx, self.Wr) + self.br)

        # Candidate hidden state
        concat_rhx = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_hat = np.tanh(np.dot(concat_rhx, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_hat

        # Output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y
