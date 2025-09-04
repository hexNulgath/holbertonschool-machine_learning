#!/usr/bin/env python3
"""0. RNN Cell"""
import numpy as np


class RNNCell:
    """class RNNCell that represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """
        Class constructor
        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Public instance method that performs forward
        propagation for one time step
        h_prev: numpy.ndarray of shape (m, h)
        containing the previous hidden state
            m: batch size for the data
            h: dimensionality of the hidden state
        x_t: numpy.ndarray of shape (m, i)
        that contains the data input for the cell
            i: dimensionality of the data
        Returns: h_next, y
            h_next: the next hidden state
            y: the output of the cell
        """
        # Concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute the next hidden state
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        # Compute the output
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
