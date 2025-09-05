#!/usr/bin/env python3
"""
bi_forward.py
"""
import numpy as np


class BidirectionalCell:
    """
    represents a bidirectional cell of an RNN

    Attributes:
        Whf: numpy.ndarray of shape (i + h, h) with weights for the
        hidden state in the forward direction
            i: dimensionality of the data
            h: dimensionality of the hidden state
        bhf: numpy.ndarray of shape (1, h) with biases for the
        hidden state in the forward direction
        Whb: numpy.ndarray of shape (i + h, h) with weights for the
        hidden state in the backward direction
        bhb: numpy.ndarray of shape (1, h) with biases for the
        hidden state in the backward direction
        Wy: numpy.ndarray of shape (2h, o) with weights for the output
            o: dimensionality of the outputs
        by: numpy.ndarray of shape (1, o) with biases for the output
    """

    def __init__(self, i, h, o):
        """
        Class constructor
        """
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the
            previous hidden state
                m: batch size
                h: dimensionality of the hidden state
            x_t: numpy.ndarray of shape (m, i) containing the data
            input for the cell
                i: dimensionality of the data

        Returns:
            h_next: next hidden state
            y: output of the cell
        """
        # Concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state using tanh activation function
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        performs backward propagation for one time step

        Args:
            h_next: numpy.ndarray of shape (m, h) containing the
            next hidden state
                m: batch size
                h: dimensionality of the hidden state
            x_t: numpy.ndarray of shape (m, i) containing the data
            input for the cell
                i: dimensionality of the data

        Returns:
            h_prev: previous hidden state
        """
        # Concatenate h_next and x_t
        concat = np.concatenate((h_next, x_t), axis=1)

        # Compute previous hidden state using tanh activation function
        h_prev = np.tanh(np.matmul(concat, self.Whb) + self.bhb)

        return h_prev
