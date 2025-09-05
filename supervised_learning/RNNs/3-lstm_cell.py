#!/usr/bin/env python3
"""Module that contains the LSTMCell class"""
import numpy as np


class LSTMCell:
    """
    Class that represents an LSTM cell
    """
    def __init__(self, i, h, o):
        """
        Initialize the LSTM cell
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        Softmax activation function
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """
        Perform forward propagation for one time step
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f_t = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        u_t = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        c_next = f_t * c_prev + u_t * c_tilde
        o_t = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)
        h_next = o_t * np.tanh(c_next)
        y_t = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y_t
