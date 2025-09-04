#!/usr/bin/env python3
"""
performs forward propagation for a simple RNN
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    performs forward propagation for a simple RNN
    """
    t, m, i = X.shape
    h = np.zeros((t, m, rnn_cell.Wh.shape[1]))
    y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    h_next = h_0

    for step in range(t):
        h_next, y_next = rnn_cell.forward(h_next, X[step])
        h[step] = h_next
        y[step] = y_next

    return h, y
