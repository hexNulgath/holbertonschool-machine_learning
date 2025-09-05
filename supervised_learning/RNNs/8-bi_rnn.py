#!/usr/bin/env python3
"""bi directional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward propagation for a bidirectional RNN

    Args:
        bi_cell: instance of BidirectionalCell that will be used
        for the forward and backward propagation
        X: numpy.ndarray of shape (t, m, i) containing the input data
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: numpy.ndarray of shape (m, h) containing the initial hidden state
        for the forward direction
            h: dimensionality of the hidden state
        h_t: numpy.ndarray of shape (m, h) containing the initial hidden state
        for the backward direction

    Returns:
        H: numpy.ndarray containing all the concatenated hidden states
        from both directions, for each time step
        Y: numpy.ndarray containing all the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[-1]  # dimensionality of hidden state

    # Initialize arrays for storing hidden states
    H = np.zeros((t, m, 2 * h))

    # Initialize hidden states for forward and backward directions
    h_forward = h_0
    h_backward = h_t

    # Forward propagation through time (left to right)
    for time_step in range(t):
        x_input = X[time_step]
        h_forward = bi_cell.forward(h_forward, x_input)
        H[time_step, :, :h] = h_forward  # Store forward hidden state

    # Backward propagation through time (right to left)
    for time_step in range(t - 1, -1, -1):
        x_input = X[time_step]
        h_backward = bi_cell.backward(h_backward, x_input)
        H[time_step, :, h:] = h_backward  # Store backward hidden state

    # Compute outputs for all time steps at once
    Y = bi_cell.output(H)

    return H, Y
