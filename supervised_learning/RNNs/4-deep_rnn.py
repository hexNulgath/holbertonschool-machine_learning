#!/usr/bin/env python3
"""deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, H_0):
    """
    performs forward propagation for a deep RNN

    Args:
        rnn_cells: list of RNNCell instances for each layer
        X: numpy.ndarray of shape (t, m, i) containing the input data
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        H_0: numpy.ndarray of shape (l, m, h) containing
        the initial hidden state
            l: number of layers
            h: dimensionality of the hidden state

    Returns:
        H: numpy.ndarray containing all the hidden states for each
        time step and layer
        Y: numpy.ndarray containing all the outputs for each time step
    """
    t, m, i = X.shape
    le = len(rnn_cells)
    h = H_0.shape[-1]  # dimensionality of hidden state

    # Initialize arrays for storing hidden states and outputs
    H = np.zeros((t + 1, le, m, h))
    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[1]))

    # Set initial hidden state
    H[0] = H_0

    # Forward propagation through time
    for time_step in range(t):
        # Process through each layer
        for layer in range(le):
            # Get the appropriate input for this layer
            if layer == 0:
                # First layer uses the input data
                x_input = X[time_step]
            else:
                # Subsequent layers use the hidden state from previous layer
                x_input = H[time_step + 1, layer - 1]

            # Get the previous hidden state for this layer
            h_prev = H[time_step, layer]

            # Forward pass through the current RNN cell
            h_next, y_output = rnn_cells[layer].forward(h_prev, x_input)

            # Store the hidden state for the next time step
            H[time_step + 1, layer] = h_next

            # If this is the last layer, store the output
            if layer == le - 1:
                Y[time_step] = y_output

    return H, Y
