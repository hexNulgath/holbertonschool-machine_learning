#!/usr/bin/env python3
"""cov_backward.py"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Args:
        dZ: Gradient of the cost with respect to the output of the conv layer
             (m, h_new, w_new, c_new)
        A_prev: Input activations from previous layer
                (m, h_prev, w_prev, c_prev)
        W: Filter weights (kh, kw, c_prev, c_new)
        b: Biases (1, 1, 1, c_new)
        padding: "same" or "valid"
        stride: (sh, sw)

    Returns:
        dA_prev: Gradient w.r.t. previous layer activations
        dW: Gradient w.r.t. filters
        db: Gradient w.r.t. biases
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Calculate padding needed for "same" padding
    if padding == "same":
        pad_h = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pad_w = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        pad_h, pad_w = 0, 0
    padded_images = np.pad(
        A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant')
    # Initialize gradients
    dA_prev = np.zeros_like(padded_images)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(dZ.shape[1]):  # Loop over hical axis
        for j in range(dZ.shape[2]):  # Loop over wontal axis
            for k in range(c_new):  # Loop over channels
                # Find the patch
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw

                # Update gradients
                dW[:, :, :, k] += np.sum(
                    padded_images[:, h_start:h_end,
                                  w_start:w_end, :] *
                    dZ[:, i, j, k, np.newaxis, np.newaxis, np.newaxis], axis=0)
                dA_prev[:, h_start:h_end, w_start:w_end, :] += (
                    W[:, :, :, k] * dZ[:, i, j, k,
                                       np.newaxis, np.newaxis, np.newaxis])
    if padding == 'same':
        dA_prev = dA_prev[:, pad_h:-pad_h, pad_w:-pad_w, :]
    return dA_prev, dW, db
