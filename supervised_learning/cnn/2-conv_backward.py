#!/usr/bin/env python3
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
        pad_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        pad_h, pad_w = 0, 0

    # Initialize gradients
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Pad A_prev and dA_prev
    if padding == "same":
        A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                            mode='constant')
        dA_prev_pad = np.pad(dA_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                             mode='constant')
    else:
        A_prev_pad = A_prev
        dA_prev_pad = dA_prev

    for i in range(dZ.shape[1]):  # Loop over vertical axis
        for j in range(dZ.shape[2]):  # Loop over horizontal axis
            for k in range(c_new):  # Loop over channels
                # Find the patch
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw

                # Get the patch from A_prev_pad
                a_slice = A_prev_pad[:, vert_start:vert_end,
                                     horiz_start:horiz_end, :]

                # Update gradients
                dW[:, :, :, k] += np.tensordot(a_slice,
                                               dZ[:, i, j, k], axes=([0], [0]))
                dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += \
                    W[:, :, :, k] * dZ[:, i, j, k,
                                       np.newaxis, np.newaxis, np.newaxis]

    # Remove padding if needed
    if padding == "same":
        dA_prev = dA_prev_pad[:, pad_h:dA_prev_pad.shape[1]-pad_h,
                              pad_w:dA_prev_pad.shape[2]-pad_w, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
