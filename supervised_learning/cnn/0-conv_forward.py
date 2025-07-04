#!/usr/bin/env python3
"""conv_forward"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional
    layer of a neural network:

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer

    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
    containing the kernels for the convolution
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output

    b is a numpy.ndarray of shape (1, 1, 1, c_new)
    containing the biases applied to the convolution

    activation is an activation function applied to the convolution

    padding is a string that is either same or valid
    indicating the type of padding used

    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width

    Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Calculate padding
    if padding == 'valid':
        ph, pw = 0, 0
    else:
        ph = (((h_prev - 1) * sh + kh - h_prev) // 2)
        pw = (((w_prev - 1) * sw + kw - w_prev) // 2)

    p_input = np.pad(A_prev,
                     ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant')

    # Calculate output dimensions
    oh = (h_prev + 2 * ph - kh) // sh + 1
    ow = (w_prev + 2 * pw - kw) // sw + 1
    conv_layer = np.zeros((m, oh, ow, c_new))

    # Reshape W shape(1, kh, kw, c_prev, c_new)
    W_reshaped = W[np.newaxis, ...]

    # Perform convolution
    for i in range(oh):
        for j in range(ow):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            # Extract patch and perform convolution
            # patch: (m, kh, kw, c, 1)
            # kernels: (1, kh, kw, c, c_new)
            # output: (m, kh, kw, c, c_new)
            patch = p_input[:, h_start:h_end, w_start:w_end, :, np.newaxis]
            conv_layer[:, i, j, :] = activation(
                np.sum(patch * W_reshaped, axis=(1, 2, 3)) + b
            )

    return conv_layer
