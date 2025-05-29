#!/usr/bin/env python3
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs back propagation over a pooling layer of a neural network:

    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the output of the pooling layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c is the number of channels

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
    output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer

    kernel_shape is a tuple of (kh, kw) containing the size of the kernel
    for the pooling
        kh is the kernel height
        kw is the kernel width

    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width

    mode is a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively

    Returns: partial derivatives with respect to the previous layer (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros_like(A_prev)
    for n in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c_new):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    if mode == 'max':
                        # extract the current slice
                        a_prev_slice = A_prev[n,
                                              h_start:h_end, w_start:w_end, ch]
                        # create a boolean mask to use only max value
                        # compare max value to each value
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        # use the gradient to affect only max value
                        dA_prev[n, h_start:h_end, w_start:w_end,
                                ch] += mask * dA[n, h, w, ch]
                    elif mode == 'avg':
                        # Compute the average gradient per pixel
                        average = dA[n, h, w, ch] / (kh * kw)
                        # Distribute equally
                        dA_prev[n, h_start:h_end, w_start:w_end,
                                ch] += np.ones((kh, kw)) * average

    return dA_prev
