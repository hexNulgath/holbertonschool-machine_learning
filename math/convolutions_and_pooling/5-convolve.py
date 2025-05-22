#!/usr/bin/env python3
"""5-convolve.py"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    performs a convolution on images using multiple kernels:

    images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernels is a numpy.ndarray with shape (kh, kw, c, nc)
    containing the kernels for the convolution
        kh is the height of a kernel
        kw is the width of a kernel
        nc is the number of kernels
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image

    Returns: a numpy.ndarray containing the convolved images
    """
    kh, kw, kc, nc = kernels.shape
    m, h, w, c = images.shape
    sh, sw = stride
    # Validate that kernel channels match image channels
    if kc != c:
        raise ValueError("Kernel channels must match image channels.")

    # Calculate output dimensions and padding
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = pw = 0
    else:  # custom padding
        ph, pw = padding

    # Apply padding with zeros
    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant')

    # Initialize output array
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, oh, ow, nc))

    # Perform convolution
    kernels = kernels.reshape((1, *kernels.shape))
    for i in range(oh):
        for j in range(ow):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            output[:, i, j, :] = np.sum(
                padded_images[:, h_start:h_end, w_start:w_end,
                              :, None] * kernels,
                axis=(1, 2, 3)
            )
    return output
