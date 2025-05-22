#!/usr/bin/env python3
"""3-convolve_grayscale.py"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    that performs a convolution on grayscale images:

    images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
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
    kh, kw = kernel.shape
    m, h, w = images.shape
    sh, sw = stride

    # Calculate output dimensions and padding
    if padding == 'same':
        # Calculate required padding
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1

        # Split padding equally on both sides
        pad_top = ph // 2
        pad_bottom = ph - pad_top
        pad_left = pw // 2
        pad_right = pw - pad_left
    elif padding == 'valid':
        pad_top = pad_bottom = pad_left = pad_right = ph = pw = 0
    else:  # custom padding
        ph, pw = padding
        pad_top = pad_bottom = ph
        pad_left = pad_right = pw

    # Apply padding with zeros
    padded_images = np.pad(
        images,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant')

    # Initialize output array
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, oh, ow))
    # Perform convolution
    for i in range(oh):
        for j in range(ow):
            h_start = i * sh
            w_start = j * sw
            h_end = h_start + kh
            w_end = w_start + kw
            output[:, i, j] = np.sum(
                padded_images[:, h_start:h_end, w_start:w_end] * kernel,
                axis=(1, 2)
            )

    return output
