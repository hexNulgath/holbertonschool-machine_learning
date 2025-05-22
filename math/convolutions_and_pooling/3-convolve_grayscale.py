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

    # Calculate padding
    if padding == 'same':
        pad_h = ((h - 1) * sh + kh - h) // 2
        pad_w = ((w - 1) * sw + kw - w) // 2
        # Compute output dimensions
        conv_h = (h + 2 * pad_h - kh) // sh + 1
        conv_w = (w + 2 * pad_w - kw) // sw + 1
    elif padding == 'valid':
        pad_h, pad_w = 0, 0
        # Compute output dimensions
        conv_h = (h - kh) // sh + 1
        conv_w = (w - kw) // sw + 1
    else:
        pad_h, pad_w = padding
        # Compute output dimensions
        conv_h = (h + 2 * pad_h - kh) // sh + 1
        conv_w = (w + 2 * pad_w - kw) // sw + 1

    # Pad images symmetrically
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant'
    )

    output = np.zeros((m, conv_h, conv_w))

    # Perform convolution
    for i in range(conv_h):
        for j in range(conv_w):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            output[:, i, j] = np.sum(
                padded_images[:, h_start:h_end, w_start:w_end] * kernel,
                axis=(1, 2)
            )
    return output
