#!/usr/bin/env python3
"""Final corrected convolution operation for grayscale images"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with given parameters.

    Args:
        images: numpy.ndarray with shape (m, h, w) containing grayscale images
        kernel: numpy.ndarray with shape (kh, kw) containing convolution kernel
        padding: either 'same', 'valid', or tuple (ph, pw)
        stride: tuple (sh, sw) specifying stride steps

    Returns:
        numpy.ndarray containing convolved images
    """
    kh, kw = kernel.shape
    m, h, w = images.shape
    sh, sw = stride

    # Calculate padding
    if padding == 'same':
        # Calculate output dimensions
        out_h = int(np.ceil(h / sh))
        out_w = int(np.ceil(w / sw))

        # Calculate required padding
        pad_h = max((out_h - 1) * sh + kh - h, 0)
        pad_w = max((out_w - 1) * sw + kw - w, 0)

        # Split padding equally on both sides
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding with zero values
        padded_images = np.pad(
            images,
            pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )
    elif padding == 'valid':
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        padded_images = images
    else:
        ph, pw = padding
        padded_images = np.pad(
            images,
            pad_width=((0, 0), (ph, ph), (pw, pw)),
            mode='constant',
            constant_values=0
        )
        pad_top, pad_bottom = ph, ph
        pad_left, pad_right = pw, pw

    # Calculate output dimensions
    out_h = (h + pad_top + pad_bottom - kh) // sh + 1
    out_w = (w + pad_left + pad_right - kw) // sw + 1

    # Initialize output array
    output = np.zeros((m, out_h, out_w))

    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            w_start = j * sw
            h_end = h_start + kh
            w_end = w_start + kw
            output[:, i, j] = np.sum(
                padded_images[:, h_start:h_end, w_start:w_end] * kernel,
                axis=(1, 2)
            )

    return output
