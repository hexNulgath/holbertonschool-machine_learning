#!/usr/bin/env python3
"""0-convolve_grayscale_valid.py"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performs a valid convolution on grayscale images:

    images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    Returns: a numpy.ndarray containing the convolved images
    """
    m, n = kernel.shape
    if m != n:
        raise ValueError("Kernel must be square (m == n)")

    z, y, x = images.shape
    y_out = y - m + 1
    x_out = x - m + 1
    new_image = np.zeros((z, y_out, x_out))

    for j in range(z):  # Loop over each image in the batch
        for i in range(y_out):  # Loop over rows
            # Extract all possible (m, m) patches in the current row
            patches = np.lib.stride_tricks.sliding_window_view(
                images[j], (m, m))[i, :x_out]
            # Compute the sum of element-wise multiplication with the kernel
            new_image[j, i] = np.sum(patches * kernel, axis=(1, 2))

    return new_image
