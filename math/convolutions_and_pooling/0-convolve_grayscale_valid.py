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
    m, h, w = images.shape
    kh, kw = kernel.shape
    conv_h = h - kh + 1
    conv_w = w - kw + 1
    output = np.zeros((m, conv_h, conv_w))

    for i in range(conv_h):
        for j in range(conv_w):
            # Extract the region from each image that aligns with the kernel
            region = images[:, i:i+kh, j:j+kw]
            # Apply the kernel by element-wise multiplication and sum
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
