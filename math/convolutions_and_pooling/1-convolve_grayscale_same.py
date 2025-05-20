#!/usr/bin/env python3
"""1-convolve_grayscale_same"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    performs a same convolution on grayscale images:

    images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    if necessary, the image should be padded with 0’s
    Returns: a numpy.ndarray containing the convolved images
    """
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded_images = np.pad(
        images, pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    m, h, w = padded_images.shape
    conv_h = h - kh + 1
    conv_w = w - kw + 1
    output = np.zeros((m, conv_h, conv_w))

    for i in range(conv_h):
        for j in range(conv_w):
            # Extract the region from each image that aligns with the kernel
            region = padded_images[:, i:i+kh, j:j+kw]
            # Apply the kernel by element-wise multiplication and sum
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
