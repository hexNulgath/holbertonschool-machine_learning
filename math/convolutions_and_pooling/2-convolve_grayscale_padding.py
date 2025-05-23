#!/usr/bin/env python3
"""2-convolve_grayscale_padding.py"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    performs a convolution on grayscale images with custom padding:

    images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    kh, kw = kernel.shape
    ph, pw = padding

    # Height padding
    if ph:
        pad_h_l = ph
        pad_h_r = ph
    else:
        pad_h_l = 0
        pad_h_r = 0

    # Width padding
    if pw:
        pad_w_l = pw
        pad_w_r = pw
    else:
        pad_w_l = 0
        pad_w_r = 0

    padded_images = np.pad(
        images,
        pad_width=((0, 0), (pad_h_l, pad_h_r), (pad_w_l, pad_w_r)),
        mode='constant'
    )

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
