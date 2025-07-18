#!/usr/bin/env python3
"""
represents a Multivariate Normal distribution
"""
import numpy as np


class MultiNormal:
    """
    Class representing a Multivariate Normal distribution
    """
    def __init__(self, data):
        """
        data is a numpy.ndarray of shape (d, n) containing the data set:
            n is the number of data points
            d is the number of dimensions in each data point
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        n = data.shape[0]

        self.mean = np.mean(data, axis=1, keepdims=True)
        deviations = data - self.mean
        self.cov = deviations @ deviations.T / (data.shape[1] - 1)

    def pdf(self, x):
        """
        x is a numpy.ndarray of shape (d, 1)
        containing the data point whose PDF should be calculated
            d is the number of dimensions of the Multinomial instance
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.ndim != 2 or x.shape[0] != d or x.shape[1] != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        deviations = x - self.mean
        cov_inv = np.linalg.inv(self.cov)
        exponent = -0.5 * (deviations.T @ cov_inv @ deviations)
        coeff = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))
        return coeff * np.exp(exponent[0, 0])
