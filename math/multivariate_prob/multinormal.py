#!/usr/bin/env python3
import numpy as np
"""
represents a Multivariate Normal distribution
"""


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
