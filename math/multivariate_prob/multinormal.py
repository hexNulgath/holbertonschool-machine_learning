#!/usr/bin/env python3
import numpy as np
"""
represents a Multivariate Normal distribution
"""


class MultiNormal:

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

        self.mean, self.cov = self.mean_cov(data)

    @staticmethod
    def mean_cov(data):
        """
        calculates the mean and covariance of the data set
        """
        mean_cov = __import__('0-mean_cov').mean_cov
        return mean_cov(data)
