#!/usr/bin/env python3
"""
Module for the Binomial distribution class.
"""


class Normal:
    """
    Represents a normal distribution
    """
    e = 2.7182818285
    π = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        data is a list of the data to be used to estimate the distribution
        mean is the mean of the distribution
        stddev is the standard deviation of the distribution
        """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            # Variance
            self.variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.variance_sample = sum(
                (x - self.mean) ** 2 for x in data) / (len(data) - 1)

            # Standard Deviation
            self.stddev = self.variance ** 0.5
        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """
        Calculates the z-score of a given x value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x value of a given z-score
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x value
        """
        return (1 / (
            self.stddev * (2 * self.π) ** 0.5)
            ) * self.e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x value
        """
        z = self.z_score(x)
        z = z / (2 ** 0.5)
        erf = (2 / (self.π ** 0.5)) * (
            z - z ** 3 / 3 + z ** 5 / 10 - z ** 7 / 42 + z ** 9 / 216)
        return 0.5 * (1 + erf)
