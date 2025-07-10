#!/usr/bin/env python3
"""
Module for the Exponential distribution class.
"""


class Exponential:
    """
    Represents an exponential distribution:
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        data is a list of the data to be used to estimate the distribution
        lambtha is the expected number of occurences in a given time frame
        """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        """
        if x < 0:
            return 0
        return self.lambtha * (self.e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        """
        if x < 0:
            return 0
        return 1 - (self.e ** (-self.lambtha * x))
