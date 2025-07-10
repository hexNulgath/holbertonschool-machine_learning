#!/usr/bin/env python3
"""
Represents a poisson distribution
"""


class Poisson:
    """
    Class to represent a Poisson distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Poisson distribution
        """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        res = self.e ** -self.lambtha * self.lambtha ** k
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
        return res / factorial

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        res = 0
        for i in range(k + 1):
            res += self.pmf(i)
        return res
