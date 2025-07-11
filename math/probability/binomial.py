#!/usr/bin/env python3
"""
Binomial distribution class
"""


class Binomial:
    """
    Class representing a binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        data is a list of the data to be used to estimate the distribution
        n is the number of Bernoulli trials
        p is the probability of a “success”
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            self.n = round(mean / (1 - (variance / mean)))
            self.p = mean / self.n

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        """
        if type(k) is not int:
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        return (self.factorial(self.n) /
                (self.factorial(k) * self.factorial(self.n - k))
                ) * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        return sum(self.pmf(i) for i in range(k + 1))

    def factorial(self, x):
        """
        calculates the factorial of a number
        """
        result = 1
        for i in range(2, x + 1):
            result *= i
        return result
