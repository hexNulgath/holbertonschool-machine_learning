#!/usr/bin/env python3
"""3-continuous.py"""
from scipy import special


# Compute the CDF at p2 and p1, then take the difference
def beta_cdf(p, a, b):
    # Regularized incomplete Beta function (CDF of Beta distribution)
    return special.betainc(a, b, p)


def posterior(x, n, p1, p2):
    """
     calculates the posterior probability for the various
     hypothetical probabilities of developing severe side
     effects given the data
     """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    prob = beta_cdf(p2, x + 1, n - x + 1) - beta_cdf(p1, x + 1, n - x + 1)

    return prob
