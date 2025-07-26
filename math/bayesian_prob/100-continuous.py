#!/usr/bin/env python3
"""3-continuous.py"""
from scipy import special


# Compute the CDF at p2 and p1, then take the difference
def beta_cdf(p, a, b):
    # Regularized incomplete Beta function (CDF of Beta distribution)
    return special.betainc(a, b, p)


def posterior(x, n, P1, P2):
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
    if not isinstance(P1, float) or P1 < 0 or P1 > 1:
        raise ValueError("P1 must be a float in the range [0, 1]")
    if not isinstance(P2, float) or P2 < 0 or P2 > 1:
        raise ValueError("P2 must be a float in the range [0, 1]")
    if P2 <= P1:
        raise ValueError("P2 must be greater than P1")
    prob = beta_cdf(P2, x + 1, n - x + 1) - beta_cdf(P1, x + 1, n - x + 1)

    return prob
