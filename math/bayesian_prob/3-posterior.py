#!/usr/bin/env python3
"""2-marginal.py"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects.
    """
    likelihoods = np.zeros(P.shape)
    for i, p in enumerate(P):
        likelihoods[i] = (np.math.comb(n, x) * (p ** x) * ((1 - p) ** (n - x)))
    return likelihoods


def intersection(x, n, P, Pr):
    """
    calculates the intersection of obtaining this
    data with the various hypothetical probabilities
    """
    likelihoods = likelihood(x, n, P)
    posterior = likelihoods * Pr
    return posterior


def marginal(x, n, P, Pr):
    """
    calculates the marginal probability of obtaining the data
    """
    intersection_values = intersection(x, n, P, Pr)
    marginal_value = np.sum(intersection_values)
    return marginal_value


def posterior(x, n, P, Pr):
    """
     calculates the posterior probability for the various
     hypothetical probabilities of developing severe side
     effects given the data
     """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray):
        raise TypeError("P must be a 1D numpy.ndarray")
    if P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    val_marginal = marginal(x, n, P, Pr)
    val_likelihood = likelihood(x, n, P)
    val_posterior = (val_likelihood * Pr) / val_marginal
    return val_posterior
