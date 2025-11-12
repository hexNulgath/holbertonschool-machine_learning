#!/usr/bin/env python3
"""policy gradient"""
import numpy as np


def policy(matrix, weight):
    """
    computes the policy with a weight of a matrix
    """
    z = matrix @ weight
    exp = np.exp(z - np.max(z))
    policy = exp / np.sum(exp)
    return policy
