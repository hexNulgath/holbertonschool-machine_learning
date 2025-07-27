#!/usr/bin/env python3
"""
3-entropy.py
"""
import numpy as np


def HP(Di, beta):
    """
    calculates the Shannon entropy and P affinities
    relative to a data point
    """
    P = np.exp(-beta * Di)
    P /= np.sum(P)
    H = -np.sum(P * np.log2(P + 1e-18))
    return H, P
