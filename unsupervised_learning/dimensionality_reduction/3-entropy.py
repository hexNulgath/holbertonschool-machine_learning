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
    # Clip beta to prevent overflow in exp(-beta * Di)
    beta = np.clip(beta, 0, 700)  # exp(-700) is close to machine precision
    
    P = np.exp(-beta * Di)
    
    # Prevent division by zero in normalization
    sum_P = np.sum(P)
    if sum_P == 0 or sum_P < np.finfo(np.float64).tiny:
        # Return uniform distribution if all probabilities are essentially zero
        P = np.ones_like(Di) / len(Di)
        H = np.log2(len(Di))
        return H, P
    
    P = P / sum_P
    
    # Calculate entropy safely
    mask = P > np.finfo(np.float64).eps
    if np.any(mask):
        H = -np.sum(P[mask] * np.log2(P[mask]))
    else:
        H = 0.0
    
    return H, P
