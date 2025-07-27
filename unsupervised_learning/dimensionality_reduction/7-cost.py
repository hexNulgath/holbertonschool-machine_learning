#!/usr/bin/env python3
"""
7-cost.py
"""
import numpy as np


def cost(P, Q):
    """
    calculates the cost of the t-SNE transformation
    """
    # Avoid division by zero by clipping both P and Q
    P_clipped = np.maximum(P, 1e-12)
    Q_clipped = np.maximum(Q, 1e-12)
    
    cost_val = np.sum(P * np.log(P_clipped / Q_clipped))

    return cost_val
