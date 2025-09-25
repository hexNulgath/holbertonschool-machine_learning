#!/usr/bin/env python3
"""
Module that contains the function positional_encoding
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Function that calculates the positional encoding for a transformer
    """
    PE = np.zeros((max_seq_len, dm))
    position = np.arange(0, max_seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))
    # add sin to even indices in the array
    PE[:, 0::2] = np.sin(position * div_term)
    # add cos to odd indices in the array
    PE[:, 1::2] = np.cos(position * div_term)
    return PE
