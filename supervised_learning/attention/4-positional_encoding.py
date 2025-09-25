#!/usr/bin/env python3
"""
Module that contains the function positional_encoding
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Function that calculates the positional encoding for a transformer
    """
    positional_encoding = np.zeros((max_seq_len, dm))
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            positional_encoding[pos, i] = np.sin(pos / (10000 ** (i / dm)))
            if i + 1 < dm:
                positional_encoding[pos, i + 1] = np.cos(
                    pos / (10000 ** ((i + 1) / dm)))
    return positional_encoding
