#!/usr/bin/env python3
"""Create a pd.DataFrame from a np.ndarray."""
import pandas as pd


def from_numpy(array):
    """
    array is the np.ndarray from which you should create the pd.DataFrame
    The columns of the pd.DataFrame should be labeled in alphabetical order
    and capitalized. There will not be more than 26 columns.
    Returns: the newly created pd.DataFrame
    """
    df = pd.DataFrame(array)
    df.columns = [chr(i) for i in range(65, 65 + df.shape[1])]
    return df
