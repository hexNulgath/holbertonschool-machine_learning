#!/usr/bin/env python3
"""2-from_file.py"""
import pandas as pd


def from_file(filename, delimiter):
    """
    loads data from a file as a pd.DataFrame:

    filename is the file to load from
    delimiter is the column separator
    Returns: the loaded pd.DataFrame
    """

    df = pd.read_csv(filename, delimiter=delimiter)
    return df
