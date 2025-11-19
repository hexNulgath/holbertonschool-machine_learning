#!/usr/bin/env python3
"""13-analyze.py"""
import pandas as pd


def analyze(df):
    """
    takes a pd.DataFrame and:

    Computes descriptive statistics for all columns
        except the Timestamp column.
    Returns a new pd.DataFrame containing these statistics.
    """
    stats = df.drop(columns=['Timestamp']).describe()
    return stats
