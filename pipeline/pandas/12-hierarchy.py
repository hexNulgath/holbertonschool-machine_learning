#!/usr/bin/env python3
""" 12-hierarchy.py """
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    takes two pd.DataFrame objects and:

    Rearranges the MultiIndex so that Timestamp is the first level.
    Concatenates the bitstamp and coinbase tables from timestamps
        1417411980 to 1417417980, inclusive.
    Adds keys to the data, labeling rows from df2 as bitstamp and
        rows from df1 as coinbase.
    Ensures the data is displayed in chronological order.
    Returns: the concatenated pd.DataFrame.
    """
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc[1417411980:1417417980]
    df1 = df1.loc[1417411980:1417417980]
    result = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    result = result.reorder_levels([1, 0]).sort_index()
    return result
