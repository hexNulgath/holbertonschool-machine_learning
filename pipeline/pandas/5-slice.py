#!/usr/bin/env python3
"""5-slice.py"""


def slice(df):
    """
    takes a pd.DataFrame and:

        Extracts the columns High, Low, Close, and Volume_BTC.
        Selects every 60th row from these columns.
        Returns: the sliced pd.DataFrame.
    """
    df = df.loc[:, ["High", "Low", "Close", "Volume_(BTC)"]]
    df = df.iloc[::60]

    return df
