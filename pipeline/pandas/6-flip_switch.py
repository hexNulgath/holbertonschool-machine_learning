#!/usr/bin/env python3
""" 6-flip_switch.py """


def flip_switch(df):
    """
    takes a pd.DataFrame and:

    Sorts the data in reverse chronological order.
    Transposes the sorted dataframe.
    Returns: the transformed pd.DataFrame.
    """
    df = df.iloc[::-1]
    df = df.transpose()
    return df
