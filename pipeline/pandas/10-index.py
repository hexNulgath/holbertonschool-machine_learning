#!/usr/bin/env python3
""" 10-index.py """


def index(df):
    """
    takes a pd.DataFrame and:

    Sets the Timestamp column as the index of the dataframe.
    Returns: the modified pd.DataFrame.
    """
    df = df.set_index('Timestamp')
    return df
