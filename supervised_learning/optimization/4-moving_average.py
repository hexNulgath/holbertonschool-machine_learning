#!/usr/bin/env python3
"""4-moving_average.py"""


def moving_average(data, beta):
    """
    calculates the weighted moving average of a data set:
    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Returns: a list containing the moving averages of data
    """
    wma = [data[0]]
    for i in range(1, len(data)):
        wma.append(beta * data[i] + (1 - beta) * wma[i - 1])
    return wma
