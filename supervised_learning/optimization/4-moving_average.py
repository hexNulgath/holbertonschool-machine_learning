#!/usr/bin/env python3
"""4-moving_average.py"""


def moving_average(data, beta):
    """
    calculates the weighted moving average of a data set:
    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Returns: a list containing the moving averages of data
    """
    average = 0
    moving_averages = []
    for i, data in enumerate(data):
        average = beta * average + (1 - beta) * data
        # Apply bias correction
        average_corrected = average / (1 - beta ** (i + 1))
        moving_averages.append(average_corrected)
    return moving_averages
