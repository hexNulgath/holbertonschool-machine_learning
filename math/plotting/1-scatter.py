#!/usr/bin/env python3
"""
Scatter Plot Visualization Module

This module generates a scatter plot of men's height vs weight
using randomly generated data from a multivariate normal distribution.
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Creates and displays a scatter plot of men's height vs weight.

    The function generates 2000 data points
    from a multivariate normal distribution
    with a specified mean and covariance to represent the relationship between
    men's heights (in inches) and weights (in pounds).

    Args:
        None

    Returns:
        None: Displays the scatter plot directly
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")
    plt.title("Men's Height vs Weight")
    plt.scatter(x, y, color="magenta")
    plt.show()
