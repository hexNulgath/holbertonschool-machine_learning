#!/usr/bin/env python3
"""
Logarithmic Scale Visualization Module

This module creates a line graph with a logarithmic y-axis to visualize
the exponential decay of Carbon-14,
a concept fundamental to radiocarbon dating.
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Creates and displays a line graph of
    exponential decay with a logarithmic y-axis.

    This function plots the exponential decay of Carbon-14 over time.
    The y-axis uses a logarithmic scale to better visualize the decay pattern
    that occurs over an extended timeframe (28,650 years).

    Args:
        None

    Returns:
        None: Displays the plot directly
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of C-14")
    plt.yscale('log')
    plt.xlim(0, 28650)
    plt.plot(x, y)
    plt.show()
