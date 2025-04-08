#!/usr/bin/env python3
"""
# 0-line.py
This script demonstrates how to plot a simple line graph using matplotlib.
It generates a cubic function and plots it with a red line.
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plot a line graph using matplotlib.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    plt.xlim(0, 10)
    plt.plot(range(0, 11), y, color='red', linestyle='-')
    plt.show()
