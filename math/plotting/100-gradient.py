#!/usr/bin/env python3
"""
This script generates a scatter plot representing mountain elevation.
"""
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    Generates a scatter plot representing mountain elevation.
    """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    plt.ylabel('y coordinate (m)')
    plt.xlabel('x coordinate (m)')
    plt.title('Mountain Elevation')
    scatter = plt.scatter(x, y, c=z, cmap='viridis')
    cbar = plt.colorbar(scatter)
    cbar.set_label('elevation (m)')
    plt.show()
