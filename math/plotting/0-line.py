#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
    Complete the following source code to plot y as a line graph:

    y should be plotted as a solid red line
    The x-axis should range from 0 to 10
"""


def line():
    """
    To plot y as a line graph:
    """
    y = np.arange(0, 11) ** 3

    plt.xlim(0, 10)
    plt.plot(y, color='red')
    plt.show()
