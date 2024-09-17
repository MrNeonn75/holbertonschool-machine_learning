#!/usr/bin/env python3
"""
Complete the following source code to plot a stacked bar graph:

fruit is a matrix representing the number of fruit various people possess
    The columns of fruit represent the number of fruit Farrah,
        Fred, and Felicia have, respectively
    The rows of fruit represent the number of apples, bananas,
        oranges, and peaches, respectively
The bars should represent the number of fruit each person possesses:
    The bars should be grouped by person, i.e, the horizontal
        axis should have one labeled tick per person
    Each fruit should be represented by a specific color:
        apples = red
        bananas = yellow
        oranges = orange (#ff8000)
        peaches = peach (#ffe5b4)
        A legend should be used to indicate which fruit is represented
            by each color
    The bars should be stacked in the same order as the rows of fruit,
        from bottom to top
    The bars should have a width of 0.5
The y-axis should be labeled Quantity of Fruit
The y-axis should range from 0 to 80 with ticks every 10 units
The title should be Number of Fruit per Person
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plot a stacked bar graph

    Args:
        None

    Return:
        None
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    rows = ('apples', 'bananas', 'oranges', 'peaches')
    columns = ('Farrah', 'Fred', 'Felicia')
    index = columns
    colors = ('red', 'yellow', '#ff8000', '#ffe5b4')
    n_rows = len(fruit)
    bar_width = 0.5
    y_offset = np.zeros(len(columns))

    for row in range(n_rows):
        plt.bar(index, fruit[row], bar_width, bottom=y_offset,
                color=colors[row], label=rows[row])
        y_offset = y_offset + fruit[row]

    plt.legend()
    plt.yticks(np.arange(0, 90, 10))
    plt.ylabel('Quantity of Fruit')
    plt.title("Number of Fruit per Person")
    plt.show()
