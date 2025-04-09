#!/usr/bin/env python3
"""
fruit is a matrix representing the number of fruit various people possess
The columns of fruit represent the number
of fruit Farrah, Fred, and Felicia have, respectively
The rows of fruit represent the number of apples,
bananas, oranges, and peaches, respectively
The bars should represent the number of fruit each person possesses:
The bars should be grouped by person, i.e,
the horizontal axis should have one labeled tick per person
Each fruit should be represented by a specific color:
apples = red
bananas = yellow
oranges = orange (#ff8000)
peaches = peach (#ffe5b4)
A legend should be used to indicate which fruit is represented by each color
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
    Bar plot with different colors
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    persons = ['Farrah', 'Fred', 'Felicia']

    plt.legend()
    plt.title('Number of Fruit per Person')
    plt.ylabel('Quantity of Fruit')
    plt.xticks(np.arange(3), persons)
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.bar(np.arange(3), fruit[0], color='red',
            width=0.5, label='apples')
    plt.bar(np.arange(3), fruit[1], color='yellow',
            width=0.5, label='bananas', bottom=fruit[0])
    plt.bar(np.arange(3), fruit[2], color='#ff8000',
            width=0.5, label='oranges', bottom=fruit[0] + fruit[1])
    plt.bar(np.arange(3), fruit[3], color='#ffe5b4',
            width=0.5, label='peaches', bottom=fruit[0] + fruit[1] + fruit[2])
    plt.legend(loc='upper right')
    plt.show()
