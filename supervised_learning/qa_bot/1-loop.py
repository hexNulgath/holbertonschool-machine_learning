#!/usr/bin/env python
"""1-loop.py"""

loop = True
exit = ['exit', 'quit', 'goodbye', 'bye']
while loop:
    x = input("Q: ")
    x = x.lower()
    if x in exit:
        print("A: Goodbye!")
        loop = False
    else:
        print("A: ")
