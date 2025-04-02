#!/usr/bin/env python3
"""
This script demonstrates how to add two arrays.
"""


def add_arrays(arr1, arr2):
    if len(arr1) != len(arr2):
        return None
    response = []
    for i in range(len(arr1)):
        new_value = arr1[i] + arr2[i]
        response.append(new_value)
    return response
