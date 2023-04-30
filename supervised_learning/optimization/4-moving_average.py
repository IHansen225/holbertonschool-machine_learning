#!/usr/bin/env python3
"""
    Moving average calculation
    module.
"""
import numpy as np


def moving_average(data, beta):
    """
        Caluculates the weighted
        moving average of a data set.
    """
    avg = 0
    avg_values = []
    for i in range(len(data)):
        avg = beta * avg + (1 - beta) * data[i]
        avg_values.append(avg / (1 - beta ** (i + 1)))
    return avg_values
