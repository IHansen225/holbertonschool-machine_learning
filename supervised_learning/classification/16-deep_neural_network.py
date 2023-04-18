#!/usr/bin/python3
"""
    Deep neural network class
    file for supervised learning
    implementation.
"""
import numpy as np


class DeepNeuralNetwork():
    """
        Deep neural network class.
    """
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not len(list(filter(lambda x: x > 0, layers))) == len(layers):
            raise TypeError("layers must be a list of positive integers")
        if not len(list(filter(lambda x: isinstance(x, int), layers))) == len(layers):
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for layer in range(self.L):
            weightName = f"W{layer}"
            biasName = f"b{layer}"
            self.weights[biasName] = np.zeros((layers[layer], 1))
            self.weights[weightName] = np.random.randn(layers[layer], nx) * np.sqrt(2 / nx)
