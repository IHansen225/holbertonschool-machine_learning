#!/usr/bin/env python3
"""
    Neuron class file for classification
    algorithm exercises
"""
import numpy as np


class Neuron():
    """
        Neuron class object
    """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
            Returns weight value of the
            corresponding neuron
        """
        return self.__W

    @property
    def b(self):
        """
            Returns bias value of the
            corresponding neuron
        """
        return self.__b

    @property
    def A(self):
        """
            Returns activated value of the
            corresponding neuron
        """
        return self.__A

    def forward_prop(self, X):
        """
            Calculates the forward propagation
            of the corresponding neuron
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A
