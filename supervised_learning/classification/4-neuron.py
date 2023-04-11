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
        # nx is the number of input features to the neuron
        # input features = neurons in the previous layer
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # W initialized using a random normal distribution
        # because it will be later modified by gradient descent
        self.__W = np.random.normal(size=(1, nx))
        # b initialized as 0 but later it will be a np.ndarray
        self.__b = 0
        # A initialized as 0 but later it will be a np.ndarray
        # with the same size as nx
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
        # z is the weighted sum of the inputs multiplied
        # by the weights added to the bias
        z = np.matmul(self.__W, X) + self.__b
        # A is the sigmoid function of z applied to
        # each element of z
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
            Calculates the cost of the
            model using logistic regression
        """
        # m is the number of training examples
        # It's [1] because [0] is the number of rows
        # and it's probably 1 anyways
        m = Y.shape[1]
        # cost is the cost of the model using logistic regression
        # This can be used as a general formula for other cases
        # where logistic regression must be used
        # Y is the correct labels for the input data and A is
        # the activated output of the neuron for each example
        cost = -(1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1.0000001-A))
        return cost

    def evaluate(self, X, Y):
        """
            Evaluates the neuron's predictions
        """
        # res is the activated output of the neuron for each example
        # being X the input data and Y the correct labels for the input data
        res = self.forward_prop(X)
        # return the neuron's prediction and the cost of the network
        # The prediction is 1 if the output of the network is >= 0.5
        # and 0 otherwise
        return np.where(res >= 0.5, 1, 0), self.cost(Y, res)
