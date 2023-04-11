#!/usr/bin/env python3
"""
    Neuron class file for classification
    algorithm exercises
"""
import numpy as np


# For anyone reading this, I'm sorry for the excessive commenting,
# but if I don't comment everything I'll forget what this thing does

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

            Basically this is the sigmoid function
            and it's what produces the output of the neuron
            based on what you give it as an input

            It produces as many outputs as there are
            input features
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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
            Calculates one pass of gradient descent on the neuron
            and updates the corresponding attributes
        """
        # X is a np.ndarray with shape (nx, m) that contains the input data
        # Y contains the correct labels for the input data
        # A contains the activated output of the neuron for each example
        ## A can't be changed because you can't touch neurons but you can
        ## change the weights and bias
        # m is the number of training examples
        m = Y.shape[1]
        # dz is the gradient of the cost with respect to z
        # Basically computes the error of the neuron's results
        # versus the expected results. It gives away the magnitude
        # for the change in the weights and bias.
        dz = A - Y
        # dw is the gradient of the cost with respect to w
        # Computes the change in the weights
        dw = (1/m) * np.matmul(dz, X.T)
        # db is the gradient of the cost with respect to b
        # Computes the change in the bias
        db = (1/m) * np.sum(dz)
        # Update the weights and bias multiplicated by the learning rate
        # being the learning rate a scalar because if I
        # leave the gradient as is, this thing will (probably) explode
        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db
