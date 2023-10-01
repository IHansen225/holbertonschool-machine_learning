#!/usr/bin/env python3
"""
    Dropout Gradient Descent
    module.
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
        Conducts forward propagation using Dropout.
    """
    m = Y.shape[1]
    for i in reversed(range(L)):
        if i == L - 1:
            dz = cache['A' + str(i + 1)] - Y
            dw = np.matmul(cache['A' + str(i)], dz.T) / m
        else:
            d1 = np.matmul(weights['W' + str(i + 2)].T, dz)
            d2 = 1 - cache['A' + str(i + 1)] ** 2
            dz = d1 * d2
            dz *= cache['D' + str(i + 1)]
            dz /= keep_prob
            dw = np.matmul(dz, cache['A' + str(i)].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights['W' + str(i + 1)] -= alpha * dw.T
        weights['b' + str(i + 1)] -= alpha * db
