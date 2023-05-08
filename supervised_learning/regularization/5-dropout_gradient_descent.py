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
    aux = {}
    aux['A' + str(L)] = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        kA = 'A' + str(i - 1)
        kA_ = 'A' + str(i)
        kW = 'W' + str(i)
        kb = 'b' + str(i)
        kdW = 'dW' + str(i)
        kdb = 'db' + str(i)
        kdA = 'dA' + str(i - 1)
        dZ = aux[kA_] * (1 - cache[kA_] ** 2)
        dZ = np.multiply(dZ, cache['D' + str(i)])
        dZ /= keep_prob
        aux[kdW] = np.matmul(dZ, aux[kA].T) / m
        aux[kdb] = np.sum(dZ, axis=1, keepdims=True) / m
        aux[kdA] = np.matmul(weights[kW].T, dZ)
        weights[kW] -= alpha * aux[kdW]
        weights[kb] -= alpha * aux[kdb]
    return weights
