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
    dZ = cache['A' + str(L)] - Y
    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]
        A = cache['A' + str(l)]
        w = weights['W' + str(l)]
        b = weights['b' + str(l)]
        dA = np.dot(w.T, dZ)
        dA *= (A > 0)
        dA /= keep_prob
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        w -= alpha * dW
        b -= alpha * db
        dZ = dA
    weights['W1'] = w
    weights['b1'] = b
    return weights
