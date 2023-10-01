#!/usr/bin/env python3
"""
    Dropout Forward Propagation
    module.
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
        Conducts forward propagation using Dropout.
    """
    cache = {}
    cache['A0'] = X
    for i in range(L):
        A = cache['A' + str(i)]
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        Z = np.matmul(W, A) + b
        if i == L - 1:
            t = np.exp(Z)
            cache['A' + str(i + 1)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            cache['A' + str(i + 1)] = np.tanh(Z)
            cache['D' + str(i + 1)] = np.random.binomial(1, keep_prob,
                                                         size=Z.shape)
            cache['A' + str(i + 1)] *= cache['D' + str(i + 1)]
            cache['A' + str(i + 1)] /= keep_prob
    return cache
