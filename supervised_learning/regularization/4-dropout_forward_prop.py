#!/usr/bin/env python3
"""
    Dropout Forward Propagation
    module.
"""
import numpy as np
import tensorflow as tf


def dropout_forward_prop(X, weights, L, keep_prob):
    """
        Conducts forward propagation using Dropout.
    """
    aux = {}
    aux['A0'] = X
    for i in range(L):
        key = 'A' + str(i)
        keyW = 'W' + str(i + 1)
        keyb = 'b' + str(i + 1)
        keyA = 'A' + str(i + 1)
        Z = np.matmul(weights[keyW], aux[key]) + weights[keyb]
        if i == L - 1:
            t = np.exp(Z)
            aux[keyA] = t / np.sum(t, axis=0, keepdims=True)
        else:
            aux[keyA] = np.tanh(Z)
            aux[keyA] = np.multiply(np.random.rand(
                aux[keyA].shape[0], aux[keyA].shape[1]) < keep_prob,
                                      aux[keyA])
            aux[keyA] /= keep_prob
    return aux
