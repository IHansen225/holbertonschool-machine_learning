#!/usr/bin/env python3
"""
    Keras training module.
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
        Tests a neural network.
    """
    return network.predict(data, verbose=verbose)
