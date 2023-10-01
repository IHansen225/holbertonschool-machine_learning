#!/usr/bin/env python3
"""
    Keras training module.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch, ep, verbose=True, shuffle=False):
    """
        Trains a model using mini-batch gradient descent.
    """
    return network.fit(data, labels, batch_size=batch, epochs=ep,
                       verbose=verbose, shuffle=shuffle)
