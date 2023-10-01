#!/usr/bin/env python3
"""
    One-hot conversion module.
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
        Converts a label vector into a one-hot matrix.
    """
    return K.utils.to_categorical(labels, classes)
