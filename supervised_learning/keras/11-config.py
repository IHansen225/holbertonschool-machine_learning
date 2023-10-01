#!/usr/bin/env python3
"""
    Keras training module.
"""
import tensorflow.keras as K


def save_config(network, filename):
    """ saves the model's config """
    with open(filename, 'w') as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """ loads the model's config """
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
