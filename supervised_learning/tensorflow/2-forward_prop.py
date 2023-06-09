#!/usr/bin/env python3
"""
    Forward propagation with tensorflow.
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
        Creates the forward propagation graph for the neural network.
    """
    # Create the first layer.
    layer = create_layer(x, layer_sizes[0], activations[0])
    # Create the rest of the layers.
    for i in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])
    # Return the last layer.
    return layer
