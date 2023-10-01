#!/usr/bin/env python3
"""
    Keras sequential model
    module to create a neural network.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lam, keep_prob):
    """
        Build a Keras sequential model.
        nx: number of input features to the network.
        layers: list containing the number of nodes
                in each layer of the network.
        activations: list containing the activation
                functions used for each layer of the network.
        lam: L2 regularization parameter.
        keep_prob: probability that a node will be kept
                for dropout regularization.
    """
    # Create empty sequential model
    model = K.Sequential()
    # Add first layer with input shape
    model.add(K.layers.Dense(layers[0], input_shape=(nx,),
                             activation=activations[0],
                             kernel_regularizer=K.regularizers.l2(lam)))
    # Add subsecuent layers with dropout according to
    # keep_prob and regularization according to lam
    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=K.regularizers.l2(lam)))
    return model
