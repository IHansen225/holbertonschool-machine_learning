#!/usr/bin/env python3
"""
    Keras input model
    module to create a neural network.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lam, keep_prob):
    """
        Build Keras model with
        the input object.
        nx: number of input features to the network.
        layers: list containing the number of nodes
                in each layer of the network.
        activations: list containing the activation
                functions used for each layer of the network.
        lam: L2 regularization parameter.
        keep_prob: probability that a node will be kept
                for dropout regularization.
    """
    # Initialize model with input shape
    inputs = K.Input(shape=(nx,))
    # Add first layer with input shape
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=K.regularizers.l2(lam))(inputs)
    # Add subsecuent layers with dropout according to
    # keep_prob and regularization according to lam
    for i in range(1, len(layers)):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=K.regularizers.l2(lam))(x)
    # Create model
    model = K.Model(inputs=inputs, outputs=x)
    return model
