#!/usr/bin/env python3
"""
Task 5. LeNet-5 (Keras)
"""
import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras

        X: K.Input of shape (m, 28, 28, 1) containing the input images for
        the network
            m: number of images

    NOTES
    - All hidden layers requiring activation should use the relu activation
    function
    - All layers requiring initialization should initialize their kernels with
    the he_normal initialization method

    THE MODEL
    - C1: Convolutional layer with 6 kernels of shape 5x5 with same padding
    - P2: Max pooling layer with kernels of shape 2x2 with 2x2 strides
    - C3: Convolutional layer with 16 kernels of shape 5x5 with valid padding
    - P4: Max pooling layer with kernels of shape 2x2 with 2x2 strides
    - F5: Fully connected layer with 120 nodes
    - F6: Fully connected layer with 84 nodes
    - O7: Fully connected softmax output layer with 10 nodes

    Returns: a K.Model compiled to use Adam optimization (with default
    hyperparameters) and accuracy metrics
    """

    # Initializate kernel:
    kernel_init = K.initializers.he_normal()

    # Create the layers one by one
    # (take the layers output in the process...):
    C1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation=K.activations.relu,
        kernel_initializer=kernel_init
    )
    result_1 = C1(X)

    P2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )
    result_2 = P2(result_1)

    C3 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation=K.activations.relu,
        kernel_initializer=kernel_init
    )
    result_3 = C3(result_2)

    # P2 = P4, thus, reuse P2 to generate the P4 output:
    result_4 = P2(result_3)

    # Now we must adapt the multidimensional output so that it can be used for
    # the fully connected layers.
    # Thus, we will create a FLATTEN layer: transform a multidimensional tensor
    # into a flat tensor (1-dimensional tensor):
    flatten_result_4 = K.layers.Flatten()(result_4)

    F5 = K.layers.Dense(
        120,
        activation=K.activations.relu,
        kernel_initializer=kernel_init
    )
    result_5 = F5(flatten_result_4)

    F6 = K.layers.Dense(
        84,
        activation=K.activations.relu,
        kernel_initializer=kernel_init
    )
    result_6 = F6(result_5)

    O7 = K.layers.Dense(
        10,
        kernel_initializer=kernel_init
    )
    result_7 = O7(result_6)

    # Define function final outputs:
    softmax_result = K.layers.Softmax()(result_7)
    model_result = K.Model(inputs=X, outputs=softmax_result)
    model_result.compile(
        optimizer=K.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model_result
