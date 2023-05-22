#!/usr/bin/env python3
"""
Task 4. LeNet-5 (Tensorflow)
"""
import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture

        x: tf.placeholder of shape (m, 28, 28, 1) containing the input images
        for the network
            m:  number of images
        y: tf.placeholder of shape (m, 10) containing the one-hot labels for
        the network

    NOTES
    - NOT use tf.keras
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

    Returns:
    - a tensor for the softmax activated output
    - a training operation that utilizes Adam optimization
    (with default hyperparam)
    - a tensor for the loss of the netowrk
    - a tensor for the accuracy of the network
    """

    # Initializate kernel:
    kernel_init = tf.contrib.layers.variance_scaling_initializer()

    # Create the layers one by one
    # (take the layers output in the process...):
    C1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=kernel_init
    )
    result_1 = C1(x)

    P2 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )
    result_2 = P2(result_1)

    C3 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=kernel_init
    )
    result_3 = C3(result_2)

    # P2 = P4, thus, reuse P2 to generate the P4 output:
    result_4 = P2(result_3)

    # Now we must adapt the multidimensional output so that it can be used for
    # the fully connected layers.
    # Thus, we will create a FLATTEN layer: transform a multidimensional tensor
    # into a flat tensor (1-dimensional tensor):
    flatten_result_4 = tf.layers.Flatten()(result_4)

    F5 = tf.layers.Dense(
        120,
        activation=tf.nn.relu,
        kernel_initializer=kernel_init
    )
    result_5 = F5(flatten_result_4)

    F6 = tf.layers.Dense(
        84,
        activation=tf.nn.relu,
        kernel_initializer=kernel_init
    )
    result_6 = F6(result_5)

    O7 = tf.layers.Dense(
        10,
        kernel_initializer=kernel_init
    )
    result_7 = O7(result_6)

    # Define function final outputs:
    softmax_result = tf.nn.softmax(result_7)
    loss = tf.losses.softmax_cross_entropy(y, logits=result_7)
    training_op = tf.train.AdamOptimizer().minimize(loss)

    # Find the predicted class label to calculate accuracy:
    y_pred = tf.math.argmax(result_7, axis=1)
    y_out = tf.math.argmax(y, axis=1)
    comparison = tf.math.equal(y_pred, y_out)
    accuracy = tf.reduce_mean(tf.cast(comparison, 'float'))

    return softmax_result, training_op, loss, accuracy
