#!/usr/bin/env python3
"""
    Creates a convolutional
    neural network using tensorflow
    and the LeNet-5 architecture.
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
        Builds a modified version
        of the LeNet-5 architecture.
    """
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
    )

    # Max pooling layer 1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=(2, 2),
        strides=(2, 2)
    )

    # Convolutional layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
    )

    # Max pooling layer 2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=(2, 2),
        strides=(2, 2)
    )

    # Flatten the pool2 output
    flatten = tf.layers.flatten(pool2)

    # Fully connected layer 1
    fc1 = tf.layers.dense(
        inputs=flatten,
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
    )

    # Fully connected layer 2
    fc2 = tf.layers.dense(
        inputs=fc1,
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
    )

    # Output layer
    logits = tf.layers.dense(
        inputs=fc2,
        units=10,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
    )

    # Softmax activated output
    output = tf.nn.softmax(logits)

    # Loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    )

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Training operation
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return output, train_op, loss, accuracy
