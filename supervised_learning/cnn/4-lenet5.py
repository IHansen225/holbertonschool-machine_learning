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
    init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=init,
    )(x)
    pool1 = tf.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2)
    )(conv1)
    conv2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=init,
    )(pool1)
    pool2 = tf.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2)
    )(conv2)
    flat = tf.layers.Flatten()(pool2)
    dense1 = tf.layers.Dense(
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=init,
    )(flat)
    dense2 = tf.layers.Dense(
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=init,
    )(dense1)
    dense3 = tf.layers.Dense(
        units=10,
        kernel_initializer=init,
    )(dense2)
    y_pred = tf.nn.softmax(dense3)
    loss = tf.losses.softmax_cross_entropy(y, dense3)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    y_pred = tf.argmax(y_pred, 1)
    y_true = tf.argmax(y, 1)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    return y_pred, train_op, loss, acc
