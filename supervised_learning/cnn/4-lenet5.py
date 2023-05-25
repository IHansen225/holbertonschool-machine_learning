#!/usr/bin/env python3
"""
    Creates a convolutional
    neural network using tensorflow
    and the LeNet-5 architecture.
"""
import tensorflow as tf


def lenet5(x, y):
    """
        Builds a modified version
        of the LeNet-5 architecture.
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                             activation=activation, kernel_initializer=init)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                             activation=activation, kernel_initializer=init)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    flatten = tf.layers.Flatten()(pool2)
    fc1 = tf.layers.Dense(units=120, activation=activation,
                          kernel_initializer=init)(flatten)
    fc2 = tf.layers.Dense(units=84, activation=activation,
                          kernel_initializer=init)(fc1)
    fc3 = tf.layers.Dense(units=10, kernel_initializer=init)(fc2)
    softmax = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc3)
    loss = tf.reduce_mean(softmax)
    train = tf.train.AdamOptimizer().minimize(loss)
    y_pred = tf.argmax(fc3, axis=1)
    y_true = tf.argmax(y, axis=1)
    equal = tf.equal(y_pred, y_true)
    acc = tf.reduce_mean(tf.cast(equal, tf.float32))
    return y_pred, train, loss, acc
