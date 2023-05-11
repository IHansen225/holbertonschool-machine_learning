#!/usr/bin/env python3
"""
    Evaluation module.
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
        Evaluates the output of a neural network.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        graph = tf.get_default_graph()
        y_pred = graph.get_tensor_by_name('y_pred:0')
        loss = graph.get_tensor_by_name('loss:0')
        accuracy = graph.get_tensor_by_name('accuracy:0')
        feed_dict = {'X:0': X, 'Y:0': Y}
        y_pred_val, loss_val, accuracy_val = sess.run([y_pred, loss, accuracy], feed_dict)

    return y_pred_val, accuracy_val, loss_val
