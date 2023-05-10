#!/usr/bin/env python3
"""
    Keras training module.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch, ep,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
        Trains a model using mini-batch gradient descent.
    """
    cb = None
    if early_stopping and validation_data is not None:
        cb = [K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)]
    return network.fit(data, labels, batch_size=batch, epochs=ep,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=cb)
