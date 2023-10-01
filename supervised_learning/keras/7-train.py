#!/usr/bin/env python3
"""
    Keras training module.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch, ep,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True,
                shuffle=False):
    """
        Trains a model using mini-batch gradient descent.
    """
    cb = []
    if validation_data is not None:
        if early_stopping:
            cb.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience))
        if learning_rate_decay:
            def scheduler(epoch):
                """ Scheduler function. """
                return alpha / (1 + decay_rate * epoch)
            cb.append(K.callbacks.LearningRateScheduler(scheduler,
                                                        verbose=1))
    return network.fit(data, labels, batch_size=batch, epochs=ep,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=cb)
