#!/usr/bin/env python3
"""
    Transfer Learning Module
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as K

def preprocess_data(X, Y):
    X = X.astype('float32') / 255.0
    Y = K.utils.to_categorical(Y, 10)
    return X, Y

(X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
X_train, Y_train = preprocess_data(X_train, Y_train)
X_test, Y_test = preprocess_data(X_test, Y_test)
base_model = K.applications.MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False
x = K.layers.Lambda(lambda image: tf.image.resize(image, (96, 96)))(base_model.output)
x = K.layers.GlobalAveragePooling2D()(x)
x = K.layers.Dense(128, activation='relu')(x)
predictions = K.layers.Dense(10, activation='softmax')(x)
model = K.models.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32)
model.save('cifar10.h5')
