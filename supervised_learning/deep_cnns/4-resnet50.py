#!/usr/bin/env python3
"""
    ResNet-50 architecture.
"""
import tensorflow.keras as K
projection_block = __import__('3-projection_block').projection_block
identity_block = __import__('2-identity_block').identity_block


def resnet50():
    """
        Creates a ResNet-50 network.
    """
    init = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                            kernel_initializer=init)(X)
    norm1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation('relu')(norm1)
    pool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(act1)
    proj1 = projection_block(pool1, [64, 64, 256], s=1)
    iden1 = identity_block(proj1, [64, 64, 256])
    iden2 = identity_block(iden1, [64, 64, 256])
    proj2 = projection_block(iden2, [128, 128, 512])
    iden3 = identity_block(proj2, [128, 128, 512])
    iden4 = identity_block(iden3, [128, 128, 512])
    iden5 = identity_block(iden4, [128, 128, 512])
    proj3 = projection_block(iden5, [256, 256, 1024])
    iden6 = identity_block(proj3, [256, 256, 1024])
    iden7 = identity_block(iden6, [256, 256, 1024])
    iden8 = identity_block(iden7, [256, 256, 1024])
    iden9 = identity_block(iden8, [256, 256, 1024])
    iden10 = identity_block(iden9, [256, 256, 1024])
    proj4 = projection_block(iden10, [512, 512, 2048])
    iden11 = identity_block(proj4, [512, 512, 2048])
    iden12 = identity_block(iden11, [512, 512, 2048])
    avgpool = K.layers.AveragePooling2D((7, 7), padding='same')(iden12)
    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(avgpool)
    model = K.models.Model(inputs=X, outputs=output)
    return model
