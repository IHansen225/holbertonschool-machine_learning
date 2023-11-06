#!/usr/bin/env python3
"""
    Yolo class module v0
"""
import numpy as np
import tensorflow.keras as K


class Yolo():
    """
        Yolo class
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
            Yolo class constructor
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
