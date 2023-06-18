#!/usr/bin/env python3
"""
    Yolo class module v1
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

    def sigmoid(self, x):
        """
            sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
            Process Darknet outputs
        """
        boxes = []
        conf = []
        probs = []
        img_h, img_w = image_size
        for output in outputs:
            boxes.append(output[..., 0:4])
            conf.append(self.sigmoid(output[..., 4, np.newaxis]))
            probs.append(self.sigmoid(output[..., 5:]))
        for i, box in enumerate(boxes):
            gr_h, gr_w, anchors_boxes, _ = box.shape
            cx = np.indices((gr_h, gr_w, anchors_boxes))[1]
            cy = np.indices((gr_h, gr_w, anchors_boxes))[0]
            t_x = box[..., 0]
            t_y = box[..., 1]
            t_w = box[..., 2]
            t_h = box[..., 3]
            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]
            bx = (self.sigmoid(t_x) + cx) / gr_w
            by = (self.sigmoid(t_y) + cy) / gr_h
            bw = (np.exp(t_w) * p_w) / self.model.input.shape[1].value
            bh = (np.exp(t_h) * p_h) / self.model.input.shape[2].value
            tl_x = bx - bw / 2
            tl_y = by - bh / 2
            br_x = bx + bw / 2
            br_y = by + bh / 2
            box[..., 0] = tl_x * img_w
            box[..., 1] = tl_y * img_h
            box[..., 2] = br_x * img_w
            box[..., 3] = br_y * img_h
        return boxes, conf, probs
