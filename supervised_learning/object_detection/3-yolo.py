#!/usr/bin/env python3
"""
    Yolo class module v3
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
            Filter box outputs
        """
        box_scores = []
        box_classes = []
        box_class_scores = []
        for i in range(len(boxes)):
            box_scores.append(box_confidences[i] * box_class_probs[i])
            box_classes.append(np.argmax(box_scores[i], axis=-1))
            box_class_scores.append(np.max(box_scores[i], axis=-1))
        box_classes = [box.reshape(-1) for box in box_classes]
        box_class_scores = [box.reshape(-1) for box in box_class_scores]
        box_classes = np.concatenate(box_classes)
        box_class_scores = np.concatenate(box_class_scores)
        filtering_mask = np.where(box_class_scores >= self.class_t)
        boxes = np.concatenate([box.reshape(-1, 4) for box in boxes])
        filtered_boxes = boxes[filtering_mask]
        box_classes = box_classes[filtering_mask]
        box_class_scores = box_class_scores[filtering_mask]
        return filtered_boxes, box_classes, box_class_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
            Perform non-max suppression
        """
        box_scores = list(box_scores)
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            cls_indices = np.where(box_classes == cls)[0]
            cls_scores = [box_scores[i] for i in cls_indices]
            cls_boxes = filtered_boxes[cls_indices]

            sorted_indices = np.argsort(cls_scores)[::-1]
            sorted_scores = [cls_scores[i] for i in sorted_indices]
            sorted_boxes = cls_boxes[sorted_indices]

            keep_indices = []
            while sorted_indices:
                best_idx = sorted_indices[0]
                best_box = sorted_boxes[0]
                box_predictions.append(best_box)
                predicted_box_classes.append(cls)
                predicted_box_scores.append(sorted_scores[0])

                iou = self.iou(best_box, sorted_boxes[1:])
                low_iou_indices = np.where(iou < self.iou_threshold)[0]
                sorted_indices = np.delete(sorted_indices, low_iou_indices + 1)
                sorted_scores = np.delete(sorted_scores, low_iou_indices + 1)
                sorted_boxes = np.delete(sorted_boxes, low_iou_indices + 1, axis=0)

            box_predictions = np.array(box_predictions)
            predicted_box_classes = np.array(predicted_box_classes)
            predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    def iou(self, box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

        iou = intersection_area / (box_area + boxes_area - intersection_area)
        return iou
