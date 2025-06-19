#!/usr/bin/env python3
"""Yolo class for loading a Darknet Keras model and its parameters."""
from tensorflow import keras as K
import numpy as np


class Yolo:
    """
    Yolo class for loading a Darknet Keras model and its associated parameters.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path is the path to where a Darknet Keras model is stored
        classes_path is the path to where the list of class names used
        for the Darknet model, listed in order of index, can be found
        class_t is a float representing the box score threshold for the
        initial filtering step
        nms_t is a float representing the IOU threshold for non-max suppression
        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes:
            outputs is the number of outputs (predictions) made by the model
            anchor_boxes is the number of anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height]

        Public instance attributes:
            model: the Darknet Keras model
            class_names: a list of the class names for the model
            class_t: the box score threshold for the initial filtering step
            nms_t: the IOU threshold for non-max suppression
            anchors: the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        self.class_names = self.load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def load_classes(classes_path):
        """
        Loads class names from a file.
        """
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        return classes

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs and convert them to
        interpretable bounding boxes.

        Args:
            outputs: List of numpy.ndarrays containing predictions from Darknet
            image_size: Original image size [height, width]

        Returns:
            Tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            # Extract components from output tensor
            box_xy = output[..., :2]  # Center coordinates (t_x, t_y)
            # sigmoid activtion
            box_xy = 1 / (1 + np.exp(-box_xy))

            box_wh = output[..., 2:4]  # Width/height (t_w, t_h)
            box_confidence = output[..., 4:5]  # Objectness score
            class_probs = output[..., 5:]  # Class probabilities

            # Convert from grid coordinates to image coordinates
            box_x1y1 = (box_xy - (box_wh / 2)) * [image_size[1], image_size[0]]
            box_x2y2 = (box_xy + (box_wh / 2)) * [image_size[1], image_size[0]]

            # Combine coordinates into [x1, y1, x2, y2] format
            box = np.concatenate([box_x1y1, box_x2y2], axis=-1)
            # Append to results
            boxes.append(box)
            # Apply sigmoid
            box_confidences.append(1 / (1 + np.exp(-box_confidence)))
            box_class_probs.append(1 / (1 + np.exp(-class_probs)))
        return boxes, box_confidences, box_class_probs
