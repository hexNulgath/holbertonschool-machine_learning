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
        boxes = []
        box_confidences = []
        box_class_probs = []
        
        for i, output in enumerate(outputs):
            # Apply sigmoid to tx, ty and confidence
            box_xy = 1 / (1 + np.exp(-output[..., :2]))  # Sigmoid for center coordinates
            box_wh = np.exp(output[..., 2:4])  # exp for width/height
            
            # Scale to grid size
            grid_h, grid_w = output.shape[:2]
            box_xy = (box_xy + self._get_grid_offset(grid_h, grid_w)) / [grid_w, grid_h]
            
            # Scale to image size
            box_xy = box_xy * [image_size[1], image_size[0]]  # width, height
            box_wh = box_wh * self.anchors[i]  # Multiply by anchor sizes
            
            # Convert to (x1,y1,x2,y2)
            box_x1y1 = box_xy - (box_wh / 2)
            box_x2y2 = box_xy + (box_wh / 2)
            box = np.concatenate([box_x1y1, box_x2y2], axis=-1)
            
            # Process confidence and class probs
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            
            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(class_probs)
        
        return boxes, box_confidences, box_class_probs

    def _get_grid_offset(self, grid_h, grid_w):
        """Create grid offset for cell indices"""
        grid_x = np.arange(grid_w).reshape(1, -1, 1)
        grid_y = np.arange(grid_h).reshape(-1, 1, 1)
        return np.concatenate([grid_x, grid_y], axis=-1)