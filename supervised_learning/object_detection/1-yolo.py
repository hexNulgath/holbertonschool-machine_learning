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
            outputs is the number of outputs (predictions) made box_y the model
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
        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            # Convert from grid coordinates to image coordinates
            x = output[..., 0:1]  # x center
            y = output[..., 1:2]  # y center
            w = output[..., 2:3]  # width
            h = output[..., 3:4]  # height

            # size of anchors
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]
            # Get the grid size
            grid_height, grid_width = output.shape[1:3]
            # grid cells coordinates for width and height
            # Create a grid of coordinates for the output
            grid_x, grid_y = np.meshgrid(np.arange(grid_width),
                                 np.arange(grid_height))

            # Add axis to match dimensions of x & y
            grid_x = np.expand_dims(grid_x, axis=-1)
            grid_y = np.expand_dims(grid_y, axis=-1)

            # Calculate bounding box coordinates
            box_x = (self.sigmoid(x) + grid_x) / grid_width
            box_y = (self.sigmoid(y) + grid_y) / grid_height
            # Calculate the center of the box (relative to the grid)
            box_w = (np.exp(w) * anchor_w) / self.model.input.shape[1]
            box_h = (np.exp(h) * anchor_h) / self.model.input.shape[2]

            # Convert to coordinates of (x1, y1) and (x2, y2)
            x1 = (box_x - box_w / 2) * image_width
            y1 = (box_y - box_h / 2) * image_height
            x2 = (box_x + box_w / 2) * image_width
            y2 = (box_y + box_h / 2) * image_height

            # Combine coordinates into [x1, y1, x2, y2] format
            box = np.stack((x1, y1, x2, y2), axis=-1)
            # Extract components from output tensor
            box_confidence = output[..., 4:5]  # Objectness score
            class_probs = output[..., 5:]  # Class probabilities
            # Append to results
            boxes.append(box)
            box_confidences.append(self.sigmoid(box_confidence))
            box_class_probs.append(self.sigmoid(class_probs))

        return boxes, box_confidences, box_class_probs

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))