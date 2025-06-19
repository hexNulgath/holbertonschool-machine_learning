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
        _, self.input_width, self.input_height, _ = self.model.input.shape


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
        Process Darknet model outputs with deterministic operations.
        Args:
            outputs: List of numpy.ndarrays containing predictions from Darknet
                    Each output has shape (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            image_size: Original image size [height, width]
        Returns:
            Tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            # Get grid dimensions
            grid_height, grid_width, num_anchors, _ = output.shape
            
            # Create grid of cell indices with proper dimensions (grid_height, grid_width, 1, 1)
            grid_y = np.arange(grid_height).reshape(grid_height, 1, 1, 1)
            grid_x = np.arange(grid_width).reshape(1, grid_width, 1, 1)
            
            # Extract predictions
            tx = output[..., 0:1]  # (grid_h, grid_w, num_anchors, 1)
            ty = output[..., 1:2]  # (grid_h, grid_w, num_anchors, 1)
            tw = output[..., 2:3]  # (grid_h, grid_w, num_anchors, 1)
            th = output[..., 3:4]  # (grid_h, grid_w, num_anchors, 1)
            box_confidence = output[..., 4:5]  # (grid_h, grid_w, num_anchors, 1)
            class_probs = output[..., 5:]  # (grid_h, grid_w, num_anchors, num_classes)
            
            # Get anchor dimensions for this output layer (1, 1, num_anchors, 1)
            anchor_w = self.anchors[i, :, 0].reshape(1, 1, num_anchors, 1)
            anchor_h = self.anchors[i, :, 1].reshape(1, 1, num_anchors, 1)
            
            # Calculate bounding box coordinates with proper broadcasting
            bx = (self.sigmoid(tx) + grid_x) / grid_width
            by = (self.sigmoid(ty) + grid_y) / grid_height
            bw = (np.exp(tw) * anchor_w) / self.input_width
            bh = (np.exp(th) * anchor_h) / self.input_height
            
            # Convert to image coordinates
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height
            
            # Stack coordinates into [x1, y1, x2, y2] format
            box = np.stack([x1, y1, x2, y2], axis=-1)
            
            # Apply sigmoid to confidence and class probabilities
            boxes.append(box)
            box_confidences.append(self.sigmoid(box_confidence))
            box_class_probs.append(self.sigmoid(class_probs))

        return boxes, box_confidences, box_class_probs

    def sigmoid(self, x):
        """Sigmoid activation function with deterministic implementation"""
        return 1 / (1 + np.exp(-x))