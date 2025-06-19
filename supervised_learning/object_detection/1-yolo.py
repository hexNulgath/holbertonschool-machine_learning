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
        Process Darknet model outputs and convert them to interpretable bounding boxes.
        
        Args:
            outputs: List of numpy.ndarrays containing predictions from Darknet for a single image
                    Each has shape (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            image_size: numpy.ndarray containing original image size [image_height, image_width]
        
        Returns:
            Tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        
        for i, output in enumerate(outputs):
            grid_h, grid_w, num_anchors, _ = output.shape
            
            # Create grid indices properly
            col = np.arange(grid_w).reshape(1, grid_w, 1, 1)
            row = np.arange(grid_h).reshape(grid_h, 1, 1, 1)
            col = np.tile(col, (grid_h, 1, num_anchors, 1))
            row = np.tile(row, (1, grid_w, num_anchors, 1))
            grid = np.concatenate((col, row), axis=-1)  # Shape: (grid_h, grid_w, num_anchors, 2)
            
            # Extract components
            box_xywh = output[..., :4]
            box_confidence = output[..., 4:5]
            class_probs = output[..., 5:]
            
            # Convert network outputs to absolute coordinates
            box_xy = 1 / (1 + np.exp(-box_xywh[..., :2])) + grid
            box_xy /= [grid_w, grid_h]  # Normalize to [0,1]
            
            # Convert width/height using anchors
            anchors_tensor = np.reshape(self.anchors[i], [1, 1, num_anchors, 2])
            box_wh = (np.exp(box_xywh[..., 2:4]) * anchors_tensor) / \
                    [image_size[1], image_size[0]]  # Normalized
            
            # Convert to x1y1x2y2 format
            box_xywh = np.concatenate([box_xy, box_wh], axis=-1)
            x, y, w, h = np.split(box_xywh, 4, axis=-1)
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            
            # Scale to original image size
            box = np.concatenate([x1, y1, x2, y2], axis=-1)
            box *= np.array([image_size[1], image_size[0], image_size[1], image_size[0]])
            
            # Apply sigmoid to confidence and class probs
            box_confidences.append(1 / (1 + np.exp(-box_confidence)))
            box_class_probs.append(1 / (1 + np.exp(-class_probs)))
            boxes.append(box)
        
        return boxes, box_confidences, box_class_probs