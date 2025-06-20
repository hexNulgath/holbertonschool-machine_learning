#!/usr/bin/env python3
"""Yolo class for loading a Darknet Keras model and its parameters."""
from tensorflow import keras as K
import numpy as np
import tensorflow as tf


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
                    Each output has shape (grid_height, grid_width,
                                            anchor_boxes, 4 + 1 + classes)
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

            # Create grid of cell indices with proper dimensions
            grid_y = np.arange(grid_height).reshape(grid_height, 1, 1, 1)
            grid_x = np.arange(grid_width).reshape(1, grid_width, 1, 1)

            # Extract predictions
            tx = output[..., 0:1]  # (grid_h, grid_w, num_anchors, 1)
            ty = output[..., 1:2]  # (grid_h, grid_w, num_anchors, 1)
            tw = output[..., 2:3]  # (grid_h, grid_w, num_anchors, 1)
            th = output[..., 3:4]  # (grid_h, grid_w, num_anchors, 1)
            box_confidence = output[..., 4:5]
            class_probs = output[..., 5:]

            # Get anchor dimensions for this output layer
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

            box = np.concatenate((x1, y1, x2, y2), axis=-1)
            # Apply sigmoid to confidence and class probabilities
            boxes.append(box)
            box_confidences.append(self.sigmoid(box_confidence))
            box_class_probs.append(self.sigmoid(class_probs))

        return boxes, box_confidences, box_class_probs

    def sigmoid(self, x):
        """Sigmoid activation function with deterministic implementation"""
        return 1 / (1 + np.exp(-x))

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        boxes: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 4)
        containing the processed boundary boxes for each output, respectively
        box_confidences: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 1)
        containing the processed box confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, classes)
        containing the processed box class probabilities for each output
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes: a numpy.ndarray of shape (?, 4)
            containing all of the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,)
            containing the class number that each box in filtered_boxes
            predicts
            box_scores: a numpy.ndarray of shape (?)
            containing the box scores for each box in filtered_boxes
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            # Reshape predictions for filtering
            box_score = box_confidences[i] * box_class_probs[i]
            # Get the class with the highest score for each box
            box_class = np.argmax(box_score, axis=-1)
            # Get the score of the highest class for each box
            box_score = np.max(box_score, axis=-1)

            # Flatten predictions
            boxes_flat = boxes[i].reshape(-1, 4)
            classes_flat = box_class.flatten()
            scores_flat = box_score.flatten()

            # Filter by threshold
            mask = scores_flat >= self.class_t
            filtered_boxes.append(boxes_flat[mask])
            box_classes.append(classes_flat[mask])
            box_scores.append(scores_flat[mask])

        # Concatenate results if any boxes remain
        # To eliminate empty arrays, we check if any boxes were filtered
        # and concatenate only if there are valid boxes
        if len(filtered_boxes) > 0:
            filtered_boxes = np.concatenate(filtered_boxes, axis=0)
            box_classes = np.concatenate(box_classes, axis=0)
            box_scores = np.concatenate(box_scores, axis=0)
        else:
            filtered_boxes = np.zeros((0, 4))
            box_classes = np.zeros((0,))
            box_scores = np.zeros((0,))

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing
            all of the filtered bounding boxes:
        box_classes: a numpy.ndarray of shape (?,) containing the class
            number for the class that filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box
            scores for each box in filtered_boxes, respectively

        Returns a tuple of
        (box_predictions, predicted_box_classes, predicted_box_scores):
        box_predictions: a numpy.ndarray of shape (?, 4)
            containing all of the predicted bounding boxes
            ordered by class and box score
        predicted_box_classes: a numpy.ndarray of shape (?,)
            containing the class number for box_predictions ordered
            by class and box score
        predicted_box_scores: a numpy.ndarray of shape (?)
            containing the box scores for box_predictions ordered
            by class and box score
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            # Get indices of boxes for this class
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            if len(cls_boxes) == 0:
                continue
            # Apply non-max suppression
            indices = tf.image.non_max_suppression(
                cls_boxes, cls_scores, max_output_size=50,
                iou_threshold=self.nms_t
            ).numpy()
            # Get the selected boxes, classes, and scores
            selected_boxes = cls_boxes[indices]
            selected_scores = cls_scores[indices]
            selected_classes = np.full(selected_scores.shape, cls)
            # Append to results
            box_predictions.append(selected_boxes)
            predicted_box_classes.append(selected_classes)
            predicted_box_scores.append(selected_scores)
        return (np.concatenate(box_predictions, axis=0),
                np.concatenate(predicted_box_classes, axis=0),
                np.concatenate(predicted_box_scores, axis=0))
