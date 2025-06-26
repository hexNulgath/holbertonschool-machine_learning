#!/usr/bin/env python3
"""performs tasks for neural style transfer"""
import numpy as np
import tensorflow as tf


class NST:
    """
    style_image - the preprocessed style image
    content_image - the preprocessed content image
    alpha - the weight for content cost
    beta - the weight for style cost
    """
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        style_image - the image used as a style reference,
            stored as a numpy.ndarray
        content_image - the image used as a content reference,
            stored as a numpy.ndarray
        alpha - the weight for content cost
        beta - the weight for style cost
        model - the Keras model used to calculate cost
        gram_style_features - a list of gram matrices calculated from the
            style layer outputs of the style image
        content_feature - the content layer output of the content image
        """
        valid = (isinstance(style_image, np.ndarray)
                 and style_image.ndim == 3 and style_image.shape[2] == 3)
        if not valid:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        valid_img = (isinstance(content_image, np.ndarray)
                     and content_image.ndim == 3
                     and content_image.shape[2] == 3)
        if not valid_img:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        rescales an image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels
        image - a numpy.ndarray of shape (h, w, 3)
        containing the image to be scaled
        Returns: the scaled image
        """
        valid = (isinstance(image, np.ndarray)
                 and image.ndim == 3 and image.shape[2] == 3)
        if not valid:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        max_dim = max(h, w)
        scale = 512 / max_dim
        new_size = (int(h * scale), int(w * scale))
        # Convert to tensor and add batch dimension
        image_tensor = tf.convert_to_tensor(image)[tf.newaxis, ...]

        # Resize using bicubic interpolation (while in [0, 255] range)
        resized_image = tf.image.resize(
            image_tensor,
            new_size,
            method=tf.image.ResizeMethod.BICUBIC
        )

        # rescale to [0, 1]
        scaled_image = tf.clip_by_value(resized_image / 255.0, 0.0, 1.0)

        return scaled_image

    def load_model(self):
        """
        creates the model used to calculate cost
        """
        # Load VGG19 without top layers and pretrained on ImageNet
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet',
        )
        # Freeze weights
        vgg.trainable = False

        # Replace MaxPooling2D layers with AveragePooling2D
        for layer in vgg.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer.__class__ = tf.keras.layers.AveragePooling2D

        # Get the desired layer outputs
        style_outputs = [
            vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        # Create a model with multi-output (content + style layers)
        self.model = tf.keras.Model(
            inputs=vgg.input,
            outputs=[content_output] + style_outputs
        )

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates gram matrices in a more efficient way"""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # Get dimensions
        _, height, width, channels = tf.unstack(tf.shape(input_layer))
    
        # Reshape to 2D matrix
        features = tf.reshape(input_layer, [-1, channels])

        # Calculate gram matrix
        gram = tf.matmul(features, features, transpose_a=True)

        # Normalize by number of locations
        return gram / tf.cast(height * width, tf.float32)

    def generate_features(self):
        """Extracts the features used to calculate neural style cost"""
        # Preprocess the images
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        # Get the style features and content features
        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)
        # Calculate the gram matrices for the style features
        self.gram_style_features = [self.gram_matrix(style_feature)
                                    for style_feature in style_outputs[:-1]]
        # Get the content feature
        self.content_feature = content_outputs[-1]
