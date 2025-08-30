#!/usr/bin/env python3
"""
Defines convolutional generator and discriminator models
for a GAN using TensorFlow/Keras.
"""
import tensorflow as tf


def convolutional_GenDiscr():
    """
    Builds and returns the generator and discriminator models.

    Returns:
        generator: the Keras Model for the generator
        discriminator: the Keras Model for the discriminator
    """
    def get_generator():
        """
        Builds the generator model for the GAN.
        """
        input_layer = tf.keras.layers.Input(shape=(16,))
        x = tf.keras.layers.Dense(2048)(input_layer)
        x = tf.keras.layers.Reshape((2, 2, 512))(x)
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('tanh')(x)
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('tanh')(x)
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('tanh')(x)
        output_layer = x
        gen = tf.keras.Model(
            inputs=input_layer, outputs=output_layer, name='generator')
        return gen

    def get_discriminator():
        """
        Builds the discriminator model for the GAN.
        """
        input_layer = tf.keras.layers.Input(shape=(16, 16, 1))
        x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same')(
            input_layer)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = tf.keras.layers.Activation('tanh')(x)
        x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = tf.keras.layers.Activation('tanh')(x)
        x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = tf.keras.layers.Activation('tanh')(x)
        x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = tf.keras.layers.Activation('tanh')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        output_layer = x
        discr = tf.keras.Model(
            inputs=input_layer, outputs=output_layer, name='discriminator')
        return discr

    return get_generator(), get_discriminator()
