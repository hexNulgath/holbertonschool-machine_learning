#!/usr/bin/env python3
"""
Implementation of a Simple Generative Adversarial Network (GAN) using TensorFlow.

This module provides a Simple_GAN class that implements a basic GAN architecture
with customizable generator and discriminator models.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """
    A simple Generative Adversarial Network implementation.
    
    This class implements a GAN with a generator and discriminator that
    are trained in an adversarial manner to generate realistic samples.
    
    Attributes:
        latent_generator: Function that generates latent space vectors
        real_examples: Tensor of real training examples
        generator: The generator model
        discriminator: The discriminator model
        batch_size: Size of training batches
        disc_iter: Number of discriminator iterations per generator iteration
        learning_rate: Learning rate for optimizers
        beta_1: Beta1 parameter for Adam optimizer
        beta_2: Beta2 parameter for Adam optimizer
    """
    
    def __init__(self, generator, discriminator, latent_generator, 
                 real_examples, batch_size=200, disc_iter=2, learning_rate=0.005):
        """
        Initialize the Simple_GAN.
        
        Args:
            generator: Generator model that creates samples from noise
            discriminator: Discriminator model that distinguishes real/fake samples
            latent_generator: Function that generates latent space vectors
            real_examples: Tensor containing real training examples
            batch_size: Size of training batches (default: 200)
            disc_iter: Number of discriminator iterations per generator iteration (default: 2)
            learning_rate: Learning rate for Adam optimizers (default: 0.005)
        """
        super().__init__()  # Run the __init__ of keras.Model first
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        
        self.learning_rate = learning_rate
        self.beta_1 = 0.5  # Standard value, but can be changed if necessary
        self.beta_2 = 0.9  # Standard value, but can be changed if necessary
        
        # Define the generator loss and optimizer
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, 
            beta_1=self.beta_1, 
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=generator.optimizer, 
            loss=generator.loss)
        
        # Define the discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape)))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, 
            beta_1=self.beta_1, 
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=discriminator.optimizer, 
            loss=discriminator.loss)
    
    def get_fake_sample(self, size=None, training=False):
        """
        Generate fake samples using the generator.
        
        Args:
            size: Number of samples to generate (defaults to batch_size)
            training: Whether the generator is in training mode
            
        Returns:
            Tensor of generated samples
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Sample real examples from the training data.
        
        Args:
            size: Number of samples to retrieve (defaults to batch_size)
            
        Returns:
            Tensor of real samples from the training data
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)
             
    def train_step(self, useless_argument):
        """
        Perform one training step of the GAN.
        
        This method implements the adversarial training process where:
        1. The discriminator is trained to distinguish real from fake samples
        2. The generator is trained to fool the discriminator
        
        Args:
            useless_argument: Required by Keras API but not used
            
        Returns:
            Dictionary with discriminator and generator loss values
        """
        # Train discriminator multiple times
        for _ in range(self.disc_iter):
            # Get real and fake samples
            real_samples = self.get_real_sample()
            fake_samples = self.get_fake_sample(training=True)
            
            # Train discriminator
            with tf.GradientTape() as tape:
                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)
                discr_loss = self.discriminator.loss(real_output, fake_output)
                
            # Calculate gradients and update discriminator
            grads = tape.gradient(
                discr_loss, 
                self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))

        # Train generator
        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_samples, training=True)
            gen_loss = self.generator.loss(fake_output)
            
        # Calculate gradients and update generator
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
