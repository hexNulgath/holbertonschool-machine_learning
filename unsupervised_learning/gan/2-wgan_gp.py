#!/usr/bin/env python3
"""
WGAN_GP module
--------------
Implements a Wasserstein GAN with Gradient Penalty
(WGAN-GP) using TensorFlow/Keras.

Classes:
    WGAN_GP: Subclass of keras.Model implementing WGAN-GP training.
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP).

    Args:
        generator: Keras model, the generator network.
        discriminator: Keras model, the discriminator (critic) network.
        latent_generator: Callable, generates latent vectors for the generator.
        real_examples: Tensor, dataset of real samples.
        batch_size: int, batch size for training.
        disc_iter: int, number of discriminator updates per generator update.
        learning_rate: float, learning rate for optimizers.
        lambda_gp: float, gradient penalty coefficient.
    """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """
        Initialize the WGAN_GP model and set up optimizers and loss functions.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size

        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9
        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        self.discriminator.loss = lambda x, y: (tf.reduce_mean(y) - tf.reduce_mean(x))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                    loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        Generate a batch of fake samples using the generator.

        Args:
            size: int, number of samples to generate. Defaults to batch_size.
            training: bool, whether to run in training mode.
        Returns:
            Tensor of generated samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Sample a batch of real examples from the dataset.

        Args:
            size: int, number of samples to draw. Defaults to batch_size.
        Returns:
            Tensor of real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generate interpolated samples between real and
        fake samples for gradient penalty.

        Args:
            real_sample: Tensor, batch of real samples.
            fake_sample: Tensor, batch of fake samples.
        Returns:
            Tensor of interpolated samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Compute the gradient penalty for WGAN-GP.

        Args:
            interpolated_sample: Tensor, batch of interpolated samples.
        Returns:
            Scalar tensor, gradient penalty value.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
        Custom training step for WGAN-GP.
        Performs multiple discriminator updates and one generator update.

        Args:
            useless_argument: Placeholder for compatibility with Keras API.
        Returns:
            Dictionary with discriminator and generator losses, and gradient
            penalty.
        """
        # Train discriminator (critic) multiple times
        for _ in range(self.disc_iter):
            with tf.GradientTape() as d_tape:
                # Get real and fake samples
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                interpolated_samples = self.get_interpolated_sample(
                                    real_samples, fake_samples)
                # Discriminator outputs
                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)
                gp = self.gradient_penalty(interpolated_samples)

                # Wasserstein loss
                d_loss = self.discriminator.loss(real_output, fake_output)

                # Gradient penalty
                gp = self.gradient_penalty(interpolated_samples)

                # Total discriminator loss
                discr_loss = d_loss + self.lambda_gp * gp

            # Apply gradients to discriminator
            d_gradients = d_tape.gradient(
                discr_loss,
                self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables))

        # Train generator once
        with tf.GradientTape() as g_tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_samples, training=True)
            g_loss = self.generator.loss(fake_output)

        # Apply gradients to generator
        g_gradients = g_tape.gradient(
            g_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": g_loss, "gp": gp}
