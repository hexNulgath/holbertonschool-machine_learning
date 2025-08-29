#!/usr/bin/env python3
""" This module defines the WGAN_GP class. """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """ This class defines a Wasserstein GAN with Gradient Penalty. """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """
        Constructor for the WGAN_GP class.
        Args:
            generator: The generator model.
            discriminator: The discriminator model.
            latent_generator: A function that generates latent vectors.
            real_examples: A tensor with real examples.
            batch_size: The batch size for training.
            disc_iter: The number of iterations to train the discriminator.
            learning_rate: The learning rate for the optimizer
            lambda_gp: The lambda value for the gradient penalty.
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

        # define the generator loss and optimizer
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        # define the discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: (
                tf.reduce_mean(y) - tf.reduce_mean(x))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer,
                                   loss=discriminator.loss)

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """
        This method generates fake samples using the generator model.
        Args:
            size: The number of samples to generate.
            training: A boolean indicating if the model is training.
        Returns:
            A tensor with the generated samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        """
        This method returns real samples from the dataset.
        Args:
            size: The number of samples to return.
        Returns:
            A tensor with the real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # generator of interpolating samples of size batch_size
    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        This method generates interpolated samples between real and fake
        samples.
        Args:
            real_sample: A tensor with real samples.
            fake_sample: A tensor with fake samples.
        Returns:
            A tensor with the interpolated samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u*real_sample + v*fake_sample

    # computing the gradient penalty
    def gradient_penalty(self, interpolated_sample):
        """
        This method computes the gradient penalty.
        Args:
            interpolated_sample: A tensor with interpolated samples.
        Returns:
            A tensor with the gradient penalty.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
        This method trains the WGAN_GP model for one step.
        Args:
            useless_argument: A useless argument.
        Returns:
            A dictionary with the losses of the generator and
            discriminator.
        """
        # training the discriminator
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample()
                interpolated_samples = self.get_interpolated_sample(
                    real_samples, fake_samples)

                # compute the output of the discriminator
                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)

                # compute the gradient penalty
                gp = self.gradient_penalty(interpolated_samples)

                # compute the loss of the discriminator
                disc_loss = self.discriminator.loss(real_output,
                                                    fake_output)
                gp = self.gradient_penalty(interpolated_samples)
                new_disc_loss = disc_loss + self.lambda_gp * gp

            # update the discriminator weights
            gradients_of_discriminator = disc_tape.gradient(
                new_disc_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(gradients_of_discriminator,
                    self.discriminator.trainable_variables))

        # training the generator
        with tf.GradientTape() as gen_tape:
            fake_samples = self.get_fake_sample()
            fake_output = self.discriminator(fake_samples, training=False)
            gen_loss = self.generator.loss(fake_output)
        # update the generator weights
        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))

        return {"discr_loss": disc_loss, "gen_loss": gen_loss, "gp": gp}