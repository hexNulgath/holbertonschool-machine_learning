import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
"""
WGAN_clip module
----------------
Implements a Wasserstein GAN (WGAN) with weight
clipping using TensorFlow/Keras.

Classes:
    WGAN_clip: Subclass of keras.Model implementing
    WGAN training with weight clipping.
"""


class WGAN_clip(keras.Model):
    """
    Wasserstein GAN with weight clipping.

    Args:
        generator: Keras model, the generator network.
        discriminator: Keras model, the discriminator (critic) network.
        latent_generator: Callable, generates latent vectors for the generator.
        real_examples: Tensor, dataset of real samples.
        batch_size: int, batch size for training.
        disc_iter: int, number of discriminator updates per generator update.
        learning_rate: float, learning rate for optimizers.
    """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """
        Initialize the WGAN_clip model and set up optimizers
        and loss functions.
        """
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary

        self.generator.loss = lambda x: -WGAN_clip.wasserstein_loss(
            tf.ones_like(x), x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1,
            beta_2=self.beta_2)

        self.discriminator.loss = lambda x, y: WGAN_clip.wasserstein_loss(
            tf.ones_like(y), y) + WGAN_clip.wasserstein_loss(
                -tf.ones_like(x), x)
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)

    # generator of real samples of size batch_size
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

    # generator of fake samples of size batch_size
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

    # overloading train_step()
    def train_step(self, useless_argument):
        """
        Custom training step for WGAN.
        Performs multiple discriminator updates and one generator update.

        Args:
            useless_argument: Placeholder for compatibility with Keras API.
        Returns:
            Dictionary with discriminator and generator losses.
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
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1, 1))

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

    @staticmethod
    def wasserstein_loss(y_true, y_pred):
        """
        Wasserstein loss function.

        Args:
            y_true: Tensor, target labels (+1 or -1).
            y_pred: Tensor, predicted values.
        Returns:
            Mean of y_true * y_pred.
        """
        return tf.reduce_mean(y_true * y_pred)
