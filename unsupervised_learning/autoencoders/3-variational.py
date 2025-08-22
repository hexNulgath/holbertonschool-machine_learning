#!/usr/bin/env python3
"""variational"""
import tensorflow.keras as keras
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates a variational autoencoder
    """
    # sampling helper function
    def sampling(args):
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon
    
    # build the encoder network
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    for units in hidden_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    z_mean = keras.layers.Dense(latent_dims, name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dims, name="z_log_var")(x)

    # Sampling layer
    z = keras.layers.Lambda(sampling, name="z")([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z, z_mean, z_log_var], name="encoder")

    # build the decoder network
    latent_inputs = keras.Input(shape=(latent_dims,), name='z_sampling')
    x = latent_inputs

    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation="relu")(x)

    decoder_outputs = keras.layers.Dense(input_dims, activation="sigmoid")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # Build the VAE model
    z_output, z_mean_output, z_log_var_output = encoder(encoder_inputs)
    outputs = decoder(z_output)
    vae = keras.Model(encoder_inputs, outputs, name="vae")

    # VAE loss: reconstruction + KL divergence
    # Use Keras backend operations instead of direct TensorFlow ops
    reconstruction_loss = keras.losses.binary_crossentropy(encoder_inputs, outputs)
    reconstruction_loss *= input_dims
    
    # Calculate KL loss using only Keras backend operations
    kl_loss = 1 + z_log_var_output
    kl_loss -= keras.backend.square(z_mean_output)
    kl_loss -= keras.backend.exp(z_log_var_output)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    
    return encoder, decoder, vae