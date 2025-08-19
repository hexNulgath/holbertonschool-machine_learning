#!/usr/bin/env python3
"""Vanilla Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates an autoencoder
    """
    input_layer = keras.Input(shape=(input_dims,))
    x = input_layer
    for layer in hidden_layers:
        x = keras.layers.Dense(layer, activation='relu')(x)
    bottleneck = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs=input_layer, outputs=bottleneck)

    input_layer_decoder = keras.Input(shape=(latent_dims,))
    x = input_layer_decoder
    for layer in reversed(hidden_layers):
        x = keras.layers.Dense(layer, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=input_layer_decoder, outputs=decoder_output)

    auto_output = decoder(bottleneck)
    auto = keras.Model(inputs=input_layer, outputs=auto_output)

    encoder.compile(optimizer='adam', loss='binary_crossentropy')
    decoder.compile(optimizer='adam', loss='binary_crossentropy')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
