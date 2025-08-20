#!/usr/bin/env python3
"""Sparse Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    creates a sparse autoencoder
    """
    # Encoder
    input_encoder = keras.Input(shape=(input_dims,))
    encoded = input_encoder

    # Build encoder hidden layers
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    # Bottleneck layer with L1 regularization for sparsity
    encoded_output = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(encoded)

    encoder = keras.Model(inputs=input_encoder, outputs=encoded_output)

    # Decoder
    input_decoder = keras.Input(shape=(latent_dims,))
    decoded = input_decoder

    # Build decoder hidden layers (reverse order of encoder hidden layers)
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    # Final output layer with sigmoid activation
    decoded_output = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoded)

    decoder = keras.Model(inputs=input_decoder, outputs=decoded_output)

    # Autoencoder (encoder + decoder)
    auto_input = keras.Input(shape=(input_dims,))
    encoder_output = encoder(auto_input)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=auto_input, outputs=decoder_output)

    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
