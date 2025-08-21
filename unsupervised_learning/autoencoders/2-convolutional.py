#!/usr/bin/env python3
"""convolutional Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    creates a convolutional autoencoder
    """
    # Encoder
    input_encoder = keras.Input(shape=input_dims)
    x = input_encoder

    # Build encoder convolutional layers
    for filter_size in filters:
        x = keras.layers.Conv2D(
            filters=filter_size,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(x)

    # Final convolution to match latent_dims
    encoded_output = keras.layers.Conv2D(
        filters=latent_dims[-1],
        kernel_size=(3, 3),
        activation='relu',
        padding='same'
    )(x)

    encoder = keras.Model(inputs=input_encoder, outputs=encoded_output)

    # Decoder
    input_decoder = keras.Input(shape=latent_dims)
    x = input_decoder

    # Build decoder convolutional layers (reverse order of filters)
    reversed_filters = filters[::-1]
    same_pad = reversed_filters[:-1]

    for i, filter_size in enumerate(same_pad):
        x = keras.layers.Conv2D(
            filters=filter_size,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    # Last two filters
    x = keras.layers.Conv2D(
        filters=reversed_filters[-1],
        kernel_size=(3, 3),
        activation='relu',
        padding='valid'
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    decoder_output = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same'
    )(x)

    decoder = keras.Model(inputs=input_decoder, outputs=decoder_output)

    # Autoencoder (encoder + decoder)
    auto_input = keras.Input(shape=input_dims)
    encoder_output = encoder(auto_input)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=auto_input, outputs=decoder_output)

    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
