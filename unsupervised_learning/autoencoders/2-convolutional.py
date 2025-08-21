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
    
    for i, filter_size in enumerate(reversed_filters):
        x = keras.layers.Conv2D(
            filters=filter_size,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    
    # We need to crop or resize to get back to exact input dimensions
    # Since we have 3 upsampling steps: 4 * 2³ = 32, but we need 28
    # So we'll crop the extra pixels
    if x.shape[1] > input_dims[0] or x.shape[2] > input_dims[1]:
        x = keras.layers.Cropping2D(((2, 2), (2, 2)))(x)  # Crop 2 pixels from each side
    
    # Final convolution to get back to original dimensions
    decoded_output = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same'
    )(x)
    
    decoder = keras.Model(inputs=input_decoder, outputs=decoded_output)

    # Autoencoder (encoder + decoder)
    auto_input = keras.Input(shape=input_dims)
    encoder_output = encoder(auto_input)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=auto_input, outputs=decoder_output)

    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto