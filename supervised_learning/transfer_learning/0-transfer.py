#!/usr/bin/env python3
"""transfer learning"""
from tensorflow import keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    pre-processes the data for your model:

    X is a numpy.ndarray of shape (m, 32, 32, 3)
    containing the CIFAR 10 data, where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    # Use standard preprocessing for EfficientNetB2
    X_p = K.applications.efficientnet.preprocess_input(X)

    # Convert Y to one-hot encoding
    Y_p = K.utils.to_categorical(Y, num_classes=10)

    return X_p, Y_p


def build_model():
    # CIFAR-10 input shape
    inputs = K.Input(shape=(32, 32, 3))

    # Lambda layer to resize images to 260x260 (required by EfficientNetB2)
    resize_layer = K.layers.Resizing(260, 260)(inputs)

    # Create base model from pre-trained model (excluding top layers)
    base_model = K.applications.EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_shape=(260, 260, 3),
        pooling='avg'
    )

    # Freeze the base model layers
    base_model.trainable = False
    x = base_model(resize_layer, training=False)
    x = K.layers.Flatten()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.BatchNormalization()(x)
    outputs = K.layers.Dense(10, activation="softmax")(x)

    return K.models.Model(inputs, outputs)


def train_model():
    """
    Trains a convolutional neural network to classify CIFAR-10 dataset
    Uses transfer learning with Keras Application EfficientNetB2
    Returns: trained model in the current working directory as cifar10.h5
    """

    # Load CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # validation split
    # Shuffle and split (80% train, 20% validation)
    val_size = int(0.15 * len(X_train))  # 15% for validation

    X_val, Y_val = X_train[:val_size], Y_train[:val_size]
    X_train, Y_train = X_train[val_size:], Y_train[val_size:]

    # Preprocess data
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_val_p, Y_val_p = preprocess_data(X_val, Y_val)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Build the model
    model = build_model()

    # Compile model
    model.compile(
        optimizer=K.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Add callbacks
    callbacks = [
        K.callbacks.ModelCheckpoint(
            'cifar10.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='avg',
        ),
        K.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
    ]

    # Add data augmentation
    datagen = K.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train_p)
    # Train the model
    model.fit(
        datagen.flow(X_train_p, Y_train_p, batch_size=64),
        validation_data=(X_val_p, Y_val_p),
        epochs=20,
        callbacks=callbacks
    )

    base_model = model.layers[1]  # Get the base model
    base_model.trainable = True  # Unfreeze the base model
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    fine_tune_optimizer = K.optimizers.Adam(1e-5)
    model.compile(
        optimizer=fine_tune_optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    fine_tuning_history = model.fit(
        datagen.flow(X_train_p, Y_train_p, batch_size=64),
        validation_data=(X_val_p, Y_val_p),
        epochs=10,
        callbacks=callbacks
    )

    # Save the model
    model.save(filepath='cifar10.h5', save_format='h5')

    # Evaluate on test set
    _, test_acc = model.evaluate(
        X_test_p, Y_test_p, batch_size=128, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    train_model()
