#!/usr/bin/env python3
"""
Bayesian Optimization with Transfer Learning (EfficientNetV2S)
"""
import tensorflow as tf
from tensorflow import keras as K
import matplotlib.pyplot as plt
import GPyOpt


def preproces(x, y):
    """Preprocess function to resize and normalize images"""

    x_p = K.applications.efficientnet_v2.preprocess_input(x)
    y_p = K.utils.to_categorical(y, num_classes=10)
    return x_p, y_p


def build_model(dropout_rate=0.5):
    """Build and compile the EfficientNetV2S model"""

    inputs = K.Input(shape=(32, 32, 3))
    resize_layer = K.layers.Resizing(260, 260)(inputs)

    base_model = K.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(260, 260, 3),
        pooling='avg'
    )

    base_model.trainable = False
    x = base_model(resize_layer, training=False)
    x = K.layers.Flatten()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(dropout_rate)(x)
    x = K.layers.BatchNormalization()(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.models.Model(inputs, outputs)

    return model


def Train(
        epochs=50, batch_size=64, beta_1=0.9,
        beta_2=0.999, dropout_rate=0.5, learning_rate=0.001
        ):
    """Function to load data, preprocess, and build the model"""

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    val_size = int(0.15 * len(x_train))  # 15% for validation
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]
    x_train_p, y_train_p = preproces(x_train, y_train)
    x_val_p, y_val_p = preproces(x_val, y_val)
    x_test_p, y_test_p = preproces(x_test, y_test)

    model = build_model(dropout_rate=dropout_rate)

    model.compile(
        optimizer=K.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    file_name = f"epochs={epochs}-batch={batch_size}-B1={beta_1}\
        -B2={beta_2}-Dropout={dropout_rate}-learning_rate={learning_rate}.weights.h5"

    callbacks = [
        K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        ),
        K.callbacks.ModelCheckpoint(
            filepath=file_name,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )
    ]

    history = model.fit(
        x_train_p, y_train_p,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val_p, y_val_p),
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(x_test_p, y_test_p)
    print(f'Test accuracy: {test_acc:.4f}')
    return model, history


def bayesian_optimization():
    """Function to perform Bayesian Optimization on the model"""

    bounds = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)},
        {'name': 'epochs', 'type': 'discrete', 'domain': (10, 20, 30)},
        {'name': 'beta_1', 'type': 'continuous', 'domain': (0.9, 0.999)},
        {'name': 'beta_2', 'type': 'continuous', 'domain': (0.999, 0.9999)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
    ]

    def objective_function(params):
        learning_rate = params[0][0]
        batch_size = int(params[0][1])
        epochs = int(params[0][2])
        beta_1 = params[0][3]
        beta_2 = params[0][4]
        dropout_rate = params[0][5]

        model, history = Train(
            epochs=epochs,
            batch_size=batch_size,
            beta_1=beta_1,
            beta_2=beta_2,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )

        return -history.history['val_accuracy'][-1]

    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective_function,
        domain=bounds,
        acquisition_type='EI',
    )

    optimizer.run_optimization(max_iter=30)
    print("Best parameters found: ", optimizer.x_opt)
    print("Best validation accuracy: ", -optimizer.fx_opt)
    optimizer.plot_convergence()
    plt.savefig('convergence.png')

    with open('bayes_opt.txt', 'w') as f:
        f.write(f"Best parameters found: {optimizer.x_opt}\n")
        f.write(f"Best validation accuracy: {-optimizer.fx_opt}\n")


if __name__ == "__main__":
    bayesian_optimization()
