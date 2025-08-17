#!/usr/bin/env python3
"""
Bayes_opt
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import GPyOpt


# import and normalize data
(X_train, Y_train),(X_test, Y_test) = keras.datasets.cifar100.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0



# One-hot encode labels for categorical_crossentropy
Y_train_cat = keras.utils.to_categorical(Y_train, 100)
Y_test_cat = keras.utils.to_categorical(Y_test, 100)

# Define the domain for GPyOpt
domain = [
    {'name': 'lr', 'type': 'continuous', 'domain': (1e-5, 1e-1)},
    {'name': 'nodes', 'type': 'discrete', 'domain': tuple(range(32, 513, 32))},
    {'name': 'activation', 'type': 'categorical', 'domain': (0, 1)},  # 0: relu, 1: tanh
    {'name': 'loss', 'type': 'categorical', 'domain': (0, 1)},  # 0: categorical_crossentropy, 1: sparse_categorical_crossentropy
    {'name': 'l2', 'type': 'continuous', 'domain': (1e-5, 1e-1)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'epochs', 'type': 'discrete', 'domain': tuple(range(5, 51, 5))},
    {'name': 'batch_size', 'type': 'discrete', 'domain': tuple(range(16, 129, 16))}
]

import numpy as np

def keras_objective(X):
    # X is shape (batch, 8)
    results = []
    for row in X:
        lr = float(row[0])
        nodes = int(row[1])
        activation = 'relu' if int(row[2]) == 0 else 'tanh'
        loss_idx = int(row[3])
        loss = 'categorical_crossentropy' if loss_idx == 0 else 'sparse_categorical_crossentropy'
        l2 = float(row[4])
        dropout = float(row[5])
        epochs = int(row[6])
        batch_size = int(row[7])

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(32, 32, 3)),
            keras.layers.Dense(nodes, activation=activation, kernel_regularizer=keras.regularizers.l2(l2)),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(100, activation='softmax')
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss=loss,
                      metrics=['accuracy'])

        # Choose correct labels for loss
        y_train = Y_train_cat if loss == 'categorical_crossentropy' else Y_train
        y_test = Y_test_cat if loss == 'categorical_crossentropy' else Y_test

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        val_loss = history.history['val_loss'][-1]
        results.append([val_loss])
    return np.array(results)

optimizer = GPyOpt.methods.BayesianOptimization(
    f=keras_objective,
    domain=domain
)
optimizer.run_optimization(max_iter=10)

# Save optimization report
with open('bayes_opt.txt', 'w') as f:
    f.write("Bayesian Optimization Report\n")
    f.write("==========================\n\n")
    f.write(f"Best hyperparameters:\n")
    f.write(f"  - Learning Rate: {optimizer.x_opt[0]:.6f}\n")
    f.write(f"  - Hidden Nodes: {int(optimizer.x_opt[1])}\n")
    f.write(f"  - Activation: {'relu' if optimizer.x_opt[2] == 0 else 'tanh'}\n")
    f.write(f"  - Loss Function: {'categorical_crossentropy' if optimizer.x_opt[3] == 0 else 'sparse_categorical_crossentropy'}\n")
    f.write(f"  - L2 Regularization: {optimizer.x_opt[4]:.6f}\n")
    f.write(f"  - Dropout Rate: {optimizer.x_opt[5]:.4f}\n")
    f.write(f"  - Epochs: {int(optimizer.x_opt[6])}\n")
    f.write(f"  - Batch Size: {int(optimizer.x_opt[7])}\n")
    f.write(f"\nBest Validation Loss: {optimizer.fx_opt:.4f}\n")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(optimizer.Y, 'r-', label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Validation Loss')
plt.title('Bayesian Optimization Convergence')
plt.grid()
plt.legend()
plt.savefig('convergence_plot.png')
plt.show()
