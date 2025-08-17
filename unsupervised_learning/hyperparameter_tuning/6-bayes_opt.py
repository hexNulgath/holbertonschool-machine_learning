#!/usr/bin/env python3
"""
Bayesian Optimization with Transfer Learning (EfficientNetV2S)
"""
import tensorflow as tf
from tensorflow import keras as K
import matplotlib.pyplot as plt
import GPyOpt
import numpy as np

# Load CIFAR-100 dataset
(X_train, Y_train), (X_test, Y_test) = K.datasets.cifar100.load_data()

# Preprocess function for EfficientNetV2S
def preprocess_data(X, Y):
    """Preprocess data for EfficientNetV2S model"""
    X_p = K.applications.efficientnet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=100)
    return X_p, Y_p

# Preprocess the data
X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

# Define the domain for GPyOpt
domain = [
    {'name': 'lr', 'type': 'continuous', 'domain': (1e-5, 1e-3)},
    {'name': 'dense_units', 'type': 'discrete', 'domain': tuple(range(128, 513, 64))},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'epochs', 'type': 'discrete', 'domain': tuple(range(5, 26, 5))},
    {'name': 'batch_size', 'type': 'discrete', 'domain': tuple(range(32, 129, 32))},
    {'name': 'fine_tune_lr', 'type': 'continuous', 'domain': (1e-6, 1e-4)},
    {'name': 'fine_tune_epochs', 'type': 'discrete', 'domain': tuple(range(5, 26, 5))}
]

def build_transfer_model(params):
    """Build the transfer learning model with given hyperparameters"""
    inputs = K.Input(shape=(32, 32, 3))
    resize_layer = K.layers.Resizing(260, 260)(inputs)
    
    # Create base model
    base_model = K.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(260, 260, 3),
        pooling='avg'
    )
    base_model.trainable = False
    
    x = base_model(resize_layer, training=False)
    x = K.layers.Flatten()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(
        int(params['dense_units']), 
        activation='relu',
        kernel_regularizer=K.regularizers.l2(params['l2'])
    )(x)
    x = K.layers.Dropout(params['dropout'])(x)
    x = K.layers.BatchNormalization()(x)
    outputs = K.layers.Dense(100, activation="softmax")(x)
    
    model = K.models.Model(inputs, outputs)
    return model

def keras_objective(X):
    """Objective function for Bayesian optimization"""
    results = []
    for row in X:
        params = {
            'lr': float(row[0]),
            'dense_units': int(row[1]),
            'dropout': float(row[2]),
            'l2': float(row[3]),
            'epochs': int(row[4]),
            'batch_size': int(row[5]),
            'fine_tune_lr': float(row[6]),
            'fine_tune_epochs': int(row[7])
        }
        
        # Build and compile initial model
        model = build_transfer_model(params)
        model.compile(
            optimizer=K.optimizers.Adam(params['lr']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train initial model
        history = model.fit(
            X_train_p, Y_train_p,
            validation_data=(X_test_p, Y_test_p),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            verbose=0
        )
        
        # Fine-tune last layers
        base_model = model.layers[2]
        base_model.trainable = True
        for layer in base_model.layers[:-10]:
            layer.trainable = False
            
        model.compile(
            optimizer=K.optimizers.Adam(params['fine_tune_lr']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history_fine = model.fit(
            X_train_p, Y_train_p,
            validation_data=(X_test_p, Y_test_p),
            epochs=params['fine_tune_epochs'],
            batch_size=params['batch_size'],
            verbose=0
        )
        
        # Get best validation accuracy
        val_acc = max(history.history['val_accuracy'] + history_fine.history['val_accuracy'])
        results.append([-val_acc])  # Negative because we want to maximize accuracy
        
    return np.array(results)

# Run Bayesian optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=keras_objective,
    domain=domain,
    acquisition_type='EI'
)
optimizer.run_optimization(max_iter=15)

# Save optimization report
with open('transfer_bayes_opt.txt', 'w') as f:
    f.write("Bayesian Optimization Report (Transfer Learning)\n")
    f.write("==============================================\n\n")
    f.write(f"Best hyperparameters:\n")
    f.write(f"  - Initial Learning Rate: {optimizer.x_opt[0]:.6f}\n")
    f.write(f"  - Dense Units: {int(optimizer.x_opt[1])}\n")
    f.write(f"  - Dropout Rate: {optimizer.x_opt[2]:.4f}\n")
    f.write(f"  - L2 Regularization: {optimizer.x_opt[3]:.6f}\n")
    f.write(f"  - Initial Epochs: {int(optimizer.x_opt[4])}\n")
    f.write(f"  - Batch Size: {int(optimizer.x_opt[5])}\n")
    f.write(f"  - Fine-tune Learning Rate: {optimizer.x_opt[6]:.6f}\n")
    f.write(f"  - Fine-tune Epochs: {int(optimizer.x_opt[7])}\n")
    f.write(f"\nBest Validation Accuracy: {-optimizer.fx_opt:.4f}\n")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(-optimizer.Y, 'b-', label='Validation Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Validation Accuracy')
plt.title('Bayesian Optimization Convergence (Transfer Learning)')
plt.grid()
plt.legend()
plt.savefig('transfer_convergence_plot.png')
plt.show()

# Train final model with best parameters
best_params = {
    'lr': optimizer.x_opt[0],
    'dense_units': int(optimizer.x_opt[1]),
    'dropout': optimizer.x_opt[2],
    'l2': optimizer.x_opt[3],
    'epochs': int(optimizer.x_opt[4]),
    'batch_size': int(optimizer.x_opt[5]),
    'fine_tune_lr': optimizer.x_opt[6],
    'fine_tune_epochs': int(optimizer.x_opt[7])
}

final_model = build_transfer_model(best_params)
final_model.compile(
    optimizer=K.optimizers.Adam(best_params['lr']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train initial phase
final_model.fit(
    X_train_p, Y_train_p,
    validation_data=(X_test_p, Y_test_p),
    epochs=best_params['epochs'],
    batch_size=best_params['batch_size']
)

# Fine-tune phase
base_model = final_model.layers[2]
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False
    
final_model.compile(
    optimizer=K.optimizers.Adam(best_params['fine_tune_lr']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

final_model.fit(
    X_train_p, Y_train_p,
    validation_data=(X_test_p, Y_test_p),
    epochs=best_params['fine_tune_epochs'],
    batch_size=best_params['batch_size']
)

# Save final model
final_model.save('cifar100_transfer.h5')

# Evaluate final model
_, test_acc = final_model.evaluate(X_test_p, Y_test_p, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
