#!/usr/bin/env python3
from tensorflow.data import Dataset
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_sets(file):
  data = np.load(file)
  x = data['x']
  y = data['y']
  BTC_norm = data['BTC']
  Currency_norm = data['Currency']
  WP_norm = data['WP']

  train_size = int(0.7 * len(x))
  val_size = int(0.2 * len(x))
  test_size = len(x) - train_size - val_size
  x_train, y_train = x[:train_size], y[:train_size]
  x_val, y_val = x[train_size:train_size+val_size], y[train_size:train_size+val_size]
  x_test, y_test = x[-test_size:], y[-test_size:]

  train_dataset = Dataset.from_tensor_slices((x_train, y_train))
  val_dataset = Dataset.from_tensor_slices((x_val, y_val))
  test_dataset = Dataset.from_tensor_slices((x_test, y_test))
  return train_dataset, val_dataset, test_dataset, BTC_norm, Currency_norm, WP_norm

def create_model():
  model = Sequential()
  model.add(LSTM(units=50, return_sequences=False, input_shape=(24, 6)))
  model.add(Dense(units=1))
  model.compile(optimizer='adam', loss='mean_squared_error')
  return model

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, BTC_norm, Currency_norm, WP_norm = get_sets('./file.npz')
    model = create_model()

    checkpoint_callback = ModelCheckpoint(
        filepath='btc_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=0
    )

    model.fit(train_dataset.batch(64), epochs=30, validation_data=val_dataset.batch(64), callbacks=[checkpoint_callback])
  
  # Collect batched validation data
    test_X = []
    test_y = []
    for batch_x, batch_y in test_dataset.batch(64):
        test_X.append(batch_x)
        test_y.append(batch_y)

    # Concatenate batches
    test_X = np.concatenate(test_X, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    # Make prediction
    model = load_model('btc_model.keras')
    predictions = model.predict(test_X)
    predictions = predictions * WP_norm + WP_norm
    test_y = test_y * WP_norm + WP_norm

    mape = np.mean(np.abs((test_y - predictions) / test_y)) * 100
    print("Test MAPE:", mape)
    print(f"Predicted Price: {predictions[-1][0]}, Actual Price: {test_y[-1][0]}")