#!/usr/bin/env python3
import tensorflow.keras as K
preprocess_data = __import__('0-transfer').preprocess_data

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10v2.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
