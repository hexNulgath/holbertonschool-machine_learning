#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
preprocess_data = __import__('preprocess_data').preprocess_data

def forecast_btc(model, dataset):
    """
    forecast_btc forecasts the Bitcoin price using the provided model and dataset
    Args:
        model: tf.keras.Model - the trained model to use for forecasting
        dataset: tf.data.Dataset - the preprocessed dataset to use for forecasting
    Returns: np.ndarray - the forecasted Bitcoin prices
    """
    predictions = model.predict(dataset)
    return predictions

if __name__ == "__main__":
    dataset = preprocess_data("bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv")
    