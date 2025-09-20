#!/usr/bin/env python3
import numpy as np
import pandas as pd

def filter_columns(file):
  data = pd.read_csv(file)
  feature_selection = [
        "Timestamp", "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price",
    ]
  data = data[feature_selection]
  return data

def missing_value_interpolation(data):
  data = data.interpolate()
  return data

def remove_seconds(data):
  data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
  hourly_data = data[data['Timestamp'].dt.minute == 0]
  hourly_data = hourly_data[hourly_data['Timestamp'].dt.second == 0]
  return hourly_data

def add_features(data):
  timestamp = data["Timestamp"]
  timestamp = timestamp.apply(lambda x: x.timestamp())
  seconds_in_day = 86400
  seconds_today = timestamp % seconds_in_day
  hour_of_day = seconds_today / 3600.0
  days_since_epoch = timestamp // seconds_in_day
  day_of_week = (days_since_epoch + 3) % 7
  is_weekend = day_of_week >= 5
  is_weekend = is_weekend.astype(int)
  data['hour_of_day'] = hour_of_day
  data['day_of_week'] = day_of_week
  data['is_weekend'] = is_weekend
  
  return data

def normalize_feature(data, feature):
  mean = data[feature].mean()
  std = data[feature].std()
  name = f"{feature}_normalized"
  data[name] = (data[feature] - mean) / std
  return data, (mean, std)

def sliding_window(data, window_size):
  input = []
  output = []
  data = data.drop(columns=["Timestamp", "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price"])
  for i in range(len(data) - window_size):
    input.append(data.iloc[i:i+window_size].values)
    output.append(data["Weighted_Price_normalized"].iloc[i+window_size:i+window_size+1].values)
  return np.array(input), np.array(output)

def preprocess_data(file, output_path, window_size):
  data = filter_columns(file)
  data = missing_value_interpolation(data)
  data = remove_seconds(data)
  data = add_features(data)
  data, BTC_norm = normalize_feature(data, "Volume_(BTC)")
  data, Currency_norm = normalize_feature(data, "Volume_(Currency)")
  data, WP_norm = normalize_feature(data, "Weighted_Price")
  input, output = sliding_window(data, window_size)
  np.savez(output_path, x=input, y=output, BTC=BTC_norm, Currency=Currency_norm, WP=WP_norm)
  print("Preprocessing Done")

if __name__ == "__main__":
  preprocess_data("./coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv", 'file', 24)