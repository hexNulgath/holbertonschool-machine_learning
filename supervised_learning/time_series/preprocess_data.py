import tensorflow as tf
import math

def preprocess_data(csv_1, window_size=60, k_ahead=1):
    # Define column types explicitly
    column_defaults = [
        tf.float64,  # Timestamp (Unix time - usually seconds)
        tf.float32,  # Close (label)
        tf.float32,  # Volume_(BTC)
        tf.float32,  # Volume_(Currency)
        tf.float32   # Weighted_Price
    ]
    
    # Use the lower-level CsvDataset for more control
    dataset = tf.data.experimental.CsvDataset(
        csv_1,
        record_defaults=column_defaults,
        header=True,
        select_cols=[0, 1, 2, 3, 4]  # Select columns by index
    )
    
    # Convert to (features, label) format
    def structure_data(timestamp, close, volume_btc, volume_currency, weighted_price):
        features = {
            "Timestamp": timestamp,
            "Volume_(BTC)": volume_btc,
            "Volume_(Currency)": volume_currency,
            "Weighted_Price": weighted_price
        }
        return features, close
    
    dataset = dataset.map(structure_data)
    
    def process_row(features, label):
        timestamp = features["Timestamp"]
        
        # CONVERT UNIX TIMESTAMP TO TIME FEATURES
        # Assuming timestamp is in SECONDS
        seconds_in_day = 86400  # 24 * 60 * 60
        
        # Calculate seconds elapsed in current day
        seconds_today = timestamp % seconds_in_day
        
        # Convert to hour of day (0-23)
        hour_of_day = seconds_today / 3600.0
        
        # Calculate day of week (0=Monday, 6=Sunday)
        # January 1, 1970 was a Thursday (which is day 3 if Monday=0)
        days_since_epoch = timestamp // seconds_in_day
        day_of_week = (days_since_epoch + 3) % 7
        
        # Weekend flag (Saturday=5, Sunday=6)
        is_weekend = tf.cast(day_of_week >= 5, tf.float32)
        
        # Add all time features
        features.update({
            "Timestamp": hour_of_day,
            "DayOfWeek": tf.cast(day_of_week, tf.float32),
            "IsWeekend": is_weekend,
        })
        
        # VALIDATION CHECK - must come AFTER all calculations
        is_valid = tf.reduce_all([
            tf.math.is_finite(timestamp),
            tf.math.is_finite(features["Volume_(BTC)"]),
            tf.math.is_finite(features["Volume_(Currency)"]),
            tf.math.is_finite(features["Weighted_Price"]),
            tf.math.is_finite(label),
            tf.math.is_finite(hour_of_day),
            timestamp > 0  # Valid Unix timestamp should be positive
        ])
        
        return features, label, is_valid
    
    def add_technical_indicators(features, label):
        """Add technical indicators - this should be a separate step"""
        # Simple price-volume ratio
        price_volume_ratio = features["Weighted_Price"] / (features["Volume_(BTC)"] + 1e-6)
        features["PriceVolumeRatio"] = price_volume_ratio
        
        # Volume difference
        volume_diff = features["Volume_(BTC)"] - features["Volume_(Currency)"]
        features["VolumeDiff"] = volume_diff
        
        return features, label
    
    # Processing pipeline
    dataset = dataset.map(process_row)
    dataset = dataset.filter(lambda features, label, is_valid: is_valid)
    dataset = dataset.map(lambda features, label, is_valid: (features, label))
    
    # Add technical indicators after validation
    dataset = dataset.map(add_technical_indicators)
    
    # Batch and optimize
    dataset = dataset.batch(64).prefetch(tf.data.AUTOTUNE)
    visualize_dataset(dataset, num_batches=1, num_samples=64)

    def create_sequences(features, label):
        """Create sequences for time series forecasting"""
        # Get the length of the sequence
        seq_length = tf.shape(label)[0]
        
        # Calculate how many complete sequences we can create
        n_sequences = seq_length - window_size - k_ahead + 1
        
        # Create feature sequences
        feature_sequences = {}
        for key, values in features.items():
            # Create windows using tensor slicing
            windows = tf.TensorArray(dtype=values.dtype, size=n_sequences)
            
            for i in range(n_sequences):
                window = values[i:i + window_size]
                windows = windows.write(i, window)
            
            feature_sequences[key] = windows.stack()
        
        # Create target sequences
        if k_ahead == 1:
            targets = label[window_size:window_size + n_sequences]
        else:
            targets = tf.TensorArray(dtype=label.dtype, size=n_sequences)
            for i in range(n_sequences):
                target_window = label[i + window_size:i + window_size + k_ahead]
                targets = targets.write(i, target_window)
            targets = targets.stack()
        
        return feature_sequences, targets
    # Apply sequence creation
    dataset = dataset.unbatch().batch(window_size + k_ahead + 100, drop_remainder=False)
    dataset = dataset.map(create_sequences)
    
    # Flatten the sequences
    dataset = dataset.flat_map(lambda features, labels: 
        tf.data.Dataset.from_tensor_slices((features, labels)))

    return dataset

# Function to visualize batch output
def visualize_dataset(dataset, num_batches=1, num_samples=3):
    print(f"\n{'='*60}")
    print("VISUALIZING DATASET BATCHES")
    print(f"{'='*60}")
    
    batch_count = 0
    for features_batch, labels_batch in dataset:
        batch_count += 1
        print(f"\n--- BATCH {batch_count} ---")
        print(f"Batch size: {labels_batch.shape[0]}")
        print(f"Labels shape: {labels_batch.shape}")
        
        # Print sample data from this batch
        for i in range(min(num_samples, labels_batch.shape[0])):
            print(f"\nSample {i+1}:")
            print(f"  Close (label): {labels_batch[i].numpy():.2f}")
            print(f"  Timestamp: {features_batch['Timestamp'][i].numpy():.2f}")
            print(f"  DayOfWeek: {features_batch['DayOfWeek'][i].numpy():.0f}")
            print(f"  IsWeekend: {features_batch['IsWeekend'][i].numpy():.0f}")
            print(f"  Volume_(BTC): {features_batch['Volume_(BTC)'][i].numpy():.6f}")
            print(f"  Weighted_Price: {features_batch['Weighted_Price'][i].numpy():.2f}")
            print(f"  PriceVolumeRatio: {features_batch['PriceVolumeRatio'][i].numpy():.6f}")
        
        # Print all feature shapes
        print(f"\nFeature shapes in batch:")
        for key, value in features_batch.items():
            print(f"  {key}: {value.shape}")
        
        if batch_count >= num_batches:
            break
    
    print(f"\n{'='*60}")
    print("END OF VISUALIZATION")
    print(f"{'='*60}")