#!/usr/bin/env python3
"""
Data Collection to Training Data Converter

Converts CSV data collected by DataCollectionController into the format
expected by EvolutionStrategy_simple.py and generate_training_data.py.

This script:
1. Reads CSV files from DataCollectionController
2. Converts to pandas DataFrame format
3. Creates sliding windows for LSTM training
4. Saves in the format expected by the neural network training pipeline

Usage:
    python convert_datacollection_to_training.py --input logs/datacollection-*.csv --output training_data/
"""

import pandas as pd
import numpy as np
import os
import argparse
import glob
from pathlib import Path
import time


def load_datacollection_csv(csv_path):
    """
    Load CSV data from DataCollectionController.
    
    Expected CSV format:
    timestamp,step,users,cores,response_time,target_rt,sla_violation,exploration_target
    """
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print(f"⚠️  Warning: {csv_path} is empty")
            return None
            
        # Verify required columns
        required_cols = ['users', 'cores', 'response_time']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Error: Missing required columns {missing_cols} in {csv_path}")
            return None
            
        print(f"✅ Loaded {len(df)} data points from {csv_path}")
        return df[required_cols]  # Keep only needed columns
        
    except Exception as e:
        print(f"❌ Error loading {csv_path}: {e}")
        return None


def load_multiple_csv_files(input_pattern):
    """
    Load and combine multiple CSV files from DataCollectionController.
    
    Args:
        input_pattern: Glob pattern for CSV files (e.g., "logs/datacollection-*.csv")
    
    Returns:
        Combined DataFrame with all data
    """
    csv_files = glob.glob(input_pattern)
    
    if not csv_files:
        print(f"❌ No CSV files found matching pattern: {input_pattern}")
        return None
    
    print(f"🔍 Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   - {f}")
    
    all_dataframes = []
    total_points = 0
    
    for csv_file in csv_files:
        df = load_datacollection_csv(csv_file)
        if df is not None and not df.empty:
            all_dataframes.append(df)
            total_points += len(df)
            print(f"   ✅ {csv_file}: {len(df)} points")
        else:
            print(f"   ⚠️  {csv_file}: skipped (empty or error)")
    
    if not all_dataframes:
        print("❌ No valid data found in any CSV files")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Remove duplicates and sort
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)
    
    print(f"\n📊 Combined Dataset Summary:")
    print(f"   Total points: {len(combined_df)} (from {total_points} raw points)")
    print(f"   Users range: [{combined_df['users'].min()}, {combined_df['users'].max()}]")
    print(f"   Cores range: [{combined_df['cores'].min()}, {combined_df['cores'].max()}]")
    print(f"   Response time range: [{combined_df['response_time'].min():.4f}, {combined_df['response_time'].max():.4f}]")
    
    return combined_df


def create_sliding_windows(df: pd.DataFrame, window_size: int = 5):
    """
    Convert DataFrame to supervised learning dataset with sliding windows.
    
    This replicates the format created by generate_training_data.py:
    - X_recurrent: (samples, window_size, 3) for LSTM [users, cores, response_time]
    - X_feedforward: (samples, 2) for Dense [users, cores]  
    - y_target: (samples,) target response_time
    
    Args:
        df: DataFrame with columns [users, cores, response_time]
        window_size: Size of the sliding window for LSTM
    
    Returns:
        Tuple of (X_recurrent, X_feedforward, y_target) as numpy arrays
    """
    print(f"\n🔄 Creating sliding windows (window_size={window_size})...")
    
    # Sort by cores then users for logical progression (like generate_training_data.py)
    df_sorted = df.sort_values(by=['cores', 'users']).reset_index(drop=True)
    
    X_recurrent = []
    X_feedforward = []
    y_target = []
    
    # Create sliding windows
    for i in range(window_size, len(df_sorted)):
        # Historical window for LSTM: [users, cores, response_time]
        recurrent_window = df_sorted[['users', 'cores', 'response_time']].iloc[i - window_size:i].values
        X_recurrent.append(recurrent_window)
        
        # Current state for feedforward: [users, cores]
        feedforward_input = df_sorted[['users', 'cores']].iloc[i].values
        X_feedforward.append(feedforward_input)
        
        # Target: response_time to predict
        target = df_sorted['response_time'].iloc[i]
        y_target.append(target)
    
    # Convert to numpy arrays
    X_recurrent = np.array(X_recurrent)
    X_feedforward = np.array(X_feedforward)
    y_target = np.array(y_target)
    
    print(f"✅ Created sliding windows:")
    print(f"   X_recurrent shape: {X_recurrent.shape} (samples, window_size, features)")
    print(f"   X_feedforward shape: {X_feedforward.shape} (samples, features)")
    print(f"   y_target shape: {y_target.shape} (samples,)")
    
    return X_recurrent, X_feedforward, y_target


def split_train_test(X_recurrent, X_feedforward, y_target, test_ratio=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X_recurrent, X_feedforward, y_target: Input arrays
        test_ratio: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of train and test sets
    """
    print(f"\n📚 Splitting data (test_ratio={test_ratio})...")
    
    np.random.seed(random_state)
    n_samples = len(X_recurrent)
    n_test = int(n_samples * test_ratio)
    
    # Random indices for test set
    test_indices = np.random.choice(n_samples, n_test, replace=False)
    train_indices = np.array([i for i in range(n_samples) if i not in test_indices])
    
    # Split arrays
    X_rec_train, X_rec_test = X_recurrent[train_indices], X_recurrent[test_indices]
    X_ff_train, X_ff_test = X_feedforward[train_indices], X_feedforward[test_indices]
    y_train, y_test = y_target[train_indices], y_target[test_indices]
    
    print(f"✅ Data split complete:")
    print(f"   Training set: {len(X_rec_train)} samples")
    print(f"   Test set: {len(X_rec_test)} samples")
    
    return (X_rec_train, X_ff_train, y_train), (X_rec_test, X_ff_test, y_test)


def save_training_data(output_dir, train_data, test_data):
    """
    Save training data in the format expected by EvolutionStrategy_simple.py.
    
    Saves:
    - X_recurrent.npy, X_feedforward.npy, y_target.npy (training)
    - X_recurrent_test.npy, X_feedforward_test.npy, y_target_test.npy (testing)
    """
    print(f"\n💾 Saving training data to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    X_rec_train, X_ff_train, y_train = train_data
    X_rec_test, X_ff_test, y_test = test_data
    
    # Save training data (no suffix)
    np.save(os.path.join(output_dir, 'X_recurrent.npy'), X_rec_train)
    np.save(os.path.join(output_dir, 'X_feedforward.npy'), X_ff_train)
    np.save(os.path.join(output_dir, 'y_target.npy'), y_train)
    
    # Save test data (with _test suffix)
    np.save(os.path.join(output_dir, 'X_recurrent_test.npy'), X_rec_test)
    np.save(os.path.join(output_dir, 'X_feedforward_test.npy'), X_ff_test)
    np.save(os.path.join(output_dir, 'y_target_test.npy'), y_test)
    
    print(f"✅ Saved training data files:")
    print(f"   Training: X_recurrent.npy, X_feedforward.npy, y_target.npy")
    print(f"   Testing: X_recurrent_test.npy, X_feedforward_test.npy, y_target_test.npy")
    
    # Print data statistics
    print(f"\n📊 Final Dataset Statistics:")
    print(f"   Training samples: {len(X_rec_train)}")
    print(f"   Test samples: {len(X_rec_test)}")
    print(f"   Feature dimensions: recurrent={X_rec_train.shape[1:]} feedforward={X_ff_train.shape[1:]}")
    print(f"   Target range: [{y_train.min():.4f}, {y_train.max():.4f}]")


def convert_datacollection_to_training(input_pattern, output_dir, window_size=5, test_ratio=0.2):
    """
    Main conversion function.
    
    Args:
        input_pattern: Glob pattern for input CSV files
        output_dir: Output directory for training data
        window_size: Sliding window size for LSTM
        test_ratio: Fraction of data for testing
    """
    print("🔄 Converting DataCollectionController CSV to Training Data")
    print("=" * 60)
    
    # Step 1: Load and combine CSV data
    combined_df = load_multiple_csv_files(input_pattern)
    if combined_df is None:
        print("❌ No data to process")
        return False
    
    # Step 2: Check data quality
    print(f"\n🔍 Data Quality Check:")
    n_missing = combined_df.isnull().sum().sum()
    if n_missing > 0:
        print(f"   ⚠️  Found {n_missing} missing values, removing...")
        combined_df = combined_df.dropna()
    
    n_zero_rt = (combined_df['response_time'] <= 0).sum()
    if n_zero_rt > 0:
        print(f"   ⚠️  Found {n_zero_rt} zero/negative response times, removing...")
        combined_df = combined_df[combined_df['response_time'] > 0]
    
    if len(combined_df) < window_size + 10:
        print(f"❌ Not enough data for sliding windows (need >{window_size + 10}, have {len(combined_df)})")
        return False
    
    print(f"   ✅ Clean dataset: {len(combined_df)} samples")
    
    # Step 3: Create sliding windows
    X_recurrent, X_feedforward, y_target = create_sliding_windows(combined_df, window_size)
    
    # Step 4: Split train/test
    train_data, test_data = split_train_test(X_recurrent, X_feedforward, y_target, test_ratio)
    
    # Step 5: Save data
    save_training_data(output_dir, train_data, test_data)
    
    print("\n🎉 Conversion completed successfully!")
    print(f"📁 Training data ready in: {output_dir}")
    print(f"🚀 You can now run: python EvolutionStrategy_simple.py --mode train")
    
    return True


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert DataCollectionController CSV to Neural Network Training Data"
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        default='../logs/datacollection-*.csv',
        help='Input CSV file pattern (default: ../logs/datacollection-*.csv)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='./training_data',
        help='Output directory for training data (default: ./training_data)'
    )
    
    parser.add_argument(
        '--window-size', 
        type=int, 
        default=5,
        help='Sliding window size for LSTM (default: 5)'
    )
    
    parser.add_argument(
        '--test-ratio', 
        type=float, 
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    print("🎯 DataCollection to Training Data Converter")
    print("=" * 50)
    print(f"Input pattern: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Window size: {args.window_size}")
    print(f"Test ratio: {args.test_ratio}")
    print("=" * 50)
    
    success = convert_datacollection_to_training(
        input_pattern=args.input,
        output_dir=args.output,
        window_size=args.window_size,
        test_ratio=args.test_ratio
    )
    
    if success:
        print("\n✅ Ready for neural network training!")
    else:
        print("\n❌ Conversion failed!")
        exit(1)


if __name__ == "__main__":
    main() 