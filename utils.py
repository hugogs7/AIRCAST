# =============================================================================
# utils.py — Beijing Air Quality Project
# Helper functions for data ingestion, preprocessing, and feature engineering.
# Import this module at the top of your main notebook.
# =============================================================================

import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# -----------------------------------------------------------------------------
# ingest_beijing_data
# -----------------------------------------------------------------------------
# Scans the given folder for all CSV files matching the Beijing dataset naming
# convention (PRSA_Data_*.csv), reads each one into a DataFrame and concatenates
# them into a single table.
#
# Parameters:
#   folder_path (str): Path to the folder containing the CSV files.
#                      Defaults to "." (current working directory).
#
# Returns:
#   pd.DataFrame: Consolidated DataFrame with all stations.
#                 Returns an empty DataFrame if no files are found.
# -----------------------------------------------------------------------------
def ingest_beijing_data(folder_path="."):
    all_files = glob.glob(os.path.join(folder_path, "PRSA_Data_*.csv"))

    if not all_files:
        print("No CSV files found! Make sure the Beijing dataset is in the folder.")
        return pd.DataFrame()

    df_list = []
    for filename in all_files:
        df_temp = pd.read_csv(filename)
        df_list.append(df_temp)

    # Consolidate all 12 stations into one DataFrame
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Successfully ingested {len(all_files)} files.")
    print(f"Total instances: {full_df.shape[0]}")
    return full_df


# -----------------------------------------------------------------------------
# preprocess_beijing_time
# -----------------------------------------------------------------------------
# Prepares the raw DataFrame for modelling by:
#   1. Building a proper datetime index from the separate year/month/day/hour cols.
#   2. Sorting chronologically so time-series splits are leakage-free.
#   3. Label-encoding the two categorical columns:
#        - 'wd'      (wind direction)  → integer codes in-place
#        - 'station' (station name)    → new column 'station_code'
#
# Parameters:
#   df (pd.DataFrame): Raw consolidated DataFrame from ingest_beijing_data().
#
# Returns:
#   pd.DataFrame: Processed DataFrame indexed by timestamp, ready for
#                 imputation and scaling.
# -----------------------------------------------------------------------------
def preprocess_beijing_time(df):
    df_p = df.copy()

    # 1. Build a single datetime column from the four time components
    df_p['timestamp'] = pd.to_datetime(df_p[['year', 'month', 'day', 'hour']])

    # 2. Set timestamp as index and sort ascending to preserve temporal order
    df_p = df_p.set_index('timestamp').sort_index()

    # 3. Encode wind direction as integer codes (16 compass points + NaN → 'nan')
    le_wd = LabelEncoder()
    df_p['wd'] = le_wd.fit_transform(df_p['wd'].astype(str))

    # 4. Encode station names into a numeric column for optional use in models
    le_station = LabelEncoder()
    df_p['station_code'] = le_station.fit_transform(df_p['station'])

    print("Timestamps aligned and categorical variables encoded.")
    return df_p


# -----------------------------------------------------------------------------
# create_multivariate_windows
# -----------------------------------------------------------------------------
# Converts a time-series DataFrame into supervised learning windows.
# For each station independently, a sliding window of `lookback` hours is
# extracted as a flat feature vector (X), and the target value `horizon`
# steps ahead is used as the label (y).
#
# Processing per station separately prevents the window from accidentally
# mixing data from two different monitoring locations.
#
# Parameters:
#   df          (pd.DataFrame): Preprocessed DataFrame (post imputation+scaling).
#   feature_cols (list[str])  : Column names to include as input features.
#   target_col   (str)        : Name of the column to predict (e.g. 'PM2.5').
#   lookback     (int)        : Number of past hours used as context. Default 24.
#   horizon      (int)        : How many hours ahead to predict. Default 1.
#
# Returns:
#   X (np.ndarray): Shape (n_samples, lookback * len(feature_cols))
#                   Each row is a flattened multivariate window.
#   y (np.ndarray): Shape (n_samples,)
#                   Target value at t + horizon for each window.
# -----------------------------------------------------------------------------
def create_multivariate_windows(df, feature_cols, target_col, lookback=24, horizon=1):
    X, y = [], []

    for station in df['station'].unique():
        # Isolate one station to avoid cross-location contamination
        station_df = df[df['station'] == station].copy()

        feature_data = station_df[feature_cols].values  # shape: (T, n_features)
        target_data  = station_df[target_col].values    # shape: (T,)

        # Slide the window across the time axis
        for i in range(lookback, len(station_df) - horizon + 1):
            # Flatten (lookback, n_features) → 1-D vector
            X.append(feature_data[i - lookback:i, :].flatten())
            # Label: value at position (i + horizon - 1)
            y.append(target_data[i + horizon - 1])

    return np.array(X), np.array(y)