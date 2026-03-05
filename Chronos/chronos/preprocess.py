"""
Basic Preprocessing Utilities – Standalone Module for Chronos
===============================================================
This module provides lightweight, function-based preprocessing helpers used by
the simple standalone training script (chronos/training.py). It handles:

  • load_timeseries  – read a CSV file with columns (timestamp, series_id, value).
  • pivot_series     – convert the long-format DataFrame into a dict of
                       {series_id: pd.Series} indexed by timestamp.
  • resample_and_fill – resample a pd.Series to a regular frequency and fill
                        any NaN gaps via forward-fill + interpolation.
  • sliding_windows  – create overlapping (X, y) input/target pairs from a
                       1-D numpy array, which is the core data transformation
                       for supervised time-series forecasting.

Note: The more feature-rich preprocessing pipeline lives in
chronos/data/preprocessor.py (the Preprocessor class with z-score normalization
and inverse_transform). This module is kept for the simple training script.
"""

import pandas as pd
import numpy as np


def load_timeseries(path):
    """Load a CSV file containing time-series data into a DataFrame.

    Expected CSV columns:
        timestamp  – datetime string (parsed automatically by pandas)
        series_id  – identifier grouping rows into distinct time series
        value      – the numeric measurement at each timestamp

    Args:
        path: File path to the CSV.

    Returns:
        pd.DataFrame with a datetime-typed 'timestamp' column.
    """
    # parse_dates converts the 'timestamp' column from string to datetime64
    # automatically, which is required for later resampling operations.
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df


def pivot_series(df):
    """Convert a long-format DataFrame into a dict of per-series pd.Series.

    Groups the DataFrame by 'series_id' and creates one pd.Series per group,
    indexed by timestamp and sorted chronologically. This format is convenient
    for per-series preprocessing (resampling, normalization).

    Args:
        df: DataFrame with columns [timestamp, series_id, value].

    Returns:
        Dictionary mapping each series_id (str) to a pd.Series of float values
        indexed by timestamp.
    """
    series = {}
    for sid, g in df.groupby('series_id'):
        # Set timestamp as the index, extract the 'value' column, cast to float,
        # and sort chronologically so time flows left-to-right.
        s = g.set_index('timestamp')['value'].astype(float).sort_index()
        series[sid] = s
    return series


def resample_and_fill(s, freq='1T'):
    """Resample a time series to a regular frequency and fill missing values.

    Real-world sensor data often arrives at irregular intervals. This function
    forces the series onto a regular grid (e.g., every 1 minute) and fills
    any resulting NaN gaps using:
      1. Forward-fill (ffill) – propagate the last known value forward.
      2. Linear interpolation – fill remaining interior NaNs.

    Args:
        s: pd.Series indexed by datetime.
        freq: Pandas offset alias for the target frequency. '1T' = 1 minute,
              '1H' = 1 hour, '1D' = 1 day, etc.

    Returns:
        pd.Series resampled to the specified frequency with no NaN values.
    """
    # resample().mean() places values onto the regular grid; timestamps that
    # had no observations become NaN.
    s2 = s.resample(freq).mean()

    # Forward-fill propagates the last valid observation to the next NaN(s).
    # interpolate() then linearly fills any remaining interior gaps.
    s2 = s2.ffill().interpolate()
    return s2


def sliding_windows(arr, input_len=60, horizon=1, stride=1):
    """Create overlapping sliding-window samples for supervised forecasting.

    Given a 1-D array of time-series values, this function generates
    (input, target) pairs where:
      - input (X) = arr[i : i + input_len]        (the look-back window)
      - target (y) = arr[i + input_len : i + input_len + horizon]  (future values)

    The window slides forward by `stride` positions each step. With stride=1,
    every possible window is extracted (maximum data utilization).

    Args:
        arr: 1-D numpy array of time-series values.
        input_len: Number of past timesteps in each input window.
        horizon: Number of future timesteps to predict.
        stride: Step size between consecutive windows.

    Returns:
        Tuple (X, y) where:
            X has shape (n_samples, input_len)
            y has shape (n_samples, horizon)
    """
    x = []  # Will collect input windows.
    y = []  # Will collect corresponding target windows.
    n = len(arr)

    # Slide from position 0 up to the last valid starting index.
    # The last valid start is n - input_len - horizon (inclusive).
    for i in range(0, n - input_len - horizon + 1, stride):
        # Extract the input window: `input_len` consecutive timesteps.
        x.append(arr[i:i+input_len])
        # Extract the target: the next `horizon` timesteps immediately after the input.
        y.append(arr[i+input_len:i+input_len+horizon])

    # Convert lists of arrays to 2-D numpy arrays.
    return np.array(x), np.array(y)
