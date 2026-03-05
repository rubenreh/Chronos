"""
Preprocessor – Time-Series Preprocessing Pipeline for Chronos
================================================================
This module contains the core data-preprocessing logic that transforms raw,
irregularly-sampled time series into clean, normalized, sliding-window samples
ready for model training and inference.

Three public interfaces are provided:

  • resample_and_fill(series, freq, method)
        Standalone function: resample a pd.Series to a regular frequency and
        fill NaN gaps using one of several strategies (interpolate, ffill,
        bfill, mean).

  • sliding_windows(arr, input_len, horizon, stride)
        Standalone function: convert a 1-D numpy array into overlapping
        (input, target) pairs for supervised forecasting.

  • Preprocessor class
        Stateful pipeline that chains: resample → fill → z-score normalize.
        It stores mu (mean) and sigma (std) after fit_transform() so that:
          - transform() can normalize new data with the same parameters.
          - inverse_transform() can denormalize model predictions back to the
            original value scale during inference.

Data flow:
    raw pd.Series → Preprocessor.fit_transform() → normalized pd.Series
    → sliding_windows() → (X, y) numpy arrays → SequenceDataset → DataLoader
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def resample_and_fill(s: pd.Series, freq: str = '1T', method: str = 'interpolate') -> pd.Series:
    """Resample a time series to a regular frequency and fill missing values.

    Real-world time-series data often arrives at irregular intervals or has
    gaps. This function forces the data onto a uniform time grid and fills
    any resulting NaN entries.

    Available fill methods:
      • 'interpolate' (default) – forward-fill then linear interpolation. This
            gives the smoothest result and handles both leading and interior gaps.
      • 'ffill' – forward-fill only (last observation carried forward).
      • 'bfill' – backward-fill only (next observation carried backward).
      • 'mean'  – fill all NaNs with the series mean (loses temporal structure).
      • anything else – fill with 0 (safe fallback).

    Args:
        s: Input pd.Series indexed by datetime.
        freq: Target frequency as a pandas offset alias.
              Examples: '1T' (1 minute), '5T' (5 minutes), '1H' (1 hour).
        method: Fill strategy (see above).

    Returns:
        pd.Series resampled to `freq` with no NaN values.
    """
    # resample().mean() places observations onto the regular grid. Timestamps
    # that had no data points become NaN.
    s2 = s.resample(freq).mean()

    if method == 'interpolate':
        # Forward-fill first to handle leading NaNs, then interpolate
        # linearly for any remaining interior gaps.
        s2 = s2.ffill().interpolate()
    elif method == 'ffill':
        # Propagate each last-known value forward through subsequent NaNs.
        s2 = s2.ffill()
    elif method == 'bfill':
        # Propagate each next-known value backward through preceding NaNs.
        s2 = s2.bfill()
    elif method == 'mean':
        # Replace all NaNs with the global mean of the series.
        s2 = s2.fillna(s2.mean())
    else:
        # Fallback: replace NaNs with zero.
        s2 = s2.fillna(0)

    return s2


def sliding_windows(
    arr: np.ndarray,
    input_len: int = 60,
    horizon: int = 1,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Create overlapping sliding-window (X, y) pairs for supervised forecasting.

    This is the key data transformation that converts a flat 1-D time-series
    array into labeled training samples:
      X[i] = arr[i        : i + input_len]            (the look-back window)
      y[i] = arr[i + input_len : i + input_len + horizon]  (the forecast target)

    With stride=1 every possible window is extracted, maximizing the number of
    training samples. Larger strides reduce overlap and dataset size.

    Args:
        arr: 1-D numpy array of preprocessed (and typically normalized) values.
        input_len: Number of past timesteps in each input window.
        horizon: Number of future timesteps to predict.
        stride: Step size between consecutive windows.

    Returns:
        Tuple (X, y):
            X – shape (n_samples, input_len)
            y – shape (n_samples, horizon)
    """
    x = []  # Accumulates input windows.
    y = []  # Accumulates corresponding target windows.
    n = len(arr)

    # The last valid starting index is n - input_len - horizon.
    for i in range(0, n - input_len - horizon + 1, stride):
        # Extract the input window of `input_len` consecutive values.
        x.append(arr[i:i+input_len])
        # Extract the target window — the next `horizon` values after the input.
        y.append(arr[i+input_len:i+input_len+horizon])

    # Stack lists into 2-D numpy arrays for efficient batch processing.
    return np.array(x), np.array(y)


class Preprocessor:
    """Stateful preprocessing pipeline: resample → fill NaN → z-score normalize.

    The Preprocessor remembers the normalization parameters (mu, sigma) computed
    during fit_transform(), so they can be reused in two critical scenarios:
      • transform()         – normalize new incoming data with the same scale
                              (e.g., a validation or test series).
      • inverse_transform() – convert model predictions back from normalized
                              space to original value units at inference time.
    """

    def __init__(
        self,
        freq: str = '1T',
        fill_method: str = 'interpolate',
        normalize: bool = True
    ):
        """Configure the preprocessing pipeline.

        Args:
            freq: Target resampling frequency (pandas offset alias).
            fill_method: Strategy for filling NaN gaps after resampling.
                         One of 'interpolate', 'ffill', 'bfill', 'mean'.
            normalize: If True, apply z-score normalization (subtract mean,
                       divide by std). If False, data is resampled and filled
                       but not scaled.
        """
        self.freq = freq              # Resampling frequency (e.g. '1T').
        self.fill_method = fill_method  # NaN fill strategy.
        self.normalize = normalize      # Whether to apply z-score scaling.
        self.mu = None                  # Will hold the series mean after fitting.
        self.sigma = None               # Will hold the series std dev after fitting.

    def fit_transform(self, series: pd.Series) -> pd.Series:
        """Resample, fill, compute normalization stats, and normalize in one step.

        This is the method used on the training data. It both *fits* (computes
        mu and sigma) and *transforms* (applies the normalization). Subsequent
        calls to transform() or inverse_transform() will reuse the mu/sigma
        computed here.

        Args:
            series: Raw pd.Series indexed by datetime.

        Returns:
            Preprocessed (and optionally normalized) pd.Series.
        """
        # Step 1: Resample to a regular frequency and fill NaN gaps.
        processed = resample_and_fill(series, freq=self.freq, method=self.fill_method)

        # Step 2: Optionally compute and apply z-score normalization.
        if self.normalize:
            # Compute the mean and standard deviation of the resampled series.
            self.mu = processed.mean()
            self.sigma = processed.std()

            if self.sigma > 0:
                # Standard z-score: (x - mu) / sigma → mean≈0, std≈1.
                processed = (processed - self.mu) / self.sigma
            else:
                # If sigma is zero (constant series), just center the data.
                # Division would cause NaN/Inf, so we skip it.
                processed = processed - self.mu

        return processed

    def transform(self, series: pd.Series) -> pd.Series:
        """Preprocess a new series using previously fitted normalization params.

        Use this for validation/test data that should be normalized with the
        *training* set's mu and sigma to avoid data leakage.

        Args:
            series: Raw pd.Series indexed by datetime.

        Returns:
            Preprocessed and normalized pd.Series (using stored mu/sigma).
        """
        # Resample and fill using the same settings as fit_transform.
        processed = resample_and_fill(series, freq=self.freq, method=self.fill_method)

        # Apply normalization only if we have previously computed mu and sigma.
        if self.normalize and self.mu is not None and self.sigma is not None:
            if self.sigma > 0:
                processed = (processed - self.mu) / self.sigma
            else:
                processed = processed - self.mu

        return processed

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        """Reverse the z-score normalization to recover original-scale values.

        This is used at inference time: the model outputs normalized predictions,
        and this method scales them back to the real-world units (e.g., dollars,
        degrees, sensor counts) by applying: original = normalized * sigma + mu.

        Args:
            series: Normalized pd.Series or numpy array.

        Returns:
            Denormalized series in the original value scale.
        """
        if self.normalize and self.mu is not None and self.sigma is not None:
            # Reverse z-score: x_original = x_normalized * sigma + mu.
            return series * self.sigma + self.mu
        # If normalization was not applied, return data unchanged.
        return series
