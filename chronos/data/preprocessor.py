"""Preprocessing utilities for time-series data."""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def resample_and_fill(s: pd.Series, freq: str = '1T', method: str = 'interpolate') -> pd.Series:
    """Resample series to regular frequency and fill missing values.
    
    Args:
        s: Input time series
        freq: Resampling frequency (e.g., '1T' for 1 minute)
        method: Fill method ('interpolate', 'ffill', 'bfill', 'mean')
    
    Returns:
        Resampled and filled series
    """
    s2 = s.resample(freq).mean()
    
    if method == 'interpolate':
        s2 = s2.ffill().interpolate()
    elif method == 'ffill':
        s2 = s2.ffill()
    elif method == 'bfill':
        s2 = s2.bfill()
    elif method == 'mean':
        s2 = s2.fillna(s2.mean())
    else:
        s2 = s2.fillna(0)
    
    return s2


def sliding_windows(
    arr: np.ndarray,
    input_len: int = 60,
    horizon: int = 1,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for time-series forecasting.
    
    Args:
        arr: 1D array of time-series values
        input_len: Length of input sequence
        horizon: Number of steps to predict
        stride: Step size for sliding window
    
    Returns:
        Tuple of (X, y) arrays where X is (n_samples, input_len) and y is (n_samples, horizon)
    """
    x = []
    y = []
    n = len(arr)
    for i in range(0, n - input_len - horizon + 1, stride):
        x.append(arr[i:i+input_len])
        y.append(arr[i+input_len:i+input_len+horizon])
    return np.array(x), np.array(y)


class Preprocessor:
    """Preprocessing pipeline for time-series data."""
    
    def __init__(
        self,
        freq: str = '1T',
        fill_method: str = 'interpolate',
        normalize: bool = True
    ):
        """Initialize preprocessor.
        
        Args:
            freq: Resampling frequency
            fill_method: Method for filling missing values
            normalize: Whether to normalize the data
        """
        self.freq = freq
        self.fill_method = fill_method
        self.normalize = normalize
        self.mu = None
        self.sigma = None
    
    def fit_transform(self, series: pd.Series) -> pd.Series:
        """Fit preprocessor and transform series."""
        # Resample and fill
        processed = resample_and_fill(series, freq=self.freq, method=self.fill_method)
        
        # Normalize if requested
        if self.normalize:
            self.mu = processed.mean()
            self.sigma = processed.std()
            if self.sigma > 0:
                processed = (processed - self.mu) / self.sigma
            else:
                processed = processed - self.mu
        
        return processed
    
    def transform(self, series: pd.Series) -> pd.Series:
        """Transform series using fitted parameters."""
        processed = resample_and_fill(series, freq=self.freq, method=self.fill_method)
        
        if self.normalize and self.mu is not None and self.sigma is not None:
            if self.sigma > 0:
                processed = (processed - self.mu) / self.sigma
            else:
                processed = processed - self.mu
        
        return processed
    
    def inverse_transform(self, series: pd.Series) -> pd.Series:
        """Inverse transform normalized series."""
        if self.normalize and self.mu is not None and self.sigma is not None:
            return series * self.sigma + self.mu
        return series

