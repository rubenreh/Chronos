"""
Behavioral feature extraction for Chronos productivity analytics.

This module is the numerical backbone of the Chronos feature pipeline. Given a
pandas Series of productivity values (e.g. task-completion scores recorded every
minute), it computes a rich dictionary of hand-crafted features that capture:

  • Statistical shape   — mean, std, skewness, kurtosis, percentiles, CV
  • Linear trend        — OLS slope, intercept, R² goodness-of-fit
  • Seasonality         — dominant FFT frequency, spectral entropy
  • Volatility          — std of per-step returns, mean absolute change
  • Temporal dependency — lag-1 autocorrelation

These features are consumed by:
  - The /patterns API endpoint to summarize a user's recent behaviour.
  - The RecommendationEngine to detect trends and bottlenecks.
  - The UserClustering module to group users with similar profiles.

The FeatureExtractor class wraps the functional API and adds optional
StandardScaler normalization so features can be fed directly to scikit-learn
models or plotted on comparable axes.
"""

import pandas as pd                          # DataFrame / Series handling for time-series input
import numpy as np                           # Fast numerical array operations
from typing import Dict, List, Optional      # Type hints for clarity and IDE support
from scipy import stats                      # Statistical functions: skew, kurtosis
from sklearn.preprocessing import StandardScaler  # Z-score normalisation for feature vectors


def extract_behavioral_features(
    series: pd.Series,
    window_size: int = 60,
    include_trend: bool = True,
    include_seasonality: bool = True,
    include_statistical: bool = True
) -> Dict[str, float]:
    """Extract a dictionary of behavioural features from a time-series window.

    This is the core feature-engineering function used across Chronos. It takes
    the most recent `window_size` values of the series and computes statistical,
    trend, seasonality, volatility, and autocorrelation features.

    Args:
        series: Time series of productivity values (pandas Series).
        window_size: How many recent data points to use (default 60 ≈ 1 hour at
                     1-minute resolution).
        include_trend: If True, compute linear-trend features (slope, R²).
        include_seasonality: If True, compute FFT-based seasonality features.
        include_statistical: If True, compute descriptive-statistics features.

    Returns:
        Dictionary mapping feature names (str) to computed values (float).
    """
    features = {}  # Accumulator dict — keys added incrementally below

    # --- Window selection ---
    # If the series is shorter than the requested window, use all available data;
    # otherwise take the most recent `window_size` points.
    if len(series) < window_size:
        window = series.values  # Use the full (short) series as a numpy array
    else:
        window = series.tail(window_size).values  # Take the last `window_size` values

    # ── Statistical features ──────────────────────────────────────────────
    # These capture the *shape* of the distribution of values in the window.
    if include_statistical:
        features['mean'] = float(np.mean(window))              # Central tendency
        features['std'] = float(np.std(window))                # Spread / dispersion
        features['min'] = float(np.min(window))                # Worst data point in window
        features['max'] = float(np.max(window))                # Best data point in window
        features['median'] = float(np.median(window))          # Robust centre (less sensitive to outliers)
        features['q25'] = float(np.percentile(window, 25))     # 25th percentile (lower quartile)
        features['q75'] = float(np.percentile(window, 75))     # 75th percentile (upper quartile)
        features['skewness'] = float(stats.skew(window))       # Asymmetry: +ve means right tail, -ve left
        features['kurtosis'] = float(stats.kurtosis(window))   # Tail heaviness: high → more outliers
        # Coefficient of variation (CV) = relative dispersion; ε avoids division by zero
        features['coefficient_of_variation'] = float(np.std(window) / (np.mean(window) + 1e-8))

    # ── Trend features ────────────────────────────────────────────────────
    # Fit a simple y = slope·x + intercept line to detect rising/falling trends.
    if include_trend and len(window) > 1:
        x = np.arange(len(window))                             # Index array [0, 1, …, N-1]
        slope, intercept = np.polyfit(x, window, 1)            # Ordinary least-squares linear fit
        features['trend_slope'] = float(slope)                 # Rate of change per timestep
        features['trend_intercept'] = float(intercept)         # Predicted value at x = 0
        features['trend_strength'] = float(np.abs(slope))      # Absolute magnitude of trend

        # R² (coefficient of determination) — how well the line explains variance.
        y_pred = slope * x + intercept                         # Predicted values from linear model
        ss_res = np.sum((window - y_pred) ** 2)                # Residual sum of squares
        ss_tot = np.sum((window - np.mean(window)) ** 2)       # Total sum of squares
        features['trend_r2'] = float(1 - (ss_res / (ss_tot + 1e-8)))  # R² in [0, 1]; ε guards zero-variance

    # ── Seasonality features (FFT) ────────────────────────────────────────
    # Use the Fast Fourier Transform to find the dominant repeating cycle.
    if include_seasonality and len(window) > 4:
        fft_vals = np.fft.rfft(window)        # Real-input FFT → complex frequency coefficients
        power = np.abs(fft_vals) ** 2          # Power spectrum (energy at each frequency bin)
        if len(power) > 1:
            # Dominant frequency: index of highest-power bin (skip DC component at index 0)
            features['dominant_frequency'] = float(np.argmax(power[1:]) + 1)
            # Spectral entropy: measures how "spread out" the energy is across frequencies.
            # Low entropy → one strong cycle; high entropy → white-noise-like, no clear pattern.
            features['spectral_entropy'] = float(-np.sum((power / (np.sum(power) + 1e-8)) *
                                                          np.log(power / (np.sum(power) + 1e-8) + 1e-8)))

    # ── Volatility features ───────────────────────────────────────────────
    # Volatility captures how erratically productivity changes step-to-step.
    if len(window) > 1:
        # Per-step fractional returns (like financial returns); ε avoids div-by-zero
        returns = np.diff(window) / (window[:-1] + 1e-8)
        features['volatility'] = float(np.std(returns))                   # Std of returns
        features['mean_absolute_change'] = float(np.mean(np.abs(np.diff(window))))  # Avg absolute Δ

    # ── Autocorrelation ───────────────────────────────────────────────────
    # Lag-1 autocorrelation: how correlated a value is with the previous one.
    # High autocorrelation → smooth momentum; low → erratic jumps.
    if len(window) > 2:
        autocorr = np.corrcoef(window[:-1], window[1:])[0, 1]  # Pearson r between t and t-1
        features['autocorr_lag1'] = float(autocorr if not np.isnan(autocorr) else 0.0)

    return features


class FeatureExtractor:
    """Sklearn-style feature extraction pipeline for productivity analytics.

    Wraps `extract_behavioral_features` with an optional StandardScaler so that
    the output feature vectors are zero-mean / unit-variance — important when
    feeding features into distance-based algorithms like KMeans or DBSCAN.
    """

    def __init__(
        self,
        window_size: int = 60,
        normalize: bool = True
    ):
        """Initialise the feature extractor.

        Args:
            window_size: Number of recent data points to consider (default 60).
            normalize: If True, fit a StandardScaler during `fit()` and apply it
                       during `transform()`.
        """
        self.window_size = window_size        # Saved for use in every extraction call
        self.normalize = normalize            # Whether to z-score normalise output vectors
        self.scaler = StandardScaler() if normalize else None  # Scaler instance (or None)
        self.feature_names_ = None            # Populated after fit() — list of feature key names

    def fit(self, series_list: List[pd.Series]):
        """Fit the scaler on a collection of time-series so it learns per-feature mean/std.

        Args:
            series_list: List of pandas Series, one per user or segment.
        """
        all_features = []  # Will hold one feature-value list per series
        for series in series_list:
            # Extract the raw feature dict for each series
            features = extract_behavioral_features(series, window_size=self.window_size)
            all_features.append(list(features.values()))  # Convert dict values to list

        if self.normalize and all_features:
            self.scaler.fit(all_features)  # Compute per-feature mean and std from all series
            # Record feature names from the first series (order is deterministic)
            self.feature_names_ = list(extract_behavioral_features(
                series_list[0], window_size=self.window_size
            ).keys())

    def transform(self, series: pd.Series) -> np.ndarray:
        """Extract and optionally normalise features from a single series.

        Args:
            series: A single productivity time-series.

        Returns:
            1-D numpy array of feature values.
        """
        features = extract_behavioral_features(series, window_size=self.window_size)
        feature_array = np.array(list(features.values()))  # Dict values → numpy array

        if self.normalize and self.scaler is not None:
            # Apply the previously-fitted scaler; wraps in list because transform expects 2-D
            feature_array = self.scaler.transform([feature_array])[0]

        return feature_array

    def fit_transform(self, series_list: List[pd.Series]) -> np.ndarray:
        """Convenience: fit on all series, then transform each one.

        Returns:
            2-D numpy array of shape (n_series, n_features).
        """
        self.fit(series_list)  # Learn normalisation parameters
        return np.array([self.transform(s) for s in series_list])  # Transform every series
