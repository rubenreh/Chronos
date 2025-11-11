"""Feature extraction for behavioral analytics."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler


def extract_behavioral_features(
    series: pd.Series,
    window_size: int = 60,
    include_trend: bool = True,
    include_seasonality: bool = True,
    include_statistical: bool = True
) -> Dict[str, float]:
    """Extract behavioral features from a time-series window.
    
    Args:
        series: Time series data
        window_size: Size of sliding window for feature extraction
        include_trend: Whether to include trend features
        include_seasonality: Whether to include seasonality features
        include_statistical: Whether to include statistical features
    
    Returns:
        Dictionary of feature names to values
    """
    features = {}
    
    if len(series) < window_size:
        window = series.values
    else:
        window = series.tail(window_size).values
    
    # Statistical features
    if include_statistical:
        features['mean'] = float(np.mean(window))
        features['std'] = float(np.std(window))
        features['min'] = float(np.min(window))
        features['max'] = float(np.max(window))
        features['median'] = float(np.median(window))
        features['q25'] = float(np.percentile(window, 25))
        features['q75'] = float(np.percentile(window, 75))
        features['skewness'] = float(stats.skew(window))
        features['kurtosis'] = float(stats.kurtosis(window))
        features['coefficient_of_variation'] = float(np.std(window) / (np.mean(window) + 1e-8))
    
    # Trend features
    if include_trend and len(window) > 1:
        x = np.arange(len(window))
        slope, intercept = np.polyfit(x, window, 1)
        features['trend_slope'] = float(slope)
        features['trend_intercept'] = float(intercept)
        features['trend_strength'] = float(np.abs(slope))
        
        # Linear regression R²
        y_pred = slope * x + intercept
        ss_res = np.sum((window - y_pred) ** 2)
        ss_tot = np.sum((window - np.mean(window)) ** 2)
        features['trend_r2'] = float(1 - (ss_res / (ss_tot + 1e-8)))
    
    # Seasonality features (using FFT)
    if include_seasonality and len(window) > 4:
        fft_vals = np.fft.rfft(window)
        power = np.abs(fft_vals) ** 2
        if len(power) > 1:
            features['dominant_frequency'] = float(np.argmax(power[1:]) + 1)
            features['spectral_entropy'] = float(-np.sum((power / (np.sum(power) + 1e-8)) * 
                                                          np.log(power / (np.sum(power) + 1e-8) + 1e-8)))
    
    # Volatility features
    if len(window) > 1:
        returns = np.diff(window) / (window[:-1] + 1e-8)
        features['volatility'] = float(np.std(returns))
        features['mean_absolute_change'] = float(np.mean(np.abs(np.diff(window))))
    
    # Autocorrelation
    if len(window) > 2:
        autocorr = np.corrcoef(window[:-1], window[1:])[0, 1]
        features['autocorr_lag1'] = float(autocorr if not np.isnan(autocorr) else 0.0)
    
    return features


class FeatureExtractor:
    """Feature extraction pipeline for productivity analytics."""
    
    def __init__(
        self,
        window_size: int = 60,
        normalize: bool = True
    ):
        """Initialize feature extractor.
        
        Args:
            window_size: Size of sliding window
            normalize: Whether to normalize features
        """
        self.window_size = window_size
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.feature_names_ = None
    
    def fit(self, series_list: List[pd.Series]):
        """Fit feature extractor on multiple series."""
        all_features = []
        for series in series_list:
            features = extract_behavioral_features(series, window_size=self.window_size)
            all_features.append(list(features.values()))
        
        if self.normalize and all_features:
            self.scaler.fit(all_features)
            self.feature_names_ = list(extract_behavioral_features(
                series_list[0], window_size=self.window_size
            ).keys())
    
    def transform(self, series: pd.Series) -> np.ndarray:
        """Extract features from a single series."""
        features = extract_behavioral_features(series, window_size=self.window_size)
        feature_array = np.array(list(features.values()))
        
        if self.normalize and self.scaler is not None:
            feature_array = self.scaler.transform([feature_array])[0]
        
        return feature_array
    
    def fit_transform(self, series_list: List[pd.Series]) -> np.ndarray:
        """Fit and transform."""
        self.fit(series_list)
        return np.array([self.transform(s) for s in series_list])

