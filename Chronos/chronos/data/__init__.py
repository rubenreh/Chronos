"""
Data Package – Loading and Preprocessing for Chronos
======================================================
This package is the data-ingestion layer of the Chronos forecasting system.
It provides two main modules:

  • loader.py       – DataLoader class that reads CSV files (expected columns:
                      timestamp, series_id, value), pivots them into per-series
                      pd.Series objects, and exposes helper methods for
                      retrieving individual series or listing available IDs.

  • preprocessor.py – Preprocessor class that handles the full preprocessing
                      pipeline: resample to a regular frequency → fill NaN
                      gaps → z-score normalize (storing mu and sigma for later
                      inverse_transform at inference time). Also provides the
                      standalone sliding_windows() function that creates
                      overlapping (X, y) pairs for supervised training.

Typical usage:
    from chronos.data import DataLoader, Preprocessor, sliding_windows
    loader = DataLoader(data_path="data/sample.csv")
    series_dict = loader.pivot_series()
    preprocessor = Preprocessor(freq='1T', normalize=True)
    processed = preprocessor.fit_transform(series_dict['sensor_01'])
"""

# DataLoader reads CSVs and provides per-series access.
# load_timeseries is a standalone function for quick CSV loading.
from .loader import DataLoader, load_timeseries

# Preprocessor handles resample → fill → normalize and stores mu/sigma.
# resample_and_fill is a standalone function for quick resampling.
# sliding_windows creates (X, y) pairs for supervised forecasting.
from .preprocessor import Preprocessor, resample_and_fill, sliding_windows

# __all__ defines the public API of this package — only these symbols are
# exported when a caller does `from chronos.data import *`.
__all__ = ['DataLoader', 'load_timeseries', 'Preprocessor', 'resample_and_fill', 'sliding_windows']
