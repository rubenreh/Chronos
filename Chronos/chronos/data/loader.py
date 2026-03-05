"""
Data Loader – CSV Ingestion for Chronos
=========================================
This module is the first step in the Chronos data pipeline. It reads raw CSV
files containing time-series observations and converts them into a format
suitable for preprocessing and model training.

Expected CSV schema:
    timestamp   – datetime string (e.g., "2025-01-15 08:30:00")
    series_id   – string identifier grouping rows into distinct time series
    value       – numeric measurement at each timestamp

Two interfaces are provided:
  • load_timeseries(path) – a standalone function that returns a pd.DataFrame.
  • DataLoader class      – an object-oriented wrapper that holds the DataFrame
                            and provides methods to pivot, retrieve, and list series.

The DataLoader here is Chronos's *custom* DataLoader (CSV → dict of pd.Series).
It is NOT the same as torch.utils.data.DataLoader (mini-batch iterator). When
both are needed in the same file, the convention is:
    from chronos.data.loader import DataLoader as ChronosDataLoader
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path


def load_timeseries(path: str) -> pd.DataFrame:
    """Load a CSV file containing time-series data into a pandas DataFrame.

    The 'timestamp' column is automatically parsed to datetime64 so that
    downstream operations like resampling and time-based indexing work
    out of the box.

    Args:
        path: File path to the CSV.

    Returns:
        DataFrame with columns [timestamp (datetime64), series_id, value].
    """
    # parse_dates=['timestamp'] tells pandas to convert that column from
    # string to datetime64 during the read, avoiding a separate pd.to_datetime call.
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df


class DataLoader:
    """Load and manage time-series data from a CSV file.

    This class wraps a pandas DataFrame and provides convenience methods
    for converting between the long format (one row per observation) and
    the wide format (one pd.Series per series_id) needed by the Preprocessor.
    """

    def __init__(self, data_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        """Initialize data loader from a file path or a pre-loaded DataFrame.

        Exactly one of data_path or df must be provided. This dual-init pattern
        is useful for testing (pass a synthetic DataFrame directly) and for
        production (pass a file path).

        Args:
            data_path: Path to a CSV file on disk.
            df: An already-loaded pandas DataFrame with the expected schema.

        Raises:
            ValueError: If neither data_path nor df is provided.
        """
        if df is not None:
            # Use the pre-loaded DataFrame directly (useful in tests/notebooks).
            self.df = df
        elif data_path:
            # Load the CSV from disk using the standalone helper function.
            self.df = load_timeseries(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")

    def pivot_series(self) -> Dict[str, pd.Series]:
        """Convert the long-format DataFrame into a dict of per-series pd.Series.

        Each series is indexed by timestamp and sorted chronologically, which is
        the required format for the Preprocessor's resample and normalize steps.

        Returns:
            Dictionary mapping each series_id (str) to a datetime-indexed pd.Series
            of float values.
        """
        series = {}
        # Group rows by series_id — each group contains all observations for one series.
        for sid, g in self.df.groupby('series_id'):
            # Set timestamp as the index, extract the 'value' column, cast to float
            # (in case it was read as object/int), and sort by time.
            s = g.set_index('timestamp')['value'].astype(float).sort_index()
            series[sid] = s
        return series

    def get_series(self, series_id: str) -> pd.Series:
        """Retrieve a single time series by its ID.

        This is a convenience method that calls pivot_series() internally.
        For repeated lookups, callers should cache the result of pivot_series()
        to avoid re-pivoting every time.

        Args:
            series_id: The identifier of the desired series.

        Returns:
            A datetime-indexed pd.Series of float values.

        Raises:
            ValueError: If the requested series_id is not in the dataset.
        """
        series = self.pivot_series()
        if series_id not in series:
            raise ValueError(f"Series {series_id} not found")
        return series[series_id]

    def get_all_series_ids(self) -> list:
        """Return a list of all unique series IDs in the dataset.

        Useful for iterating over every available series during batch
        preprocessing or training.

        Returns:
            List of unique series_id strings.
        """
        # .unique() returns a numpy array; .tolist() converts it to a plain Python list.
        return self.df['series_id'].unique().tolist()
