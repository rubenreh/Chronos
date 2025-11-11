"""Data loading utilities for Chronos."""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path


def load_timeseries(path: str) -> pd.DataFrame:
    """Load time-series data from CSV.
    
    Expected format: timestamp, series_id, value
    """
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df


class DataLoader:
    """Load and manage time-series data."""
    
    def __init__(self, data_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        """Initialize data loader.
        
        Args:
            data_path: Path to CSV file
            df: Optional pre-loaded DataFrame
        """
        if df is not None:
            self.df = df
        elif data_path:
            self.df = load_timeseries(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")
    
    def pivot_series(self) -> Dict[str, pd.Series]:
        """Convert DataFrame to dict of series_id -> pd.Series.
        
        Returns:
            Dictionary mapping series_id to time-indexed Series
        """
        series = {}
        for sid, g in self.df.groupby('series_id'):
            s = g.set_index('timestamp')['value'].astype(float).sort_index()
            series[sid] = s
        return series
    
    def get_series(self, series_id: str) -> pd.Series:
        """Get a specific series by ID."""
        series = self.pivot_series()
        if series_id not in series:
            raise ValueError(f"Series {series_id} not found")
        return series[series_id]
    
    def get_all_series_ids(self) -> list:
        """Get all unique series IDs."""
        return self.df['series_id'].unique().tolist()

