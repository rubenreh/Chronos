"""Basic preprocessing utilities: load CSV, pivot to wide format per series, resample, and create sliding windows."""

import pandas as pd
import numpy as np


def load_timeseries(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df

def pivot_series(df):
    # returns dict series_id -> pd.Series indexed by timestamp
    series = {}
    for sid, g in df.groupby('series_id'):
        s = g.set_index('timestamp')['value'].astype(float).sort_index()
        series[sid] = s
    return series

def resample_and_fill(s, freq='1T'):
    # ensure regular frequency, forward fill then interpolate
    s2 = s.resample(freq).mean()
    s2 = s2.ffill().interpolate()
    return s2

def sliding_windows(arr, input_len=60, horizon=1, stride=1):
    x = []
    y = []
    n = len(arr)
    for i in range(0, n - input_len - horizon + 1, stride):
        x.append(arr[i:i+input_len])
        y.append(arr[i+input_len:i+input_len+horizon])
    return np.array(x), np.array(y)
