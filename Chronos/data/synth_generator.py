"""
Basic synthetic time-series generator for the Chronos demo.

Produces a CSV file with columns: timestamp, series_id, value.
Each generated series is a superposition of:
  • A linear trend   — gradual upward or downward drift.
  • A sinusoidal seasonality component — repeating cycle (daily, 12-h, hourly,
    or weekly period chosen at random).
  • Gaussian noise   — random jitter proportional to the seasonal amplitude.
  • Optional injected anomalies — sudden spikes or dips that test the model's
    ability to detect outliers.

This generator is intentionally simple; for more realistic data with behavioural
phases, missingness, and concept drift, see advanced_synth_generator.py.

Usage (CLI):
    python data/synth_generator.py --out data/sample_timeseries.csv --n-series 3 --length 1440 --seed 42
"""

import argparse                           # Command-line argument parsing
from datetime import datetime, timedelta  # Timestamp generation
import math                               # sin / pi for seasonality
import random                             # Pseudo-random number generation for reproducibility
import csv                                # Writing output CSV files
import os                                 # File-system utilities (makedirs)


def generate_series(length=1440, freq_minutes=1, seed=None, inject_anomalies=True):
    """Generate a single synthetic time-series with trend + seasonality + noise.

    The function creates `length` data points at `freq_minutes`-minute intervals
    starting from 2025-01-01 00:00. A random linear trend, a sinusoidal
    seasonal pattern, and Gaussian noise are combined to form the series.
    Optionally, a handful of extreme spikes/dips are injected as anomalies.

    Args:
        length: Number of data points to generate (default 1440 ≈ 1 day at 1-min).
        freq_minutes: Interval between consecutive points in minutes.
        seed: Random seed for reproducibility (None = non-deterministic).
        inject_anomalies: If True, inject ~1 anomaly per 400 points.

    Returns:
        Tuple of (timestamps, values) where timestamps is a list of datetime
        objects and values is a list of floats.
    """
    if seed is not None:
        random.seed(seed)  # Fix the random state so the same seed always produces the same series

    # Create evenly-spaced timestamps starting from midnight on 2025-01-01
    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(minutes=i * freq_minutes) for i in range(length)]

    # ── Randomise series parameters ──────────────────────────────────────
    trend = random.uniform(-0.001, 0.001)          # Small linear drift per timestep
    amp = random.uniform(5, 20)                    # Amplitude of the seasonal wave
    # Period of the seasonal cycle (daily, 12-h, hourly, or weekly)
    period = random.choice([24 * 60, 12 * 60, 60, 7 * 24 * 60])

    # ── Build the series by summing components ───────────────────────────
    values = []
    baseline = random.uniform(20, 80)  # Random vertical offset so series sit at different levels
    for i in range(length):
        t = i  # Timestep index
        seasonal = amp * math.sin(2 * math.pi * t / period)   # Sinusoidal seasonal component
        # Combine baseline + linear trend + seasonal + Gaussian noise (σ = 15 % of amplitude)
        val = baseline + trend * t + seasonal + random.gauss(0, amp * 0.15)
        values.append(val)

    # ── Inject anomalies (optional) ──────────────────────────────────────
    # Anomalies are sudden spikes or dips meant to test outlier detection.
    if inject_anomalies:
        # Roughly 1 anomaly per 400 data points (at least 1)
        for _ in range(max(1, length // 400)):
            idx = random.randint(20, length - 20)  # Avoid the very first/last points
            if random.random() < 0.6:
                # 60 % chance of a positive spike (3× – 7× amplitude above normal)
                values[idx] += random.uniform(amp * 3, amp * 7)
            else:
                # 40 % chance of a negative dip
                values[idx] -= random.uniform(amp * 3, amp * 7)

    return timestamps, values


# ── CLI entry point ──────────────────────────────────────────────────────
if __name__ == '__main__':
    # Parse command-line arguments for output path, number of series, length, and seed
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/sample_timeseries.csv')  # Output CSV path
    p.add_argument('--n-series', type=int, default=3)              # How many series to generate
    p.add_argument('--length', type=int, default=1440)             # Points per series
    p.add_argument('--seed', type=int, default=42)                 # Base random seed
    args = p.parse_args()

    # Ensure the output directory exists (e.g. "data/")
    try:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    except Exception:
        pass  # Silently ignore if directory creation fails (e.g. empty dirname)

    # Write all generated series into a single CSV file
    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'series_id', 'value'])  # CSV header row

        for s in range(args.n_series):
            # Each series gets a unique seed (base_seed + series_index) for reproducibility
            ts, vals = generate_series(length=args.length, seed=args.seed + s)
            sid = f"series_{s}"  # Series identifier string, e.g. "series_0"
            # Write one row per data point: ISO-8601 timestamp, series id, value to 4 decimal places
            for t, v in zip(ts, vals):
                writer.writerow([t.isoformat(), sid, f"{v:.4f}"])

    print(f"Wrote sample data to {args.out}")
