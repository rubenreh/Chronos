"""
Advanced synthetic time-series generator with behavioural phases, missingness,
concept drift, multiple seasonality types, and configurable noise models.

This generator produces realistic productivity datasets for stress-testing
Chronos models. Unlike the basic synth_generator.py, it simulates:

  • Behavioural phases — Normal, High-Performance, Burnout, Recovery — each with
    distinct trend slopes and variance profiles.
  • Missingness patterns — Random, Burst (contiguous gap), Periodic (e.g.
    weekends) — to test imputation robustness.
  • Concept drift — Gradual (linear shift), Sudden (step change at midpoint),
    Recurring (periodic ramps) — to test adaptive forecasting.
  • Seasonality types — Daily (24 h), Weekly (7 d), Monthly (30 d), Multi
    (daily + weekly combined).
  • Noise models — Gaussian (constant σ), Uniform, Heteroscedastic (σ ∝ value).

Output CSV columns: timestamp, series_id, value  (NaN rows are omitted).

Usage (CLI):
    python data/advanced_synth_generator.py --out data/advanced_timeseries.csv \\
        --n-series 10 --length 2880 --seed 42 --drift-type gradual
"""

import argparse                           # CLI argument parsing
from datetime import datetime, timedelta  # Timestamp arithmetic
import math                               # sin / pi for seasonal waves
import random                             # Python-level random for seed control
import csv                                # CSV writing
import os                                 # File-system utilities
import numpy as np                        # Vectorised numerical operations
from typing import List, Tuple, Optional  # Type annotations


class BehavioralPhase:
    """Enum-like constants for the four productivity phases.

    Each phase models a different regime of user behaviour:
      HIGH_PERFORMANCE — sustained high output with a positive trend.
      BURNOUT          — declining output with high volatility.
      RECOVERY         — rising from a low baseline after burnout.
      NORMAL           — stable output around the mean.
    """
    HIGH_PERFORMANCE = "high_performance"
    BURNOUT = "burnout"
    RECOVERY = "recovery"
    NORMAL = "normal"


def generate_phase_pattern(
    phase: str,
    length: int,
    baseline: float,
    amplitude: float
) -> np.ndarray:
    """Generate a segment of productivity values that follow a given behavioural phase.

    Each phase has its own trend direction, offset from baseline, and noise level
    to simulate realistic shifts in user behaviour.

    Args:
        phase: One of the BehavioralPhase constants.
        length: Number of timesteps to generate for this phase.
        baseline: Global baseline productivity value (shared across phases).
        amplitude: Controls the magnitude of offsets and noise.

    Returns:
        1-D numpy array of `length` productivity values.
    """
    values = []

    if phase == BehavioralPhase.HIGH_PERFORMANCE:
        # Elevated baseline (+30 % of amplitude) with a mild positive trend
        trend = random.uniform(0.001, 0.003)  # Positive slope per timestep
        for i in range(length):
            val = baseline + amplitude * 0.3 + trend * i + random.gauss(0, amplitude * 0.1)
            values.append(val)

    elif phase == BehavioralPhase.BURNOUT:
        # Depressed baseline (−20 % of amplitude) with a negative trend and high noise
        trend = random.uniform(-0.003, -0.001)  # Negative slope per timestep
        for i in range(length):
            val = baseline - amplitude * 0.2 + trend * i + random.gauss(0, amplitude * 0.25)
            values.append(val)

    elif phase == BehavioralPhase.RECOVERY:
        # Start low (−30 % of amplitude) and trend upward with moderate noise
        trend = random.uniform(0.002, 0.004)       # Steeper positive slope than high-performance
        start_baseline = baseline - amplitude * 0.3  # Starting point below global baseline
        for i in range(length):
            val = start_baseline + trend * i + random.gauss(0, amplitude * 0.15)
            values.append(val)

    else:  # NORMAL
        # Flat: values scatter around the baseline with moderate noise
        for i in range(length):
            val = baseline + random.gauss(0, amplitude * 0.15)
            values.append(val)

    return np.array(values)


def inject_missingness(
    values: np.ndarray,
    missing_rate: float = 0.05,
    missing_pattern: str = "random"
) -> np.ndarray:
    """Replace a fraction of values with NaN to simulate missing data.

    Three missingness patterns model different real-world scenarios:
      random   — sensor drop-outs scattered uniformly.
      burst    — a contiguous block of missing data (e.g. system outage).
      periodic — regular gaps (e.g. no data on weekends).

    Args:
        values: Original time-series values.
        missing_rate: Proportion of values to remove (0.0–1.0).
        missing_pattern: One of 'random', 'burst', 'periodic'.

    Returns:
        Copy of values with NaN inserted at the chosen positions.
    """
    result = values.copy()  # Work on a copy to avoid mutating the original
    n_missing = int(len(values) * missing_rate)  # Total number of values to blank out

    if missing_pattern == "random":
        # Pick n_missing distinct random indices and set them to NaN
        missing_indices = random.sample(range(len(values)), n_missing)
        for idx in missing_indices:
            result[idx] = np.nan

    elif missing_pattern == "burst":
        # One contiguous block of NaNs starting at a random offset
        start_idx = random.randint(0, len(values) - n_missing)
        result[start_idx:start_idx + n_missing] = np.nan

    elif missing_pattern == "periodic":
        # Drop one value every `period` timesteps (simulates regular down-time)
        period = random.choice([7, 14, 30])  # Gap period in timesteps
        for i in range(0, len(values), period):
            if i < len(values):
                result[i] = np.nan

    return result


def inject_time_drift(
    values: np.ndarray,
    drift_type: str = "gradual",
    drift_magnitude: float = 0.1
) -> np.ndarray:
    """Apply concept drift to the time-series to simulate non-stationarity.

    Concept drift means the underlying data distribution changes over time,
    which is a common challenge in production ML systems.

    Three drift patterns are supported:
      gradual   — linear ramp from 0 to drift_magnitude × mean over the series.
      sudden    — step-function shift at the midpoint.
      recurring — repeated linear ramps within each quarter of the series.

    Args:
        values: Original time-series values.
        drift_type: One of 'gradual', 'sudden', 'recurring'.
        drift_magnitude: Fractional strength of the drift relative to the mean.

    Returns:
        Copy of values with drift added.
    """
    result = values.copy()
    n = len(values)

    if drift_type == "gradual":
        # Linearly increasing offset from 0 to drift_magnitude × global_mean
        drift = np.linspace(0, drift_magnitude * np.mean(values), n)
        result = result + drift

    elif drift_type == "sudden":
        # No change in the first half; constant positive shift in the second half
        shift_point = n // 2  # Midpoint of the series
        shift = drift_magnitude * np.mean(values[:shift_point])  # Shift size based on first-half mean
        result[shift_point:] = result[shift_point:] + shift

    elif drift_type == "recurring":
        # Divide the series into 4 equal segments; apply a fresh ramp in each
        period = n // 4
        for i in range(0, n, period):
            end = min(i + period, n)
            drift = np.linspace(0, drift_magnitude * np.mean(values), end - i)
            result[i:end] = result[i:end] + drift

    return result


def add_seasonality(
    values: np.ndarray,
    seasonality_type: str = "daily",
    amplitude: float = 5.0
) -> np.ndarray:
    """Overlay a sinusoidal seasonal pattern onto the time-series.

    Seasonality captures repeating cycles in productivity — e.g. people are more
    productive mid-morning and less productive after lunch (daily cycle).

    Args:
        values: Original time-series values.
        seasonality_type: 'daily' (24 h), 'weekly' (7 d), 'monthly' (30 d),
                          or 'multi' (daily + weekly superimposed).
        amplitude: Peak-to-trough magnitude of the seasonal wave.

    Returns:
        Copy of values with the seasonal component added.
    """
    result = values.copy()
    n = len(values)

    if seasonality_type == "daily":
        period = 24 * 60  # 24 hours expressed in minutes (1440 minutes)
        for i in range(n):
            seasonal = amplitude * math.sin(2 * math.pi * i / period)  # Sinusoidal wave
            result[i] += seasonal

    elif seasonality_type == "weekly":
        period = 7 * 24 * 60  # 7 days expressed in minutes (10080 minutes)
        for i in range(n):
            seasonal = amplitude * math.sin(2 * math.pi * i / period)
            result[i] += seasonal

    elif seasonality_type == "monthly":
        period = 30 * 24 * 60  # 30 days expressed in minutes (43200 minutes)
        for i in range(n):
            seasonal = amplitude * math.sin(2 * math.pi * i / period)
            result[i] += seasonal

    elif seasonality_type == "multi":
        # Superimpose two seasonal waves: daily + weekly, each at half amplitude
        for period_mult in [24 * 60, 7 * 24 * 60]:
            for i in range(n):
                seasonal = (amplitude / 2) * math.sin(2 * math.pi * i / period_mult)
                result[i] += seasonal

    return result


def add_noise(
    values: np.ndarray,
    noise_model: str = "gaussian",
    noise_level: float = 0.1
) -> np.ndarray:
    """Add random noise to the time-series using the chosen noise model.

    Three noise models capture different real-world measurement errors:
      gaussian         — constant-variance Gaussian noise (most common assumption).
      uniform          — bounded noise with a uniform distribution.
      heteroscedastic  — noise variance proportional to the signal magnitude
                         (common in financial data and activity metrics).

    Args:
        values: Original time-series values.
        noise_model: One of 'gaussian', 'uniform', 'heteroscedastic'.
        noise_level: Multiplier on the series std that sets the noise σ.

    Returns:
        Copy of values with noise added.
    """
    result = values.copy()
    std_val = np.std(values)                # Standard deviation of the original series
    noise_std = noise_level * std_val       # Scale noise magnitude by this factor

    if noise_model == "gaussian":
        # i.i.d. Gaussian noise with zero mean and constant σ
        noise = np.random.normal(0, noise_std, len(values))
        result = result + noise

    elif noise_model == "uniform":
        # Uniformly distributed noise in [-2σ, +2σ]
        noise = np.random.uniform(-noise_std * 2, noise_std * 2, len(values))
        result = result + noise

    elif noise_model == "heteroscedastic":
        # Noise σ depends on the local value: larger values → more noise
        for i in range(len(values)):
            local_std = noise_level * abs(values[i]) * 0.1  # σ proportional to |value|
            noise = np.random.normal(0, local_std)
            result[i] += noise

    return result


def generate_advanced_series(
    length: int = 2880,
    freq_minutes: int = 1,
    seed: Optional[int] = None,
    phases: Optional[List[Tuple[str, int]]] = None,
    missing_rate: float = 0.05,
    missing_pattern: str = "random",
    drift_type: Optional[str] = None,
    drift_magnitude: float = 0.1,
    seasonality_type: str = "daily",
    seasonality_amplitude: float = 5.0,
    noise_model: str = "gaussian",
    noise_level: float = 0.1,
    inject_anomalies: bool = True
) -> Tuple[List[datetime], np.ndarray]:
    """Generate a single advanced synthetic time-series with realistic characteristics.

    This is the main orchestrator function. It composes the series in layers:
      1. Phase patterns  (behavioural regime segments)
      2. Seasonality     (periodic cycles)
      3. Noise           (random perturbation)
      4. Concept drift   (non-stationarity)
      5. Anomalies       (outlier spikes / dips)
      6. Missingness     (NaN gaps)

    Args:
        length: Total number of data points.
        freq_minutes: Minutes between consecutive points.
        seed: Random seed for reproducibility.
        phases: List of (phase_type, phase_length) tuples. If None, defaults to
                four equal-length phases: Normal → High → Burnout → Recovery.
        missing_rate: Proportion of values to make missing (0.0–1.0).
        missing_pattern: 'random', 'burst', or 'periodic'.
        drift_type: 'gradual', 'sudden', 'recurring', or None for no drift.
        drift_magnitude: Fractional drift strength.
        seasonality_type: 'daily', 'weekly', 'monthly', 'multi'.
        seasonality_amplitude: Peak-to-trough seasonal magnitude.
        noise_model: 'gaussian', 'uniform', 'heteroscedastic'.
        noise_level: Noise magnitude as a fraction of series std.
        inject_anomalies: If True, inject ~1 anomaly per 400 points.

    Returns:
        Tuple of (timestamps, values) where timestamps is a list of datetime
        objects and values is a numpy array (may contain NaN).
    """
    # Fix both Python and NumPy random seeds for full reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ── Step 0: Generate evenly-spaced timestamps ────────────────────────
    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(minutes=i * freq_minutes) for i in range(length)]

    # ── Randomise global parameters ──────────────────────────────────────
    baseline = random.uniform(30, 70)   # Global productivity baseline
    amplitude = random.uniform(5, 20)   # Controls offset magnitudes and noise scaling

    # ── Step 1: Build behavioural-phase segments ─────────────────────────
    if phases is None:
        # Default: split the series into 4 equal phases cycling through all regimes
        phase_lengths = [length // 4] * 4
        phase_types = [
            BehavioralPhase.NORMAL,
            BehavioralPhase.HIGH_PERFORMANCE,
            BehavioralPhase.BURNOUT,
            BehavioralPhase.RECOVERY
        ]
        phases = list(zip(phase_types, phase_lengths))

    all_values = []       # Accumulator for the concatenated phase segments
    current_idx = 0       # Tracks how many points have been generated so far

    for phase_type, phase_length in phases:
        if current_idx >= length:
            break  # Already have enough data points

        # Clamp phase length so we don't exceed the total requested length
        actual_length = min(phase_length, length - current_idx)
        # Generate productivity values for this phase
        phase_values = generate_phase_pattern(phase_type, actual_length, baseline, amplitude)
        all_values.extend(phase_values)
        current_idx += actual_length

    # Trim to exactly `length` points (handles rounding from phase splits)
    values = np.array(all_values[:length])

    # ── Step 2: Add seasonality ──────────────────────────────────────────
    if seasonality_type:
        values = add_seasonality(values, seasonality_type, seasonality_amplitude)

    # ── Step 3: Add noise ────────────────────────────────────────────────
    if noise_model:
        values = add_noise(values, noise_model, noise_level)

    # ── Step 4: Inject concept drift ─────────────────────────────────────
    if drift_type:
        values = inject_time_drift(values, drift_type, drift_magnitude)

    # ── Step 5: Inject anomalies (outlier spikes / dips) ─────────────────
    if inject_anomalies:
        n_anomalies = max(1, length // 400)  # At least 1 anomaly
        for _ in range(n_anomalies):
            idx = random.randint(20, length - 20)  # Avoid edges
            if random.random() < 0.6:
                # 60 % chance: positive spike (2×–5× amplitude above normal)
                values[idx] += random.uniform(amplitude * 2, amplitude * 5)
            else:
                # 40 % chance: negative dip
                values[idx] -= random.uniform(amplitude * 2, amplitude * 5)

    # ── Step 6: Inject missingness (NaN gaps) ────────────────────────────
    if missing_rate > 0:
        values = inject_missingness(values, missing_rate, missing_pattern)

    return timestamps, values


# ── CLI entry point ──────────────────────────────────────────────────────
if __name__ == '__main__':
    # Set up argument parser with all configurable generation parameters
    p = argparse.ArgumentParser(description='Generate advanced synthetic time-series data')
    p.add_argument('--out', default='data/advanced_timeseries.csv', help='Output CSV path')
    p.add_argument('--n-series', type=int, default=10, help='Number of series to generate')
    p.add_argument('--length', type=int, default=2880, help='Length of each series')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--missing-rate', type=float, default=0.05, help='Missing value rate')
    p.add_argument('--missing-pattern', default='random', choices=['random', 'burst', 'periodic'],
                   help='Missing value pattern')
    p.add_argument('--drift-type', default=None, choices=['gradual', 'sudden', 'recurring'],
                   help='Time drift type')
    p.add_argument('--seasonality', default='daily', choices=['daily', 'weekly', 'monthly', 'multi'],
                   help='Seasonality type')
    p.add_argument('--noise-model', default='gaussian', choices=['gaussian', 'uniform', 'heteroscedastic'],
                   help='Noise model')
    args = p.parse_args()

    # Ensure the output directory exists
    try:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    except Exception:
        pass  # Silently ignore if directory already exists or dirname is empty

    # Write all series to a single CSV file
    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'series_id', 'value'])  # CSV header

        for s in range(args.n_series):
            # Generate one advanced series per iteration, each with a unique seed offset
            ts, vals = generate_advanced_series(
                length=args.length,
                seed=args.seed + s,           # Unique seed per series for variety
                missing_rate=args.missing_rate,
                missing_pattern=args.missing_pattern,
                drift_type=args.drift_type,
                seasonality_type=args.seasonality,
                noise_model=args.noise_model
            )
            sid = f"series_{s}"  # Series identifier, e.g. "series_0"

            for t, v in zip(ts, vals):
                # Skip NaN values (missing data) — they are omitted from the CSV
                if not np.isnan(v):
                    writer.writerow([t.isoformat(), sid, f"{v:.4f}"])

    print(f"Generated {args.n_series} advanced time series to {args.out}")
