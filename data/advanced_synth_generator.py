"""Advanced synthetic time-series generator with missingness, drift, phases, and seasonality.
Produces realistic productivity datasets for testing and benchmarking.
"""
import argparse
from datetime import datetime, timedelta
import math
import random
import csv
import os
import numpy as np
from typing import List, Tuple, Optional


class BehavioralPhase:
    """Represents a behavioral phase in productivity."""
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
    """Generate productivity pattern for a specific phase.
    
    Args:
        phase: Phase type (high_performance, burnout, recovery, normal)
        length: Length of phase in timesteps
        baseline: Baseline productivity value
        amplitude: Amplitude of variation
    
    Returns:
        Array of productivity values
    """
    values = []
    
    if phase == BehavioralPhase.HIGH_PERFORMANCE:
        # High values with positive trend
        trend = random.uniform(0.001, 0.003)
        for i in range(length):
            val = baseline + amplitude * 0.3 + trend * i + random.gauss(0, amplitude * 0.1)
            values.append(val)
    
    elif phase == BehavioralPhase.BURNOUT:
        # Declining values with high volatility
        trend = random.uniform(-0.003, -0.001)
        for i in range(length):
            val = baseline - amplitude * 0.2 + trend * i + random.gauss(0, amplitude * 0.25)
            values.append(val)
    
    elif phase == BehavioralPhase.RECOVERY:
        # Rising values from low baseline
        trend = random.uniform(0.002, 0.004)
        start_baseline = baseline - amplitude * 0.3
        for i in range(length):
            val = start_baseline + trend * i + random.gauss(0, amplitude * 0.15)
            values.append(val)
    
    else:  # NORMAL
        # Stable values around baseline
        for i in range(length):
            val = baseline + random.gauss(0, amplitude * 0.15)
            values.append(val)
    
    return np.array(values)


def inject_missingness(
    values: np.ndarray,
    missing_rate: float = 0.05,
    missing_pattern: str = "random"
) -> np.ndarray:
    """Inject missing values into time series.
    
    Args:
        values: Original values
        missing_rate: Proportion of values to make missing
        missing_pattern: Pattern type ('random', 'burst', 'periodic')
    
    Returns:
        Array with NaN values inserted
    """
    result = values.copy()
    n_missing = int(len(values) * missing_rate)
    
    if missing_pattern == "random":
        # Random missing values
        missing_indices = random.sample(range(len(values)), n_missing)
        for idx in missing_indices:
            result[idx] = np.nan
    
    elif missing_pattern == "burst":
        # Burst of missing values
        start_idx = random.randint(0, len(values) - n_missing)
        result[start_idx:start_idx + n_missing] = np.nan
    
    elif missing_pattern == "periodic":
        # Periodic missing values (e.g., weekends)
        period = random.choice([7, 14, 30])  # Days
        for i in range(0, len(values), period):
            if i < len(values):
                result[i] = np.nan
    
    return result


def inject_time_drift(
    values: np.ndarray,
    drift_type: str = "gradual",
    drift_magnitude: float = 0.1
) -> np.ndarray:
    """Inject time drift (concept drift) into time series.
    
    Args:
        values: Original values
        drift_type: Type of drift ('gradual', 'sudden', 'recurring')
        drift_magnitude: Magnitude of drift
    
    Returns:
        Array with drift applied
    """
    result = values.copy()
    n = len(values)
    
    if drift_type == "gradual":
        # Gradual linear drift
        drift = np.linspace(0, drift_magnitude * np.mean(values), n)
        result = result + drift
    
    elif drift_type == "sudden":
        # Sudden shift at midpoint
        shift_point = n // 2
        shift = drift_magnitude * np.mean(values[:shift_point])
        result[shift_point:] = result[shift_point:] + shift
    
    elif drift_type == "recurring":
        # Recurring drift pattern
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
    """Add seasonality patterns to time series.
    
    Args:
        values: Original values
        seasonality_type: Type of seasonality ('daily', 'weekly', 'monthly')
        amplitude: Amplitude of seasonal variation
    
    Returns:
        Array with seasonality added
    """
    result = values.copy()
    n = len(values)
    
    if seasonality_type == "daily":
        # Daily pattern (24-hour cycle)
        period = 24 * 60  # 24 hours in minutes
        for i in range(n):
            seasonal = amplitude * math.sin(2 * math.pi * i / period)
            result[i] += seasonal
    
    elif seasonality_type == "weekly":
        # Weekly pattern (7-day cycle)
        period = 7 * 24 * 60
        for i in range(n):
            seasonal = amplitude * math.sin(2 * math.pi * i / period)
            result[i] += seasonal
    
    elif seasonality_type == "monthly":
        # Monthly pattern (30-day cycle)
        period = 30 * 24 * 60
        for i in range(n):
            seasonal = amplitude * math.sin(2 * math.pi * i / period)
            result[i] += seasonal
    
    elif seasonality_type == "multi":
        # Multiple seasonalities
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
    """Add noise to time series.
    
    Args:
        values: Original values
        noise_model: Type of noise ('gaussian', 'uniform', 'heteroscedastic')
        noise_level: Level of noise (as proportion of std)
    
    Returns:
        Array with noise added
    """
    result = values.copy()
    std_val = np.std(values)
    noise_std = noise_level * std_val
    
    if noise_model == "gaussian":
        noise = np.random.normal(0, noise_std, len(values))
        result = result + noise
    
    elif noise_model == "uniform":
        noise = np.random.uniform(-noise_std * 2, noise_std * 2, len(values))
        result = result + noise
    
    elif noise_model == "heteroscedastic":
        # Noise variance depends on value
        for i in range(len(values)):
            local_std = noise_level * abs(values[i]) * 0.1
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
    """Generate advanced synthetic time series with multiple realistic patterns.
    
    Args:
        length: Length of series in timesteps
        freq_minutes: Frequency in minutes
        seed: Random seed
        phases: List of (phase_type, phase_length) tuples
        missing_rate: Rate of missing values
        missing_pattern: Pattern of missingness
        drift_type: Type of time drift
        drift_magnitude: Magnitude of drift
        seasonality_type: Type of seasonality
        seasonality_amplitude: Amplitude of seasonality
        noise_model: Type of noise
        noise_level: Level of noise
        inject_anomalies: Whether to inject anomalies
    
    Returns:
        Tuple of (timestamps, values)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate timestamps
    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(minutes=i * freq_minutes) for i in range(length)]
    
    # Baseline parameters
    baseline = random.uniform(30, 70)
    amplitude = random.uniform(5, 20)
    
    # Generate phases if not provided
    if phases is None:
        # Default: mix of phases
        phase_lengths = [length // 4] * 4
        phase_types = [
            BehavioralPhase.NORMAL,
            BehavioralPhase.HIGH_PERFORMANCE,
            BehavioralPhase.BURNOUT,
            BehavioralPhase.RECOVERY
        ]
        phases = list(zip(phase_types, phase_lengths))
    
    # Generate values for each phase
    all_values = []
    current_idx = 0
    
    for phase_type, phase_length in phases:
        if current_idx >= length:
            break
        
        actual_length = min(phase_length, length - current_idx)
        phase_values = generate_phase_pattern(phase_type, actual_length, baseline, amplitude)
        all_values.extend(phase_values)
        current_idx += actual_length
    
    # Ensure we have the right length
    values = np.array(all_values[:length])
    
    # Add seasonality
    if seasonality_type:
        values = add_seasonality(values, seasonality_type, seasonality_amplitude)
    
    # Add noise
    if noise_model:
        values = add_noise(values, noise_model, noise_level)
    
    # Inject time drift
    if drift_type:
        values = inject_time_drift(values, drift_type, drift_magnitude)
    
    # Inject anomalies
    if inject_anomalies:
        n_anomalies = max(1, length // 400)
        for _ in range(n_anomalies):
            idx = random.randint(20, length - 20)
            if random.random() < 0.6:
                # Spike
                values[idx] += random.uniform(amplitude * 2, amplitude * 5)
            else:
                # Dip
                values[idx] -= random.uniform(amplitude * 2, amplitude * 5)
    
    # Inject missingness
    if missing_rate > 0:
        values = inject_missingness(values, missing_rate, missing_pattern)
    
    return timestamps, values


if __name__ == '__main__':
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
    
    try:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    except Exception:
        pass
    
    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'series_id', 'value'])
        
        for s in range(args.n_series):
            ts, vals = generate_advanced_series(
                length=args.length,
                seed=args.seed + s,
                missing_rate=args.missing_rate,
                missing_pattern=args.missing_pattern,
                drift_type=args.drift_type,
                seasonality_type=args.seasonality,
                noise_model=args.noise_model
            )
            sid = f"series_{s}"
            
            for t, v in zip(ts, vals):
                if not np.isnan(v):
                    writer.writerow([t.isoformat(), sid, f"{v:.4f}"])
    
    print(f"Generated {args.n_series} advanced time series to {args.out}")
