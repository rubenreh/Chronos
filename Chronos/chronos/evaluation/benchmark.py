"""
Benchmark Module – Baseline Model Comparison for Chronos
==========================================================
This module lets you compare Chronos's deep-learning forecasters against
classical statistical baselines. Having baseline benchmarks is essential for
demonstrating that the neural models actually add value over simpler methods.

Baselines implemented:
  • ARIMA   – AutoRegressive Integrated Moving Average (statsmodels).
  • Prophet – Facebook Prophet, an additive decomposition model.
  • SimpleRNN – a pre-trained Keras/TF RNN passed in by the caller.

Each benchmark:
  1. Fits the baseline on `train_data`.
  2. Forecasts `len(test_data)` steps.
  3. Measures wall-clock latency.
  4. Computes accuracy metrics (MSE, RMSE, MAE, MAPE, R²).
  5. Computes a composite score (accuracy + stability + latency).
  6. Returns a BenchmarkResult dataclass.

Usage:
    runner = BenchmarkRunner()
    results = runner.run_all_benchmarks(train_data, test_data)
    for r in results:
        print(f"{r.model_name}: RMSE={r.metrics['rmse']:.4f}")
"""

import time          # Wall-clock timing for latency measurement
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Conditional imports for optional baseline libraries.
# If the user hasn't installed statsmodels/prophet, the corresponding
# benchmark is silently skipped rather than crashing the whole module.
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Chronos evaluation metrics used to score each baseline.
from chronos.evaluation.metrics import calculate_metrics, calculate_latency_score, composite_score


@dataclass
class BenchmarkResult:
    """Immutable container for the results of a single benchmark run.

    Attributes:
        model_name: Human-readable name of the baseline (e.g. "ARIMA").
        metrics: Dictionary of accuracy metrics (MSE, RMSE, MAE, MAPE, R²).
        latency_ms: Total wall-clock time for fit + forecast, in milliseconds.
        composite_score: Weighted blend of accuracy, stability, and latency.
    """
    model_name: str
    metrics: Dict[str, float]
    latency_ms: float
    composite_score: float


class BenchmarkRunner:
    """Orchestrates benchmark runs for multiple baseline models.

    Instantiate once, then call run_all_benchmarks() to evaluate every
    available baseline. Results are stored in self.results and also returned.
    """

    def __init__(self):
        """Initialize the benchmark runner with an empty results list."""
        self.results: List[BenchmarkResult] = []

    def benchmark_arima(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        order: tuple = (1, 1, 1)
    ) -> Optional[BenchmarkResult]:
        """Fit an ARIMA model and measure its forecasting performance.

        ARIMA (AutoRegressive Integrated Moving Average) is a classical
        statistical baseline. The order (p, d, q) controls the number of
        AR terms (p), differencing passes (d), and MA terms (q).

        Args:
            train_data: 1-D numpy array of training observations.
            test_data: 1-D numpy array of ground-truth future values.
            order: ARIMA (p, d, q) order tuple.

        Returns:
            BenchmarkResult if successful, None if ARIMA is unavailable or fails.
        """
        # Skip silently if statsmodels is not installed.
        if not ARIMA_AVAILABLE:
            return None

        try:
            # Start the wall-clock timer — includes both fitting and forecasting.
            start_time = time.time()

            # Fit the ARIMA model on the training data.
            model = ARIMA(train_data, order=order)
            fitted = model.fit()

            # Generate forecasts for len(test_data) future steps.
            forecast = fitted.forecast(steps=len(test_data))

            # Record total elapsed time in milliseconds.
            latency_ms = (time.time() - start_time) * 1000

            # Compute accuracy metrics by comparing forecast to ground truth.
            metrics = calculate_metrics(test_data, forecast)

            # Convert raw latency to a 0–1 score (lower latency = higher score).
            latency_score = calculate_latency_score(latency_ms)

            # Compute the composite score (0.8 is a fixed stability estimate for ARIMA
            # since it's deterministic and we don't run multiple times here).
            comp_score = composite_score(metrics, 0.8, latency_score)

            return BenchmarkResult(
                model_name='ARIMA',
                metrics=metrics,
                latency_ms=latency_ms,
                composite_score=comp_score
            )
        except Exception as e:
            # ARIMA can fail on non-stationary or short series; log and move on.
            print(f"ARIMA benchmark failed: {e}")
            return None

    def benchmark_prophet(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Optional[BenchmarkResult]:
        """Fit a Facebook Prophet model and measure its forecasting performance.

        Prophet is an additive decomposition model (trend + seasonality +
        holidays). It requires a DataFrame with columns 'ds' (datetime) and
        'y' (value).

        Args:
            train_data: 1-D numpy array of training observations.
            test_data: 1-D numpy array of ground-truth future values.
            timestamps: Optional datetime index. If None, a synthetic hourly
                        index starting from 2025-01-01 is generated.

        Returns:
            BenchmarkResult if successful, None if Prophet is unavailable or fails.
        """
        # Skip silently if the prophet package is not installed.
        if not PROPHET_AVAILABLE:
            return None

        try:
            start_time = time.time()

            # Generate a synthetic timestamp index if none was provided.
            if timestamps is None:
                timestamps = pd.date_range(start='2025-01-01', periods=len(train_data), freq='1H')

            # Prophet requires a DataFrame with specific column names: 'ds' and 'y'.
            df = pd.DataFrame({
                'ds': timestamps[:len(train_data)],
                'y': train_data
            })

            # Instantiate and fit the Prophet model.
            model = Prophet()
            model.fit(df)

            # Build a future DataFrame extending len(test_data) steps beyond training.
            future = model.make_future_dataframe(periods=len(test_data), freq='1H')

            # Predict over the entire range (training + future).
            forecast = model.predict(future)

            # Extract only the forecasted values corresponding to the test period.
            predictions = forecast['yhat'].tail(len(test_data)).values

            latency_ms = (time.time() - start_time) * 1000

            # Compute accuracy metrics against the ground-truth test data.
            metrics = calculate_metrics(test_data, predictions)
            latency_score = calculate_latency_score(latency_ms)

            # 0.8 is a fixed stability estimate for Prophet (deterministic model).
            comp_score = composite_score(metrics, 0.8, latency_score)

            return BenchmarkResult(
                model_name='Prophet',
                metrics=metrics,
                latency_ms=latency_ms,
                composite_score=comp_score
            )
        except Exception as e:
            print(f"Prophet benchmark failed: {e}")
            return None

    def benchmark_simple_rnn(
        self,
        model: Any,
        train_data: np.ndarray,
        test_data: np.ndarray,
        input_len: int = 60
    ) -> Optional[BenchmarkResult]:
        """Benchmark a pre-trained SimpleRNN (Keras/TF) model via autoregressive rollout.

        The RNN predicts one step at a time. After each prediction, the input
        window is shifted forward and the prediction is appended, creating an
        autoregressive forecast loop.

        Args:
            model: A pre-trained Keras model with a .predict() method.
            train_data: 1-D numpy array — the last input_len values are used as
                        the initial input window.
            test_data: 1-D numpy array of ground-truth future values.
            input_len: Length of the model's input window.

        Returns:
            BenchmarkResult if successful, None if the model fails.
        """
        try:
            start_time = time.time()

            # Need at least input_len values to form the initial input window.
            if len(train_data) < input_len:
                return None

            # Take the last input_len values from training data as the seed input.
            # Reshape to (1, input_len, 1) for Keras: (batch, timesteps, features).
            x_input = train_data[-input_len:].reshape(1, input_len, 1)

            # Autoregressive forecast loop: predict one step, shift window, repeat.
            predictions = []
            current_input = x_input.copy()

            for _ in range(len(test_data)):
                # Predict the next single value.
                pred = model.predict(current_input, verbose=0)[0, 0]
                predictions.append(pred)

                # Shift the input window left by 1 and insert the prediction at the end.
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = pred

            predictions = np.array(predictions)
            latency_ms = (time.time() - start_time) * 1000

            # Compute accuracy metrics.
            metrics = calculate_metrics(test_data, predictions)
            latency_score = calculate_latency_score(latency_ms)

            # 0.8 is a fixed stability estimate for the SimpleRNN.
            comp_score = composite_score(metrics, 0.8, latency_score)

            return BenchmarkResult(
                model_name='SimpleRNN',
                metrics=metrics,
                latency_ms=latency_ms,
                composite_score=comp_score
            )
        except Exception as e:
            print(f"SimpleRNN benchmark failed: {e}")
            return None

    def run_all_benchmarks(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        rnn_model: Optional[Any] = None
    ) -> List[BenchmarkResult]:
        """Run every available baseline benchmark and collect results.

        Tries ARIMA, Prophet, and (optionally) SimpleRNN. Baselines whose
        dependencies are missing are silently skipped.

        Args:
            train_data: 1-D numpy array of training observations.
            test_data: 1-D numpy array of ground-truth test observations.
            timestamps: Optional datetime index for Prophet.
            rnn_model: Optional pre-trained Keras RNN model.

        Returns:
            List of BenchmarkResult objects for each successful baseline.
        """
        results = []

        # --- ARIMA baseline ---
        arima_result = self.benchmark_arima(train_data, test_data)
        if arima_result:
            results.append(arima_result)

        # --- Prophet baseline ---
        prophet_result = self.benchmark_prophet(train_data, test_data, timestamps)
        if prophet_result:
            results.append(prophet_result)

        # --- SimpleRNN baseline (only if a model was provided) ---
        if rnn_model is not None:
            rnn_result = self.benchmark_simple_rnn(rnn_model, train_data, test_data)
            if rnn_result:
                results.append(rnn_result)

        # Store results on the instance for later inspection.
        self.results = results
        return results


def run_baseline_benchmarks(
    train_data: np.ndarray,
    test_data: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None
) -> List[BenchmarkResult]:
    """Convenience function to run all baselines in a single call.

    Creates a BenchmarkRunner internally and delegates to run_all_benchmarks().
    Useful in scripts and notebooks where you don't need to keep the runner.

    Args:
        train_data: 1-D numpy array of training observations.
        test_data: 1-D numpy array of test observations.
        timestamps: Optional datetime index for Prophet.

    Returns:
        List of BenchmarkResult objects.
    """
    runner = BenchmarkRunner()
    return runner.run_all_benchmarks(train_data, test_data, timestamps)
