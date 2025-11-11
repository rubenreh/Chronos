"""Benchmarking utilities for comparing models."""
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

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

from chronos.evaluation.metrics import calculate_metrics, calculate_latency_score, composite_score


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    model_name: str
    metrics: Dict[str, float]
    latency_ms: float
    composite_score: float


class BenchmarkRunner:
    """Run benchmarks on multiple models."""
    
    def __init__(self):
        """Initialize benchmark runner."""
        self.results: List[BenchmarkResult] = []
    
    def benchmark_arima(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        order: tuple = (1, 1, 1)
    ) -> Optional[BenchmarkResult]:
        """Benchmark ARIMA model."""
        if not ARIMA_AVAILABLE:
            return None
        
        try:
            start_time = time.time()
            
            # Fit ARIMA
            model = ARIMA(train_data, order=order)
            fitted = model.fit()
            
            # Forecast
            forecast = fitted.forecast(steps=len(test_data))
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate metrics
            metrics = calculate_metrics(test_data, forecast)
            latency_score = calculate_latency_score(latency_ms)
            comp_score = composite_score(metrics, 0.8, latency_score)
            
            return BenchmarkResult(
                model_name='ARIMA',
                metrics=metrics,
                latency_ms=latency_ms,
                composite_score=comp_score
            )
        except Exception as e:
            print(f"ARIMA benchmark failed: {e}")
            return None
    
    def benchmark_prophet(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Optional[BenchmarkResult]:
        """Benchmark Prophet model."""
        if not PROPHET_AVAILABLE:
            return None
        
        try:
            start_time = time.time()
            
            # Prepare data for Prophet
            if timestamps is None:
                timestamps = pd.date_range(start='2025-01-01', periods=len(train_data), freq='1H')
            
            df = pd.DataFrame({
                'ds': timestamps[:len(train_data)],
                'y': train_data
            })
            
            # Fit Prophet
            model = Prophet()
            model.fit(df)
            
            # Forecast
            future = model.make_future_dataframe(periods=len(test_data), freq='1H')
            forecast = model.predict(future)
            
            predictions = forecast['yhat'].tail(len(test_data)).values
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate metrics
            metrics = calculate_metrics(test_data, predictions)
            latency_score = calculate_latency_score(latency_ms)
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
        """Benchmark a simple RNN model."""
        try:
            start_time = time.time()
            
            # Prepare input
            if len(train_data) < input_len:
                return None
            
            # Use last input_len values as input
            x_input = train_data[-input_len:].reshape(1, input_len, 1)
            
            # Predict
            predictions = []
            current_input = x_input.copy()
            
            for _ in range(len(test_data)):
                pred = model.predict(current_input, verbose=0)[0, 0]
                predictions.append(pred)
                # Update input (shift and append prediction)
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = pred
            
            predictions = np.array(predictions)
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate metrics
            metrics = calculate_metrics(test_data, predictions)
            latency_score = calculate_latency_score(latency_ms)
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
        """Run all available benchmarks."""
        results = []
        
        # ARIMA
        arima_result = self.benchmark_arima(train_data, test_data)
        if arima_result:
            results.append(arima_result)
        
        # Prophet
        prophet_result = self.benchmark_prophet(train_data, test_data, timestamps)
        if prophet_result:
            results.append(prophet_result)
        
        # Simple RNN
        if rnn_model is not None:
            rnn_result = self.benchmark_simple_rnn(rnn_model, train_data, test_data)
            if rnn_result:
                results.append(rnn_result)
        
        self.results = results
        return results


def run_baseline_benchmarks(
    train_data: np.ndarray,
    test_data: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None
) -> List[BenchmarkResult]:
    """Convenience function to run baseline benchmarks."""
    runner = BenchmarkRunner()
    return runner.run_all_benchmarks(train_data, test_data, timestamps)

