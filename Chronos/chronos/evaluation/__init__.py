"""
Evaluation Package – Metrics, Benchmarking, and Explainability for Chronos
============================================================================
This package provides everything needed to assess and compare Chronos models:

  • metrics.py         – Core metric functions:
        calculate_metrics()       → MSE, RMSE, MAE, MAPE, directional accuracy, R²
        calculate_stability_score() → measures prediction variance across multiple runs
        calculate_latency_score()   → normalizes inference latency to a 0–1 score
        composite_score()          → weighted blend of accuracy + stability + latency

  • benchmark.py       – BenchmarkRunner class that compares a Chronos model
                         against classical baselines (ARIMA, Prophet, SimpleRNN).
                         Each baseline is timed, scored, and returned as a
                         BenchmarkResult dataclass.

  • explainability.py  – ModelExplainer class that uses SHAP (SHapley Additive
                         exPlanations) to explain individual predictions by
                         computing per-feature importance values.

Typical usage:
    from chronos.evaluation import calculate_metrics, BenchmarkRunner, ModelExplainer
    metrics = calculate_metrics(y_true, y_pred)
    runner = BenchmarkRunner()
    results = runner.run_all_benchmarks(train_data, test_data)
"""

# Core forecasting metrics (MSE, RMSE, MAE, MAPE, directional accuracy, R²).
from .metrics import (
    calculate_metrics,
    calculate_stability_score,
    calculate_latency_score,
    composite_score
)

# Baseline comparison (ARIMA, Prophet, SimpleRNN).
from .benchmark import BenchmarkRunner, run_baseline_benchmarks

# SHAP-based model interpretation.
from .explainability import ModelExplainer, explain_prediction

# __all__ defines the public API — these symbols are exported by `from chronos.evaluation import *`.
__all__ = [
    'calculate_metrics',
    'calculate_stability_score',
    'calculate_latency_score',
    'composite_score',
    'BenchmarkRunner',
    'run_baseline_benchmarks',
    'ModelExplainer',
    'explain_prediction'
]
