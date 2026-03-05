"""
Evaluation Metrics – Comprehensive Forecasting Metrics for Chronos
====================================================================
This module computes all the quantitative metrics used to evaluate Chronos
forecasting models. The metrics fall into three categories:

  1. **Accuracy metrics** (calculated by calculate_metrics):
       • MSE  – Mean Squared Error: average of squared prediction errors.
       • RMSE – Root Mean Squared Error: square root of MSE, same units as data.
       • MAE  – Mean Absolute Error: average of absolute prediction errors.
       • MAPE – Mean Absolute Percentage Error: percentage-based error metric.
       • Directional Accuracy – fraction of times the predicted direction
         (up/down) matches the actual direction.
       • R²   – Coefficient of determination: 1 means perfect fit, 0 means
         the model is no better than predicting the mean.

  2. **Stability score** (calculate_stability_score):
       Measures how consistent predictions are across multiple inference runs.
       A stable model produces nearly identical outputs each time. The score
       uses the coefficient of variation (CV) across runs and maps it to [0, 1].

  3. **Latency score** (calculate_latency_score):
       Maps raw inference latency (in ms) to a 0–1 score where lower latency
       yields a higher score. Used to reward faster models.

  4. **Composite score** (composite_score):
       A single number that blends accuracy, stability, and latency scores
       with configurable weights (default: 60% accuracy, 25% stability,
       15% latency). This is the primary metric for model comparison.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# scikit-learn provides optimized implementations of common regression metrics.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_r2: bool = True
) -> Dict[str, float]:
    """Calculate a comprehensive suite of forecasting metrics.

    This is the primary evaluation function called after every epoch (on the
    validation set) and at the end of training (on the test set).

    Args:
        y_true: 1-D array of ground-truth values.
        y_pred: 1-D array of predicted values (same length as y_true).
        include_r2: Whether to include the R² metric (sometimes omitted when
                    the number of samples is very small).

    Returns:
        Dictionary mapping metric names to their float values.
    """
    metrics = {}

    # --- Basic regression metrics ---

    # MSE: Mean Squared Error — average of (y_true - y_pred)².
    # Penalizes large errors more heavily due to squaring.
    metrics['mse'] = float(mean_squared_error(y_true, y_pred))

    # RMSE: square root of MSE — same units as the original data, making it
    # more interpretable than MSE.
    metrics['rmse'] = float(np.sqrt(metrics['mse']))

    # MAE: Mean Absolute Error — average of |y_true - y_pred|.
    # Less sensitive to outliers than MSE/RMSE.
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))

    # MAPE: Mean Absolute Percentage Error — measures error as a percentage
    # of the true value. The 1e-8 in the denominator prevents division by zero
    # when y_true contains values near zero.
    metrics['mape'] = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)

    # --- Directional accuracy ---
    # Measures how often the model correctly predicts whether the next value
    # goes up or down. Important for trading and trend-following applications.
    if len(y_true) > 1:
        # np.diff computes consecutive differences; np.sign maps to -1, 0, or +1.
        true_direction = np.sign(np.diff(y_true))   # Actual direction of change.
        pred_direction = np.sign(np.diff(y_pred))    # Predicted direction of change.
        # Fraction of timesteps where predicted direction matches actual direction.
        metrics['directional_accuracy'] = float(np.mean(true_direction == pred_direction))

    # --- R² (coefficient of determination) ---
    # R² = 1 means perfect predictions; R² = 0 means the model is equivalent
    # to always predicting the mean; R² < 0 means worse than the mean.
    if include_r2:
        metrics['r2'] = float(r2_score(y_true, y_pred))

    return metrics


def calculate_stability_score(
    predictions: List[np.ndarray],
    window_size: int = 10
) -> float:
    """Measure how consistent a model's predictions are across multiple runs.

    A perfectly deterministic model (no dropout at inference, fixed seed) would
    score 1.0. Models with stochastic elements (e.g., Monte Carlo dropout) will
    score lower if their outputs vary significantly between runs.

    The score is computed as:
        stability = 1 / (1 + mean_CV)
    where CV is the coefficient of variation (std / |mean|) across runs.

    Args:
        predictions: List of prediction arrays from multiple inference runs.
                     Each array should have the same shape.
        window_size: (Reserved for future rolling-window variance; currently unused.)

    Returns:
        Stability score in (0, 1]. Higher is more stable.
    """
    # Need at least two runs to measure variation; a single run is trivially stable.
    if len(predictions) < 2:
        return 1.0

    # Stack predictions into a 2-D matrix: rows = runs, columns = timesteps.
    pred_matrix = np.array(predictions)

    # Compute the mean prediction at each timestep across all runs.
    mean_pred = np.mean(pred_matrix, axis=0)

    # Compute the standard deviation at each timestep across all runs.
    std_pred = np.std(pred_matrix, axis=0)

    # Coefficient of variation: relative spread of predictions.
    # 1e-8 prevents division by zero when the mean prediction is near zero.
    cv = std_pred / (np.abs(mean_pred) + 1e-8)

    # Map mean CV to a 0–1 stability score using 1/(1+x) transform.
    # As CV → 0 (no variation), stability → 1. As CV → ∞, stability → 0.
    stability = 1.0 / (1.0 + np.mean(cv))

    return float(stability)


def calculate_latency_score(
    latency_ms: float,
    max_latency_ms: float = 1000.0
) -> float:
    """Convert raw inference latency into a normalized 0–1 score.

    The score linearly decreases as latency increases, reaching 0 at
    max_latency_ms. This rewards faster models in the composite score.

    Args:
        latency_ms: Measured inference latency in milliseconds.
        max_latency_ms: The latency threshold above which the score is 0.
                        Default is 1000 ms (1 second).

    Returns:
        Latency score between 0.0 (very slow) and 1.0 (instantaneous).
    """
    # Guard against nonsensical negative latency values.
    if latency_ms <= 0:
        return 1.0

    # Linear mapping: score = 1 - (latency / max_latency), clamped to [0, 1].
    score = max(0.0, 1.0 - (latency_ms / max_latency_ms))
    return float(score)


def composite_score(
    accuracy_metrics: Dict[str, float],
    stability_score: float,
    latency_score: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """Compute a single composite score blending accuracy, stability, and latency.

    This is the headline metric used to rank and compare different models and
    configurations in Chronos. A higher composite score is better.

    Default weights:
        accuracy  : 60%  – the model must be accurate above all.
        stability : 25%  – consistent predictions are important for trust.
        latency   : 15%  – fast inference matters for real-time applications.

    Args:
        accuracy_metrics: Dictionary from calculate_metrics() (needs 'rmse' and 'mae').
        stability_score: Output of calculate_stability_score(), range [0, 1].
        latency_score: Output of calculate_latency_score(), range [0, 1].
        weights: Optional custom weights. Must have keys 'accuracy', 'stability', 'latency'.

    Returns:
        Composite score (higher is better).
    """
    # Use default weights if none are provided.
    if weights is None:
        weights = {
            'accuracy': 0.6,
            'stability': 0.25,
            'latency': 0.15
        }

    # Extract RMSE and MAE from the accuracy metrics dictionary.
    rmse = accuracy_metrics.get('rmse', 1.0)
    mae = accuracy_metrics.get('mae', 1.0)

    # Convert error metrics to a 0–1 accuracy score using an inverse transform:
    # accuracy_score = 1 / (1 + rmse + mae).
    # When rmse + mae → 0 (perfect), accuracy_score → 1.
    # When rmse + mae → ∞ (terrible), accuracy_score → 0.
    accuracy_score = 1.0 / (1.0 + rmse + mae)

    # Weighted sum of the three component scores.
    composite = (
        weights['accuracy'] * accuracy_score +
        weights['stability'] * stability_score +
        weights['latency'] * latency_score
    )

    return float(composite)
