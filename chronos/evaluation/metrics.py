"""Evaluation metrics for time-series forecasting."""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_r2: bool = True
) -> Dict[str, float]:
    """Calculate comprehensive forecasting metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        include_r2: Whether to include R² score
    
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = float(mean_squared_error(y_true, y_pred))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    metrics['mape'] = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    
    # Directional accuracy
    if len(y_true) > 1:
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        metrics['directional_accuracy'] = float(np.mean(true_direction == pred_direction))
    
    # R² score
    if include_r2:
        metrics['r2'] = float(r2_score(y_true, y_pred))
    
    return metrics


def calculate_stability_score(
    predictions: List[np.ndarray],
    window_size: int = 10
) -> float:
    """Calculate stability score based on prediction variance.
    
    Args:
        predictions: List of prediction arrays from multiple runs
        window_size: Window size for rolling variance calculation
    
    Returns:
        Stability score (higher is more stable)
    """
    if len(predictions) < 2:
        return 1.0
    
    # Stack predictions
    pred_matrix = np.array(predictions)
    
    # Calculate coefficient of variation across runs
    mean_pred = np.mean(pred_matrix, axis=0)
    std_pred = np.std(pred_matrix, axis=0)
    
    # Avoid division by zero
    cv = std_pred / (np.abs(mean_pred) + 1e-8)
    stability = 1.0 / (1.0 + np.mean(cv))
    
    return float(stability)


def calculate_latency_score(
    latency_ms: float,
    max_latency_ms: float = 1000.0
) -> float:
    """Calculate latency score (higher is better, lower latency).
    
    Args:
        latency_ms: Prediction latency in milliseconds
        max_latency_ms: Maximum acceptable latency
    
    Returns:
        Latency score between 0 and 1
    """
    if latency_ms <= 0:
        return 1.0
    
    score = max(0.0, 1.0 - (latency_ms / max_latency_ms))
    return float(score)


def composite_score(
    accuracy_metrics: Dict[str, float],
    stability_score: float,
    latency_score: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """Calculate composite score combining accuracy, stability, and latency.
    
    Args:
        accuracy_metrics: Dictionary of accuracy metrics
        stability_score: Stability score (0-1)
        latency_score: Latency score (0-1)
        weights: Optional weights for different components
    
    Returns:
        Composite score (higher is better)
    """
    if weights is None:
        weights = {
            'accuracy': 0.6,
            'stability': 0.25,
            'latency': 0.15
        }
    
    # Normalize accuracy metrics (use inverse of RMSE, normalized)
    rmse = accuracy_metrics.get('rmse', 1.0)
    mae = accuracy_metrics.get('mae', 1.0)
    
    # Normalize to 0-1 scale (assuming reasonable bounds)
    accuracy_score = 1.0 / (1.0 + rmse + mae)
    
    # Weighted combination
    composite = (
        weights['accuracy'] * accuracy_score +
        weights['stability'] * stability_score +
        weights['latency'] * latency_score
    )
    
    return float(composite)

