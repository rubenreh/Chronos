"""Ensemble models for improved forecasting."""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
from enum import Enum


class EnsembleMethod(Enum):
    """Ensemble combination methods."""
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    STACKING = "stacking"


class ModelEnsemble:
    """Ensemble of multiple forecasting models."""
    
    def __init__(
        self,
        models: List[Any],
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
        weights: Optional[List[float]] = None,
        meta_model: Optional[Any] = None
    ):
        """Initialize ensemble.
        
        Args:
            models: List of trained models
            method: Ensemble combination method
            weights: Optional weights for weighted average (must sum to 1)
            meta_model: Optional meta-model for stacking
        """
        self.models = models
        self.method = method
        self.weights = weights
        self.meta_model = meta_model
        
        if weights is None and method == EnsembleMethod.WEIGHTED_AVERAGE:
            # Equal weights by default
            self.weights = [1.0 / len(models)] * len(models)
        
        if weights and abs(sum(weights) - 1.0) > 1e-6:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make ensemble prediction.
        
        Args:
            x: Input data (shape: batch, seq_len, features) or (seq_len, features)
        
        Returns:
            Ensemble prediction
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            if isinstance(x, np.ndarray):
                x_tensor = torch.tensor(x, dtype=torch.float32)
            else:
                x_tensor = x
            
            model.eval()
            with torch.no_grad():
                if len(x_tensor.shape) == 2:
                    x_tensor = x_tensor.unsqueeze(0)
                pred = model(x_tensor)
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)  # (n_models, batch, output_dim)
        
        # Combine predictions
        if self.method == EnsembleMethod.AVERAGE:
            return np.mean(predictions, axis=0)
        elif self.method == EnsembleMethod.WEIGHTED_AVERAGE:
            weighted = np.zeros_like(predictions[0])
            for i, weight in enumerate(self.weights):
                weighted += weight * predictions[i]
            return weighted
        elif self.method == EnsembleMethod.MEDIAN:
            return np.median(predictions, axis=0)
        elif self.method == EnsembleMethod.STACKING:
            if self.meta_model is None:
                # Fallback to average if no meta-model
                return np.mean(predictions, axis=0)
            # Stack predictions and use meta-model
            stacked = predictions.transpose(1, 0, 2).reshape(predictions.shape[1], -1)
            return self.meta_model.predict(stacked)
        else:
            return np.mean(predictions, axis=0)
    
    def predict_single(self, x: np.ndarray) -> float:
        """Predict single value (for single instance)."""
        pred = self.predict(x)
        if len(pred.shape) > 1:
            return float(pred[0, 0] if pred.shape[1] > 0 else pred[0])
        return float(pred[0])


def create_ensemble(
    models: List[Any],
    method: str = "weighted_average",
    weights: Optional[List[float]] = None
) -> ModelEnsemble:
    """Convenience function to create an ensemble."""
    method_enum = EnsembleMethod(method)
    return ModelEnsemble(models, method=method_enum, weights=weights)

