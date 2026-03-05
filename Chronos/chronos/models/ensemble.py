"""
Ensemble models for improved forecasting.

This module implements `ModelEnsemble`, which combines predictions from
multiple trained models (e.g. LSTM, TCN, Transformer) into a single,
more robust forecast. Ensembling reduces variance and often outperforms
any individual model because different architectures capture different
patterns in the data.

Supported combination strategies (via `EnsembleMethod`):
    AVERAGE          – simple arithmetic mean of all model predictions
    WEIGHTED_AVERAGE – weighted sum (weights must sum to 1); allows
                       giving more influence to better-performing models
    MEDIAN           – element-wise median; robust to outlier predictions
    STACKING         – concatenate base-model predictions and feed them
                       into a secondary "meta-model" that learns the
                       optimal combination

Usage:
    ensemble = create_ensemble(
        models=[lstm, tcn, transformer],
        method="weighted_average",
        weights=[0.5, 0.3, 0.2]
    )
    prediction = ensemble.predict(input_window)
"""

import torch                                    # Tensor operations for model inference
import torch.nn as nn                           # Not directly used but available for extensions
import numpy as np                              # Array operations for combining predictions
from typing import List, Dict, Any, Optional    # Type annotations for clarity
from enum import Enum                           # Enumeration for combination strategies


class EnsembleMethod(Enum):
    """
    Enumeration of supported ensemble combination strategies.

    Each value maps to a string that can be stored in configuration files
    or passed through the API, making it easy to select the strategy at
    runtime.
    """
    AVERAGE = "average"                  # Simple mean of all predictions
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted sum (must sum to 1)
    MEDIAN = "median"                    # Element-wise median (outlier-robust)
    STACKING = "stacking"                # Meta-model trained on base predictions


class ModelEnsemble:
    """
    Ensemble of multiple forecasting models.

    Wraps a list of trained PyTorch models and combines their predictions
    according to a chosen strategy.  All models must accept the same input
    shape and produce outputs of the same dimension.
    """

    def __init__(
        self,
        models: List[Any],
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
        weights: Optional[List[float]] = None,
        meta_model: Optional[Any] = None
    ):
        """
        Initialize the ensemble.

        Args:
            models: List of trained PyTorch models. Each must implement a
                    standard forward() accepting (batch, seq_len, features).
            method: The combination strategy to use when merging predictions.
            weights: Optional list of floats for WEIGHTED_AVERAGE.  If None,
                     equal weights are assigned automatically.  Weights are
                     normalised if they don't already sum to 1.
            meta_model: An optional sklearn-style model (with .predict())
                        used only when method is STACKING.
        """
        self.models = models           # Store the list of base forecasting models
        self.method = method           # Store the chosen combination strategy
        self.weights = weights         # Store the per-model weights (may be None)
        self.meta_model = meta_model   # Store the optional stacking meta-model

        # If using weighted average but no weights were supplied, default to
        # equal weights (each model contributes 1/N of the prediction)
        if weights is None and method == EnsembleMethod.WEIGHTED_AVERAGE:
            self.weights = [1.0 / len(models)] * len(models)

        # If weights were supplied but don't sum to 1, normalise them so the
        # weighted average is properly scaled
        if weights and abs(sum(weights) - 1.0) > 1e-6:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate an ensemble prediction by running all base models and
        combining their outputs.

        Args:
            x: Input data as a NumPy array or PyTorch tensor.
               Expected shape: (batch, seq_len, features) or (seq_len, features).

        Returns:
            NumPy array of combined predictions with shape (batch, output_dim).
        """
        predictions = []  # Will collect each model's output

        for model in self.models:
            # Convert NumPy input to a PyTorch float tensor if necessary
            if isinstance(x, np.ndarray):
                x_tensor = torch.tensor(x, dtype=torch.float32)
            else:
                x_tensor = x

            # Set the model to evaluation mode (disables dropout, batchnorm uses running stats)
            model.eval()

            # Disable gradient computation for inference efficiency
            with torch.no_grad():
                # If input is 2-D (seq_len, features), add a batch dimension
                # so the model receives the expected 3-D tensor
                if len(x_tensor.shape) == 2:
                    x_tensor = x_tensor.unsqueeze(0)

                # Run the model's forward pass
                pred = model(x_tensor)

                # Convert the prediction back to a NumPy array for uniform handling
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()

                predictions.append(pred)

        # Stack into (n_models, batch, output_dim) for easy aggregation
        predictions = np.array(predictions)

        # ---- Combine predictions according to the chosen strategy ----

        if self.method == EnsembleMethod.AVERAGE:
            # Simple arithmetic mean across models (axis=0 = model axis)
            return np.mean(predictions, axis=0)

        elif self.method == EnsembleMethod.WEIGHTED_AVERAGE:
            # Weighted sum: multiply each model's output by its weight
            weighted = np.zeros_like(predictions[0])
            for i, weight in enumerate(self.weights):
                weighted += weight * predictions[i]
            return weighted

        elif self.method == EnsembleMethod.MEDIAN:
            # Element-wise median across models; robust to a single model
            # producing an outlier prediction
            return np.median(predictions, axis=0)

        elif self.method == EnsembleMethod.STACKING:
            if self.meta_model is None:
                # Fall back to simple average if no meta-model was provided
                return np.mean(predictions, axis=0)

            # Reshape so each sample's row concatenates all base-model outputs:
            # (n_models, batch, output_dim) → (batch, n_models * output_dim)
            stacked = predictions.transpose(1, 0, 2).reshape(predictions.shape[1], -1)

            # Feed the stacked features into the meta-model for final prediction
            return self.meta_model.predict(stacked)

        else:
            # Unknown method – gracefully default to averaging
            return np.mean(predictions, axis=0)

    def predict_single(self, x: np.ndarray) -> float:
        """
        Convenience method that returns a single scalar prediction.

        Useful when the API needs to return one number (e.g. next-step
        productivity forecast) rather than a batch of arrays.

        Args:
            x: Input data (same as `predict`).

        Returns:
            A single float – the first element of the first batch item.
        """
        pred = self.predict(x)

        # If the result is multi-dimensional, extract the scalar at [0, 0]
        if len(pred.shape) > 1:
            return float(pred[0, 0] if pred.shape[1] > 0 else pred[0])

        # If the result is already 1-D, just take the first element
        return float(pred[0])


def create_ensemble(
    models: List[Any],
    method: str = "weighted_average",
    weights: Optional[List[float]] = None
) -> ModelEnsemble:
    """
    Convenience factory that creates a ModelEnsemble from a string method name.

    This is the preferred entry point used by the FastAPI service layer
    and CLI tools, since they pass the method as a plain string from
    configuration / request parameters.

    Args:
        models: List of trained PyTorch models.
        method: Combination strategy as a string (e.g. "average",
                "weighted_average", "median", "stacking").
        weights: Optional per-model weights for weighted averaging.

    Returns:
        A configured `ModelEnsemble` instance.
    """
    # Convert the string to the corresponding EnsembleMethod enum value
    method_enum = EnsembleMethod(method)

    return ModelEnsemble(models, method=method_enum, weights=weights)
