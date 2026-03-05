"""
Explainability Module – SHAP-Based Model Interpretation for Chronos
=====================================================================
This module adds model-agnostic explainability to Chronos using SHAP (SHapley
Additive exPlanations). SHAP assigns each input feature (i.e., each past
timestep in the look-back window) an importance value that quantifies how much
it pushed the model's prediction above or below the baseline (expected) value.

Key concepts:
  • SHAP values – one value per input feature per prediction. Positive values
    mean the feature pushed the prediction higher; negative means lower.
  • KernelExplainer – a model-agnostic SHAP algorithm that works with any
    black-box predict function. It samples perturbations of the input and
    observes how the output changes.
  • Background data – a representative sample of input data used by SHAP to
    approximate the expected model output when features are "missing."

Components:
  • ModelExplainer class   – wraps a trained model and background data,
                             initializes the SHAP KernelExplainer, and provides
                             methods to explain single or batch predictions.
  • explain_prediction()   – convenience function that creates a ModelExplainer
                             and explains a single instance in one call.

Usage:
    explainer = ModelExplainer(model, background_data)
    explanation = explainer.explain_prediction(input_instance, top_features=10)
    print(explanation['feature_importance'])  # e.g. {'feature_59': 0.12, ...}
"""

import numpy as np
from typing import Dict, List, Optional, Any
import warnings

# SHAP is an optional heavy dependency — gracefully degrade if not installed.
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    # Warn at import time so the user knows SHAP features won't work.
    warnings.warn("SHAP not available. Install with: pip install shap")


class ModelExplainer:
    """Explain model predictions using SHAP feature-importance values.

    This class wraps a trained model and a background dataset, initializes a
    SHAP KernelExplainer, and provides methods to explain individual or
    batched predictions.
    """

    def __init__(self, model: Any, background_data: np.ndarray):
        """Initialize the explainer with a model and background data.

        Args:
            model: A trained model with a .predict() method or __call__
                   interface. For PyTorch models, wrap them in a function
                   that accepts numpy input and returns numpy output.
            background_data: A 2-D numpy array (n_samples, n_features) used
                             by SHAP as the reference distribution. SHAP
                             measures feature importance relative to this
                             baseline.

        Raises:
            ImportError: If SHAP is not installed.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for explainability. Install with: pip install shap")

        self.model = model
        self.background_data = background_data
        self.explainer = None  # Will hold the SHAP KernelExplainer.

        # Attempt to initialize the SHAP explainer immediately.
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Create the SHAP KernelExplainer using the background data.

        KernelExplainer is model-agnostic: it only needs a callable that
        maps inputs → predictions. We sample up to 100 background points
        for efficiency (SHAP computation scales with background size).
        """
        try:
            # KernelExplainer requires: (predict_fn, background_data).
            # We use a wrapper method to handle different model interfaces.
            self.explainer = shap.KernelExplainer(
                self._model_predict_wrapper,
                self.background_data[:100]  # Subsample for speed; 100 is a good trade-off.
            )
        except Exception as e:
            # If initialization fails (e.g., incompatible model), log and continue.
            print(f"Warning: Could not initialize SHAP explainer: {e}")
            self.explainer = None

    def _model_predict_wrapper(self, X: np.ndarray) -> np.ndarray:
        """Adapter that makes any model compatible with SHAP's expected interface.

        SHAP calls this function repeatedly with different input perturbations.
        It must accept a 2-D numpy array and return a 1-D array of predictions.

        Args:
            X: Input array of shape (n_perturbations, n_features).

        Returns:
            Predictions array of shape (n_perturbations,).

        Raises:
            ValueError: If the model has neither .predict() nor __call__.
        """
        # Try .predict() first (Keras, sklearn, custom wrappers).
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        # Fall back to __call__ (PyTorch modules wrapped in a lambda).
        elif hasattr(self.model, '__call__'):
            return self.model(X)
        else:
            raise ValueError("Model must have predict method or be callable")

    def explain_prediction(
        self,
        instance: np.ndarray,
        top_features: int = 10
    ) -> Dict[str, Any]:
        """Explain a single prediction by computing SHAP values.

        For a time-series input of length N, this returns N SHAP values — one
        per timestep. Each value indicates how much that timestep contributed
        to pushing the prediction above or below the expected baseline.

        Args:
            instance: 1-D numpy array of shape (n_features,) representing the
                      input to explain (e.g., a single look-back window).
            top_features: Number of most-important features to include in the
                          sorted feature_importance dictionary.

        Returns:
            Dictionary with keys:
                feature_importance – dict of the top_features most impactful
                                     features, sorted by absolute SHAP value.
                shap_values        – full list of SHAP values for every feature.
                prediction         – the model's output for this instance.
                base_value         – the expected (average) model output over the
                                     background data.
            If the explainer is not initialized, returns an error dict.
        """
        # Guard: return a stub result if the explainer failed to initialize.
        if self.explainer is None:
            return {
                'error': 'Explainer not initialized',
                'feature_importance': {},
                'shap_values': None
            }

        try:
            # Reshape the 1-D instance to (1, n_features) for the explainer.
            shap_values = self.explainer.shap_values(instance.reshape(1, -1))

            # shap_values can be a list (multi-output models); take the first.
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Flatten to a 1-D array of length n_features.
            shap_values = shap_values.flatten()

            # Build a dictionary mapping feature indices to their SHAP values.
            # In time-series context, feature_0 is the oldest timestep and
            # feature_{N-1} is the most recent.
            feature_importance = {
                f'feature_{i}': float(shap_values[i])
                for i in range(len(shap_values))
            }

            # Sort features by absolute SHAP value (most impactful first)
            # and keep only the top_features.
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),  # Sort by absolute importance.
                reverse=True              # Highest absolute value first.
            )[:top_features]

            return {
                'feature_importance': dict(sorted_features),
                'shap_values': shap_values.tolist(),
                # Include the raw prediction and the baseline value for context.
                'prediction': float(self._model_predict_wrapper(instance.reshape(1, -1))[0]),
                'base_value': float(self.explainer.expected_value)
            }
        except Exception as e:
            # Return an error dict rather than raising — keeps batch processing robust.
            return {
                'error': str(e),
                'feature_importance': {},
                'shap_values': None
            }

    def explain_batch(
        self,
        instances: np.ndarray,
        top_features: int = 10
    ) -> List[Dict[str, Any]]:
        """Explain multiple predictions by calling explain_prediction on each.

        Args:
            instances: 2-D numpy array of shape (n_instances, n_features).
            top_features: Number of top features to include per explanation.

        Returns:
            List of explanation dictionaries, one per instance.
        """
        explanations = []
        # Iterate over each row (instance) and explain individually.
        for instance in instances:
            explanations.append(self.explain_prediction(instance, top_features))
        return explanations


def explain_prediction(
    model: Any,
    instance: np.ndarray,
    background_data: np.ndarray,
    top_features: int = 10
) -> Dict[str, Any]:
    """Convenience function to explain a single prediction in one call.

    Creates a ModelExplainer internally, so the caller doesn't need to
    manage the explainer lifecycle. Best for one-off explanations; for
    repeated explanations reuse a ModelExplainer instance to avoid
    re-initializing the SHAP KernelExplainer each time.

    Args:
        model: Trained model with .predict() or __call__ interface.
        instance: 1-D numpy array of the input to explain.
        background_data: 2-D numpy array for SHAP's background distribution.
        top_features: Number of most-important features to return.

    Returns:
        Explanation dictionary (see ModelExplainer.explain_prediction).
    """
    # Create a fresh explainer (re-initializes SHAP KernelExplainer).
    explainer = ModelExplainer(model, background_data)
    return explainer.explain_prediction(instance, top_features)
