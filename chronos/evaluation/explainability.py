"""Model explainability using SHAP."""
import numpy as np
from typing import Dict, List, Optional, Any
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")


class ModelExplainer:
    """Explain model predictions using SHAP."""
    
    def __init__(self, model: Any, background_data: np.ndarray):
        """Initialize model explainer.
        
        Args:
            model: Trained model with predict method
            background_data: Background data for SHAP (shape: n_samples, n_features)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for explainability. Install with: pip install shap")
        
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer."""
        try:
            # Use KernelExplainer for general models
            self.explainer = shap.KernelExplainer(
                self._model_predict_wrapper,
                self.background_data[:100]  # Sample for efficiency
            )
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def _model_predict_wrapper(self, X: np.ndarray) -> np.ndarray:
        """Wrapper for model prediction compatible with SHAP."""
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        elif hasattr(self.model, '__call__'):
            return self.model(X)
        else:
            raise ValueError("Model must have predict method or be callable")
    
    def explain_prediction(
        self,
        instance: np.ndarray,
        top_features: int = 10
    ) -> Dict[str, Any]:
        """Explain a single prediction.
        
        Args:
            instance: Input instance to explain (shape: n_features,)
            top_features: Number of top features to return
        
        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            return {
                'error': 'Explainer not initialized',
                'feature_importance': {},
                'shap_values': None
            }
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(instance.reshape(1, -1))
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            shap_values = shap_values.flatten()
            
            # Get feature importance
            feature_importance = {
                f'feature_{i}': float(shap_values[i])
                for i in range(len(shap_values))
            }
            
            # Sort by absolute importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_features]
            
            return {
                'feature_importance': dict(sorted_features),
                'shap_values': shap_values.tolist(),
                'prediction': float(self._model_predict_wrapper(instance.reshape(1, -1))[0]),
                'base_value': float(self.explainer.expected_value)
            }
        except Exception as e:
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
        """Explain multiple predictions.
        
        Args:
            instances: Input instances (shape: n_instances, n_features)
            top_features: Number of top features to return per instance
        
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        for instance in instances:
            explanations.append(self.explain_prediction(instance, top_features))
        return explanations


def explain_prediction(
    model: Any,
    instance: np.ndarray,
    background_data: np.ndarray,
    top_features: int = 10
) -> Dict[str, Any]:
    """Convenience function to explain a single prediction."""
    explainer = ModelExplainer(model, background_data)
    return explainer.explain_prediction(instance, top_features)

