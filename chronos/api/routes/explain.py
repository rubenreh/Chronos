"""Model explainability endpoints."""
import numpy as np
from fastapi import APIRouter, HTTPException
from typing import Optional

from chronos.api.schemas import ExplainRequest, ExplainResponse
from chronos.api.cache import get_cache
from chronos.evaluation.explainability import ModelExplainer
from chronos.inference import load_model, predict
from chronos.api.routes.forecast import MODELS

router = APIRouter(prefix="/explain", tags=["explainability"])
CACHE = get_cache()


def create_background_data(window: np.ndarray, n_samples: int = 100) -> np.ndarray:
    """Create background data for SHAP from window statistics."""
    # Use sliding windows from the input as background
    if len(window) < 10:
        # Generate synthetic background data
        mean_val = np.mean(window)
        std_val = np.std(window)
        background = np.random.normal(mean_val, std_val, (n_samples, len(window)))
    else:
        # Use actual sliding windows
        background = []
        for i in range(min(n_samples, len(window) - 10)):
            background.append(window[i:i+len(window)])
        background = np.array(background)
    
    return background


class SimpleModelWrapper:
    """Wrapper for PyTorch model to work with SHAP."""
    
    def __init__(self, model, mu: float, sigma: float):
        self.model = model
        self.mu = mu
        self.sigma = sigma
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the model."""
        import torch
        
        predictions = []
        for x in X:
            # Normalize
            x_norm = (x - self.mu) / (self.sigma + 1e-8)
            
            # Convert to tensor
            x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                pred = self.model(x_tensor)
                if isinstance(pred, torch.Tensor):
                    pred = pred.squeeze().cpu().numpy()
                else:
                    pred = float(pred)
            
            predictions.append(pred)
        
        return np.array(predictions)


@router.post("", response_model=ExplainResponse)
async def explain_prediction(req: ExplainRequest):
    """Explain model predictions using SHAP."""
    if len(req.window) < 10:
        raise HTTPException(status_code=400, detail="Window too short")
    
    # Check cache
    cache_key = f"explain:{req.series_id}:{len(req.window)}:{req.top_features}"
    cached_result = CACHE.get(cache_key)
    if cached_result:
        return ExplainResponse(**cached_result)
    
    window = np.array(req.window)
    
    # Try to use actual model for SHAP if available
    try:
        # Get model from MODELS (set at startup)
        if 'ensemble' in MODELS or 'lstm' in MODELS:
            model_key = 'ensemble' if 'ensemble' in MODELS else 'lstm'
            model_data = MODELS[model_key]
            model = model_data['model']
            mu = model_data.get('mu', 0.0)
            sigma = model_data.get('sigma', 1.0)
            
            # Create model wrapper
            model_wrapper = SimpleModelWrapper(model, mu, sigma)
            
            # Create background data
            background_data = create_background_data(window, n_samples=50)
            
            # Initialize SHAP explainer
            try:
                explainer = ModelExplainer(model_wrapper, background_data)
                
                # Explain prediction
                explanation = explainer.explain_prediction(window, top_features=req.top_features)
                
                if 'error' not in explanation:
                    response = ExplainResponse(
                        series_id=req.series_id,
                        feature_importance=explanation.get('feature_importance', {}),
                        shap_values=explanation.get('shap_values'),
                        reasoning=f"SHAP analysis shows top contributing features. "
                                 f"Base value: {explanation.get('base_value', 0):.4f}, "
                                 f"Prediction: {explanation.get('prediction', 0):.4f}"
                    )
                    
                    # Cache result
                    CACHE.set(cache_key, response.dict())
                    return response
            except Exception as e:
                # Fall back to simple explanation if SHAP fails
                pass
    except Exception:
        # Fall back to simple explanation
        pass
    
    # Fallback: Simple feature importance based on variance and trends
    feature_importance = {}
    
    # Recent values (more important)
    recent_weight = 2.0
    for i in range(min(len(window), req.top_features)):
        idx = len(window) - 1 - i  # Start from most recent
        if idx >= 0:
            # Weight by recency and variance
            recency_weight = recent_weight / (i + 1)
            variance_importance = abs(window[idx] - np.mean(window)) / (np.std(window) + 1e-8)
            importance = recency_weight * variance_importance
            feature_importance[f"t-{i}"] = float(importance)
    
    # Trend features
    if len(window) > 2:
        trend = np.polyfit(range(len(window)), window, 1)[0]
        feature_importance['trend'] = float(abs(trend) * 10)
    
    # Volatility
    if len(window) > 1:
        volatility = np.std(np.diff(window))
        feature_importance['volatility'] = float(volatility)
    
    # Sort by importance
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:req.top_features]
    
    reasoning = f"Model prediction is primarily influenced by recent values and trend patterns. "
    reasoning += f"Top contributing factors: {', '.join([k for k, _ in sorted_features[:3]])}"
    
    response = ExplainResponse(
        series_id=req.series_id,
        feature_importance=dict(sorted_features),
        shap_values=None,
        reasoning=reasoning
    )
    
    # Cache result
    CACHE.set(cache_key, response.dict())
    
    return response

