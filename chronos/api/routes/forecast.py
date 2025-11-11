"""Forecasting endpoints."""
import time
import numpy as np
from fastapi import APIRouter, HTTPException
from typing import Optional

from chronos.api.schemas import PredictRequest, PredictResponse
from chronos.api.cache import get_cache
from chronos.inference import load_model, predict
from chronos.models.ensemble import ModelEnsemble, EnsembleMethod

router = APIRouter(prefix="/forecast", tags=["forecasting"])

# Global model storage (will be initialized at startup)
MODELS = {}
CACHE = get_cache()


@router.post("", response_model=PredictResponse)
async def forecast(req: PredictRequest):
    """Generate productivity forecasts.
    
    Supports multiple model types: LSTM, TCN, Transformer, and Ensemble.
    """
    if len(req.window) < 10:
        raise HTTPException(status_code=400, detail="Window too short (minimum 10)")
    
    # Check cache
    cache_key = f"forecast:{req.series_id}:{req.model_type}:{len(req.window)}:{req.horizon}"
    cached_result = CACHE.get(cache_key)
    if cached_result:
        return PredictResponse(**cached_result)
    
    start_time = time.time()
    
    # Get model
    model_key = req.model_type or "ensemble"
    if model_key not in MODELS:
        raise HTTPException(status_code=503, detail=f"Model {model_key} not loaded")
    
    model_data = MODELS[model_key]
    model = model_data['model']
    mu = model_data.get('mu', 0.0)
    sigma = model_data.get('sigma', 1.0)
    
    # Prepare input
    window = np.array(req.window[-60:], dtype=np.float32)
    
    # Generate predictions
    predictions = []
    for _ in range(req.horizon):
        pred = predict(model, mu, sigma, window)
        predictions.append(float(pred))
        # Update window for next prediction (simple approach)
        window = np.roll(window, -1)
        window[-1] = pred
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Generate confidence intervals (simple heuristic)
    std = np.std(req.window)
    confidence_intervals = [
        {'lower': p - 1.96 * std, 'upper': p + 1.96 * std}
        for p in predictions
    ]
    
    response = PredictResponse(
        series_id=req.series_id,
        predictions=predictions,
        model=model_key,
        confidence_intervals=confidence_intervals,
        latency_ms=latency_ms
    )
    
    # Cache result
    CACHE.set(cache_key, response.dict())
    
    return response

