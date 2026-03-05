"""
Forecasting endpoints.

This module implements the core POST /forecast endpoint — the primary reason
the Chronos API exists. A client sends a window of recent time-series values
and a forecast horizon; the endpoint runs the requested PyTorch model in an
autoregressive loop (predict one step → append to window → repeat) and returns
the predicted values along with 95 % confidence intervals.

Inference pipeline:
    1. Validate that the input window is long enough (≥ 10 data points)
    2. Check the in-memory cache for a previous identical request
    3. Look up the requested model in the global MODELS dict (populated at startup)
    4. Truncate the window to the last 60 values (model's expected input length)
    5. Autoregressive loop: for each horizon step, call predict() which internally
       normalizes the window with (mu, sigma), runs a forward pass, and
       denormalizes the output back to the original scale
    6. Compute heuristic 95 % confidence intervals using ±1.96 × σ of the window
    7. Cache and return the response
"""
import time                                     # Wall-clock timing for latency measurement
import numpy as np                              # Array operations for window manipulation
from fastapi import APIRouter, HTTPException    # Router groups endpoints; HTTPException for error responses
from typing import Optional                     # Type hint for nullable model_type

from chronos.api.schemas import PredictRequest, PredictResponse  # Pydantic request/response models
from chronos.api.cache import get_cache                          # Singleton in-memory cache
from chronos.inference import load_model, predict                # Model loader and single-step predict function
from chronos.models.ensemble import ModelEnsemble, EnsembleMethod  # Ensemble wrapper (used if multiple models loaded)

# Create a router with the /forecast URL prefix and "forecasting" tag for OpenAPI grouping
router = APIRouter(prefix="/forecast", tags=["forecasting"])

# MODELS is populated at startup by server.py's load_models() function.
# It maps model names → {"model": nn.Module, "mu": float, "sigma": float}.
MODELS = {}

# Module-level reference to the global cache for fast lookups
CACHE = get_cache()


@router.post("", response_model=PredictResponse)
async def forecast(req: PredictRequest):
    """Generate productivity forecasts using a trained PyTorch model.

    Supports multiple model types: LSTM, TCN, Transformer, and Ensemble.
    Uses an autoregressive strategy: each predicted value is appended to
    the input window before predicting the next step, allowing the model
    to condition on its own previous outputs for multi-step forecasting.
    """
    # --- Input validation ---
    if len(req.window) < 10:                                     # Minimum context length for meaningful prediction
        raise HTTPException(status_code=400, detail="Window too short (minimum 10)")

    # --- Cache lookup ---
    # Build a composite key from request parameters so identical requests hit the cache
    cache_key = f"forecast:{req.series_id}:{req.model_type}:{len(req.window)}:{req.horizon}"
    cached_result = CACHE.get(cache_key)                         # Returns None on miss
    if cached_result:
        return PredictResponse(**cached_result)                  # Deserialize cached dict back into Pydantic model

    start_time = time.time()                                     # Start the latency stopwatch

    # --- Model lookup ---
    model_key = req.model_type or "ensemble"                     # Default to ensemble if client sends None
    if model_key not in MODELS:                                  # Model not loaded at startup
        raise HTTPException(status_code=503, detail=f"Model {model_key} not loaded")

    model_data = MODELS[model_key]                               # Retrieve model + normalization stats
    model = model_data['model']                                  # The PyTorch nn.Module (or ensemble wrapper)
    mu = model_data.get('mu', 0.0)                               # Training-set mean for z-score normalization
    sigma = model_data.get('sigma', 1.0)                         # Training-set std for z-score normalization

    # --- Prepare input window ---
    # Take at most the last 60 values — this matches the model's expected
    # sequence length and prevents unnecessarily large tensors.
    window = np.array(req.window[-60:], dtype=np.float32)

    # --- Autoregressive forecasting loop ---
    # For each step in the horizon, predict one value, then shift the window
    # forward by dropping the oldest value and appending the new prediction.
    predictions = []
    for _ in range(req.horizon):
        pred = predict(model, mu, sigma, window)                 # Single-step inference (normalize → forward → denorm)
        predictions.append(float(pred))                          # Convert numpy scalar to Python float for JSON
        window = np.roll(window, -1)                             # Shift all elements left by 1 (oldest drops off)
        window[-1] = pred                                        # Place the new prediction at the end of the window

    # --- Latency calculation ---
    latency_ms = (time.time() - start_time) * 1000               # Convert seconds → milliseconds

    # --- Confidence intervals ---
    # Heuristic: use ±1.96 standard deviations of the original window to form
    # approximate 95 % confidence bands. A more rigorous approach would use
    # Monte Carlo dropout or quantile regression.
    std = np.std(req.window)                                     # Standard deviation of the original input data
    confidence_intervals = [
        {'lower': p - 1.96 * std, 'upper': p + 1.96 * std}      # 95 % CI for each predicted value
        for p in predictions
    ]

    # --- Build response ---
    response = PredictResponse(
        series_id=req.series_id,                                 # Echo back the series identifier
        predictions=predictions,                                 # List of forecasted float values
        model=model_key,                                         # Which model produced these predictions
        confidence_intervals=confidence_intervals,               # 95 % CI for each step
        latency_ms=latency_ms                                    # How long inference took (for monitoring)
    )

    # --- Cache the result for future identical requests ---
    CACHE.set(cache_key, response.dict())                        # Store as a plain dict (JSON-serializable)

    return response
