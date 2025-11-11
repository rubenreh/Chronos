"""Pattern detection endpoints."""
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta

from chronos.api.schemas import PatternRequest, PatternResponse
from chronos.api.cache import get_cache
from chronos.features.extractor import extract_behavioral_features
from chronos.recommendations.engine import RecommendationEngine

router = APIRouter(prefix="/patterns", tags=["patterns"])
CACHE = get_cache()
REC_ENGINE = RecommendationEngine()


@router.post("", response_model=PatternResponse)
async def detect_patterns(req: PatternRequest):
    """Detect patterns and trends in productivity data."""
    if len(req.window) < req.window_size:
        raise HTTPException(status_code=400, detail="Window too short for requested window_size")
    
    # Check cache
    cache_key = f"patterns:{req.series_id}:{len(req.window)}:{req.window_size}"
    cached_result = CACHE.get(cache_key)
    if cached_result:
        return PatternResponse(**cached_result)
    
    # Convert to pandas Series
    timestamps = pd.date_range(
        end=datetime.now(),
        periods=len(req.window),
        freq='1H'
    )
    series = pd.Series(req.window, index=timestamps)
    
    # Extract features
    features = extract_behavioral_features(series, window_size=req.window_size)
    
    # Detect trend
    trend = REC_ENGINE.detect_trend(series, window_size=req.window_size)
    
    # Calculate trend strength
    if len(req.window) >= req.window_size * 2:
        recent = np.array(req.window[-req.window_size:])
        earlier = np.array(req.window[-req.window_size*2:-req.window_size])
        change = (np.mean(recent) - np.mean(earlier)) / (np.mean(earlier) + 1e-8)
        trend_strength = min(1.0, abs(change))
    else:
        trend_strength = 0.5
    
    # Detect anomalies (simple threshold-based)
    anomalies = []
    mean_val = np.mean(req.window)
    std_val = np.std(req.window)
    threshold = mean_val + 3 * std_val
    
    for i, val in enumerate(req.window):
        if abs(val - mean_val) > threshold:
            anomalies.append({
                'index': i,
                'value': float(val),
                'severity': 'high' if abs(val - mean_val) > 4 * std_val else 'medium'
            })
    
    # Detect behavioral phase
    behavioral_phase = None
    if trend == 'declining' and trend_strength > 0.2:
        behavioral_phase = 'burnout'
    elif trend == 'recovering':
        behavioral_phase = 'recovery'
    elif trend == 'rising' and trend_strength > 0.15:
        behavioral_phase = 'high_performance'
    else:
        behavioral_phase = 'normal'
    
    response = PatternResponse(
        series_id=req.series_id,
        trend=trend,
        trend_strength=float(trend_strength),
        features=features,
        anomalies=anomalies[:10],  # Limit to top 10
        behavioral_phase=behavioral_phase
    )
    
    # Cache result
    CACHE.set(cache_key, response.dict())
    
    return response

