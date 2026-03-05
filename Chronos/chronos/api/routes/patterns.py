"""
Pattern detection endpoints.

This module implements the POST /patterns endpoint, which analyses a window
of productivity time-series data to detect:

    1. **Trends** — whether the series is rising, declining, plateauing, or recovering
    2. **Anomalies** — individual data points that deviate more than 3 standard
       deviations from the mean (statistical outlier detection)
    3. **Behavioral phases** — higher-level labels (burnout, recovery,
       high_performance, normal) derived from trend direction and strength

The endpoint also extracts a dictionary of statistical features (mean, std,
slope, etc.) via the feature extractor module, which downstream endpoints
like /recommend use to generate personalized recommendations.
"""
import numpy as np                              # Numerical operations for statistics and array math
import pandas as pd                             # Pandas Series/DatetimeIndex used by the feature extractor
from fastapi import APIRouter, HTTPException    # Router for endpoint grouping; HTTPException for error responses
from datetime import datetime, timedelta        # Timestamp generation for synthetic DatetimeIndex

from chronos.api.schemas import PatternRequest, PatternResponse  # Pydantic request/response models
from chronos.api.cache import get_cache                          # Singleton in-memory cache
from chronos.features.extractor import extract_behavioral_features  # Computes rolling statistics & behavioral features
from chronos.recommendations.engine import RecommendationEngine     # Trend detection helper lives here

# Create router under /patterns prefix, grouped as "patterns" in OpenAPI docs
router = APIRouter(prefix="/patterns", tags=["patterns"])

# Module-level references to shared singletons
CACHE = get_cache()              # In-memory cache to avoid recomputing patterns for identical requests
REC_ENGINE = RecommendationEngine()  # Re-used for its detect_trend() utility method


@router.post("", response_model=PatternResponse)
async def detect_patterns(req: PatternRequest):
    """Detect trends, anomalies, and behavioral phases in productivity data.

    Takes a raw time-series window and a rolling window size, then runs
    statistical analysis to surface actionable patterns. Results are cached
    to avoid redundant computation for repeated identical requests.
    """
    # --- Input validation ---
    # Ensure the data window is at least as long as the requested rolling window
    if len(req.window) < req.window_size:
        raise HTTPException(status_code=400, detail="Window too short for requested window_size")

    # --- Cache lookup ---
    cache_key = f"patterns:{req.series_id}:{len(req.window)}:{req.window_size}"
    cached_result = CACHE.get(cache_key)                         # Check for a previously computed result
    if cached_result:
        return PatternResponse(**cached_result)                  # Return cached response immediately

    # --- Convert raw list to a pandas Series with a datetime index ---
    # The feature extractor expects a pd.Series indexed by timestamps.
    # We synthesize hourly timestamps ending at "now" so the extractor
    # can compute time-aware features (e.g. hour-of-day patterns).
    timestamps = pd.date_range(
        end=datetime.now(),                                      # Most recent timestamp = current time
        periods=len(req.window),                                 # One timestamp per data point
        freq='1H'                                                # Hourly frequency assumption
    )
    series = pd.Series(req.window, index=timestamps)             # Wrap the raw values as a time-indexed Series

    # --- Feature extraction ---
    # Computes rolling statistics (mean, std, slope, etc.) over the specified window size
    features = extract_behavioral_features(series, window_size=req.window_size)

    # --- Trend detection ---
    # Delegates to the recommendation engine's detect_trend method, which
    # classifies the series as rising / declining / plateauing / recovering
    trend = REC_ENGINE.detect_trend(series, window_size=req.window_size)

    # --- Trend strength calculation ---
    # Compare the mean of the most recent window to the preceding window.
    # The normalized absolute change gives a 0–1 strength score.
    if len(req.window) >= req.window_size * 2:
        recent = np.array(req.window[-req.window_size:])         # Most recent window_size values
        earlier = np.array(req.window[-req.window_size*2:-req.window_size])  # Previous window_size values
        change = (np.mean(recent) - np.mean(earlier)) / (np.mean(earlier) + 1e-8)  # Relative change (epsilon avoids /0)
        trend_strength = min(1.0, abs(change))                   # Clamp to [0, 1]
    else:
        trend_strength = 0.5                                     # Default mid-range strength when data is too short

    # --- Anomaly detection (3-sigma threshold) ---
    # A data point is flagged as anomalous if it deviates from the mean by
    # more than 3 standard deviations — a classical statistical outlier test.
    anomalies = []
    mean_val = np.mean(req.window)                               # Global mean of the input window
    std_val = np.std(req.window)                                 # Global standard deviation
    threshold = mean_val + 3 * std_val                           # 3-sigma threshold for anomaly detection

    for i, val in enumerate(req.window):                         # Iterate over every data point
        if abs(val - mean_val) > threshold:                      # Check if deviation exceeds 3σ
            anomalies.append({
                'index': i,                                      # Position in the window
                'value': float(val),                             # Actual observed value
                'severity': 'high' if abs(val - mean_val) > 4 * std_val else 'medium'  # 4σ = high, 3σ = medium
            })

    # --- Behavioral phase classification ---
    # Maps trend + strength into a human-readable productivity phase
    behavioral_phase = None
    if trend == 'declining' and trend_strength > 0.2:
        behavioral_phase = 'burnout'                             # Declining productivity above threshold → burnout
    elif trend == 'recovering':
        behavioral_phase = 'recovery'                            # Any recovering trend → recovery phase
    elif trend == 'rising' and trend_strength > 0.15:
        behavioral_phase = 'high_performance'                    # Rising productivity above threshold → high performance
    else:
        behavioral_phase = 'normal'                              # Everything else → stable / normal

    # --- Build and cache response ---
    response = PatternResponse(
        series_id=req.series_id,                                 # Echo back the series identifier
        trend=trend,                                             # Detected trend direction
        trend_strength=float(trend_strength),                    # Numeric strength score [0, 1]
        features=features,                                       # Dict of statistical features
        anomalies=anomalies[:10],                                # Limit to top 10 anomalies to keep response compact
        behavioral_phase=behavioral_phase                        # High-level productivity phase label
    )

    CACHE.set(cache_key, response.dict())                        # Store result in cache for future identical requests

    return response
