"""
Recommendation endpoints.

This module implements the POST /recommend endpoint, which generates
personalized, actionable productivity recommendations based on a user's
time-series data. It leverages:

    - RecommendationEngine: analyses trends and patterns to produce ranked
      suggestions (e.g. "take a break", "try a focus session")
    - UserClustering (optional): groups users with similar behavioral patterns
      so that recommendations can be tailored to a cluster archetype

The endpoint accepts an optional user_profile dict for extra personalization
and supports multi-day recommendation plans.
"""
import numpy as np                              # Numerical operations (available for future extensions)
import pandas as pd                             # Pandas Series with DatetimeIndex for the recommendation engine
from fastapi import APIRouter, HTTPException    # Router for endpoint grouping; HTTPException for validation errors
from datetime import datetime                   # Used to generate synthetic timestamps for the Series index

from chronos.api.schemas import RecommendRequest, RecommendResponse  # Pydantic request/response models
from chronos.api.cache import get_cache                              # Singleton in-memory cache
from chronos.recommendations.engine import RecommendationEngine      # Core recommendation logic
from chronos.recommendations.clustering import UserClustering        # Behavioral clustering (optional)

# Create router under /recommend prefix, grouped as "recommendations" in OpenAPI docs
router = APIRouter(prefix="/recommend", tags=["recommendations"])

# Module-level singletons
CACHE = get_cache()                  # Shared in-memory cache to avoid redundant recommendation generation
REC_ENGINE = RecommendationEngine()  # Stateless engine that analyses a Series and returns recommendations
USER_CLUSTERING = None               # Will be initialized later if a fitted clustering model is available


@router.post("", response_model=RecommendResponse)
async def get_recommendations(req: RecommendRequest):
    """Generate personalized productivity recommendations.

    Analyses the user's recent time-series data, detects patterns and trends,
    and returns a ranked list of actionable recommendations. Optionally
    identifies the user's behavioral cluster for further personalization.
    """
    # --- Input validation ---
    # At least 30 data points are needed for the engine to detect meaningful patterns
    if len(req.window) < 30:
        raise HTTPException(status_code=400, detail="Window too short (minimum 30)")

    # --- Cache lookup ---
    cache_key = f"recommend:{req.series_id}:{len(req.window)}:{req.num_recommendations}"
    cached_result = CACHE.get(cache_key)                         # Attempt to retrieve a cached response
    if cached_result:
        return RecommendResponse(**cached_result)                # Return immediately on cache hit

    # --- Convert raw values to a time-indexed pandas Series ---
    # The recommendation engine expects a pd.Series with a DatetimeIndex so
    # it can reason about temporal patterns (time-of-day, day-of-week, etc.)
    timestamps = pd.date_range(
        end=datetime.now(),                                      # Most recent timestamp = now
        periods=len(req.window),                                 # One timestamp per data point
        freq='1H'                                                # Hourly granularity assumption
    )
    series = pd.Series(req.window, index=timestamps)             # Create the indexed Series

    # --- Generate recommendations ---
    # The engine internally detects trends, extracts features, and maps them
    # to a curated set of recommendation templates ranked by relevance.
    recommendations = REC_ENGINE.generate_recommendations(
        series=series,                                           # The user's productivity time series
        user_profile=req.user_profile,                           # Optional dict with role, preferences, etc.
        num_recommendations=req.num_recommendations,             # How many suggestions to return
        multi_day=req.multi_day                                  # Whether to include multi-day action plans
    )

    # --- Optional: user clustering ---
    # If a fitted UserClustering model is available, predict which behavioral
    # cluster this user belongs to. This enables cluster-specific recommendations
    # and identification of similar users.
    user_cluster = None
    similar_users = None

    if USER_CLUSTERING is not None:                              # Only run if clustering model has been loaded
        try:
            user_cluster = USER_CLUSTERING.predict(series)       # Assign cluster ID based on feature vector
            # Identifying similar_users would require access to the full user database;
            # omitted in this lightweight in-memory implementation
        except Exception:
            pass                                                 # Gracefully degrade — clustering is optional

    # --- Build and cache response ---
    response = RecommendResponse(
        series_id=req.series_id,                                 # Echo the series identifier back to the client
        recommendations=recommendations,                         # List of recommendation dicts
        user_cluster=user_cluster,                               # Cluster ID (int) or None
        similar_users=similar_users                              # List of similar user IDs or None
    )

    CACHE.set(cache_key, response.dict())                        # Cache for future identical requests

    return response
