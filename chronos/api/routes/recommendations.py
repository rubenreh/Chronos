"""Recommendation endpoints."""
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from datetime import datetime

from chronos.api.schemas import RecommendRequest, RecommendResponse
from chronos.api.cache import get_cache
from chronos.recommendations.engine import RecommendationEngine
from chronos.recommendations.clustering import UserClustering

router = APIRouter(prefix="/recommend", tags=["recommendations"])
CACHE = get_cache()
REC_ENGINE = RecommendationEngine()
USER_CLUSTERING = None  # Will be initialized if user data available


@router.post("", response_model=RecommendResponse)
async def get_recommendations(req: RecommendRequest):
    """Get personalized productivity recommendations."""
    if len(req.window) < 30:
        raise HTTPException(status_code=400, detail="Window too short (minimum 30)")
    
    # Check cache
    cache_key = f"recommend:{req.series_id}:{len(req.window)}:{req.num_recommendations}"
    cached_result = CACHE.get(cache_key)
    if cached_result:
        return RecommendResponse(**cached_result)
    
    # Convert to pandas Series
    timestamps = pd.date_range(
        end=datetime.now(),
        periods=len(req.window),
        freq='1H'
    )
    series = pd.Series(req.window, index=timestamps)
    
    # Generate recommendations
    recommendations = REC_ENGINE.generate_recommendations(
        series=series,
        user_profile=req.user_profile,
        num_recommendations=req.num_recommendations,
        multi_day=req.multi_day
    )
    
    # Get user cluster if clustering is available
    user_cluster = None
    similar_users = None
    
    if USER_CLUSTERING is not None:
        try:
            user_cluster = USER_CLUSTERING.predict(series)
            # Similar users would require access to all user data
            # For now, we'll skip this
        except Exception:
            pass
    
    response = RecommendResponse(
        series_id=req.series_id,
        recommendations=recommendations,
        user_cluster=user_cluster,
        similar_users=similar_users
    )
    
    # Cache result
    CACHE.set(cache_key, response.dict())
    
    return response

