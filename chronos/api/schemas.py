"""Pydantic schemas for Chronos API."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PredictRequest(BaseModel):
    """Request for forecasting."""
    series_id: str = Field(..., description="Time series identifier")
    window: List[float] = Field(..., description="Most recent values, ordered oldest->newest")
    horizon: int = Field(1, ge=1, le=30, description="Number of steps to forecast")
    model_type: Optional[str] = Field("ensemble", description="Model type: lstm, tcn, transformer, ensemble")


class PredictResponse(BaseModel):
    """Response from forecasting."""
    series_id: str
    predictions: List[float]
    model: str
    confidence_intervals: Optional[List[Dict[str, float]]] = None
    latency_ms: float


class PatternRequest(BaseModel):
    """Request for pattern detection."""
    series_id: str
    window: List[float]
    window_size: int = Field(60, ge=10, le=200)


class PatternResponse(BaseModel):
    """Response from pattern detection."""
    series_id: str
    trend: str = Field(..., description="Trend: rising, declining, plateauing, recovering")
    trend_strength: float = Field(..., ge=0.0, le=1.0)
    features: Dict[str, float]
    anomalies: List[Dict[str, Any]]
    behavioral_phase: Optional[str] = None


class RecommendRequest(BaseModel):
    """Request for recommendations."""
    series_id: str
    window: List[float]
    user_profile: Optional[Dict[str, Any]] = None
    num_recommendations: int = Field(3, ge=1, le=10)
    multi_day: bool = True


class RecommendResponse(BaseModel):
    """Response from recommendation engine."""
    series_id: str
    recommendations: List[Dict[str, Any]]
    user_cluster: Optional[int] = None
    similar_users: Optional[List[str]] = None


class ExplainRequest(BaseModel):
    """Request for model explanation."""
    series_id: str
    window: List[float]
    prediction: Optional[float] = None
    top_features: int = Field(10, ge=1, le=50)


class ExplainResponse(BaseModel):
    """Response from explainability."""
    series_id: str
    feature_importance: Dict[str, float]
    shap_values: Optional[List[float]] = None
    reasoning: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: List[str]
    cache_stats: Optional[Dict[str, Any]] = None
