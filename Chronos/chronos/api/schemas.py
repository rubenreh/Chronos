"""
Pydantic schemas for Chronos API.

This module defines all request and response models used by the FastAPI endpoints.
Pydantic's BaseModel provides automatic JSON parsing, type coercion, and validation
so that every incoming request is guaranteed to match the expected shape before any
business logic runs. FastAPI also uses these schemas to auto-generate the OpenAPI
(Swagger) documentation visible at /docs.

Schema overview:
    - PredictRequest / PredictResponse   — forecasting (POST /forecast)
    - PatternRequest / PatternResponse   — pattern detection (POST /patterns)
    - RecommendRequest / RecommendResponse — recommendations (POST /recommend)
    - ExplainRequest / ExplainResponse   — model explainability (POST /explain)
    - HealthResponse                     — health check (GET /health)
"""
from pydantic import BaseModel, Field           # BaseModel = schema base class; Field = per-field metadata & validation
from typing import List, Optional, Dict, Any    # Generic type hints for collections and nullable fields
from datetime import datetime                   # Available for timestamp fields in future extensions


# ---------------------------------------------------------------------------
# Forecasting schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for the POST /forecast endpoint.

    The client sends a recent window of time-series values and asks the API
    to forecast ``horizon`` steps into the future using the specified model.
    """
    # Unique identifier for the time series (e.g. a user ID or sensor ID)
    series_id: str = Field(..., description="Time series identifier")

    # The most recent observed values, ordered oldest → newest.
    # The model uses this window as context to autoregressively forecast ahead.
    window: List[float] = Field(..., description="Most recent values, ordered oldest->newest")

    # How many future time steps to predict (1–30). Defaults to 1-step-ahead.
    horizon: int = Field(1, ge=1, le=30, description="Number of steps to forecast")

    # Which model architecture to use. "ensemble" averages all loaded models.
    model_type: Optional[str] = Field("ensemble", description="Model type: lstm, tcn, transformer, ensemble")


class PredictResponse(BaseModel):
    """Response body returned by POST /forecast.

    Contains the predicted values, the name of the model that produced them,
    optional 95 % confidence intervals, and the wall-clock inference latency.
    """
    series_id: str                                                   # Echoed back so the client can match request ↔ response
    predictions: List[float]                                         # Forecasted values for each step in the horizon
    model: str                                                       # Which model was used (e.g. "lstm", "ensemble")
    confidence_intervals: Optional[List[Dict[str, float]]] = None    # List of {"lower": …, "upper": …} per step
    latency_ms: float                                                # Inference time in milliseconds for performance tracking


# ---------------------------------------------------------------------------
# Pattern detection schemas
# ---------------------------------------------------------------------------

class PatternRequest(BaseModel):
    """Request body for the POST /patterns endpoint.

    The client sends a time-series window and a rolling-window size; the API
    analyses trends, detects anomalies, and classifies behavioral phases.
    """
    series_id: str                                                               # Identifies which series to analyze
    window: List[float]                                                          # Raw time-series values
    window_size: int = Field(60, ge=10, le=200)  # Rolling window length for trend/feature calculations


class PatternResponse(BaseModel):
    """Response body returned by POST /patterns.

    Describes the detected trend direction and strength, extracted statistical
    features, any anomalous data points, and the inferred behavioral phase.
    """
    series_id: str                                                                                  # Echoed series identifier
    trend: str = Field(..., description="Trend: rising, declining, plateauing, recovering")         # Detected trend category
    trend_strength: float = Field(..., ge=0.0, le=1.0)                                             # 0 = no trend, 1 = very strong
    features: Dict[str, float]                                                                     # Extracted statistical features (mean, std, slope, etc.)
    anomalies: List[Dict[str, Any]]                                                                # List of anomaly dicts with index, value, severity
    behavioral_phase: Optional[str] = None  # Human-readable phase: burnout, recovery, high_performance, normal


# ---------------------------------------------------------------------------
# Recommendation schemas
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """Request body for the POST /recommend endpoint.

    Takes productivity data plus an optional user profile and returns
    personalized, actionable recommendations.
    """
    series_id: str                                                               # Identifies the user/series
    window: List[float]                                                          # Recent productivity values
    user_profile: Optional[Dict[str, Any]] = None                                # Optional metadata (role, preferences) for personalization
    num_recommendations: int = Field(3, ge=1, le=10)                             # How many recommendations to generate
    multi_day: bool = True  # Whether to generate multi-day (longitudinal) recommendations


class RecommendResponse(BaseModel):
    """Response body returned by POST /recommend.

    Delivers a ranked list of recommendations plus optional clustering info.
    """
    series_id: str                                                   # Echoed series identifier
    recommendations: List[Dict[str, Any]]                            # Each dict has title, description, priority, etc.
    user_cluster: Optional[int] = None                               # Cluster ID from user-behavior clustering (if available)
    similar_users: Optional[List[str]] = None                        # IDs of similar users in the same cluster


# ---------------------------------------------------------------------------
# Explainability schemas
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    """Request body for the POST /explain endpoint.

    Asks the API to explain *why* the model made a particular prediction by
    computing SHAP values or a heuristic feature-importance fallback.
    """
    series_id: str                                                   # Identifies the series being explained
    window: List[float]                                              # Input data that the model would forecast from
    prediction: Optional[float] = None                               # Optional specific prediction to explain
    top_features: int = Field(10, ge=1, le=50)                       # How many top features to return


class ExplainResponse(BaseModel):
    """Response body returned by POST /explain.

    Contains feature importance scores, optional raw SHAP values, and a
    natural-language reasoning string summarizing the explanation.
    """
    series_id: str                                                   # Echoed series identifier
    feature_importance: Dict[str, float]                             # Feature name → importance score mapping
    shap_values: Optional[List[float]] = None                        # Raw SHAP values per input feature (if SHAP succeeded)
    reasoning: str                                                   # Human-readable explanation of the prediction drivers


# ---------------------------------------------------------------------------
# Health check schema
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response body for the GET /health endpoint.

    Reports whether the server is operational, which models are loaded, and
    current cache statistics. Used by infrastructure probes and dashboards.
    """
    status: str                                                      # "healthy" when the server is running correctly
    models_loaded: List[str]                                         # Names of models currently in memory (e.g. ["lstm", "ensemble"])
    cache_stats: Optional[Dict[str, Any]] = None                     # Cache size and TTL info
