"""
API routes package for Chronos.

This __init__.py re-exports all APIRouter instances from the individual route
modules so that server.py can import them from a single location:

    from chronos.api.routes import forecast_router, patterns_router, ...

Each router is a self-contained group of related endpoints:
    - forecast_router        → POST /forecast          (time-series predictions)
    - patterns_router        → POST /patterns          (trend & anomaly detection)
    - recommendations_router → POST /recommend         (actionable suggestions)
    - explain_router         → POST /explain           (SHAP / feature importance)
    - training_router        → POST/GET /training/*    (background training jobs)
"""
from .forecast import router as forecast_router                  # Forecasting endpoints
from .patterns import router as patterns_router                  # Pattern detection endpoints
from .recommendations import router as recommendations_router    # Recommendation endpoints
from .explain import router as explain_router                    # Explainability endpoints
from .training import router as training_router                  # Training management endpoints

# Explicit public API — controls what ``from chronos.api.routes import *`` exports
__all__ = [
    'forecast_router',
    'patterns_router',
    'recommendations_router',
    'explain_router',
    'training_router'
]
