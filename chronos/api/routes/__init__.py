"""API routes for Chronos."""
from .forecast import router as forecast_router
from .patterns import router as patterns_router
from .recommendations import router as recommendations_router
from .explain import router as explain_router
from .training import router as training_router

__all__ = [
    'forecast_router',
    'patterns_router',
    'recommendations_router',
    'explain_router',
    'training_router'
]

