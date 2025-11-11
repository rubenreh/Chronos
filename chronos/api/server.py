"""Enhanced FastAPI server for Chronos."""
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

from chronos.api.schemas import HealthResponse
from chronos.api.cache import get_cache
from chronos.api.routes import forecast_router, patterns_router, recommendations_router, explain_router, training_router
from chronos.inference import load_model

# Model storage
MODELS = {}
MODEL_PATHS = {
    'lstm': os.environ.get('CHRONOS_LSTM_MODEL', 'artifacts/lstm_model.pth'),
    'tcn': os.environ.get('CHRONOS_TCN_MODEL', None),
    'transformer': os.environ.get('CHRONOS_TRANSFORMER_MODEL', None),
    'ensemble': None  # Will be created if multiple models available
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    print("Starting Chronos API server...")
    await load_models()
    yield
    # Shutdown
    print("Shutting down Chronos API server...")


async def load_models():
    """Load models asynchronously."""
    global MODELS
    
    # Load LSTM model (baseline)
    lstm_path = MODEL_PATHS['lstm']
    if os.path.exists(lstm_path):
        try:
            model, mu, sigma = load_model(lstm_path)
            MODELS['lstm'] = {'model': model, 'mu': mu, 'sigma': sigma}
            print(f"✓ Loaded LSTM model from {lstm_path}")
        except Exception as e:
            print(f"✗ Failed to load LSTM model: {e}")
    
    # Create ensemble if multiple models available
    if len(MODELS) > 1:
        from chronos.models.ensemble import ModelEnsemble, EnsembleMethod
        models_list = [m['model'] for m in MODELS.values()]
        ensemble = ModelEnsemble(models_list, method=EnsembleMethod.WEIGHTED_AVERAGE)
        MODELS['ensemble'] = {'model': ensemble, 'mu': 0.0, 'sigma': 1.0}
        print("✓ Created ensemble model")
    elif 'lstm' in MODELS:
        # Use LSTM as default ensemble
        MODELS['ensemble'] = MODELS['lstm']
    
    # Make models available to routers
    import chronos.api.routes.forecast as forecast_module
    forecast_module.MODELS = MODELS


app = FastAPI(
    title="Chronos Productivity Forecasting API",
    description="AI-driven productivity forecasting and recommendation system",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(forecast_router)
app.include_router(patterns_router)
app.include_router(recommendations_router)
app.include_router(explain_router)
app.include_router(training_router)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    cache = get_cache()
    return HealthResponse(
        status="healthy",
        models_loaded=list(MODELS.keys()),
        cache_stats=cache.get_stats()
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Chronos Productivity Forecasting API",
        "version": "2.0.0",
        "endpoints": {
            "forecast": "/forecast",
            "patterns": "/patterns",
            "recommendations": "/recommend",
            "explain": "/explain",
            "training": "/training",
            "health": "/health",
            "docs": "/docs"
        }
    }


if __name__ == '__main__':
    uvicorn.run(
        'chronos.api.server:app',
        host='0.0.0.0',
        port=8000,
        reload=True
    )
