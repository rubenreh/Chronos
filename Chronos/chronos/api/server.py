"""
Enhanced FastAPI server for Chronos.

This is the main entry point for the Chronos REST API. It creates and configures
the FastAPI application, loads trained PyTorch models from disk at startup via an
async lifespan context manager, attaches CORS middleware for cross-origin requests,
and registers all route modules (forecast, patterns, recommendations, explain,
training). When the server starts, models are loaded from .pth checkpoint files
into a global MODELS dictionary that route handlers reference for inference.

Key responsibilities:
    - Application factory: instantiates the FastAPI app with metadata and lifespan
    - Model loading: reads serialized PyTorch models (LSTM, TCN, Transformer) at
      startup, along with their normalization statistics (mu, sigma)
    - Ensemble creation: if multiple model architectures are available, assembles a
      weighted-average ensemble for improved forecast accuracy
    - Router registration: mounts all endpoint routers under their respective prefixes
    - Health check: exposes /health for liveness probes (Docker, K8s, load balancers)
"""
import os                                       # File-path and environment variable access
import asyncio                                  # Async primitives (not directly used here but available for future async work)
from contextlib import asynccontextmanager      # Enables the modern FastAPI lifespan pattern (replaces on_event)
from fastapi import FastAPI, HTTPException, BackgroundTasks  # Core FastAPI framework components
from fastapi.middleware.cors import CORSMiddleware           # Middleware for handling Cross-Origin Resource Sharing
import uvicorn                                  # ASGI server that runs the FastAPI app
import numpy as np                              # Numerical library (available for any array ops at the server level)

# Internal Chronos imports
from chronos.api.schemas import HealthResponse  # Pydantic model that structures the /health JSON response
from chronos.api.cache import get_cache         # Retrieves the singleton in-memory cache instance
from chronos.api.routes import (                # All API route modules packaged from the routes sub-package
    forecast_router,
    patterns_router,
    recommendations_router,
    explain_router,
    training_router
)
from chronos.inference import load_model        # Utility that deserializes a .pth checkpoint into a PyTorch model + stats

# ---------------------------------------------------------------------------
# Global model registry
# ---------------------------------------------------------------------------
# MODELS is a dict that maps model names (e.g. "lstm", "ensemble") to dicts
# containing the PyTorch model object, the training-set mean (mu), and
# standard deviation (sigma) used for z-score normalization of inputs.
MODELS = {}

# MODEL_PATHS maps each supported architecture to its checkpoint file path.
# Paths can be overridden via environment variables so the same Docker image
# can serve different model artifacts without a rebuild.
MODEL_PATHS = {
    'lstm': os.environ.get('CHRONOS_LSTM_MODEL', 'artifacts/lstm_model.pth'),       # Default LSTM checkpoint
    'tcn': os.environ.get('CHRONOS_TCN_MODEL', None),                               # Optional TCN checkpoint
    'transformer': os.environ.get('CHRONOS_TRANSFORMER_MODEL', None),               # Optional Transformer checkpoint
    'ensemble': None  # Dynamically created at startup if multiple models are loaded
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown.

    FastAPI's lifespan pattern replaces the deprecated @app.on_event decorators.
    Code before ``yield`` runs once at startup (model loading), and code after
    ``yield`` runs once when the server is shutting down (cleanup).
    """
    # --- Startup phase ---
    print("Starting Chronos API server...")
    await load_models()  # Deserialize all available model checkpoints into MODELS
    yield
    # --- Shutdown phase ---
    print("Shutting down Chronos API server...")


async def load_models():
    """Load all available PyTorch models from disk into the global MODELS dict.

    For each architecture whose checkpoint file exists on disk, this function
    calls ``load_model`` to deserialize the state dict and retrieve the
    normalization statistics (mu, sigma) that were saved during training.
    If more than one model is loaded, an ensemble wrapper is created
    automatically using a weighted-average strategy. Otherwise the single
    loaded model is aliased as 'ensemble' so forecast code can always
    reference MODELS['ensemble'] without caring about the underlying arch.
    Finally, the populated MODELS dict is injected into the forecast route
    module so that endpoint handlers can access it at request time.
    """
    global MODELS  # Modify the module-level MODELS dict

    # --- Load LSTM model (baseline architecture) ---
    lstm_path = MODEL_PATHS['lstm']                        # Resolve the LSTM checkpoint path
    if os.path.exists(lstm_path):                          # Only attempt load if the file actually exists
        try:
            model, mu, sigma = load_model(lstm_path)       # Deserialize checkpoint → (nn.Module, float, float)
            MODELS['lstm'] = {                             # Store the model and its normalization stats
                'model': model,
                'mu': mu,
                'sigma': sigma
            }
            print(f"✓ Loaded LSTM model from {lstm_path}")
        except Exception as e:
            print(f"✗ Failed to load LSTM model: {e}")     # Log but don't crash — other models may still load

    # --- Create ensemble if multiple architectures were loaded ---
    if len(MODELS) > 1:
        # Import ensemble utilities only when needed (avoids circular imports)
        from chronos.models.ensemble import ModelEnsemble, EnsembleMethod
        models_list = [m['model'] for m in MODELS.values()]  # Collect raw nn.Module objects
        ensemble = ModelEnsemble(                             # Build a weighted-average ensemble
            models_list,
            method=EnsembleMethod.WEIGHTED_AVERAGE
        )
        # Ensemble operates on already-normalized data, so mu=0, sigma=1
        MODELS['ensemble'] = {'model': ensemble, 'mu': 0.0, 'sigma': 1.0}
        print("✓ Created ensemble model")
    elif 'lstm' in MODELS:
        # Only one model available — alias it as 'ensemble' so route code
        # can always look up MODELS['ensemble'] regardless of how many archs loaded
        MODELS['ensemble'] = MODELS['lstm']

    # --- Inject models into the forecast route module ---
    # This is necessary because the forecast module maintains its own MODELS
    # reference at module scope; updating it here keeps everything in sync.
    import chronos.api.routes.forecast as forecast_module
    forecast_module.MODELS = MODELS


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------
# The FastAPI constructor receives metadata shown in the auto-generated
# OpenAPI docs at /docs, plus the lifespan handler that orchestrates startup.
app = FastAPI(
    title="Chronos Productivity Forecasting API",                            # Shown in Swagger UI header
    description="AI-driven productivity forecasting and recommendation system",  # Shown in Swagger UI description
    version="2.0.0",                                                         # Semantic version displayed in docs
    lifespan=lifespan                                                        # Registers the startup/shutdown handler
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
# CORSMiddleware allows the API to be called from web frontends hosted on
# different origins (e.g. a React dashboard on localhost:3000 calling the
# API on localhost:8000). The wildcard "*" settings are permissive and
# suitable for development; in production these should be tightened.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Which origins can call the API (wildcard = any)
    allow_credentials=True,    # Allow cookies / Authorization headers
    allow_methods=["*"],       # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],       # Allow all request headers
)

# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------
# Each router is a self-contained group of related endpoints. include_router
# mounts them onto the main app so their paths become accessible.
app.include_router(forecast_router)          # POST /forecast — time-series predictions
app.include_router(patterns_router)          # POST /patterns — trend & anomaly detection
app.include_router(recommendations_router)   # POST /recommend — actionable suggestions
app.include_router(explain_router)           # POST /explain  — SHAP-based explanations
app.include_router(training_router)          # POST /training/* — background training jobs


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint.

    Returns the server status, which models are currently loaded, and cache
    statistics (size, TTL). Used by Docker HEALTHCHECK, Kubernetes liveness
    probes, and load balancers to verify the service is operational.
    """
    cache = get_cache()                          # Retrieve the singleton cache instance
    return HealthResponse(
        status="healthy",                        # Static string indicating the server is up
        models_loaded=list(MODELS.keys()),       # e.g. ["lstm", "ensemble"]
        cache_stats=cache.get_stats()            # {"size": N, "ttl_seconds": 300}
    )


@app.get("/")
async def root():
    """Root endpoint with API information.

    Returns a JSON directory of all available endpoints so that developers
    hitting the base URL can discover the API surface without reading docs.
    """
    return {
        "name": "Chronos Productivity Forecasting API",
        "version": "2.0.0",
        "endpoints": {
            "forecast": "/forecast",             # Time-series forecasting
            "patterns": "/patterns",             # Pattern & anomaly detection
            "recommendations": "/recommend",     # Personalized recommendations
            "explain": "/explain",               # Model explainability
            "training": "/training",             # Background training management
            "health": "/health",                 # Liveness / readiness probe
            "docs": "/docs"                      # Auto-generated Swagger UI
        }
    }


# ---------------------------------------------------------------------------
# Direct execution entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # When running ``python -m chronos.api.server``, Uvicorn starts the ASGI
    # app with hot-reload enabled for development convenience.
    uvicorn.run(
        'chronos.api.server:app',   # Import path for the FastAPI app object
        host='0.0.0.0',             # Bind to all network interfaces
        port=8000,                  # Default HTTP port for the API
        reload=True                 # Auto-restart on code changes (dev only)
    )
