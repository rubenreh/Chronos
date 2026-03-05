"""
Feature-engineering sub-package for Chronos productivity time-series.

This package turns raw time-series data into machine-learnable representations
used throughout the Chronos system:

  1. FeatureExtractor / extract_behavioral_features (extractor.py)
     — Computes statistical moments (mean, std, skew, kurtosis), trend parameters
       (slope, R²), dominant FFT frequency, volatility, and autocorrelation from
       a sliding window of productivity values.  These features feed the /patterns
       API endpoint and the recommendation engine.

  2. EmbeddingGenerator / generate_task_embeddings (embeddings.py)
     — Uses a sentence-transformer model (all-MiniLM-L6-v2) to convert task
       descriptions into dense vector embeddings, enabling cosine-similarity
       search for matching tasks and recommendations.

Downstream consumers:
  - chronos.recommendations.engine  (trend detection, bottleneck analysis)
  - chronos.recommendations.clustering (KMeans / DBSCAN user grouping)
  - FastAPI /patterns and /recommend endpoints
"""

# Import the public API of the features sub-package so callers can do:
#   from chronos.features import FeatureExtractor, extract_behavioral_features
from .extractor import FeatureExtractor, extract_behavioral_features  # noqa: F401 — behavioral feature pipeline
from .embeddings import EmbeddingGenerator, generate_task_embeddings  # noqa: F401 — semantic embedding pipeline

# Explicit public API declaration so `from chronos.features import *` only
# exposes the four main symbols.
__all__ = [
    'FeatureExtractor',
    'extract_behavioral_features',
    'EmbeddingGenerator',
    'generate_task_embeddings'
]
