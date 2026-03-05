"""
Recommendation-engine sub-package for Chronos productivity coaching.

This package analyses behavioural features extracted from a user's time-series
data and generates personalised, actionable recommendations to improve
productivity. It consists of two modules:

  1. engine.py — RecommendationEngine / generate_recommendations
     Detects macro trends (rising, declining, plateauing, recovering), identifies
     bottlenecks (high volatility, low momentum), selects template-based coaching
     messages, and optionally builds multi-day action plans.

  2. clustering.py — UserClustering / cluster_users
     Groups users by their behavioural feature profiles using KMeans or DBSCAN,
     enabling "users like you" recommendations and cohort analysis.

Downstream consumers:
  - FastAPI /recommend endpoint (invokes RecommendationEngine.generate_recommendations)
  - Enhanced Streamlit dashboard Recommendations tab
"""

# Import the public API so consumers can do:
#   from chronos.recommendations import RecommendationEngine, cluster_users
from .engine import RecommendationEngine, generate_recommendations    # noqa: F401 — trend + coaching logic
from .clustering import UserClustering, cluster_users                 # noqa: F401 — user-grouping logic

# Declare public symbols for wildcard imports
__all__ = [
    'RecommendationEngine',
    'generate_recommendations',
    'UserClustering',
    'cluster_users'
]
