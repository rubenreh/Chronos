"""
Chronos — AI-Driven Time-Series Forecasting & Productivity Analytics Package.

This is the top-level package for the Chronos project. Chronos is an end-to-end
system that combines deep-learning forecasting (LSTM, TCN, Transformer ensembles)
with behavioral feature extraction, semantic embeddings, and a recommendation engine
to help users understand and improve their productivity patterns.

Architecture overview:
  - chronos.features   : Extracts statistical, trend, seasonality, and volatility
                         features from raw time-series data; generates semantic
                         embeddings via sentence-transformers.
  - chronos.recommendations : Detects trends/bottlenecks and produces personalized
                              coaching recommendations with multi-day action plans.
  - The FastAPI server (not in this subpackage) exposes /predict, /forecast,
    /patterns, /recommend, and /explain endpoints backed by these modules.
  - MLflow is used for experiment tracking; Docker for reproducible deployments.
"""

# Chronos package
