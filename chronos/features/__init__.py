"""Feature engineering for productivity time-series."""
from .extractor import FeatureExtractor, extract_behavioral_features
from .embeddings import EmbeddingGenerator, generate_task_embeddings

__all__ = [
    'FeatureExtractor',
    'extract_behavioral_features',
    'EmbeddingGenerator',
    'generate_task_embeddings'
]

