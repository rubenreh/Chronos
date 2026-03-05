"""
Semantic vector embeddings for task and recommendation similarity in Chronos.

This module uses a pre-trained sentence-transformer model (default:
all-MiniLM-L6-v2, 384-dim embeddings) to convert free-text task descriptions
and recommendation strings into dense vectors. Cosine similarity on those
vectors enables:

  • Finding tasks that are semantically related to a query.
  • Matching recommendations to a user's current context.
  • Powering the /recommend API's ability to surface relevant advice.

Key classes / functions:
  EmbeddingGenerator   — stateful wrapper around SentenceTransformer.
  generate_task_embeddings — one-shot convenience function.
"""

import numpy as np                                # Numerical array operations
from typing import List, Dict, Optional           # Type annotations for function signatures
from sentence_transformers import SentenceTransformer  # Pre-trained transformer encoder for text → vector
from sklearn.metrics.pairwise import cosine_similarity  # Efficient pairwise cosine similarity computation


class EmbeddingGenerator:
    """Generate dense vector embeddings for tasks and recommendation strings.

    Internally loads a SentenceTransformer model once and reuses it for all
    encoding calls, amortising the model-load cost across the application
    lifetime.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialise the embedding generator by loading the transformer model.

        Args:
            model_name: HuggingFace model identifier for a sentence-transformer.
                        'all-MiniLM-L6-v2' is a lightweight, fast model that
                        produces 384-dimensional embeddings.
        """
        self.model = SentenceTransformer(model_name)  # Download (if needed) and load the model into memory
        self.model_name = model_name                   # Store for reference / logging

    def encode_tasks(self, task_descriptions: List[str]) -> np.ndarray:
        """Batch-encode a list of task descriptions into embedding vectors.

        Args:
            task_descriptions: Plain-text descriptions of tasks, e.g.
                               ["Write unit tests", "Review PR #42"].

        Returns:
            2-D numpy array of shape (n_tasks, embedding_dim).
        """
        # convert_to_numpy=True ensures we get a numpy array instead of a torch tensor
        return self.model.encode(task_descriptions, convert_to_numpy=True)

    def encode_recommendation(self, recommendation: str) -> np.ndarray:
        """Encode a single recommendation string into its embedding vector.

        Returns:
            1-D numpy array of length embedding_dim.
        """
        # Encode expects a list; we take the first (and only) result
        return self.model.encode([recommendation], convert_to_numpy=True)[0]

    def find_similar_tasks(
        self,
        query_embedding: np.ndarray,
        task_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[int]:
        """Find the `top_k` tasks most similar to a query via cosine similarity.

        This powers the recommendation engine's ability to match a user's current
        context (encoded as `query_embedding`) against a library of known tasks.

        Args:
            query_embedding: 1-D embedding vector for the query.
            task_embeddings: 2-D array of pre-computed task embeddings.
            top_k: How many nearest neighbours to return.

        Returns:
            List of integer indices into `task_embeddings`, sorted by descending
            similarity.
        """
        # Compute cosine similarity between the single query and every task embedding
        similarities = cosine_similarity([query_embedding], task_embeddings)[0]
        # argsort ascending, then reverse for descending; take the first top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices.tolist()  # Convert numpy int64 array to plain Python list


def generate_task_embeddings(
    tasks: List[str],
    model_name: str = 'all-MiniLM-L6-v2'
) -> np.ndarray:
    """One-shot convenience function: create an EmbeddingGenerator and encode tasks.

    Useful for scripts or notebooks where you don't need to keep the generator
    around for repeated calls.

    Args:
        tasks: List of task description strings.
        model_name: Sentence-transformer model identifier.

    Returns:
        2-D numpy array of shape (len(tasks), embedding_dim).
    """
    generator = EmbeddingGenerator(model_name=model_name)  # Instantiate (loads model)
    return generator.encode_tasks(tasks)                    # Encode and return
