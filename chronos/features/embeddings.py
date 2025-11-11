"""Vector embeddings for task and user similarity."""
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingGenerator:
    """Generate embeddings for tasks and recommendations."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize embedding generator.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def encode_tasks(self, task_descriptions: List[str]) -> np.ndarray:
        """Generate embeddings for task descriptions.
        
        Args:
            task_descriptions: List of task description strings
        
        Returns:
            Array of embeddings (n_tasks, embedding_dim)
        """
        return self.model.encode(task_descriptions, convert_to_numpy=True)
    
    def encode_recommendation(self, recommendation: str) -> np.ndarray:
        """Generate embedding for a single recommendation."""
        return self.model.encode([recommendation], convert_to_numpy=True)[0]
    
    def find_similar_tasks(
        self,
        query_embedding: np.ndarray,
        task_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[int]:
        """Find most similar tasks to query.
        
        Args:
            query_embedding: Query embedding vector
            task_embeddings: Array of task embeddings
            top_k: Number of similar tasks to return
        
        Returns:
            List of indices of most similar tasks
        """
        similarities = cosine_similarity([query_embedding], task_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices.tolist()


def generate_task_embeddings(
    tasks: List[str],
    model_name: str = 'all-MiniLM-L6-v2'
) -> np.ndarray:
    """Convenience function to generate task embeddings."""
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.encode_tasks(tasks)

