"""Embedding service for text vectorization."""

import time
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings
from utils.helpers import Timer


class EmbeddingService:
    """Service for creating text embeddings."""

    def __init__(self):
        self.model_name = settings.embedding_model
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize embedding model."""
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")

    def create_embedding(
        self,
        text: str
    ) -> Tuple[np.ndarray, float]:
        """
        Create embedding for single text.
        
        Returns:
            Tuple of (embedding, time_taken)
        """
        with Timer() as timer:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        return embedding, timer.elapsed

    def create_embeddings_batch(
        self,
        texts: List[str]
    ) -> Tuple[List[np.ndarray], float]:
        """
        Create embeddings for batch of texts.
        
        Returns:
            Tuple of (embeddings, time_taken)
        """
        with Timer() as timer:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False
            )
        
        return embeddings.tolist(), timer.elapsed

    def get_embedding_dimension(self) -> int:
        """Get dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()