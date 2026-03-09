"""Embeddings provider wrapper for sentence-transformers."""

from __future__ import annotations

from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config


class EmbeddingProvider:
    """Wrapper for sentence-transformers embedding models."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize embedding provider with specified model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       If None, uses the default from config.
        """
        self.model_name = model_name or config.get_embedding_model()
        self.model = SentenceTransformer(self.model_name)

    def encode(
        self, 
        texts: Union[str, List[str]], 
        normalize_embeddings: bool = False
    ) -> np.ndarray:
        """Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(texts, normalize_embeddings=normalize_embeddings)
        
        # Ensure we always return a numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        # Ensure 2D shape for consistency
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        return embeddings
