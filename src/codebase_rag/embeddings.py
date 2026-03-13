"""Embeddings provider wrapper for sentence-transformers and transformers."""

from __future__ import annotations

from typing import List, Union
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from . import config

# Models that require to full transformers backend with manual mean pooling.
# SentenceTransformer wrapping doesn't configure pooling correctly for these.
_TRANSFORMERS_MODELS = {"microsoft/unixcoder-base"}


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Singleton thread-safe embedding model loader.
    
    Carrega o modelo apenas na primeira chamada, subsequentes usam cache.
    Reduz tempo de resposta de 30-60s para <1s.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=1)
def get_transformer_model(model_name: str):
    """Singleton thread-safe transformer model loader."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = AutoModel.from_pretrained(model_name)
    transformer = transformer.to(device)  # Ensure device assignment
    return tokenizer, transformer


class EmbeddingProvider:
    """Wrapper for sentence-transformers and transformers embedding models."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.get_embedding_model()
        # NÃO instanciar modelos no construtor - usar lazy loading
        self._backend = None
        self._model = None
        self._tokenizer = None
        self._transformer = None

    def _ensure_models_loaded(self):
        """Lazy loading dos modelos apenas quando necessário."""
        if self._backend is None:
            if self.model_name in _TRANSFORMERS_MODELS:
                self._backend = "transformers"
                self._tokenizer, self._transformer = get_transformer_model(self.model_name)
            else:
                self._backend = "sentence_transformers"
                self._model = get_embedding_model(self.model_name)

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        """Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts to encode.
            normalize_embeddings: Whether to L2-normalize the output embeddings.

        Returns:
            Numpy array of shape (n_texts, embedding_dim).
        """
        # Lazy loading - carrega modelos apenas na primeira chamada
        self._ensure_models_loaded()
        
        if self._backend == "transformers":
            return self._encode_with_transformers(texts, normalize_embeddings)

        embeddings = self._model.encode(texts, normalize_embeddings=normalize_embeddings)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def _encode_with_transformers(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        encoded = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        attention_mask = encoded["attention_mask"]  # (batch, seq_len)

        self._transformer.eval()
        with torch.no_grad():
            outputs = self._transformer(**encoded)

        # Mean pooling over non-padding tokens
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        sum_embeddings = (last_hidden * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        embeddings = (sum_embeddings / sum_mask).numpy()

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms == 0, 1, norms)

        return embeddings

    @staticmethod
    def get_cache_info():
        """Retorna informações do cache para monitoramento."""
        return {
            "embedding_model_cache": get_embedding_model.cache_info()._asdict(),
            "transformer_model_cache": get_transformer_model.cache_info()._asdict(),
        }
