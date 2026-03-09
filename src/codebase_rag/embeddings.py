"""Embeddings provider wrapper for sentence-transformers and transformers."""

from __future__ import annotations

from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from . import config

# Models that require the full transformers backend with manual mean pooling.
# SentenceTransformer wrapping doesn't configure pooling correctly for these.
_TRANSFORMERS_MODELS = {"microsoft/unixcoder-base"}


class EmbeddingProvider:
    """Wrapper for sentence-transformers and transformers embedding models."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.get_embedding_model()
        if self.model_name in _TRANSFORMERS_MODELS:
            self._backend = "transformers"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._transformer = AutoModel.from_pretrained(self.model_name)
        else:
            self._backend = "sentence_transformers"
            self.model = SentenceTransformer(self.model_name)

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
        if self._backend == "transformers":
            return self._encode_with_transformers(texts, normalize_embeddings)

        embeddings = self.model.encode(texts, normalize_embeddings=normalize_embeddings)
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
        import torch

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
