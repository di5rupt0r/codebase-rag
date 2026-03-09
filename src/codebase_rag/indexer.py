from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import numpy as np
import chromadb
from chromadb.api.models import Collection

from . import config
from .embeddings import EmbeddingProvider


@dataclass
class IndexStats:
    """Simple statistics about an indexing run."""

    documents_indexed: int


def _iter_source_files(root: Path) -> Iterable[Path]:
    """Yield all supported, non-ignored files under the given root."""
    root = root.resolve()
    for path in root.rglob("*"):
        if path.is_dir():
            if config.should_ignore_path(path):
                # Skip whole ignored directory trees
                dir_path = str(path)
                for sub in list(path.rglob("*")):
                    # Effectively skip; rglob will still walk, but we'll filter
                    pass
                continue
            continue

        if not config.is_supported_file(path):
            continue

        if config.should_ignore_path(path):
            continue

        yield path


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    """Split text into overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    chunks: List[Dict[str, Any]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append({"start": start, "end": end, "text": chunk})
        if end == text_len:
            break
        start = end - overlap if overlap < (end - start) else end

    return chunks


def _get_client(db_path: Optional[Path] = None) -> chromadb.PersistentClient:
    """Create or reuse a Chroma persistent client."""
    if db_path is None:
        db_path = config.get_chroma_db_path()
    else:
        db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def _get_collection(
    client: chromadb.PersistentClient,
    name: str,
) -> Collection:
    return client.get_or_create_collection(name=name)


def index_codebase(
    root: Path,
    db_path: Optional[Path] = None,
    *,
    embedding_provider: Optional[EmbeddingProvider] = None,
    collection_name: str = "codebase-rag",
) -> int:
    """Index all supported files under `root` into ChromaDB.

    Returns number of chunks indexed.
    """
    root = root.resolve()
    provider = embedding_provider or EmbeddingProvider()
    client = _get_client(db_path)
    collection = _get_collection(client, collection_name)

    chunk_size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for file_path in _iter_source_files(root):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        chunks = _chunk_text(text, chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "path": str(file_path),
                    "start": chunk["start"],
                    "end": chunk["end"],
                }
            )
            ids.append(f"{file_path}:{idx}")

    if not documents:
        return 0

    # Compute embeddings in batches to avoid large single calls
    embeddings: List[List[float]] = []
    batch_size = 64
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_embeddings = provider.encode(batch_docs)
        # Ensure 2D numpy array, then convert to list
        if not isinstance(batch_embeddings, np.ndarray):
            batch_embeddings = np.array(batch_embeddings)
        if batch_embeddings.ndim == 1:
            batch_embeddings = batch_embeddings.reshape(1, -1)
        embeddings.extend(batch_embeddings.astype("float32").tolist())

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )

    return len(documents)


def list_indexed_files(
    *,
    db_path: Optional[Path] = None,
    collection_name: str = "codebase-rag",
) -> List[Dict[str, Any]]:
    """Return a list of unique indexed files with basic metadata."""
    client = _get_client(db_path)
    collection = _get_collection(client, collection_name)

    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []

    files: Dict[str, Dict[str, Any]] = {}
    for meta in metadatas:
        if not meta:
            continue
        path = meta.get("path")
        if not path:
            continue
        existing = files.get(path)
        if existing is None:
            files[path] = {"path": path}

    return list(files.values())


def search_codebase(
    *,
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    db_path: Optional[Path] = None,
    collection_name: str = "codebase-rag",
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> List[Dict[str, Any]]:
    """Search indexed codebase and return ranked matches."""
    provider = embedding_provider or EmbeddingProvider()
    client = _get_client(db_path)
    collection = _get_collection(client, collection_name)

    query_embedding = provider.encode([query])
    # Ensure numpy array
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.array(query_embedding)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    result = collection.query(
        query_embeddings=query_embedding.astype("float32").tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    matches: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        path = (meta or {}).get("path", "")
        score = float(1.0 - float(dist))  # convert distance to similarity-ish score
        matches.append(
            {
                "path": path,
                "score": score,
                "content": doc,
            }
        )

    return matches


def get_file_content(path: str) -> str:
    """Read and return the full content of a file from disk."""
    file_path = Path(path)
    return file_path.read_text(encoding="utf-8", errors="ignore")

