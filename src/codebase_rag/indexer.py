from __future__ import annotations

import ast
import hashlib
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
            continue
        if not config.is_supported_file(path):
            continue
        if config.should_ignore_path(path):
            continue
        yield path


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    """Split text into overlapping character chunks (kept for backward compat)."""
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


def _extract_imports(content: str) -> List[str]:
    """Extract all imported module names from Python source."""
    imports: List[str] = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
    except SyntaxError:
        pass
    return list(set(imports))


def _chunk_by_ast(file_path: Path, content: str) -> List[Dict[str, Any]]:
    """Chunk Python source by function/class definitions using AST.

    Falls back to line-based chunking on parse errors or when no
    top-level definitions exist.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _chunk_by_lines(file_path, content)

    imports = _extract_imports(content)
    lines = content.splitlines()
    file_hash = hashlib.sha256(content.encode()).hexdigest()

    chunks: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        start_line = node.lineno - 1  # 0-indexed
        end_line = getattr(node, "end_lineno", start_line + 20)
        chunk_content = "\n".join(lines[start_line:end_line])

        calls = [
            n.func.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and hasattr(n.func, "id")
        ]

        chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"

        chunks.append(
            {
                "content": chunk_content,
                "type": chunk_type,
                "name": node.name,
                "line_start": node.lineno,
                "line_end": end_line,
                "path": str(file_path),
                "imports": imports,
                "calls": list(set(calls)),
                "docstring": ast.get_docstring(node) or "",
                "file_hash": file_hash,
            }
        )

    if not chunks:
        return _chunk_by_lines(file_path, content)

    return chunks


def _chunk_by_lines(
    file_path: Path,
    content: str,
    size: int = config.CHUNK_SIZE_LINES,
    overlap: int = config.CHUNK_OVERLAP_LINES,
) -> List[Dict[str, Any]]:
    """Chunk any source file by line count with overlap (fallback for non-Python)."""
    lines = content.splitlines()
    file_hash = hashlib.sha256(content.encode()).hexdigest()
    chunks: List[Dict[str, Any]] = []
    step = max(1, size - overlap)

    for i in range(0, len(lines), step):
        end = min(i + size, len(lines))
        chunk_content = "\n".join(lines[i:end])
        chunks.append(
            {
                "content": chunk_content,
                "type": "block",
                "name": "",
                "line_start": i + 1,
                "line_end": end,
                "path": str(file_path),
                "imports": [],
                "calls": [],
                "docstring": "",
                "file_hash": file_hash,
            }
        )
        if end == len(lines):
            break

    return chunks


def chunk_file(file_path: Path, content: str) -> List[Dict[str, Any]]:
    """Dispatch to the best chunking strategy for the given file."""
    if file_path.suffix == ".py":
        return _chunk_by_ast(file_path, content)
    return _chunk_by_lines(file_path, content)


def _get_client(db_path: Optional[Path] = None) -> chromadb.PersistentClient:
    """Create or reuse a Chroma persistent client."""
    if db_path is None:
        db_path = config.get_chroma_db_path()
    db_path = Path(db_path)
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

    Performs incremental indexing: files whose content hash matches the
    existing index entry are skipped, avoiding redundant re-embedding.

    Returns the number of new/updated chunks written.
    """
    root = root.resolve()
    provider = embedding_provider or EmbeddingProvider()
    client = _get_client(db_path)
    collection = _get_collection(client, collection_name)

    # Build a map of the latest known file_hash per path for incremental check.
    existing = collection.get(include=["metadatas"])
    existing_hashes: Dict[str, str] = {}
    for meta in (existing.get("metadatas") or []):
        if meta and meta.get("path") and meta.get("file_hash"):
            existing_hashes[meta["path"]] = meta["file_hash"]

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for file_path in _iter_source_files(root):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        file_hash = hashlib.sha256(content.encode()).hexdigest()
        path_str = str(file_path)

        # Skip files whose content hasn't changed since last index.
        if existing_hashes.get(path_str) == file_hash:
            continue

        # Remove stale chunks so we don't accumulate duplicates.
        try:
            collection.delete(where={"path": path_str})
        except Exception:
            pass

        for idx, chunk in enumerate(chunk_file(file_path, content)):
            documents.append(chunk["content"])
            metadatas.append(
                {
                    "path": path_str,
                    "line_start": chunk["line_start"],
                    "line_end": chunk["line_end"],
                    "type": chunk.get("type", "block"),
                    "name": chunk.get("name", ""),
                    "docstring": chunk.get("docstring", ""),
                    "file_hash": chunk["file_hash"],
                    # ChromaDB metadata values must be scalars; join lists
                    "imports": ",".join(chunk.get("imports", [])),
                    "calls": ",".join(chunk.get("calls", [])),
                }
            )
            ids.append(f"{path_str}:{chunk['line_start']}:{idx}")

    if not documents:
        return 0

    # Compute embeddings in batches to keep memory usage reasonable.
    embeddings: List[List[float]] = []
    batch_size = 64
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_embeddings = provider.encode(batch_docs)
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
        if path not in files:
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
        score = float(1.0 - float(dist))
        matches.append(
            {
                "path": path,
                "score": score,
                "content": doc,
                "line_start": (meta or {}).get("line_start"),
                "line_end": (meta or {}).get("line_end"),
                "type": (meta or {}).get("type", "block"),
                "name": (meta or {}).get("name", ""),
                "docstring": (meta or {}).get("docstring", ""),
            }
        )

    return matches


def get_file_content(path: str) -> str:
    """Read and return the full content of a file from disk."""
    file_path = Path(path)
    return file_path.read_text(encoding="utf-8", errors="ignore")

