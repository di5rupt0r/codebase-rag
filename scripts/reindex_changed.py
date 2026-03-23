#!/usr/bin/env python3
"""Reindex files changed in the latest Git commit (called by post-commit hook)."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from codebase_rag import config
from codebase_rag.indexer import (
    EmbeddingProvider,
    _resolve_client,
    _get_collection,
    _chunk_by_treesitter,
)


def reindex_files(project_name: str, files: list[str]) -> None:
    """Reindex only the listed files under project_name."""
    code_files = [
        f for f in files if Path(f).suffix.lower() in config.SUPPORTED_EXTENSIONS
    ]
    if not code_files:
        return

    provider = EmbeddingProvider()
    client = _resolve_client()
    collection = _get_collection(client, project_name)

    for path_str in code_files:
        file_path = Path(path_str).resolve()
        if not file_path.exists():
            # File deleted — remove from index
            try:
                collection.delete(where={"path": str(file_path)})
                print(f"Removed deleted file from index: {file_path.name}")
            except Exception:
                pass
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            file_hash = hashlib.sha256(content.encode()).hexdigest()

            # Check if already up-to-date
            existing = collection.get(where={"path": str(file_path)}, include=["metadatas"])
            metas = existing.get("metadatas") or []
            if metas and metas[0] and metas[0].get("file_hash") == file_hash:
                print(f"Skipping {file_path.name} (unchanged)")
                continue

            # Remove stale chunks and re-embed
            collection.delete(where={"path": str(file_path)})
            chunks = _chunk_by_treesitter(content, file_path.suffix)
            if not chunks:
                continue

            docs = [c["text"] for c in chunks]
            chunk_metas = [
                {
                    "path": str(file_path),
                    "line_start": c.get("line_start", 0),
                    "line_end": c.get("line_end", 0),
                    "type": c.get("type", "block"),
                    "name": c.get("name", ""),
                    "file_hash": file_hash,
                }
                for c in chunks
            ]
            
            # encode output handling to ensure 2D float list
            batch_embeddings = provider.encode(docs)
            if not isinstance(batch_embeddings, np.ndarray):
                batch_embeddings = np.array(batch_embeddings)
            if batch_embeddings.ndim == 1:
                batch_embeddings = batch_embeddings.reshape(1, -1)
            embeddings = batch_embeddings.astype("float32").tolist()
            
            ids = [f"{file_path}:{i}" for i in range(len(chunks))]
            collection.add(documents=docs, metadatas=chunk_metas, ids=ids, embeddings=embeddings)
            print(f"Reindexed {file_path.name} — {len(chunks)} chunk(s)")

        except Exception as exc:
            print(f"Failed to reindex {path_str}: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reindex specific files after a Git commit"
    )
    parser.add_argument("--project", required=True, help="Project name (ChromaDB collection)")
    parser.add_argument(
        "--files",
        required=True,
        help="Newline or space-separated list of changed files",
    )
    args = parser.parse_args()

    files = args.files.split() if "\n" not in args.files else args.files.splitlines()
    files = [f.strip() for f in files if f.strip()]
    reindex_files(args.project, files)


if __name__ == "__main__":
    main()
