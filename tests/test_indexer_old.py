from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pytest

from codebase_rag import config


@pytest.fixture
def tmp_codebase(tmp_path: Path) -> Path:
    """Create a small temporary codebase for indexing tests."""
    project = tmp_path / "project"
    project.mkdir()

    (project / "main.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n"
    )

    (project / "utils" ).mkdir()
    (project / "utils" / "math_utils.py").write_text(
        "def mul(a, b):\n"
        "    return a * b\n"
    )

    # Ignored directory
    (project / "node_modules").mkdir()
    (project / "node_modules" / "ignore.js").write_text("console.log('ignore');\n")

    return project


class DummyEmbeddingProvider:
    """Simple deterministic embedding provider for tests."""

    def __init__(self, dimension: int = config.EMBEDDING_DIMENSION) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        # Simple deterministic embeddings based on text length
        embeddings = []
        for text in texts:
            vec = np.zeros(self._dimension, dtype=np.float32)
            vec[0] = len(text)
            embeddings.append(vec)
        return np.stack(embeddings, axis=0)


def test_index_and_search_roundtrip(tmp_codebase: Path, tmp_path: Path) -> None:
    """Index a tiny codebase and perform a basic semantic search."""
    from codebase_rag.indexer import index_codebase, search_codebase, list_indexed_files

    db_path = tmp_path / "chroma_db"
    provider = DummyEmbeddingProvider()

    indexed_count = index_codebase(
        root=tmp_codebase,
        db_path=db_path,
        embedding_provider=provider,
        collection_name="test-codebase",
    )
    assert indexed_count > 0

    files = list_indexed_files(
        db_path=db_path,
        collection_name="test-codebase",
    )
    file_paths = {Path(f["path"]) for f in files}
    # Only supported files, ignoring node_modules
    assert tmp_codebase / "main.py" in file_paths
    assert tmp_codebase / "utils" / "math_utils.py" in file_paths
    assert all("node_modules" not in str(p) for p in file_paths)

    results = search_codebase(
        query="multiply two numbers",
        top_k=3,
        db_path=db_path,
        collection_name="test-codebase",
        embedding_provider=provider,
    )

    assert results, "search should return at least one result"
    # Ensure result schema is stable
    first: Dict[str, Any] = results[0]
    assert first["path"].endswith(".py")
    assert "score" in first
    assert "content" in first
    assert isinstance(first["score"], float)


def test_get_file_content_reads_from_disk(tmp_codebase: Path, tmp_path: Path) -> None:
    """get_file_content should return the full file content from disk."""
    from codebase_rag.indexer import get_file_content

    target = tmp_codebase / "main.py"
    content = get_file_content(str(target))

    assert "def add" in content
    assert "return a + b" in content

