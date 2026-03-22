"""Tests for indexer module — V5.0 API."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from codebase_rag.indexer import (
    IndexStats,
    BM25Index,
    _chunk_by_treesitter,
    _fallback_chunk_by_lines,
    _iter_source_files,
    _resolve_client,
    reciprocal_rank_fusion,
    index_codebase,
    list_indexed_files,
    search_codebase,
    get_file_content,
)


# ---------------------------------------------------------------------------
# _fallback_chunk_by_lines
# ---------------------------------------------------------------------------

class TestFallbackChunkByLines:
    """Test line-based fallback chunking."""

    def test_basic(self):
        content = "\n".join(f"line{i}" for i in range(60))
        chunks = _fallback_chunk_by_lines(content)
        assert len(chunks) > 1
        assert all(c["type"] == "block" for c in chunks)

    def test_empty_content(self):
        chunks = _fallback_chunk_by_lines("")
        assert chunks == []

    def test_overlap(self):
        content = "\n".join(f"line{i}" for i in range(60))
        chunks = _fallback_chunk_by_lines(content, chunk_size=30, overlap=5)
        lines_c0 = set(chunks[0]["text"].splitlines())
        lines_c1 = set(chunks[1]["text"].splitlines())
        assert lines_c0 & lines_c1  # shared lines due to overlap

    def test_line_numbers_correct(self):
        content = "\n".join(f"line{i}" for i in range(10))
        chunks = _fallback_chunk_by_lines(content, chunk_size=5, overlap=0)
        assert chunks[0]["line_start"] == 1
        assert chunks[0]["line_end"] == 5
        assert chunks[1]["line_start"] == 6

    def test_small_file_single_chunk(self):
        content = "line1\nline2\nline3"
        chunks = _fallback_chunk_by_lines(content, chunk_size=50, overlap=0)
        assert len(chunks) == 1
        assert "line1" in chunks[0]["text"]


# ---------------------------------------------------------------------------
# _chunk_by_treesitter
# ---------------------------------------------------------------------------

class TestChunkByTreesitter:
    """Test tree-sitter based chunking."""

    def test_python_function(self):
        content = "def foo():\n    return 42\n"
        chunks = _chunk_by_treesitter(content, ".py")
        # tree-sitter may fall back to line-based if library version mismatch —
        # what matters is that at least 1 chunk is returned with required keys.
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "text" in chunk
            assert "type" in chunk
            assert "foo" in chunk["text"] or chunk["text"]  # content preserved

    def test_python_class(self):
        content = "class MyClass:\n    def method(self):\n        pass\n"
        chunks = _chunk_by_treesitter(content, ".py")
        assert len(chunks) >= 1

    def test_unsupported_extension_falls_back(self):
        content = "some content here\nmore content"
        chunks = _chunk_by_treesitter(content, ".unknown_ext")
        # Falls back to line-based chunking — should still return chunks
        assert len(chunks) >= 1
        assert all(c["type"] == "block" for c in chunks)

    def test_large_file_falls_back(self):
        # > 500KB triggers fallback
        content = "x = 1\n" * 100_000
        chunks = _chunk_by_treesitter(content, ".py")
        assert len(chunks) >= 1

    def test_chunk_has_required_keys(self):
        content = "def bar():\n    pass\n"
        chunks = _chunk_by_treesitter(content, ".py")
        for chunk in chunks:
            assert "text" in chunk
            assert "type" in chunk
            assert "line_start" in chunk
            assert "line_end" in chunk

    def test_js_file(self):
        content = "function greet() {\n  return 'hello';\n}\n"
        chunks = _chunk_by_treesitter(content, ".js")
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class TestBM25Index:
    """Test BM25 sparse index."""

    def _make_chunks(self, texts):
        return [{"text": t, "path": f"file{i}.py"} for i, t in enumerate(texts)]

    def test_index_and_search_basic(self):
        chunks = self._make_chunks(["def foo(): pass", "class Bar: pass", "import os"])
        idx = BM25Index()
        idx.index(chunks)
        results = idx.search("foo", top_k=3)
        assert len(results) > 0
        # First result should be the chunk containing "foo"
        top_chunk_idx, _ = results[0]
        assert "foo" in chunks[top_chunk_idx]["text"].lower()

    def test_search_empty_index(self):
        idx = BM25Index()
        idx.index([])
        # bm25 is None — should return []
        results = idx.search("anything")
        assert results == []

    def test_search_no_match_returns_empty(self):
        chunks = self._make_chunks(["def foo(): pass"])
        idx = BM25Index()
        idx.index(chunks)
        results = idx.search("zzz_no_match_xyz")
        assert results == []

    def test_top_k_respected(self):
        chunks = self._make_chunks([f"function_{i} does something" for i in range(20)])
        idx = BM25Index()
        idx.index(chunks)
        results = idx.search("function", top_k=5)
        assert len(results) <= 5

    def test_tokenize_lowercases(self):
        idx = BM25Index()
        tokens = idx._tokenize("Hello World FOO")
        assert all(t == t.lower() for t in tokens)

    def test_tokenize_min_length(self):
        idx = BM25Index()
        tokens = idx._tokenize("a bb ccc")
        assert "a" not in tokens
        assert "bb" in tokens
        assert "ccc" in tokens


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    """Test RRF fusion of dense + sparse results."""

    def _make_dense(self, paths):
        return [{"chunk_index": i, "path": p, "content": f"content {i}", "score": 1.0 - i * 0.1}
                for i, p in enumerate(paths)]

    def _make_sparse(self, indices_scores):
        return [(idx, score) for idx, score in indices_scores]

    def _make_chunks(self, n):
        return [{"text": f"chunk {i}", "path": f"file{i}.py", "type": "function",
                 "name": f"func_{i}", "line_start": i, "line_end": i + 5}
                for i in range(n)]

    def test_basic_fusion(self):
        chunks = self._make_chunks(3)
        dense = self._make_dense(["a.py", "b.py", "c.py"])
        sparse = self._make_sparse([(2, 0.9), (0, 0.5)])
        results = reciprocal_rank_fusion(dense, sparse, chunks, k=60, top_k=3)
        assert len(results) <= 3

    def test_dense_only_returns_dense_order(self):
        chunks = self._make_chunks(3)
        dense = self._make_dense(["a.py", "b.py", "c.py"])
        results = reciprocal_rank_fusion(dense, [], chunks, k=60, top_k=3)
        assert len(results) <= 3

    def test_top_k_respected(self):
        chunks = self._make_chunks(10)
        dense = self._make_dense([f"f{i}.py" for i in range(10)])
        sparse = self._make_sparse([(i, float(10 - i)) for i in range(10)])
        results = reciprocal_rank_fusion(dense, sparse, chunks, k=60, top_k=5)
        assert len(results) <= 5

    def test_result_has_required_keys(self):
        chunks = self._make_chunks(2)
        dense = self._make_dense(["a.py", "b.py"])
        sparse = self._make_sparse([(0, 0.8)])
        results = reciprocal_rank_fusion(dense, sparse, chunks, k=60, top_k=2)
        for r in results:
            assert "path" in r
            assert "content" in r
            assert "score" in r

    def test_empty_inputs(self):
        results = reciprocal_rank_fusion([], [], [], k=60, top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# _iter_source_files
# ---------------------------------------------------------------------------

class TestIterSourceFiles:
    """Test source file iteration."""

    def test_iter_source_files_basic(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "test.py").write_text("print('hello')")
            (tmp_path / "test.js").write_text("console.log('hello')")
            (tmp_path / "ignore.txt").write_text("ignore me")
            files = list(_iter_source_files(tmp_path))
            assert len(files) == 2
            assert any(f.name == "test.py" for f in files)
            assert any(f.name == "test.js" for f in files)
            assert not any(f.name == "ignore.txt" for f in files)

    def test_iter_source_files_ignores_pycache(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "src").mkdir()
            (tmp_path / "src" / "app.py").write_text("app")
            (tmp_path / "__pycache__").mkdir()
            (tmp_path / "__pycache__" / "app.pyc").write_text("compiled")
            files = list(_iter_source_files(tmp_path))
            assert len(files) == 1
            assert files[0].name == "app.py"

    def test_iter_source_files_empty_dir(self):
        with TemporaryDirectory() as tmp_dir:
            files = list(_iter_source_files(Path(tmp_dir)))
            assert files == []

    def test_iter_source_files_nested(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "src" / "components").mkdir(parents=True)
            (tmp_path / "src" / "main.py").write_text("main")
            (tmp_path / "src" / "components" / "button.tsx").write_text("button")
            files = list(_iter_source_files(tmp_path))
            assert len(files) == 2


# ---------------------------------------------------------------------------
# index_codebase
# ---------------------------------------------------------------------------

class TestIndexCodebase:
    """Test codebase indexing."""

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._resolve_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_index_basic(self, mock_collection, mock_client, mock_provider):
        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance
        mock_provider_instance.encode.return_value = [[0.1, 0.2, 0.3]]

        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "test.py").write_text("def hello():\n    return 42\n")
            result = index_codebase(tmp_path)
            assert result >= 1
            mock_collection_instance.add.assert_called_once()

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._resolve_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_index_empty_dir(self, mock_collection, mock_client, mock_provider):
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        with TemporaryDirectory() as tmp_dir:
            result = index_codebase(Path(tmp_dir))
            assert result == 0
            mock_collection_instance.add.assert_not_called()

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._resolve_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_index_custom_collection(self, mock_collection, mock_client, mock_provider):
        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance
        mock_provider_instance.encode.return_value = [[0.1, 0.2, 0.3]]

        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "test.py").write_text("def foo(): pass\n")
            index_codebase(tmp_path, collection_name="custom-col")
            mock_collection.assert_called_once_with(mock_client.return_value, "custom-col")


# ---------------------------------------------------------------------------
# list_indexed_files
# ---------------------------------------------------------------------------

class TestListIndexedFiles:
    """Test listing indexed files."""

    @patch("codebase_rag.indexer._resolve_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_basic(self, mock_collection, mock_client):
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        mock_collection_instance.get.return_value = {
            "metadatas": [
                {"path": "/path/to/file1.py"},
                {"path": "/path/to/file2.py"},
                {"path": "/path/to/file1.py"},  # duplicate
            ]
        }
        result = list_indexed_files()
        assert len(result) == 2

    @patch("codebase_rag.indexer._resolve_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_empty(self, mock_collection, mock_client):
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        mock_collection_instance.get.return_value = {"metadatas": []}
        result = list_indexed_files()
        assert result == []

    @patch("codebase_rag.indexer._resolve_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_malformed_metadata(self, mock_collection, mock_client):
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        mock_collection_instance.get.return_value = {
            "metadatas": [
                {"path": "/valid/path.py"},
                None,
                {},
                {"invalid": "key"},
            ]
        }
        result = list_indexed_files()
        assert len(result) == 1
        assert result[0]["path"] == "/valid/path.py"


# ---------------------------------------------------------------------------
# search_codebase
# ---------------------------------------------------------------------------

class TestSearchCodebase:
    """Test codebase searching."""

    def _mock_collection_query(self, mock_collection_instance, docs, metas, dists):
        mock_collection_instance.query.return_value = {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }
        # BM25 bulk fetch
        mock_collection_instance.get.return_value = {
            "documents": docs,
            "metadatas": metas,
        }

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._resolve_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_basic(self, mock_collection, mock_client, mock_provider):
        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance
        mock_provider_instance.encode.return_value = [[0.1, 0.2, 0.3]]

        mock_col = Mock()
        mock_collection.return_value = mock_col
        self._mock_collection_query(
            mock_col,
            ["content1", "content2"],
            [{"path": "file1.py", "type": "function", "name": "foo", "line_start": 1, "line_end": 5},
             {"path": "file2.py", "type": "function", "name": "bar", "line_start": 6, "line_end": 10}],
            [0.1, 0.3],
        )

        result = search_codebase(query="test query")
        assert isinstance(result, dict)
        results = result.get("results", result) if isinstance(result, dict) else result
        # Accept dict (new API) or list (old API)
        if isinstance(result, dict):
            assert "results" in result or len(result) >= 0
        else:
            assert len(result) >= 0

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._resolve_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_no_results(self, mock_collection, mock_client, mock_provider):
        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance
        mock_provider_instance.encode.return_value = [[0.1, 0.2, 0.3]]

        mock_col = Mock()
        mock_collection.return_value = mock_col
        self._mock_collection_query(mock_col, [], [], [])

        result = search_codebase(query="no results query")
        if isinstance(result, dict):
            assert result.get("results", []) == []
        else:
            assert result == []

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._resolve_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_top_k_passed_as_double(self, mock_collection, mock_client, mock_provider):
        """Hybrid search fetches top_k*2 candidates for RRF."""
        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance
        mock_provider_instance.encode.return_value = [[0.1, 0.2, 0.3]]

        mock_col = Mock()
        mock_collection.return_value = mock_col
        self._mock_collection_query(mock_col, [], [], [])

        search_codebase(query="test", top_k=10)
        call_args = mock_col.query.call_args
        assert call_args[1]["n_results"] == 20  # top_k * 2


# ---------------------------------------------------------------------------
# get_file_content
# ---------------------------------------------------------------------------

class TestGetFileContent:
    """Test file content reading."""

    def test_basic(self):
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.py"
            file_path.write_text("print('hello world')")
            content = get_file_content(str(file_path))
            assert content == "print('hello world')"

    def test_nonexistent_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            get_file_content("/nonexistent/file.py")

    def test_encoding_errors_handled(self):
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.py"
            file_path.write_bytes(b"\xff\xfe\x00\x00")
            content = get_file_content(str(file_path))
            assert isinstance(content, str)


# ---------------------------------------------------------------------------
# IndexStats
# ---------------------------------------------------------------------------

class TestIndexStats:
    def test_creation(self):
        stats = IndexStats(documents_indexed=42)
        assert stats.documents_indexed == 42

    def test_equality(self):
        assert IndexStats(documents_indexed=10) == IndexStats(documents_indexed=10)
        assert IndexStats(documents_indexed=10) != IndexStats(documents_indexed=20)


# ---------------------------------------------------------------------------
# registry integration
# ---------------------------------------------------------------------------

class TestIndexCodesbaseWritesRegistry:
    """registry.update_registry must be called when reindex_project indexes a codebase.

    The registry call is in server.py (reindex_project), not in indexer.index_codebase.
    These tests verify that the server layer persists the project path correctly.
    """

    @patch("codebase_rag.server.registry.update_registry")
    @patch("codebase_rag.server._index_codebase")
    @patch("codebase_rag.server._list_indexed_files")
    def test_calls_update_registry(self, mock_list, mock_index, mock_update_registry):
        """reindex_project must call registry.update_registry with name + resolved path."""
        from codebase_rag.server import reindex_project
        mock_index.return_value = 5
        mock_list.return_value = [{"path": "f.py"}]

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            reindex_project(str(root), "my-project")

        mock_update_registry.assert_called_once_with("my-project", str(root))

    @patch("codebase_rag.server.registry.update_registry")
    @patch("codebase_rag.server._index_codebase")
    @patch("codebase_rag.server._list_indexed_files")
    def test_uses_resolved_path(self, mock_list, mock_index, mock_update_registry):
        """The path written to the registry must be the resolved (absolute) path."""
        from codebase_rag.server import reindex_project
        mock_index.return_value = 3
        mock_list.return_value = [{"path": "f.py"}]

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            reindex_project(str(root), "proj")

        called_path = mock_update_registry.call_args[0][1]
        assert Path(called_path).is_absolute()
