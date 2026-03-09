"""Tests for indexer module."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from codebase_rag.indexer import (
    IndexStats,
    _chunk_text,
    _iter_source_files,
    _extract_imports,
    _chunk_by_ast,
    _chunk_by_lines,
    chunk_file,
    index_codebase,
    list_indexed_files,
    search_codebase,
    get_file_content,
    _extract_keywords,
    _rerank,
)


class TestChunkText:
    """Test text chunking functionality."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "a" * 100  # 100 characters
        chunks = _chunk_text(text, chunk_size=30, overlap=5)
        
        assert len(chunks) == 4
        assert chunks[0]["start"] == 0
        assert chunks[0]["end"] == 30
        assert chunks[1]["start"] == 25  # 30 - 5 overlap
        assert chunks[1]["end"] == 55
        assert chunks[2]["start"] == 50
        assert chunks[2]["end"] == 80
        assert chunks[3]["start"] == 75
        assert chunks[3]["end"] == 100

    def test_chunk_text_no_overlap(self):
        """Test chunking without overlap."""
        text = "a" * 100
        chunks = _chunk_text(text, chunk_size=30, overlap=0)
        
        assert len(chunks) == 4
        assert chunks[0]["start"] == 0
        assert chunks[0]["end"] == 30
        assert chunks[1]["start"] == 30
        assert chunks[2]["start"] == 60
        assert chunks[3]["start"] == 90

    def test_chunk_text_single_chunk(self):
        """Test text smaller than chunk size."""
        text = "short"
        chunks = _chunk_text(text, chunk_size=10, overlap=2)
        
        assert len(chunks) == 1
        assert chunks[0]["start"] == 0
        assert chunks[0]["end"] == 5
        assert chunks[0]["text"] == "short"

    def test_chunk_text_exact_multiple(self):
        """Test text exactly divisible by chunk size."""
        text = "a" * 60
        chunks = _chunk_text(text, chunk_size=20, overlap=5)
        
        assert len(chunks) == 4  # 4 chunks due to overlap logic
        assert chunks[0]["end"] == 20
        assert chunks[1]["end"] == 35
        assert chunks[2]["end"] == 50
        assert chunks[3]["end"] == 60

    def test_chunk_text_invalid_params(self):
        """Test invalid parameters."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            _chunk_text("test", chunk_size=0, overlap=0)
        
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            _chunk_text("test", chunk_size=10, overlap=-1)

    def test_chunk_text_large_overlap(self):
        """Test overlap larger than chunk size."""
        text = "a" * 50
        chunks = _chunk_text(text, chunk_size=20, overlap=15)
        
        # Should still work, just with smaller effective steps
        assert len(chunks) >= 2


class TestIterSourceFiles:
    """Test source file iteration."""

    def test_iter_source_files_basic(self):
        """Test basic file iteration."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test files
            (tmp_path / "test.py").write_text("print('hello')")
            (tmp_path / "test.js").write_text("console.log('hello')")
            (tmp_path / "ignore.txt").write_text("ignore me")
            
            files = list(_iter_source_files(tmp_path))
            
            assert len(files) == 2
            assert any(f.name == "test.py" for f in files)
            assert any(f.name == "test.js" for f in files)
            assert not any(f.name == "ignore.txt" for f in files)

    def test_iter_source_files_ignored_dirs(self):
        """Test ignoring directories."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create directory structure
            (tmp_path / "src").mkdir()
            (tmp_path / "src" / "app.py").write_text("app")
            (tmp_path / "__pycache__").mkdir()
            (tmp_path / "__pycache__" / "app.pyc").write_text("compiled")
            
            files = list(_iter_source_files(tmp_path))
            
            assert len(files) == 1
            assert files[0].name == "app.py"

    def test_iter_source_files_empty_dir(self):
        """Test empty directory."""
        with TemporaryDirectory() as tmp_dir:
            files = list(_iter_source_files(Path(tmp_dir)))
            assert len(files) == 0

    def test_iter_source_files_nested(self):
        """Test nested directory structure."""
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            (tmp_path / "src" / "components").mkdir(parents=True)
            (tmp_path / "src" / "main.py").write_text("main")
            (tmp_path / "src" / "components" / "button.tsx").write_text("button")
            
            files = list(_iter_source_files(tmp_path))
            
            assert len(files) == 2
            assert any(f.name == "main.py" for f in files)
            assert any(f.name == "button.tsx" for f in files)


class TestIndexCodebase:
    """Test codebase indexing."""

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._get_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_index_codebase_basic(self, mock_collection, mock_client, mock_provider):
        """Test basic indexing functionality."""
        # Setup mocks
        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance
        mock_provider_instance.encode.return_value = [[0.1, 0.2, 0.3]]
        
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        # Incremental check: empty existing index
        mock_collection_instance.get.return_value = {"metadatas": []}
        
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "test.py").write_text("print('hello world')")
            
            result = index_codebase(tmp_path)
            
            assert result >= 1  # at least 1 chunk indexed
            mock_collection_instance.add.assert_called_once()

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._get_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_index_codebase_multiple_files(self, mock_collection, mock_client, mock_provider):
        """Test indexing multiple files."""
        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance
        # Return embeddings for multiple chunks
        mock_provider_instance.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        # Incremental check: empty existing index
        mock_collection_instance.get.return_value = {"metadatas": []}
        
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "test1.py").write_text("a" * 600)  # Will create multiple chunks
            (tmp_path / "test2.py").write_text("b" * 300)  # Will create some chunks
            
            result = index_codebase(tmp_path)
            
            assert result >= 2  # at least 2 chunks total

    def test_index_codebase_empty_dir(self):
        """Test indexing empty directory."""
        with TemporaryDirectory() as tmp_dir:
            result = index_codebase(Path(tmp_dir))
            assert result == 0

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._get_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_index_codebase_custom_collection(self, mock_collection, mock_client, mock_provider):
        """Test indexing with custom collection name."""
        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance
        mock_provider_instance.encode.return_value = [[0.1, 0.2, 0.3]]
        
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        mock_collection_instance.get.return_value = {"metadatas": []}
        
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "test.py").write_text("print('hello')")
            
            index_codebase(tmp_path, collection_name="custom-collection")
            
            mock_collection.assert_called_once_with(mock_client.return_value, "custom-collection")

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._get_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_index_codebase_incremental_skip_unchanged(self, mock_collection, mock_client, mock_provider):
        """Test that unchanged files are skipped during incremental indexing."""
        import hashlib

        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance

        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            content = "print('hello')"
            file_path = tmp_path / "test.py"
            file_path.write_text(content)

            file_hash = hashlib.sha256(content.encode()).hexdigest()
            # Simulate file already indexed with matching hash
            mock_collection_instance.get.return_value = {
                "metadatas": [{"path": str(file_path), "file_hash": file_hash}]
            }

            result = index_codebase(tmp_path)

            # No new chunks since file is unchanged
            assert result == 0
            mock_collection_instance.add.assert_not_called()

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._get_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_index_codebase_incremental_reindex_changed(self, mock_collection, mock_client, mock_provider):
        """Test that changed files are re-indexed and old chunks deleted."""
        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance
        mock_provider_instance.encode.return_value = [[0.1, 0.2, 0.3]]

        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            content = "print('hello')"
            file_path = tmp_path / "test.py"
            file_path.write_text(content)

            # Simulate file indexed with a DIFFERENT hash (file changed)
            mock_collection_instance.get.return_value = {
                "metadatas": [{"path": str(file_path), "file_hash": "old_hash_value"}]
            }

            result = index_codebase(tmp_path)

            # Old chunks deleted and new ones added
            mock_collection_instance.delete.assert_called_once()
            mock_collection_instance.add.assert_called_once()
            assert result >= 1


class TestListIndexedFiles:
    """Test listing indexed files."""

    @patch("codebase_rag.indexer._get_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_list_indexed_files_basic(self, mock_collection, mock_client):
        """Test basic file listing."""
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        mock_collection_instance.get.return_value = {
            "metadatas": [
                {"path": "/path/to/file1.py"},
                {"path": "/path/to/file2.py"},
                {"path": "/path/to/file1.py"},  # Duplicate
            ]
        }
        
        result = list_indexed_files()
        
        assert len(result) == 2  # Duplicates removed
        assert result[0]["path"] == "/path/to/file1.py"
        assert result[1]["path"] == "/path/to/file2.py"

    @patch("codebase_rag.indexer._get_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_list_indexed_files_empty(self, mock_collection, mock_client):
        """Test empty collection."""
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        mock_collection_instance.get.return_value = {"metadatas": []}
        
        result = list_indexed_files()
        
        assert len(result) == 0

    @patch("codebase_rag.indexer._get_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_list_indexed_files_malformed_metadata(self, mock_collection, mock_client):
        """Test handling malformed metadata."""
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        mock_collection_instance.get.return_value = {
            "metadatas": [
                {"path": "/valid/path.py"},
                None,  # Null metadata
                {},    # Empty metadata
                {"invalid": "metadata"},  # Missing path
            ]
        }
        
        result = list_indexed_files()
        
        assert len(result) == 1
        assert result[0]["path"] == "/valid/path.py"


class TestSearchCodebase:
    """Test codebase searching."""

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._get_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_search_codebase_basic(self, mock_collection, mock_client, mock_provider):
        """Test basic search functionality."""
        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance
        mock_provider_instance.encode.return_value = [[0.1, 0.2, 0.3]]
        
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        mock_collection_instance.query.return_value = {
            "documents": [["content1", "content2"]],
            "metadatas": [[{"path": "file1.py"}, {"path": "file2.py"}]],
            "distances": [[0.1, 0.3]]
        }
        
        result = search_codebase(query="test query")
        
        assert len(result) == 2
        assert result[0]["path"] == "file1.py"
        assert result[0]["content"] == "content1"
        assert result[0]["score"] == 0.9  # 1.0 - 0.1
        assert result[1]["path"] == "file2.py"
        assert result[1]["score"] == 0.7  # 1.0 - 0.3

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._get_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_search_codebase_custom_top_k(self, mock_collection, mock_client, mock_provider):
        """Test search with custom top_k."""
        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance
        mock_provider_instance.encode.return_value = [[0.1, 0.2, 0.3]]
        
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        mock_collection_instance.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        search_codebase(query="test", top_k=10)
        
        mock_collection_instance.query.assert_called_once()
        call_args = mock_collection_instance.query.call_args
        # hybrid search fetches top_k*2 candidates for reranking
        assert call_args[1]["n_results"] == 20

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._get_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_search_codebase_no_results(self, mock_collection, mock_client, mock_provider):
        """Test search with no results."""
        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance
        mock_provider_instance.encode.return_value = [[0.1, 0.2, 3.3]]
        
        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        mock_collection_instance.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        result = search_codebase(query="no results query")
        
        assert len(result) == 0


class TestGetFileContent:
    """Test file content reading."""

    def test_get_file_content_basic(self):
        """Test basic file content reading."""
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.py"
            file_path.write_text("print('hello world')")
            
            content = get_file_content(str(file_path))
            
            assert content == "print('hello world')"

    def test_get_file_content_nonexistent(self):
        """Test reading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            get_file_content("/nonexistent/file.py")

    def test_get_file_content_encoding_errors(self):
        """Test handling encoding errors."""
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.py"
            # Write invalid UTF-8 bytes
            file_path.write_bytes(b'\xff\xfe\x00\x00')
            
            # Should not raise, but return empty or partial content
            content = get_file_content(str(file_path))
            assert isinstance(content, str)


class TestIndexStats:
    """Test IndexStats dataclass."""

    def test_index_stats_creation(self):
        """Test IndexStats creation."""
        stats = IndexStats(documents_indexed=42)
        assert stats.documents_indexed == 42

    def test_index_stats_equality(self):
        """Test IndexStats equality."""
        stats1 = IndexStats(documents_indexed=42)
        stats2 = IndexStats(documents_indexed=42)
        stats3 = IndexStats(documents_indexed=24)
        
        assert stats1 == stats2
        assert stats1 != stats3


class TestExtractImports:
    """Test Python import extraction."""

    def test_extract_imports_basic(self):
        content = "import os\nimport sys\n\nprint('hello')"
        imports = _extract_imports(content)
        assert "os" in imports
        assert "sys" in imports

    def test_extract_imports_from_import(self):
        content = "from pathlib import Path\nfrom typing import List, Optional"
        imports = _extract_imports(content)
        assert "pathlib" in imports
        assert "typing" in imports

    def test_extract_imports_mixed(self):
        content = "import os\nfrom pathlib import Path\nimport json"
        imports = _extract_imports(content)
        assert "os" in imports
        assert "pathlib" in imports
        assert "json" in imports

    def test_extract_imports_empty(self):
        imports = _extract_imports("x = 1")
        assert imports == []

    def test_extract_imports_syntax_error(self):
        # Should not raise, just return empty list
        imports = _extract_imports("def broken(:")
        assert imports == []

    def test_extract_imports_deduplicated(self):
        content = "import os\nimport os"
        imports = _extract_imports(content)
        assert imports.count("os") == 1


class TestChunkByAst:
    """Test AST-based Python chunking."""

    def test_chunk_function(self):
        content = "def foo():\n    return 42\n"
        chunks = _chunk_by_ast(Path("test.py"), content)
        assert len(chunks) == 1
        assert chunks[0]["type"] == "function"
        assert chunks[0]["name"] == "foo"
        assert "def foo" in chunks[0]["content"]

    def test_chunk_class(self):
        content = "class MyClass:\n    pass\n"
        chunks = _chunk_by_ast(Path("test.py"), content)
        assert len(chunks) >= 1
        class_chunks = [c for c in chunks if c["type"] == "class"]
        assert len(class_chunks) == 1
        assert class_chunks[0]["name"] == "MyClass"

    def test_chunk_multiple_functions(self):
        content = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        chunks = _chunk_by_ast(Path("test.py"), content)
        names = [c["name"] for c in chunks]
        assert "foo" in names
        assert "bar" in names

    def test_chunk_docstring_extracted(self):
        content = 'def greet():\n    """Say hello."""\n    return "hello"\n'
        chunks = _chunk_by_ast(Path("test.py"), content)
        assert chunks[0]["docstring"] == "Say hello."

    def test_chunk_imports_extracted(self):
        content = "import os\n\ndef foo():\n    pass\n"
        chunks = _chunk_by_ast(Path("test.py"), content)
        assert "os" in chunks[0]["imports"]

    def test_chunk_file_hash_present(self):
        content = "def foo():\n    pass\n"
        chunks = _chunk_by_ast(Path("test.py"), content)
        assert "file_hash" in chunks[0]
        assert len(chunks[0]["file_hash"]) == 64  # SHA256 hex digest

    def test_chunk_line_numbers(self):
        content = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        chunks = _chunk_by_ast(Path("test.py"), content)
        foo_chunk = next(c for c in chunks if c["name"] == "foo")
        bar_chunk = next(c for c in chunks if c["name"] == "bar")
        assert foo_chunk["line_start"] == 1
        assert bar_chunk["line_start"] == 4

    def test_chunk_syntax_error_falls_back_to_lines(self):
        # Invalid Python — must fall back gracefully
        content = "def broken(:"
        chunks = _chunk_by_ast(Path("test.py"), content)
        assert len(chunks) >= 1
        assert all(c["type"] == "block" for c in chunks)

    def test_chunk_no_functions_falls_back_to_lines(self):
        # Valid Python but no function/class definitions
        content = "x = 1\ny = 2\nz = x + y\n"
        chunks = _chunk_by_ast(Path("test.py"), content)
        assert len(chunks) >= 1
        assert all(c["type"] == "block" for c in chunks)

    def test_chunk_calls_extracted(self):
        content = "def foo():\n    bar()\n    baz()\n"
        chunks = _chunk_by_ast(Path("test.py"), content)
        assert "bar" in chunks[0]["calls"] or "baz" in chunks[0]["calls"]


class TestChunkByLines:
    """Test line-based chunking fallback."""

    def test_basic_chunking(self):
        content = "\n".join(f"line{i}" for i in range(60))
        chunks = _chunk_by_lines(Path("test.js"), content)
        assert len(chunks) > 1
        assert all(c["type"] == "block" for c in chunks)

    def test_overlap(self):
        content = "\n".join(f"line{i}" for i in range(60))
        chunks = _chunk_by_lines(Path("test.js"), content, size=30, overlap=5)
        # With overlap, adjacent chunks share lines
        lines_c0 = set(chunks[0]["content"].splitlines())
        lines_c1 = set(chunks[1]["content"].splitlines())
        assert lines_c0 & lines_c1  # some shared lines

    def test_file_hash_present(self):
        content = "line1\nline2\n"
        chunks = _chunk_by_lines(Path("test.js"), content)
        assert "file_hash" in chunks[0]
        assert len(chunks[0]["file_hash"]) == 64

    def test_empty_content(self):
        chunks = _chunk_by_lines(Path("test.js"), "")
        assert chunks == []

    def test_line_numbers_correct(self):
        content = "\n".join(f"line{i}" for i in range(10))
        chunks = _chunk_by_lines(Path("test.js"), content, size=5, overlap=0)
        assert chunks[0]["line_start"] == 1
        assert chunks[0]["line_end"] == 5
        assert chunks[1]["line_start"] == 6

    def test_imports_and_calls_empty_for_non_python(self):
        content = "const x = 1;\nconsole.log(x);\n"
        chunks = _chunk_by_lines(Path("test.js"), content)
        assert chunks[0]["imports"] == []
        assert chunks[0]["calls"] == []


class TestChunkFile:
    """Test chunk_file dispatcher."""

    def test_dispatches_ast_for_py(self):
        content = "def foo():\n    pass\n"
        chunks = chunk_file(Path("test.py"), content)
        # AST chunking returns function-level chunks
        assert any(c["type"] == "function" for c in chunks)

    def test_dispatches_lines_for_js(self):
        content = "\n".join(f"line{i}" for i in range(40))
        chunks = chunk_file(Path("test.js"), content)
        assert all(c["type"] == "block" for c in chunks)

    def test_dispatches_lines_for_ts(self):
        content = "const x = 1;\n"
        chunks = chunk_file(Path("app.ts"), content)
        assert all(c["type"] == "block" for c in chunks)

    def test_dispatches_lines_for_yaml(self):
        content = "key: value\n"
        chunks = chunk_file(Path("config.yaml"), content)
        assert all(c["type"] == "block" for c in chunks)


class TestSearchCodebaseEnrichedMetadata:
    """Test that search returns enriched metadata fields."""

    @patch("codebase_rag.indexer.EmbeddingProvider")
    @patch("codebase_rag.indexer._get_client")
    @patch("codebase_rag.indexer._get_collection")
    def test_search_returns_line_numbers(self, mock_collection, mock_client, mock_provider):
        mock_provider_instance = Mock()
        mock_provider.return_value = mock_provider_instance
        mock_provider_instance.encode.return_value = [[0.1, 0.2, 0.3]]

        mock_collection_instance = Mock()
        mock_collection.return_value = mock_collection_instance
        mock_collection_instance.query.return_value = {
            "documents": [["def foo(): pass"]],
            "metadatas": [[{
                "path": "foo.py",
                "line_start": 10,
                "line_end": 15,
                "type": "function",
                "name": "foo",
                "docstring": "Does foo.",
            }]],
            "distances": [[0.1]],
        }

        results = search_codebase(query="foo function")
        assert results[0]["line_start"] == 10
        assert results[0]["line_end"] == 15
        assert results[0]["type"] == "function"
        assert results[0]["name"] == "foo"
        assert results[0]["docstring"] == "Does foo."


class TestExtractKeywords:
    """Test keyword extraction for hybrid search."""

    def test_basic_words(self):
        keywords = _extract_keywords("search for score function")
        assert "search" in keywords
        assert "score" in keywords
        assert "function" in keywords

    def test_stopwords_removed(self):
        keywords = _extract_keywords("where is the function for parsing")
        assert "the" not in keywords
        assert "is" not in keywords
        assert "for" not in keywords
        assert "where" not in keywords
        assert "parsing" in keywords

    def test_short_words_excluded(self):
        keywords = _extract_keywords("do it at my place")
        # words with 2 or fewer chars are excluded
        assert "do" not in keywords
        assert "it" not in keywords
        assert "at" not in keywords
        assert "my" not in keywords

    def test_empty_query(self):
        keywords = _extract_keywords("")
        assert keywords == []

    def test_underscore_identifiers(self):
        keywords = _extract_keywords("_extract_keywords function")
        assert any("extract" in kw or "_extract_keywords" in kw for kw in keywords)

    def test_numbers_excluded(self):
        # numbers alone should not appear (regex requires letter or underscore start)
        keywords = _extract_keywords("version 42 update")
        assert "42" not in keywords

    def test_case_insensitive(self):
        keywords = _extract_keywords("PARSE Parse parse")
        assert keywords.count("parse") == 1


class TestRerank:
    """Test reranking of search results with keyword heuristics."""

    def _make_match(self, name: str, score: float, docstring: str = "") -> dict:
        return {
            "path": "test.py",
            "score": score,
            "content": f"def {name}(): pass",
            "name": name,
            "docstring": docstring,
            "line_start": 1,
            "line_end": 5,
            "type": "function",
        }

    def test_no_keywords_preserves_order(self):
        matches = [
            self._make_match("foo", 0.9),
            self._make_match("bar", 0.7),
        ]
        result = _rerank(matches, keywords=[])
        assert [m["name"] for m in result] == ["foo", "bar"]

    def test_name_match_boosts_score(self):
        # "bar" match should be boosted over "foo" for keyword "bar"
        matches = [
            self._make_match("foo", 0.9),
            self._make_match("bar", 0.7),
        ]
        result = _rerank(matches, keywords=["bar"])
        assert result[0]["name"] == "bar"

    def test_docstring_match_boosts_score(self):
        matches = [
            self._make_match("foo", 0.9, docstring=""),
            self._make_match("bar", 0.8, docstring="parses input"),
        ]
        result = _rerank(matches, keywords=["parses"])
        assert result[0]["name"] == "bar"

    def test_returns_all_matches(self):
        matches = [self._make_match(f"f{i}", 0.9 - i * 0.1) for i in range(5)]
        result = _rerank(matches, keywords=["test"])
        assert len(result) == 5

    def test_empty_matches(self):
        assert _rerank([], keywords=["test"]) == []
