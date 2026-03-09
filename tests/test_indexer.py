"""Tests for indexer module."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from codebase_rag.indexer import (
    IndexStats,
    _chunk_text,
    _iter_source_files,
    index_codebase,
    list_indexed_files,
    search_codebase,
    get_file_content,
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
        
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "test.py").write_text("print('hello world')")
            
            result = index_codebase(tmp_path)
            
            assert result == 1  # 1 chunk indexed
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
        
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "test1.py").write_text("a" * 600)  # Will create 2 chunks
            (tmp_path / "test2.py").write_text("b" * 300)  # Will create 1 chunk
            
            result = index_codebase(tmp_path)
            
            assert result == 3  # 3 chunks total

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
        
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "test.py").write_text("print('hello')")
            
            index_codebase(tmp_path, collection_name="custom-collection")
            
            mock_collection.assert_called_once_with(mock_client.return_value, "custom-collection")


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
        assert call_args[1]["n_results"] == 10

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
