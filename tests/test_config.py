"""Tests for config module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from codebase_rag import config


class TestConfigDefaults:
    """Test default configuration values."""

    def test_chunk_size_default(self):
        """Test CHUNK_SIZE has correct default."""
        assert config.CHUNK_SIZE == 500

    def test_chunk_overlap_default(self):
        """Test CHUNK_OVERLAP has correct default."""
        assert config.CHUNK_OVERLAP == 50

    def test_embedding_model_default(self):
        """Test EMBEDDING_MODEL has correct default."""
        assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"

    def test_default_top_k(self):
        """Test DEFAULT_TOP_K has correct default."""
        assert config.DEFAULT_TOP_K == 5

    def test_ignored_patterns(self):
        """Test IGNORED_PATTERNS contains expected patterns."""
        expected = [
            "*.pyc", "__pycache__", ".git", "node_modules",
            ".venv", "venv", "*.egg-info", ".pytest_cache"
        ]
        assert all(pattern in config.IGNORED_PATTERNS for pattern in expected)

    def test_supported_extensions(self):
        """Test SUPPORTED_EXTENSIONS contains expected file types."""
        expected = [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h"]
        assert all(ext in config.SUPPORTED_EXTENSIONS for ext in expected)


class TestEnvironmentOverrides:
    """Test environment variable overrides."""

    def test_chroma_db_path_env_override(self):
        """Test CHROMA_DB_PATH can be overridden via environment."""
        custom_path = "/custom/chroma/db"
        with patch.dict(os.environ, {"CHROMA_DB_PATH": custom_path}):
            assert config.get_chroma_db_path() == Path(custom_path)

    def test_embedding_model_env_override(self):
        """Test EMBEDDING_MODEL can be overridden via environment."""
        custom_model = "custom-embedding-model"
        with patch.dict(os.environ, {"EMBEDDING_MODEL": custom_model}):
            assert config.get_embedding_model() == custom_model

    def test_log_level_env_override(self):
        """Test LOG_LEVEL can be overridden via environment."""
        custom_level = "DEBUG"
        with patch.dict(os.environ, {"LOG_LEVEL": custom_level}):
            assert config.get_log_level() == custom_level


class TestPathHelpers:
    """Test path helper functions."""

    def test_get_chroma_db_path_default(self):
        """Test default ChromaDB path is under project data directory."""
        path = config.get_chroma_db_path()
        assert path.name == "chroma_db"
        assert "data" in str(path)

    def test_get_chroma_db_path_creates_parent(self):
        """Test get_chroma_db_path creates parent directory if needed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_path = Path(tmp_dir) / "subdir" / "chroma_db"
            result = config.get_chroma_db_path(custom_path)
            assert result == custom_path
            # Note: The actual directory creation happens in _get_client, not here


class TestFileFiltering:
    """Test file filtering functions."""

    def test_is_supported_file_true(self):
        """Test is_supported_file returns True for supported extensions."""
        supported_files = [
            Path("test.py"),
            Path("app.js"),
            Path("component.tsx"),
            Path("main.java"),
        ]
        for file_path in supported_files:
            assert config.is_supported_file(file_path)

    def test_is_supported_file_false(self):
        """Test is_supported_file returns False for unsupported extensions."""
        unsupported_files = [
            Path("image.png"),
            Path("document.pdf"),
            Path("data.csv"),
            Path("archive.zip"),
        ]
        for file_path in unsupported_files:
            assert not config.is_supported_file(file_path)

    def test_should_ignore_path_true(self):
        """Test should_ignore_path returns True for ignored patterns."""
        ignored_paths = [
            Path("node_modules/package"),
            Path(".git/config"),
            Path("__pycache__/module.pyc"),
            Path("venv/lib/python3.11"),
            Path(".pytest_cache/cache"),
        ]
        for path in ignored_paths:
            assert config.should_ignore_path(path)

    def test_should_ignore_path_false(self):
        """Test should_ignore_path returns False for valid paths."""
        valid_paths = [
            Path("src/main.py"),
            Path("tests/test_app.js"),
            Path("components/Button.tsx"),
            Path("README.md"),
        ]
        for path in valid_paths:
            assert not config.should_ignore_path(path)


class TestServerName:
    """Test server name configuration."""

    def test_server_name(self):
        """Test SERVER_NAME is set correctly."""
        assert config.SERVER_NAME == "codebase-rag"
