"""Configuration module for Codebase RAG server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


# Default configuration values
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
CHUNK_SIZE_LINES: int = 30   # for line-based fallback chunking
CHUNK_OVERLAP_LINES: int = 5  # line overlap for non-Python files
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
DEFAULT_TOP_K: int = 5
SERVER_NAME: str = "codebase-rag"

IGNORED_PATTERNS: List[str] = [
    "*.pyc",
    "__pycache__",
    ".git",
    "node_modules",
    "vendor",
    "packages",
    "Packages",
    ".venv",
    "venv",
    "env",
    ".env",
    ".tox",
    "build",
    "dist",
    "target",
    "bin",
    "obj",
    "out",
    ".idea",
    ".vscode",
    ".vs",
    "*.egg-info",
    ".pytest_cache",
    ".mypy_cache",
    ".coverage",
    "htmlcov",
]

SUPPORTED_EXTENSIONS: List[str] = [
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".cs",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
]


def get_chroma_db_path(custom_path: Path | None = None) -> Path:
    """Get ChromaDB path, allowing environment override."""
    if custom_path:
        return custom_path
    
    env_path = os.environ.get("CHROMA_DB_PATH")
    if env_path:
        return Path(env_path)
    
    # Default to ./data/chroma_db relative to project root
    return Path(__file__).parent.parent.parent / "data" / "chroma_db"


def get_embedding_model() -> str:
    """Get embedding model name, allowing environment override."""
    env_model = os.environ.get("EMBEDDING_MODEL")
    return env_model or EMBEDDING_MODEL


def get_log_level() -> str:
    """Get log level, allowing environment override."""
    env_level = os.environ.get("LOG_LEVEL", "INFO")
    return env_level


def is_supported_file(file_path: Path) -> bool:
    """Check if file has supported extension."""
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def should_ignore_path(path: Path) -> bool:
    """Check if path should be ignored based on patterns."""
    path_str = str(path)
    
    for pattern in IGNORED_PATTERNS:
        # Simple pattern matching - handles directory names and simple wildcards
        if "*" in pattern:
            # Handle wildcard patterns (e.g., "*.pyc")
            if pattern.endswith("*"):
                # Prefix match (e.g., "__pycache__*")
                if path_str.startswith(pattern.rstrip("*")):
                    return True
            elif pattern.startswith("*"):
                # Suffix match (e.g., "*.pyc")
                if path_str.endswith(pattern.lstrip("*")):
                    return True
        else:
            # Exact match for directory names
            if pattern in path.parts:
                return True
    
    return False
