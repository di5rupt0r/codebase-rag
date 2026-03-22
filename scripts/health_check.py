#!/usr/bin/env python3
"""Health check script for MCP Codebase RAG server."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from codebase_rag.indexer import list_indexed_files, search_codebase
from codebase_rag.embeddings import EmbeddingProvider


def check_embedding_provider() -> bool:
    """Check if embedding provider is working."""
    try:
        provider = EmbeddingProvider()
        # Test encoding with a simple query
        embeddings = provider.encode("test query")
        return embeddings.shape[0] == 1 and embeddings.shape[1] > 0
    except Exception as e:
        print(f"❌ Embedding provider check failed: {e}")
        return False


def check_chromadb_connection() -> bool:
    """Check if ChromaDB is accessible."""
    try:
        files = list_indexed_files()
        return True  # If we can list files, connection works
    except Exception as e:
        print(f"❌ ChromaDB connection check failed: {e}")
        return False


def check_search_functionality() -> bool:
    """Check if search functionality is working."""
    try:
        # Try a search query
        results = search_codebase(query="test", top_k=1)
        res_list = results.get("results") if isinstance(results, dict) else results
        return isinstance(res_list, list)
    except Exception as e:
        print(f"❌ Search functionality check failed: {e}")
        return False


def check_data_directory() -> bool:
    """Check if data directory exists and is writable."""
    try:
        from codebase_rag.config import get_chroma_db_path
        db_path = get_chroma_db_path()
        
        # Check if parent directory exists
        parent_dir = db_path.parent
        if not parent_dir.exists():
            print(f"❌ Data directory parent does not exist: {parent_dir}")
            return False
        
        # Try to create the data directory if it doesn't exist
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Test write permission by creating a test file
        test_file = db_path / ".health_check"
        test_file.write_text("ok")
        test_file.unlink()
        
        return True
    except Exception as e:
        print(f"❌ Data directory check failed: {e}")
        return False


def main() -> None:
    """Run all health checks."""
    print("🔍 MCP Codebase RAG Server Health Check")
    print("=" * 50)
    
    checks = [
        ("Embedding Provider", check_embedding_provider),
        ("ChromaDB Connection", check_chromadb_connection),
        ("Search Functionality", check_search_functionality),
        ("Data Directory", check_data_directory),
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\nChecking {name}...", end=" ")
        sys.stdout.flush()
        
        start_time = time.time()
        result = check_func()
        duration = time.time() - start_time
        
        if result:
            print(f"✓ OK ({duration:.2f}s)")
            passed += 1
        else:
            print(f"❌ FAILED")
    
    print("\n" + "=" * 50)
    print(f"Health Check Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All systems operational!")
        sys.exit(0)
    else:
        print("⚠️  Some checks failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
