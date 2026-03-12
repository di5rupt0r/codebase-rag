"""Tests for tree-sitter chunking functionality."""

import pytest
from unittest.mock import Mock, patch
from codebase_rag.indexer import _chunk_by_treesitter, _fallback_chunk_by_lines


class TestTreeSitterChunking:
    """Test tree-sitter based chunking."""

    def test_chunk_python_function(self):
        """Test chunking Python function definitions."""
        python_code = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")

class MyClass:
    def method(self):
        pass
'''
        chunks = _chunk_by_treesitter(python_code, ".py")
        
        # Should find at least function or class (may find both or just one)
        assert len(chunks) >= 1
        
        # Check if any function or class chunks found
        func_chunks = [c for c in chunks if c["type"] == "function"]
        class_chunks = [c for c in chunks if c["type"] == "class"]
        
        # At least one should exist
        assert len(func_chunks) >= 1 or len(class_chunks) >= 1
        
        # Check names if found
        if func_chunks:
            assert any("hello_world" in c.get("name", "") for c in func_chunks)
        if class_chunks:
            assert any("MyClass" in c.get("name", "") for c in class_chunks)

    def test_chunk_javascript_function(self):
        """Test chunking JavaScript function definitions."""
        js_code = '''
function helloWorld() {
    console.log("Hello, World!");
}

class MyClass {
    method() {
        return true;
    }
}
'''
        chunks = _chunk_by_treesitter(js_code, ".js")
        
        # Should find at least function or class
        assert len(chunks) >= 1
        
        # Check if any function or class chunks found
        func_chunks = [c for c in chunks if c["type"] == "function"]
        class_chunks = [c for c in chunks if c["type"] == "class"]
        
        # At least one should exist
        assert len(func_chunks) >= 1 or len(class_chunks) >= 1

    def test_chunk_unsupported_extension(self):
        """Test fallback to line-based for unsupported extensions."""
        content = "some random content\nwith multiple lines"
        chunks = _chunk_by_treesitter(content, ".unknown")
        
        # Should fallback to line-based chunking
        assert len(chunks) > 0
        assert all(c["type"] == "block" for c in chunks)

    @patch('codebase_rag.indexer.Language')
    @patch('codebase_rag.indexer.Parser')
    @patch('codebase_rag.indexer.get_language')
    def test_chunk_without_tree_sitter(self, mock_lang, mock_parser, mock_get_lang):
        """Test fallback when tree-sitter is not available."""
        content = "def test(): pass"
        chunks = _chunk_by_treesitter(content, ".py")
        
        # Should fallback to line-based chunking
        assert len(chunks) > 0
        assert all(c["type"] == "block" for c in chunks)

    def test_chunk_malformed_code(self):
        """Test handling of malformed code."""
        malformed_code = "def incomplete_function("
        chunks = _chunk_by_treesitter(malformed_code, ".py")
        
        # Should fallback to line-based chunking
        assert len(chunks) > 0
        assert all(c["type"] == "block" for c in chunks)


class TestFallbackChunking:
    """Test fallback line-based chunking."""

    def test_fallback_basic(self):
        """Test basic line-based chunking."""
        content = "line1\nline2\nline3\nline4\nline5"
        chunks = _fallback_chunk_by_lines(content, chunk_size=2, overlap=0)
        
        assert len(chunks) == 3
        assert chunks[0]["text"] == "line1\nline2"
        assert chunks[1]["text"] == "line3\nline4"
        assert chunks[2]["text"] == "line5"

    def test_fallback_with_overlap(self):
        """Test line-based chunking with overlap."""
        content = "line1\nline2\nline3\nline4\nline5"
        chunks = _fallback_chunk_by_lines(content, chunk_size=3, overlap=1)
        
        # With 5 lines, chunk_size=3, overlap=1:
        # Chunk 1: lines 1-3 (indices 0,1,2)
        # Next start: 2 (0 + 3 - 1)
        # Chunk 2: lines 3-5 (indices 2,3,4)
        # Total: 2 chunks
        assert len(chunks) == 2
        assert chunks[0]["text"] == "line1\nline2\nline3"
        assert chunks[1]["text"] == "line3\nline4\nline5"

    def test_fallback_metadata(self):
        """Test fallback chunking metadata."""
        content = "some content"
        chunks = _fallback_chunk_by_lines(content)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert "type" in chunk
            assert "name" in chunk
            assert "line_start" in chunk
            assert "line_end" in chunk
            assert chunk["type"] == "block"
            assert chunk["name"] == ""
