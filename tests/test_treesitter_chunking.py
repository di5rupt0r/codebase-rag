"""Tests for tree-sitter chunking functionality."""

import pytest
from unittest.mock import Mock, patch
from codebase_rag.indexer import _chunk_by_treesitter, _fallback_chunk_by_lines


class TestTreeSitterChunking:
    """Test tree-sitter based chunking."""

    @patch('codebase_rag.indexer.get_language')
    @patch('codebase_rag.indexer.Parser')
    def test_chunk_python_function(self, mock_parser, mock_get_language):
        """Test chunking Python function definitions."""
        # Mock the tree-sitter behavior
        mock_language = Mock()
        mock_get_language.return_value = mock_language
        mock_parser.return_value.parse.return_value = Mock()
        
        # Mock the tree structure
        mock_node = Mock()
        mock_node.type = "function_definition"
        mock_node.start_byte = 0
        mock_node.end_byte = 50
        mock_node.start_point = (0, 0)
        mock_node.end_point = (5, 0)
        mock_node.children = []
        
        # Mock the root node to return our function node
        mock_root = Mock()
        mock_root.children = [mock_node]
        mock_parser.return_value.parse.return_value.root_node = mock_root
        
        python_code = "def hello_world(): pass"
        chunks = _chunk_by_treesitter(python_code, ".py")
        
        # Should find function
        assert len(chunks) >= 1
        func_chunks = [c for c in chunks if c["type"] == "function"]
        assert len(func_chunks) >= 1

    @patch('codebase_rag.indexer.get_language')
    @patch('codebase_rag.indexer.Parser')
    def test_chunk_javascript_function(self, mock_parser, mock_get_language):
        """Test chunking JavaScript function definitions."""
        # Mock the tree-sitter behavior
        mock_language = Mock()
        mock_get_language.return_value = mock_language
        mock_parser.return_value.parse.return_value = Mock()
        
        # Mock the tree structure with function and class
        mock_func_node = Mock()
        mock_func_node.type = "function_definition"
        mock_func_node.start_byte = 0
        mock_func_node.end_byte = 40
        mock_func_node.start_point = (0, 0)
        mock_func_node.end_point = (4, 0)
        mock_func_node.children = []
        
        mock_class_node = Mock()
        mock_class_node.type = "class_definition"
        mock_class_node.start_byte = 50
        mock_class_node.end_byte = 80
        mock_class_node.start_point = (6, 0)
        mock_class_node.end_point = (10, 0)
        mock_class_node.children = []
        
        # Mock the root node to return both nodes
        mock_root = Mock()
        mock_root.children = [mock_func_node, mock_class_node]
        mock_parser.return_value.parse.return_value.root_node = mock_root
        
        js_code = "function helloWorld() { console.log('Hello'); } class MyClass { method() { return true; } }"
        chunks = _chunk_by_treesitter(js_code, ".js")
        
        # Should find function and class
        assert len(chunks) >= 2
        func_chunks = [c for c in chunks if c["type"] == "function"]
        class_chunks = [c for c in chunks if c["type"] == "class"]
        assert len(func_chunks) >= 1
        assert len(class_chunks) >= 1

    def test_chunk_unsupported_extension(self):
        """Test fallback to line-based for unsupported extensions."""
        content = "some random content\nwith multiple lines"
        chunks = _chunk_by_treesitter(content, ".unknown")
        
        # Should fallback to line-based chunking
        assert len(chunks) > 0
        assert all(c["type"] == "block" for c in chunks)

    @patch('codebase_rag.indexer.get_language')
    @patch('codebase_rag.indexer.Parser')
    def test_chunk_without_tree_sitter(self, mock_parser, mock_get_language):
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
        
        # Let's trace through the logic:
        # lines = ['line1', 'line2', 'line3', 'line4', 'line5']
        # step = max(1, 3-1) = 2
        # i=0: end=min(0+3,5)=3 -> chunk lines[0:3] = ['line1','line2','line3']
        # i=2: end=min(2+3,5)=5 -> chunk lines[2:5] = ['line3','line4','line5']
        # i=4: end=min(4+3,5)=5 -> chunk lines[4:5] = ['line5']
        # Total: 3 chunks
        assert len(chunks) == 3
        assert chunks[0]["text"] == "line1\nline2\nline3"
        assert chunks[1]["text"] == "line3\nline4\nline5"
        assert chunks[2]["text"] == "line5"

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

    def test_fallback_large_file(self):
        """Test fallback for files larger than 500KB."""
        # Create content larger than 500KB
        content = "line\n" * 60000  # ~600KB
        chunks = _chunk_by_treesitter(content, ".py")
        
        # Should fallback to line-based chunking for large files
        assert len(chunks) > 0
        assert all(c["type"] == "block" for c in chunks)
