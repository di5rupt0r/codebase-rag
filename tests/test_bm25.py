"""Tests for BM25 sparse search functionality."""

import pytest
from unittest.mock import Mock, patch
from codebase_rag.indexer import BM25Index


class TestBM25Index:
    """Test BM25 index functionality."""

    def test_index_initialization(self):
        """Test BM25 index initialization."""
        with patch('codebase_rag.indexer.BM25Okapi'):
            index = BM25Index()
            assert index.bm25 is None
            assert index.chunks == []
            assert index.tokenized_corpus == []

    @patch('codebase_rag.indexer.BM25Okapi', None)
    def test_index_without_bm25_dependency(self):
        """Test error when BM25 is not available."""
        with pytest.raises(ImportError, match="rank-bm25 not available"):
            BM25Index()

    def test_index_chunks(self):
        """Test indexing chunks."""
        mock_chunks = [
            {"text": "function test() { return true; }"},
            {"text": "class TestClass { constructor() {} }"},
            {"text": "variable x = 42;"},
        ]
        
        with patch('codebase_rag.indexer.BM25Okapi') as mock_bm25:
            index = BM25Index()
            index.index(mock_chunks)
            
            assert index.chunks == mock_chunks
            assert len(index.tokenized_corpus) == 3
            mock_bm25.assert_called_once()

    def test_tokenize_text(self):
        """Test text tokenization."""
        with patch('codebase_rag.indexer.BM25Okapi'):
            index = BM25Index()
            
            # Test basic tokenization
            tokens = index._tokenize("function_name variable_123 test")
            expected = ["function_name", "variable_123", "test"]
            assert tokens == expected
            
            # Test case insensitivity
            tokens = index._tokenize("FUNCTION_NAME Variable_123 TEST")
            expected = ["function_name", "variable_123", "test"]
            assert tokens == expected
            
            # Test minimum length filter
            tokens = index._tokenize("a b c d e")
            expected = []  # all tokens < 2 chars
            assert tokens == expected

    def test_search_empty_index(self):
        """Test search on empty index."""
        with patch('codebase_rag.indexer.BM25Okapi'):
            index = BM25Index()
            results = index.search("test query")
            assert results == []

    def test_search_with_results(self):
        """Test search with mocked results."""
        mock_chunks = [
            {"text": "function test_function() { return true; }"},
            {"text": "class TestClass { method() {} }"},
            {"text": "variable test_variable = 42;"},
        ]
        
        with patch('codebase_rag.indexer.BM25Okapi') as mock_bm25:
            # Mock BM25 to return specific scores
            mock_instance = mock_bm25.return_value
            mock_instance.get_scores.return_value = [0.8, 0.6, 0.4]
            
            index = BM25Index()
            index.index(mock_chunks)
            
            results = index.search("test", top_k=2)
            
            # Should return top 2 results sorted by score
            assert len(results) == 2
            assert results[0] == (0, 0.8)  # highest score
            assert results[1] == (1, 0.6)  # second highest
            
            # Should filter zero scores
            mock_instance.get_scores.return_value = [0.0, 0.0, 0.0]
            results = index.search("test", top_k=5)
            assert results == []

    def test_search_top_k(self):
        """Test top_k parameter in search."""
        mock_chunks = [
            {"text": f"content_{i}"} for i in range(10)
        ]
        
        with patch('codebase_rag.indexer.BM25Okapi') as mock_bm25:
            mock_instance = mock_bm25.return_value
            # Mock scores that will result in specific ordering
            mock_instance.get_scores.return_value = list(range(10, 0, -1))  # 10, 9, 8, ..., 1
            
            index = BM25Index()
            index.index(mock_chunks)
            
            results = index.search("test", top_k=3)
            assert len(results) == 3
            # Should return indices with highest scores (0, 1, 2 in our mock)
            assert results[0][0] == 0
            assert results[1][0] == 1  
            assert results[2][0] == 2
