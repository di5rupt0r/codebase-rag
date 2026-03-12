"""Tests for Reciprocal Rank Fusion functionality."""

import pytest
from codebase_rag.indexer import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    """Test RRF fusion functionality."""

    def test_rrf_basic_fusion(self):
        """Test basic RRF fusion of dense and sparse results."""
        dense_results = [
            {"chunk_index": 0, "score": 0.9},
            {"chunk_index": 1, "score": 0.8},
            {"chunk_index": 2, "score": 0.7},
        ]
        
        sparse_results = [(1, 0.5), (0, 0.3), (2, 0.1)]  # (chunk_idx, score)
        
        chunks = [
            {"text": "chunk0", "path": "file0.py", "type": "function", "name": "func0"},
            {"text": "chunk1", "path": "file1.py", "type": "class", "name": "Class1"},
            {"text": "chunk2", "path": "file2.py", "type": "function", "name": "func2"},
        ]
        
        results = reciprocal_rank_fusion(dense_results, sparse_results, chunks, k=60, top_k=3)
        
        assert len(results) == 3
        
        # Check that results have expected structure
        for result in results:
            assert "path" in result
            assert "content" in result
            assert "score" in result
            assert "type" in result
            assert "name" in result

    def test_rrf_score_calculation(self):
        """Test RRF score calculation."""
        # Dense: chunk 0 (rank 1), chunk 1 (rank 2)
        dense_results = [
            {"chunk_index": 0, "score": 0.9},
            {"chunk_index": 1, "score": 0.8},
        ]
        
        # Sparse: chunk 1 (rank 1), chunk 0 (rank 2)  
        sparse_results = [(1, 0.5), (0, 0.3)]
        
        chunks = [
            {"text": "chunk0", "path": "file0.py"},
            {"text": "chunk1", "path": "file1.py"},
        ]
        
        results = reciprocal_rank_fusion(dense_results, sparse_results, chunks, k=60, top_k=2)
        
        # Calculate expected RRF scores:
        # Chunk 0: dense rank 1 -> 1/(60+1) = 0.0164, sparse rank 2 -> 1/(60+2) = 0.0161, total = 0.0325
        # Chunk 1: dense rank 2 -> 1/(60+2) = 0.0161, sparse rank 1 -> 1/(60+1) = 0.0164, total = 0.0325
        
        # Should be sorted by combined score (both equal in this case, but chunk 0 appears first in dense)
        assert len(results) == 2
        assert results[0]["chunk_index"] == 0
        assert results[1]["chunk_index"] == 1

    def test_rrf_empty_dense_results(self):
        """Test RRF with only sparse results."""
        dense_results = []
        sparse_results = [(0, 0.8), (1, 0.6)]
        
        chunks = [
            {"text": "chunk0", "path": "file0.py"},
            {"text": "chunk1", "path": "file1.py"},
        ]
        
        results = reciprocal_rank_fusion(dense_results, sparse_results, chunks, k=60, top_k=2)
        
        assert len(results) == 2
        assert results[0]["chunk_index"] == 0
        assert results[1]["chunk_index"] == 1

    def test_rrf_empty_sparse_results(self):
        """Test RRF with only dense results."""
        dense_results = [
            {"chunk_index": 0, "score": 0.9},
            {"chunk_index": 1, "score": 0.8},
        ]
        sparse_results = []
        
        chunks = [
            {"text": "chunk0", "path": "file0.py"},
            {"text": "chunk1", "path": "file1.py"},
        ]
        
        results = reciprocal_rank_fusion(dense_results, sparse_results, chunks, k=60, top_k=2)
        
        assert len(results) == 2
        assert results[0]["chunk_index"] == 0
        assert results[1]["chunk_index"] == 1

    def test_rrf_top_k_limiting(self):
        """Test top_k parameter limits results."""
        dense_results = [
            {"chunk_index": i, "score": 0.9 - i * 0.1} for i in range(10)
        ]
        sparse_results = [(i, 0.8 - i * 0.1) for i in range(10)]
        
        chunks = [
            {"text": f"chunk{i}", "path": f"file{i}.py"} for i in range(10)
        ]
        
        results = reciprocal_rank_fusion(dense_results, sparse_results, chunks, k=60, top_k=5)
        
        assert len(results) == 5

    def test_rrf_different_k_values(self):
        """Test RRF with different k values."""
        dense_results = [{"chunk_index": 0, "score": 0.9}]
        sparse_results = [(0, 0.8)]
        chunks = [{"text": "chunk0", "path": "file0.py"}]
        
        # Test with different k values
        result_k30 = reciprocal_rank_fusion(dense_results, sparse_results, chunks, k=30, top_k=1)
        result_k90 = reciprocal_rank_fusion(dense_results, sparse_results, chunks, k=90, top_k=1)
        
        # Lower k should give higher scores
        assert result_k30[0]["score"] > result_k90[0]["score"]

    def test_rrf_missing_chunk_data(self):
        """Test RRF with missing chunk data."""
        dense_results = [
            {"chunk_index": 0, "score": 0.9},
            {"chunk_index": 999, "score": 0.8},  # missing chunk
        ]
        sparse_results = [(0, 0.8)]
        
        chunks = [
            {"text": "chunk0", "path": "file0.py"},
            # No chunk with index 999
        ]
        
        results = reciprocal_rank_fusion(dense_results, sparse_results, chunks, k=60, top_k=5)
        
        # Should only return results for existing chunks
        assert len(results) == 1
        assert results[0]["chunk_index"] == 0
