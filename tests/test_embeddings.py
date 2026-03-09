"""Tests for embeddings module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestEmbeddingProvider:
    """Test EmbeddingProvider class."""

    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_init_default_model(self, mock_st):
        """Test EmbeddingProvider initializes with default model."""
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        
        mock_st.assert_called_once_with("all-MiniLM-L6-v2")
        assert provider.model == mock_model

    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_init_custom_model(self, mock_st):
        """Test EmbeddingProvider initializes with custom model."""
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider(model_name="custom-model")
        
        mock_st.assert_called_once_with("custom-model")
        assert provider.model == mock_model

    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_encode_single_text(self, mock_st):
        """Test encoding a single text returns correct shape."""
        mock_model = Mock()
        # Simulate sentence-transformers output
        mock_embedding = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        mock_model.encode.return_value = mock_embedding
        mock_st.return_value = mock_model
        
        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        result = provider.encode("test text")
        
        mock_model.encode.assert_called_once_with("test text", normalize_embeddings=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 5)  # 1 text, 5 dimensions
        np.testing.assert_array_equal(result, mock_embedding)

    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_encode_multiple_texts(self, mock_st):
        """Test encoding multiple texts returns correct shape."""
        mock_model = Mock()
        # Simulate sentence-transformers output for 2 texts
        mock_embedding = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],  # text 1
            [0.6, 0.7, 0.8, 0.9, 1.0],  # text 2
        ])
        mock_model.encode.return_value = mock_embedding
        mock_st.return_value = mock_model
        
        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        texts = ["text 1", "text 2"]
        result = provider.encode(texts)
        
        mock_model.encode.assert_called_once_with(texts, normalize_embeddings=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 5)  # 2 texts, 5 dimensions
        np.testing.assert_array_equal(result, mock_embedding)

    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_encode_with_normalize_embeddings(self, mock_st):
        """Test encoding passes through normalize_embeddings parameter."""
        mock_model = Mock()
        mock_embedding = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embedding
        mock_st.return_value = mock_model
        
        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        provider.encode("test", normalize_embeddings=True)
        
        mock_model.encode.assert_called_once_with("test", normalize_embeddings=True)

    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_encode_empty_list(self, mock_st):
        """Test encoding empty list returns empty array."""
        mock_model = Mock()
        mock_embedding = np.array([]).reshape(0, 384)  # Empty with correct shape
        mock_model.encode.return_value = mock_embedding
        mock_st.return_value = mock_model
        
        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        result = provider.encode([])
        
        mock_model.encode.assert_called_once_with([], normalize_embeddings=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 384)

    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_encode_returns_numpy_array(self, mock_st):
        """Test encode always returns numpy array even if model returns list."""
        mock_model = Mock()
        # Model returns list instead of numpy array
        mock_embedding = [[0.1, 0.2, 0.3]]
        mock_model.encode.return_value = mock_embedding
        mock_st.return_value = mock_model
        
        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        result = provider.encode("test")
        
        # Should convert to numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)
        np.testing.assert_array_equal(result, np.array(mock_embedding))

    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_model_loading_error_handling(self, mock_st):
        """Test error handling when model loading fails."""
        mock_st.side_effect = Exception("Model loading failed")
        
        from codebase_rag.embeddings import EmbeddingProvider
        
        with pytest.raises(Exception, match="Model loading failed"):
            EmbeddingProvider()

    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_encode_error_handling(self, mock_st):
        """Test error handling when encoding fails."""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_st.return_value = mock_model
        
        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()
        
        with pytest.raises(Exception, match="Encoding failed"):
            provider.encode("test text")
