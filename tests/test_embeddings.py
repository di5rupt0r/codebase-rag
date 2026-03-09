"""Tests for embeddings module."""

from unittest.mock import MagicMock, Mock, patch

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


class TestUniXCoderEmbeddingProvider:
    """Test EmbeddingProvider with transformers backend (unixcoder-base)."""

    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_init_uses_sentence_transformers_backend_for_default(self, mock_st):
        """Default model uses sentence_transformers backend, not transformers."""
        mock_st.return_value = Mock()

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider()

        assert provider._backend == "sentence_transformers"
        mock_st.assert_called_once()

    @patch("codebase_rag.embeddings.AutoModel")
    @patch("codebase_rag.embeddings.AutoTokenizer")
    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_init_uses_transformers_backend_for_unixcoder(self, mock_st, mock_tok, mock_model):
        """microsoft/unixcoder-base uses transformers backend, not SentenceTransformer."""
        mock_tok.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider(model_name="microsoft/unixcoder-base")

        assert provider._backend == "transformers"
        mock_st.assert_not_called()
        mock_tok.from_pretrained.assert_called_once_with("microsoft/unixcoder-base")
        mock_model.from_pretrained.assert_called_once_with("microsoft/unixcoder-base")

    @patch("codebase_rag.embeddings.AutoModel")
    @patch("codebase_rag.embeddings.AutoTokenizer")
    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_encode_unixcoder_returns_2d_numpy_array(self, mock_st, mock_tok, mock_automodel):
        """Encoding with unixcoder backend returns 2D numpy array."""
        import torch

        mock_tokenizer = Mock()
        mock_tok.from_pretrained.return_value = mock_tokenizer

        seq_len, hidden_size = 10, 768
        last_hidden = torch.ones(1, seq_len, hidden_size)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        mock_tokenizer.return_value = {"attention_mask": attention_mask}

        mock_transformer = Mock()
        mock_transformer.eval.return_value = None
        outputs = Mock()
        outputs.last_hidden_state = last_hidden
        mock_transformer.return_value = outputs
        mock_automodel.from_pretrained.return_value = mock_transformer

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider(model_name="microsoft/unixcoder-base")
        result = provider.encode("def foo(): pass")

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (1, hidden_size)

    @patch("codebase_rag.embeddings.AutoModel")
    @patch("codebase_rag.embeddings.AutoTokenizer")
    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_encode_unixcoder_wraps_single_string_in_list(self, mock_st, mock_tok, mock_automodel):
        """Single string input is handled correctly (wrapped internally)."""
        import torch

        mock_tokenizer = Mock()
        mock_tok.from_pretrained.return_value = mock_tokenizer

        hidden_size = 768
        last_hidden = torch.ones(1, 5, hidden_size)
        attention_mask = torch.ones(1, 5, dtype=torch.long)
        mock_tokenizer.return_value = {"attention_mask": attention_mask}

        mock_transformer = Mock()
        mock_transformer.eval.return_value = None
        outputs = Mock()
        outputs.last_hidden_state = last_hidden
        mock_transformer.return_value = outputs
        mock_automodel.from_pretrained.return_value = mock_transformer

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider(model_name="microsoft/unixcoder-base")
        result = provider.encode("single string")

        # Tokenizer must have been called with a list
        call_args = mock_tokenizer.call_args
        assert isinstance(call_args[0][0], list)
        assert result.shape[0] == 1

    @patch("codebase_rag.embeddings.AutoModel")
    @patch("codebase_rag.embeddings.AutoTokenizer")
    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_encode_unixcoder_normalizes_embeddings(self, mock_st, mock_tok, mock_automodel):
        """normalize_embeddings=True produces unit-norm vectors."""
        import torch

        mock_tokenizer = Mock()
        mock_tok.from_pretrained.return_value = mock_tokenizer

        hidden_size = 4
        # Make last_hidden state such that mean pooling gives [3, 4, 0, 0]
        # which has norm 5 → normalized: [0.6, 0.8, 0, 0]
        embeddings_raw = torch.tensor([[[3.0, 4.0, 0.0, 0.0]] * 3])  # (1, 3, 4)
        attention_mask = torch.ones(1, 3, dtype=torch.long)
        mock_tokenizer.return_value = {"attention_mask": attention_mask}

        mock_transformer = Mock()
        mock_transformer.eval.return_value = None
        outputs = Mock()
        outputs.last_hidden_state = embeddings_raw
        mock_transformer.return_value = outputs
        mock_automodel.from_pretrained.return_value = mock_transformer

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider(model_name="microsoft/unixcoder-base")
        result = provider.encode("code", normalize_embeddings=True)

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0], atol=1e-6)

    @patch("codebase_rag.embeddings.AutoModel")
    @patch("codebase_rag.embeddings.AutoTokenizer")
    @patch("codebase_rag.embeddings.SentenceTransformer")
    def test_encode_unixcoder_multiple_texts(self, mock_st, mock_tok, mock_automodel):
        """Encoding multiple texts returns correct batch shape."""
        import torch

        mock_tokenizer = Mock()
        mock_tok.from_pretrained.return_value = mock_tokenizer

        batch, seq_len, hidden_size = 3, 8, 768
        last_hidden = torch.ones(batch, seq_len, hidden_size)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
        mock_tokenizer.return_value = {"attention_mask": attention_mask}

        mock_transformer = Mock()
        mock_transformer.eval.return_value = None
        outputs = Mock()
        outputs.last_hidden_state = last_hidden
        mock_transformer.return_value = outputs
        mock_automodel.from_pretrained.return_value = mock_transformer

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider(model_name="microsoft/unixcoder-base")
        result = provider.encode(["def foo():", "class Bar:", "import os"])

        assert result.shape == (batch, hidden_size)
