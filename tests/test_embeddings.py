"""Tests for embeddings module."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest


class TestEmbeddingProvider:
    """Test EmbeddingProvider class — V5.0 lazy-loading API."""

    def test_init_does_not_load_model(self):
        """EmbeddingProvider.__init__ must NOT load the model (lazy loading)."""
        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider("any-model")
        assert provider._backend is None
        assert provider._model is None
        assert provider._tokenizer is None
        assert provider._transformer is None

    @patch("codebase_rag.embeddings.get_embedding_model")
    def test_backend_set_on_first_encode(self, mock_get_model):
        """Backend is resolved on the first call to encode(), not at init."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_get_model.return_value = mock_model

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider("some-model")
        assert provider._backend is None  # not yet set
        provider.encode("test")
        assert provider._backend == "sentence_transformers"

    @patch("codebase_rag.embeddings.get_embedding_model")
    def test_encode_single_text(self, mock_get_model):
        """Encoding a single string returns a 2D numpy array."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_get_model.return_value = mock_model

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider("some-model")
        result = provider.encode("hello")

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    @patch("codebase_rag.embeddings.get_embedding_model")
    def test_encode_multiple_texts(self, mock_get_model):
        """Encoding a list of texts returns shape (n, dim)."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_get_model.return_value = mock_model

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider("some-model")
        result = provider.encode(["a", "b"])

        assert result.shape == (2, 2)

    @patch("codebase_rag.embeddings.get_embedding_model")
    def test_encode_normalize_flag_forwarded(self, mock_get_model):
        """normalize_embeddings=True is forwarded to the underlying model."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.6, 0.8]])
        mock_get_model.return_value = mock_model

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider("some-model")
        provider.encode("code", normalize_embeddings=True)

        mock_model.encode.assert_called_once_with("code", normalize_embeddings=True)

    @patch("codebase_rag.embeddings.get_embedding_model")
    def test_encode_returns_numpy_array_when_model_returns_list(self, mock_get_model):
        """Result must be numpy array even if the underlying model returns a list."""
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_get_model.return_value = mock_model

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider("some-model")
        result = provider.encode("test")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)

    @patch("codebase_rag.embeddings.get_embedding_model")
    def test_encode_error_propagates(self, mock_get_model):
        """Exceptions from the underlying model are not swallowed."""
        mock_model = Mock()
        mock_model.encode.side_effect = RuntimeError("GPU OOM")
        mock_get_model.return_value = mock_model

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider("some-model")

        with pytest.raises(RuntimeError, match="GPU OOM"):
            provider.encode("test")


class TestUniXCoderEmbeddingProvider:
    """Test EmbeddingProvider with transformers backend (unixcoder-base).
    
    Note: get_embedding_model and get_transformer_model use @lru_cache.
    The fixture clears those caches so patches are actually applied.
    _backend is lazy-loaded, so encode() must be called before checking it.
    """

    @pytest.fixture(autouse=True)
    def clear_lru_caches(self):
        from codebase_rag.embeddings import get_embedding_model, get_transformer_model
        get_embedding_model.cache_clear()
        get_transformer_model.cache_clear()
        yield
        get_embedding_model.cache_clear()
        get_transformer_model.cache_clear()

    @patch("codebase_rag.embeddings.get_embedding_model")
    def test_init_uses_sentence_transformers_backend_for_default(self, mock_get_st):
        """EmbeddingProvider with a non-unixcoder model name uses sentence_transformers backend."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_get_st.return_value = mock_model

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider("all-MiniLM-L6-v2")  # explicit non-unixcoder
        provider.encode("test")  # trigger lazy load

        assert provider._backend == "sentence_transformers"
        mock_get_st.assert_called_once_with("all-MiniLM-L6-v2")

    @patch("codebase_rag.embeddings.get_transformer_model")
    def test_init_uses_transformers_backend_for_unixcoder(self, mock_get_transformer):
        """microsoft/unixcoder-base uses transformers backend after first encode()."""
        import torch
        mock_tokenizer = Mock()
        hidden = torch.zeros(1, 3, 768)
        attention_mask = torch.ones(1, 3, dtype=torch.long)
        mock_tokenizer.return_value = {"attention_mask": attention_mask, "input_ids": torch.zeros(1, 3, dtype=torch.long)}
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = hidden
        mock_transformer = Mock()
        mock_transformer.return_value = mock_outputs
        mock_get_transformer.return_value = (mock_tokenizer, mock_transformer)

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider(model_name="microsoft/unixcoder-base")
        provider.encode("test")  # trigger lazy load

        assert provider._backend == "transformers"
        mock_get_transformer.assert_called_once_with("microsoft/unixcoder-base")

    @patch("codebase_rag.embeddings.get_transformer_model")
    def test_encode_unixcoder_returns_2d_numpy_array(self, mock_get_transformer):
        """Encoding with unixcoder backend returns 2D numpy array."""
        import torch
        seq_len, hidden_size = 10, 768
        mock_tokenizer = Mock()
        last_hidden = torch.ones(1, seq_len, hidden_size)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        mock_tokenizer.return_value = {"attention_mask": attention_mask, "input_ids": torch.zeros(1, seq_len, dtype=torch.long)}
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = last_hidden
        mock_transformer = Mock()
        mock_transformer.return_value = mock_outputs
        mock_get_transformer.return_value = (mock_tokenizer, mock_transformer)

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider(model_name="microsoft/unixcoder-base")
        result = provider.encode("def foo(): pass")

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (1, hidden_size)

    @patch("codebase_rag.embeddings.get_transformer_model")
    def test_encode_unixcoder_wraps_single_string_in_list(self, mock_get_transformer):
        """Single string input is wrapped in a list before tokenization."""
        import torch
        hidden_size = 768
        mock_tokenizer = Mock()
        last_hidden = torch.ones(1, 5, hidden_size)
        attention_mask = torch.ones(1, 5, dtype=torch.long)
        mock_tokenizer.return_value = {"attention_mask": attention_mask, "input_ids": torch.zeros(1, 5, dtype=torch.long)}
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = last_hidden
        mock_transformer = Mock()
        mock_transformer.return_value = mock_outputs
        mock_get_transformer.return_value = (mock_tokenizer, mock_transformer)

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider(model_name="microsoft/unixcoder-base")
        result = provider.encode("single string")

        call_args = mock_tokenizer.call_args
        assert isinstance(call_args[0][0], list)
        assert result.shape[0] == 1

    @patch("codebase_rag.embeddings.get_transformer_model")
    def test_encode_unixcoder_normalizes_embeddings(self, mock_get_transformer):
        """normalize_embeddings=True produces unit-norm vectors."""
        import torch
        embeddings_raw = torch.tensor([[[3.0, 4.0, 0.0, 0.0]] * 3])  # (1, 3, 4)
        attention_mask = torch.ones(1, 3, dtype=torch.long)
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"attention_mask": attention_mask, "input_ids": torch.zeros(1, 3, dtype=torch.long)}
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = embeddings_raw
        mock_transformer = Mock()
        mock_transformer.return_value = mock_outputs
        mock_get_transformer.return_value = (mock_tokenizer, mock_transformer)

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider(model_name="microsoft/unixcoder-base")
        result = provider.encode("code", normalize_embeddings=True)

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0], atol=1e-6)

    @patch("codebase_rag.embeddings.get_transformer_model")
    def test_encode_unixcoder_multiple_texts(self, mock_get_transformer):
        """Encoding multiple texts returns correct batch shape."""
        import torch
        batch, seq_len, hidden_size = 3, 8, 768
        mock_tokenizer = Mock()
        last_hidden = torch.ones(batch, seq_len, hidden_size)
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
        mock_tokenizer.return_value = {"attention_mask": attention_mask, "input_ids": torch.zeros(batch, seq_len, dtype=torch.long)}
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = last_hidden
        mock_transformer = Mock()
        mock_transformer.return_value = mock_outputs
        mock_get_transformer.return_value = (mock_tokenizer, mock_transformer)

        from codebase_rag.embeddings import EmbeddingProvider
        provider = EmbeddingProvider(model_name="microsoft/unixcoder-base")
        result = provider.encode(["def foo():", "class Bar:", "import os"])

        assert result.shape == (batch, hidden_size)

    @patch("codebase_rag.embeddings.AutoModel.from_pretrained")
    @patch("codebase_rag.embeddings.AutoTokenizer.from_pretrained")
    @patch("codebase_rag.embeddings.snapshot_download")
    def test_get_transformer_model_suppresses_warnings_and_checks_cache(
        self, mock_snapshot, mock_tokenizer, mock_automodel
    ):
        """Test get_transformer_model suppresses warnings, checks snapshot, and uses local_files_only."""
        from codebase_rag.embeddings import get_transformer_model
        
        mock_snapshot.return_value = "/mock/path"
        mock_automodel.return_value = Mock()
        mock_automodel.return_value.to.return_value = Mock()
        
        with patch("transformers.utils.logging.set_verbosity_error") as mock_set_verbosity:
            tokenizer, transformer = get_transformer_model("microsoft/unixcoder-base")
            
            # Assert verbosity suppression
            mock_set_verbosity.assert_called_once()
            
            # Assert snapshot_download was called to checking for updates
            mock_snapshot.assert_called_once()
            _, kwargs = mock_snapshot.call_args
            assert kwargs.get("repo_id") == "microsoft/unixcoder-base"
            
            # Assert AutoModel instantiated with ignore_mismatched_sizes and local_files_only=False
            mock_automodel.assert_called_once()
            _, kwargs = mock_automodel.call_args
            assert kwargs.get("ignore_mismatched_sizes") is True
            assert kwargs.get("local_files_only") is False

    @patch("codebase_rag.embeddings.AutoModel.from_pretrained")
    @patch("codebase_rag.embeddings.AutoTokenizer.from_pretrained")
    @patch("codebase_rag.embeddings.snapshot_download", side_effect=Exception("Network error"))
    def test_get_transformer_model_fallback_on_network_error(
        self, mock_snapshot, mock_tokenizer, mock_automodel
    ):
        """Test that if snapshot_download fails (offline), it loads local_files_only."""
        from codebase_rag.embeddings import get_transformer_model
        
        mock_automodel.return_value = Mock()
        mock_automodel.return_value.to.return_value = Mock()
        
        tokenizer, transformer = get_transformer_model("microsoft/unixcoder-base")
        
        mock_snapshot.assert_called_once()
        mock_automodel.assert_called_once()
        _, kwargs = mock_automodel.call_args
        assert kwargs.get("local_files_only") is True





class TestLazyLoading:
    """Test lazy loading functionality for embeddings."""

    @patch("codebase_rag.embeddings.SentenceTransformer")
    @patch("codebase_rag.embeddings.torch")
    def test_get_embedding_model_singleton(self, mock_torch, mock_st):
        """Test get_embedding_model uses @lru_cache singleton."""
        from codebase_rag.embeddings import get_embedding_model
        
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        # First call should create model
        result1 = get_embedding_model("test-model")
        mock_st.assert_called_once_with("test-model", device="cpu")
        
        # Second call should use cache (no new instantiation)
        result2 = get_embedding_model("test-model")
        mock_st.assert_called_once()  # Still only called once
        
        assert result1 is result2  # Same object from cache

    @patch("codebase_rag.embeddings.snapshot_download")
    @patch("codebase_rag.embeddings.AutoModel")
    @patch("codebase_rag.embeddings.AutoTokenizer")
    @patch("codebase_rag.embeddings.torch")
    def test_get_transformer_model_singleton(self, mock_torch, mock_autotok, mock_automodel, mock_snapshot):
        """Test get_transformer_model uses @lru_cache singleton."""
        from codebase_rag.embeddings import get_transformer_model
        
        mock_snapshot.return_value = "/mock/path"
        mock_torch.cuda.is_available.return_value = True
        mock_tokenizer = Mock()
        mock_transformer = Mock()
        mock_autotok.from_pretrained.return_value = mock_tokenizer
        mock_automodel.from_pretrained.return_value = mock_transformer
        
        # First call should create models
        result1 = get_transformer_model("test-model")
        mock_autotok.from_pretrained.assert_called_once_with("test-model")
        mock_automodel.from_pretrained.assert_called_once_with(
            "test-model", ignore_mismatched_sizes=True, local_files_only=False
        )
        
        # Second call should use cache
        result2 = get_transformer_model("test-model")
        mock_autotok.from_pretrained.assert_called_once()  # Still only called once
        mock_automodel.from_pretrained.assert_called_once()  # Still only called once
        
        assert result1 == result2  # Same objects from cache

    @patch("codebase_rag.embeddings.get_embedding_model")
    def test_lazy_loading_in_provider(self, mock_get_model):
        """Test EmbeddingProvider uses lazy loading."""
        from codebase_rag.embeddings import EmbeddingProvider
        
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        
        provider = EmbeddingProvider("test-model")
        
        # Models should NOT be loaded during initialization
        assert provider._backend is None
        assert provider._model is None
        mock_get_model.assert_not_called()
        
        # Models should be loaded only when encode() is called
        provider.encode("test text")
        mock_get_model.assert_called_once_with("test-model")

    @patch("codebase_rag.embeddings.get_embedding_model")
    def test_cache_info_monitoring(self, mock_get_model):
        """Test cache info is available for monitoring."""
        from codebase_rag.embeddings import EmbeddingProvider
        from codebase_rag.embeddings import get_embedding_model
        
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        
        # Call multiple times to populate cache
        get_embedding_model("test-model")
        get_embedding_model("test-model")
        get_embedding_model("test-model")
        
        cache_info = EmbeddingProvider.get_cache_info()
        
        assert "embedding_model_cache" in cache_info
        assert "transformer_model_cache" in cache_info
        # Check that cache info is accessible (not exact values due to mocking)
