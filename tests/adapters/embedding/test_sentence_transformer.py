import logging

import pytest
from confluence_gateway.adapters.embedding.sentence_transformer import (
    SentenceTransformerProvider,
    _torch_available,
    torch,
)
from confluence_gateway.core.config import EmbeddingConfig
from confluence_gateway.core.exceptions import EmbeddingProviderError

TEST_MODEL_NAME = "all-MiniLM-L6-v2"
TEST_MODEL_DIMENSION = 384


@pytest.fixture
def valid_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="sentence-transformers",
        model_name=TEST_MODEL_NAME,
        dimension=TEST_MODEL_DIMENSION,
        device="cpu",
    )


@pytest.fixture
def invalid_model_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="sentence-transformers",
        model_name="this-model-does-not-exist-hopefully",
        dimension=TEST_MODEL_DIMENSION,
    )


@pytest.fixture
def missing_dimension_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="sentence-transformers",
        model_name=TEST_MODEL_NAME,
        dimension=None,
    )


@pytest.fixture
def mismatched_dimension_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="sentence-transformers",
        model_name=TEST_MODEL_NAME,
        dimension=TEST_MODEL_DIMENSION + 1,
    )


@pytest.fixture
def cpu_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="sentence-transformers",
        model_name=TEST_MODEL_NAME,
        dimension=TEST_MODEL_DIMENSION,
        device="cpu",
    )


@pytest.fixture
def cuda_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="sentence-transformers",
        model_name=TEST_MODEL_NAME,
        dimension=TEST_MODEL_DIMENSION,
        device="cuda",
    )


@pytest.fixture
def auto_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="sentence-transformers",
        model_name=TEST_MODEL_NAME,
        dimension=TEST_MODEL_DIMENSION,
        device=None,
    )


@pytest.fixture
def initialized_provider(valid_config: EmbeddingConfig) -> SentenceTransformerProvider:
    provider = SentenceTransformerProvider(valid_config)
    try:
        provider.initialize()
        yield provider
    finally:
        provider.close()


@pytest.mark.integration
class TestSentenceTransformerInitialization:
    def test_initialize_success(self, valid_config: EmbeddingConfig):
        provider = SentenceTransformerProvider(valid_config)
        provider.initialize()
        assert provider.model is not None
        assert provider.device == "cpu"
        assert provider.get_dimension() == TEST_MODEL_DIMENSION
        provider.close()

    def test_initialize_missing_model_name(self):
        config = EmbeddingConfig(
            provider="sentence-transformers", dimension=TEST_MODEL_DIMENSION
        )
        provider = SentenceTransformerProvider(config)
        with pytest.raises(
            EmbeddingProviderError, match="No embedding model name provided"
        ):
            provider.initialize()

    def test_initialize_missing_dimension(
        self, missing_dimension_config: EmbeddingConfig
    ):
        provider = SentenceTransformerProvider(missing_dimension_config)
        with pytest.raises(
            EmbeddingProviderError, match="No embedding dimension provided"
        ):
            provider.initialize()

    def test_initialize_invalid_model_name(self, invalid_model_config: EmbeddingConfig):
        provider = SentenceTransformerProvider(invalid_model_config)
        with pytest.raises(
            EmbeddingProviderError, match="Could not load sentence-transformer model"
        ):
            provider.initialize()

    def test_initialize_mismatched_dimension(
        self, mismatched_dimension_config: EmbeddingConfig
    ):
        provider = SentenceTransformerProvider(mismatched_dimension_config)
        with pytest.raises(
            EmbeddingProviderError, match="does not match configured dimension"
        ):
            provider.initialize()

    def test_initialize_device_cpu(self, cpu_config: EmbeddingConfig):
        provider = SentenceTransformerProvider(cpu_config)
        provider.initialize()
        assert provider.device == "cpu"
        provider.close()

    @pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")
    def test_initialize_device_cuda_available(
        self, cuda_config: EmbeddingConfig, monkeypatch
    ):
        if not (torch and torch.cuda.is_available()):
            pytest.skip("CUDA not available on this system")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        provider = SentenceTransformerProvider(cuda_config)
        provider.initialize()
        assert provider.device == "cuda"
        provider.close()

    @pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")
    def test_initialize_device_cuda_unavailable(
        self, cuda_config: EmbeddingConfig, monkeypatch, caplog
    ):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        provider = SentenceTransformerProvider(cuda_config)
        with caplog.at_level(logging.WARNING):
            provider.initialize()
        assert provider.device == "cpu"
        assert "CUDA requested but not available. Falling back to CPU." in caplog.text
        provider.close()

    @pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")
    def test_initialize_device_auto_cuda(
        self, auto_config: EmbeddingConfig, monkeypatch
    ):
        if not (torch and torch.cuda.is_available()):
            pytest.skip("CUDA not available on this system for auto-detection")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        provider = SentenceTransformerProvider(auto_config)
        provider.initialize()
        assert provider.device == "cuda"
        provider.close()

    @pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")
    def test_initialize_device_auto_cpu_no_torch(
        self, auto_config: EmbeddingConfig, monkeypatch
    ):
        monkeypatch.setattr(
            "confluence_gateway.adapters.embedding.sentence_transformer._torch_available",
            False,
        )
        provider = SentenceTransformerProvider(auto_config)
        provider.initialize()
        assert provider.device == "cpu"
        provider.close()

    @pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")
    def test_initialize_device_auto_cpu_no_cuda(
        self, auto_config: EmbeddingConfig, monkeypatch
    ):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        provider = SentenceTransformerProvider(auto_config)
        provider.initialize()
        assert provider.device == "cpu"
        provider.close()

    def test_initialize_device_invalid(self, valid_config: EmbeddingConfig, caplog):
        invalid_device_config = valid_config.model_copy()
        invalid_device_config.device = "tpu"
        provider = SentenceTransformerProvider(invalid_device_config)
        with caplog.at_level(logging.WARNING):
            provider.initialize()
        assert provider.device == "cpu"
        assert "Invalid device 'tpu' requested. Falling back to CPU." in caplog.text
        provider.close()


@pytest.mark.integration
class TestSentenceTransformerEmbedding:
    def test_embed_text_success(
        self, initialized_provider: SentenceTransformerProvider
    ):
        text = "This is a test sentence."
        embedding = initialized_provider.embed_text(text)
        assert isinstance(embedding, list)
        assert len(embedding) == TEST_MODEL_DIMENSION
        assert all(isinstance(f, float) for f in embedding)

    def test_embed_texts_success(
        self, initialized_provider: SentenceTransformerProvider
    ):
        texts = ["First sentence.", "Second sentence."]
        embeddings = initialized_provider.embed_texts(texts)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(isinstance(e, list) for e in embeddings)
        assert all(len(e) == TEST_MODEL_DIMENSION for e in embeddings)
        assert all(isinstance(f, float) for e in embeddings for f in e)

    def test_embed_text_empty(self, initialized_provider: SentenceTransformerProvider):
        embedding = initialized_provider.embed_text("")
        assert embedding == []

    def test_embed_text_invalid(
        self, initialized_provider: SentenceTransformerProvider
    ):
        embedding = initialized_provider.embed_text(None)  # type: ignore
        assert embedding == []

    def test_embed_texts_empty_list(
        self, initialized_provider: SentenceTransformerProvider
    ):
        embeddings = initialized_provider.embed_texts([])
        assert embeddings == []

    def test_embed_texts_with_empty_invalid_strings(
        self, initialized_provider: SentenceTransformerProvider, caplog
    ):
        texts = ["Valid sentence.", "", "Another valid one.", None, "  "]  # type: ignore
        with caplog.at_level(logging.WARNING):
            embeddings = initialized_provider.embed_texts(texts)
        assert len(embeddings) == 2
        assert "Filtered out 3 empty/invalid strings" in caplog.text

    def test_embed_texts_all_empty_invalid(
        self, initialized_provider: SentenceTransformerProvider, caplog
    ):
        texts = ["", None, "   "]  # type: ignore
        with caplog.at_level(logging.WARNING):
            embeddings = initialized_provider.embed_texts(texts)
        assert embeddings == []
        assert "All texts in the batch were empty or invalid" in caplog.text

    def test_embed_before_initialize(self, valid_config: EmbeddingConfig):
        provider = SentenceTransformerProvider(valid_config)
        with pytest.raises(EmbeddingProviderError, match="not initialized"):
            provider.embed_text("test")
        with pytest.raises(EmbeddingProviderError, match="not initialized"):
            provider.embed_texts(["test"])

    def test_embedding_consistency(
        self, initialized_provider: SentenceTransformerProvider
    ):
        text = "This sentence should have a consistent embedding."
        embedding1 = initialized_provider.embed_text(text)
        embedding2 = initialized_provider.embed_text(text)
        assert embedding1 == embedding2

        texts = ["Sentence A", "Sentence B"]
        embeddings1 = initialized_provider.embed_texts(texts)
        embeddings2 = initialized_provider.embed_texts(texts)
        assert embeddings1 == embeddings2


class TestSentenceTransformerMisc:
    def test_get_dimension_success(
        self, initialized_provider: SentenceTransformerProvider
    ):
        assert initialized_provider.get_dimension() == TEST_MODEL_DIMENSION

    def test_get_dimension_not_configured(self, valid_config: EmbeddingConfig):
        config_no_dim = valid_config.model_copy()
        config_no_dim.dimension = None
        provider = SentenceTransformerProvider(config_no_dim)
        with pytest.raises(EmbeddingProviderError, match="dimension is not configured"):
            provider.get_dimension()

    def test_get_dimension_before_initialize(self, valid_config: EmbeddingConfig):
        provider = SentenceTransformerProvider(valid_config)
        assert provider.config.dimension is not None
        with pytest.raises(EmbeddingProviderError, match="dimension is not configured"):
            provider.get_dimension()

    def test_close_success(self, initialized_provider: SentenceTransformerProvider):
        assert initialized_provider.model is not None
        initialized_provider.close()
        assert initialized_provider.model is None

    def test_close_multiple_times(
        self, initialized_provider: SentenceTransformerProvider
    ):
        initialized_provider.close()
        initialized_provider.close()
        assert initialized_provider.model is None

    def test_close_before_initialize(self, valid_config: EmbeddingConfig):
        provider = SentenceTransformerProvider(valid_config)
        provider.close()
        assert provider.model is None

    @pytest.mark.skipif(
        not _torch_available or not torch, reason="PyTorch or torch.cuda not available"
    )
    def test_close_cuda_cache_clear(self, cuda_config: EmbeddingConfig, monkeypatch):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        mock_empty_cache_called = False

        def mock_empty_cache():
            nonlocal mock_empty_cache_called
            mock_empty_cache_called = True

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty_cache)

        provider = SentenceTransformerProvider(cuda_config)
        provider.initialize()
        assert provider.device == "cuda"
        provider.close()
        assert mock_empty_cache_called is True
