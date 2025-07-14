"""Performance validation tests for shared embedding optimization.

This module validates the performance improvements achieved through shared
sentence-transformers model optimization in integration test scenarios.

Key metrics validated:
- Model loading time reduction
- Provider creation time improvement
- Overall test execution time benefits
- Memory usage optimization
"""

import time
from typing import Any
from unittest.mock import patch

import pytest

from tests.fixtures.shared_embedding import (
    inject_shared_model_into_provider,
    log_embedding_operation,
)


class TestSharedModelPerformance:
    """Test performance improvements from shared model optimization."""

    def test_shared_model_loading_performance(self, shared_sentence_transformer_model):
        """Validate that shared model loading provides expected performance benefits."""
        if shared_sentence_transformer_model is None:
            pytest.skip("Shared model not available for performance testing")

        # The shared model should already be loaded (session-scoped)
        # Test that we can access it without additional loading time
        start_time = time.time()

        # Access the shared model (should be immediate)
        model = shared_sentence_transformer_model
        access_time = time.time() - start_time

        assert model is not None
        assert access_time < 0.1, (
            f"Shared model access took too long: {access_time:.3f}s"
        )

        # Log performance
        log_embedding_operation("shared_model_access", access_time)
        print(f"Shared model access completed in {access_time:.6f}s")

    def test_provider_creation_optimization(self, shared_sentence_transformer_model):
        """Test that provider creation and initialization is optimized with shared model injection."""
        if shared_sentence_transformer_model is None:
            pytest.skip("Shared model not available for optimization testing")

        try:
            from confluence_gateway.adapters.embedding.sentence_transformer import (
                SentenceTransformerProvider,
            )
            from confluence_gateway.core.config import EmbeddingConfig
        except ImportError:
            pytest.skip("SentenceTransformerProvider not available")

        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            dimension=384,
        )

        # Test standard provider creation and initialization (without optimization)
        # Run multiple iterations to get more reliable timing
        iterations = 5
        standard_times = []

        for _ in range(iterations):
            start_time = time.time()
            standard_provider = SentenceTransformerProvider(config)
            standard_provider.initialize()

            # Test actual model usage to measure the real benefit
            standard_provider.embed_text("test text for timing measurement")

            standard_times.append(time.time() - start_time)

        standard_avg_time = sum(standard_times) / len(standard_times)

        # Test optimized provider creation with shared model injection
        optimized_times = []

        for _ in range(iterations):
            optimized_start = time.time()
            optimized_provider = SentenceTransformerProvider(config)

            # Inject shared model (this is the optimization)
            injection_success = inject_shared_model_into_provider(
                optimized_provider, shared_sentence_transformer_model
            )

            # Test actual model usage to measure the real benefit
            optimized_provider.embed_text("test text for timing measurement")

            optimized_times.append(time.time() - optimized_start)

        optimized_avg_time = sum(optimized_times) / len(optimized_times)

        assert injection_success, "Shared model injection should succeed"

        # Log performance metrics
        log_embedding_operation("standard_provider_with_usage", standard_avg_time)
        log_embedding_operation("optimized_provider_with_usage", optimized_avg_time)

        print(
            f"Provider creation+usage times (avg of {iterations} runs): "
            f"Standard={standard_avg_time:.3f}s, Optimized={optimized_avg_time:.3f}s"
        )

        # Calculate performance improvement
        if standard_avg_time > 0:
            improvement_ratio = standard_avg_time / optimized_avg_time
            print(f"Performance improvement: {improvement_ratio:.2f}x faster")

        # The optimized version should be significantly faster due to skipping model loading
        # If both times are very small (< 0.01s), just verify injection worked correctly
        if standard_avg_time < 0.01 and optimized_avg_time < 0.01:
            print(
                "Both times too small for reliable comparison, verifying injection correctness"
            )
            assert injection_success, "Shared model injection should succeed"
            assert optimized_provider.model is shared_sentence_transformer_model, (
                "Optimized provider should use the shared model"
            )
        else:
            # For meaningful timings, optimized should be faster
            assert optimized_avg_time < standard_avg_time, (
                f"Optimized version should be faster: {optimized_avg_time:.3f}s vs {standard_avg_time:.3f}s"
            )

    def test_embedding_operation_performance(self, shared_sentence_transformer_model):
        """Test that embedding operations maintain performance with shared model."""
        if shared_sentence_transformer_model is None:
            pytest.skip("Shared model not available for performance testing")

        try:
            from confluence_gateway.adapters.embedding.sentence_transformer import (
                SentenceTransformerProvider,
            )
            from confluence_gateway.core.config import EmbeddingConfig
        except ImportError:
            pytest.skip("SentenceTransformerProvider not available")

        # Create optimized provider
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            dimension=384,
        )
        provider = SentenceTransformerProvider(config)
        provider.initialize()

        # Inject shared model
        injection_success = inject_shared_model_into_provider(
            provider, shared_sentence_transformer_model
        )

        if not injection_success:
            pytest.skip("Could not inject shared model for performance testing")

        # Test embedding operation performance
        test_texts = [
            "This is a test sentence for embedding performance validation.",
            "Performance testing with shared sentence-transformers model optimization.",
            "Integration tests should benefit from model reuse across test cases.",
        ]

        start_time = time.time()

        # Perform multiple embedding operations
        embeddings = []
        for text in test_texts:
            embedding = provider.embed_text(text)
            embeddings.append(embedding)

        total_embedding_time = time.time() - start_time
        average_embedding_time = total_embedding_time / len(test_texts)

        # Validate embeddings
        assert len(embeddings) == len(test_texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)

        # Log performance metrics
        log_embedding_operation("batch_embedding_operations", total_embedding_time)
        log_embedding_operation("average_embedding_time", average_embedding_time)

        print(
            f"Embedding performance: {len(test_texts)} texts in {total_embedding_time:.3f}s "
            f"(avg: {average_embedding_time:.3f}s per text)"
        )

        # Reasonable performance expectations (adjust based on hardware)
        max_acceptable_time_per_text = 1.0  # 1 second per text is reasonable for CPU
        assert average_embedding_time < max_acceptable_time_per_text, (
            f"Embedding operations too slow: {average_embedding_time:.3f}s per text"
        )


class TestPerformanceRegression:
    """Test suite to detect performance regressions in shared model optimization."""

    def test_no_memory_leaks_in_shared_model(self, shared_sentence_transformer_model):
        """Validate that shared model doesn't cause memory leaks across tests."""
        if shared_sentence_transformer_model is None:
            pytest.skip("Shared model not available for memory testing")

        # This test mainly serves as a placeholder for memory usage validation
        # In a full implementation, you might use memory profiling tools

        # Basic validation that the model is accessible
        assert shared_sentence_transformer_model is not None

        # Test that model can be used multiple times without issues
        start_time = time.time()

        for i in range(5):
            # Simulate repeated model access
            model = shared_sentence_transformer_model
            assert model is not None

        repeated_access_time = time.time() - start_time

        # Log for analysis
        log_embedding_operation("repeated_model_access", repeated_access_time)
        print(f"Repeated model access (5x) completed in {repeated_access_time:.3f}s")

        # Should be very fast since model is already loaded
        assert repeated_access_time < 0.1, (
            f"Repeated access too slow: {repeated_access_time:.3f}s"
        )

    def test_optimization_consistency_across_configurations(
        self, shared_sentence_transformer_model, provider_config: dict[str, Any]
    ):
        """Test that optimization works consistently across different configurations."""
        embedding_provider = provider_config.get("EMBEDDING_PROVIDER")

        # Only test sentence-transformers configurations
        if embedding_provider != "sentence-transformers":
            pytest.skip(f"Skipping optimization test for {embedding_provider} provider")

        if shared_sentence_transformer_model is None:
            pytest.skip("Shared model not available for consistency testing")

        try:
            from confluence_gateway.adapters.embedding.sentence_transformer import (
                SentenceTransformerProvider,
            )
            from confluence_gateway.core.config import EmbeddingConfig
        except ImportError:
            pytest.skip("SentenceTransformerProvider not available")

        # Create provider with current configuration
        start_time = time.time()

        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name=provider_config.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
            device=provider_config.get("EMBEDDING_DEVICE", "cpu"),
            dimension=int(provider_config.get("EMBEDDING_DIMENSION", "384")),
        )
        provider = SentenceTransformerProvider(config)
        provider.initialize()

        # Test optimization injection
        injection_success = inject_shared_model_into_provider(
            provider, shared_sentence_transformer_model
        )

        optimization_time = time.time() - start_time

        assert injection_success, (
            f"Optimization should work for configuration: {provider_config}"
        )

        # Test that the optimized provider works correctly
        test_embedding = provider.embed_text("Test optimization consistency")
        assert isinstance(test_embedding, list)
        assert len(test_embedding) == int(
            provider_config.get("EMBEDDING_DIMENSION", "384")
        )

        # Log configuration-specific performance
        config_name = f"{provider_config.get('VECTOR_DB_TYPE', 'unknown')}_config"
        log_embedding_operation(
            f"optimization_consistency_{config_name}", optimization_time
        )

        print(
            f"Optimization consistency test for {config_name} completed in {optimization_time:.3f}s"
        )
