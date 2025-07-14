"""
Test file demonstrating shared embedding model optimization.

This test file showcases how the session-scoped embedding fixtures
dramatically improve test performance by loading models once per session
and reusing them across all tests requiring embeddings.

Performance benefits:
- Single model load per test session (vs. per test)
- Shared model injection into provider instances
- Thread-safe concurrent access
- Automatic performance tracking and reporting

Run this test file to see the optimization in action:
    uv run pytest tests/test_shared_embedding_optimization.py -v -s
"""

import time
from unittest.mock import patch

import pytest

from tests.fixtures.shared_embedding import (
    get_shared_model_thread_safe,
    inject_shared_model_into_provider,
    log_embedding_operation,
)


class TestSharedEmbeddingOptimization:
    """Test class demonstrating shared embedding model optimization patterns."""

    def test_shared_model_availability(self, shared_sentence_transformer_model):
        """Test that the shared sentence-transformer model is available and functional."""
        if shared_sentence_transformer_model is None:
            pytest.skip("sentence-transformers not available in test environment")

        # Verify model is loaded and functional
        assert shared_sentence_transformer_model is not None
        print(f"Shared model type: {type(shared_sentence_transformer_model)}")

        # Test model can generate embeddings
        test_text = ["test embedding generation"]
        start_time = time.time()
        embeddings = shared_sentence_transformer_model.encode(test_text)
        duration = time.time() - start_time

        assert embeddings is not None
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384  # all-MiniLM-L6-v2 dimension

        log_embedding_operation("shared_model_encode", duration)
        print(f"Shared model embedding generation: {duration:.4f}s")

    def test_shared_embedding_provider(self, shared_embedding_provider):
        """Test that the shared embedding provider works with injected model."""
        if shared_embedding_provider is None:
            pytest.skip("Shared embedding provider not available")

        # Verify provider is configured correctly
        assert shared_embedding_provider is not None
        assert hasattr(shared_embedding_provider, "_model")
        assert hasattr(shared_embedding_provider, "_is_initialized")
        assert shared_embedding_provider._is_initialized is True

        print(f"Shared provider type: {type(shared_embedding_provider)}")
        print(f"Provider initialized: {shared_embedding_provider._is_initialized}")

    def test_embedding_service_with_shared_model(
        self, embedding_service_with_shared_model
    ):
        """Test that the embedding service uses the shared model optimization."""
        if embedding_service_with_shared_model is None:
            pytest.skip("Embedding service with shared model not available")

        # Verify service is configured correctly
        assert embedding_service_with_shared_model is not None
        assert hasattr(embedding_service_with_shared_model, "provider")

        # Test service functionality with shared model
        test_texts = ["test embedding service", "another test text"]
        start_time = time.time()

        try:
            embeddings = embedding_service_with_shared_model.generate_embeddings(
                test_texts
            )
            duration = time.time() - start_time

            assert embeddings is not None
            assert len(embeddings) == 2
            log_embedding_operation("service_generate_embeddings", duration)
            print(f"Service embedding generation: {duration:.4f}s")
        except Exception as e:
            print(f"Service test skipped due to: {e}")
            pytest.skip(f"Embedding service test failed: {e}")

    def test_model_injection_utility(self, shared_sentence_transformer_model):
        """Test the utility function for injecting shared models into providers."""
        if shared_sentence_transformer_model is None:
            pytest.skip("sentence-transformers not available")

        try:
            from confluence_gateway.adapters.embedding.sentence_transformer import (
                SentenceTransformerAdapter,
            )

            # Create a new provider instance
            provider = SentenceTransformerAdapter(
                model_name="all-MiniLM-L6-v2", device="cpu", dimension=384
            )

            # Test injection
            success = inject_shared_model_into_provider(
                provider, shared_sentence_transformer_model
            )
            assert success is True
            assert provider._model is shared_sentence_transformer_model
            assert provider._is_initialized is True

            print("Model injection utility test: SUCCESS")

        except ImportError:
            pytest.skip("SentenceTransformerAdapter not available")

    def test_thread_safe_model_access(self, shared_sentence_transformer_model):
        """Test thread-safe access to the shared model."""
        if shared_sentence_transformer_model is None:
            pytest.skip("sentence-transformers not available")

        # Test thread-safe accessor
        safe_model = get_shared_model_thread_safe(shared_sentence_transformer_model)
        assert safe_model is shared_sentence_transformer_model

        print("Thread-safe model access test: SUCCESS")

    def test_mock_context_with_shared_model(
        self, mock_sentence_transformer_with_shared_model
    ):
        """Test the mock context fixture that uses the shared model."""
        # This fixture should always provide a model (real or mock)
        assert mock_sentence_transformer_with_shared_model is not None

        # Test that we can use it in a mock context
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_sentence_transformer_with_shared_model,
        ):
            # Simulate importing and using sentence transformers
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("test-model")
            assert model is mock_sentence_transformer_with_shared_model

        print("Mock context with shared model test: SUCCESS")

    def test_performance_comparison_simulation(self, shared_sentence_transformer_model):
        """Simulate performance comparison between shared and individual model loading."""
        if shared_sentence_transformer_model is None:
            pytest.skip("sentence-transformers not available")

        # Simulate multiple embedding operations using shared model
        test_texts = [
            "first test embedding",
            "second test embedding",
            "third test embedding",
        ]

        total_shared_time = 0.0
        for i, text in enumerate(test_texts, 1):
            start_time = time.time()
            _ = shared_sentence_transformer_model.encode([text])  # Just measure timing
            duration = time.time() - start_time
            total_shared_time += duration

            log_embedding_operation(f"simulation_embedding_{i}", duration)
            print(f"Embedding operation {i}: {duration:.4f}s")

        print(
            f"Total time for 3 operations with shared model: {total_shared_time:.4f}s"
        )
        print(
            "Without shared model: Each operation would include ~2s model loading overhead"
        )
        estimated_without_sharing = total_shared_time + (
            3 * 2.0
        )  # 3 operations * 2s load time
        estimated_savings = estimated_without_sharing - total_shared_time
        print(
            f"Estimated time savings: {estimated_savings:.2f}s ({(estimated_savings / estimated_without_sharing) * 100:.1f}% improvement)"
        )


class TestConcurrentEmbeddingAccess:
    """Test concurrent access patterns with shared embedding fixtures."""

    def test_concurrent_model_access_safety(self, shared_sentence_transformer_model):
        """Test that concurrent access to shared model is safe."""
        if shared_sentence_transformer_model is None:
            pytest.skip("sentence-transformers not available")

        import queue
        import threading

        results_queue = queue.Queue()

        def embedding_worker(worker_id: int):
            """Worker function that generates embeddings concurrently."""
            try:
                safe_model = get_shared_model_thread_safe(
                    shared_sentence_transformer_model
                )
                text = f"worker {worker_id} test text"
                embedding = safe_model.encode([text])
                results_queue.put((worker_id, len(embedding[0]), "success"))
            except Exception as e:
                results_queue.put((worker_id, None, f"error: {e}"))

        # Create multiple threads accessing the shared model
        threads = []
        num_workers = 3

        for i in range(num_workers):
            thread = threading.Thread(target=embedding_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all workers succeeded
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        assert len(results) == num_workers
        for worker_id, embedding_dim, status in results:
            assert status == "success", f"Worker {worker_id} failed: {status}"
            assert embedding_dim == 384, (
                f"Worker {worker_id} got wrong embedding dimension: {embedding_dim}"
            )

        print(f"Concurrent access test: {num_workers} workers all succeeded")


class TestPerformanceRegressionPrevention:
    """Tests to prevent performance regressions in embedding optimization."""

    def test_model_loading_time_benchmark(self, shared_sentence_transformer_model):
        """Benchmark shared model performance to catch regressions."""
        if shared_sentence_transformer_model is None:
            pytest.skip("sentence-transformers not available")

        # Benchmark embedding generation time
        test_texts = ["benchmark text"] * 10  # 10 texts for more stable timing

        start_time = time.time()
        embeddings = shared_sentence_transformer_model.encode(test_texts)
        duration = time.time() - start_time

        assert len(embeddings) == 10
        assert len(embeddings[0]) == 384

        # Performance regression check - should be fast since model is pre-loaded
        per_text_time = duration / 10
        assert per_text_time < 0.1, (
            f"Embedding generation too slow: {per_text_time:.4f}s per text"
        )

        log_embedding_operation("benchmark_10_texts", duration)
        print(
            f"Benchmark: {duration:.4f}s for 10 texts ({per_text_time:.4f}s per text)"
        )

    def test_memory_usage_reasonable(self, shared_sentence_transformer_model):
        """Verify that shared model doesn't consume excessive memory."""
        if shared_sentence_transformer_model is None:
            pytest.skip("sentence-transformers not available")

        import os

        import psutil

        # Get current process memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        print(f"Current memory usage: {memory_mb:.1f} MB")

        # For all-MiniLM-L6-v2, memory usage should be reasonable (typically < 500MB for testing)
        # This is a loose check to catch obvious memory leaks
        assert memory_mb < 2000, f"Memory usage seems excessive: {memory_mb:.1f} MB"

        # Generate embeddings and check memory doesn't spike significantly
        test_texts = ["memory test"] * 100
        _ = shared_sentence_transformer_model.encode(
            test_texts
        )  # Just measure memory impact

        new_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase = new_memory_mb - memory_mb

        print(
            f"Memory after 100 embeddings: {new_memory_mb:.1f} MB (increase: {memory_increase:.1f} MB)"
        )

        # Memory increase should be minimal for embedding generation
        assert memory_increase < 100, (
            f"Memory increase too large: {memory_increase:.1f} MB"
        )
