import logging
from types import ModuleType
from typing import TYPE_CHECKING, Any

from confluence_gateway.adapters.embedding.base import EmbeddingProvider
from confluence_gateway.core.config import (
    get_development_context,
    get_environment_context,
)
from confluence_gateway.core.exceptions import EmbeddingProviderError

_torch: ModuleType | None = None
_torch_available: bool | None = None
_sentence_transformers_available: bool | None = None
_SentenceTransformer: Any | None = None

if TYPE_CHECKING:
    from confluence_gateway.core.config import EmbeddingConfig

logger = logging.getLogger(__name__)


def _check_torch_available() -> bool:
    """Check if torch is available, loading it if needed."""
    global _torch, _torch_available
    if _torch_available is None:
        try:
            import torch

            _torch = torch
            _torch_available = True
        except ImportError:
            _torch_available = False
            _torch = None
    return _torch_available


def _get_sentence_transformer_class() -> Any:
    """Get SentenceTransformer class, loading it if needed."""
    global _SentenceTransformer, _sentence_transformers_available
    if _SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer

            _SentenceTransformer = SentenceTransformer
            _sentence_transformers_available = True
        except ImportError:
            _sentence_transformers_available = False
            raise ImportError(
                "The 'sentence-transformers' library is required for SentenceTransformerProvider. "
                "Please install it using 'pip install sentence-transformers'."
            )
    return _SentenceTransformer


def _log_gpu_memory_usage(operation: str = "operation") -> None:
    """Log current GPU memory usage if CUDA is available."""
    if not _check_torch_available() or not _torch:
        return

    try:
        if _torch.cuda.is_available() and _torch.cuda.device_count() > 0:
            current_device = _torch.cuda.current_device()
            allocated = _torch.cuda.memory_allocated(current_device) / (1024**3)  # GB
            reserved = _torch.cuda.memory_reserved(current_device) / (1024**3)  # GB
            logger.info(
                f"GPU memory usage after {operation}: "
                f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
            )
    except Exception as e:
        logger.debug(f"Could not log GPU memory usage: {e}")


def _log_cuda_device_info() -> None:
    """Log detailed CUDA device information."""
    if not _check_torch_available() or not _torch:
        return

    try:
        if _torch.cuda.is_available():
            device_count = _torch.cuda.device_count()
            logger.info(f"CUDA devices available: {device_count}")

            for i in range(device_count):
                props = _torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # GB
                logger.info(
                    f"CUDA Device {i}: {props.name} "
                    f"(Compute Capability: {props.major}.{props.minor}, "
                    f"Total Memory: {total_memory:.2f} GB, "
                    f"Multiprocessors: {props.multi_processor_count})"
                )

            current_device = _torch.cuda.current_device()
            logger.info(f"Current CUDA device: {current_device}")
            _log_gpu_memory_usage("device info check")
    except Exception as e:
        logger.warning(f"Could not retrieve CUDA device information: {e}")


def _gpu_warmup(model: Any, device: str) -> None:
    """Perform GPU warmup for better performance."""
    if device != "cuda" or not _check_torch_available() or not _torch:
        return

    try:
        if model and _torch.cuda.is_available():
            logger.info("Performing GPU warmup for better performance...")
            # Run a small warmup embedding to initialize CUDA kernels
            warmup_text = "GPU warmup test"
            _ = model.encode(warmup_text, convert_to_numpy=False)
            _log_gpu_memory_usage("GPU warmup")
            logger.info("GPU warmup completed successfully")
    except Exception as e:
        logger.warning(f"GPU warmup failed, but model should still work: {e}")


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, config: "EmbeddingConfig") -> None:
        super().__init__(config)
        self.model: Any | None = None
        self.device: str | None = None
        self.cache_dir: Any | None = None
        self.dev_context = get_development_context()
        self.env_context = get_environment_context()
        self.dev_mode = self.dev_context.enabled
        self._auto_detected_dimension: int | None = None

        if self.dev_mode:
            self.dev_context.log_stub("SentenceTransformerProvider")
            logger.info(
                f"SentenceTransformerProvider initialized in DEV MODE - stub implementation only. "
                f"Model='{self.config.model_name}', Dimension='{self.config.dimension}'"
            )
        else:
            logger.info(
                f"SentenceTransformerProvider initialized with config: "
                f"Model='{self.config.model_name}', Dimension='{self.config.dimension}', "
                f"Requested Device='{self.config.device or 'auto'}'"
            )

    def _determine_device(self) -> str:
        # Check if we're in CI environment first to avoid expensive CUDA detection
        if not self.config.device and self.env_context.is_ci:
            logger.info(
                "CI environment detected and no explicit device configured. "
                "Forcing CPU device to skip expensive CUDA detection for faster CI execution."
            )
            return "cpu"

        if not self.config.device:
            if _check_torch_available():
                torch = _torch
                if torch and torch.cuda.is_available():
                    logger.info("Auto-detected CUDA availability. Using CUDA.")
                    _log_cuda_device_info()
                    return "cuda"
            logger.info(
                "Auto-detection: CUDA not available or torch not installed. Using CPU."
            )
            return "cpu"

        if self.config.device == "cuda":
            if not _check_torch_available():
                logger.warning(
                    "Torch library not found. Cannot use CUDA. Falling back to CPU."
                )
                return "cpu"

            torch = _torch
            if not torch or not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return "cpu"

            logger.info("CUDA requested and available. Using CUDA.")
            # Log CI info if explicit device is being used in CI
            if self.env_context.is_ci:
                logger.info(
                    "CI environment detected with explicit CUDA device configuration. "
                    "Proceeding with CUDA detection as requested."
                )
            _log_cuda_device_info()
            return "cuda"

        if self.config.device == "cpu":
            logger.info("CPU explicitly requested. Using CPU.")
            if self.env_context.is_ci:
                logger.info(
                    "CI environment detected with explicit CPU device configuration."
                )
            return "cpu"

        logger.warning(  # type: ignore[unreachable]
            f"Invalid device '{self.config.device}' requested. Falling back to CPU."
        )
        return "cpu"

    def _auto_detect_dimension(self) -> int:
        """Auto-detect the model's output dimension."""
        if not self.model:
            raise EmbeddingProviderError(
                "Cannot auto-detect dimension, model not loaded."
            )

        try:
            # Try to use the model's built-in dimension method if available
            if hasattr(self.model, "get_sentence_embedding_dimension"):
                dimension = int(self.model.get_sentence_embedding_dimension())
                logger.info(f"Auto-detected dimension using model method: {dimension}")
                return dimension

            # Fallback: encode a test string and measure output
            test_embedding = self.model.encode("test", convert_to_numpy=False)
            dimension = int(len(test_embedding))
            logger.info(f"Auto-detected dimension using test encoding: {dimension}")
            return dimension

        except Exception as e:
            logger.error(
                f"Failed to auto-detect dimension for '{self.config.model_name}': {e}",
                exc_info=True,
            )
            raise EmbeddingProviderError(
                f"Failed to auto-detect dimension for '{self.config.model_name}'"
            ) from e

    def _validate_dimension(self) -> None:
        if not self.model:
            raise EmbeddingProviderError("Cannot validate dimension, model not loaded.")

        # Auto-detect dimension if not configured
        if self.config.dimension is None:
            self._auto_detected_dimension = self._auto_detect_dimension()
            logger.info(
                f"Auto-detected dimension for model '{self.config.model_name}': {self._auto_detected_dimension}"
            )
            return

        # Validate explicit dimension configuration
        try:
            test_embedding = self.model.encode("test", convert_to_numpy=False)
            actual_dimension = len(test_embedding)

            if actual_dimension != self.config.dimension:
                raise EmbeddingProviderError(
                    f"Model '{self.config.model_name}' output dimension ({actual_dimension}) "
                    f"does not match configured dimension ({self.config.dimension}). "
                    "Please ensure EMBEDDING_DIMENSION is set correctly for the chosen model."
                )
            logger.info(
                f"Model dimension validated: {actual_dimension} == {self.config.dimension}"
            )
        except Exception as e:
            logger.error(
                f"Failed during model dimension validation for '{self.config.model_name}': {e}",
                exc_info=True,
            )
            self.model = None
            raise EmbeddingProviderError(
                f"Failed during model dimension validation for '{self.config.model_name}'"
            ) from e

    def _ensure_model_loaded(self) -> None:
        """Load the model on first use if not already loaded."""
        if self.model is not None:
            return  # Model already loaded

        if self.dev_mode:
            self.dev_context.log_skip(
                f"sentence-transformer model '{self.config.model_name}' loading"
            )
            return  # Skip model loading in dev mode

        logger.info(
            f"Loading sentence-transformer model '{self.config.model_name}' onto device '{self.device}' "
            f"with cache directory: {self.cache_dir}"
        )

        # Log initial GPU memory usage if using CUDA
        if self.device == "cuda":
            _log_gpu_memory_usage("pre-model loading")

        try:
            # Lazy load SentenceTransformer class
            SentenceTransformer = _get_sentence_transformer_class()

            # Apply GPU-specific optimizations for large models
            model_kwargs = {"device": self.device}
            if self.device == "cuda" and _check_torch_available() and _torch:
                try:
                    # Enable memory-efficient loading for large models on GPU
                    if _torch.cuda.is_available():
                        # Clear cache before loading to maximize available memory
                        _torch.cuda.empty_cache()
                        _log_gpu_memory_usage("after CUDA cache clear")

                        logger.info(
                            "Applied GPU memory optimizations for model loading"
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not apply GPU optimizations, proceeding with standard loading: {e}"
                    )

            self.model = SentenceTransformer(
                self.config.model_name,
                cache_folder=str(self.cache_dir),
                **model_kwargs,
            )

            # Log memory usage after model loading
            if self.device == "cuda":
                _log_gpu_memory_usage("model loading")

            logger.info(
                f"Successfully loaded sentence-transformer model '{self.config.model_name}'."
            )

            # Perform GPU warmup for better performance
            if self.device == "cuda":
                _gpu_warmup(self.model, self.device)

            self._validate_dimension()

        except Exception as e:
            # Enhanced error handling for GPU-related issues
            error_msg = (
                f"Failed to load sentence-transformer model '{self.config.model_name}'"
            )

            if self.device == "cuda" and _check_torch_available() and _torch:
                try:
                    if _torch.cuda.is_available():
                        # Log GPU memory state for debugging
                        _log_gpu_memory_usage("model loading failure")

                        # Check for common GPU issues
                        if (
                            "out of memory" in str(e).lower()
                            or "cuda out of memory" in str(e).lower()
                        ):
                            error_msg += " (GPU out of memory - consider using a smaller model or CPU device)"
                        elif "cuda" in str(e).lower():
                            error_msg += (
                                " (CUDA error - check GPU availability and drivers)"
                            )
                except Exception:
                    pass  # Don't let GPU debugging interfere with original error

            logger.error(
                f"{error_msg} from source: {e}",
                exc_info=True,
            )
            self.model = None
            raise EmbeddingProviderError(
                f"{error_msg}. "
                f"Ensure the model name is correct and accessible. Original error: {e}"
            ) from e

    def initialize(self) -> None:
        """Initialize lightweight configuration only - no model loading."""
        if not self.config.model_name:
            raise EmbeddingProviderError(
                "Initialization failed: No embedding model name provided in configuration (EMBEDDING_MODEL_NAME)."
            )
        # Allow dimension=None for auto-detection
        if self.config.dimension is None:
            logger.info(
                "No dimension configured - will auto-detect from model output dimension."
            )

        if self.dev_mode:
            # Minimal setup for dev mode
            self.device = "cpu"  # Use CPU for consistency in dev mode
            logger.info(
                f"SentenceTransformerProvider initialized in DEV MODE. "
                f"Model '{self.config.model_name}' will NOT be loaded."
            )
            return

        # Lightweight setup only - no model loading
        self.device = self._determine_device()

        # Set up cache directory for model storage
        from pathlib import Path

        self.cache_dir = Path.home() / ".cache" / "confluence-gateway" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"SentenceTransformerProvider initialized with model '{self.config.model_name}' for device '{self.device}'. "
            f"Model will be loaded on first use."
        )

    def _check_initialization(self) -> None:
        if not hasattr(self, "device") or self.device is None:
            raise EmbeddingProviderError(
                "SentenceTransformerProvider is not initialized. Call initialize() first."
            )
        # Allow dimension=None for auto-detection, will be validated when model loads
        pass

    def _validate_embedding(
        self, embedding: Any, index: int | None = None
    ) -> list[float]:
        expected_dimension = self.get_dimension()
        if not isinstance(embedding, list) or len(embedding) != expected_dimension:
            index_info = f" at index {index}" if index is not None else ""
            logger.error(
                f"Unexpected embedding format{index_info}: Expected {expected_dimension}D list, "
                f"got {type(embedding)} with length {len(embedding) if isinstance(embedding, list) else 'N/A'}."
            )
            raise EmbeddingProviderError(
                f"Unexpected embedding format received from model{index_info}."
            )
        return embedding

    def embed_text(self, text: str) -> list[float]:
        if not text or not isinstance(text, str):
            logger.warning(
                "Received empty or invalid text for embedding, returning empty list."
            )
            return []

        if self.dev_mode:
            # Return stub embedding in dev mode
            import random

            random.seed(hash(text) % (2**32))  # Deterministic but varied
            dimension = self.config.dimension or self._auto_detected_dimension or 384
            stub_embedding = [random.uniform(-1.0, 1.0) for _ in range(dimension)]
            logger.debug(
                f"DEV MODE: Generated stub embedding for text: '{text[:30]}...'"
            )
            return stub_embedding

        self._check_initialization()
        self._ensure_model_loaded()
        assert self.model is not None

        try:
            embedding = self.model.encode(text, convert_to_numpy=False).tolist()
            return self._validate_embedding(embedding)
        except EmbeddingProviderError:
            raise
        except Exception as e:
            # Enhanced GPU error handling for embedding operations
            error_msg = f"Failed to embed text using model '{self.config.model_name}'"

            if self.device == "cuda" and _check_torch_available() and _torch:
                try:
                    if _torch.cuda.is_available():
                        # Check for GPU-specific errors
                        if (
                            "out of memory" in str(e).lower()
                            or "cuda out of memory" in str(e).lower()
                        ):
                            error_msg += " (GPU out of memory during embedding - consider reducing batch size or using CPU)"
                            _log_gpu_memory_usage("embedding failure")
                        elif "cuda" in str(e).lower():
                            error_msg += (
                                " (CUDA error during embedding - check GPU state)"
                            )
                            _log_gpu_memory_usage("embedding failure")
                except Exception:
                    pass  # Don't let GPU debugging interfere with original error

            logger.error(
                f"Error during single text embedding with model '{self.config.model_name}': {e}",
                exc_info=True,
            )
            raise EmbeddingProviderError(error_msg) from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning(
                "Received empty list for batch embedding, returning empty list."
            )
            return []

        if self.dev_mode:
            # Return stub embeddings in dev mode
            import random

            stub_embeddings = []
            for text in texts:
                if text and isinstance(text, str):
                    random.seed(hash(text) % (2**32))  # Deterministic but varied
                    dimension = (
                        self.config.dimension or self._auto_detected_dimension or 384
                    )
                    stub_embedding = [
                        random.uniform(-1.0, 1.0) for _ in range(dimension)
                    ]
                    stub_embeddings.append(stub_embedding)
            logger.debug(
                f"DEV MODE: Generated {len(stub_embeddings)} stub embeddings for batch"
            )
            return stub_embeddings

        self._check_initialization()
        self._ensure_model_loaded()
        assert self.model is not None

        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if not valid_texts:
            logger.warning(
                "All texts in the batch were empty or invalid, returning empty list."
            )
            return []

        if len(valid_texts) < len(texts):
            logger.warning(
                f"Filtered out {len(texts) - len(valid_texts)} empty/invalid strings from batch embedding request."
            )

        try:
            embedding_tensors = self.model.encode(
                valid_texts,
                convert_to_numpy=False,
                show_progress_bar=False,
            )

            if not isinstance(embedding_tensors, list):
                logger.error(
                    f"Model returned non-list output ({type(embedding_tensors).__name__}) for batch embedding when expecting list[Tensor]."
                )
                raise EmbeddingProviderError(
                    "Unexpected batch embedding format received from model."
                )

            validated_embeddings = []
            for i, tensor in enumerate(embedding_tensors):
                if not (hasattr(tensor, "tolist") and callable(tensor.tolist)):
                    logger.error(
                        f"Item at index {i} in embedding result is not a Tensor, but {type(tensor).__name__}."
                    )
                    raise EmbeddingProviderError(
                        f"Unexpected item type in batch embedding result at index {i}."
                    )
                embedding_list = tensor.tolist()
                validated_embeddings.append(self._validate_embedding(embedding_list, i))

            return validated_embeddings

        except EmbeddingProviderError:
            raise
        except Exception as e:
            # Enhanced GPU error handling for batch embedding operations
            error_msg = (
                f"Failed to embed batch of texts using model '{self.config.model_name}'"
            )

            if self.device == "cuda" and _check_torch_available() and _torch:
                try:
                    if _torch.cuda.is_available():
                        # Check for GPU-specific errors
                        if (
                            "out of memory" in str(e).lower()
                            or "cuda out of memory" in str(e).lower()
                        ):
                            batch_size = (
                                len(valid_texts)
                                if "valid_texts" in locals()
                                else len(texts)
                            )
                            error_msg += f" (GPU out of memory with batch size {batch_size} - consider reducing batch size or using CPU)"
                            _log_gpu_memory_usage("batch embedding failure")
                        elif "cuda" in str(e).lower():
                            error_msg += (
                                " (CUDA error during batch embedding - check GPU state)"
                            )
                            _log_gpu_memory_usage("batch embedding failure")
                except Exception:
                    pass  # Don't let GPU debugging interfere with original error

            logger.error(
                f"Original error during batch text embedding with model '{self.config.model_name}': {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise EmbeddingProviderError(f"{error_msg} due to provider error.") from e

    def get_dimension(self) -> int:
        # Return configured dimension if available
        if self.config.dimension is not None:
            return self.config.dimension

        # For auto-detection case, handle dev mode first
        if self.dev_mode:
            # In dev mode, return a default dimension
            return 384

        # Return auto-detected dimension if already available
        if self._auto_detected_dimension is not None:
            return self._auto_detected_dimension

        # Need to load model to auto-detect dimension
        self._ensure_model_loaded()

        # Return the auto-detected dimension (should be set after model loading)
        return (
            self._auto_detected_dimension or 384
        )  # Fallback to default if somehow not set

    def close(self) -> None:
        logger.info(
            f"Closing SentenceTransformerProvider for model '{self.config.model_name}'."
        )
        if self.model:
            # Log GPU memory usage before cleanup if using CUDA
            if self.device == "cuda":
                _log_gpu_memory_usage("before model cleanup")

            del self.model
            self.model = None
            logger.debug("SentenceTransformer model reference removed.")

            if (
                self.device == "cuda"
                and _torch_available
                and _torch
                and hasattr(_torch.cuda, "empty_cache")
            ):
                try:
                    # Comprehensive GPU cleanup
                    _torch.cuda.empty_cache()
                    logger.debug("Cleared PyTorch CUDA cache.")

                    # Additional GPU cleanup if available
                    if hasattr(_torch.cuda, "synchronize"):
                        _torch.cuda.synchronize()
                        logger.debug("Synchronized CUDA operations.")

                    # Log memory usage after cleanup
                    _log_gpu_memory_usage("after GPU cleanup")

                except Exception as e:
                    logger.warning(
                        f"Failed to perform complete GPU cleanup: {e}", exc_info=True
                    )
        else:
            logger.debug("No model loaded, nothing to close.")
