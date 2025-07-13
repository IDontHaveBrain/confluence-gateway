import logging
from types import ModuleType
from typing import TYPE_CHECKING, Any

from confluence_gateway.adapters.embedding.base import EmbeddingProvider
from confluence_gateway.core.config import (
    dev_mode_log_skip,
    dev_mode_log_stub,
    is_dev_mode,
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


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, config: "EmbeddingConfig") -> None:
        super().__init__(config)
        self.model: Any | None = None
        self.device: str | None = None
        self.cache_dir: Any | None = None
        self.dev_mode = is_dev_mode()

        if self.dev_mode:
            dev_mode_log_stub("SentenceTransformerProvider")
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
        if not self.config.device:
            if _check_torch_available():
                torch = _torch
                if torch and torch.cuda.is_available():
                    logger.info("Auto-detected CUDA availability. Using CUDA.")
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
            return "cuda"

        if self.config.device == "cpu":
            logger.info("CPU explicitly requested. Using CPU.")
            return "cpu"

        logger.warning(  # type: ignore[unreachable]
            f"Invalid device '{self.config.device}' requested. Falling back to CPU."
        )
        return "cpu"

    def _validate_dimension(self) -> None:
        if not self.model:
            raise EmbeddingProviderError("Cannot validate dimension, model not loaded.")
        if self.config.dimension is None:
            raise EmbeddingProviderError(
                "Cannot validate dimension, expected dimension not configured."
            )

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
            dev_mode_log_skip(
                f"sentence-transformer model '{self.config.model_name}' loading"
            )
            return  # Skip model loading in dev mode

        logger.info(
            f"Loading sentence-transformer model '{self.config.model_name}' onto device '{self.device}' "
            f"with cache directory: {self.cache_dir}"
        )

        try:
            # Lazy load SentenceTransformer class
            SentenceTransformer = _get_sentence_transformer_class()
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.device,
                cache_folder=str(self.cache_dir),
            )
            logger.info(
                f"Successfully loaded sentence-transformer model '{self.config.model_name}'."
            )

            self._validate_dimension()

        except Exception as e:
            logger.error(
                f"Failed to load sentence-transformer model '{self.config.model_name}' from source: {e}",
                exc_info=True,
            )
            self.model = None
            raise EmbeddingProviderError(
                f"Could not load sentence-transformer model '{self.config.model_name}'. "
                f"Ensure the model name is correct and accessible. Original error: {e}"
            ) from e

    def initialize(self) -> None:
        """Initialize lightweight configuration only - no model loading."""
        if not self.config.model_name:
            raise EmbeddingProviderError(
                "Initialization failed: No embedding model name provided in configuration (EMBEDDING_MODEL_NAME)."
            )
        if self.config.dimension is None:
            raise EmbeddingProviderError(
                "Initialization failed: No embedding dimension provided in configuration (EMBEDDING_DIMENSION)."
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
        if not self.config.dimension:
            raise EmbeddingProviderError("Configuration dimension is missing.")

    def _validate_embedding(
        self, embedding: Any, index: int | None = None
    ) -> list[float]:
        if not isinstance(embedding, list) or len(embedding) != self.config.dimension:
            index_info = f" at index {index}" if index is not None else ""
            logger.error(
                f"Unexpected embedding format{index_info}: Expected {self.config.dimension}D list, "
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
            stub_embedding = [
                random.uniform(-1.0, 1.0) for _ in range(self.config.dimension or 384)
            ]
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
            logger.error(
                f"Error during single text embedding with model '{self.config.model_name}': {e}",
                exc_info=True,
            )
            raise EmbeddingProviderError(
                f"Failed to embed text using model '{self.config.model_name}'"
            ) from e

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
                    stub_embedding = [
                        random.uniform(-1.0, 1.0)
                        for _ in range(self.config.dimension or 384)
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
            logger.error(
                f"Original error during batch text embedding with model '{self.config.model_name}': {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise EmbeddingProviderError(
                f"Failed to embed batch of texts using model '{self.config.model_name}' due to provider error."
            ) from e

    def get_dimension(self) -> int:
        if self.config.dimension is None:
            raise EmbeddingProviderError(
                "Embedding dimension is not configured for this provider."
            )
        return self.config.dimension

    def close(self) -> None:
        logger.info(
            f"Closing SentenceTransformerProvider for model '{self.config.model_name}'."
        )
        if self.model:
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
                    _torch.cuda.empty_cache()
                    logger.debug("Cleared PyTorch CUDA cache.")
                except Exception as e:
                    logger.warning(
                        f"Failed to clear PyTorch CUDA cache: {e}", exc_info=True
                    )
        else:
            logger.debug("No model loaded, nothing to close.")
