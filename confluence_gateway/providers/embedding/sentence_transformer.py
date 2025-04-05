import logging
from types import ModuleType
from typing import TYPE_CHECKING, Optional

from confluence_gateway.providers.embedding.base import (
    EmbeddingProvider,
    EmbeddingProviderError,
)

torch: Optional[ModuleType] = None
try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "The 'sentence-transformers' library is required for SentenceTransformerProvider. "
        "Please install it using 'pip install sentence-transformers'."
    )

if TYPE_CHECKING:
    from confluence_gateway.core.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, config: "EmbeddingConfig") -> None:
        super().__init__(config)
        self.model: Optional[SentenceTransformer] = None
        self.device: Optional[str] = None
        logger.info(
            f"SentenceTransformerProvider initialized with config: "
            f"Model='{self.config.model_name}', Dimension='{self.config.dimension}', "
            f"Requested Device='{self.config.device or 'auto'}'"
        )

    def _determine_device(self) -> str:
        requested_device = self.config.device

        if requested_device:
            if requested_device == "cuda":
                if not _torch_available:
                    logger.warning(
                        "Torch library not found. Cannot use CUDA. Falling back to CPU."
                    )
                    return "cpu"
                if not torch or not torch.cuda.is_available():
                    logger.warning(
                        "CUDA requested but not available (or torch check failed). Falling back to CPU."
                    )
                    return "cpu"
                logger.info("CUDA requested and available. Using CUDA.")
                return "cuda"
            elif requested_device == "cpu":
                logger.info("CPU explicitly requested. Using CPU.")
                return "cpu"
            else:
                logger.warning(
                    f"Invalid device '{requested_device}' requested. Falling back to CPU."
                )
                return "cpu"
        else:
            if _torch_available and torch and torch.cuda.is_available():
                logger.info("Auto-detected CUDA availability. Using CUDA.")
                return "cuda"
            else:
                logger.info(
                    "Auto-detection: CUDA not available or torch not installed. Using CPU."
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

    def initialize(self) -> None:
        if not self.config.model_name:
            raise EmbeddingProviderError(
                "Initialization failed: No embedding model name provided in configuration (EMBEDDING_MODEL_NAME)."
            )
        if self.config.dimension is None:
            raise EmbeddingProviderError(
                "Initialization failed: No embedding dimension provided in configuration (EMBEDDING_DIMENSION)."
            )

        self.device = self._determine_device()
        logger.info(
            f"Attempting to load sentence-transformer model '{self.config.model_name}' onto device '{self.device}'..."
        )

        try:
            self.model = SentenceTransformer(self.config.model_name, device=self.device)
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

    def embed_text(self, text: str) -> list[float]:
        if not self.model:
            raise EmbeddingProviderError(
                "SentenceTransformerProvider is not initialized. Call initialize() first."
            )
        if not self.config.dimension:
            raise EmbeddingProviderError("Configuration dimension is missing.")

        if not text or not isinstance(text, str):
            logger.warning(
                "Received empty or invalid text for embedding, returning empty list."
            )
            return []

        try:
            embedding: list[float] = self.model.encode(
                text, convert_to_numpy=False
            ).tolist()

            if (
                not isinstance(embedding, list)
                or len(embedding) != self.config.dimension
            ):
                logger.error(
                    f"Unexpected embedding format or dimension returned by model. "
                    f"Expected {self.config.dimension}D list[float], got {type(embedding)} "
                    f"with length {len(embedding) if isinstance(embedding, list) else 'N/A'}."
                )
                raise EmbeddingProviderError(
                    "Unexpected embedding format received from model."
                )

            return embedding
        except Exception as e:
            logger.error(
                f"Error during single text embedding with model '{self.config.model_name}': {e}",
                exc_info=True,
            )
            raise EmbeddingProviderError(
                f"Failed to embed text using model '{self.config.model_name}'"
            ) from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.model:
            raise EmbeddingProviderError(
                "SentenceTransformerProvider is not initialized. Call initialize() first."
            )
        if not self.config.dimension:
            raise EmbeddingProviderError("Configuration dimension is missing.")

        if not texts:
            logger.warning(
                "Received empty list for batch embedding, returning empty list."
            )
            return []

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
            embeddings: list[list[float]] = self.model.encode(
                valid_texts,
                convert_to_numpy=False,
                show_progress_bar=False,
            ).tolist()

            if not isinstance(embeddings, list) or not all(
                isinstance(emb, list) and len(emb) == self.config.dimension
                for emb in embeddings
            ):
                logger.error(
                    f"Unexpected batch embedding format or dimension returned by model. "
                    f"Expected list[list[float]] with inner dim {self.config.dimension}, "
                    f"got {type(embeddings)}."
                )
                raise EmbeddingProviderError(
                    "Unexpected batch embedding format received from model."
                )

            return embeddings
        except Exception as e:
            logger.error(
                f"Error during batch text embedding with model '{self.config.model_name}': {e}",
                exc_info=True,
            )
            raise EmbeddingProviderError(
                f"Failed to embed batch of texts using model '{self.config.model_name}'"
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
                and torch
                and hasattr(torch.cuda, "empty_cache")
            ):
                try:
                    torch.cuda.empty_cache()
                    logger.debug("Cleared PyTorch CUDA cache.")
                except Exception as e:
                    logger.warning(
                        f"Failed to clear PyTorch CUDA cache: {e}", exc_info=True
                    )
        else:
            logger.debug("No model loaded, nothing to close.")
