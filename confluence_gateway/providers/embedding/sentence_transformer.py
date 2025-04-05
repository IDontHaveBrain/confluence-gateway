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
    torch = None  # Assign None if torch is not available

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
    """
    Embedding provider using the sentence-transformers library for local model execution.
    """

    def __init__(self, config: "EmbeddingConfig") -> None:
        """
        Initialize the SentenceTransformerProvider.

        Args:
            config: The EmbeddingConfig object containing settings.
        """
        super().__init__(config)
        self.model: Optional[SentenceTransformer] = None
        self.device: Optional[str] = None
        logger.info(
            f"SentenceTransformerProvider initialized with config: "
            f"Model='{self.config.model_name}', Dimension='{self.config.dimension}', "
            f"Requested Device='{self.config.device or 'auto'}'"
        )

    def _determine_device(self) -> str:
        """Determines the compute device (CPU or CUDA) based on config and availability."""
        requested_device = self.config.device

        if requested_device:
            if requested_device == "cuda":
                if not _torch_available:
                    logger.warning(
                        "Torch library not found. Cannot use CUDA. Falling back to CPU."
                    )
                    return "cpu"
                # Explicit check for torch helps mypy
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
                # Should be caught by Pydantic validation, but handle defensively
                logger.warning(
                    f"Invalid device '{requested_device}' requested. Falling back to CPU."
                )
                return "cpu"
        else:
            # Auto-detection
            # Explicit check for torch helps mypy
            if _torch_available and torch and torch.cuda.is_available():
                logger.info("Auto-detected CUDA availability. Using CUDA.")
                return "cuda"
            else:
                logger.info(
                    "Auto-detection: CUDA not available or torch not installed. Using CPU."
                )
                return "cpu"

    def _validate_dimension(self) -> None:
        """Validates the loaded model's output dimension against the configuration."""
        if not self.model:
            raise EmbeddingProviderError("Cannot validate dimension, model not loaded.")
        if self.config.dimension is None:
            # This should be caught by EmbeddingConfig validation, but check again
            raise EmbeddingProviderError(
                "Cannot validate dimension, expected dimension not configured."
            )

        try:
            # Encode a short test string
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
            # Clean up potentially loaded model if validation fails
            self.model = None
            raise EmbeddingProviderError(
                f"Failed during model dimension validation for '{self.config.model_name}'"
            ) from e

    def initialize(self) -> None:
        """
        Loads the sentence-transformer model and validates its dimension.

        Raises:
            EmbeddingProviderError: If model loading or dimension validation fails.
        """
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

            # Validate dimension after successful loading
            self._validate_dimension()

        except Exception as e:
            # Catch potential errors from SentenceTransformer loading (e.g., model not found, network issues)
            logger.error(
                f"Failed to load sentence-transformer model '{self.config.model_name}' from source: {e}",
                exc_info=True,
            )
            self.model = None  # Ensure model is None if loading fails
            raise EmbeddingProviderError(
                f"Could not load sentence-transformer model '{self.config.model_name}'. "
                f"Ensure the model name is correct and accessible. Original error: {e}"
            ) from e

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding for a single piece of text.

        Args:
            text: The input text string.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingProviderError: If the provider is not initialized or embedding fails.
        """
        if not self.model:
            raise EmbeddingProviderError(
                "SentenceTransformerProvider is not initialized. Call initialize() first."
            )
        if not self.config.dimension:
            raise EmbeddingProviderError(
                "Configuration dimension is missing."
            )  # Should not happen if initialized

        if not text or not isinstance(text, str):
            logger.warning(
                "Received empty or invalid text for embedding, returning empty list."
            )
            # Return empty list matching the expected dimension? Or just empty?
            # Let's return an empty list for now, consistent with previous behavior.
            return []
            # Alternative: raise ValueError("Input text cannot be empty or non-string")

        try:
            embedding: list[float] = self.model.encode(
                text, convert_to_numpy=False
            ).tolist()

            # Basic check on output format (should be guaranteed by validation, but good practice)
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
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: A list of input text strings.

        Returns:
            A list of lists of floats, where each inner list is an embedding vector.

        Raises:
            EmbeddingProviderError: If the provider is not initialized or embedding fails.
        """
        if not self.model:
            raise EmbeddingProviderError(
                "SentenceTransformerProvider is not initialized. Call initialize() first."
            )
        if not self.config.dimension:
            raise EmbeddingProviderError(
                "Configuration dimension is missing."
            )  # Should not happen if initialized

        if not texts:
            logger.warning(
                "Received empty list for batch embedding, returning empty list."
            )
            return []

        # Filter out empty or non-string inputs
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
                show_progress_bar=False,  # Disable progress bar for non-interactive use
            ).tolist()

            # Basic check on output format
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
        """
        Return the expected dimension of the embeddings generated by this provider.

        Returns:
            The embedding dimension as an integer.

        Raises:
            EmbeddingProviderError: If the dimension is not configured.
        """
        if self.config.dimension is None:
            # This should ideally be caught during initialization
            raise EmbeddingProviderError(
                "Embedding dimension is not configured for this provider."
            )
        return self.config.dimension

    def close(self) -> None:
        """
        Clean up resources used by the provider (release model from memory).
        """
        logger.info(
            f"Closing SentenceTransformerProvider for model '{self.config.model_name}'."
        )
        if self.model:
            # Remove reference to the model to allow garbage collection
            del self.model
            self.model = None
            logger.debug("SentenceTransformer model reference removed.")

            # If using CUDA, optionally clear cache (can be aggressive)
            # Explicit check for torch helps mypy
            if (
                self.device == "cuda"
                and _torch_available
                and torch  # Add check here
                and hasattr(torch.cuda, "empty_cache")
            ):
                try:
                    # Now mypy knows torch is not None here
                    torch.cuda.empty_cache()
                    logger.debug("Cleared PyTorch CUDA cache.")
                except Exception as e:
                    logger.warning(
                        f"Failed to clear PyTorch CUDA cache: {e}", exc_info=True
                    )
        else:
            logger.debug("No model loaded, nothing to close.")
