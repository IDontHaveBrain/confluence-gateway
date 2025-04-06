import logging
from typing import TYPE_CHECKING

try:
    import litellm
    from litellm.exceptions import (
        APIConnectionError,
        AuthenticationError,
        BadRequestError,
        RateLimitError,
        ServiceUnavailableError,
    )
except ImportError:
    raise ImportError(
        "The 'litellm' library is required for LiteLLMProvider. "
        "Please install it using 'pip install litellm'."
    )

from confluence_gateway.adapters.embedding.base import EmbeddingProvider
from confluence_gateway.core.exceptions import EmbeddingProviderError

if TYPE_CHECKING:
    from confluence_gateway.core.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class LiteLLMProvider(EmbeddingProvider):
    def __init__(self, config: "EmbeddingConfig") -> None:
        super().__init__(config)
        logger.info(
            f"LiteLLMProvider initialized with config: "
            f"Model='{self.config.model_name}', Dimension='{self.config.dimension}', "
            f"Base URL='{self.config.litellm_api_base}', API Key Provided={'Yes' if self.config.litellm_api_key else 'No'}"
        )
        if not self.config.model_name:
            raise EmbeddingProviderError(
                "LiteLLMProvider requires a model name in the configuration."
            )
        if self.config.dimension is None:
            raise EmbeddingProviderError(
                "LiteLLMProvider requires an embedding dimension in the configuration."
            )

    def initialize(self) -> None:
        logger.info(
            f"Initializing LiteLLM provider for model: {self.config.model_name}"
        )

        original_key = litellm.api_key
        original_base = litellm.api_base

        try:
            # Configure LiteLLM global settings
            litellm.api_key = self.config.litellm_api_key
            logger.debug(
                f"LiteLLM API key {'set from config' if self.config.litellm_api_key else 'not provided'}"
            )

            litellm.api_base = (
                str(self.config.litellm_api_base)
                if self.config.litellm_api_base
                else None
            )
            logger.debug(
                f"LiteLLM API base {'set to: ' + str(litellm.api_base) if litellm.api_base else 'not provided'}"
            )

            # Validate with a test call
            test_text = "validate provider initialization"
            logger.debug(
                f"Performing test embedding call with model '{self.config.model_name}'..."
            )

            response = litellm.embedding(
                model=self.config.model_name, input=[test_text]
            )
            logger.debug("Test embedding call successful.")

            try:
                response_data = self._validate_embedding_response(response)
                embedding = self._extract_embedding_from_item(response_data[0])

                logger.info(
                    f"LiteLLM provider initialized and dimension ({len(embedding)}D) validated "
                    f"successfully for model '{self.config.model_name}'."
                )
            except EmbeddingProviderError as e:
                raise EmbeddingProviderError(
                    f"Test embedding validation failed: {str(e)}"
                )

        except (
            AuthenticationError,
            APIConnectionError,
            BadRequestError,
            RateLimitError,
            ServiceUnavailableError,
            Exception,
        ) as e:
            error_message = f"Failed to initialize LiteLLM provider for model '{self.config.model_name}'. Error: {type(e).__name__}: {e}"
            logger.error(error_message, exc_info=True)

            # Restore original settings
            litellm.api_key = original_key
            litellm.api_base = original_base
            logger.debug(
                "Restored original global LiteLLM settings after initialization failure."
            )

            raise EmbeddingProviderError(error_message) from e

    def _validate_embedding_response(self, response, expected_count=1):
        """Validate embedding response format and structure."""
        if (
            not response.data
            or not isinstance(response.data, list)
            or len(response.data) != expected_count
        ):
            raise EmbeddingProviderError(
                f"LiteLLM embedding response data mismatch. Expected {expected_count} embeddings, "
                f"got {len(response.data) if response.data else 0}."
            )

        return response.data

    def _extract_embedding_from_item(self, item, index=None):
        """Extract and validate a single embedding from a response item."""
        index_info = f" at index {index}" if index is not None else ""

        if not isinstance(item, dict) or "embedding" not in item:
            raise EmbeddingProviderError(
                f"LiteLLM response missing 'embedding' key{index_info}."
            )

        embedding = item["embedding"]
        if not isinstance(embedding, list):
            raise EmbeddingProviderError(
                f"LiteLLM response 'embedding' is not a list{index_info}."
            )

        if len(embedding) != self.config.dimension:
            logger.warning(
                f"LiteLLM returned embedding dimension {len(embedding)}, expected {self.config.dimension}. Check model consistency."
            )
            raise EmbeddingProviderError(
                f"Dimension mismatch{index_info}: Expected {self.config.dimension}, got {len(embedding)}"
            )

        return embedding

    def _check_configuration(self):
        """Verify the provider is properly configured."""
        if not self.config.model_name or self.config.dimension is None:
            raise EmbeddingProviderError(
                "LiteLLM provider is not properly configured (missing model name or dimension)."
            )

    def embed_text(self, text: str) -> list[float]:
        if not text or not isinstance(text, str):
            logger.warning(
                "Received empty or invalid text for embedding, returning empty list."
            )
            return []

        self._check_configuration()

        try:
            response = litellm.embedding(model=self.config.model_name, input=[text])
            response_data = self._validate_embedding_response(response)
            return self._extract_embedding_from_item(response_data[0])

        except EmbeddingProviderError:
            raise
        except Exception as e:
            logger.error(
                f"Error during single text embedding with LiteLLM model '{self.config.model_name}': {e}",
                exc_info=True,
            )
            raise EmbeddingProviderError(
                f"LiteLLM failed to embed text using model '{self.config.model_name}'"
            ) from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning(
                "Received empty list for batch embedding, returning empty list."
            )
            return []

        self._check_configuration()

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
            response = litellm.embedding(
                model=self.config.model_name, input=valid_texts
            )
            response_data = self._validate_embedding_response(
                response, expected_count=len(valid_texts)
            )

            return [
                self._extract_embedding_from_item(item, i)
                for i, item in enumerate(response_data)
            ]

        except EmbeddingProviderError:
            raise
        except Exception as e:
            logger.error(
                f"Error during batch text embedding with LiteLLM model '{self.config.model_name}': {e}",
                exc_info=True,
            )
            raise EmbeddingProviderError(
                f"LiteLLM failed to embed batch of texts using model '{self.config.model_name}'"
            ) from e

    def get_dimension(self) -> int:
        if self.config.dimension is None:
            raise EmbeddingProviderError(
                "Embedding dimension is not configured for the LiteLLM provider."
            )
        return self.config.dimension

    def close(self) -> None:
        logger.info(f"Closing LiteLLMProvider for model '{self.config.model_name}'.")
        pass
