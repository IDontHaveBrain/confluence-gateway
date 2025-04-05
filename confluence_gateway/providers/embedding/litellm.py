import logging
from typing import TYPE_CHECKING, Optional

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


from confluence_gateway.providers.embedding.base import (
    EmbeddingProvider,
    EmbeddingProviderError,
)

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
            if self.config.litellm_api_key:
                litellm.api_key = self.config.litellm_api_key
                logger.debug("LiteLLM API key set globally from provider config.")
            else:
                litellm.api_key = None
                logger.debug(
                    "LiteLLM API key set globally to None (not provided in config)."
                )

            if self.config.litellm_api_base:
                litellm.api_base = str(self.config.litellm_api_base)
                logger.debug(f"LiteLLM API base set globally to: {litellm.api_base}")
            else:
                litellm.api_base = None
                logger.debug(
                    "LiteLLM API base set globally to None (not provided in config)."
                )

            test_text = "validate provider initialization"
            logger.debug(
                f"Performing test embedding call with model '{self.config.model_name}'..."
            )

            response = litellm.embedding(
                model=self.config.model_name, input=[test_text]
            )
            logger.debug("Test embedding call successful.")

            if (
                not response.data
                or not isinstance(response.data, list)
                or len(response.data) == 0
            ):
                raise EmbeddingProviderError(
                    "Test embedding call returned empty or invalid data structure."
                )

            first_item = response.data[0]
            if not isinstance(first_item, dict) or "embedding" not in first_item:
                raise EmbeddingProviderError(
                    "Test embedding call response missing 'embedding' key in data item."
                )

            first_embedding = first_item["embedding"]
            if not first_embedding or not isinstance(first_embedding, list):
                raise EmbeddingProviderError(
                    "Test embedding call response 'embedding' is not a valid list."
                )

            actual_dimension = len(first_embedding)
            if actual_dimension != self.config.dimension:
                raise EmbeddingProviderError(
                    f"Model '{self.config.model_name}' output dimension ({actual_dimension}) "
                    f"does not match configured dimension ({self.config.dimension}). "
                    "Please ensure EMBEDDING_DIMENSION is set correctly for the chosen model."
                )

            logger.info(
                f"LiteLLM provider initialized and dimension ({actual_dimension}D) validated successfully for model '{self.config.model_name}'."
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

            litellm.api_key = original_key
            litellm.api_base = original_base
            logger.debug(
                "Restored original global LiteLLM settings after initialization failure."
            )

            raise EmbeddingProviderError(error_message) from e

    def embed_text(self, text: str) -> list[float]:
        if not text or not isinstance(text, str):
            logger.warning(
                "Received empty or invalid text for embedding, returning empty list."
            )
            return []
        if not self.config.model_name or self.config.dimension is None:
            raise EmbeddingProviderError(
                "LiteLLM provider is not properly configured (missing model name or dimension)."
            )

        try:
            response = litellm.embedding(model=self.config.model_name, input=[text])

            if (
                not response.data
                or not isinstance(response.data, list)
                or len(response.data) == 0
            ):
                raise EmbeddingProviderError(
                    "LiteLLM embedding response missing or invalid data structure."
                )

            first_item = response.data[0]
            if not isinstance(first_item, dict) or "embedding" not in first_item:
                raise EmbeddingProviderError(
                    "LiteLLM embedding response missing 'embedding' key in data item."
                )

            embedding = first_item["embedding"]
            if not isinstance(embedding, list):
                raise EmbeddingProviderError(
                    "LiteLLM embedding response 'embedding' is not a valid list."
                )

            if len(embedding) != self.config.dimension:
                logger.warning(
                    f"LiteLLM returned embedding dimension {len(embedding)}, expected {self.config.dimension}. Check model consistency."
                )
                raise EmbeddingProviderError(
                    f"Dimension mismatch: Expected {self.config.dimension}, got {len(embedding)}"
                )

            return embedding

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
        if not self.config.model_name or self.config.dimension is None:
            raise EmbeddingProviderError(
                "LiteLLM provider is not properly configured (missing model name or dimension)."
            )

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

            if (
                not response.data
                or not isinstance(response.data, list)
                or len(response.data) != len(valid_texts)
            ):
                raise EmbeddingProviderError(
                    f"LiteLLM batch embedding response data mismatch or missing. "
                    f"Expected {len(valid_texts)} embeddings, got {len(response.data) if response.data else 0}."
                )

            embeddings: list[Optional[list[float]]] = []
            for i, item in enumerate(response.data):
                if not isinstance(item, dict) or "embedding" not in item:
                    raise EmbeddingProviderError(
                        f"LiteLLM batch response missing 'embedding' key in item {i}."
                    )
                embedding = item["embedding"]
                if not isinstance(embedding, list):
                    raise EmbeddingProviderError(
                        f"LiteLLM batch response 'embedding' is not a list in item {i}."
                    )
                embeddings.append(embedding)

            final_embeddings: list[list[float]] = []
            for i, emb in enumerate(embeddings):
                if emb is None:
                    raise EmbeddingProviderError(
                        f"Missing embedding for text at index {i} in the batch."
                    )
                if len(emb) != self.config.dimension:
                    logger.error(
                        f"Unexpected batch embedding dimension from LiteLLM model '{self.config.model_name}' at index {i}. Expected {self.config.dimension}, got {len(emb)}."
                    )
                    raise EmbeddingProviderError(
                        f"Unexpected batch embedding dimension received from LiteLLM at index {i}."
                    )
                final_embeddings.append(emb)

            return final_embeddings

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
        logger.info(
            f"Closing LiteLLMProvider for model '{self.config.model_name}'. (No specific action taken)"
        )
        pass
