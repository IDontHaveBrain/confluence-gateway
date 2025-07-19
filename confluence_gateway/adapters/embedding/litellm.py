import logging
from typing import TYPE_CHECKING, Any

from confluence_gateway.adapters.embedding.base import EmbeddingProvider
from confluence_gateway.core.config import (
    get_development_context,
)
from confluence_gateway.core.exceptions import EmbeddingProviderError

_litellm: Any | None = None
_litellm_available: bool | None = None
_litellm_exceptions: Any | None = None
_EmbeddingResponse: Any | None = None

if TYPE_CHECKING:
    from confluence_gateway.core.config import EmbeddingConfig

logger = logging.getLogger(__name__)


def _get_litellm() -> Any:
    """Get litellm module, loading it if needed."""
    global _litellm, _litellm_available, _litellm_exceptions, _EmbeddingResponse
    if _litellm is None:
        try:
            import litellm
            from litellm.exceptions import (
                APIConnectionError,
                AuthenticationError,
                BadRequestError,
                RateLimitError,
                ServiceUnavailableError,
            )
            from litellm.utils import EmbeddingResponse

            _litellm = litellm
            _litellm_exceptions = {
                "APIConnectionError": APIConnectionError,
                "AuthenticationError": AuthenticationError,
                "BadRequestError": BadRequestError,
                "RateLimitError": RateLimitError,
                "ServiceUnavailableError": ServiceUnavailableError,
            }
            _EmbeddingResponse = EmbeddingResponse
            _litellm_available = True
        except ImportError:
            _litellm_available = False
            raise ImportError(
                "The 'litellm' library is required for LiteLLMProvider. "
                "Please install it using 'pip install litellm'."
            )
    return _litellm, _litellm_exceptions, _EmbeddingResponse


class LiteLLMProvider(EmbeddingProvider):
    def __init__(self, config: "EmbeddingConfig") -> None:
        super().__init__(config)
        self.dev_context = get_development_context()
        self.dev_mode = self.dev_context.enabled
        self._auto_detected_dimension: int | None = None

        if self.dev_mode:
            self.dev_context.log_stub("LiteLLMProvider")
            logger.info(
                f"LiteLLMProvider initialized in DEV MODE - stub implementation only. "
                f"Model='{self.config.model_name}', Dimension='{self.config.dimension}'"
            )
            return

        logger.info(
            f"LiteLLMProvider initialized with config: "
            f"Model='{self.config.model_name}', Dimension='{self.config.dimension}', "
            f"Base URL='{self.config.litellm_api_base}', API Key Provided={'Yes' if self.config.litellm_api_key else 'No'}"
        )
        if not self.config.model_name:
            raise EmbeddingProviderError(
                "LiteLLMProvider requires a model name in the configuration."
            )

        # Auto-detect dimension if not provided
        if self.config.dimension is None:
            logger.info(
                f"Dimension not specified for model '{self.config.model_name}', will auto-detect during initialization."
            )
            self._auto_detected_dimension = self._auto_detect_dimension()

    def _auto_detect_dimension(self) -> int:
        """Auto-detect embedding dimension by making a test embedding call."""
        logger.info(
            f"Auto-detecting embedding dimension for model: {self.config.model_name}"
        )

        try:
            litellm, litellm_exceptions, EmbeddingResponse = _get_litellm()

            test_text = "dimension detection test"
            logger.debug(
                f"Making test embedding call to auto-detect dimension for model '{self.config.model_name}'..."
            )

            response = litellm.embedding(
                model=self.config.model_name,
                input=[test_text],
                api_key=self.config.litellm_api_key,
                api_base=(
                    str(self.config.litellm_api_base)
                    if self.config.litellm_api_base
                    else None
                ),
            )

            # Extract dimension from response
            if (
                not response.data
                or not isinstance(response.data, list)
                or len(response.data) == 0
            ):
                raise EmbeddingProviderError(
                    f"Invalid response during dimension auto-detection for model '{self.config.model_name}'"
                )

            item = response.data[0]
            if not isinstance(item, dict) or "embedding" not in item:
                raise EmbeddingProviderError(
                    f"Invalid embedding response structure during dimension auto-detection for model '{self.config.model_name}'"
                )

            embedding = item["embedding"]
            if not isinstance(embedding, list):
                raise EmbeddingProviderError(
                    f"Embedding is not a list during dimension auto-detection for model '{self.config.model_name}'"
                )

            detected_dimension = len(embedding)
            logger.info(
                f"Auto-detected embedding dimension: {detected_dimension}D for model '{self.config.model_name}'"
            )
            return detected_dimension

        except Exception as e:
            error_message = f"Failed to auto-detect dimension for model '{self.config.model_name}'. Error: {type(e).__name__}: {e}"
            logger.error(error_message, exc_info=True)
            raise EmbeddingProviderError(error_message) from e

    def initialize(self) -> None:
        if self.dev_mode:
            self.dev_context.log_skip(
                f"LiteLLM provider initialization and test call for model '{self.config.model_name}'"
            )
            logger.info("LiteLLM provider initialized in DEV MODE - no test calls made")
            return

        logger.info(
            f"Initializing LiteLLM provider for model: {self.config.model_name}"
        )

        try:
            litellm, litellm_exceptions, EmbeddingResponse = _get_litellm()

            test_text = "validate provider initialization"
            logger.debug(
                f"Performing test embedding call with model '{self.config.model_name}'..."
            )

            response = litellm.embedding(
                model=self.config.model_name,
                input=[test_text],
                api_key=self.config.litellm_api_key,
                api_base=(
                    str(self.config.litellm_api_base)
                    if self.config.litellm_api_base
                    else None
                ),
            )
            logger.debug("Test embedding call successful.")

            try:
                response_data = self._validate_embedding_response(response)
                embedding = self._extract_embedding_from_item(response_data[0])

                expected_dim = self.get_dimension()
                logger.info(
                    f"LiteLLM provider initialized and dimension ({len(embedding)}D) validated "
                    f"successfully for model '{self.config.model_name}'. Expected: {expected_dim}D"
                )
            except EmbeddingProviderError as e:
                raise EmbeddingProviderError(
                    f"Test embedding validation failed: {str(e)}"
                )

        except Exception as e:
            error_message = f"Failed to initialize LiteLLM provider for model '{self.config.model_name}'. Error: {type(e).__name__}: {e}"
            logger.error(error_message, exc_info=True)
            raise EmbeddingProviderError(error_message) from e

    def _validate_embedding_response(
        self, response: Any, expected_count: int = 1
    ) -> list[Any]:
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

        return list(response.data)

    def _extract_embedding_from_item(
        self, item: dict[str, Any], index: int | None = None
    ) -> list[float]:
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

        expected_dimension = self.get_dimension()
        if len(embedding) != expected_dimension:
            logger.warning(
                f"LiteLLM returned embedding dimension {len(embedding)}, expected {expected_dimension}. Check model consistency."
            )
            raise EmbeddingProviderError(
                f"Dimension mismatch{index_info}: Expected {expected_dimension}, got {len(embedding)}"
            )

        return embedding

    def _check_configuration(self) -> None:
        """Verify the provider is properly configured."""
        if not self.config.model_name:
            raise EmbeddingProviderError(
                "LiteLLM provider is not properly configured (missing model name)."
            )
        if self.config.dimension is None and self._auto_detected_dimension is None:
            raise EmbeddingProviderError(
                "LiteLLM provider is not properly configured (no dimension configured or auto-detected)."
            )

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
                random.uniform(-1.0, 1.0) for _ in range(self.get_dimension())
            ]
            logger.debug(
                f"DEV MODE: Generated stub LiteLLM embedding for text: '{text[:30]}...'"
            )
            return stub_embedding

        self._check_configuration()

        try:
            litellm, _, _ = _get_litellm()

            response = litellm.embedding(
                model=self.config.model_name,
                input=[text],
                api_key=self.config.litellm_api_key,
                api_base=(
                    str(self.config.litellm_api_base)
                    if self.config.litellm_api_base
                    else None
                ),
            )
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

        if self.dev_mode:
            # Return stub embeddings in dev mode
            import random

            stub_embeddings = []
            for text in texts:
                if text and isinstance(text, str):
                    random.seed(hash(text) % (2**32))  # Deterministic but varied
                    stub_embedding = [
                        random.uniform(-1.0, 1.0) for _ in range(self.get_dimension())
                    ]
                    stub_embeddings.append(stub_embedding)
            logger.debug(
                f"DEV MODE: Generated {len(stub_embeddings)} stub LiteLLM embeddings for batch"
            )
            return stub_embeddings

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
            litellm, _, _ = _get_litellm()

            response = litellm.embedding(
                model=self.config.model_name,
                input=valid_texts,
                api_key=self.config.litellm_api_key,
                api_base=(
                    str(self.config.litellm_api_base)
                    if self.config.litellm_api_base
                    else None
                ),
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
        """Get the embedding dimension (configured or auto-detected)."""
        if self.config.dimension is not None:
            return self.config.dimension
        elif self._auto_detected_dimension is not None:
            return self._auto_detected_dimension
        else:
            raise EmbeddingProviderError(
                "Embedding dimension is not configured or auto-detected for the LiteLLM provider."
            )

    def close(self) -> None:
        logger.info(f"Closing LiteLLMProvider for model '{self.config.model_name}'.")
        pass
