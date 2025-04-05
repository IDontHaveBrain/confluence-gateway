import logging
from typing import Optional

from confluence_gateway.core.config import EmbeddingConfig
from confluence_gateway.providers.embedding.base import (
    EmbeddingProvider,
    EmbeddingProviderError,
)

logger = logging.getLogger(__name__)


def get_embedding_provider(
    config: Optional[EmbeddingConfig],
) -> Optional[EmbeddingProvider]:
    if config is None:
        logger.info(
            "Embedding configuration not loaded. Embedding provider cannot be created."
        )
        return None

    if config.provider == "none":
        logger.info("Embedding provider explicitly disabled (provider='none').")
        return None

    provider: Optional[EmbeddingProvider] = None
    provider_name = config.provider

    try:
        logger.info(f"Attempting to create embedding provider: {provider_name}")

        if provider_name == "sentence-transformers":
            from .sentence_transformer import SentenceTransformerProvider

            provider = SentenceTransformerProvider(config)

        elif provider_name == "litellm":
            from .litellm import LiteLLMProvider

            provider = LiteLLMProvider(config)

        else:
            logger.error(
                f"Unsupported embedding provider type configured: {provider_name}"
            )
            return None

        if provider:
            logger.info(f"Initializing {provider.__class__.__name__}...")
            provider.initialize()
            logger.info(
                f"{provider.__class__.__name__} for model '{config.model_name}' initialized successfully."
            )
            return provider
        else:
            logger.error(
                f"Provider object creation failed unexpectedly for type: {provider_name}"
            )
            return None

    except EmbeddingProviderError as e:
        logger.error(
            f"Failed to initialize embedding provider '{provider_name}' for model '{config.model_name}': {e}",
            exc_info=True,
        )
        return None

    except ImportError as e:
        logger.error(
            f"Failed to import dependency for provider '{provider_name}': {e}. "
            f"Ensure required libraries are installed.",
            exc_info=False,
        )
        return None

    except Exception as e:
        logger.error(
            f"An unexpected error occurred while creating/initializing embedding provider '{provider_name}': {e}",
            exc_info=True,
        )
        return None
