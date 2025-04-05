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
    """
    Factory function to create and initialize the appropriate embedding provider.

    Reads the provided EmbeddingConfig, instantiates the corresponding
    EmbeddingProvider implementation, calls its initialize() method,
    and returns the initialized instance.

    Args:
        config: The loaded EmbeddingConfig object.

    Returns:
        An initialized EmbeddingProvider instance if configuration is valid
        and initialization succeeds, otherwise None.
    """
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
            # Import locally to avoid dependency if not used
            from .sentence_transformer import SentenceTransformerProvider

            provider = SentenceTransformerProvider(config)

        elif provider_name == "litellm":
            # Import locally to avoid dependency if not used
            from .litellm import LiteLLMProvider

            provider = LiteLLMProvider(config)

        else:
            # This case should be prevented by EmbeddingConfig validation, but handle defensively
            logger.error(
                f"Unsupported embedding provider type configured: {provider_name}"
            )
            return None

        # Initialize the created provider
        if provider:
            logger.info(f"Initializing {provider.__class__.__name__}...")
            provider.initialize()  # This can raise EmbeddingProviderError
            logger.info(
                f"{provider.__class__.__name__} for model '{config.model_name}' initialized successfully."
            )
            return provider
        else:
            # Should not happen if the logic above is correct
            logger.error(
                f"Provider object creation failed unexpectedly for type: {provider_name}"
            )
            return None

    except EmbeddingProviderError as e:
        # Catch errors specifically from provider initialization/validation
        logger.error(
            f"Failed to initialize embedding provider '{provider_name}' for model '{config.model_name}': {e}",
            exc_info=True,
        )
        # Optional: Attempt cleanup if provider object exists but failed init?
        # if provider and hasattr(provider, 'close'):
        #     try: provider.close()
        #     except Exception: pass # Ignore close errors during failure handling
        return None  # Signal failure: provider could not be initialized

    except ImportError as e:
        # Catch missing dependencies (e.g., sentence-transformers or litellm not installed)
        logger.error(
            f"Failed to import dependency for provider '{provider_name}': {e}. "
            f"Ensure required libraries are installed.",
            exc_info=False,  # Keep log cleaner, error message is informative enough
        )
        return None  # Signal failure: missing dependency

    except Exception as e:
        # Catch any other unexpected errors during instantiation or initialization
        logger.error(
            f"An unexpected error occurred while creating/initializing embedding provider '{provider_name}': {e}",
            exc_info=True,
        )
        return None  # Signal failure: unexpected error
