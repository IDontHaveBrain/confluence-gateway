import logging
from typing import Optional

from confluence_gateway.adapters.embedding.base import EmbeddingProvider
from confluence_gateway.core.config import EmbeddingConfig
from confluence_gateway.core.exceptions import EmbeddingProviderError

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

    provider_name = config.provider
    logger.info(f"Attempting to create embedding provider: {provider_name}")

    provider_classes = {
        "sentence-transformers": ".sentence_transformer.SentenceTransformerProvider",
        "litellm": ".litellm.LiteLLMProvider",
    }

    if provider_name not in provider_classes:
        logger.error(f"Unsupported embedding provider type configured: {provider_name}")
        return None

    try:
        provider_module_path = provider_classes[provider_name]
        try:
            module_name, class_name = provider_module_path.rsplit(".", 1)
            module = __import__(module_name, fromlist=["*"], globals=globals(), level=1)
            provider_class = getattr(module, class_name)
        except ImportError as e:
            logger.error(
                f"Failed to import dependency for provider '{provider_name}': {e}. "
                "Ensure required libraries are installed."
            )
            return None

        provider = provider_class(config)
        logger.info(f"Initializing {provider.__class__.__name__}...")
        provider.initialize()
        logger.info(
            f"{provider.__class__.__name__} for model '{config.model_name}' initialized successfully."
        )
        return provider

    except EmbeddingProviderError as e:
        logger.error(
            f"Failed to initialize embedding provider '{provider_name}' for model '{config.model_name}': {e}",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while creating/initializing embedding provider '{provider_name}': {e}",
            exc_info=True,
        )

    return None
