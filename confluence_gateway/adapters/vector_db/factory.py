import logging
from typing import Optional

from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.core.config import vector_db_config

logger = logging.getLogger(__name__)


def get_vector_db_adapter() -> Optional["VectorDBAdapter"]:
    if vector_db_config is None:
        logger.warning("Vector DB configuration is not loaded. Cannot get adapter.")
        return None

    adapter: Optional[VectorDBAdapter] = None
    adapter_type = vector_db_config.type
    logger.info(f"Attempting to get Vector DB adapter for type: {adapter_type}")

    try:
        if adapter_type == "none":
            logger.info("Vector DB is disabled (type='none').")
            return None

        elif adapter_type == "chroma":
            from confluence_gateway.adapters.vector_db.chroma_adapter import (
                ChromaDBAdapter,
            )

            logger.info("Creating ChromaDB adapter.")
            adapter = ChromaDBAdapter(vector_db_config)
            adapter.initialize()
            logger.info("ChromaDB adapter initialized successfully.")

        elif adapter_type == "qdrant":
            from confluence_gateway.adapters.vector_db.qdrant_adapter import (
                QdrantAdapter,
            )

            logger.info("Creating Qdrant adapter.")
            adapter = QdrantAdapter(vector_db_config)
            adapter.initialize()
            logger.info("Qdrant adapter initialized successfully.")

        else:
            logger.error(f"Unsupported vector database type configured: {adapter_type}")
            return None

        return adapter

    except Exception as e:
        logger.error(
            f"Failed to initialize vector database adapter ({adapter_type}): {e}",
            exc_info=True,
        )
        if adapter and hasattr(adapter, "close"):
            try:
                adapter.close()
            except Exception as close_e:
                logger.error(
                    f"Error during cleanup after initialization failure: {close_e}",
                    exc_info=True,
                )
        return None
