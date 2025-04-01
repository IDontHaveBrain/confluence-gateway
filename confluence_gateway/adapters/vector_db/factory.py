from typing import Optional

from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.core.config import vector_db_config


def get_vector_db_adapter() -> Optional["VectorDBAdapter"]:
    """
    Factory function that returns the configured vector database adapter.

    Returns:
        An initialized VectorDBAdapter instance based on configuration,
        or None if vector database is disabled or configuration is invalid.
    """
    if vector_db_config is None:
        return None

    if vector_db_config.type == "none":
        return None

    if vector_db_config.type == "chroma":
        # Import here to avoid circular imports
        from confluence_gateway.adapters.vector_db.chroma_adapter import ChromaDBAdapter

        adapter = ChromaDBAdapter(vector_db_config)
        adapter.initialize()
        return adapter

    # TODO: Uncomment and implement other adapters as needed
    # if vector_db_config.type == "qdrant":
    #     # Import here to avoid circular imports
    #     from confluence_gateway.adapters.vector_db.qdrant_adapter import QdrantAdapter
    #
    #     adapter = QdrantAdapter(vector_db_config)
    #     adapter.initialize()
    #     return adapter
    #
    # if vector_db_config.type == "pgvector":
    #     # Import here to avoid circular imports
    #     from confluence_gateway.adapters.vector_db.pgvector_adapter import (
    #         PgvectorAdapter,
    #     )
    #
    #     adapter = PgvectorAdapter(vector_db_config)
    #     adapter.initialize()
    #     return adapter

    # Should never reach here due to validation in VectorDBConfig
    return None
