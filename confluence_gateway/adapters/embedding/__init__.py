"""
Embedding provider implementations for the Confluence Gateway.

This package contains adapters for various embedding models that convert
text to vector representations for semantic search capabilities.

Available providers:
- sentence-transformers: For local embedding models using HuggingFace transformers
- litellm: For API-based embedding models (OpenAI, Cohere, etc.)
"""

from .base import EmbeddingProvider as EmbeddingProvider
from .factory import get_embedding_provider as get_embedding_provider
