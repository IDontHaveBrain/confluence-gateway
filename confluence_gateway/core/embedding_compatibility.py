import logging
from dataclasses import dataclass
from typing import Any

from confluence_gateway.core.exceptions import EmbeddingCompatibilityError

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelInfo:
    """Information about an embedding model for compatibility checking."""

    model_name: str
    provider: str
    dimension: int
    created_at: str  # ISO format timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage in vector DB metadata."""
        return {
            "embedding_model_name": self.model_name,
            "embedding_provider": self.provider,
            "embedding_dimension": self.dimension,
            "embedding_model_created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingModelInfo":
        """Create from dictionary loaded from vector DB metadata."""
        return cls(
            model_name=data["embedding_model_name"],
            provider=data["embedding_provider"],
            dimension=data["embedding_dimension"],
            created_at=data["embedding_model_created_at"],
        )


class EmbeddingCompatibilityChecker:
    """Utility class for checking embedding model compatibility."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def validate_dimension_compatibility(
        self,
        current_dimension: int,
        stored_dimension: int,
        current_model: str,
        stored_model: str,
    ) -> None:
        """
        Validate that embedding dimensions match between current and stored models.

        Args:
            current_dimension: Dimension of current embedding model
            stored_dimension: Dimension of stored embedding model
            current_model: Name of current embedding model
            stored_model: Name of stored embedding model

        Raises:
            EmbeddingCompatibilityError: If dimensions don't match
        """
        if current_dimension != stored_dimension:
            error_msg = (
                f"Embedding dimension mismatch detected! "
                f"Current model '{current_model}' has dimension {current_dimension}, "
                f"but existing data was created with model '{stored_model}' "
                f"having dimension {stored_dimension}. "
                f"You must either:\n"
                f"1. Use the same embedding model ('{stored_model}'), or\n"
                f"2. Clear the vector database and re-index all content, or\n"
                f"3. Create a new collection with a different name"
            )
            raise EmbeddingCompatibilityError(error_msg)

    def validate_model_compatibility(
        self,
        current_model_info: EmbeddingModelInfo,
        stored_model_info: EmbeddingModelInfo,
        strict_model_match: bool = False,
    ) -> None:
        """
        Validate compatibility between current and stored embedding models.

        Args:
            current_model_info: Information about current embedding model
            stored_model_info: Information about stored embedding model
            strict_model_match: If True, require exact model name match

        Raises:
            EmbeddingCompatibilityError: If models are incompatible
        """
        # Always validate dimensions
        self.validate_dimension_compatibility(
            current_model_info.dimension,
            stored_model_info.dimension,
            current_model_info.model_name,
            stored_model_info.model_name,
        )

        # Strict model name validation if requested
        if (
            strict_model_match
            and current_model_info.model_name != stored_model_info.model_name
        ):
            error_msg = (
                f"Strict model compatibility check failed! "
                f"Current model '{current_model_info.model_name}' "
                f"differs from stored model '{stored_model_info.model_name}'. "
                f"For strict compatibility, you must use the exact same model."
            )
            raise EmbeddingCompatibilityError(error_msg)

        # Log compatibility information
        if current_model_info.model_name != stored_model_info.model_name:
            self.logger.warning(
                f"Using different embedding model than stored data: "
                f"current='{current_model_info.model_name}' vs "
                f"stored='{stored_model_info.model_name}'. "
                f"Dimensions match ({current_model_info.dimension}), so proceeding."
            )
        else:
            self.logger.info(
                f"Embedding model compatibility confirmed: '{current_model_info.model_name}' "
                f"with dimension {current_model_info.dimension}"
            )

    def create_model_info_from_provider(
        self,
        provider: Any,  # EmbeddingProvider
        provider_name: str,
        timestamp: str,
    ) -> EmbeddingModelInfo:
        """
        Create EmbeddingModelInfo from an embedding provider instance.

        Args:
            provider: The embedding provider instance
            provider_name: Name of the provider (e.g., "sentence-transformers", "litellm")
            timestamp: ISO format timestamp

        Returns:
            EmbeddingModelInfo instance
        """
        model_name = getattr(provider.config, "model_name", "unknown")
        dimension = provider.get_dimension()

        return EmbeddingModelInfo(
            model_name=model_name,
            provider=provider_name,
            dimension=dimension,
            created_at=timestamp,
        )

    def extract_model_info_from_metadata(
        self, metadata_list: list[dict[str, Any]]
    ) -> EmbeddingModelInfo | None:
        """
        Extract embedding model info from vector DB metadata.

        Args:
            metadata_list: List of metadata dictionaries from vector DB

        Returns:
            EmbeddingModelInfo if found, None otherwise
        """
        if not metadata_list:
            return None

        # Look for embedding model metadata in the first entry
        first_metadata = metadata_list[0]
        required_fields = [
            "embedding_model_name",
            "embedding_provider",
            "embedding_dimension",
            "embedding_model_created_at",
        ]

        if all(field in first_metadata for field in required_fields):
            try:
                return EmbeddingModelInfo.from_dict(first_metadata)
            except (KeyError, TypeError, ValueError) as e:
                self.logger.warning(
                    f"Failed to parse embedding model info from metadata: {e}"
                )
                return None

        return None

    def should_validate_compatibility(
        self,
        operation_type: str,  # "search" or "index"
        has_existing_data: bool,
    ) -> bool:
        """
        Determine if compatibility validation should be performed.

        Args:
            operation_type: Type of operation ("search" or "index")
            has_existing_data: Whether the collection has existing data

        Returns:
            True if validation should be performed
        """
        return has_existing_data
