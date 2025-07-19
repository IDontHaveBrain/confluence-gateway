"""
Simplified model information system for embedding model compatibility.

This module replaces the complex model metadata architecture with a simple,
version-based approach that maintains essential functionality while eliminating
over-engineering.

Key simplifications:
- Single ModelInfo dataclass replaces 4+ overlapping classes
- Simple version-based comparison instead of hash-based change detection
- Reduced from 1,372 lines to ~500 lines total
- Maintains critical dimension compatibility checking
- Provides clear reindex guidance
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from confluence_gateway.core.exceptions import EmbeddingCompatibilityError

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """
    Simplified model information for embedding compatibility checking.

    Replaces the complex ModelMetadata, EmbeddingModelInfo, ModelChangeInfo
    and ModelChangeType classes with a single, simple dataclass.
    """

    provider: str
    model_name: str
    dimension: int
    version: str  # Format: "provider:model:dimension" for simple comparison
    created_at: str  # ISO format timestamp

    @classmethod
    def create(cls, provider: str, model_name: str, dimension: int) -> "ModelInfo":
        """Create a new ModelInfo with auto-generated version."""
        version = f"{provider}:{model_name}:{dimension}"
        created_at = datetime.now(timezone.utc).isoformat()

        return cls(
            provider=provider,
            model_name=model_name,
            dimension=dimension,
            version=version,
            created_at=created_at,
        )

    @classmethod
    def from_provider(cls, provider: Any, provider_name: str) -> "ModelInfo":
        """
        Create ModelInfo from an embedding provider instance.

        Args:
            provider: The embedding provider instance
            provider_name: Name of the provider (e.g., "sentence-transformers", "litellm")

        Returns:
            ModelInfo instance
        """
        model_name = getattr(provider.config, "model_name", "unknown")
        dimension = provider.get_dimension()

        return cls.create(provider_name, model_name, dimension)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage in vector DB metadata."""
        return {
            "embedding_model_name": self.model_name,
            "embedding_provider": self.provider,
            "embedding_dimension": self.dimension,
            "embedding_model_created_at": self.created_at,
            "embedding_model_version": self.version,
        }

    @classmethod
    def from_metadata_dict(cls, data: dict[str, Any]) -> Optional["ModelInfo"]:
        """Create from dictionary loaded from vector DB metadata."""
        try:
            # Handle both new format (with version) and legacy format (without version)
            provider = data["embedding_provider"]
            model_name = data["embedding_model_name"]
            dimension = data["embedding_dimension"]
            created_at = data["embedding_model_created_at"]

            # Generate version if not present (legacy data)
            version = data.get(
                "embedding_model_version", f"{provider}:{model_name}:{dimension}"
            )

            return cls(
                provider=provider,
                model_name=model_name,
                dimension=dimension,
                version=version,
                created_at=created_at,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse model info from metadata: {e}")
            return None


class ModelCompatibilityValidator:
    """
    Simplified model compatibility validation.

    Replaces the complex EmbeddingCompatibilityChecker with simple,
    focused validation logic.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def validate_dimension_compatibility(
        self, current_info: ModelInfo, stored_info: ModelInfo
    ) -> None:
        """
        Validate that embedding dimensions match between current and stored models.

        Args:
            current_info: Current model information
            stored_info: Stored model information

        Raises:
            EmbeddingCompatibilityError: If dimensions don't match
        """
        if current_info.dimension != stored_info.dimension:
            error_msg = (
                f"Embedding dimension mismatch detected! "
                f"Current model '{current_info.model_name}' has dimension {current_info.dimension}, "
                f"but existing data was created with model '{stored_info.model_name}' "
                f"having dimension {stored_info.dimension}. "
                f"You must either:\n"
                f"1. Use the same embedding model ('{stored_info.model_name}'), or\n"
                f"2. Clear the vector database and re-index all content, or\n"
                f"3. Create a new collection with a different name"
            )
            raise EmbeddingCompatibilityError(error_msg)

    def check_reindex_required(
        self, current_info: ModelInfo, stored_info: ModelInfo
    ) -> tuple[bool, str]:
        """
        Check if reindexing is required based on model changes.

        Args:
            current_info: Current model information
            stored_info: Stored model information

        Returns:
            Tuple of (reindex_required, guidance_message)
        """
        # Same version = no changes
        if current_info.version == stored_info.version:
            return False, "No changes detected. Models are compatible."

        # Different dimensions = always requires reindex
        if current_info.dimension != stored_info.dimension:
            return True, (
                f"Dimension changed from {stored_info.dimension} to {current_info.dimension}. "
                f"Full reindexing required: confluence-gateway index trigger --full-reindex"
            )

        # Different provider = requires reindex
        if current_info.provider != stored_info.provider:
            return True, (
                f"Provider changed from '{stored_info.provider}' to '{current_info.provider}'. "
                f"Full reindexing required: confluence-gateway index trigger --full-reindex"
            )

        # Different model = may require reindex for optimal results
        if current_info.model_name != stored_info.model_name:
            return True, (
                f"Model changed from '{stored_info.model_name}' to '{current_info.model_name}'. "
                f"Reindexing recommended for optimal results: confluence-gateway index trigger --full-reindex"
            )

        # Should not reach here, but safe fallback
        return False, "Models appear compatible."

    def validate_compatibility(
        self,
        current_info: ModelInfo,
        stored_info: ModelInfo,
        operation_type: str = "search",
    ) -> None:
        """
        Validate compatibility between current and stored embedding models.

        Args:
            current_info: Current model information
            stored_info: Stored model information
            operation_type: Type of operation ("search" or "index")

        Raises:
            EmbeddingCompatibilityError: If models are incompatible
        """
        # Always validate dimensions (critical for data integrity)
        self.validate_dimension_compatibility(current_info, stored_info)

        # Check if reindexing is needed
        reindex_required, guidance = self.check_reindex_required(
            current_info, stored_info
        )

        if reindex_required:
            self.logger.warning(f"Model compatibility warning: {guidance}")
        else:
            self.logger.info(
                f"Embedding model compatibility confirmed: '{current_info.model_name}' "
                f"with dimension {current_info.dimension}"
            )

    def extract_model_info_from_metadata(
        self, metadata_list: list[dict[str, Any]]
    ) -> ModelInfo | None:
        """
        Extract model info from vector DB metadata.

        Args:
            metadata_list: List of metadata dictionaries from vector DB

        Returns:
            ModelInfo if found, None otherwise
        """
        if not metadata_list:
            return None

        # Look for embedding model metadata in the first entry
        first_metadata = metadata_list[0]
        return ModelInfo.from_metadata_dict(first_metadata)


class ModelInfoStorage:
    """
    Simplified storage for model information.

    Replaces complex metadata persistence with simple JSON storage.
    """

    def __init__(self) -> None:
        self.storage_path = Path.home() / ".confluence_gateway_model_info.json"
        self.logger = logging.getLogger(__name__)

    def save(self, model_info: ModelInfo, collection_name: str) -> None:
        """Save model info for a collection."""
        try:
            # Load existing data
            data = self._load_storage()

            # Store model info by collection name
            data[collection_name] = asdict(model_info)

            # Save updated data
            with self.storage_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Saved model info for collection '{collection_name}'")

        except OSError as e:
            self.logger.error(f"Failed to save model info: {e}")

    def load(self, collection_name: str) -> ModelInfo | None:
        """Load model info for a collection."""
        try:
            data = self._load_storage()

            if collection_name not in data:
                return None

            model_data = data[collection_name]
            return ModelInfo(**model_data)

        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            self.logger.warning(
                f"Could not load model info for collection '{collection_name}': {e}"
            )
            return None

    def list_all(self) -> dict[str, ModelInfo]:
        """Load all model info."""
        try:
            data = self._load_storage()
            result = {}

            for collection_name, model_data in data.items():
                try:
                    result[collection_name] = ModelInfo(**model_data)
                except (ValueError, TypeError) as e:
                    self.logger.warning(
                        f"Could not parse model info for collection '{collection_name}': {e}"
                    )
                    continue

            return result

        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(f"Could not load model info storage: {e}")
            return {}

    def _load_storage(self) -> dict[str, Any]:
        """Load storage file."""
        if not self.storage_path.exists():
            return {}

        try:
            with self.storage_path.open(encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                self.logger.warning("Storage file contains invalid data. Resetting.")
                return {}

            return data

        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(f"Could not read storage file: {e}")
            return {}


# Convenience functions for backward compatibility and easy usage
def create_model_info(provider: str, model_name: str, dimension: int) -> ModelInfo:
    """Create a new ModelInfo instance."""
    return ModelInfo.create(provider, model_name, dimension)


def validate_model_compatibility(
    current_info: ModelInfo, stored_info: ModelInfo, operation_type: str = "search"
) -> None:
    """Validate model compatibility."""
    validator = ModelCompatibilityValidator()
    validator.validate_compatibility(current_info, stored_info, operation_type)


def check_dimension_compatibility(
    current_info: ModelInfo, stored_info: ModelInfo
) -> None:
    """Check dimension compatibility (critical validation)."""
    validator = ModelCompatibilityValidator()
    validator.validate_dimension_compatibility(current_info, stored_info)


def get_reindex_guidance(
    current_info: ModelInfo, stored_info: ModelInfo
) -> tuple[bool, str]:
    """Get reindex guidance for model changes."""
    validator = ModelCompatibilityValidator()
    return validator.check_reindex_required(current_info, stored_info)
