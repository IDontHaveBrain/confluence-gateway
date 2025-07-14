import json
import logging
import os
import platform
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


# Development mode configuration
def is_dev_mode() -> bool:
    """Check if development mode is enabled via CONFLUENCE_GATEWAY_DEV_MODE environment variable."""
    dev_mode = os.getenv("CONFLUENCE_GATEWAY_DEV_MODE", "").lower() in [
        "true",
        "1",
        "t",
        "yes",
        "y",
    ]
    if dev_mode:
        logger.info(
            "🚀 Development mode ENABLED - heavy services will be skipped for faster startup"
        )
    return dev_mode


def dev_mode_log_skip(service_name: str) -> None:
    """Log that a service is being skipped in development mode."""
    logger.info(
        f"⚡ DEV MODE: Skipping {service_name} initialization for faster development iteration"
    )


def dev_mode_log_stub(service_name: str) -> None:
    """Log that a service is using a stub implementation in development mode."""
    logger.info(f"🔧 DEV MODE: Using stub implementation for {service_name}")


def is_pytest_running() -> bool:
    """Check if code is running under pytest using multiple detection methods."""
    import sys

    # Method 1: PYTEST_VERSION (pytest >= 8.2.0) - Most reliable
    if os.environ.get("PYTEST_VERSION"):
        return True

    # Method 2: PYTEST_CURRENT_TEST (available during test execution)
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True

    # Method 3: sys.modules fallback for older versions
    if "pytest" in sys.modules:
        return True

    return False


def is_ci_running() -> bool:
    """Check if code is running in a CI environment using common CI environment variables."""
    # Common CI environment variables
    ci_indicators = [
        "CI",  # Generic CI indicator
        "CONTINUOUS_INTEGRATION",  # Generic alternative
        "GITHUB_ACTIONS",  # GitHub Actions
        "TRAVIS",  # Travis CI
        "JENKINS_URL",  # Jenkins
        "BUILDKITE",  # Buildkite
        "CIRCLECI",  # CircleCI
        "GITLAB_CI",  # GitLab CI
        "AZURE_PIPELINES",  # Azure Pipelines
        "APPVEYOR",  # AppVeyor
        "DRONE",  # Drone CI
        "SEMAPHORE",  # Semaphore CI
        "BITBUCKET_BUILD_NUMBER",  # Bitbucket Pipelines
        "CODEBUILD_BUILD_ID",  # AWS CodeBuild
        "BUILD_ID",  # Generic build ID (Jenkins, etc.)
        "TF_BUILD",  # Azure DevOps Server/TFS
    ]

    for indicator in ci_indicators:
        if indicator in os.environ:
            return True

    return False


DEFAULT_EMBEDDING_PROVIDER_TYPE: Literal["sentence-transformers", "litellm", "none"] = (
    "sentence-transformers"
)
DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_EMBEDDING_DEVICE: Literal["cpu", "cuda"] | None = None


class ModelMetadata(BaseModel):
    """Metadata for tracking embedding model information."""

    provider: str
    model_name: str
    dimension: int
    device: str | None = None
    created_at: datetime
    collection_name: str
    configuration_hash: str

    @classmethod
    def from_embedding_config(
        cls, embedding_config: "EmbeddingConfig", collection_name: str
    ) -> "ModelMetadata":
        """Create ModelMetadata from EmbeddingConfig."""
        config_dict = {
            "provider": embedding_config.provider,
            "model_name": embedding_config.model_name,
            "dimension": embedding_config.dimension,
            "device": embedding_config.device,
        }
        # Create a hash of the configuration for change detection
        config_hash = str(hash(str(sorted(config_dict.items()))))

        return cls(
            provider=embedding_config.provider,
            model_name=embedding_config.model_name or "",
            dimension=embedding_config.dimension or 0,
            device=embedding_config.device,
            created_at=datetime.now(),
            collection_name=collection_name,
            configuration_hash=config_hash,
        )


class ModelChangeType(str, Enum):
    """Enum for different types of model changes."""

    COMPATIBLE = "compatible"  # Same provider, model, dimension
    DIMENSION_CHANGE = "dimension_change"  # Different dimension (requires migration)
    MODEL_CHANGE = "model_change"  # Different model (may require migration)
    PROVIDER_CHANGE = "provider_change"  # Different provider (requires migration)
    INCOMPATIBLE = "incompatible"  # Major changes requiring full reindexing


class ModelChangeInfo(BaseModel):
    """Information about detected model changes."""

    change_type: ModelChangeType
    current_metadata: ModelMetadata | None = None
    new_config: dict[str, Any]
    migration_required: bool
    warning_message: str
    migration_guidance: str


class ConfluenceConfig(BaseModel):
    url: HttpUrl
    username: str
    api_token: str
    timeout: int = 10


class SearchConfig(BaseModel):
    default_limit: int = 20
    max_limit: int = 100
    default_expand: list[str] = ["body.view", "space"]

    hybrid_search_enabled: bool = True
    hybrid_keyword_fetch_limit: int = Field(default=50, gt=0)
    hybrid_semantic_fetch_limit: int = Field(default=50, gt=0)
    hybrid_rrf_k: int = Field(default=60, gt=0)


class EmbeddingConfig(BaseModel):
    provider: Literal["sentence-transformers", "litellm", "none"] = "none"
    model_name: str | None = None
    dimension: int | None = None
    litellm_api_key: str | None = Field(default=None, exclude=True)
    litellm_api_base: HttpUrl | None = None
    device: Literal["cpu", "cuda"] | None = None

    @model_validator(mode="after")
    def check_conditional_requirements(self) -> "EmbeddingConfig":
        if self.provider != "none":
            if not self.model_name:
                raise ValueError(
                    "EMBEDDING_MODEL_NAME must be set if EMBEDDING_PROVIDER is not 'none'."
                )
            # Note: dimension=None is allowed for auto-detection with providers that support it
            # The dimension will be automatically detected and set during model initialization

            if self.provider == "litellm":
                if self.model_name and self.model_name.startswith("ollama/"):
                    if not self.litellm_api_base:
                        raise ValueError(
                            "LITELLM_API_BASE must be set when using an 'ollama/' model with the 'litellm' provider."
                        )

            if self.provider == "sentence-transformers":
                if self.litellm_api_key or self.litellm_api_base:
                    logger.warning(
                        "LITELLM_API_KEY and LITELLM_API_BASE are ignored when EMBEDDING_PROVIDER is 'sentence-transformers'."
                    )
            elif self.provider == "litellm":
                if self.device:
                    logger.warning(
                        "EMBEDDING_DEVICE is ignored when EMBEDDING_PROVIDER is 'litellm'."
                    )

        return self


VectorDBType = Literal["chroma", "qdrant", "none"]


def sanitize_model_name_for_collection(model_name: str) -> str:
    """
    Sanitize model name for use in collection names.
    Replace special characters with underscores and convert to lowercase.
    """
    if not model_name:
        return "unknown"

    # Replace special characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", model_name)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    # Convert to lowercase
    sanitized = sanitized.lower()

    # If empty after sanitization, use "unknown"
    return sanitized if sanitized else "unknown"


def generate_model_specific_collection_name(
    base_name: str, model_name: str | None, dimension: int | None
) -> str:
    """
    Generate a model-specific collection name.
    Format: {base_name}_{sanitized_model_name}_{dimension}d
    """
    if not model_name or dimension is None:
        return base_name

    sanitized_model = sanitize_model_name_for_collection(model_name)
    return f"{base_name}_{sanitized_model}_{dimension}d"


class IndexingConfig(BaseModel):
    include_spaces: list[str] | None = None
    exclude_spaces: list[str] | None = None
    html_parser: Literal["markitdown", "unstructured"] = "markitdown"
    include_attachments: bool = False
    max_attachment_size_mb: int = Field(default=10, ge=0)
    allowed_attachment_extensions: list[str] | None = [
        "pdf",
        "docx",
        "pptx",
        "txt",
        "md",
    ]
    attachment_parser: Literal["markitdown", "unstructured"] = "markitdown"

    @field_validator("html_parser", "attachment_parser")
    @classmethod
    def check_parser_name(cls, v: str) -> str:
        valid_parsers = {"markitdown", "unstructured"}
        if v.lower() not in valid_parsers:
            raise ValueError(
                f"Invalid parser name '{v}'. Must be one of: {', '.join(valid_parsers)}"
            )
        return v.lower()

    @model_validator(mode="before")
    @classmethod
    def normalize_extensions(cls, values: dict[str, Any]) -> dict[str, Any]:
        extensions = values.get("allowed_attachment_extensions")
        if isinstance(extensions, list):
            values["allowed_attachment_extensions"] = [
                ext.lower().lstrip(".") for ext in extensions if isinstance(ext, str)
            ]
        return values


def get_user_config_path() -> Path:
    return Path.home() / ".confluence_gateway_config.json"


def get_default_config_path() -> Path:
    return Path(__file__).parent.parent / "confluence_gateway_config.json"


def get_model_metadata_path() -> Path:
    """Get the path to the model metadata file."""
    return Path.home() / ".confluence_gateway_model_metadata.json"


def save_model_metadata(metadata: ModelMetadata) -> None:
    """Save model metadata to persistent storage."""
    metadata_path = get_model_metadata_path()

    # Load existing metadata if it exists
    existing_metadata = {}
    if metadata_path.exists():
        try:
            with metadata_path.open(encoding="utf-8") as f:
                existing_metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read existing model metadata: {e}")
            existing_metadata = {}

    # Store metadata by collection name
    existing_metadata[metadata.collection_name] = metadata.model_dump()
    existing_metadata[metadata.collection_name]["created_at"] = (
        metadata.created_at.isoformat()
    )

    # Save updated metadata
    try:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(existing_metadata, f, indent=2)
        logger.info(f"Saved model metadata for collection '{metadata.collection_name}'")
    except OSError as e:
        logger.error(f"Failed to save model metadata: {e}")


def load_model_metadata(collection_name: str) -> ModelMetadata | None:
    """Load model metadata for a specific collection."""
    metadata_path = get_model_metadata_path()

    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open(encoding="utf-8") as f:
            all_metadata = json.load(f)

        if collection_name not in all_metadata:
            return None

        metadata_dict = all_metadata[collection_name]
        # Parse the ISO datetime string
        metadata_dict["created_at"] = datetime.fromisoformat(
            metadata_dict["created_at"]
        )

        return ModelMetadata(**metadata_dict)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning(
            f"Could not load model metadata for collection '{collection_name}': {e}"
        )
        return None


def list_all_model_metadata() -> dict[str, ModelMetadata]:
    """Load all model metadata."""
    metadata_path = get_model_metadata_path()

    if not metadata_path.exists():
        return {}

    try:
        with metadata_path.open(encoding="utf-8") as f:
            all_metadata = json.load(f)

        result = {}
        for collection_name, metadata_dict in all_metadata.items():
            try:
                # Parse the ISO datetime string
                metadata_dict["created_at"] = datetime.fromisoformat(
                    metadata_dict["created_at"]
                )
                result[collection_name] = ModelMetadata(**metadata_dict)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not parse metadata for collection '{collection_name}': {e}"
                )
                continue

        return result
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load model metadata: {e}")
        return {}


def detect_model_changes(
    embedding_config: EmbeddingConfig, vector_db_config: "VectorDBConfig"
) -> "ModelChangeInfo | None":
    """Detect changes in embedding model configuration."""
    if embedding_config.provider == "none" or vector_db_config.type == "none":
        return None

    collection_name = vector_db_config.get_effective_collection_name()
    current_metadata = load_model_metadata(collection_name)

    if current_metadata is None:
        # No existing metadata, this is a new setup
        return None

    # Create new metadata from current config
    new_metadata = ModelMetadata.from_embedding_config(
        embedding_config, collection_name
    )

    # Compare configurations
    change_type = ModelChangeType.COMPATIBLE
    migration_required = False
    warning_message = ""
    migration_guidance = ""

    if current_metadata.configuration_hash == new_metadata.configuration_hash:
        # No changes detected
        return None

    # Detect specific changes
    changes = []

    if current_metadata.provider != new_metadata.provider:
        change_type = ModelChangeType.PROVIDER_CHANGE
        migration_required = True
        changes.append(
            f"provider: {current_metadata.provider} → {new_metadata.provider}"
        )

    if current_metadata.model_name != new_metadata.model_name:
        if change_type == ModelChangeType.COMPATIBLE:
            change_type = ModelChangeType.MODEL_CHANGE
        migration_required = True
        changes.append(
            f"model: {current_metadata.model_name} → {new_metadata.model_name}"
        )

    if current_metadata.dimension != new_metadata.dimension:
        change_type = ModelChangeType.DIMENSION_CHANGE
        migration_required = True
        changes.append(
            f"dimension: {current_metadata.dimension} → {new_metadata.dimension}"
        )

    if current_metadata.device != new_metadata.device:
        changes.append(f"device: {current_metadata.device} → {new_metadata.device}")

    # Build warning message
    if changes:
        warning_message = (
            f"Embedding model configuration has changed: {', '.join(changes)}"
        )

        if migration_required:
            if change_type == ModelChangeType.DIMENSION_CHANGE:
                migration_guidance = (
                    f"The embedding dimension has changed from {current_metadata.dimension} "
                    f"to {new_metadata.dimension}. This requires reindexing all content.\n"
                    "Run: confluence-gateway index trigger --full-reindex"
                )
            elif change_type == ModelChangeType.PROVIDER_CHANGE:
                migration_guidance = (
                    f"The embedding provider has changed from {current_metadata.provider} "
                    f"to {new_metadata.provider}. This requires reindexing all content.\n"
                    "Run: confluence-gateway index trigger --full-reindex"
                )
            elif change_type == ModelChangeType.MODEL_CHANGE:
                migration_guidance = (
                    f"The embedding model has changed from {current_metadata.model_name} "
                    f"to {new_metadata.model_name}. This may require reindexing for optimal results.\n"
                    "Consider running: confluence-gateway index trigger --full-reindex"
                )
        else:
            migration_guidance = "No migration required. Changes are compatible."

    new_config = {
        "provider": new_metadata.provider,
        "model_name": new_metadata.model_name,
        "dimension": new_metadata.dimension,
        "device": new_metadata.device,
    }

    return ModelChangeInfo(
        change_type=change_type,
        current_metadata=current_metadata,
        new_config=new_config,
        migration_required=migration_required,
        warning_message=warning_message,
        migration_guidance=migration_guidance,
    )


def _load_config_from_file(path: Path) -> dict[str, Any]:
    config_data = {}
    if path.exists() and path.is_file():
        try:
            with path.open(encoding="utf-8") as f:
                config_data = json.load(f)
            if not isinstance(config_data, dict):
                logger.warning(
                    f"Config file at {path} does not contain a valid JSON object. Ignoring."
                )
                return {}
            logger.info(f"Loaded configuration from {path}")
            logger.debug(
                f"Configuration content from {path}: {json.dumps(config_data, indent=2)}"
            )
        except json.JSONDecodeError:
            logger.warning(
                f"Could not parse JSON from config file at {path}. Ignoring."
            )
            return {}
        except Exception as e:
            logger.warning(f"Error reading config file at {path}: {e}. Ignoring.")
            return {}
    else:
        logger.info(f"Config file not found at {path}")
    return config_data


class VectorDBConfig(BaseModel):
    type: VectorDBType = "none"
    collection_name: str | None = None
    embedding_dimension: int | None = None
    chunk_size: int = 512
    chunk_overlap: int = 50
    chroma_persist_path: str | None = None
    chroma_host: str | None = None
    chroma_port: int | None = None
    qdrant_url: HttpUrl | Literal[":memory:"] | None = None
    qdrant_local_path: str | None = None
    qdrant_api_key: str | None = None
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = False

    # Internal fields for model-specific naming
    _embedding_model_name: str | None = None
    _embedding_dimension: int | None = None

    def get_effective_collection_name(
        self, model_name: str | None = None, dimension: int | None = None
    ) -> str:
        """
        Get the effective collection name based on configuration.

        If collection_name is explicitly set, use it as-is (user override).
        If collection_name is None/empty, auto-generate with cg_ prefix and model info.
        """
        # If collection_name is explicitly set, use it as-is (backward compatibility)
        if self.collection_name:
            return self.collection_name

        # Auto-generate model-specific collection name with "cg" prefix
        # Use provided parameters first, then internal fields, then config fields
        effective_model = (
            model_name
            or self._embedding_model_name
            or getattr(self, "embedding_model_name", None)
        )
        effective_dimension = (
            dimension or self._embedding_dimension or self.embedding_dimension
        )

        if effective_model and effective_dimension:
            return generate_model_specific_collection_name(
                "cg", effective_model, effective_dimension
            )

        # Fallback to base "cg" name if model info not available
        return "cg"

    def set_embedding_info(self, model_name: str | None, dimension: int | None) -> None:
        """
        Set embedding model information for collection naming.
        This method is called during configuration loading.
        """
        self._embedding_model_name = model_name
        self._embedding_dimension = dimension

    @model_validator(mode="after")
    def check_conditional_requirements(self) -> "VectorDBConfig":
        if self.type != "none":
            # Note: embedding_dimension=None is allowed for auto-detection
            # The dimension will be set from EmbeddingConfig during configuration loading
            # or auto-detected during embedding service initialization
            pass

        if self.type == "qdrant":
            if self.qdrant_url is None and self.qdrant_local_path is None:
                raise ValueError(
                    "Either QDRANT_URL or QDRANT_LOCAL_PATH must be set if VECTOR_DB_TYPE is 'qdrant'."
                )

        # Block in-memory mode when not running under pytest
        if not is_pytest_running():
            if self.type == "qdrant" and self.qdrant_url == ":memory:":
                raise ValueError(
                    "In-memory mode (:memory:) is only allowed during testing. "
                    "Please use a persistent Qdrant configuration for production use."
                )
            elif self.type == "chroma" and (
                self.chroma_persist_path is None or self.chroma_persist_path == ""
            ):
                raise ValueError(
                    "In-memory mode (empty persist_path) is only allowed during testing. "
                    "Please set CHROMA_PERSIST_PATH for production use."
                )

        return self


class GenerationConfig(BaseModel):
    enable: bool = False
    provider: Literal["litellm"] = "litellm"
    model_name: str | None = None
    litellm_api_key: str | None = Field(default=None, exclude=True)
    litellm_api_base: HttpUrl | None = None
    prompt_template: str = "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    max_context_tokens: int = Field(default=8000, gt=0)
    max_output_tokens: int = Field(default=500, gt=0)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    generation_timeout: int = Field(default=60, gt=0)

    @model_validator(mode="after")
    def check_conditional_requirements(self) -> "GenerationConfig":
        if self.enable and not self.model_name:
            raise ValueError(
                "GENERATION_MODEL_NAME must be set if GENERATION_ENABLE is True."
            )
        return self


def _load_raw_env_vars(prefix: str, case_sensitive: bool = False) -> dict[str, Any]:
    env_vars = {}

    is_windows = platform.system().lower() == "windows"
    effective_case_sensitive = case_sensitive and not is_windows

    prefix_upper = prefix.upper()
    prefix_len = len(prefix)

    for key, value in os.environ.items():
        key_to_check = key if effective_case_sensitive else key.upper()
        prefix_to_check = prefix if effective_case_sensitive else prefix_upper

        if key_to_check.startswith(prefix_to_check):
            config_key = (
                key[prefix_len:].lower()
                if not effective_case_sensitive
                else key[prefix_len:]
            )
            env_vars[config_key] = value

    return env_vars


def _try_convert_to_int(config: dict[str, Any], key: str, error_message: str) -> None:
    if key in config and isinstance(config[key], str):
        try:
            config[key] = int(config[key])
        except ValueError:
            logger.warning(error_message)
            del config[key]


def _load_raw_search_env() -> dict[str, Any]:
    validations = {
        "default_limit": int,
        "max_limit": int,
        "default_expand": "comma_list",
        "hybrid_search_enabled": bool,
        "hybrid_keyword_fetch_limit": int,
        "hybrid_semantic_fetch_limit": int,
        "hybrid_rrf_k": int,
    }

    return _load_env_with_validation("SEARCH_", validations)


def _load_raw_confluence_env() -> dict[str, Any]:
    validations = {
        "timeout": int,
    }
    return _load_env_with_validation("CONFLUENCE_", validations)


def _load_raw_vector_db_env() -> dict[str, Any]:
    validations = {
        "type": get_args(VectorDBType),
        "embedding_dimension": int,
        "chunk_size": int,
        "chunk_overlap": int,
    }

    raw_config = _load_env_with_validation("VECTOR_DB_", validations)

    for env_var, config_key in [
        ("CHROMA_PERSIST_PATH", "chroma_persist_path"),
        ("CHROMA_HOST", "chroma_host"),
    ]:
        if value := os.getenv(env_var):
            raw_config[config_key] = value

    if port_str := os.getenv("CHROMA_PORT"):
        try:
            raw_config["chroma_port"] = int(port_str)
        except ValueError:
            raw_config["chroma_port"] = port_str

    for env_var, config_key in [
        ("QDRANT_URL", "qdrant_url"),
        ("QDRANT_LOCAL_PATH", "qdrant_local_path"),
        ("QDRANT_API_KEY", "qdrant_api_key"),
    ]:
        if value := os.getenv(env_var):
            raw_config[config_key] = value

    if grpc_port_str := os.getenv("QDRANT_GRPC_PORT"):
        try:
            raw_config["qdrant_grpc_port"] = int(grpc_port_str)
        except ValueError:
            raw_config["qdrant_grpc_port"] = grpc_port_str

    if prefer_grpc_str := os.getenv("QDRANT_PREFER_GRPC"):
        raw_config["qdrant_prefer_grpc"] = prefer_grpc_str.lower() in [
            "true",
            "1",
            "t",
            "yes",
            "y",
        ]

    return raw_config


def _load_env_with_validation(
    prefix: str, validations: dict[str, Any] = None
) -> dict[str, Any]:
    raw_config = _load_raw_env_vars(prefix)

    if not validations:
        return raw_config

    keys_to_process = list(raw_config.keys())

    for key in keys_to_process:
        if key not in validations:
            continue

        validator = validations[key]
        value = raw_config[key]

        if isinstance(validator, type):
            try:
                if validator is int:
                    raw_config[key] = int(value)
                elif validator is float:
                    raw_config[key] = float(value)
                elif validator is bool and isinstance(value, str):
                    raw_config[key] = value.lower() in ["true", "1", "t", "yes", "y"]
            except ValueError:
                logger.warning(
                    f"Invalid value '{value}' for environment variable {prefix}{key.upper()}. "
                    f"Expected {validator.__name__}. Using default or ignoring."
                )
                del raw_config[key]

        elif isinstance(validator, tuple) and all(
            isinstance(t, type) for t in validator
        ):
            if isinstance(value, str) and value.lower() not in [
                str(v).lower() for v in validator
            ]:
                logger.warning(
                    f"Invalid value '{value}' for environment variable {prefix}{key.upper()}. "
                    f"Expected one of: {', '.join(map(str, validator))}. Using default or ignoring."
                )
                del raw_config[key]
            elif isinstance(value, str):
                for literal_val in validator:
                    if str(literal_val).lower() == value.lower():
                        raw_config[key] = literal_val
                        break

        elif validator == "comma_list" and isinstance(value, str):
            raw_config[key] = [s.strip() for s in value.split(",") if s.strip()]

    return raw_config


def _load_raw_embedding_env() -> dict[str, Any]:
    validations = {
        "provider": get_args(Literal["sentence-transformers", "litellm", "none"]),
        "dimension": int,
        "device": get_args(Literal["cpu", "cuda"]) + (None,),
    }

    raw_config = _load_env_with_validation("EMBEDDING_", validations)

    for env_var, config_key in [
        ("LITELLM_API_KEY", "litellm_api_key"),
        ("LITELLM_API_BASE", "litellm_api_base"),
    ]:
        if value := os.getenv(env_var):
            raw_config[config_key] = value

    return raw_config


def _load_raw_indexing_env() -> dict[str, Any]:
    validations = {
        "include_spaces": "comma_list",
        "exclude_spaces": "comma_list",
        "html_parser": get_args(Literal["markitdown", "unstructured"]),
        "include_attachments": bool,
        "max_attachment_size_mb": int,
        "allowed_attachment_extensions": "comma_list",
        "attachment_parser": get_args(Literal["markitdown", "unstructured"]),
    }

    raw_config = _load_env_with_validation("INDEXING_", validations)

    if "allowed_attachment_extensions" in raw_config and isinstance(
        raw_config["allowed_attachment_extensions"], list
    ):
        raw_config["allowed_attachment_extensions"] = [
            ext.lower().lstrip(".")
            for ext in raw_config["allowed_attachment_extensions"]
        ]

    return raw_config


def _load_raw_generation_env() -> dict[str, Any]:
    validations = {
        "enable": bool,
        "provider": get_args(Literal["litellm"]),
        "max_context_tokens": int,
        "max_output_tokens": int,
        "temperature": float,
        "generation_timeout": int,
    }

    raw_config = _load_env_with_validation("GENERATION_", validations)

    for env_var, config_key in [
        ("GENERATION_MODEL_NAME", "model_name"),
        ("GENERATION_LITELLM_API_KEY", "litellm_api_key"),
        ("GENERATION_LITELLM_API_BASE", "litellm_api_base"),
        ("GENERATION_PROMPT_TEMPLATE", "prompt_template"),
    ]:
        if value := os.getenv(env_var):
            raw_config[config_key] = value

    return raw_config


def load_configurations() -> tuple[
    ConfluenceConfig | None,
    SearchConfig,
    VectorDBConfig | None,
    EmbeddingConfig | None,
    IndexingConfig,
    GenerationConfig | None,
    ModelChangeInfo | None,
]:
    # Load default config first
    default_config_path = get_default_config_path()
    logger.info(f"Default config path resolved to: {default_config_path}")
    logger.info(f"Default config path exists: {default_config_path.exists()}")
    default_config = _load_config_from_file(default_config_path)

    # Load user config
    user_config_path = get_user_config_path()
    logger.info(f"User config path resolved to: {user_config_path}")
    user_config = _load_config_from_file(user_config_path)

    # Load environment variables
    env_confluence_raw = _load_raw_confluence_env()
    env_search_raw = _load_raw_search_env()
    env_vector_db_raw = _load_raw_vector_db_env()
    env_embedding_raw = _load_raw_embedding_env()
    env_indexing_raw = _load_raw_indexing_env()
    env_generation_raw = _load_raw_generation_env()

    # Priority: user home config > env vars > default config
    # For each section, start with default, override with env vars, then override with user config
    def merge_configs(
        default_section: dict[str, Any],
        env_section: dict[str, Any],
        user_section: dict[str, Any],
    ) -> dict[str, Any]:
        # Start with default
        result = default_section.copy() if default_section else {}
        # Override with env vars
        result.update(env_section)
        # Override with user config (highest priority)
        if user_section:
            result.update(user_section)
        return result

    final_confluence_config = merge_configs(
        default_config.get("confluence", {}),
        env_confluence_raw,
        user_config.get("confluence", {}),
    )

    final_search_config = merge_configs(
        default_config.get("search", {}), env_search_raw, user_config.get("search", {})
    )

    final_vector_db_config = merge_configs(
        default_config.get("vector_db", {}),
        env_vector_db_raw,
        user_config.get("vector_db", {}),
    )

    final_embedding_config = merge_configs(
        default_config.get("embedding", {}),
        env_embedding_raw,
        user_config.get("embedding", {}),
    )

    final_indexing_config = merge_configs(
        default_config.get("indexing", {}),
        env_indexing_raw,
        user_config.get("indexing", {}),
    )

    final_generation_config = merge_configs(
        default_config.get("generation", {}),
        env_generation_raw,
        user_config.get("generation", {}),
    )

    loaded_confluence_config: ConfluenceConfig | None = None
    required_confluence_fields = ["url", "username", "api_token"]
    if all(field in final_confluence_config for field in required_confluence_fields):
        try:
            loaded_confluence_config = ConfluenceConfig(**final_confluence_config)
        except ValidationError as e:
            logger.error(f"Invalid Confluence configuration: {e}")
            logger.warning("Confluence client cannot be initialized.")
    else:
        logger.info(
            "Essential Confluence configuration (url, username, api_token) not found. Confluence client disabled."
        )

    try:
        loaded_search_config = SearchConfig(**final_search_config)
    except ValidationError as e:
        logger.error(f"Invalid Search configuration: {e}. Using defaults.")
        loaded_search_config = SearchConfig()

    loaded_embedding_config: EmbeddingConfig | None = None
    embedding_load_error = False

    if final_embedding_config:
        try:
            filtered_emb_config = {
                k: v for k, v in final_embedding_config.items() if v is not None
            }
            config_instance = EmbeddingConfig(**filtered_emb_config)
            if config_instance.provider != "none":
                loaded_embedding_config = config_instance
                logger.info(
                    f"Loaded Embedding configuration (Provider: {config_instance.provider})."
                )
            else:
                logger.info("Embedding features disabled (provider='none').")
        except (ValidationError, ValueError) as e:
            logger.error(f"Invalid Embedding configuration: {e}")
            embedding_load_error = True

    if loaded_embedding_config is None and embedding_load_error:
        logger.warning("Embedding features disabled due to invalid configuration.")

    # Synchronize embedding dimensions between EmbeddingConfig and VectorDBConfig
    # This handles both explicit dimensions and auto-detected dimensions
    if loaded_embedding_config:
        vdb_type = final_vector_db_config.get("type", "none")

        if vdb_type != "none":
            embedding_dim = loaded_embedding_config.dimension
            vdb_embedding_dim = final_vector_db_config.get("embedding_dimension")

            if embedding_dim is not None:
                # Embedding dimension is explicitly set
                if "embedding_dimension" not in final_vector_db_config:
                    # Set VectorDB dimension from explicit embedding dimension
                    final_vector_db_config["embedding_dimension"] = embedding_dim
                    logger.info(
                        f"Setting VectorDB embedding_dimension from EmbeddingConfig: {embedding_dim}"
                    )
                elif (
                    vdb_embedding_dim is not None and vdb_embedding_dim != embedding_dim
                ):
                    # Both have explicit dimensions but they differ - warn and use VectorDB value
                    logger.warning(
                        f"VECTOR_DB_EMBEDDING_DIMENSION ({vdb_embedding_dim}) "
                        f"differs from EMBEDDING_DIMENSION ({embedding_dim}). Using the VectorDB specific value."
                    )
            else:
                # Embedding dimension will be auto-detected
                if "embedding_dimension" not in final_vector_db_config:
                    # Both will use auto-detection - VectorDB will get dimension after embedding initialization
                    final_vector_db_config["embedding_dimension"] = None
                    logger.info(
                        "Embedding dimension will be auto-detected. VectorDB will synchronize after embedding initialization."
                    )
                elif vdb_embedding_dim is not None:
                    # VectorDB has explicit dimension but embedding will auto-detect
                    logger.info(
                        f"VectorDB has explicit embedding_dimension ({vdb_embedding_dim}) while embedding dimension will be auto-detected. "
                        "Dimension compatibility will be validated after auto-detection."
                    )
                # Note: If VectorDB also has None, both will auto-detect and sync during initialization

    loaded_vector_db_config: VectorDBConfig | None = None
    if final_vector_db_config:
        if "type" not in final_vector_db_config:
            final_vector_db_config["type"] = "none"
            logger.info("Vector DB type missing, defaulting to 'none'.")

        try:
            filtered_vdb_config = {
                k: v for k, v in final_vector_db_config.items() if v is not None
            }
            if "type" not in filtered_vdb_config and "type" in final_vector_db_config:
                filtered_vdb_config["type"] = final_vector_db_config["type"]

            config_instance = VectorDBConfig(**filtered_vdb_config)

            if config_instance.type != "none":
                # Set embedding info for collection naming
                if loaded_embedding_config:
                    config_instance.set_embedding_info(
                        loaded_embedding_config.model_name,
                        loaded_embedding_config.dimension,
                    )
                    effective_name = config_instance.get_effective_collection_name()
                    if config_instance.collection_name:
                        logger.info(f"Using explicit collection name: {effective_name}")
                    else:
                        logger.info(
                            f"Using auto-generated collection name: {effective_name}"
                        )

                loaded_vector_db_config = config_instance
            else:
                logger.info("Vector database integration is disabled (type='none').")

        except (ValidationError, ValueError) as e:
            logger.error(f"Invalid Vector DB configuration: {e}")
    else:
        logger.info("No Vector DB configuration found. Vector DB features disabled.")

    try:
        loaded_indexing_config = IndexingConfig(**final_indexing_config)
    except ValidationError as e:
        logger.error(f"Invalid Indexing configuration: {e}. Using defaults.")
        loaded_indexing_config = IndexingConfig()

    loaded_generation_config: GenerationConfig | None = None
    if final_generation_config:
        if "enable" not in final_generation_config:
            final_generation_config["enable"] = False
            logger.info("Generation 'enable' flag missing, defaulting to False.")

        try:
            filtered_gen_config = {
                k: v for k, v in final_generation_config.items() if v is not None
            }
            if (
                "enable" not in filtered_gen_config
                and "enable" in final_generation_config
            ):
                filtered_gen_config["enable"] = final_generation_config["enable"]

            config_instance = GenerationConfig(**filtered_gen_config)

            if config_instance.enable:
                loaded_generation_config = config_instance
                logger.info("RAG Generation features enabled.")
            else:
                logger.info("RAG Generation features disabled (enable=False).")

        except (ValidationError, ValueError) as e:
            logger.error(f"Invalid Generation configuration: {e}")
            logger.warning("RAG Generation features will be disabled.")
    else:
        logger.info("No Generation configuration found. RAG features disabled.")

    # Detect model changes if both embedding and vector db configs are loaded
    model_change_info = None
    if loaded_embedding_config and loaded_vector_db_config:
        model_change_info = detect_model_changes(
            loaded_embedding_config, loaded_vector_db_config
        )

        if model_change_info:
            logger.warning(f"🔄 {model_change_info.warning_message}")
            if model_change_info.migration_required:
                logger.warning(
                    f"⚠️  MIGRATION REQUIRED: {model_change_info.migration_guidance}"
                )
            else:
                logger.info(f"ℹ️  {model_change_info.migration_guidance}")

        # Save current model metadata for future change detection
        # Only save if this is not a test environment
        if not is_pytest_running():
            try:
                effective_collection_name = (
                    loaded_vector_db_config.get_effective_collection_name()
                )
                new_metadata = ModelMetadata.from_embedding_config(
                    loaded_embedding_config, effective_collection_name
                )
                save_model_metadata(new_metadata)
            except Exception as e:
                logger.warning(f"Could not save model metadata: {e}")

    return (
        loaded_confluence_config,
        loaded_search_config,
        loaded_vector_db_config,
        loaded_embedding_config,
        loaded_indexing_config,
        loaded_generation_config,
        model_change_info,
    )


# Configuration cache for lazy loading - typed variables for type safety
_configs_loaded = False
_cached_confluence_config: ConfluenceConfig | None = None
_cached_search_config: SearchConfig | None = None
_cached_vector_db_config: VectorDBConfig | None = None
_cached_embedding_config: EmbeddingConfig | None = None
_cached_indexing_config: IndexingConfig | None = None
_cached_generation_config: GenerationConfig | None = None
_cached_model_change_info: ModelChangeInfo | None = None


def _ensure_configs_loaded() -> None:
    """Ensure all configurations are loaded into cache."""
    global _configs_loaded, _cached_confluence_config, _cached_search_config
    global _cached_vector_db_config, _cached_embedding_config, _cached_indexing_config
    global _cached_generation_config, _cached_model_change_info

    if not _configs_loaded:
        logger.debug("Loading all configurations into cache")
        (
            confluence_cfg,
            search_cfg,
            vector_db_cfg,
            embedding_cfg,
            indexing_cfg,
            generation_cfg,
            model_change_cfg,
        ) = load_configurations()

        _cached_confluence_config = confluence_cfg
        _cached_search_config = search_cfg
        _cached_vector_db_config = vector_db_cfg
        _cached_embedding_config = embedding_cfg
        _cached_indexing_config = indexing_cfg
        _cached_generation_config = generation_cfg
        _cached_model_change_info = model_change_cfg
        _configs_loaded = True


def get_confluence_config() -> ConfluenceConfig | None:
    """Get Confluence configuration, loading lazily if needed."""
    _ensure_configs_loaded()
    return _cached_confluence_config


def get_search_config() -> SearchConfig:
    """Get Search configuration, loading lazily if needed."""
    _ensure_configs_loaded()
    assert _cached_search_config is not None  # SearchConfig is never None
    return _cached_search_config


def get_vector_db_config() -> VectorDBConfig | None:
    """Get VectorDB configuration, loading lazily if needed."""
    _ensure_configs_loaded()
    return _cached_vector_db_config


def get_embedding_config() -> EmbeddingConfig | None:
    """Get Embedding configuration, loading lazily if needed."""
    _ensure_configs_loaded()
    return _cached_embedding_config


def get_indexing_config() -> IndexingConfig:
    """Get Indexing configuration, loading lazily if needed."""
    _ensure_configs_loaded()
    assert _cached_indexing_config is not None  # IndexingConfig is never None
    return _cached_indexing_config


def get_generation_config() -> GenerationConfig | None:
    """Get Generation configuration, loading lazily if needed."""
    _ensure_configs_loaded()
    return _cached_generation_config


def get_model_change_info() -> ModelChangeInfo | None:
    """Get Model change information, loading lazily if needed."""
    _ensure_configs_loaded()
    return _cached_model_change_info


def clear_config_cache() -> None:
    """Clear the configuration cache to force reload on next access."""
    global _configs_loaded, _cached_confluence_config, _cached_search_config
    global _cached_vector_db_config, _cached_embedding_config, _cached_indexing_config
    global _cached_generation_config, _cached_model_change_info

    _configs_loaded = False
    _cached_confluence_config = None
    _cached_search_config = None
    _cached_vector_db_config = None
    _cached_embedding_config = None
    _cached_indexing_config = None
    _cached_generation_config = None
    _cached_model_change_info = None


def synchronize_auto_detected_dimension(auto_detected_dimension: int) -> None:
    """
    Synchronize vector DB configuration with auto-detected embedding dimension.

    This function should be called by embedding services after they auto-detect
    the embedding dimension at runtime. It will update the cached vector DB
    configuration to use the auto-detected dimension and validate compatibility.

    Args:
        auto_detected_dimension: The dimension value that was auto-detected

    Raises:
        ValueError: If dimension validation fails
    """
    global _cached_vector_db_config, _cached_embedding_config

    # Ensure configs are loaded first
    _ensure_configs_loaded()

    if _cached_embedding_config is None:
        logger.warning(
            "Cannot synchronize auto-detected dimension: no embedding config loaded"
        )
        return

    if _cached_vector_db_config is None:
        logger.debug("No vector DB config to synchronize with auto-detected dimension")
        return

    # Update the cached embedding config with auto-detected dimension
    _cached_embedding_config.dimension = auto_detected_dimension
    logger.info(
        f"Updated cached embedding dimension with auto-detected value: {auto_detected_dimension}"
    )

    # Check if vector DB config needs dimension synchronization
    vdb_embedding_dim = _cached_vector_db_config.embedding_dimension

    if vdb_embedding_dim is None:
        # Vector DB also needs the auto-detected dimension
        _cached_vector_db_config.embedding_dimension = auto_detected_dimension
        logger.info(
            f"Synchronized vector DB embedding_dimension with auto-detected value: {auto_detected_dimension}"
        )
    elif vdb_embedding_dim != auto_detected_dimension:
        # Vector DB has explicit dimension that differs from auto-detected
        logger.warning(
            f"Dimension mismatch after auto-detection: VectorDB expects {vdb_embedding_dim} "
            f"but embedding model auto-detected {auto_detected_dimension}. "
            "This may cause compatibility issues."
        )
        raise ValueError(
            f"Incompatible dimensions: VectorDB configured for {vdb_embedding_dim} "
            f"but embedding model auto-detected {auto_detected_dimension}"
        )
    else:
        # Dimensions match - validation successful
        logger.info(
            f"Dimension validation successful: auto-detected dimension {auto_detected_dimension} "
            "matches vector DB configuration"
        )


# Backward compatibility: module-level __getattr__ for lazy loading of old global config variables
def __getattr__(name: str) -> Any:
    """Module-level __getattr__ for lazy loading of global config variables."""
    if name == "confluence_config":
        return get_confluence_config()
    elif name == "search_config":
        return get_search_config()
    elif name == "vector_db_config":
        return get_vector_db_config()
    elif name == "embedding_config":
        return get_embedding_config()
    elif name == "indexing_config":
        return get_indexing_config()
    elif name == "generation_config":
        return get_generation_config()
    elif name == "model_change_info":
        return get_model_change_info()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
