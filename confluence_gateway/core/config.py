import json
import logging
import os
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, get_args

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


# Environment Detection - Consolidated Implementation


@dataclass
class EnvironmentContext:
    """Unified environment detection context."""

    is_pytest: bool
    is_ci: bool
    testing_mode: Literal["ci", "local", "production"]
    use_memory_mode: bool
    ci_platform: str | None = None


@dataclass
class DevelopmentContext:
    """Development mode detection and logging context."""

    enabled: bool
    log_skip: Callable[[str], None]
    log_stub: Callable[[str], None]


def get_environment_context() -> EnvironmentContext:
    """Single source of truth for all environment detection."""
    import sys

    # Unified pytest detection
    is_pytest = (
        bool(os.environ.get("PYTEST_VERSION"))
        or "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in sys.modules
    )

    # Unified CI detection with platform identification
    ci_indicators = {
        "GITHUB_ACTIONS": "GitHub Actions",
        "TRAVIS": "Travis CI",
        "JENKINS_URL": "Jenkins",
        "CIRCLECI": "CircleCI",
        "GITLAB_CI": "GitLab CI",
        "CI": "Generic CI",
        "CONTINUOUS_INTEGRATION": "Generic CI",
    }

    ci_platform = None
    is_ci = False
    for env_var, platform_name in ci_indicators.items():
        if env_var in os.environ:
            is_ci = True
            ci_platform = platform_name
            break

    # Unified testing mode determination with proper Literal typing
    testing_mode: Literal["ci", "local", "production"]
    if is_pytest:
        testing_mode = "ci" if is_ci else "local"
    else:
        testing_mode = "production"

    # Unified memory mode decision
    use_memory_mode = testing_mode == "local"

    return EnvironmentContext(
        is_pytest=is_pytest,
        is_ci=is_ci,
        testing_mode=testing_mode,
        use_memory_mode=use_memory_mode,
        ci_platform=ci_platform,
    )


def get_development_context() -> DevelopmentContext:
    """Unified development mode detection with integrated logging."""
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

    def log_skip(service_name: str) -> None:
        logger.info(
            f"⚡ DEV MODE: Skipping {service_name} initialization for faster development iteration"
        )

    def log_stub(service_name: str) -> None:
        logger.info(f"🔧 DEV MODE: Using stub implementation for {service_name}")

    return DevelopmentContext(enabled=dev_mode, log_skip=log_skip, log_stub=log_stub)


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

    def get_effective_collection_name(self) -> str:
        """Get the effective collection name: explicit override OR auto-generated."""
        # If collection_name is explicitly set, use it (user override)
        if self.collection_name:
            return self.collection_name

        # Auto-generate with model info if available
        if hasattr(self, "embedding_dimension") and self.embedding_dimension:
            return f"cg_{self.embedding_dimension}d"

        # Simple fallback
        return "cg"

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

        # Block in-memory mode when not running under pytest or in production
        testing_mode = get_environment_context().testing_mode

        if testing_mode == "production":
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

        # Log the testing mode for debugging
        if testing_mode != "production":
            logger.info(f"Testing mode detected: {testing_mode}")
            if testing_mode == "ci":
                logger.info(
                    "CI environment detected - using file storage for vector databases with caching benefits"
                )
            elif testing_mode == "local":
                logger.info(
                    "Local testing environment detected - using memory mode for vector databases"
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


# Unified Environment Variable Loading System


class EnvConfigSection(TypedDict, total=False):
    """Type definition for environment configuration sections."""

    prefix: str
    validations: dict[str, Any]
    manual_mappings: list[tuple[str, str]]
    special_handling: dict[str, tuple[str, type]]
    post_process: str


class EnvironmentVariableLoader:
    """Unified environment variable loader replacing 7 separate functions."""

    # Configuration schema for all sections
    ENV_LOADING_CONFIG: dict[str, EnvConfigSection] = {
        "search": {
            "prefix": "SEARCH_",
            "validations": {
                "default_limit": int,
                "max_limit": int,
                "default_expand": "comma_list",
                "hybrid_search_enabled": bool,
                "hybrid_keyword_fetch_limit": int,
                "hybrid_semantic_fetch_limit": int,
                "hybrid_rrf_k": int,
            },
        },
        "confluence": {"prefix": "CONFLUENCE_", "validations": {"timeout": int}},
        "vector_db": {
            "prefix": "VECTOR_DB_",
            "validations": {
                "type": get_args(VectorDBType),
                "embedding_dimension": int,
                "chunk_size": int,
                "chunk_overlap": int,
            },
            "manual_mappings": [
                ("CHROMA_PERSIST_PATH", "chroma_persist_path"),
                ("CHROMA_HOST", "chroma_host"),
                ("QDRANT_URL", "qdrant_url"),
                ("QDRANT_LOCAL_PATH", "qdrant_local_path"),
                ("QDRANT_API_KEY", "qdrant_api_key"),
            ],
            "special_handling": {
                "CHROMA_PORT": ("chroma_port", int),
                "QDRANT_GRPC_PORT": ("qdrant_grpc_port", int),
                "QDRANT_PREFER_GRPC": ("qdrant_prefer_grpc", bool),
            },
        },
        "embedding": {
            "prefix": "EMBEDDING_",
            "validations": {
                "provider": get_args(
                    Literal["sentence-transformers", "litellm", "none"]
                ),
                "dimension": int,
                "device": get_args(Literal["cpu", "cuda"]) + (None,),
            },
            "manual_mappings": [
                ("LITELLM_API_KEY", "litellm_api_key"),
                ("LITELLM_API_BASE", "litellm_api_base"),
            ],
        },
        "indexing": {
            "prefix": "INDEXING_",
            "validations": {
                "include_spaces": "comma_list",
                "exclude_spaces": "comma_list",
                "html_parser": get_args(Literal["markitdown", "unstructured"]),
                "include_attachments": bool,
                "max_attachment_size_mb": int,
                "allowed_attachment_extensions": "comma_list",
                "attachment_parser": get_args(Literal["markitdown", "unstructured"]),
            },
            "post_process": "_process_indexing_extensions",
        },
        "generation": {
            "prefix": "GENERATION_",
            "validations": {
                "enable": bool,
                "provider": get_args(Literal["litellm"]),
                "max_context_tokens": int,
                "max_output_tokens": int,
                "temperature": float,
                "generation_timeout": int,
            },
            "manual_mappings": [
                ("GENERATION_MODEL_NAME", "model_name"),
                ("GENERATION_LITELLM_API_KEY", "litellm_api_key"),
                ("GENERATION_LITELLM_API_BASE", "litellm_api_base"),
                ("GENERATION_PROMPT_TEMPLATE", "prompt_template"),
            ],
        },
    }

    @classmethod
    def load_section(cls, section_name: str) -> dict[str, Any]:
        """Load environment variables for a specific section."""
        config: EnvConfigSection = cls.ENV_LOADING_CONFIG[section_name]
        prefix = config["prefix"]
        validations = config["validations"]

        # Load with validation
        raw_config = cls._load_with_validation(prefix, validations)

        # Apply manual mappings if present
        if "manual_mappings" in config:
            for env_var, config_key in config["manual_mappings"]:
                if value := os.getenv(env_var):
                    raw_config[config_key] = value

        # Apply special handling if present
        if "special_handling" in config:
            for env_var, (config_key, type_converter) in config[
                "special_handling"
            ].items():
                if value := os.getenv(env_var):
                    try:
                        if type_converter is int:
                            raw_config[config_key] = int(value)
                        elif type_converter is bool:
                            raw_config[config_key] = value.lower() in [
                                "true",
                                "1",
                                "t",
                                "yes",
                                "y",
                            ]
                        else:
                            raw_config[config_key] = value
                    except ValueError:
                        raw_config[config_key] = value

        # Apply post-processing if present
        if "post_process" in config:
            method_name = config["post_process"]
            raw_config = getattr(cls, method_name)(raw_config)

        return raw_config

    @classmethod
    def _load_with_validation(
        cls, prefix: str, validations: dict[str, Any]
    ) -> dict[str, Any]:
        """Core validation logic consolidated from _load_env_with_validation."""
        raw_config = _load_raw_env_vars(prefix)

        for key in list(raw_config.keys()):
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
                        raw_config[key] = value.lower() in [
                            "true",
                            "1",
                            "t",
                            "yes",
                            "y",
                        ]
                except ValueError:
                    logger.warning(
                        f"Invalid value '{value}' for environment variable {prefix}{key.upper()}. "
                        f"Expected {validator.__name__}. Using default or ignoring."
                    )
                    del raw_config[key]

            elif isinstance(validator, tuple):
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

    @classmethod
    def _process_indexing_extensions(cls, raw_config: dict[str, Any]) -> dict[str, Any]:
        """Post-process indexing allowed_attachment_extensions."""
        if "allowed_attachment_extensions" in raw_config and isinstance(
            raw_config["allowed_attachment_extensions"], list
        ):
            raw_config["allowed_attachment_extensions"] = [
                ext.lower().lstrip(".")
                for ext in raw_config["allowed_attachment_extensions"]
            ]
        return raw_config


def load_configurations() -> tuple[
    ConfluenceConfig | None,
    SearchConfig,
    VectorDBConfig | None,
    EmbeddingConfig | None,
    IndexingConfig,
    GenerationConfig | None,
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

    # Load environment variables using unified loader
    env_confluence_raw = EnvironmentVariableLoader.load_section("confluence")
    env_search_raw = EnvironmentVariableLoader.load_section("search")
    env_vector_db_raw = EnvironmentVariableLoader.load_section("vector_db")
    env_embedding_raw = EnvironmentVariableLoader.load_section("embedding")
    env_indexing_raw = EnvironmentVariableLoader.load_section("indexing")
    env_generation_raw = EnvironmentVariableLoader.load_section("generation")

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
                    # Removed set_embedding_info call - using simplified naming logic
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

    # Model change detection removed - now handled by simplified ModelInfo system

    return (
        loaded_confluence_config,
        loaded_search_config,
        loaded_vector_db_config,
        loaded_embedding_config,
        loaded_indexing_config,
        loaded_generation_config,
    )


# Configuration cache for lazy loading - typed variables for type safety
_configs_loaded = False
_cached_confluence_config: ConfluenceConfig | None = None
_cached_search_config: SearchConfig | None = None
_cached_vector_db_config: VectorDBConfig | None = None
_cached_embedding_config: EmbeddingConfig | None = None
_cached_indexing_config: IndexingConfig | None = None
_cached_generation_config: GenerationConfig | None = None


def _ensure_configs_loaded() -> None:
    """Ensure all configurations are loaded into cache."""
    global _configs_loaded, _cached_confluence_config, _cached_search_config
    global _cached_vector_db_config, _cached_embedding_config, _cached_indexing_config
    global _cached_generation_config

    if not _configs_loaded:
        logger.debug("Loading all configurations into cache")
        (
            confluence_cfg,
            search_cfg,
            vector_db_cfg,
            embedding_cfg,
            indexing_cfg,
            generation_cfg,
        ) = load_configurations()

        _cached_confluence_config = confluence_cfg
        _cached_search_config = search_cfg
        _cached_vector_db_config = vector_db_cfg
        _cached_embedding_config = embedding_cfg
        _cached_indexing_config = indexing_cfg
        _cached_generation_config = generation_cfg
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
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
