import json
import logging
import os
import platform
from pathlib import Path
from typing import Any, Literal, Optional, Union, get_args

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


class ConfluenceConfig(BaseModel):
    url: HttpUrl
    username: str
    api_token: str
    timeout: int = 10


class SearchConfig(BaseModel):
    default_limit: int = 20
    max_limit: int = 100
    default_expand: list[str] = ["body.view", "space"]

    hybrid_search_enabled: bool = False
    hybrid_keyword_fetch_limit: int = Field(default=50, gt=0)
    hybrid_semantic_fetch_limit: int = Field(default=50, gt=0)
    hybrid_rrf_k: int = Field(default=60, gt=0)


class EmbeddingConfig(BaseModel):
    provider: Literal["sentence-transformers", "litellm", "none"] = "none"
    model_name: Optional[str] = None
    dimension: Optional[int] = None
    litellm_api_key: Optional[str] = Field(default=None, exclude=True)
    litellm_api_base: Optional[HttpUrl] = None
    device: Optional[Literal["cpu", "cuda"]] = None

    @model_validator(mode="after")
    def check_conditional_requirements(self) -> "EmbeddingConfig":
        if self.provider != "none":
            if not self.model_name:
                raise ValueError(
                    "EMBEDDING_MODEL_NAME must be set if EMBEDDING_PROVIDER is not 'none'."
                )
            if self.dimension is None:
                raise ValueError(
                    "EMBEDDING_DIMENSION must be set if EMBEDDING_PROVIDER is not 'none'."
                )

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
    include_spaces: Optional[list[str]] = None
    exclude_spaces: Optional[list[str]] = None
    html_parser: Literal["markitdown", "unstructured"] = "markitdown"
    include_attachments: bool = False
    max_attachment_size_mb: int = Field(default=10, ge=0)
    allowed_attachment_extensions: Optional[list[str]] = [
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


def _load_config_from_file(path: Path) -> dict[str, Any]:
    config_data = {}
    if path.exists() and path.is_file():
        try:
            with open(path, encoding="utf-8") as f:
                config_data = json.load(f)
            if not isinstance(config_data, dict):
                logger.warning(
                    f"Config file at {path} does not contain a valid JSON object. Ignoring."
                )
                return {}
            logger.info(f"Loaded configuration from {path}")
        except json.JSONDecodeError:
            logger.warning(
                f"Could not parse JSON from config file at {path}. Ignoring."
            )
            return {}
        except Exception as e:
            logger.warning(f"Error reading config file at {path}: {e}. Ignoring.")
            return {}
    return config_data


class VectorDBConfig(BaseModel):
    type: VectorDBType = "none"
    collection_name: str = "confluence_embeddings"
    embedding_dimension: Optional[int] = None
    chunk_size: int = 512
    chunk_overlap: int = 50
    chroma_persist_path: Optional[str] = None
    chroma_host: Optional[str] = None
    chroma_port: Optional[int] = None
    qdrant_url: Optional[Union[HttpUrl, Literal[":memory:"]]] = None
    qdrant_api_key: Optional[str] = None
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = False

    @model_validator(mode="after")
    def check_conditional_requirements(self) -> "VectorDBConfig":
        if self.type != "none":
            if self.embedding_dimension is None:
                raise ValueError(
                    "VECTOR_DB_EMBEDDING_DIMENSION must be set if VECTOR_DB_TYPE is not 'none'."
                )

        if self.type == "qdrant":
            if self.qdrant_url is None:
                raise ValueError(
                    "QDRANT_URL must be set if VECTOR_DB_TYPE is 'qdrant'."
                )

        return self


class GenerationConfig(BaseModel):
    enable: bool = False
    provider: Literal["litellm"] = "litellm"
    model_name: Optional[str] = None
    litellm_api_key: Optional[str] = Field(default=None, exclude=True)
    litellm_api_base: Optional[HttpUrl] = None
    prompt_template: str = "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    max_context_tokens: int = Field(default=3000, gt=0)
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
    confluence_env = _load_raw_env_vars("CONFLUENCE_")

    if "timeout" in confluence_env and isinstance(confluence_env["timeout"], str):
        try:
            confluence_env["timeout"] = int(confluence_env["timeout"])
        except ValueError:
            logger.warning("Invalid CONFLUENCE_TIMEOUT value, using default.")
            del confluence_env["timeout"]

    return confluence_env


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

    for key, validator in validations.items():
        if key not in raw_config:
            continue

        value = raw_config[key]

        if isinstance(validator, type):
            try:
                if validator is int:
                    raw_config[key] = int(value)
                elif validator is bool and isinstance(value, str):
                    raw_config[key] = value.lower() in ["true", "1", "t", "yes", "y"]
            except ValueError:
                pass

        elif isinstance(validator, list):
            if isinstance(value, str) and value.lower() not in [
                str(v).lower() for v in validator
            ]:
                logger.warning(
                    f"Invalid {prefix}{key} '{value}' in environment. Using default."
                )
            elif isinstance(value, str):
                raw_config[key] = value.lower()

        elif validator == "comma_list" and isinstance(value, str):
            raw_config[key] = [s.strip() for s in value.split(",") if s.strip()]

    return raw_config


def _load_raw_embedding_env() -> dict[str, Any]:
    validations = {
        "provider": get_args(Literal["sentence-transformers", "litellm", "none"]),
        "dimension": int,
        "device": get_args(Literal["cpu", "cuda"]),
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
    Optional[ConfluenceConfig],
    SearchConfig,
    Optional[VectorDBConfig],
    Optional[EmbeddingConfig],
    IndexingConfig,
    Optional["GenerationConfig"],
]:
    user_config_path = get_user_config_path()
    file_config = _load_config_from_file(user_config_path)

    env_confluence_raw = _load_raw_confluence_env()
    env_search_raw = _load_raw_search_env()
    env_vector_db_raw = _load_raw_vector_db_env()
    env_embedding_raw = _load_raw_embedding_env()
    env_indexing_raw = _load_raw_indexing_env()
    env_generation_raw = _load_raw_generation_env()

    file_confluence = file_config.get("confluence", {})
    file_search = file_config.get("search", {})
    file_vector_db = file_config.get("vector_db", {})
    file_embedding = file_config.get("embedding", {})
    file_indexing = file_config.get("indexing", {})
    file_generation = file_config.get("generation", {})

    final_confluence_config = env_confluence_raw.copy()
    final_confluence_config.update(file_confluence)

    final_search_config = env_search_raw.copy()
    final_search_config.update(file_search)

    final_vector_db_config = env_vector_db_raw.copy()
    final_vector_db_config.update(file_vector_db)

    final_embedding_config = env_embedding_raw.copy()
    final_embedding_config.update(file_embedding)

    final_indexing_config = env_indexing_raw.copy()
    final_indexing_config.update(file_indexing)

    final_generation_config = env_generation_raw.copy()
    final_generation_config.update(file_generation)

    loaded_confluence_config: Optional[ConfluenceConfig] = None
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

    loaded_embedding_config: Optional[EmbeddingConfig] = None
    if final_embedding_config:
        if "provider" not in final_embedding_config:
            final_embedding_config["provider"] = "none"
            logger.info("Embedding provider type missing, defaulting to 'none'.")

        try:
            filtered_emb_config = {
                k: v for k, v in final_embedding_config.items() if v is not None
            }
            if (
                "provider" not in filtered_emb_config
                and "provider" in final_embedding_config
            ):
                filtered_emb_config["provider"] = final_embedding_config["provider"]

            config_instance = EmbeddingConfig(**filtered_emb_config)
            if config_instance.provider != "none":
                loaded_embedding_config = config_instance
            else:
                logger.info("Embedding provider is disabled (provider='none').")

        except (ValidationError, ValueError) as e:
            logger.error(f"Invalid Embedding configuration: {e}")
    else:
        logger.info("No Embedding configuration found. Embedding features disabled.")

    if loaded_embedding_config and loaded_embedding_config.dimension is not None:
        vdb_type = final_vector_db_config.get("type", "none")
        if vdb_type != "none" and "embedding_dimension" not in final_vector_db_config:
            final_vector_db_config["embedding_dimension"] = (
                loaded_embedding_config.dimension
            )
            logger.info(
                f"Setting VectorDB embedding_dimension from EmbeddingConfig: {loaded_embedding_config.dimension}"
            )
        elif (
            vdb_type != "none"
            and "embedding_dimension" in final_vector_db_config
            and final_vector_db_config.get("embedding_dimension")
            != loaded_embedding_config.dimension
        ):
            logger.warning(
                f"VECTOR_DB_EMBEDDING_DIMENSION ({final_vector_db_config.get('embedding_dimension')}) "
                f"differs from EMBEDDING_DIMENSION ({loaded_embedding_config.dimension}). Using the VectorDB specific value."
            )

    loaded_vector_db_config: Optional[VectorDBConfig] = None
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

    loaded_generation_config: Optional[GenerationConfig] = None
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

    return (
        loaded_confluence_config,
        loaded_search_config,
        loaded_vector_db_config,
        loaded_embedding_config,
        loaded_indexing_config,
        loaded_generation_config,
    )


(
    confluence_config,
    search_config,
    vector_db_config,
    embedding_config,
    indexing_config,
    generation_config,
) = load_configurations()
