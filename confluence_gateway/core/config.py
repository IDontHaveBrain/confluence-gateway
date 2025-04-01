import os
import platform
from typing import Any, Literal, Optional, get_args

from pydantic import BaseModel, Field, HttpUrl, ValidationError, model_validator


class ConfluenceConfig(BaseModel):
    url: HttpUrl
    username: str
    api_token: str
    timeout: int = 10


class SearchConfig(BaseModel):
    default_limit: int = 20
    max_limit: int = 100
    default_expand: list[str] = ["body.view", "space"]


VectorDBType = Literal["chroma", "qdrant", "pgvector", "none"]


class VectorDBConfig(BaseModel):
    """Configuration for the Vector Database adapter."""

    type: VectorDBType = Field(
        default="none", description="The type of vector database to use."
    )
    collection_name: str = Field(
        default="confluence_embeddings",
        description="Name of the collection/table in the vector database.",
    )
    embedding_dimension: Optional[int] = Field(
        default=None,
        description="Dimension of the text embeddings. Required if type is not 'none'.",
    )

    # ChromaDB specific
    chroma_persist_path: Optional[str] = Field(
        default=None,
        description="Path for ChromaDB persistent storage. Client mode takes precedence if host/port are set.",
    )
    chroma_host: Optional[str] = Field(
        default=None, description="Hostname for ChromaDB client/server mode."
    )
    chroma_port: Optional[int] = Field(
        default=None, description="Port for ChromaDB client/server mode."
    )

    # Qdrant specific
    qdrant_url: Optional[HttpUrl] = Field(
        default=None,
        description="URL for the Qdrant instance. Required if type is 'qdrant'.",
    )
    qdrant_api_key: Optional[str] = Field(
        default=None, description="API key for Qdrant authentication."
    )
    qdrant_grpc_port: int = Field(
        default=6334, description="Port for Qdrant gRPC connections."
    )
    qdrant_prefer_grpc: bool = Field(
        default=False, description="Prefer gRPC over HTTP for Qdrant."
    )

    # pgvector specific
    pgvector_connection_string: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection string. Required if type is 'pgvector'.",
    )

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

        if self.type == "pgvector":
            if self.pgvector_connection_string is None:
                raise ValueError(
                    "PGVECTOR_CONNECTION_STRING must be set if VECTOR_DB_TYPE is 'pgvector'."
                )

        return self


def load_from_env(prefix: str, case_sensitive: bool = False) -> dict[str, Any]:
    env_vars = {}

    is_windows = platform.system().lower() == "windows"
    effective_case_sensitive = case_sensitive and not is_windows

    for key, value in os.environ.items():
        if effective_case_sensitive:
            if key.startswith(prefix):
                config_key = key[len(prefix) :]
                env_vars[config_key] = value
        else:
            if key.upper().startswith(prefix.upper()):
                config_key = key[len(prefix) :].lower()
                env_vars[config_key] = value

    return env_vars


def load_search_config_from_env() -> SearchConfig:
    search_env = load_from_env("SEARCH_")

    # Convert string values to appropriate types
    if "default_limit" in search_env:
        search_env["default_limit"] = int(search_env["default_limit"])
    if "max_limit" in search_env:
        search_env["max_limit"] = int(search_env["max_limit"])
    if "default_expand" in search_env and isinstance(search_env["default_expand"], str):
        search_env["default_expand"] = search_env["default_expand"].split(",")

    return SearchConfig(**search_env)


def load_confluence_config_from_env() -> Optional[ConfluenceConfig]:
    confluence_env = load_from_env("CONFLUENCE_")
    required_fields = ["url", "username", "api_token"]

    if not all(field in confluence_env for field in required_fields):
        return None

    # Convert timeout to int if present
    if "timeout" in confluence_env and isinstance(confluence_env["timeout"], str):
        confluence_env["timeout"] = int(confluence_env["timeout"])

    return ConfluenceConfig(**confluence_env)


def load_vector_db_config_from_env() -> Optional[VectorDBConfig]:
    """Loads Vector Database configuration from environment variables."""
    raw_config: dict[str, Any] = {}

    # Common
    vector_db_type_str = os.getenv("VECTOR_DB_TYPE", "none").lower()
    if vector_db_type_str not in get_args(VectorDBType):
        print(
            f"Warning: Invalid VECTOR_DB_TYPE '{vector_db_type_str}'. Defaulting to 'none'."
        )
        vector_db_type_str = "none"
    raw_config["type"] = vector_db_type_str

    if raw_config["type"] != "none":
        dim_str = os.getenv("VECTOR_DB_EMBEDDING_DIMENSION")
        if dim_str:
            raw_config["embedding_dimension"] = dim_str

    raw_config["collection_name"] = os.getenv("VECTOR_DB_COLLECTION_NAME")

    # ChromaDB specific
    raw_config["chroma_persist_path"] = os.getenv("CHROMA_PERSIST_PATH")
    raw_config["chroma_host"] = os.getenv("CHROMA_HOST")
    port_str = os.getenv("CHROMA_PORT")
    if port_str:
        raw_config["chroma_port"] = port_str

    # Qdrant specific
    raw_config["qdrant_url"] = os.getenv("QDRANT_URL")
    raw_config["qdrant_api_key"] = os.getenv("QDRANT_API_KEY")
    grpc_port_str = os.getenv("QDRANT_GRPC_PORT")
    if grpc_port_str:
        raw_config["qdrant_grpc_port"] = grpc_port_str

    prefer_grpc_str = os.getenv("QDRANT_PREFER_GRPC", "false").lower()
    raw_config["qdrant_prefer_grpc"] = prefer_grpc_str in ["true", "1", "t", "yes", "y"]

    # pgvector specific
    raw_config["pgvector_connection_string"] = os.getenv("PGVECTOR_CONNECTION_STRING")

    # Filter out None values so Pydantic defaults apply correctly
    filtered_config = {k: v for k, v in raw_config.items() if v is not None}

    try:
        config = VectorDBConfig(**filtered_config)
        if config.type == "none":
            print(
                "Info: Vector database integration is disabled (VECTOR_DB_TYPE='none')."
            )
        return config
    except ValidationError as e:
        print(f"Error: Invalid Vector DB configuration: {e}")
        print(
            "Error: Vector database configuration failed. Vector DB features will be disabled."
        )
        return None
    except ValueError as e:
        print(f"Error: Invalid Vector DB configuration: {e}")
        print(
            "Error: Vector database configuration failed. Vector DB features will be disabled."
        )
        return None


# Global config instances
confluence_config = load_confluence_config_from_env()
search_config = load_search_config_from_env()
vector_db_config = load_vector_db_config_from_env()
