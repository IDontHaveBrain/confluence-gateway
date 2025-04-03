import json
import os
import platform
from pathlib import Path
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


VectorDBType = Literal["chroma", "qdrant", "none"]


def get_user_config_path() -> Path:
    """Returns the path to the user-specific configuration file."""
    return Path.home() / ".confluence_gateway_config.json"


def _load_config_from_file(path: Path) -> dict[str, Any]:
    """Loads configuration from a JSON file, returning an empty dict if not found or invalid."""
    config_data = {}
    if path.exists() and path.is_file():
        try:
            with open(path, encoding="utf-8") as f:
                config_data = json.load(f)
            if not isinstance(config_data, dict):
                print(
                    f"Warning: Config file at {path} does not contain a valid JSON object. Ignoring."
                )
                return {}
            print(f"Info: Loaded configuration from {path}")
        except json.JSONDecodeError:
            print(
                f"Warning: Could not parse JSON from config file at {path}. Ignoring."
            )
            return {}
        except Exception as e:
            print(f"Warning: Error reading config file at {path}: {e}. Ignoring.")
            return {}
    return config_data


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


def _load_raw_env_vars(prefix: str, case_sensitive: bool = False) -> dict[str, Any]:
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


def _load_raw_search_env() -> dict[str, Any]:
    search_env = _load_raw_env_vars("SEARCH_")

    # Convert string values to appropriate types
    if "default_limit" in search_env and isinstance(search_env["default_limit"], str):
        try:
            search_env["default_limit"] = int(search_env["default_limit"])
        except ValueError:
            print("Warning: Invalid SEARCH_DEFAULT_LIMIT value, using default.")
            del search_env["default_limit"]
    if "max_limit" in search_env and isinstance(search_env["max_limit"], str):
        try:
            search_env["max_limit"] = int(search_env["max_limit"])
        except ValueError:
            print("Warning: Invalid SEARCH_MAX_LIMIT value, using default.")
            del search_env["max_limit"]
    if "default_expand" in search_env and isinstance(search_env["default_expand"], str):
        search_env["default_expand"] = [
            s.strip() for s in search_env["default_expand"].split(",") if s.strip()
        ]

    return search_env


def _load_raw_confluence_env() -> dict[str, Any]:
    confluence_env = _load_raw_env_vars("CONFLUENCE_")

    # Convert timeout to int if present
    if "timeout" in confluence_env and isinstance(confluence_env["timeout"], str):
        try:
            confluence_env["timeout"] = int(confluence_env["timeout"])
        except ValueError:
            print("Warning: Invalid CONFLUENCE_TIMEOUT value, using default.")
            del confluence_env["timeout"]

    return confluence_env


def _load_raw_vector_db_env() -> dict[str, Any]:
    """Loads raw Vector Database configuration from environment variables."""
    raw_config: dict[str, Any] = {}

    # Common
    vector_db_type_str = os.getenv("VECTOR_DB_TYPE", "").lower()
    if vector_db_type_str:
        if vector_db_type_str not in get_args(VectorDBType):
            print(
                f"Warning: Invalid VECTOR_DB_TYPE '{vector_db_type_str}' in environment. Check config file or defaults."
            )
        else:
            raw_config["type"] = vector_db_type_str

    # Load other env vars only if they exist
    if dim_str := os.getenv("VECTOR_DB_EMBEDDING_DIMENSION"):
        try:
            raw_config["embedding_dimension"] = int(dim_str)
        except ValueError:
            # Keep as string, Pydantic will handle validation
            raw_config["embedding_dimension"] = dim_str

    if col_name := os.getenv("VECTOR_DB_COLLECTION_NAME"):
        raw_config["collection_name"] = col_name

    # ChromaDB specific
    if path := os.getenv("CHROMA_PERSIST_PATH"):
        raw_config["chroma_persist_path"] = path
    if host := os.getenv("CHROMA_HOST"):
        raw_config["chroma_host"] = host
    if port_str := os.getenv("CHROMA_PORT"):
        try:
            raw_config["chroma_port"] = int(port_str)
        except ValueError:
            # Keep as string for validation
            raw_config["chroma_port"] = port_str

    # Qdrant specific
    if url := os.getenv("QDRANT_URL"):
        raw_config["qdrant_url"] = url
    if key := os.getenv("QDRANT_API_KEY"):
        raw_config["qdrant_api_key"] = key
    if grpc_port_str := os.getenv("QDRANT_GRPC_PORT"):
        try:
            raw_config["qdrant_grpc_port"] = int(grpc_port_str)
        except ValueError:
            # Keep as string for validation
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


def load_configurations() -> tuple[
    Optional[ConfluenceConfig], SearchConfig, Optional[VectorDBConfig]
]:
    """
    Loads configuration from the user config file and environment variables,
    with the file taking precedence.
    """
    user_config_path = get_user_config_path()
    file_config = _load_config_from_file(user_config_path)

    # Load raw environment variables
    env_confluence_raw = _load_raw_confluence_env()
    env_search_raw = _load_raw_search_env()
    env_vector_db_raw = _load_raw_vector_db_env()

    # Get config sections from file (defaulting to empty dict if section missing)
    file_confluence = file_config.get("confluence", {})
    file_search = file_config.get("search", {})
    file_vector_db = file_config.get("vector_db", {})

    # --- Merge configurations (File overrides Environment) ---
    # Start with environment config, then update with file config
    final_confluence_config = env_confluence_raw.copy()
    final_confluence_config.update(file_confluence)

    final_search_config = env_search_raw.copy()
    final_search_config.update(file_search)

    final_vector_db_config = env_vector_db_raw.copy()
    final_vector_db_config.update(file_vector_db)

    # --- Instantiate Pydantic Models ---

    # Confluence Config
    loaded_confluence_config: Optional[ConfluenceConfig] = None
    required_confluence_fields = ["url", "username", "api_token"]
    if all(field in final_confluence_config for field in required_confluence_fields):
        try:
            loaded_confluence_config = ConfluenceConfig(**final_confluence_config)
        except ValidationError as e:
            print(f"Error: Invalid Confluence configuration: {e}")
            print("Warning: Confluence client cannot be initialized.")
    else:
        print(
            "Info: Essential Confluence configuration (url, username, api_token) not found. Confluence client disabled."
        )

    # Search Config (Always load, uses defaults)
    try:
        loaded_search_config = SearchConfig(**final_search_config)
    except ValidationError as e:
        print(f"Error: Invalid Search configuration: {e}. Using defaults.")
        loaded_search_config = SearchConfig()  # Fallback to defaults

    # Vector DB Config
    loaded_vector_db_config: Optional[VectorDBConfig] = None
    # Only attempt to load if 'type' is specified and not 'none', or if other keys exist
    if (
        final_vector_db_config.get("type", "none") != "none"
        or len(final_vector_db_config) > 1
    ):
        # Ensure 'type' defaults to 'none' if completely missing after merge
        if "type" not in final_vector_db_config:
            final_vector_db_config["type"] = "none"

        try:
            # Filter out None values before validation
            filtered_vdb_config = {
                k: v for k, v in final_vector_db_config.items() if v is not None
            }
            config_instance = VectorDBConfig(**filtered_vdb_config)
            if config_instance.type != "none":
                loaded_vector_db_config = config_instance
                print(
                    f"Info: Vector DB configured: Type='{config_instance.type}', Collection='{config_instance.collection_name}'"
                )
            else:
                print("Info: Vector database integration is disabled (type='none').")

        except (ValidationError, ValueError) as e:
            print(f"Error: Invalid Vector DB configuration: {e}")
            print(
                "Warning: Vector database configuration failed. Vector DB features will be disabled."
            )
    else:
        print("Info: No Vector DB configuration found. Vector DB features disabled.")

    return loaded_confluence_config, loaded_search_config, loaded_vector_db_config


# Global config instances
confluence_config, search_config, vector_db_config = load_configurations()
