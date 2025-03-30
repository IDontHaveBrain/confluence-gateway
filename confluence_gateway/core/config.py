import os
import platform
from typing import Any, Optional

from pydantic import BaseModel, HttpUrl


class ConfluenceConfig(BaseModel):
    """Configuration for Confluence API."""

    url: HttpUrl
    username: str
    api_token: str
    timeout: int = 10


class SearchConfig(BaseModel):
    """Configuration for search functionality."""

    default_limit: int = 20
    max_limit: int = 100
    default_expand: list[str] = ["body.view", "space"]


def load_from_env(prefix: str, case_sensitive: bool = False) -> dict[str, Any]:
    """Load configuration values from environment variables with the given prefix."""
    env_vars = {}

    # Check if we're on Windows, where environment variables are case-insensitive
    is_windows = platform.system().lower() == "windows"

    # Skip case-sensitive matching on Windows as it's not supported by the OS
    effective_case_sensitive = case_sensitive and not is_windows

    for key, value in os.environ.items():
        if effective_case_sensitive:
            # Case-sensitive matching (non-Windows only)
            if key.startswith(prefix):
                config_key = key[len(prefix) :]
                env_vars[config_key] = value
        else:
            # Case-insensitive matching
            if key.upper().startswith(prefix.upper()):
                config_key = key[len(prefix) :].lower()
                env_vars[config_key] = value

    return env_vars


def load_search_config_from_env() -> SearchConfig:
    """Load search configuration from environment variables."""
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
    """Load Confluence configuration from environment variables."""
    confluence_env = load_from_env("CONFLUENCE_")

    # Check for required fields
    required_fields = ["url", "username", "api_token"]
    if not all(field in confluence_env for field in required_fields):
        return None

    # Convert timeout to int if present
    if "timeout" in confluence_env and isinstance(confluence_env["timeout"], str):
        confluence_env["timeout"] = int(confluence_env["timeout"])

    return ConfluenceConfig(**confluence_env)


# Global config instances - will be None if required env vars aren't set
confluence_config = load_confluence_config_from_env()
search_config = load_search_config_from_env()
