import os
from typing import Any

from pydantic import BaseModel, HttpUrl


class ConfluenceConfig(BaseModel):
    """
    Configuration for Confluence API.

    Attributes:
        url: Base URL for Confluence instance (e.g., 'https://your-domain.atlassian.net')
        username: Confluence username or email
        api_token: API token for authentication
        timeout: Connection timeout in seconds
    """

    url: HttpUrl
    username: str
    api_token: str
    timeout: int = 10


class SearchConfig(BaseModel):
    """
    Configuration for search functionality.

    Attributes:
        default_limit: Default number of results per page
        max_limit: Maximum allowed results per page
        default_expand: Default fields to expand in API response
    """

    default_limit: int = 20
    max_limit: int = 100
    default_expand: list[str] = ["body.view", "space"]


def load_from_env(prefix: str, case_sensitive: bool = False) -> dict[str, Any]:
    """Load configuration values from environment variables with the given prefix."""
    env_vars = {}
    for key, value in os.environ.items():
        env_key = key if case_sensitive else key.upper()
        prefix_upper = prefix if case_sensitive else prefix.upper()

        if env_key.startswith(prefix_upper):
            config_key = (
                key[len(prefix) :] if case_sensitive else key[len(prefix) :].lower()
            )
            env_vars[config_key] = value
    return env_vars


# Load configuration values from environment variables.
# Required environment variables: CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN
# Optional environment variables: CONFLUENCE_TIMEOUT, SEARCH_DEFAULT_LIMIT, SEARCH_MAX_LIMIT, SEARCH_DEFAULT_EXPAND
confluence_env = load_from_env("CONFLUENCE_")
search_env = load_from_env("SEARCH_")

# Convert string values to appropriate types for search config
if "default_limit" in search_env:
    search_env["default_limit"] = int(search_env["default_limit"])
if "max_limit" in search_env:
    search_env["max_limit"] = int(search_env["max_limit"])
if "default_expand" in search_env and isinstance(search_env["default_expand"], str):
    search_env["default_expand"] = search_env["default_expand"].split(",")

confluence_config = ConfluenceConfig(**confluence_env)
search_config = SearchConfig(**search_env)
