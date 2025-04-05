import json
from pathlib import Path

import pytest
from confluence_gateway.core.config import (
    ConfluenceConfig,
    SearchConfig,
    get_user_config_path,
    load_configurations,
)
from pydantic import ValidationError


@pytest.fixture
def mock_confluence_env(monkeypatch):
    env_vars = {
        "CONFLUENCE_URL": "https://test-confluence.atlassian.net",
        "CONFLUENCE_USERNAME": "test-user@example.com",
        "CONFLUENCE_API_TOKEN": "test-api-token-123",
        "CONFLUENCE_TIMEOUT": "15",
    }
    for name, value in env_vars.items():
        monkeypatch.setenv(name, value)
    return env_vars


@pytest.fixture
def mock_search_env(monkeypatch):
    env_vars = {
        "SEARCH_DEFAULT_LIMIT": "25",
        "SEARCH_MAX_LIMIT": "200",
        "SEARCH_DEFAULT_EXPAND": "body.storage,version,space",
    }
    for name, value in env_vars.items():
        monkeypatch.setenv(name, value)
    return env_vars


@pytest.fixture
def mock_vector_db_env(monkeypatch):
    env_vars = {
        "VECTOR_DB_TYPE": "chroma",
        "VECTOR_DB_EMBEDDING_DIMENSION": "768",
        "VECTOR_DB_COLLECTION_NAME": "test_collection",
        "CHROMA_PERSIST_PATH": "/tmp/chroma",
    }
    for name, value in env_vars.items():
        monkeypatch.setenv(name, value)
    return env_vars


@pytest.fixture
def mock_user_config_file(tmp_path, monkeypatch):
    config_path = tmp_path / ".confluence_gateway_config.json"

    # Patch the function to return the temp path
    monkeypatch.setattr(
        "confluence_gateway.core.config.get_user_config_path", lambda: config_path
    )

    def _create_config(content: dict):
        config_path.write_text(json.dumps(content))
        return config_path

    # Ensure the file doesn't exist initially for some tests
    if config_path.exists():
        config_path.unlink()

    return _create_config  # Return a function to create the file content on demand


class TestConfluenceConfig:
    def test_valid_config(self):
        config = ConfluenceConfig(
            url="https://example.atlassian.net",
            username="test@example.com",
            api_token="test-token",
        )
        assert str(config.url) == "https://example.atlassian.net/"
        assert config.username == "test@example.com"
        assert config.api_token == "test-token"
        assert config.timeout == 10  # Default value

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            ConfluenceConfig(username="test@example.com", api_token="test-token")

        with pytest.raises(ValidationError):
            ConfluenceConfig(
                url="https://example.atlassian.net", api_token="test-token"
            )

        with pytest.raises(ValidationError):
            ConfluenceConfig(
                url="https://example.atlassian.net", username="test@example.com"
            )

    def test_invalid_url(self):
        with pytest.raises(ValidationError):
            ConfluenceConfig(
                url="invalid-url",  # Not a valid URL format
                username="test@example.com",
                api_token="test-token",
            )

    def test_custom_timeout(self):
        config = ConfluenceConfig(
            url="https://example.atlassian.net",
            username="test@example.com",
            api_token="test-token",
            timeout=30,
        )
        assert config.timeout == 30


class TestSearchConfig:
    def test_default_values(self):
        config = SearchConfig()
        assert config.default_limit == 20
        assert config.max_limit == 100
        assert config.default_expand == ["body.view", "space"]

    def test_custom_values(self):
        config = SearchConfig(
            default_limit=10,
            max_limit=50,
            default_expand=["body.storage", "version"],
        )
        assert config.default_limit == 10
        assert config.max_limit == 50
        assert config.default_expand == ["body.storage", "version"]


class TestLoadConfigurations:
    def test_load_from_environment_only(
        self,
        mock_confluence_env,
        mock_search_env,
        mock_vector_db_env,
        mock_user_config_file,
    ):
        # Don't create the config file, just make sure it doesn't exist
        if Path(get_user_config_path()).exists():
            Path(get_user_config_path()).unlink()

        # Load configurations
        confluence_config, search_config, vector_db_config, _ = load_configurations()

        # Verify configurations loaded from environment
        assert confluence_config is not None
        assert str(confluence_config.url) == "https://test-confluence.atlassian.net/"
        assert confluence_config.username == "test-user@example.com"
        assert confluence_config.api_token == "test-api-token-123"
        assert confluence_config.timeout == 15

        assert search_config.default_limit == 25
        assert search_config.max_limit == 200
        assert search_config.default_expand == ["body.storage", "version", "space"]

        assert vector_db_config is not None
        assert vector_db_config.type == "chroma"
        assert vector_db_config.embedding_dimension == 768
        assert vector_db_config.collection_name == "test_collection"
        assert vector_db_config.chroma_persist_path == "/tmp/chroma"

    def test_load_from_file_only(self, monkeypatch, mock_user_config_file):
        # Clear environment variables
        for var in [
            "CONFLUENCE_URL",
            "CONFLUENCE_USERNAME",
            "CONFLUENCE_API_TOKEN",
            "CONFLUENCE_TIMEOUT",
            "SEARCH_DEFAULT_LIMIT",
            "SEARCH_MAX_LIMIT",
            "SEARCH_DEFAULT_EXPAND",
            "VECTOR_DB_TYPE",
            "VECTOR_DB_EMBEDDING_DIMENSION",
        ]:
            monkeypatch.delenv(var, raising=False)

        # Create config file
        mock_user_config_file(
            {
                "confluence": {
                    "url": "https://file-confluence.atlassian.net",
                    "username": "file-user@example.com",
                    "api_token": "file-api-token",
                    "timeout": 30,
                },
                "search": {
                    "default_limit": 50,
                    "max_limit": 300,
                    "default_expand": ["body.view", "version"],
                },
                "vector_db": {
                    "type": "qdrant",
                    "embedding_dimension": 384,
                    "collection_name": "file_collection",
                    "qdrant_url": "http://localhost:6333",
                },
            }
        )

        # Load configurations
        confluence_config, search_config, vector_db_config, _ = load_configurations()

        # Verify configurations loaded from file
        assert confluence_config is not None
        assert str(confluence_config.url) == "https://file-confluence.atlassian.net/"
        assert confluence_config.username == "file-user@example.com"
        assert confluence_config.api_token == "file-api-token"
        assert confluence_config.timeout == 30

        assert search_config.default_limit == 50
        assert search_config.max_limit == 300
        assert search_config.default_expand == ["body.view", "version"]

        assert vector_db_config is not None
        assert vector_db_config.type == "qdrant"
        assert vector_db_config.embedding_dimension == 384
        assert vector_db_config.collection_name == "file_collection"
        assert str(vector_db_config.qdrant_url) == "http://localhost:6333/"

    def test_file_precedence_over_environment(
        self,
        mock_confluence_env,
        mock_search_env,
        mock_vector_db_env,
        mock_user_config_file,
    ):
        # Create config file with different values
        mock_user_config_file(
            {
                "confluence": {
                    "url": "https://file-confluence.atlassian.net",
                    "username": "file-user@example.com",
                    "api_token": "file-api-token",
                    "timeout": 30,
                },
                "search": {"default_limit": 50, "max_limit": 300},
                "vector_db": {
                    "type": "qdrant",
                    "embedding_dimension": 384,
                    "qdrant_url": "http://localhost:6333",
                },
            }
        )

        # Load configurations
        confluence_config, search_config, vector_db_config, _ = load_configurations()

        # Verify file values take precedence
        assert confluence_config is not None
        assert str(confluence_config.url) == "https://file-confluence.atlassian.net/"
        assert confluence_config.username == "file-user@example.com"
        assert confluence_config.api_token == "file-api-token"
        assert confluence_config.timeout == 30

        assert search_config.default_limit == 50
        assert search_config.max_limit == 300
        # Default expand should be from environment as it's not in the file
        assert "body.storage" in search_config.default_expand
        assert "version" in search_config.default_expand
        assert "space" in search_config.default_expand

        assert vector_db_config is not None
        assert vector_db_config.type == "qdrant"
        assert vector_db_config.embedding_dimension == 384
        assert str(vector_db_config.qdrant_url) == "http://localhost:6333/"
        # Collection name from environment as it's not in file
        assert vector_db_config.collection_name == "test_collection"

    def test_merged_configuration(self, monkeypatch, mock_user_config_file):
        # Set some environment variables
        monkeypatch.setenv("CONFLUENCE_URL", "https://env-confluence.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_USERNAME", "env-user@example.com")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "env-api-token")
        monkeypatch.setenv("SEARCH_DEFAULT_LIMIT", "25")
        monkeypatch.setenv("VECTOR_DB_COLLECTION_NAME", "env_collection")

        # Create config file with different values
        mock_user_config_file(
            {
                "confluence": {"timeout": 30},
                "search": {"max_limit": 300},
                "vector_db": {
                    "type": "chroma",
                    "embedding_dimension": 768,
                    "chroma_persist_path": "/tmp/chroma_db",
                },
            }
        )

        # Load configurations
        confluence_config, search_config, vector_db_config, _ = load_configurations()

        # Verify merged configuration
        assert confluence_config is not None
        assert (
            str(confluence_config.url) == "https://env-confluence.atlassian.net/"
        )  # From env
        assert confluence_config.username == "env-user@example.com"  # From env
        assert confluence_config.api_token == "env-api-token"  # From env
        assert confluence_config.timeout == 30  # From file

        assert search_config.default_limit == 25  # From env
        assert search_config.max_limit == 300  # From file
        assert search_config.default_expand == ["body.view", "space"]  # Default

        assert vector_db_config is not None
        assert vector_db_config.type == "chroma"  # From file
        assert vector_db_config.embedding_dimension == 768  # From file
        assert vector_db_config.collection_name == "env_collection"  # From env
        assert vector_db_config.chroma_persist_path == "/tmp/chroma_db"  # From file

    def test_invalid_json_file(self, monkeypatch, mock_user_config_file, caplog):
        # Set environment variables as fallback
        monkeypatch.setenv("CONFLUENCE_URL", "https://env-confluence.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_USERNAME", "env-user@example.com")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "env-api-token")

        # Create an invalid JSON file
        config_path = mock_user_config_file({})  # Get the path
        with open(config_path, "w") as f:
            f.write("This is not valid JSON")

        # Load configurations - should fall back to env vars
        confluence_config, search_config, vector_db_config, _ = load_configurations()

        # Verify fallback to environment
        assert confluence_config is not None
        assert str(confluence_config.url) == "https://env-confluence.atlassian.net/"
        assert confluence_config.username == "env-user@example.com"
        assert confluence_config.api_token == "env-api-token"

    def test_file_not_json_object(self, monkeypatch, mock_user_config_file, caplog):
        # Set environment variables as fallback
        monkeypatch.setenv("CONFLUENCE_URL", "https://env-confluence.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_USERNAME", "env-user@example.com")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "env-api-token")

        # Create a JSON file that's not an object (array instead)
        config_path = mock_user_config_file({})  # Get the path
        with open(config_path, "w") as f:
            f.write(json.dumps(["item1", "item2"]))

        # Load configurations - should fall back to env vars
        confluence_config, search_config, vector_db_config, _ = load_configurations()

        # Verify fallback to environment
        assert confluence_config is not None
        assert str(confluence_config.url) == "https://env-confluence.atlassian.net/"
        assert confluence_config.username == "env-user@example.com"
        assert confluence_config.api_token == "env-api-token"

    def test_missing_required_confluence_fields(
        self, monkeypatch, mock_user_config_file
    ):
        # Clear environment variables
        for var in ["CONFLUENCE_URL", "CONFLUENCE_USERNAME", "CONFLUENCE_API_TOKEN"]:
            monkeypatch.delenv(var, raising=False)

        # Create config without required fields
        mock_user_config_file({"confluence": {"timeout": 30}})

        # Load configurations
        confluence_config, search_config, vector_db_config, _ = load_configurations()

        # Verify confluence_config is None due to missing required fields
        assert confluence_config is None
        # Search config should still be loaded with defaults
        assert search_config is not None
        assert search_config.default_limit == 20  # Default

    def test_vector_db_type_validation(self, monkeypatch, mock_user_config_file):
        # Set vector db with type but missing required fields
        mock_user_config_file(
            {
                "vector_db": {
                    "type": "qdrant"  # Missing embedding_dimension and qdrant_url
                }
            }
        )

        # Load configurations
        confluence_config, search_config, vector_db_config, _ = load_configurations()

        # Vector DB config should be None due to validation errors
        assert vector_db_config is None

    def test_vectordb_defaults_to_none(self, monkeypatch, mock_user_config_file):
        # Clear all vector db env vars
        for var in [
            "VECTOR_DB_TYPE",
            "VECTOR_DB_EMBEDDING_DIMENSION",
            "VECTOR_DB_COLLECTION_NAME",
        ]:
            monkeypatch.delenv(var, raising=False)

        # Create minimal config without vector_db
        mock_user_config_file(
            {
                "confluence": {
                    "url": "https://file-confluence.atlassian.net",
                    "username": "file-user@example.com",
                    "api_token": "file-api-token",
                }
            }
        )

        # Load configurations
        confluence_config, search_config, vector_db_config, _ = load_configurations()

        # VectorDBConfig should be None as type defaults to 'none'
        assert vector_db_config is None
