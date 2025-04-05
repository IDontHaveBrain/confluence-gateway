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

    monkeypatch.setattr(
        "confluence_gateway.core.config.get_user_config_path", lambda: config_path
    )

    def _create_config(content: dict):
        config_path.write_text(json.dumps(content))
        return config_path

    if config_path.exists():
        config_path.unlink()

    return _create_config


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
        assert config.timeout == 10

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
                url="invalid-url",
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
        if Path(get_user_config_path()).exists():
            Path(get_user_config_path()).unlink()

        confluence_config, search_config, vector_db_config, _ = load_configurations()

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

        confluence_config, search_config, vector_db_config, _ = load_configurations()

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

        confluence_config, search_config, vector_db_config, _ = load_configurations()

        assert confluence_config is not None
        assert str(confluence_config.url) == "https://file-confluence.atlassian.net/"
        assert confluence_config.username == "file-user@example.com"
        assert confluence_config.api_token == "file-api-token"
        assert confluence_config.timeout == 30

        assert search_config.default_limit == 50
        assert search_config.max_limit == 300
        assert "body.storage" in search_config.default_expand
        assert "version" in search_config.default_expand
        assert "space" in search_config.default_expand

        assert vector_db_config is not None
        assert vector_db_config.type == "qdrant"
        assert vector_db_config.embedding_dimension == 384
        assert str(vector_db_config.qdrant_url) == "http://localhost:6333/"
        assert vector_db_config.collection_name == "test_collection"

    def test_merged_configuration(self, monkeypatch, mock_user_config_file):
        monkeypatch.setenv("CONFLUENCE_URL", "https://env-confluence.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_USERNAME", "env-user@example.com")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "env-api-token")
        monkeypatch.setenv("SEARCH_DEFAULT_LIMIT", "25")
        monkeypatch.setenv("VECTOR_DB_COLLECTION_NAME", "env_collection")

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

        confluence_config, search_config, vector_db_config, _ = load_configurations()

        assert confluence_config is not None
        assert str(confluence_config.url) == "https://env-confluence.atlassian.net/"
        assert confluence_config.username == "env-user@example.com"
        assert confluence_config.api_token == "env-api-token"
        assert confluence_config.timeout == 30

        assert search_config.default_limit == 25
        assert search_config.max_limit == 300
        assert search_config.default_expand == ["body.view", "space"]

        assert vector_db_config is not None
        assert vector_db_config.type == "chroma"
        assert vector_db_config.embedding_dimension == 768
        assert vector_db_config.collection_name == "env_collection"
        assert vector_db_config.chroma_persist_path == "/tmp/chroma_db"

    def test_invalid_json_file(self, monkeypatch, mock_user_config_file, caplog):
        monkeypatch.setenv("CONFLUENCE_URL", "https://env-confluence.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_USERNAME", "env-user@example.com")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "env-api-token")

        config_path = mock_user_config_file({})
        with open(config_path, "w") as f:
            f.write("This is not valid JSON")

        confluence_config, search_config, vector_db_config, _ = load_configurations()

        assert confluence_config is not None
        assert str(confluence_config.url) == "https://env-confluence.atlassian.net/"
        assert confluence_config.username == "env-user@example.com"
        assert confluence_config.api_token == "env-api-token"

    def test_file_not_json_object(self, monkeypatch, mock_user_config_file, caplog):
        monkeypatch.setenv("CONFLUENCE_URL", "https://env-confluence.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_USERNAME", "env-user@example.com")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "env-api-token")

        config_path = mock_user_config_file({})
        with open(config_path, "w") as f:
            f.write(json.dumps(["item1", "item2"]))

        confluence_config, search_config, vector_db_config, _ = load_configurations()

        assert confluence_config is not None
        assert str(confluence_config.url) == "https://env-confluence.atlassian.net/"
        assert confluence_config.username == "env-user@example.com"
        assert confluence_config.api_token == "env-api-token"

    def test_missing_required_confluence_fields(
        self, monkeypatch, mock_user_config_file
    ):
        for var in ["CONFLUENCE_URL", "CONFLUENCE_USERNAME", "CONFLUENCE_API_TOKEN"]:
            monkeypatch.delenv(var, raising=False)

        mock_user_config_file({"confluence": {"timeout": 30}})

        confluence_config, search_config, vector_db_config, _ = load_configurations()

        assert confluence_config is None
        assert search_config is not None
        assert search_config.default_limit == 20

    def test_vector_db_type_validation(self, monkeypatch, mock_user_config_file):
        mock_user_config_file({"vector_db": {"type": "qdrant"}})

        confluence_config, search_config, vector_db_config, _ = load_configurations()

        assert vector_db_config is None

    def test_vectordb_defaults_to_none(self, monkeypatch, mock_user_config_file):
        for var in [
            "VECTOR_DB_TYPE",
            "VECTOR_DB_EMBEDDING_DIMENSION",
            "VECTOR_DB_COLLECTION_NAME",
        ]:
            monkeypatch.delenv(var, raising=False)

        mock_user_config_file(
            {
                "confluence": {
                    "url": "https://file-confluence.atlassian.net",
                    "username": "file-user@example.com",
                    "api_token": "file-api-token",
                }
            }
        )

        confluence_config, search_config, vector_db_config, _ = load_configurations()

        assert vector_db_config is None
