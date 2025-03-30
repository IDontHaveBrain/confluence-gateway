import pytest
from confluence_gateway.core.config import (
    ConfluenceConfig,
    SearchConfig,
    load_from_env,
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


class TestLoadFromEnv:
    def test_load_from_env_with_prefix(self, monkeypatch):
        monkeypatch.setenv("TEST_URL", "https://example.atlassian.net")
        monkeypatch.setenv("TEST_USERNAME", "test@example.com")
        monkeypatch.setenv("TEST_API_TOKEN", "test-token")
        monkeypatch.setenv("OTHER_VAR", "other-value")

        env_vars = load_from_env("TEST_")
        assert env_vars == {
            "url": "https://example.atlassian.net",
            "username": "test@example.com",
            "api_token": "test-token",
        }

    def test_load_from_env_case_sensitive(self, monkeypatch):
        import platform

        monkeypatch.setenv("Test_Url", "https://example.atlassian.net")
        monkeypatch.setenv("test_Username", "test@example.com")
        monkeypatch.setenv("TEST_API_TOKEN", "test-token")

        # Case-insensitive (default)
        env_vars = load_from_env("TEST_")
        assert env_vars == {
            "url": "https://example.atlassian.net",
            "username": "test@example.com",
            "api_token": "test-token",
        }

        # Case-sensitive test depends on platform
        is_windows = platform.system().lower() == "windows"
        env_vars = load_from_env("Test_", case_sensitive=True)

        if is_windows:
            # On Windows, environment variables are case-insensitive
            # So we'll check that we get some results, possibly all of them
            assert "url" in env_vars
            assert env_vars["url"] == "https://example.atlassian.net"
        else:
            # On Unix/Linux, we expect exact case matching
            assert env_vars == {
                "Url": "https://example.atlassian.net",
            }

    def test_empty_prefix(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "test-value")
        monkeypatch.setenv("ANOTHER_VAR", "another-value")

        env_vars = load_from_env("")
        assert "test_var" in env_vars
        assert "another_var" in env_vars

    def test_config_from_environment(self, mock_confluence_env, mock_search_env):
        from confluence_gateway.core.config import (
            load_confluence_config_from_env,
            load_search_config_from_env,
        )

        # Load configs using the proper loading functions
        confluence_config = load_confluence_config_from_env()
        search_config = load_search_config_from_env()

        # Verify Confluence config
        assert str(confluence_config.url) == "https://test-confluence.atlassian.net/"
        assert confluence_config.username == "test-user@example.com"
        assert confluence_config.api_token == "test-api-token-123"
        assert confluence_config.timeout == 15

        # Verify Search config
        assert search_config.default_limit == 25
        assert search_config.max_limit == 200
        assert search_config.default_expand == ["body.storage", "version", "space"]
