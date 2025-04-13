import os
import pathlib
from pathlib import Path
from unittest.mock import patch

import pytest
from confluence_gateway.core.config import (
    ConfluenceConfig,
    EmbeddingConfig,
    GenerationConfig,
    IndexingConfig,
    SearchConfig,
    VectorDBConfig,
    get_user_config_path,
    load_configurations,
)
from pydantic import ValidationError


@pytest.fixture
def mock_path_home(mocker, tmp_path):
    return mocker.patch.object(pathlib.Path, "home", return_value=tmp_path)


pytestmark = pytest.mark.usefixtures("mock_path_home")


@pytest.fixture
def mock_env(mocker):
    env_vars = {}

    def _set_env(vars_dict):
        nonlocal env_vars
        env_vars = vars_dict
        mocker.patch.dict(os.environ, env_vars, clear=True)

    return _set_env


@pytest.fixture
def mock_config_file(mocker):
    config_content = {}

    def _set_content(content_dict):
        nonlocal config_content
        config_content = content_dict
        mocker.patch(
            "confluence_gateway.core.config._load_config_from_file",
            return_value=config_content,
        )
        mocker.patch.object(Path, "exists", return_value=bool(config_content))

    mocker.patch(
        "confluence_gateway.core.config._load_config_from_file", return_value={}
    )
    mocker.patch.object(Path, "exists", return_value=False)
    return _set_content


def test_load_defaults(mock_env, mock_config_file):
    mock_env({})
    mock_config_file({})
    (conf_cfg, search_cfg, vdb_cfg, emb_cfg, idx_cfg, gen_cfg) = load_configurations()

    assert conf_cfg is None
    assert isinstance(search_cfg, SearchConfig)
    assert search_cfg.default_limit == 20
    assert vdb_cfg is None
    assert emb_cfg is None
    assert isinstance(idx_cfg, IndexingConfig)
    assert idx_cfg.html_parser == "markitdown"
    assert gen_cfg is None


def test_load_from_env(mock_env, mock_config_file):
    mock_env(
        {
            "CONFLUENCE_URL": "https://test.atlassian.net",
            "CONFLUENCE_USERNAME": "user@test.com",
            "CONFLUENCE_API_TOKEN": "test_token",
            "CONFLUENCE_TIMEOUT": "30",
            "SEARCH_DEFAULT_LIMIT": "50",
            "SEARCH_HYBRID_SEARCH_ENABLED": "true",
            "EMBEDDING_PROVIDER": "sentence-transformers",
            "EMBEDDING_MODEL_NAME": "test-model",
            "EMBEDDING_DIMENSION": "128",
            "VECTOR_DB_TYPE": "qdrant",
            "VECTOR_DB_EMBEDDING_DIMENSION": "128",
            "QDRANT_URL": "http://localhost:6333",
            "INDEXING_INCLUDE_SPACES": "DEV,PROD",
            "INDEXING_HTML_PARSER": "unstructured",
            "GENERATION_ENABLE": "true",
            "GENERATION_MODEL_NAME": "gen-model",
        }
    )
    mock_config_file({})
    (conf_cfg, search_cfg, vdb_cfg, emb_cfg, idx_cfg, gen_cfg) = load_configurations()

    assert isinstance(conf_cfg, ConfluenceConfig)
    assert str(conf_cfg.url) == "https://test.atlassian.net/"
    assert conf_cfg.username == "user@test.com"
    assert conf_cfg.api_token == "test_token"
    assert conf_cfg.timeout == 30

    assert isinstance(search_cfg, SearchConfig)
    assert search_cfg.default_limit == 50
    assert search_cfg.hybrid_search_enabled is True

    assert isinstance(emb_cfg, EmbeddingConfig)
    assert emb_cfg.provider == "sentence-transformers"
    assert emb_cfg.model_name == "test-model"
    assert emb_cfg.dimension == 128

    assert isinstance(vdb_cfg, VectorDBConfig)
    assert vdb_cfg.type == "qdrant"
    assert vdb_cfg.embedding_dimension == 128
    assert str(vdb_cfg.qdrant_url) == "http://localhost:6333/"

    assert isinstance(idx_cfg, IndexingConfig)
    assert idx_cfg.include_spaces == ["DEV", "PROD"]
    assert idx_cfg.html_parser == "unstructured"

    assert isinstance(gen_cfg, GenerationConfig)
    assert gen_cfg.enable is True
    assert gen_cfg.model_name == "gen-model"


def test_load_from_file(mock_env, mock_config_file):
    mock_env({})
    mock_config_file(
        {
            "confluence": {
                "url": "https://file.atlassian.net",
                "username": "file@test.com",
                "api_token": "file_token",
            },
            "search": {
                "default_limit": 15,
            },
            "embedding": {
                "provider": "litellm",
                "model_name": "file-model",
                "dimension": 256,
                "litellm_api_base": "http://localhost:8000",
            },
            "vector_db": {
                "type": "chroma",
                "embedding_dimension": 256,
                "chroma_persist_path": "/data",
            },
            "indexing": {"exclude_spaces": ["ARCHIVE"]},
            "generation": {"enable": True, "model_name": "file-gen-model"},
        }
    )
    (conf_cfg, search_cfg, vdb_cfg, emb_cfg, idx_cfg, gen_cfg) = load_configurations()

    assert isinstance(conf_cfg, ConfluenceConfig)
    assert str(conf_cfg.url) == "https://file.atlassian.net/"
    assert conf_cfg.username == "file@test.com"

    assert search_cfg.default_limit == 15

    assert isinstance(emb_cfg, EmbeddingConfig)
    assert emb_cfg.provider == "litellm"
    assert emb_cfg.model_name == "file-model"
    assert emb_cfg.dimension == 256
    assert str(emb_cfg.litellm_api_base) == "http://localhost:8000/"

    assert isinstance(vdb_cfg, VectorDBConfig)
    assert vdb_cfg.type == "chroma"
    assert vdb_cfg.embedding_dimension == 256
    assert vdb_cfg.chroma_persist_path == "/data"

    assert idx_cfg.exclude_spaces == ["ARCHIVE"]

    assert isinstance(gen_cfg, GenerationConfig)
    assert gen_cfg.enable is True
    assert gen_cfg.model_name == "file-gen-model"


def test_env_overrides_file(mock_env, mock_config_file):
    mock_env(
        {
            "CONFLUENCE_URL": "https://env.atlassian.net",
            "CONFLUENCE_USERNAME": "env@test.com",
            "CONFLUENCE_API_TOKEN": "env_token",
            "SEARCH_DEFAULT_LIMIT": "99",
        }
    )
    mock_config_file(
        {
            "confluence": {
                "url": "https://file.atlassian.net",
                "username": "file@test.com",
                "api_token": "file_token",
            },
            "search": {
                "default_limit": 15,
            },
        }
    )
    (conf_cfg, search_cfg, _, _, _, _) = load_configurations()

    assert isinstance(conf_cfg, ConfluenceConfig)
    assert str(conf_cfg.url) == "https://env.atlassian.net/"
    assert conf_cfg.username == "env@test.com"
    assert conf_cfg.api_token == "env_token"

    assert search_cfg.default_limit == 99


def test_missing_required_confluence_config(mock_env, mock_config_file):
    mock_env(
        {
            "CONFLUENCE_URL": "https://test.atlassian.net",
        }
    )
    mock_config_file({})
    (conf_cfg, _, _, _, _, _) = load_configurations()
    assert conf_cfg is None


def test_invalid_value_types(mock_env, mock_config_file, caplog):
    mock_env(
        {
            "CONFLUENCE_URL": "https://test.atlassian.net",
            "CONFLUENCE_USERNAME": "user@test.com",
            "CONFLUENCE_API_TOKEN": "test_token",
            "CONFLUENCE_TIMEOUT": "not-an-integer",
            "SEARCH_DEFAULT_LIMIT": "also-not-int",
        }
    )
    mock_config_file({})

    with patch("logging.Logger.warning") as mock_warning:
        (conf_cfg, search_cfg, _, _, _, _) = load_configurations()

        assert conf_cfg.timeout == 10
        assert search_cfg.default_limit == 20

        assert mock_warning.call_count >= 2


def test_embedding_dimension_propagation(mock_env, mock_config_file):
    mock_env(
        {
            "EMBEDDING_PROVIDER": "sentence-transformers",
            "EMBEDDING_MODEL_NAME": "test-model",
            "EMBEDDING_DIMENSION": "384",
            "VECTOR_DB_TYPE": "qdrant",
            "QDRANT_URL": ":memory:",
        }
    )
    mock_config_file({})
    (_, _, vdb_cfg, emb_cfg, _, _) = load_configurations()

    assert emb_cfg is not None
    assert emb_cfg.dimension == 384
    assert vdb_cfg is not None
    assert vdb_cfg.embedding_dimension == 384


def test_embedding_dimension_mismatch_warning(mock_env, mock_config_file, caplog):
    mock_env(
        {
            "EMBEDDING_PROVIDER": "sentence-transformers",
            "EMBEDDING_MODEL_NAME": "test-model",
            "EMBEDDING_DIMENSION": "384",
            "VECTOR_DB_TYPE": "qdrant",
            "QDRANT_URL": ":memory:",
            "VECTOR_DB_EMBEDDING_DIMENSION": "768",
        }
    )
    mock_config_file({})

    with patch("logging.Logger.warning") as mock_warning:
        (_, _, vdb_cfg, emb_cfg, _, _) = load_configurations()

        assert emb_cfg.dimension == 384
        assert vdb_cfg.embedding_dimension == 768
        assert mock_warning.call_count >= 1
        assert any(
            "differs from EMBEDDING_DIMENSION" in call.args[0]
            for call in mock_warning.call_args_list
        )


def test_boolean_env_var_parsing(mock_env, mock_config_file):
    mock_env(
        {
            "SEARCH_HYBRID_SEARCH_ENABLED": "true",
            "INDEXING_INCLUDE_ATTACHMENTS": "1",
            "GENERATION_ENABLE": "false",
            "QDRANT_PREFER_GRPC": "0",
        }
    )
    mock_config_file({})
    (_, search_cfg, vdb_cfg, _, idx_cfg, gen_cfg) = load_configurations()

    assert search_cfg.hybrid_search_enabled is True
    assert idx_cfg.include_attachments is True

    mock_env(
        {
            "SEARCH_HYBRID_SEARCH_ENABLED": "true",
            "INDEXING_INCLUDE_ATTACHMENTS": "1",
            "GENERATION_ENABLE": "false",
            "GENERATION_MODEL_NAME": "gen-model",
            "QDRANT_PREFER_GRPC": "0",
            "VECTOR_DB_TYPE": "qdrant",
            "VECTOR_DB_EMBEDDING_DIMENSION": "128",
            "QDRANT_URL": ":memory:",
        }
    )
    (_, search_cfg, vdb_cfg, _, idx_cfg, gen_cfg) = load_configurations()

    assert search_cfg.hybrid_search_enabled is True
    assert idx_cfg.include_attachments is True
    assert gen_cfg is None
    assert vdb_cfg.qdrant_prefer_grpc is False


def test_comma_list_env_var_parsing(mock_env, mock_config_file):
    mock_env(
        {
            "INDEXING_INCLUDE_SPACES": "DEV, PROD , TEST ",
            "INDEXING_ALLOWED_ATTACHMENT_EXTENSIONS": "pdf,docx",
        }
    )
    mock_config_file({})
    (_, _, _, _, idx_cfg, _) = load_configurations()

    assert idx_cfg.include_spaces == ["DEV", "PROD", "TEST"]
    assert idx_cfg.allowed_attachment_extensions == ["pdf", "docx"]


import logging


def test_ollama_litellm_requires_api_base(mock_env, mock_config_file, caplog):
    mock_env(
        {
            "EMBEDDING_PROVIDER": "litellm",
            "EMBEDDING_MODEL_NAME": "ollama/nomic-embed-text",
            "EMBEDDING_DIMENSION": "768",
        }
    )
    mock_config_file({})

    with caplog.at_level(logging.ERROR):
        (_, _, _, emb_cfg, _, _) = load_configurations()

    assert emb_cfg is None
    assert "Invalid Embedding configuration" in caplog.text
    assert "LITELLM_API_BASE must be set" in caplog.text


def test_generation_requires_model_if_enabled(mock_env, mock_config_file, caplog):
    mock_env(
        {
            "GENERATION_ENABLE": "true",
        }
    )
    mock_config_file({})

    with caplog.at_level(logging.ERROR):
        (_, _, _, _, _, gen_cfg) = load_configurations()

    assert gen_cfg is None
    assert "Invalid Generation configuration" in caplog.text
    assert "GENERATION_MODEL_NAME must be set" in caplog.text
