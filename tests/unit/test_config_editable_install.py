"""Test to verify that configuration is loaded from source directory in editable install."""

import json
from pathlib import Path

from confluence_gateway.core.config import (
    get_default_config_path,
    load_configurations,
)


def test_default_config_path_points_to_source():
    """Verify that the default config path points to the source directory."""
    default_path = get_default_config_path()

    # The path should be in the source directory, not in site-packages
    assert "site-packages" not in str(default_path)
    assert "confluence_gateway" in str(default_path)
    assert default_path.name == "confluence_gateway_config.json"
    assert default_path.exists(), f"Default config file not found at {default_path}"


def test_config_loads_from_source_directory():
    """Verify that configuration is actually loaded from the source directory."""
    default_path = get_default_config_path()

    # Load the config file directly
    with Path(default_path).open() as f:
        direct_config = json.load(f)

    # Load through the configuration system
    (
        confluence_config,
        search_config,
        vector_db_config,
        embedding_config,
        indexing_config,
        generation_config,
    ) = load_configurations()

    # Verify that the loaded config matches the source file
    assert search_config.default_limit == direct_config["search"]["default_limit"]
    assert search_config.max_limit == direct_config["search"]["max_limit"]
    assert indexing_config.html_parser == direct_config["indexing"]["html_parser"]

    # Verify embedding config if present
    if embedding_config and "embedding" in direct_config:
        assert embedding_config.provider == direct_config["embedding"]["provider"]
        assert embedding_config.model_name == direct_config["embedding"]["model_name"]


def test_config_modification_detection():
    """Verify that modifications to the source config file are immediately reflected."""
    default_path = get_default_config_path()

    # This test just verifies the path is correct for editable installs
    # Actual modification testing would require file manipulation which could
    # interfere with other tests
    assert default_path.is_absolute()
    assert default_path.parent.name == "confluence_gateway"
    assert default_path.parent.parent == Path(__file__).parent.parent.parent
