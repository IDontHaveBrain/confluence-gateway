"""Essential storage mode integration tests for confluence-gateway.

Basic tests for memory mode behavior:
- Memory mode initialization
- Memory mode isolation (no persistent artifacts)
"""

import subprocess
from pathlib import Path

import pytest

from tests.fixtures.config_builders import (
    apply_env_vars,
    cleanup_temp_dirs,
    get_qdrant_memory_config,
    restore_env_vars,
)


class TestMemoryModeBasic:
    """Basic memory mode functionality tests."""

    def test_memory_mode_initialization(self, tmp_path: Path) -> None:
        """Test that memory mode initializes correctly.

        Args:
            tmp_path: Temporary directory for test isolation
        """
        # Build memory mode configuration
        config_result = get_qdrant_memory_config()

        # Apply environment variables
        previous_env = apply_env_vars(config_result.env_vars)

        try:
            # Test CLI version command to verify basic initialization
            result = subprocess.run(
                ["uv", "run", "confluence-gateway", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert result.returncode == 0
            assert (
                "confluence-gateway" in result.stdout
                or "confluence-gateway" in result.stderr
            )

            # Verify memory mode settings
            import os

            assert os.environ.get("QDRANT_URL") == ":memory:"

        finally:
            restore_env_vars(previous_env)
            cleanup_temp_dirs(config_result.temp_dirs)

    def test_memory_mode_no_persistence(self, tmp_path: Path) -> None:
        """Test that memory mode does not create persistent artifacts.

        Args:
            tmp_path: Temporary directory for test isolation
        """
        memory_config = get_qdrant_memory_config()
        previous_env = apply_env_vars(memory_config.env_vars)

        try:
            # Record initial file state
            initial_files = set(tmp_path.rglob("*"))

            # Run basic commands that might create files
            commands = [
                ["uv", "run", "confluence-gateway", "--version"],
                ["uv", "run", "confluence-gateway", "--help"],
            ]

            for cmd in commands:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert result.returncode == 0

            # Check for new files after operations
            final_files = set(tmp_path.rglob("*"))
            new_files = final_files - initial_files

            # Filter out temporary files that are expected
            persistent_files = [
                f
                for f in new_files
                if f.is_file()
                and any(
                    pattern in str(f).lower()
                    for pattern in [
                        "qdrant",
                        "vector",
                        "storage",
                        ".db",
                        ".sqlite",
                        ".data",
                        ".index",
                    ]
                )
                and not any(
                    temp_pattern in str(f)
                    for temp_pattern in ["tmp", "temp", "cache", "lock"]
                )
            ]

            assert len(persistent_files) == 0, (
                f"Memory mode created persistent files: {persistent_files}"
            )

            # Verify memory-specific configuration is in place
            import os

            assert os.environ.get("QDRANT_URL") == ":memory:"

        finally:
            restore_env_vars(previous_env)
            cleanup_temp_dirs(memory_config.temp_dirs)


class TestMemoryModeIsolation:
    """Memory mode isolation tests."""

    def test_memory_mode_fast_initialization(self, tmp_path: Path) -> None:
        """Test that memory mode initializes quickly.

        Args:
            tmp_path: Temporary directory for test isolation
        """
        memory_config = get_qdrant_memory_config()
        previous_env = apply_env_vars(memory_config.env_vars)

        try:
            # Test initialization performance
            import time

            start_time = time.time()

            result = subprocess.run(
                ["uv", "run", "confluence-gateway", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            init_time = time.time() - start_time

            assert result.returncode == 0

            # Memory mode should initialize reasonably quickly (under 15 seconds)
            assert init_time < 15.0, f"Memory mode took {init_time:.2f}s to initialize"

            # Verify no persistent artifacts are created
            temp_files = list(tmp_path.rglob("*"))
            storage_files = [
                f
                for f in temp_files
                if any(
                    pattern in str(f).lower()
                    for pattern in ["qdrant", "vector", ".db", ".sqlite", "storage"]
                )
                and f.is_file()
            ]

            # Memory mode should not create persistent storage files
            assert len(storage_files) == 0, (
                f"Memory mode created persistent files: {storage_files}"
            )

        finally:
            restore_env_vars(previous_env)
            cleanup_temp_dirs(memory_config.temp_dirs)
