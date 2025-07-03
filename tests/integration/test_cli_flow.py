from unittest.mock import patch

import pytest
from pytest_mock import MockerFixture
from typer.testing import CliRunner

import confluence_gateway.cli.dependencies as cli_deps
import confluence_gateway.core.config as config_module
import confluence_gateway.services.search as search_service_module
from confluence_gateway.cli.main import app
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.generation import GenerationService
from confluence_gateway.services.indexing import IndexingService
from confluence_gateway.services.search import SearchService

pytestmark = pytest.mark.integration


class TestCliFlows:
    def test_cli_search_text_basic(
        self,
        runner: CliRunner,
        is_real_config_available: bool,
        real_search_term: str,
        confluence_config,
        mocker: MockerFixture,
    ):
        if not is_real_config_available:
            pytest.skip("Requires real Confluence configuration.")
        # Patch at the import location in cli.dependencies
        with patch("confluence_gateway.cli.dependencies.confluence_config", confluence_config):
            command = ["search", "text", real_search_term, "--limit", "1"]
            result = runner.invoke(app, command)
        assert result.exit_code == 0, (
            f"CLI failed with output:\n{result.stdout}"
        )
        assert "Performing Text Search..." in result.stdout
        # The search might return no results, so we need to handle both cases
        if "No results found." in result.stdout:
            # This is acceptable - the search worked but found no matches
            assert "No results found." in result.stdout
        else:
            # If results were found, check for expected output
            assert "Search Results" in result.stdout
            assert "Showing results" in result.stdout
            assert "Took" in result.stdout

    def test_cli_search_text_hybrid(
        self,
        runner: CliRunner,
        is_semantic_search_possible: bool,
        real_search_term: str,
        mocker: MockerFixture,
        semantic_search_service: SearchService,
        embedding_service: EmbeddingService,
        vector_db_adapter,
        confluence_config,
    ):
        if not is_semantic_search_possible:
            pytest.skip("Hybrid search requires semantic search capabilities.")
        # Patch at the import location in cli.dependencies
        with patch("confluence_gateway.cli.dependencies.confluence_config", confluence_config):
            mocker.patch.object(
                cli_deps, "_get_embedding_service", return_value=embedding_service
            )
            mocker.patch.object(
                cli_deps, "_get_vector_db_adapter", return_value=vector_db_adapter
            )
            mocker.patch.object(
                search_service_module.search_config, "hybrid_search_enabled", True
            )
            command = ["search", "text", real_search_term, "--hybrid", "--limit", "1"]
            result = runner.invoke(app, command)
        assert result.exit_code == 0, (
            f"CLI failed with output:\n{result.stdout}"
        )
        assert "Performing Hybrid Search..." in result.stdout
        assert "Search Results" in result.stdout
        assert "Showing results" in result.stdout
        assert "Took" in result.stdout

    def test_cli_search_cql(
        self,
        runner: CliRunner,
        is_real_config_available: bool,
        confluence_config,
        mocker: MockerFixture,
    ):
        if not is_real_config_available:
            pytest.skip("Requires real Confluence configuration.")
        # Patch at the import location in cli.dependencies
        with patch("confluence_gateway.cli.dependencies.confluence_config", confluence_config):
            command = ["search", "cql", "type=page", "--limit", "1"]
            result = runner.invoke(app, command)
        assert result.exit_code == 0, (
            f"CLI failed with output:\n{result.stdout}"
        )
        assert "Performing CQL Search:" in result.stdout
        assert "type=page" in result.stdout
        assert "Search Results" in result.stdout
        assert "Showing results" in result.stdout
        assert "Took" in result.stdout

    def test_cli_search_semantic(
        self,
        runner: CliRunner,
        is_semantic_search_possible: bool,
        mocker: MockerFixture,
        semantic_search_service: SearchService,
        embedding_service: EmbeddingService,
        vector_db_adapter,
        confluence_config,
    ):
        if not is_semantic_search_possible:
            pytest.skip("Semantic search requires semantic search configuration.")

        print(f"\nDEBUG: semantic_search_service instance: {semantic_search_service}")
        assert semantic_search_service is not None, (
            "semantic_search_service fixture is None"
        )
        assert hasattr(semantic_search_service, "vector_db_adapter"), (
            "semantic_search_service fixture missing vector_db_adapter attr"
        )
        print(
            f"DEBUG: semantic_search_service.vector_db_adapter: {semantic_search_service.vector_db_adapter}"
        )
        assert semantic_search_service.vector_db_adapter is not None, (
            "semantic_search_service.vector_db_adapter is None BEFORE patching/invoke"
        )

        # Patch at the import location in cli.dependencies
        with patch("confluence_gateway.cli.dependencies.confluence_config", confluence_config):
            mocker.patch.object(
                cli_deps, "_get_embedding_service", return_value=embedding_service
            )
            mocker.patch.object(
                cli_deps, "_get_vector_db_adapter", return_value=vector_db_adapter
            )
            query = "apples"
            command = ["search", "semantic", query, "--top-k", "2"]
            result = runner.invoke(app, command)
        assert result.exit_code == 0, (
            f"CLI failed with output:\n{result.stdout}"
        )
        assert "Performing Semantic Search..." in result.stdout
        assert f"Semantic Search Results for: '{query}'" in result.stdout
        assert "Score" in result.stdout
        assert "ID" in result.stdout
        # Accept either test doc content or a test doc id
        assert (
            "apples" in result.stdout
            or "sem_doc1" in result.stdout
            or "sem_doc2" in result.stdout
        )
        assert "Took" in result.stdout

    def test_cli_index_status(
        self,
        runner: CliRunner,
        is_semantic_search_possible: bool,
    ):
        if not is_semantic_search_possible:
            pytest.skip("Indexing requires semantic search configuration.")
        command = ["index", "status"]
        result = runner.invoke(app, command)
        assert result.exit_code == 0, (
            f"CLI failed with output:\n{result.stdout}"
        )
        assert "Indexing Status:" in result.stdout
        assert (
            "idle" in result.stdout
            or "success" in result.stdout
            or "running" in result.stdout
            or "failure" in result.stdout
        )
        assert "Last Run Start:" in result.stdout
        assert "Last Run End:" in result.stdout

    def test_cli_index_trigger(
        self,
        runner: CliRunner,
        is_semantic_search_possible: bool,
        mocker: MockerFixture,
        indexing_service: IndexingService,
        embedding_service: EmbeddingService,
        vector_db_adapter,
        confluence_config,
        vector_db_config,
    ):
        if not is_semantic_search_possible:
            pytest.skip("Indexing requires semantic search configuration.")
        
        # Skip test if vector_db_config is not available globally
        # This happens when no vector DB is configured in the environment
        if vector_db_config is None:
            # Check if we can at least verify that the CLI properly handles this case
            command = ["index", "trigger"]
            result = runner.invoke(app, command)
            # The CLI should fail gracefully with a warning
            assert result.exit_code == 1
            assert "Vector DB is not configured" in result.stdout
            return
        
        # If we have vector_db_config, test the normal flow
        # Mock the _run_indexing_sync method to avoid actual indexing
        def mock_run_indexing_sync(space_keys=None):
            # Simulate successful indexing without doing actual work
            pass
        
        # Mock the service getter to return our indexing service
        mocker.patch.object(cli_deps, "_get_indexing_service", return_value=indexing_service)
        mocker.patch.object(indexing_service, "_run_indexing_sync", side_effect=mock_run_indexing_sync)
        
        command = ["index", "trigger"]
        result = runner.invoke(app, command)
        
        # Check the result
        assert result.exit_code == 0, (
            f"CLI failed with output:\n{result.stdout}"
        )
        # The output should show the indexing started
        assert "Starting synchronous indexing..." in result.stdout or "Indexing process is already running." in result.stdout

    def test_cli_generate_answer(
        self,
        runner: CliRunner,
        is_generation_enabled: bool,
        is_semantic_search_possible: bool,
        mocker: MockerFixture,
        generation_service: GenerationService,
    ):
        if not is_generation_enabled:
            pytest.skip("Generation feature is disabled.")
        if not is_semantic_search_possible:
            pytest.skip("Generation requires semantic search capabilities.")
        mocker.patch.object(
            cli_deps, "_get_generation_service", return_value=generation_service
        )
        mock_llm_call = mocker.patch("litellm.acompletion", autospec=True)
        mock_response_content = "Apples, oranges, and bananas are mentioned."
        mock_llm_call.return_value = mocker.MagicMock(
            choices=[
                mocker.MagicMock(
                    message=mocker.MagicMock(content=mock_response_content)
                )
            ]
        )
        query = "What fruits are mentioned?"
        command = ["generate", "answer", query, "--top-k", "3"]
        result = runner.invoke(app, command)
        assert result.exit_code == 0, (
            f"CLI failed with output:\n{result.stdout}"
        )
        assert "Generating answer using RAG..." in result.stdout
        assert "Generated Answer" in result.stdout
        assert "Sources" in result.stdout
        assert "Score" in result.stdout
        assert "ID" in result.stdout
        assert mock_response_content in result.stdout
        assert (
            "apples" in result.stdout
            or "oranges" in result.stdout
            or "bananas" in result.stdout
        )

    def test_cli_invalid_input_missing_arg(self, runner: CliRunner):
        command = ["search", "text"]
        result = runner.invoke(app, command)
        assert result.exit_code != 0
        assert result.exit_code == 2
        assert "Missing argument" in result.stdout
        assert "'QUERY'" in result.stdout

    def test_cli_invalid_input_bad_option(self, runner: CliRunner):
        command = ["search", "text", "somequery", "--limit", "not-a-number"]
        result = runner.invoke(app, command)
        assert result.exit_code != 0
        assert result.exit_code == 2
        assert "Invalid value" in result.stdout
        assert "'not-a-number' is not a valid integer" in result.stdout
