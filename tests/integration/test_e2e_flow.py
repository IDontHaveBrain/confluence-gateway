from typing import Optional

import pytest
from fastapi.testclient import TestClient

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.vector_db.factory import get_vector_db_adapter
from confluence_gateway.adapters.vector_db.qdrant_adapter import QdrantAdapter
from confluence_gateway.api.app import app
from confluence_gateway.api.dependencies import get_embedding_provider_dependency
from confluence_gateway.core.config import (
    EmbeddingConfig,
    VectorDBConfig,
    embedding_config,
    load_configurations,
    vector_db_config,
)
from confluence_gateway.providers.embedding.sentence_transformer import SentenceTransformerProvider

client = TestClient(app)

_confluence_config, _, _, _ = load_configurations()
real_config_available = _confluence_config is not None

# Default settings for embedding provider override
DEFAULT_EMBEDDING_PROVIDER = "sentence-transformers"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Relatively small and fast
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_EMBEDDING_DEVICE = "cpu"

# Determine if the core components for semantic search are configured or can be defaulted
_is_confluence_available = real_config_available

_global_embedding_config = embedding_config  # Cache for clarity
_global_vector_db_config = vector_db_config  # Cache for clarity

_is_global_embedding_configured = (
    _global_embedding_config is not None and _global_embedding_config.provider != "none"
)
_is_global_vector_db_configured = (
    _global_vector_db_config is not None and _global_vector_db_config.type != "none"
)

# Check if an embedding dimension is obtainable (globally or via default)
_embedding_dimension_available = (
    _global_embedding_config.dimension if _is_global_embedding_configured
    else DEFAULT_EMBEDDING_DIMENSION  # Assume default can provide dimension if no global config
) is not None

# Semantic search can run if Confluence is available,
# AND an embedding dimension can be determined (needed by VDB),
# AND a vector DB is either configured globally or can be defaulted (which requires the dimension).
# We assume the default embedding provider *can* be loaded if not configured globally.
# We assume the default vector DB *can* be loaded if not configured globally.
# Actual loading errors are handled within the fixtures.
semantic_search_fully_enabled = (
    _is_confluence_available and
    _embedding_dimension_available and
    (_is_global_vector_db_configured or (_global_vector_db_config is None or _global_vector_db_config.type == "none"))  # VDB is configured or can be defaulted
    # We don't need to explicitly check if embedding provider can be defaulted here,
    # as _embedding_dimension_available covers the essential outcome.
)

semantic_search_skip_reason = (
    "Semantic search requires: configured Confluence, an available embedding dimension "
    "(from config or default), and an available Vector DB (from config or default in-memory Qdrant)."
)


@pytest.fixture
def confluence_client():
    config, _, _, _ = load_configurations()
    if not config:
        pytest.skip(
            "Confluence configuration not available - skipping integration tests"
        )
    return ConfluenceClient(config=config)


@pytest.fixture
def search_term(confluence_client):
    try:
        common_terms = ["the", "and", "is", "in", "to"]

        for term in common_terms:
            result = confluence_client.search(query=term, limit=1)
            if result.total_size > 0:
                return term

        spaces_result = confluence_client.atlassian_api.get_all_spaces(limit=1)
        if spaces_result and "results" in spaces_result and spaces_result["results"]:
            space_name = spaces_result["results"][0].get("name")
            if space_name:
                return space_name

        pytest.skip("Could not find any search terms that return results")
    except Exception:
        pytest.skip("Error occurred trying to find valid search term")


@pytest.fixture(scope="class", autouse=True)
def override_embedding_provider_for_test(request):
    """
    Overrides the embedding provider dependency for semantic search tests
    if no embedding provider is configured globally. Defaults to a
    lightweight sentence-transformer model on CPU.
    """
    original_emb_config = embedding_config
    test_provider_instance = None
    original_override = app.dependency_overrides.get(get_embedding_provider_dependency)
    override_applied = False

    # Only intervene if NO embedding provider is configured globally
    if original_emb_config is None or original_emb_config.provider == "none":
        print(f"\nINFO: No Embedding Provider configured globally. Attempting override with default: "
              f"{DEFAULT_EMBEDDING_PROVIDER}/{DEFAULT_EMBEDDING_MODEL} (Dim: {DEFAULT_EMBEDDING_DIMENSION}, Device: {DEFAULT_EMBEDDING_DEVICE})")

        # Configure the default provider
        test_emb_config = EmbeddingConfig(
            provider=DEFAULT_EMBEDDING_PROVIDER,
            model_name=DEFAULT_EMBEDDING_MODEL,
            dimension=DEFAULT_EMBEDDING_DIMENSION,
            device=DEFAULT_EMBEDDING_DEVICE,
        )

        try:
            # Create and initialize the test provider instance
            test_provider_instance = SentenceTransformerProvider(test_emb_config)
            print(f"INFO: Initializing default SentenceTransformerProvider...")
            # Initialization might download the model on first run
            test_provider_instance.initialize()
            print("INFO: Default SentenceTransformerProvider initialized successfully.")

            # Define the override function
            def get_test_embedding_provider():
                # print("DEBUG: Providing overridden default Embedding Provider") # Optional debug
                return test_provider_instance

            # Apply the override
            app.dependency_overrides[get_embedding_provider_dependency] = get_test_embedding_provider
            override_applied = True
            print("INFO: Embedding Provider dependency overridden.")

        except Exception as e:
            print(f"\nERROR: Failed to initialize default Embedding Provider ({DEFAULT_EMBEDDING_MODEL}): {e}")
            print("INFO: Semantic search tests requiring embeddings will likely be skipped or fail.")
            # Do not apply override if initialization fails
            test_provider_instance = None  # Ensure cleanup doesn't try to close failed instance
            # Fall through to yield without override

    else:
        # Embedding provider IS configured globally, do nothing.
        print(f"\nINFO: Using globally configured Embedding Provider: "
              f"Type='{original_emb_config.provider}', Model='{original_emb_config.model_name}', "
              f"Dimension='{original_emb_config.dimension}'")
        # Fall through to yield without override

    # Yield control to the tests within the class
    yield

    # --- Teardown ---
    if test_provider_instance:  # Only cleanup if we created and initialized an instance
        print("\nINFO: Cleaning up default Embedding Provider instance...")
        try:
            test_provider_instance.close()
            print("INFO: Default Embedding Provider closed.")
        except Exception as close_e:
             print(f"ERROR: Exception during default Embedding Provider close: {close_e}")

    if override_applied:  # Only restore if we applied an override
        if original_override:
            app.dependency_overrides[get_embedding_provider_dependency] = original_override
        else:
            # Ensure key exists before deleting
            if get_embedding_provider_dependency in app.dependency_overrides:
                 del app.dependency_overrides[get_embedding_provider_dependency]
        print("INFO: Restored original Embedding Provider dependency.")
    # Add similar safety check as in VDB override
    elif get_embedding_provider_dependency in app.dependency_overrides and app.dependency_overrides[get_embedding_provider_dependency] == get_test_embedding_provider:
         print("\nWARN: Cleaning up potentially stale Embedding Provider override.")
         if original_override:
             app.dependency_overrides[get_embedding_provider_dependency] = original_override
         else:
             del app.dependency_overrides[get_embedding_provider_dependency]


@pytest.fixture(scope="class", autouse=True)
def override_vector_db_for_test(request):
    """
    Overrides the vector DB dependency for semantic search tests
    if no vector DB is configured globally. Defaults to in-memory Qdrant.
    Relies on obtaining an embedding dimension from either the global config
    or the potentially overridden embedding provider dependency.
    """
    original_vdb_config = vector_db_config
    test_adapter = None
    original_vdb_override = app.dependency_overrides.get(get_vector_db_adapter)
    override_applied = False

    # Only intervene if NO vector DB is configured globally
    if original_vdb_config is None or original_vdb_config.type == "none":
        print("\nINFO: No Vector DB configured globally. Checking if override is possible.")

        # --- Determine the effective embedding dimension ---
        effective_embedding_dimension: Optional[int] = None
        try:
            # Check if the embedding provider dependency is overridden
            embedding_override_func = app.dependency_overrides.get(get_embedding_provider_dependency)
            if embedding_override_func:
                print("INFO: Embedding provider dependency is overridden. Getting dimension from override.")
                # Call the override function to get the actual provider instance
                effective_provider = embedding_override_func()
                if effective_provider:
                    effective_embedding_dimension = effective_provider.get_dimension()
                else:
                     print("WARN: Embedding provider override function returned None.")
            else:
                # No override, check the global config
                print("INFO: Embedding provider dependency not overridden. Getting dimension from global config.")
                global_embedding_config = embedding_config  # Reload just in case? Or use cached one? Let's use cached.
                if global_embedding_config and global_embedding_config.provider != "none":
                    effective_embedding_dimension = global_embedding_config.dimension
                else:
                    print("INFO: Global embedding provider not configured or disabled.")

            if effective_embedding_dimension is None:
                 # This case should ideally be caught by the class skip logic, but double-check here.
                 print("ERROR: Could not determine embedding dimension (required for Vector DB override).")
                 pytest.skip("Cannot override Vector DB: Embedding dimension could not be determined.")

        except Exception as dim_e:
             print(f"ERROR: Failed to determine embedding dimension: {dim_e}")
             pytest.skip(f"Cannot override Vector DB: Failed to determine embedding dimension: {dim_e}")
        # --- End Determine Dimension ---


        print(f"INFO: Effective embedding dimension for Vector DB override: {effective_embedding_dimension}")
        print("INFO: Attempting override with in-memory Qdrant for testing.")

        # Configure in-memory Qdrant for testing
        test_vdb_config = VectorDBConfig(
            type="qdrant",
            qdrant_url=":memory:",  # Directly use ":memory:" as supported value
            collection_name="test_integration_embeddings",
            embedding_dimension=effective_embedding_dimension,  # Use the determined dimension
        )

        try:
            # Create and initialize the test adapter instance
            test_adapter = QdrantAdapter(test_vdb_config)
            print(f"INFO: Initializing in-memory Qdrant adapter for collection '{test_vdb_config.collection_name}'...")
            test_adapter.initialize()
            print("INFO: In-memory Qdrant adapter initialized.")

            # Define the override function
            def get_test_vector_db_adapter():
                # print("DEBUG: Providing overridden in-memory Qdrant adapter") # Optional debug
                return test_adapter

            # Apply the override
            app.dependency_overrides[get_vector_db_adapter] = get_test_vector_db_adapter
            override_applied = True
            print("INFO: Vector DB dependency overridden.")

        except Exception as e:
            print(f"\nERROR: Failed to initialize in-memory Qdrant for testing: {e}")
            # If override fails, tests relying on it will fail or be skipped by class decorator
            test_adapter = None  # Ensure cleanup doesn't try to close failed instance
            pass  # Fall through to yield

    else:
        # Vector DB IS configured globally, do nothing.
        print(f"\nINFO: Using globally configured Vector DB: Type='{original_vdb_config.type}', Collection='{original_vdb_config.collection_name}'")
        # Fall through to yield without override

    # Yield control to the tests within the class
    yield

    # --- Teardown ---
    if test_adapter:  # Only cleanup if we created an adapter
        print("\nINFO: Cleaning up in-memory Qdrant test adapter...")
        try:
            test_adapter.close()
            print("INFO: In-memory Qdrant adapter closed.")
        except Exception as close_e:
             print(f"ERROR: Exception during test adapter close: {close_e}")

    if override_applied:  # Only restore if we applied an override
        if original_vdb_override:
            app.dependency_overrides[get_vector_db_adapter] = original_vdb_override
        else:
            # Ensure key exists before deleting
            if get_vector_db_adapter in app.dependency_overrides:
                 del app.dependency_overrides[get_vector_db_adapter]
        print("INFO: Restored original Vector DB dependency.")
    # Add similar safety check as before
    elif get_vector_db_adapter in app.dependency_overrides and app.dependency_overrides[get_vector_db_adapter] == get_test_vector_db_adapter:
         print("\nWARN: Cleaning up potentially stale Vector DB override.")
         if original_vdb_override:
             app.dependency_overrides[get_vector_db_adapter] = original_vdb_override
         else:
             del app.dependency_overrides[get_vector_db_adapter]


@pytest.mark.skipif(
    not real_config_available, reason="Confluence configuration not available"
)
@pytest.mark.integration
class TestBasicSearchFlow:
    def test_search_api_endpoint_returns_results(self, search_term):
        response = client.get(f"/api/search?query={search_term}")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] >= 1
        assert len(data["results"]) >= 1

    def test_search_result_format(self, search_term):
        response = client.get(f"/api/search?query={search_term}")
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "results" in data
        assert "total" in data
        assert "start" in data
        assert "limit" in data
        assert "took_ms" in data
        assert "page_count" in data
        assert "current_page" in data
        assert "has_more" in data

        # Check result item structure
        if data["total"] > 0 and len(data["results"]) > 0:
            first_result = data["results"][0]
            assert "id" in first_result
            assert "title" in first_result
            assert "type" in first_result
            assert "space_key" in first_result
            assert "space_name" in first_result
            assert "url" in first_result
            assert "last_modified" in first_result


@pytest.mark.skipif(
    not real_config_available, reason="Confluence configuration not available"
)
@pytest.mark.integration
class TestCQLSearchFlow:
    def test_cql_search_api_endpoint(self, search_term, confluence_client):
        try:
            spaces_result = confluence_client.atlassian_api.get_all_spaces(limit=1)
            space_key = (
                spaces_result["results"][0]["key"]
                if spaces_result.get("results")
                else None
            )
        except Exception:
            space_key = None

        cql = f'text ~ "{search_term}"'
        if space_key:
            cql += f' AND space = "{space_key}"'

        response = client.post("/api/search/cql", json={"cql": cql, "limit": 10})

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] >= 1
        assert len(data["results"]) >= 1


@pytest.mark.skipif(
    not real_config_available, reason="Confluence configuration not available"
)
@pytest.mark.integration
class TestAdvancedSearchFlow:
    def test_advanced_search_api_endpoint(self, search_term):
        response = client.post(
            "/api/search/advanced",
            json={
                "query": search_term,
                "limit": 10,
                "sort_by": ["title"],
                "sort_direction": ["asc"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] >= 1
        assert len(data["results"]) >= 1

        # If we have multiple results, verify sorting works
        if len(data["results"]) > 1:
            titles = [item["title"] for item in data["results"]]
            sorted_titles = sorted(titles)
            assert titles == sorted_titles

    def test_advanced_search_with_filters(self, search_term):
        response = client.post(
            "/api/search/advanced",
            json={"query": search_term, "content_type": "page", "limit": 10},
        )

        assert response.status_code == 200
        data = response.json()

        if data["total"] > 0:
            for result in data["results"]:
                assert result["type"] == "page"


@pytest.mark.skipif(
    not real_config_available, reason="Confluence configuration not available"
)
@pytest.mark.integration
class TestPaginationFlow:
    def test_pagination_links_and_navigation(self, search_term):
        response = client.get(f"/api/search?query={search_term}&limit=2")
        assert response.status_code == 200
        first_page = response.json()

        if first_page["total"] <= 2:
            pytest.skip("Not enough results to test pagination")

        assert first_page["has_more"]
        assert "links" in first_page
        assert "next" in first_page["links"]

        next_link = first_page["links"]["next"]
        next_path = (
            next_link.split("://")[-1].split("/", 1)[-1]
            if "://" in next_link
            else next_link
        )

        response = client.get(f"/{next_path}")
        assert response.status_code == 200
        second_page = response.json()

        assert second_page["start"] > first_page["start"]
        assert "links" in second_page
        assert "previous" in second_page["links"]

        first_page_ids = [item["id"] for item in first_page["results"]]
        second_page_ids = [item["id"] for item in second_page["results"]]

        if first_page_ids == second_page_ids:
            assert second_page["current_page"] > first_page["current_page"]
        else:
            assert not any(id in first_page_ids for id in second_page_ids)


@pytest.mark.skipif(
    not semantic_search_fully_enabled,
    reason=semantic_search_skip_reason,
)
@pytest.mark.integration
class TestSemanticSearchFlow:
    def test_semantic_search_api_endpoint(self, search_term):
        """Tests the basic functionality of the semantic search endpoint."""
        payload = {"query": search_term, "top_k": 5}
        response = client.post("/api/search/semantic", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "results" in data
        assert "took_ms" in data
        assert "query" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["took_ms"], (float, int))
        assert data["took_ms"] >= 0
        assert data["query"] == search_term

        # If results are returned, check the structure of the first item
        if data["results"]:
            first_result = data["results"][0]
            assert "id" in first_result
            assert "score" in first_result
            assert isinstance(first_result["score"], (float, int))
            assert "metadata" in first_result
            assert isinstance(first_result["metadata"], dict)

    def test_semantic_search_invalid_input(self):
        """Tests sending invalid input (empty query) to the semantic search endpoint."""
        payload = {"query": "", "top_k": 5}
        response = client.post("/api/search/semantic", json=payload)

        # Expecting FastAPI/Pydantic validation error
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        # Check if the error is related to the 'query' field
        assert any(
            "query" in item.get("loc", []) for item in data["detail"]
        ), "Error detail should mention the 'query' field"


@pytest.mark.skipif(
    not real_config_available, reason="Confluence configuration not available"
)
@pytest.mark.integration
class TestErrorPropagation:
    def test_invalid_parameter_error(self):
        response = client.get("/api/search?query=a")
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        assert "loc" in data["detail"][0]
        assert "msg" in data["detail"][0]

    def test_invalid_cql_error(self):
        response = client.post("/api/search/cql", json={"cql": "&&&invalidcql!!!"})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        # Validation errors have a different structure in FastAPI default responses
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        assert "loc" in data["detail"][0]
        assert "msg" in data["detail"][0]

    def test_content_type_validation(self):
        response = client.post(
            "/api/search/advanced",
            json={"query": "test", "content_type": "invalid_type"},
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        # Validation errors have a different structure in FastAPI default responses
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
        assert "loc" in data["detail"][0]
        assert "msg" in data["detail"][0]


@pytest.mark.skipif(
    not real_config_available, reason="Confluence configuration not available"
)
@pytest.mark.integration
class TestDataConsistency:
    def test_cross_endpoint_result_consistency(self, search_term):
        # Get results from both endpoints with the same query
        basic_response = client.get(f"/api/search?query={search_term}&limit=5")
        advanced_response = client.post(
            "/api/search/advanced", json={"query": search_term, "limit": 5}
        )

        assert basic_response.status_code == 200
        assert advanced_response.status_code == 200

        basic_data = basic_response.json()
        advanced_data = advanced_response.json()

        # Total should be the same
        assert basic_data["total"] == advanced_data["total"]

        # Result counts should match
        # (might be less than limit if not enough results)
        assert len(basic_data["results"]) == len(advanced_data["results"])
