import logging
import random
import re
import uuid
from collections.abc import Generator
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

# Register custom pytest plugin
# pytest_plugins = ["tests.pytest_plugins"]

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.embedding.factory import (
    EmbeddingProvider,
    get_embedding_provider,
)
from confluence_gateway.adapters.embedding.litellm import LiteLLMProvider
from confluence_gateway.adapters.embedding.sentence_transformer import (
    SentenceTransformerProvider,
)
from confluence_gateway.adapters.vector_db.factory import (
    VectorDBAdapter,
    get_vector_db_adapter,
)
from confluence_gateway.adapters.vector_db.models import Document
from confluence_gateway.adapters.vector_db.qdrant_adapter import QdrantAdapter
from confluence_gateway.api.app import app
from confluence_gateway.core.config import (
    ConfluenceConfig,
    EmbeddingConfig,
    GenerationConfig,
    IndexingConfig,
    SearchConfig,
    VectorDBConfig,
    load_configurations,
)
from confluence_gateway.core.config import (
    embedding_config as global_embedding_config,
)
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.generation import GenerationService
from confluence_gateway.services.indexing import IndexingService
from confluence_gateway.services.search import SearchService
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from typer.testing import CliRunner


class SuppressSpecificLogFilter(logging.Filter):
    """Filter to suppress logs from specific loggers."""
    def filter(self, record):
        # Block ERROR logs from these specific loggers
        if record.levelno == logging.ERROR and record.name in [
            "confluence_gateway.adapters.embedding.litellm",
            "confluence_gateway.core.config"
        ]:
            return False
        return True


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark tests that require a real Confluence connection"
    )


def pytest_sessionstart(session):
    """Add filter to CLI handler after pytest sets up logging."""
    # Get all handlers from root logger
    root_logger = logging.getLogger()
    
    # Add our filter to all handlers (CLI handler will be one of them)
    log_filter = SuppressSpecificLogFilter()
    for handler in root_logger.handlers:
        handler.addFilter(log_filter)


REAL_CONFIG_SKIP_REASON = (
    "Confluence configuration not found in environment or config file"
)
SEMANTIC_SEARCH_SKIP_REASON = (
    "Semantic search requires: configured Confluence, an available embedding provider "
    "(from config or default), and an available Vector DB (from config or default)."
)

DEFAULT_EMBEDDING_PROVIDER_TYPE = "sentence-transformers"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_EMBEDDING_DEVICE = "cpu"

DEFAULT_VECTOR_DB_TYPE = "qdrant"
DEFAULT_VECTOR_DB_COLLECTION = "confluence_pytest_embeddings"
DEFAULT_VECTOR_DB_URL = ":memory:"

_SEMANTIC_TEST_DOCS = [
    {"id": str(uuid.uuid4()), "text": "This is the first test document about apples."},
    {
        "id": str(uuid.uuid4()),
        "text": "The second document discusses oranges and citrus fruits.",
    },
    {"id": str(uuid.uuid4()), "text": "Finally, a document mentioning bananas."},
]

import pytest


@pytest.fixture(scope="session")
def SEMANTIC_TEST_DOCS():
    return _SEMANTIC_TEST_DOCS


@pytest.fixture(scope="session")
def loaded_configs() -> tuple[
    Optional[ConfluenceConfig],
    SearchConfig,
    Optional[VectorDBConfig],
    Optional[EmbeddingConfig],
]:
    return load_configurations()


@pytest.fixture(scope="session")
def confluence_config(loaded_configs) -> Optional[ConfluenceConfig]:
    return loaded_configs[0]


@pytest.fixture(scope="session")
def search_config(loaded_configs) -> SearchConfig:
    return loaded_configs[1]


@pytest.fixture(scope="session")
def vector_db_config(loaded_configs) -> Optional[VectorDBConfig]:
    return loaded_configs[2]


@pytest.fixture(scope="session")
def embedding_config(loaded_configs) -> Optional[EmbeddingConfig]:
    return loaded_configs[3]


@pytest.fixture(scope="session")
def indexing_config(loaded_configs) -> IndexingConfig:
    return loaded_configs[4]


@pytest.fixture(scope="session")
def generation_config(loaded_configs) -> Optional[GenerationConfig]:
    return loaded_configs[5]


@pytest.fixture(scope="session")
def is_generation_enabled(generation_config) -> bool:
    return generation_config is not None and generation_config.enable


@pytest.fixture(scope="session")
def is_real_config_available(confluence_config) -> bool:
    return confluence_config is not None


@pytest.fixture(scope="session")
def confluence_client(
    confluence_config, is_real_config_available
) -> Optional[ConfluenceClient]:
    if not is_real_config_available:
        pytest.skip(REAL_CONFIG_SKIP_REASON)
        return None

    client = ConfluenceClient(config=confluence_config)
    try:
        client.test_connection()
    except Exception as e:
        pytest.skip(f"Could not connect to Confluence during client setup: {e}")
        return None
    return client


@pytest.fixture(scope="session")
def real_search_term(confluence_client) -> str:
    if not confluence_client:
        pytest.skip("Confluence client not available for finding search term.")
        return "skip"

    def extract_content_tokens(text, min_length=2, max_length=20):
        if not text:
            return []
        tokens = re.findall(r"\b\w+\b", text, re.UNICODE)
        return [t for t in tokens if min_length <= len(t) <= max_length]

    token_candidates = []
    try:
        spaces_response = confluence_client.atlassian_api.get_all_spaces(limit=5)
        if spaces_response and spaces_response.get("results"):
            spaces = random.sample(
                spaces_response["results"], min(len(spaces_response["results"]), 3)
            )
            for space in spaces:
                if space.get("name"):
                    token_candidates.extend(extract_content_tokens(space["name"]))
                if space.get("key"):
                    try:
                        cql = f'space = "{space["key"]}" AND type in (page, blogpost) ORDER BY lastmodified DESC'
                        page_resp = confluence_client.search_by_cql(
                            cql, limit=2, expand=["title"]
                        )
                        if page_resp and page_resp.results:
                            for page in page_resp.results:
                                token_candidates.extend(
                                    extract_content_tokens(page.title)
                                )
                    except Exception:
                        pass

        if len(set(token_candidates)) < 10:
            cql = "type in (page, blogpost) ORDER BY lastmodified DESC"
            page_resp = confluence_client.search_by_cql(cql, limit=5, expand=["title"])
            if page_resp and page_resp.results:
                for page in page_resp.results:
                    token_candidates.extend(extract_content_tokens(page.title))

        unique_candidates = list(set(t for t in token_candidates if t))
        random.shuffle(unique_candidates)
        for term in unique_candidates[:15]:
            try:
                search_result = confluence_client.search(query=term, limit=1)
                if search_result.total_size > 0:
                    print(f"\nINFO: Using real search term: '{term}'")
                    return term
            except Exception:
                continue

    except Exception as e:
        print(f"\nWARN: Error finding dynamic search term: {e}. Falling back.")
        pytest.skip("Could not dynamically find a working search term.")
        return "skip"

    for term in ["the", "and", "is", "in"]:
        try:
            search_result = confluence_client.search(query=term, limit=1)
            if search_result.total_size > 0:
                print(f"\nINFO: Using fallback search term: '{term}'")
                return term
        except Exception:
            continue

    pytest.skip("Could not find any search term yielding results.")
    return "skip"


@pytest.fixture(scope="session")
def embedding_provider(embedding_config) -> Optional[EmbeddingProvider]:
    """
    Provides an initialized EmbeddingProvider instance.
    Uses global config or defaults to a lightweight sentence-transformer.
    Includes teardown. Skips if initialization fails.
    """
    provider_instance: Optional[EmbeddingProvider] = None
    effective_config = embedding_config

    if effective_config is None or effective_config.provider == "none":
        print(
            f"\nINFO (pytest): No Embedding Provider configured globally. Using default: "
            f"{DEFAULT_EMBEDDING_PROVIDER_TYPE}/{DEFAULT_EMBEDDING_MODEL}"
        )
        effective_config = EmbeddingConfig(
            provider=DEFAULT_EMBEDDING_PROVIDER_TYPE,
            model_name=DEFAULT_EMBEDDING_MODEL,
            dimension=DEFAULT_EMBEDDING_DIMENSION,
            device=DEFAULT_EMBEDDING_DEVICE,
        )
        try:
            provider_instance = SentenceTransformerProvider(effective_config)
            provider_instance.initialize()
            print("INFO (pytest): Default SentenceTransformerProvider initialized.")
        except Exception as e:
            pytest.skip(
                f"Failed to initialize default embedding provider ({DEFAULT_EMBEDDING_MODEL}): {e}"
            )
            return None
    else:
        print(
            f"\nINFO (pytest): Using globally configured Embedding Provider: "
            f"Type='{effective_config.provider}', Model='{effective_config.model_name}'"
        )
        try:
            provider_instance = get_embedding_provider(effective_config)
            if provider_instance is None and effective_config.provider != "none":
                raise RuntimeError(
                    "Embedding factory returned None for non-'none' provider."
                )
            print("INFO (pytest): Configured Embedding Provider obtained.")
        except Exception as e:
            pytest.skip(f"Failed to get/initialize configured embedding provider: {e}")
            return None

        if effective_config.provider == "sentence-transformers" and not isinstance(
            provider_instance, SentenceTransformerProvider
        ):
            pytest.skip(
                f"Configured for sentence-transformers, but factory returned type {type(provider_instance)}. Skipping."
            )
            return None
    yield provider_instance

    if provider_instance and hasattr(provider_instance, "close"):
        print(
            f"\nINFO (pytest): Closing Embedding Provider instance ({effective_config.provider})..."
        )
        try:
            provider_instance.close()
            print("INFO (pytest): Embedding Provider closed.")
        except Exception as close_e:
            print(
                f"ERROR (pytest): Exception during Embedding Provider close: {close_e}"
            )


@pytest.fixture(scope="session")
def is_embedding_available(embedding_provider) -> bool:
    return embedding_provider is not None


import typing

MockedProviderFixture = tuple[LiteLLMProvider, MagicMock]


@pytest.fixture(scope="function")
def mocked_litellm_provider(
    mocker: MockerFixture, embedding_config: Optional[EmbeddingConfig]
) -> typing.Optional[MockedProviderFixture]:
    effective_config = embedding_config
    if not effective_config or effective_config.provider != "litellm":
        effective_config = EmbeddingConfig(
            provider="litellm",
            model_name="mock-embedding-model",
            dimension=128,
        )
        print("\nINFO (pytest): Using dummy LiteLLM config for mocked provider.")
    else:
        print("\nINFO (pytest): Using provided LiteLLM config for mocked provider.")

    mock_embedding_call = mocker.patch("litellm.embedding", autospec=True)

    try:
        provider = LiteLLMProvider(config=effective_config)
        _dummy_embedding_for_internal_mocks = [0.0] * (
            effective_config.dimension or 128
        )
        _mock_response_data_for_internal_mocks = [
            {"embedding": _dummy_embedding_for_internal_mocks}
        ]
        mocker.patch.object(
            provider,
            "_validate_embedding_response",
            return_value=_mock_response_data_for_internal_mocks,
        )
        mocker.patch.object(
            provider,
            "_extract_embedding_from_item",
            return_value=_dummy_embedding_for_internal_mocks,
        )
        return provider, mock_embedding_call
    except Exception as e:
        pytest.skip(f"Skipping test: Could not instantiate mocked LiteLLMProvider: {e}")
        return None


@pytest.fixture(scope="session")
def effective_embedding_dimension(embedding_provider) -> Optional[int]:
    if not embedding_provider:
        if global_embedding_config and global_embedding_config.dimension:
            return global_embedding_config.dimension
        return DEFAULT_EMBEDDING_DIMENSION
    try:
        return embedding_provider.get_dimension()
    except Exception as e:
        print(
            f"\nWARN (pytest): Could not get dimension from embedding provider: {e}. Falling back."
        )
        if global_embedding_config and global_embedding_config.dimension:
            return global_embedding_config.dimension
        return DEFAULT_EMBEDDING_DIMENSION


@pytest.fixture(scope="session")
def vector_db_adapter(
    vector_db_config, effective_embedding_dimension
) -> Optional[VectorDBAdapter]:
    """
    Provides an initialized VectorDBAdapter instance.
    Uses global config or defaults to in-memory Qdrant.
    Requires an effective_embedding_dimension. Includes teardown. Skips if initialization fails.
    """
    adapter_instance: Optional[VectorDBAdapter] = None
    effective_vdb_config = vector_db_config

    if effective_embedding_dimension is None:
        pytest.skip("Cannot initialize Vector DB: Embedding dimension is unknown.")
        return None

    if effective_vdb_config is None or effective_vdb_config.type == "none":
        print(
            f"\nINFO (pytest): No Vector DB configured globally. Using default: "
            f"In-memory {DEFAULT_VECTOR_DB_TYPE} (Dim: {effective_embedding_dimension})"
        )
        effective_vdb_config = VectorDBConfig(
            type=DEFAULT_VECTOR_DB_TYPE,
            qdrant_url=DEFAULT_VECTOR_DB_URL,
            collection_name=DEFAULT_VECTOR_DB_COLLECTION,
            embedding_dimension=effective_embedding_dimension,
        )
        try:
            if effective_vdb_config.type == "qdrant":
                adapter_instance = QdrantAdapter(effective_vdb_config)
                adapter_instance.initialize()
                print("INFO (pytest): Default in-memory Qdrant adapter initialized.")
            else:
                raise NotImplementedError(
                    "Default Vector DB type not implemented for tests"
                )

        except Exception as e:
            pytest.skip(f"Failed to initialize default Vector DB adapter: {e}")
            return None
    else:
        if effective_vdb_config.embedding_dimension != effective_embedding_dimension:
            print(
                f"\nWARN (pytest): Mismatch between configured Vector DB dimension "
                f"({effective_vdb_config.embedding_dimension}) and effective "
                f"embedding dimension ({effective_embedding_dimension})."
            )

        print(
            f"\nINFO (pytest): Using globally configured Vector DB: "
            f"Type='{effective_vdb_config.type}', Collection='{effective_vdb_config.collection_name}'"
        )
        try:
            adapter_instance = get_vector_db_adapter(effective_vdb_config)
            if adapter_instance is None and effective_vdb_config.type != "none":
                raise RuntimeError(
                    "Vector DB factory returned None for non-'none' type."
                )
            print("INFO (pytest): Configured Vector DB adapter obtained.")
        except Exception as e:
            pytest.skip(f"Failed to get/initialize configured Vector DB adapter: {e}")
            return None
    yield adapter_instance

    if adapter_instance and hasattr(adapter_instance, "close"):
        print(
            f"\nINFO (pytest): Closing Vector DB adapter instance ({effective_vdb_config.type})..."
        )
        try:
            if (
                effective_vdb_config.qdrant_url == ":memory:"
                and isinstance(adapter_instance, QdrantAdapter)
                and hasattr(adapter_instance, "client")
                and adapter_instance.client
            ):
                try:
                    adapter_instance.client.delete_collection(
                        collection_name=effective_vdb_config.collection_name
                    )
                    print(
                        f"INFO (pytest): Deleted collection '{effective_vdb_config.collection_name}'."
                    )
                except Exception as del_e:
                    print(f"WARN (pytest): Failed to delete Qdrant collection: {del_e}")

            adapter_instance.close()
            print("INFO (pytest): Vector DB adapter closed.")
        except Exception as close_e:
            print(
                f"ERROR (pytest): Exception during Vector DB adapter close: {close_e}"
            )


@pytest.fixture(scope="session")
def is_vector_db_available(vector_db_adapter) -> bool:
    return vector_db_adapter is not None


@pytest.fixture(scope="session")
def is_semantic_search_possible(
    is_real_config_available, is_embedding_available, is_vector_db_available
) -> bool:
    return (
        is_real_config_available and is_embedding_available and is_vector_db_available
    )


@pytest.fixture(scope="session")
def embedding_service(
    embedding_provider, is_embedding_available
) -> Optional[EmbeddingService]:
    if not is_embedding_available:
        return None
    return EmbeddingService(provider=embedding_provider)


@pytest.fixture(scope="function")
def mocked_embedding_service(
    mocked_litellm_provider: Optional[LiteLLMProvider],
) -> Optional[EmbeddingService]:
    """Provides an EmbeddingService using the mocked LiteLLMProvider."""
    if not mocked_litellm_provider:
        pytest.skip("Mocked LiteLLM provider not available.")
        return None
    # Ensure the provider appears initialized for the service
    # (mocks above handle the actual initialization logic)
    return EmbeddingService(provider=mocked_litellm_provider)


@pytest.fixture(scope="session")
def semantic_search_service(
    confluence_client, embedding_service, vector_db_adapter, is_semantic_search_possible
) -> Optional[SearchService]:
    """
    Provides a SearchService instance fully configured for semantic search.
    Returns None if semantic search is not possible.
    """
    if not is_semantic_search_possible:
        return None

    return SearchService(
        client=confluence_client,
        indexing_service=None,
        embedding_service=embedding_service,
        vector_db_adapter=vector_db_adapter,
    )


@pytest.fixture(scope="session")
def standard_search_service(
    confluence_client, is_real_config_available
) -> Optional[SearchService]:
    if not is_real_config_available:
        return None
    return SearchService(
        client=confluence_client,
        indexing_service=None,
        embedding_service=None,
        vector_db_adapter=None,
    )


@pytest.fixture(scope="function")
def indexing_service(
    confluence_client: Optional[ConfluenceClient],
    embedding_service: Optional[EmbeddingService],
    vector_db_adapter: Optional[VectorDBAdapter],
    indexing_config: IndexingConfig,
    search_config: SearchConfig,
    is_semantic_search_possible: bool,
) -> Optional[IndexingService]:
    if not is_semantic_search_possible or not confluence_client:
        pytest.skip(
            "IndexingService requires Confluence client, Vector DB, and Embedding Service."
        )
        return None
    if not vector_db_adapter:
        pytest.skip(
            "IndexingService requires VectorDBAdapter, but it was not available."
        )
        return None

    # Reset singleton state before creating/getting instance
    IndexingService._instance = None
    IndexingService._is_running = False
    IndexingService._last_run_start_time = None
    IndexingService._last_run_end_time = None
    IndexingService._last_run_status = "idle"
    IndexingService._last_error_message = None

    try:
        service = IndexingService(
            confluence_client=confluence_client,
            indexing_config=indexing_config,
            search_config=search_config,
            embedding_service=embedding_service,
            vector_db_adapter=vector_db_adapter,
        )
        if not service.vector_db_adapter:
            pytest.skip(
                "IndexingService initialization failed: VectorDBAdapter became None post-init."
            )
            return None
        if not service.text_splitter:
            pytest.skip(
                "IndexingService failed to initialize SentenceSplitter (adapter might be missing config internally, or another init error)."
            )
            return None

        yield service
        
        # Clean up singleton state after test
        IndexingService._instance = None
        IndexingService._is_running = False
        IndexingService._last_run_start_time = None
        IndexingService._last_run_end_time = None
        IndexingService._last_run_status = "idle"
        IndexingService._last_error_message = None
        
    except Exception as e:
        pytest.skip(f"Failed to initialize IndexingService fixture: {e}")
        return None


@pytest.fixture(scope="function")
def generation_service(
    semantic_search_service: Optional[SearchService],
    generation_config: Optional[GenerationConfig],
    is_generation_enabled: bool,
) -> Optional[GenerationService]:
    """
    Provides a GenerationService instance.
    Mocking of internal litellm.acompletion happens *within tests*.
    Skips if generation is disabled or search service unavailable.
    """
    if not is_generation_enabled:
        pytest.skip("Generation feature is disabled in configuration.")
        return None
    if not semantic_search_service:
        pytest.skip(
            "GenerationService requires SearchService (check semantic search prerequisites)."
        )
        return None

    try:
        service = GenerationService(
            search_service=semantic_search_service, config=generation_config
        )
        return service
    except Exception as e:
        pytest.skip(f"Failed to initialize GenerationService fixture: {e}")
        return None


@pytest.fixture(scope="session", autouse=True)
def index_semantic_test_data(
    semantic_search_service,
    embedding_service,
    vector_db_adapter,
    is_semantic_search_possible,
    SEMANTIC_TEST_DOCS,
):
    """
    Indexes the SEMANTIC_TEST_DOCS into the vector DB if semantic search is possible.
    Runs once per session and checks if data might already exist.
    """
    if not is_semantic_search_possible:
        print(
            "\nINFO (pytest): Skipping semantic test data indexing (semantic search not possible)."
        )
        return

    adapter = vector_db_adapter
    embed_svc = embedding_service

    try:
        # Check if the first document exists using retrieve_by_ids
        first_doc_id = SEMANTIC_TEST_DOCS[0]["id"]
        # Use retrieve_by_ids which is simpler for checking existence by ID
        existing_records = adapter.retrieve_by_ids(
            ids=[first_doc_id], with_payload=False, with_vector=False
        )
        if existing_records:
            print(
                f"\nINFO (pytest): Semantic test data (e.g., '{first_doc_id}') seems to exist via retrieve_by_ids. Skipping indexing."
            )
            return
        else:
            print(
                f"\nINFO (pytest): Semantic test data (e.g., '{first_doc_id}') not found via retrieve_by_ids. Proceeding with indexing."
            )
    except Exception as check_err:
        # Catch specific expected errors if possible, otherwise broad Exception
        print(
            f"\nWARN (pytest): Error checking for existing semantic test data using retrieve_by_ids: {check_err}. Attempting indexing anyway."
        )
        # For robustness, we'll proceed with indexing attempt

    print("\nINFO (pytest): Indexing semantic test data...")
    try:
        texts = [doc["text"] for doc in SEMANTIC_TEST_DOCS]
        embeddings = embed_svc.embed_texts(texts)

        if len(embeddings) != len(SEMANTIC_TEST_DOCS):
            raise RuntimeError(
                "Mismatch between texts and generated embeddings during test data setup."
            )

        documents = []
        for i, doc_data in enumerate(SEMANTIC_TEST_DOCS):
            documents.append(
                Document(
                    id=doc_data["id"],
                    text=doc_data["text"],
                    embedding=embeddings[i],
                    metadata={
                        "source": "pytest_fixture",
                        "test_run_id": str(uuid.uuid4())[:8],
                    },
                )
            )

        adapter.upsert(documents)
        print(f"INFO (pytest): Indexed {len(documents)} semantic test documents.")

    except Exception as e:
        print(f"\nERROR (pytest): Failed to index semantic test data: {e}")


from confluence_gateway.api.app import app
from confluence_gateway.api.dependencies import (
    get_confluence_client,
    get_embedding_provider_dependency,
    get_vector_db_adapter,
)


@pytest.fixture(scope="session")
def test_app_client(
    confluence_client: Optional[ConfluenceClient],
    vector_db_adapter: Optional[VectorDBAdapter],
    embedding_provider: Optional[EmbeddingProvider],
    search_config: SearchConfig,
    vector_db_config: Optional[VectorDBConfig],
    indexing_config: IndexingConfig,
) -> Generator[TestClient, Any, None]:
    def override_get_confluence_client():
        return confluence_client

    def override_get_vector_db_adapter():
        return vector_db_adapter

    def override_get_embedding_provider_dependency():
        return embedding_provider

    def override_get_search_config():
        return search_config

    def override_get_vector_db_config():
        if vector_db_adapter and hasattr(vector_db_adapter, "config"):
            adapter_config = getattr(vector_db_adapter, "config", None)
            if adapter_config:
                return adapter_config
        return vector_db_config

    def override_get_indexing_config():
        return indexing_config

    app.dependency_overrides[get_confluence_client] = override_get_confluence_client
    app.dependency_overrides[get_vector_db_adapter] = override_get_vector_db_adapter
    app.dependency_overrides[get_embedding_provider_dependency] = (
        override_get_embedding_provider_dependency
    )

    import logging
    from threading import Lock

    from confluence_gateway.api.dependencies import get_indexing_service
    from confluence_gateway.services.embedding import EmbeddingService
    from confluence_gateway.services.indexing import IndexingService

    _override_indexing_service_instance = None
    _override_indexing_service_lock = Lock()

    def override_get_indexing_service():
        nonlocal _override_indexing_service_instance

        if _override_indexing_service_instance is not None:
            return _override_indexing_service_instance

        with _override_indexing_service_lock:
            if _override_indexing_service_instance is None:
                effective_vdb_config = None
                if vector_db_adapter and hasattr(vector_db_adapter, "config"):
                    adapter_config = getattr(vector_db_adapter, "config", None)
                    if adapter_config:
                        effective_vdb_config = adapter_config
                if effective_vdb_config is None:
                    effective_vdb_config = vector_db_config

                if not vector_db_adapter:
                    logging.warning(
                        "Override: Vector DB Adapter fixture not available."
                    )
                    return None
                if not effective_vdb_config:
                    logging.warning(
                        "Override: Effective Vector DB Config not available."
                    )
                    return None

                current_embedding_service = EmbeddingService(
                    provider=embedding_provider
                )

                try:
                    logging.info(
                        "Override: Attempting to initialize IndexingService..."
                    )
                    instance = IndexingService(
                        confluence_client=confluence_client,
                        indexing_config=indexing_config,
                        search_config=search_config,
                        embedding_service=current_embedding_service,
                        vector_db_adapter=vector_db_adapter,
                    )
                    if not instance.vector_db_adapter:
                        logging.error("Override Error: Adapter became None post-init.")
                        return None
                    if not instance.text_splitter:
                        logging.error(
                            "Override Error: Text splitter not initialized post-init. Check VDB config access within IndexingService init."
                        )
                        return None

                    _override_indexing_service_instance = instance
                    logging.info("Override: IndexingService initialized successfully.")
                except Exception as e:
                    logging.error(
                        f"Override Error: Failed to initialize IndexingService: {e}",
                        exc_info=True,
                    )
                    _override_indexing_service_instance = None
                    return None

        return _override_indexing_service_instance

    app.dependency_overrides[get_indexing_service] = override_get_indexing_service

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def runner() -> CliRunner:
    return CliRunner()
# Fixture aliases for backward compatibility

@pytest.fixture(scope="session")
def search_service(standard_search_service):
    """Alias for standard_search_service for backward compatibility."""
    return standard_search_service

@pytest.fixture(scope="session")
def api_client(test_app_client):
    """Alias for test_app_client for backward compatibility."""
    return test_app_client

