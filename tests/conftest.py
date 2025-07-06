import logging
import random
import re
import uuid
from collections.abc import Generator
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.embedding.factory import (
    EmbeddingProvider,
    get_embedding_provider,
)
from confluence_gateway.adapters.vector_db.factory import (
    VectorDBAdapter,
    get_vector_db_adapter,
)
from confluence_gateway.adapters.vector_db.models import Document
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
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from typer.testing import CliRunner


class SuppressSpecificLogFilter(logging.Filter):
    """Filter to suppress logs from specific loggers."""

    def filter(self, record):
        if record.levelno == logging.ERROR and record.name in [
            "confluence_gateway.adapters.embedding.litellm",
            "confluence_gateway.core.config",
        ]:
            return False
        return True


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark tests that require a real Confluence connection"
    )


def pytest_collection_modifyitems(config, items):
    import os

    if not os.environ.get("CONFLUENCE_URL"):
        skip_integration = pytest.mark.skip(
            reason="No Confluence config - set CONFLUENCE_URL env var"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_sessionstart(session):
    """Add filter to CLI handler after pytest sets up logging."""
    root_logger = logging.getLogger()

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

_FALLBACK_SEMANTIC_TEST_DOCS = [
    {"id": str(uuid.uuid4()), "text": "This is the first test document about apples."},
    {
        "id": str(uuid.uuid4()),
        "text": "The second document discusses oranges and citrus fruits.",
    },
    {"id": str(uuid.uuid4()), "text": "Finally, a document mentioning bananas."},
]

import pytest


@pytest.fixture(scope="session")
def SEMANTIC_TEST_DOCS(
    dummy_data_spaces: list[dict] | None, confluence_client
) -> list[dict]:
    """Provides semantic test documents, preferring real generated content over fallback docs."""
    if dummy_data_spaces and confluence_client:
        real_docs = []
        target_doc_count = 5

        for space in dummy_data_spaces[:2]:
            if not space["has_content"]:
                continue

            try:
                search_result = confluence_client.search(
                    query=f"space={space['key']}",
                    limit=3,
                )

                for page in search_result.results[:3]:
                    if len(real_docs) >= target_doc_count:
                        break

                    try:
                        page_detail = confluence_client.get_page(
                            page.id, expand="body.storage"
                        )

                        content = (
                            getattr(page_detail.body.storage, "value", "")
                            if page_detail.body
                            else ""
                        )
                        if content:
                            import re

                            text_content = re.sub(r"<[^>]+>", " ", content)
                            text_content = " ".join(text_content.split())

                            if len(text_content) > 50:
                                real_docs.append(
                                    {
                                        "id": page.id,
                                        "text": text_content[:1000],
                                        "title": page.title,
                                        "space_key": space["key"],
                                        "source": "real_generated_data",
                                    }
                                )
                    except Exception as e:
                        continue

                if len(real_docs) >= target_doc_count:
                    break

            except Exception as e:
                continue

        if real_docs:
            return real_docs

    return _FALLBACK_SEMANTIC_TEST_DOCS


@pytest.fixture(scope="session")
def loaded_configs() -> tuple[
    ConfluenceConfig | None,
    SearchConfig,
    VectorDBConfig | None,
    EmbeddingConfig | None,
]:
    return load_configurations()


@pytest.fixture(scope="session")
def confluence_config(loaded_configs) -> ConfluenceConfig | None:
    return loaded_configs[0]


@pytest.fixture(scope="session")
def search_config(loaded_configs) -> SearchConfig:
    return loaded_configs[1]


@pytest.fixture(scope="session")
def vector_db_config(loaded_configs) -> VectorDBConfig | None:
    return loaded_configs[2]


@pytest.fixture(scope="session")
def embedding_config(loaded_configs) -> EmbeddingConfig | None:
    return loaded_configs[3]


@pytest.fixture(scope="session")
def indexing_config(loaded_configs) -> IndexingConfig:
    return loaded_configs[4]


@pytest.fixture(scope="session")
def generation_config(loaded_configs) -> GenerationConfig | None:
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
) -> ConfluenceClient | None:
    if not is_real_config_available:
        pytest.skip(REAL_CONFIG_SKIP_REASON)
        return None

    client = ConfluenceClient(config=confluence_config)
    return client


@pytest.fixture(scope="session")
def real_search_term(real_search_terms: list[str]) -> str:
    """Provides a single search term, preferring real extracted terms."""
    import os

    env_term = os.environ.get("CONFLUENCE_TEST_SEARCH_TERM")
    if env_term:
        return env_term

    if real_search_terms:
        return real_search_terms[0]

    return "confluence"


@pytest.fixture(scope="session")
def embedding_provider(embedding_config) -> EmbeddingProvider | None:
    """
    Provides an initialized EmbeddingProvider instance.
    Uses global config or defaults to a lightweight sentence-transformer.
    Includes teardown. Skips if initialization fails.
    """
    provider_instance: EmbeddingProvider | None = None
    effective_config = embedding_config

    if effective_config is None or effective_config.provider == "none":
        effective_config = EmbeddingConfig(
            provider=DEFAULT_EMBEDDING_PROVIDER_TYPE,
            model_name=DEFAULT_EMBEDDING_MODEL,
            dimension=DEFAULT_EMBEDDING_DIMENSION,
            device=DEFAULT_EMBEDDING_DEVICE,
        )
        try:
            from pathlib import Path

            cache_dir = Path.home() / ".cache" / "confluence-gateway" / "models"
            model_path = cache_dir / f"sentence-transformers_{DEFAULT_EMBEDDING_MODEL}"

            from confluence_gateway.adapters.embedding.sentence_transformer import (
                SentenceTransformerProvider,
            )

            provider_instance = SentenceTransformerProvider(effective_config)
            provider_instance.initialize()
        except Exception as e:
            pytest.skip(
                f"Failed to initialize default embedding provider ({DEFAULT_EMBEDDING_MODEL}): {e}"
            )
            return None
    else:
        try:
            provider_instance = get_embedding_provider(effective_config)
            if provider_instance is None and effective_config.provider != "none":
                raise RuntimeError(
                    "Embedding factory returned None for non-'none' provider."
                )
        except Exception as e:
            pytest.skip(f"Failed to get/initialize configured embedding provider: {e}")
            return None

        if effective_config.provider == "sentence-transformers":
            from confluence_gateway.adapters.embedding.sentence_transformer import (
                SentenceTransformerProvider,
            )

            if not isinstance(provider_instance, SentenceTransformerProvider):
                pytest.skip(
                    f"Configured for sentence-transformers, but factory returned type {type(provider_instance)}. Skipping."
                )
                return None
    yield provider_instance

    if provider_instance and hasattr(provider_instance, "close"):
        try:
            provider_instance.close()
        except Exception as close_e:
            pass


@pytest.fixture(scope="session")
def is_embedding_available(embedding_provider) -> bool:
    return embedding_provider is not None


import typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from confluence_gateway.adapters.embedding.litellm import LiteLLMProvider

    MockedProviderFixture = tuple[LiteLLMProvider, MagicMock]
else:
    MockedProviderFixture = tuple[Any, MagicMock]


@pytest.fixture(scope="function")
def mocked_litellm_provider(
    mocker: MockerFixture, embedding_config: EmbeddingConfig | None
) -> MockedProviderFixture | None:
    effective_config = embedding_config
    if not effective_config or effective_config.provider != "litellm":
        effective_config = EmbeddingConfig(
            provider="litellm",
            model_name="mock-embedding-model",
            dimension=128,
        )

    mock_embedding_call = mocker.patch("litellm.embedding", autospec=True)

    try:
        from confluence_gateway.adapters.embedding.litellm import LiteLLMProvider

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
def effective_embedding_dimension(embedding_provider) -> int | None:
    if not embedding_provider:
        if global_embedding_config and global_embedding_config.dimension:
            return global_embedding_config.dimension
        return DEFAULT_EMBEDDING_DIMENSION
    try:
        return embedding_provider.get_dimension()
    except Exception as e:
        if global_embedding_config and global_embedding_config.dimension:
            return global_embedding_config.dimension
        return DEFAULT_EMBEDDING_DIMENSION


@pytest.fixture(scope="session")
def vector_db_adapter(
    vector_db_config, effective_embedding_dimension
) -> VectorDBAdapter | None:
    """
    Provides an initialized VectorDBAdapter instance.
    Uses global config or defaults to in-memory Qdrant.
    Requires an effective_embedding_dimension. Includes teardown. Skips if initialization fails.
    """
    adapter_instance: VectorDBAdapter | None = None
    effective_vdb_config = vector_db_config

    if effective_embedding_dimension is None:
        pytest.skip("Cannot initialize Vector DB: Embedding dimension is unknown.")
        return None

    if effective_vdb_config is None or effective_vdb_config.type == "none":
        effective_vdb_config = VectorDBConfig(
            type=DEFAULT_VECTOR_DB_TYPE,
            qdrant_url=DEFAULT_VECTOR_DB_URL,
            collection_name=DEFAULT_VECTOR_DB_COLLECTION,
            embedding_dimension=effective_embedding_dimension,
        )
        try:
            if effective_vdb_config.type == "qdrant":
                from confluence_gateway.adapters.vector_db.qdrant_adapter import (
                    QdrantAdapter,
                )

                adapter_instance = QdrantAdapter(effective_vdb_config)
                adapter_instance.initialize()
            else:
                raise NotImplementedError(
                    "Default Vector DB type not implemented for tests"
                )

        except Exception as e:
            pytest.skip(f"Failed to initialize default Vector DB adapter: {e}")
            return None
    else:
        try:
            adapter_instance = get_vector_db_adapter(effective_vdb_config)
            if adapter_instance is None and effective_vdb_config.type != "none":
                raise RuntimeError(
                    "Vector DB factory returned None for non-'none' type."
                )
        except Exception as e:
            pytest.skip(f"Failed to get/initialize configured Vector DB adapter: {e}")
            return None
    yield adapter_instance

    if adapter_instance and hasattr(adapter_instance, "close"):
        try:
            if (
                effective_vdb_config.qdrant_url == ":memory:"
                and effective_vdb_config.type == "qdrant"
                and hasattr(adapter_instance, "client")
                and adapter_instance.client
            ):
                try:
                    adapter_instance.client.delete_collection(
                        collection_name=effective_vdb_config.collection_name
                    )
                except Exception as del_e:
                    pass

            adapter_instance.close()
        except Exception as close_e:
            pass


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
def embedding_service(embedding_provider, is_embedding_available) -> Any | None:
    if not is_embedding_available:
        return None
    from confluence_gateway.services.embedding import EmbeddingService

    return EmbeddingService(provider=embedding_provider)


@pytest.fixture(scope="function")
def mocked_embedding_service(
    mocked_litellm_provider: MockedProviderFixture | None,
) -> Any | None:
    """Provides an EmbeddingService using the mocked LiteLLMProvider."""
    if not mocked_litellm_provider:
        pytest.skip("Mocked LiteLLM provider not available.")
        return None
    provider, _ = mocked_litellm_provider
    from confluence_gateway.services.embedding import EmbeddingService

    return EmbeddingService(provider=provider)


@pytest.fixture(scope="session")
def semantic_search_service(
    confluence_client, embedding_service, vector_db_adapter, is_semantic_search_possible
) -> Any | None:
    """
    Provides a SearchService instance fully configured for semantic search.
    Returns None if semantic search is not possible.
    """
    if not is_semantic_search_possible:
        return None

    from confluence_gateway.services.search import SearchService

    return SearchService(
        client=confluence_client,
        indexing_service=None,
        embedding_service=embedding_service,
        vector_db_adapter=vector_db_adapter,
    )


@pytest.fixture(scope="session")
def standard_search_service(confluence_client, is_real_config_available) -> Any | None:
    if not is_real_config_available:
        return None
    from confluence_gateway.services.search import SearchService

    return SearchService(
        client=confluence_client,
        indexing_service=None,
        embedding_service=None,
        vector_db_adapter=None,
    )


@pytest.fixture(scope="function")
def indexing_service(
    confluence_client: ConfluenceClient | None,
    embedding_service: Any | None,
    vector_db_adapter: VectorDBAdapter | None,
    indexing_config: IndexingConfig,
    search_config: SearchConfig,
    is_semantic_search_possible: bool,
) -> Any | None:
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

    from confluence_gateway.services.indexing import IndexingService

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
    semantic_search_service: Any | None,
    generation_config: GenerationConfig | None,
    is_generation_enabled: bool,
) -> Any | None:
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
        from confluence_gateway.services.generation import GenerationService

        service = GenerationService(
            search_service=semantic_search_service, config=generation_config
        )
        return service
    except Exception as e:
        pytest.skip(f"Failed to initialize GenerationService fixture: {e}")
        return None


@pytest.fixture(scope="session", autouse=False)
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
    Now supports both real generated content and fallback test docs.
    """
    if not is_semantic_search_possible:
        return

    adapter = vector_db_adapter
    embed_svc = embedding_service

    using_real_data = any(
        doc.get("source") == "real_generated_data" for doc in SEMANTIC_TEST_DOCS
    )

    try:
        first_doc_id = SEMANTIC_TEST_DOCS[0]["id"]
        existing_records = adapter.retrieve_by_ids(
            ids=[first_doc_id], with_payload=False, with_vector=False
        )
        if existing_records:
            return
    except Exception as check_err:
        pass

    try:
        texts = [doc["text"] for doc in SEMANTIC_TEST_DOCS]
        embeddings = embed_svc.embed_texts(texts)

        if len(embeddings) != len(SEMANTIC_TEST_DOCS):
            raise RuntimeError(
                "Mismatch between texts and generated embeddings during test data setup."
            )

        documents = []
        for i, doc_data in enumerate(SEMANTIC_TEST_DOCS):
            metadata = {
                "source": doc_data.get("source", "pytest_fixture"),
                "test_run_id": str(uuid.uuid4())[:8],
            }

            if using_real_data:
                metadata.update(
                    {
                        "title": doc_data.get("title", "Unknown"),
                        "space_key": doc_data.get("space_key", "Unknown"),
                        "data_type": "real_generated",
                    }
                )
            else:
                metadata["data_type"] = "fallback"

            documents.append(
                Document(
                    id=doc_data["id"],
                    text=doc_data["text"],
                    embedding=embeddings[i],
                    metadata=metadata,
                )
            )

        adapter.upsert(documents)

    except Exception as e:
        pass


@pytest.fixture(scope="session")
def test_app_client(
    confluence_client: ConfluenceClient | None,
    vector_db_adapter: VectorDBAdapter | None,
    embedding_provider: EmbeddingProvider | None,
    search_config: SearchConfig,
    vector_db_config: VectorDBConfig | None,
    indexing_config: IndexingConfig,
) -> Generator[TestClient, Any, None]:
    from confluence_gateway.api.app import app
    from confluence_gateway.api.dependencies import (
        get_confluence_client,
        get_embedding_provider_dependency,
        get_vector_db_adapter,
    )

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

                from confluence_gateway.services.embedding import EmbeddingService
                from confluence_gateway.services.indexing import IndexingService

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


@pytest.fixture(scope="session")
def search_service(standard_search_service):
    """Alias for standard_search_service for backward compatibility."""
    return standard_search_service


@pytest.fixture(scope="session")
def api_client(test_app_client):
    """Alias for test_app_client for backward compatibility."""
    return test_app_client


@pytest.fixture(scope="session")
def minimal_confluence_client(confluence_config) -> ConfluenceClient | None:
    """Client without connection test for unit tests."""
    if not confluence_config:
        return None
    return ConfluenceClient(config=confluence_config)


@pytest.fixture(scope="session")
def validated_confluence_client(minimal_confluence_client) -> ConfluenceClient | None:
    """Client with validated connection for integration tests."""
    if minimal_confluence_client:
        try:
            minimal_confluence_client.test_connection()
        except Exception as e:
            pytest.skip(f"Could not connect to Confluence: {e}")
            return None
    return minimal_confluence_client


@pytest.fixture
def semantic_test_setup(request):
    """Load semantic search fixtures only when needed."""
    if "semantic" in request.node.name:
        return request.getfixturevalue("index_semantic_test_data")
    return None


@pytest.fixture(scope="session")
def lazy_embedding_provider():
    """Lazy embedding provider that only initializes when accessed."""
    _provider = None

    def get_provider():
        nonlocal _provider
        if _provider is None:
            config = EmbeddingConfig(
                provider=DEFAULT_EMBEDDING_PROVIDER_TYPE,
                model_name=DEFAULT_EMBEDDING_MODEL,
                dimension=DEFAULT_EMBEDDING_DIMENSION,
                device=DEFAULT_EMBEDDING_DEVICE,
            )
            from confluence_gateway.adapters.embedding.sentence_transformer import (
                SentenceTransformerProvider,
            )

            _provider = SentenceTransformerProvider(config)
            _provider.initialize()
        return _provider

    yield get_provider

    if _provider and hasattr(_provider, "close"):
        _provider.close()


@pytest.fixture(scope="session")
def dummy_data_spaces(confluence_client: ConfluenceClient | None) -> list[dict] | None:
    """
    Check for and return available dummy data spaces (TESTDUM*).
    Returns None if no dummy data is found or confluence client is unavailable.
    """
    if not confluence_client:
        return None

    try:
        all_spaces = confluence_client.list_all_spaces(limit=100)
        dummy_spaces = [
            space for space in all_spaces if space.key.startswith("TESTDUM")
        ]

        if dummy_spaces:
            spaces_dict = []
            for space in dummy_spaces:
                try:
                    search_result = confluence_client.search(
                        query=f"space={space.key}",
                        limit=1,
                    )
                    page_count = (
                        len(search_result.results) if search_result.results else 0
                    )

                    spaces_dict.append(
                        {
                            "key": space.key,
                            "name": space.name,
                            "id": getattr(space, "id", None),
                            "page_count": page_count,
                            "has_content": page_count > 0,
                        }
                    )
                except Exception as e:
                    spaces_dict.append(
                        {
                            "key": space.key,
                            "name": space.name,
                            "id": getattr(space, "id", None),
                            "page_count": 0,
                            "has_content": False,
                        }
                    )

            spaces_with_content = [s for s in spaces_dict if s["has_content"]]

            if spaces_with_content:
                return spaces_with_content
            else:
                return None

        else:
            return None

    except Exception as e:
        return None


@pytest.fixture(scope="session")
def test_space_with_content(dummy_data_spaces: list[dict] | None) -> dict | None:
    """
    Provides a single test space with content for tests that need a reliable space.
    Prefers TESTDUMTECH spaces, falls back to any available dummy space with content.
    """
    if not dummy_data_spaces:
        return None

    tech_spaces = [
        s for s in dummy_data_spaces if "TECH" in s["key"] and s["has_content"]
    ]
    if tech_spaces:
        return tech_spaces[0]

    content_spaces = [s for s in dummy_data_spaces if s["has_content"]]
    if content_spaces:
        return content_spaces[0]

    return None


@pytest.fixture(scope="session")
def test_space_with_attachments(
    dummy_data_spaces: list[dict] | None, confluence_client: ConfluenceClient | None
) -> dict | None:
    """
    Provides a test space that has pages with attachments for attachment testing.
    """
    if not dummy_data_spaces or not confluence_client:
        return None

    for space in dummy_data_spaces:
        if not space["has_content"]:
            continue

        try:
            search_result = confluence_client.search(
                query=f"space={space['key']}",
                limit=10,
            )

            if not search_result.results:
                continue

            for page in search_result.results:
                try:
                    if hasattr(confluence_client, "get_attachments"):
                        attachments = confluence_client.get_attachments(page.id)
                    else:
                        attachments = confluence_client.atlassian_api.get_attachments_from_content(
                            page.id
                        )

                    if attachments and (
                        (isinstance(attachments, dict) and attachments.get("results"))
                        or (isinstance(attachments, list) and attachments)
                    ):
                        attachment_count = (
                            len(attachments["results"])
                            if isinstance(attachments, dict)
                            else len(attachments)
                        )

                        space_with_attachments = space.copy()
                        space_with_attachments["has_attachments"] = True
                        space_with_attachments["sample_page_id"] = page.id
                        space_with_attachments["attachment_count"] = attachment_count

                        return space_with_attachments

                except Exception as e:
                    continue

        except Exception as e:
            continue

    return None


@pytest.fixture(scope="session")
def real_search_terms(
    dummy_data_spaces: list[dict] | None, confluence_client
) -> list[str]:
    """
    Provides real search terms extracted from dummy data content for more realistic testing.
    """
    if not dummy_data_spaces or not confluence_client:
        return ["confluence", "documentation", "test"]

    search_terms = set()

    for space in dummy_data_spaces[:2]:
        if not space["has_content"]:
            continue

        try:
            search_result = confluence_client.search(
                query=f"space={space['key']}",
                limit=3,
            )

            for page in search_result.results[:3]:
                title_words = page.title.lower().split()
                for word in title_words:
                    clean_word = "".join(c for c in word if c.isalnum())
                    if len(clean_word) > 4 and clean_word not in [
                        "testdum",
                        "confluence",
                        "documentation",
                    ]:
                        search_terms.add(clean_word)

                if len(search_terms) >= 10:
                    break

        except Exception as e:
            continue

    terms_list = (
        list(search_terms)[:5]
        if search_terms
        else ["confluence", "documentation", "test"]
    )
    return terms_list


@pytest.fixture(scope="session")
def real_content_samples(
    dummy_data_spaces: list[dict] | None, confluence_client
) -> list[dict]:
    """
    Provides samples of real content from dummy data for testing various content processing scenarios.
    """
    if not dummy_data_spaces or not confluence_client:
        return []

    content_samples = []

    for space in dummy_data_spaces:
        if not space["has_content"] or len(content_samples) >= 5:
            continue

        try:
            search_result = confluence_client.search(
                query=f"space={space['key']}",
                limit=2,
            )

            for page in search_result.results[:2]:
                try:
                    page_detail = confluence_client.get_page(
                        page.id, expand="body.storage"
                    )
                    content = (
                        getattr(page_detail.body.storage, "value", "")
                        if page_detail.body
                        else ""
                    )

                    if content and len(content) > 100:
                        content_samples.append(
                            {
                                "id": page.id,
                                "title": page.title,
                                "space_key": space["key"],
                                "content": content,
                                "content_length": len(content),
                                "source": "real_dummy_data",
                            }
                        )

                except Exception as e:
                    continue

        except Exception as e:
            continue

    return content_samples
