import random
import re
import uuid
from collections.abc import Generator
from typing import Any, Optional

import pytest
from confluence_gateway.adapters.confluence.client import ConfluenceClient
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
    SearchConfig,
    VectorDBConfig,
    load_configurations,
)
from confluence_gateway.core.config import (
    embedding_config as global_embedding_config,
)
from confluence_gateway.providers.embedding.factory import (
    EmbeddingProvider,
    get_embedding_provider,
)
from confluence_gateway.providers.embedding.sentence_transformer import (
    SentenceTransformerProvider,
)
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.search import SearchService
from fastapi.testclient import TestClient


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark tests that require a real Confluence connection"
    )


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

SEMANTIC_TEST_DOCS = [
    {"id": "sem_doc1", "text": "This is the first test document about apples."},
    {
        "id": "sem_doc2",
        "text": "The second document discusses oranges and citrus fruits.",
    },
    {"id": "sem_doc3", "text": "Finally, a document mentioning bananas."},
]


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


@pytest.fixture(scope="session", autouse=True)
def index_semantic_test_data(
    semantic_search_service,
    embedding_service,
    vector_db_adapter,
    is_semantic_search_possible,
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
        existing_doc = adapter.get_by_id(SEMANTIC_TEST_DOCS[0]["id"])
        if existing_doc:
            print(
                f"\nINFO (pytest): Semantic test data (e.g., '{SEMANTIC_TEST_DOCS[0]['id']}') seems to exist. Skipping indexing."
            )
            return
    except Exception as get_err:
        print(
            f"\nWARN (pytest): Error checking for existing semantic test data: {get_err}. Attempting indexing."
        )

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


@pytest.fixture(scope="session")
def test_app_client() -> Generator[TestClient, Any, None]:
    with TestClient(app) as client:
        yield client
