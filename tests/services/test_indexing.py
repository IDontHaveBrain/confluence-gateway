import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from confluence_gateway.adapters.confluence.models import (
    ConfluenceAttachment,
    ConfluencePage,
    ConfluenceSpace,
    ContentType,
    Version,
)
from confluence_gateway.adapters.vector_db.models import Document
from confluence_gateway.core.config import IndexingConfig, SearchConfig, VectorDBConfig
from confluence_gateway.core.exceptions import ConfluenceAPIError
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.indexing import IndexingService

from tests.conftest import (
    REAL_CONFIG_SKIP_REASON,
    SEMANTIC_SEARCH_SKIP_REASON,
)

TEST_SPACE_KEY = "INDEXTEST"
TEST_SPACE_NAME = "Indexing Test Space"
TEST_PAGE_ID_1 = "11111"
TEST_PAGE_TITLE_1 = "Test Page One"
TEST_PAGE_ID_2 = "22222"
TEST_PAGE_TITLE_2 = "Test Page Two"
TEST_ATTACHMENT_ID_1 = "att111"
TEST_ATTACHMENT_TITLE_1 = "test_document.txt"
TEST_HTML_CONTENT = "<p>This is HTML content for page one.</p>"
TEST_STORAGE_CONTENT = "<p>This is storage content for page one.</p>"
TEST_ATTACHMENT_CONTENT = b"This is the text content of the attachment."
TEST_EMBEDDING_DIM = 4
TEST_EMBEDDING_1 = [0.1] * TEST_EMBEDDING_DIM
TEST_EMBEDDING_2 = [0.2] * TEST_EMBEDDING_DIM


@pytest.fixture(scope="module")
def test_indexing_config() -> IndexingConfig:
    return IndexingConfig(
        include_spaces=[TEST_SPACE_KEY],
        include_attachments=True,
        allowed_attachment_extensions=["txt"],
        max_attachment_size_mb=1,
        html_parser="markitdown",
        attachment_parser="unstructured",
    )


@pytest.fixture(scope="module")
def test_vector_db_config(
    vector_db_config, effective_embedding_dimension
) -> VectorDBConfig:
    if vector_db_config and vector_db_config.type != "none":
        if vector_db_config.embedding_dimension != TEST_EMBEDDING_DIM:
            pytest.skip(
                f"Vector DB config dimension ({vector_db_config.embedding_dimension}) doesn't match test dimension ({TEST_EMBEDDING_DIM})"
            )
        return vector_db_config
    else:
        return VectorDBConfig(
            type="qdrant",
            qdrant_url=":memory:",
            collection_name=f"pytest_indexing_{TEST_SPACE_KEY.lower()}",
            embedding_dimension=TEST_EMBEDDING_DIM,
            chunk_size=50,
            chunk_overlap=10,
        )


@pytest.fixture(scope="function")
def indexing_service(
    confluence_client,
    test_indexing_config,
    search_config,
    test_vector_db_config,
    embedding_service,
    vector_db_adapter,
    request,
) -> IndexingService:
    if request.node.get_closest_marker("skipif_semantic_search_unavailable"):
        if not pytest.lazy_fixture("is_semantic_search_possible")(request):
            pytest.skip(SEMANTIC_SEARCH_SKIP_REASON)

    if not confluence_client:
        pytest.skip(REAL_CONFIG_SKIP_REASON)
    if not embedding_service:
        pytest.skip("Embedding service fixture not available.")
    if not vector_db_adapter:
        pytest.skip("Vector DB adapter fixture not available.")

    try:
        vector_db_adapter.delete_by_metadata(filters={"space_key": TEST_SPACE_KEY})
        count = vector_db_adapter.count()
        if count > 0 and vector_db_adapter.search_by_metadata(
            filters={"space_key": TEST_SPACE_KEY}
        ):
            print(
                f"WARN: Vector DB collection '{test_vector_db_config.collection_name}' for space {TEST_SPACE_KEY} not empty after cleanup attempt ({count} items)."
            )
    except Exception as e:
        pytest.fail(f"Failed to clean vector DB before test: {e}")

    service = IndexingService(
        confluence_client=confluence_client,
        indexing_config=test_indexing_config,
        search_config=search_config,
        vector_db_config=test_vector_db_config,
        embedding_service=embedding_service,
    )
    service.vector_db_adapter = vector_db_adapter

    yield service

    try:
        vector_db_adapter.delete_by_metadata(filters={"space_key": TEST_SPACE_KEY})
    except Exception as e:
        print(f"WARN: Failed to clean vector DB after test: {e}")


@pytest.fixture
def mock_space() -> ConfluenceSpace:
    return ConfluenceSpace(
        id="1234", key=TEST_SPACE_KEY, name=TEST_SPACE_NAME, title=TEST_SPACE_NAME
    )


@pytest.fixture
def mock_page_1_summary() -> ConfluencePage:
    return ConfluencePage(
        id=TEST_PAGE_ID_1,
        title=TEST_PAGE_TITLE_1,
        type=ContentType.PAGE,
        status="current",
        created_at=datetime.now(timezone.utc) - timedelta(days=2),
        updated_at=datetime.now(timezone.utc) - timedelta(days=1),
        version=Version(number=1, when=datetime.now(timezone.utc) - timedelta(days=1)),
        space={"key": TEST_SPACE_KEY, "name": TEST_SPACE_NAME},
    )


@pytest.fixture
def mock_page_1_details(mock_page_1_summary) -> ConfluencePage:
    page = mock_page_1_summary.model_copy(deep=True)
    page.body = {
        "storage": {"value": TEST_STORAGE_CONTENT},
        "view": {"value": TEST_HTML_CONTENT},
    }
    return page


@pytest.fixture
def mock_page_2_summary() -> ConfluencePage:
    return ConfluencePage(
        id=TEST_PAGE_ID_2,
        title=TEST_PAGE_TITLE_2,
        type=ContentType.PAGE,
        status="current",
        created_at=datetime.now(timezone.utc) - timedelta(days=3),
        updated_at=datetime.now(timezone.utc) - timedelta(days=3),
        version=Version(number=1, when=datetime.now(timezone.utc) - timedelta(days=3)),
        space={"key": TEST_SPACE_KEY, "name": TEST_SPACE_NAME},
    )


@pytest.fixture
def mock_page_2_details(mock_page_2_summary) -> ConfluencePage:
    page = mock_page_2_summary.model_copy(deep=True)
    page.body = {
        "storage": {"value": "<p>Content for page two.</p>"},
        "view": {"value": "<p>Content for page two.</p>"},
    }
    return page


@pytest.fixture
def mock_attachment_1(mock_page_1_summary) -> ConfluenceAttachment:
    now = datetime.now(timezone.utc)
    return ConfluenceAttachment(
        id=TEST_ATTACHMENT_ID_1,
        title=TEST_ATTACHMENT_TITLE_1,
        type=ContentType.ATTACHMENT,
        status="current",
        created_at=now - timedelta(hours=1),
        updated_at=now - timedelta(hours=1),
        version=Version(number=1, when=now - timedelta(hours=1)),
        extensions={
            "mediaType": "text/plain",
            "fileSize": len(TEST_ATTACHMENT_CONTENT),
            "comment": "",
        },
        _links={
            "download": f"/download/attachments/{TEST_ATTACHMENT_ID_1}/{TEST_ATTACHMENT_TITLE_1}",
            "webui": f"/pages/viewpage.action?pageId={mock_page_1_summary.id}&preview=/{TEST_ATTACHMENT_ID_1}/{TEST_ATTACHMENT_TITLE_1}",
        },
        container={"id": mock_page_1_summary.id, "type": "page"},
    )


def get_vector_db_chunks(adapter, content_id: str) -> list[Document]:
    try:
        metadata_results = adapter.search_by_metadata(
            filters={"original_content_id": content_id},
            select=["text", "chunk_sequence_number", "last_modified"],
        )
        docs = []
        for meta in metadata_results:
            docs.append(
                Document(
                    id=meta.get(
                        "id",
                        f"{content_id}_chunk_{meta.get('chunk_sequence_number', 'unknown')}",
                    ),
                    text=meta.get("text", ""),
                    embedding=[],
                    metadata=meta,
                )
            )
        docs.sort(key=lambda d: d.metadata.get("chunk_sequence_number", -1))
        return docs
    except Exception as e:
        print(f"Error retrieving chunks for {content_id} from vector DB: {e}")
        return []


@pytest.mark.skipif(
    not pytest.lazy_fixture("is_semantic_search_possible"),
    reason=SEMANTIC_SEARCH_SKIP_REASON,
)
@pytest.mark.integration
class TestIndexingServiceStatus:
    @pytest.mark.asyncio
    async def test_initial_status(self, indexing_service):
        assert indexing_service.status["status"] == "idle"
        assert indexing_service.status["last_run_start_time"] is None
        assert indexing_service.status["last_run_end_time"] is None
        assert indexing_service.status["last_error_message"] is None

    @pytest.mark.asyncio
    @patch(
        "confluence_gateway.services.indexing.IndexingService._run_indexing_sync",
        new_callable=MagicMock,
    )
    async def test_status_after_success(self, mock_run_sync, indexing_service):
        mock_run_sync.return_value = None

        start_time = datetime.now(timezone.utc)
        await indexing_service.run_indexing()
        end_time = datetime.now(timezone.utc)

        status = indexing_service.status
        assert status["status"] == "success"
        assert status["last_run_start_time"] >= start_time
        assert status["last_run_end_time"] <= end_time
        assert status["last_run_start_time"] < status["last_run_end_time"]
        assert status["last_error_message"] is None
        mock_run_sync.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "confluence_gateway.services.indexing.IndexingService._run_indexing_sync",
        new_callable=MagicMock,
    )
    async def test_status_after_failure(self, mock_run_sync, indexing_service):
        error_message = "Simulated indexing failure"
        mock_run_sync.side_effect = Exception(error_message)

        start_time = datetime.now(timezone.utc)
        await indexing_service.run_indexing()
        end_time = datetime.now(timezone.utc)

        status = indexing_service.status
        assert status["status"] == "failure"
        assert status["last_run_start_time"] >= start_time
        assert status["last_run_end_time"] <= end_time
        assert status["last_run_start_time"] < status["last_run_end_time"]
        assert error_message in status["last_error_message"]
        mock_run_sync.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "confluence_gateway.services.indexing.IndexingService._run_indexing_sync",
        new_callable=AsyncMock,
    )
    async def test_status_while_running(self, mock_run_sync, indexing_service):
        async def long_running_task(*args, **kwargs):
            await asyncio.sleep(0.2)

        mock_run_sync.side_effect = long_running_task

        task = asyncio.create_task(indexing_service.run_indexing())
        await asyncio.sleep(0.05)

        status_running = indexing_service.status
        assert status_running["status"] == "running"
        assert status_running["last_run_start_time"] is not None
        assert status_running["last_run_end_time"] is None
        assert status_running["last_error_message"] is None

        await task

        status_done = indexing_service.status
        assert status_done["status"] == "success"
        assert status_done["last_run_end_time"] is not None


@pytest.mark.skipif(
    not pytest.lazy_fixture("is_semantic_search_possible"),
    reason=SEMANTIC_SEARCH_SKIP_REASON,
)
@pytest.mark.integration
class TestIndexingServiceRunIndexing:
    @pytest.fixture(autouse=True)
    def setup_mocks(
        self,
        monkeypatch,
        mock_space,
        mock_page_1_summary,
        mock_page_1_details,
        mock_page_2_summary,
        mock_page_2_details,
        mock_attachment_1,
    ):
        self.mock_confluence_client = MagicMock(spec=IndexingService.confluence_client)

        self.spaces = {TEST_SPACE_KEY: mock_space}
        self.pages = {
            TEST_PAGE_ID_1: mock_page_1_details,
            TEST_PAGE_ID_2: mock_page_2_details,
        }
        self.attachments = {TEST_PAGE_ID_1: [mock_attachment_1]}

        def mock_list_all_spaces(*args, **kwargs):
            logging.debug("MOCK: list_all_spaces called")
            return list(self.spaces.values())

        def mock_list_pages_in_space(space_key, *args, **kwargs):
            logging.debug(f"MOCK: list_pages_in_space called for {space_key}")
            return [
                p
                for p in self.pages.values()
                if p.space and p.space.get("key") == space_key
            ]

        def mock_get_page(page_id, *args, **kwargs):
            logging.debug(f"MOCK: get_page called for {page_id}")
            if page_id in self.pages:
                return self.pages[page_id]
            raise ConfluenceAPIError(
                status_code=404, error_message=f"Page {page_id} not found"
            )

        def mock_list_attachments(page_id, *args, **kwargs):
            logging.debug(f"MOCK: list_attachments called for {page_id}")
            return self.attachments.get(page_id, [])

        def mock_download_attachment(attachment_id, *args, **kwargs):
            logging.debug(f"MOCK: download_attachment called for {attachment_id}")
            if attachment_id == TEST_ATTACHMENT_ID_1:
                return TEST_ATTACHMENT_CONTENT
            raise ConfluenceAPIError(
                status_code=404, error_message=f"Attachment {attachment_id} not found"
            )

        monkeypatch.setattr(
            IndexingService.confluence_client, "list_all_spaces", mock_list_all_spaces
        )
        monkeypatch.setattr(
            IndexingService.confluence_client, "search_by_cql", mock_list_pages_in_space
        )
        monkeypatch.setattr(
            IndexingService.confluence_client, "get_page", mock_get_page
        )
        monkeypatch.setattr(
            IndexingService.confluence_client, "list_attachments", mock_list_attachments
        )
        monkeypatch.setattr(
            IndexingService.confluence_client,
            "download_attachment",
            mock_download_attachment,
        )

        self.mock_embedding_service = MagicMock(spec=EmbeddingService)

        def mock_embed_texts(texts):
            return [
                TEST_EMBEDDING_1 if i % 2 == 0 else TEST_EMBEDDING_2
                for i in range(len(texts))
            ]

        self.mock_embedding_service.embed_texts.side_effect = mock_embed_texts
        monkeypatch.setattr(
            IndexingService, "embedding_service", self.mock_embedding_service
        )

    @pytest.mark.asyncio
    async def test_initial_indexing(self, indexing_service, vector_db_adapter):
        await indexing_service.run_indexing()

        status = indexing_service.status
        assert status["status"] == "success"

        page1_chunks = get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_1)
        assert len(page1_chunks) > 0
        assert page1_chunks[0].metadata["title"] == TEST_PAGE_TITLE_1
        assert page1_chunks[0].metadata["space_key"] == TEST_SPACE_KEY
        assert page1_chunks[0].metadata["document_type"] == "page"
        assert "This is storage content" in page1_chunks[0].text

        page2_chunks = get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_2)
        assert len(page2_chunks) > 0
        assert page2_chunks[0].metadata["title"] == TEST_PAGE_TITLE_2

        attach1_chunks = get_vector_db_chunks(vector_db_adapter, TEST_ATTACHMENT_ID_1)
        assert len(attach1_chunks) > 0
        assert attach1_chunks[0].metadata["title"] == TEST_ATTACHMENT_TITLE_1
        assert attach1_chunks[0].metadata["space_key"] == TEST_SPACE_KEY
        assert attach1_chunks[0].metadata["document_type"] == "attachment"
        assert attach1_chunks[0].metadata["parent_page_id"] == TEST_PAGE_ID_1
        assert "text content of the attachment" in attach1_chunks[0].text

    @pytest.mark.asyncio
    async def test_reindexing_updated_page(
        self, indexing_service, vector_db_adapter, mock_page_1_details
    ):
        await indexing_service.run_indexing()
        initial_chunks = get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_1)
        assert len(initial_chunks) > 0
        initial_timestamp = initial_chunks[0].metadata.get("last_modified")
        assert initial_timestamp is not None

        new_update_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        mock_page_1_details.updated_at = new_update_time
        mock_page_1_details.version = Version(number=2, when=new_update_time)
        mock_page_1_details.body["storage"]["value"] = "<p>Updated storage content.</p>"
        self.pages[TEST_PAGE_ID_1] = mock_page_1_details

        await indexing_service.run_indexing()
        status = indexing_service.status
        assert status["status"] == "success"

        updated_chunks = get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_1)
        assert len(updated_chunks) > 0
        assert updated_chunks[0].metadata.get("last_modified") > initial_timestamp
        assert "Updated storage content" in updated_chunks[0].text

    @pytest.mark.asyncio
    async def test_indexing_no_change(self, indexing_service, vector_db_adapter):
        await indexing_service.run_indexing()
        initial_chunks_p1 = get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_1)
        initial_chunks_p2 = get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_2)
        initial_chunks_a1 = get_vector_db_chunks(
            vector_db_adapter, TEST_ATTACHMENT_ID_1
        )
        assert len(initial_chunks_p1) > 0
        assert len(initial_chunks_p2) > 0
        assert len(initial_chunks_a1) > 0

        with (
            patch.object(
                vector_db_adapter, "upsert", wraps=vector_db_adapter.upsert
            ) as mock_upsert,
            patch.object(
                vector_db_adapter,
                "delete_by_metadata",
                wraps=vector_db_adapter.delete_by_metadata,
            ) as mock_delete,
        ):
            await indexing_service.run_indexing()
            status = indexing_service.status
            assert status["status"] == "success"

            assert mock_upsert.call_count == 0
            delete_calls_for_content = [
                call
                for call in mock_delete.call_args_list
                if call[1].get("filters", {}).get("original_content_id")
                in [TEST_PAGE_ID_1, TEST_PAGE_ID_2, TEST_ATTACHMENT_ID_1]
            ]
            assert len(delete_calls_for_content) == 0

        final_chunks_p1 = get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_1)
        assert final_chunks_p1 == initial_chunks_p1

    @pytest.mark.asyncio
    async def test_indexing_deleted_page(self, indexing_service, vector_db_adapter):
        await indexing_service.run_indexing()
        assert len(get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_1)) > 0
        assert len(get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_2)) > 0

        del self.pages[TEST_PAGE_ID_2]

        await indexing_service.run_indexing()
        status = indexing_service.status
        assert status["status"] == "success"

        assert len(get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_1)) > 0
        assert len(get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_2)) == 0

    @pytest.mark.asyncio
    async def test_indexing_skip_attachment_by_config(
        self, indexing_service, vector_db_adapter, test_indexing_config
    ):
        test_indexing_config.include_attachments = False
        indexing_service.indexing_config = test_indexing_config

        await indexing_service.run_indexing()
        status = indexing_service.status
        assert status["status"] == "success"

        assert len(get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_1)) > 0
        assert len(get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_2)) > 0
        assert len(get_vector_db_chunks(vector_db_adapter, TEST_ATTACHMENT_ID_1)) == 0

        test_indexing_config.include_attachments = True

    @pytest.mark.asyncio
    async def test_indexing_api_error_handling(
        self, indexing_service, vector_db_adapter, caplog
    ):
        await indexing_service.run_indexing()
        assert len(get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_1)) > 0
        assert len(get_vector_db_chunks(vector_db_adapter, TEST_PAGE_ID_2)) > 0
        assert len(get_vector_db_chunks(vector_db_adapter, TEST_ATTACHMENT_ID_1)) > 0

        original_get_page = IndexingService.confluence_client.get_page

        def failing_get_page(page_id, *args, **kwargs):
            if page_id == TEST_PAGE_ID_1:
                logging.debug(f"MOCK: Simulating API error for get_page {page_id}")
                raise ConfluenceAPIError(status_code=500, error_message="Server error")
            return original_get_page(page_id, *args, **kwargs)

        with patch.object(
            IndexingService.confluence_client, "get_page", side_effect=failing_get_page
        ):
            with caplog.at_level(logging.ERROR):
                await indexing_service.run_indexing()

        status = indexing_service.status
        assert status["status"] == "failure"
        assert "Failed to fetch/process page 11111" in caplog.text
