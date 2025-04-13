from unittest.mock import AsyncMock, MagicMock, call

import pytest
from confluence_gateway.adapters.confluence.models import (
    ConfluenceAttachment,
    ConfluencePage,
    ConfluenceSpace,
)
from confluence_gateway.adapters.vector_db import Document
from confluence_gateway.adapters.vector_db.base_adapter import VectorDBAdapter
from confluence_gateway.services.embedding import EmbeddingService
from confluence_gateway.services.indexing import IndexingService


def create_mock_page(
    page_id: str,
    title: str,
    space_key: str,
    version: int = 1,
    last_modified: str = "2023-01-01T10:00:00Z",
) -> MagicMock:
    page = MagicMock(spec=ConfluencePage)
    page.id = page_id
    page.title = title
    page.space = MagicMock(spec=ConfluenceSpace)
    page.space.key = space_key
    page.version = {"number": version}
    page.updated_at = None
    page.created_at = None
    from datetime import datetime, timezone

    mock_dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
    page.updated_at = mock_dt
    page.storage_content = f"<html><body>Content for {title}</body></html>"
    return page


def create_mock_attachment(
    attach_id: str,
    title: str,
    page_id: str,
    last_modified: str = "2023-01-01T10:00:00Z",
) -> MagicMock:
    attachment = MagicMock(spec=ConfluenceAttachment)
    attachment.id = attach_id
    attachment.title = title
    attachment.container = {"id": page_id}
    from datetime import datetime, timezone

    mock_dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
    attachment.updated_at = mock_dt
    attachment.file_size = 1024
    attachment.media_type = "application/pdf"
    return attachment


@pytest.mark.integration
class TestIndexingServiceLogic:
    def setup_confluence_mocks(self, mocker, indexing_service: IndexingService):
        mock_client = MagicMock(spec=indexing_service.confluence_client)
        indexing_service.confluence_client = mock_client
        mock_client.list_all_spaces = MagicMock(return_value=[])
        mock_client.get_space = MagicMock(
            return_value=MagicMock(spec=ConfluenceSpace, key="TEST", title="Test Space")
        )
        mock_client.search_by_cql = MagicMock(
            return_value=MagicMock(results=[], total_size=0)
        )
        mock_client.get_page = MagicMock(return_value=None)
        mock_client.list_attachments = MagicMock(return_value=[])
        mock_client.download_attachment = MagicMock(
            return_value=b"mock attachment content"
        )
        mock_client.extract_content_fields = MagicMock(
            side_effect=lambda obj: {
                "id": obj.id,
                "title": obj.title,
                "space_key": getattr(getattr(obj, "space", None), "key", "TEST"),
                "updated_at": obj.updated_at,
            }
        )
        return mock_client

    @pytest.mark.asyncio
    async def test_index_new_page_no_attachments(
        self,
        indexing_service: IndexingService,
        mocker,
        vector_db_adapter: VectorDBAdapter,
        embedding_service: EmbeddingService,
    ):
        mock_confluence_client = self.setup_confluence_mocks(mocker, indexing_service)
        space_key = "NEWSPACE"
        page1 = create_mock_page(
            "page1",
            "New Page One",
            space_key,
            version=1,
            last_modified="2024-01-01T10:00:00Z",
        )
        mock_confluence_client.list_all_spaces.return_value = [
            MagicMock(spec=ConfluenceSpace, key=space_key, title="New Space")
        ]
        mock_confluence_client.search_by_cql.return_value = MagicMock(
            results=[page1], total_size=1
        )
        mock_confluence_client.get_page.return_value = page1
        mock_confluence_client.list_attachments.return_value = []
        search_meta_spy = mocker.spy(vector_db_adapter, "search_by_metadata")
        search_meta_spy.side_effect = [
            [],  # _should_index_content (timestamp check)
            [],  # _cleanup_deleted_content_for_space
        ]
        mock_confluence_client.get_space.return_value = MagicMock(
            spec=ConfluenceSpace, key=space_key, title="New Space"
        )
        upsert_spy = mocker.spy(vector_db_adapter, "upsert")
        delete_spy = mocker.spy(vector_db_adapter, "delete_by_metadata")
        embed_texts_spy = mocker.spy(embedding_service, "embed_texts")
        await indexing_service.run_indexing(space_keys=[space_key])
        mock_confluence_client.get_space.assert_called_once_with(space_key)
        mock_confluence_client.search_by_cql.assert_called_once()
        mock_confluence_client.get_page.assert_any_call(
            page1.id, expand=["version", "space"]
        )
        mock_confluence_client.get_page.assert_any_call(
            page1.id, expand=["body.storage", "version", "space"]
        )
        search_meta_spy.assert_any_call(
            filters={"original_content_id": page1.id}, select=["last_modified"], limit=1
        )
        embed_texts_spy.assert_called_once()
        texts_passed_to_embed = embed_texts_spy.call_args[0][0]
        assert texts_passed_to_embed, "No texts were passed to embed_texts"
        assert texts_passed_to_embed[0] == "Content for New Page One"
        upsert_spy.assert_called_once()
        assert isinstance(upsert_spy.call_args[1]["documents"][0], Document)
        assert (
            upsert_spy.call_args[1]["documents"][0].metadata["original_content_id"]
            == page1.id
        )
        assert (
            upsert_spy.call_args[1]["documents"][0].metadata["document_type"] == "page"
        )
        delete_spy.assert_not_called()
        assert len(search_meta_spy.call_args_list) >= 2, (
            "Expected at least two calls to search_by_metadata"
        )
        cleanup_call = search_meta_spy.call_args_list[1]
        cleanup_call.assert_called_with(
            filters={"space_key": space_key}, select=["original_content_id"]
        )

    @pytest.mark.asyncio
    async def test_index_updated_page(
        self,
        indexing_service: IndexingService,
        mocker,
        vector_db_adapter: VectorDBAdapter,
        embedding_service: EmbeddingService,
    ):
        mock_confluence_client = self.setup_confluence_mocks(mocker, indexing_service)
        space_key = "UPDSPACE"
        page_updated = create_mock_page(
            "page_upd",
            "Updated Page",
            space_key,
            version=2,
            last_modified="2024-02-01T12:00:00Z",
        )
        mock_confluence_client.list_all_spaces.return_value = [
            MagicMock(spec=ConfluenceSpace, key=space_key, title="Update Space")
        ]
        mock_confluence_client.search_by_cql.return_value = MagicMock(
            results=[page_updated], total_size=1
        )
        mock_confluence_client.get_page.return_value = page_updated
        mock_confluence_client.list_attachments.return_value = []
        search_meta_spy = mocker.spy(vector_db_adapter, "search_by_metadata")
        search_meta_spy.side_effect = [
            [
                {
                    "original_content_id": page_updated.id,
                    "last_modified": "2024-01-01T10:00:00Z",
                }
            ],  # _should_index_content
            [],  # _cleanup_deleted_content_for_space
        ]
        mock_confluence_client.get_space.return_value = MagicMock(
            spec=ConfluenceSpace, key=space_key, title="Update Space"
        )
        upsert_spy = mocker.spy(vector_db_adapter, "upsert")
        delete_spy = mocker.spy(vector_db_adapter, "delete_by_metadata")
        embed_texts_spy = mocker.spy(embedding_service, "embed_texts")
        await indexing_service.run_indexing(space_keys=[space_key])
        search_meta_spy.assert_any_call(
            filters={"original_content_id": page_updated.id},
            select=["last_modified"],
            limit=1,
        )
        delete_spy.assert_called_once_with(
            filters={"original_content_id": page_updated.id}
        )
        upsert_spy.assert_called_once()
        embed_texts_spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_unchanged_page(
        self,
        indexing_service: IndexingService,
        mocker,
        vector_db_adapter: VectorDBAdapter,
        embedding_service: EmbeddingService,
    ):
        mock_confluence_client = self.setup_confluence_mocks(mocker, indexing_service)
        space_key = "SKIPSPACE"
        page_unchanged = create_mock_page(
            "page_skip",
            "Unchanged Page",
            space_key,
            version=1,
            last_modified="2024-01-01T10:00:00Z",
        )
        mock_confluence_client.list_all_spaces.return_value = [
            MagicMock(spec=ConfluenceSpace, key=space_key, title="Skip Space")
        ]
        mock_confluence_client.search_by_cql.return_value = MagicMock(
            results=[page_unchanged], total_size=1
        )
        mock_confluence_client.get_page.return_value = page_unchanged
        mock_confluence_client.list_attachments.return_value = []
        search_meta_spy = mocker.spy(vector_db_adapter, "search_by_metadata")
        search_meta_spy.side_effect = [
            [
                {
                    "original_content_id": page_unchanged.id,
                    "last_modified": "2024-01-01T10:00:00Z",
                }
            ],  # _should_index_content
            [],  # _cleanup_deleted_content_for_space
        ]
        mock_confluence_client.get_space.return_value = MagicMock(
            spec=ConfluenceSpace, key=space_key, title="Skip Space"
        )
        upsert_spy = mocker.spy(vector_db_adapter, "upsert")
        delete_spy = mocker.spy(vector_db_adapter, "delete_by_metadata")
        embed_texts_spy = mocker.spy(embedding_service, "embed_texts")
        await indexing_service.run_indexing(space_keys=[space_key])
        search_meta_spy.assert_any_call(
            filters={"original_content_id": page_unchanged.id},
            select=["last_modified"],
            limit=1,
        )
        upsert_spy.assert_not_called()
        delete_spy.assert_not_called()
        embed_texts_spy.assert_not_called()
        mock_confluence_client.get_page.assert_called_once_with(
            page_unchanged.id, expand=["version", "space"]
        )

    @pytest.mark.asyncio
    async def test_cleanup_deleted_page(
        self,
        indexing_service: IndexingService,
        mocker,
        vector_db_adapter: VectorDBAdapter,
    ):
        mock_confluence_client = self.setup_confluence_mocks(mocker, indexing_service)
        space_key = "DELSPACE"
        mock_confluence_client.list_all_spaces.return_value = [
            MagicMock(spec=ConfluenceSpace, key=space_key, title="Delete Space")
        ]
        mock_confluence_client.search_by_cql.return_value = MagicMock(
            results=[], total_size=0
        )
        search_meta_spy = mocker.spy(vector_db_adapter, "search_by_metadata")
        search_meta_spy.side_effect = [
            [{"original_content_id": "deleted_page"}],
        ]
        mock_confluence_client.get_space.return_value = MagicMock(
            spec=ConfluenceSpace, key=space_key, title="Delete Space"
        )
        upsert_spy = mocker.spy(vector_db_adapter, "upsert")
        delete_spy = mocker.spy(vector_db_adapter, "delete_by_metadata")
        await indexing_service.run_indexing(space_keys=[space_key])
        upsert_spy.assert_not_called()
        delete_spy.assert_called_once_with(
            filters={"original_content_id": "deleted_page"}
        )
        search_meta_spy.assert_called_once_with(
            filters={"space_key": space_key}, select=["original_content_id"]
        )

    @pytest.mark.asyncio
    async def test_index_page_with_new_attachment(
        self,
        indexing_service: IndexingService,
        mocker,
        vector_db_adapter: VectorDBAdapter,
        embedding_service: EmbeddingService,
    ):
        mock_confluence_client = self.setup_confluence_mocks(mocker, indexing_service)
        space_key = "ATTACHSPACE"
        page_attach = create_mock_page(
            "page_att",
            "Page With Attachment",
            space_key,
            last_modified="2024-03-01T10:00:00Z",
        )
        attach1 = create_mock_attachment(
            "att1",
            "new_document.pdf",
            page_attach.id,
            last_modified="2024-03-01T11:00:00Z",
        )
        mock_confluence_client.list_all_spaces.return_value = [
            MagicMock(spec=ConfluenceSpace, key=space_key, title="Attachment Space")
        ]
        mock_confluence_client.search_by_cql.return_value = MagicMock(
            results=[page_attach], total_size=1
        )
        mock_confluence_client.get_page.return_value = page_attach
        mock_confluence_client.list_attachments.return_value = [attach1]
        mock_confluence_client.download_attachment.return_value = (
            b"Mock PDF content for attachment"
        )
        search_meta_spy = mocker.spy(vector_db_adapter, "search_by_metadata")
        search_meta_spy.side_effect = [
            [],  # Timestamp check for page
            [],  # Timestamp check for attachment
            [],  # Cleanup check
        ]
        mock_confluence_client.get_space.return_value = MagicMock(
            spec=ConfluenceSpace, key=space_key, title="Attachment Space"
        )
        upsert_spy = mocker.spy(vector_db_adapter, "upsert")
        delete_spy = mocker.spy(vector_db_adapter, "delete_by_metadata")
        embed_texts_spy = mocker.spy(embedding_service, "embed_texts")
        original_include_attachments = (
            indexing_service.indexing_config.include_attachments
        )
        indexing_service.indexing_config.include_attachments = True
        try:
            await indexing_service.run_indexing(space_keys=[space_key])
        finally:
            indexing_service.indexing_config.include_attachments = (
                original_include_attachments
            )
        mock_confluence_client.list_attachments.assert_called_once_with(page_attach.id)
        mock_confluence_client.download_attachment.assert_called_once_with(attach1.id)
        search_meta_spy.assert_any_call(
            filters={"original_content_id": page_attach.id},
            select=["last_modified"],
            limit=1,
        )
        search_meta_spy.assert_any_call(
            filters={"original_content_id": attach1.id},
            select=["last_modified"],
            limit=1,
        )
        assert embed_texts_spy.call_count == 2
        assert (
            "Content for Page With Attachment"
            in embed_texts_spy.call_args_list[0][0][0][0]
        )
        assert upsert_spy.call_count == 2
        page_doc_found = False
        attach_doc_found = False
        for call_args in upsert_spy.call_args_list:
            doc = call_args[1]["documents"][0]
            if doc.metadata["original_content_id"] == page_attach.id:
                assert doc.metadata["document_type"] == "page"
                page_doc_found = True
            elif doc.metadata["original_content_id"] == attach1.id:
                assert doc.metadata["document_type"] == "attachment"
                assert doc.metadata["parent_page_id"] == page_attach.id
                attach_doc_found = True
        assert page_doc_found and attach_doc_found
        delete_spy.assert_not_called()
        assert len(search_meta_spy.call_args_list) >= 3, (
            "Expected at least three calls to search_by_metadata"
        )
        cleanup_call = search_meta_spy.call_args_list[2]
        cleanup_call.assert_called_with(
            filters={"space_key": space_key}, select=["original_content_id"]
        )
