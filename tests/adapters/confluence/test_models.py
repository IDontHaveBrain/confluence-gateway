from datetime import datetime

import pytest
from confluence_gateway.adapters.confluence.models import (
    BodyContent,
    ConfluenceAttachment,
    ConfluenceObject,
    ConfluencePage,
    ConfluenceSpace,
    ContentType,
    SearchResult,
    SpaceType,
    Version,
)
from confluence_gateway.core.exceptions import ConfluenceAPIError

from tests.conftest import REAL_CONFIG_SKIP_REASON


class TestConfluenceObject:
    def test_confluence_object_required_fields(self):
        obj = ConfluenceObject(id="123", title="Test Object")
        assert obj.id == "123"
        assert obj.title == "Test Object"
        assert obj.created_at is None
        assert obj.updated_at is None

    def test_confluence_object_all_fields(self):
        now = datetime.now()
        obj = ConfluenceObject(
            id="123", title="Test Object", created_at=now, updated_at=now
        )
        assert obj.id == "123"
        assert obj.title == "Test Object"
        assert obj.created_at == now
        assert obj.updated_at == now

    def test_confluence_object_date_field_mapping(self):
        now = datetime.now()
        obj = ConfluenceObject(
            id="123",
            title="Test Object",
            created=now,
            updated=now,
        )
        assert obj.created_at == now
        assert obj.updated_at == now


class TestConfluenceSpace:
    def test_confluence_space_required_fields(self):
        space = ConfluenceSpace(id="123", title="Test Space", key="TEST")
        assert space.id == "123"
        assert space.title == "Test Space"
        assert space.key == "TEST"
        assert space.description is None
        assert space.type is None

    def test_confluence_space_all_fields(self):
        now = datetime.now()
        space = ConfluenceSpace(
            id="123",
            title="Test Space",
            key="TEST",
            description={"plain": {"value": "Description"}},
            type=SpaceType.GLOBAL,
            created_at=now,
            updated_at=now,
        )
        assert space.id == "123"
        assert space.title == "Test Space"
        assert space.key == "TEST"
        assert space.description == {"plain": {"value": "Description"}}
        assert space.type == SpaceType.GLOBAL
        assert space.created_at == now
        assert space.updated_at == now

    def test_confluence_space_name_title_mapping(self):
        space = ConfluenceSpace(id="123", name="Test Space", key="TEST")
        assert space.title == "Test Space"

    def test_confluence_space_description_handling(self):
        space = ConfluenceSpace(
            id="123",
            title="Test Space",
            key="TEST",
            description={
                "plain": {"value": "Description text", "representation": "plain"}
            },
        )
        assert space.description == {
            "plain": {"value": "Description text", "representation": "plain"}
        }


class TestConfluencePage:
    def test_confluence_page_required_fields(self):
        page = ConfluencePage(id="123", title="Test Page")
        assert page.id == "123"
        assert page.title == "Test Page"
        assert page.content_type == ContentType.PAGE
        assert page.space is None
        assert page.body is None
        assert page.version is None
        assert page.status is None

    def test_confluence_page_all_fields(self):
        now = datetime.now()
        page = ConfluencePage(
            id="123",
            title="Test Page",
            content_type=ContentType.BLOGPOST,
            space={"key": "TEST", "name": "Test Space"},
            body={
                "view": {"value": "<p>HTML content</p>"},
                "storage": {"value": "<p>Storage content</p>"},
                "plain": {"value": "Plain content"},
            },
            version={"number": 5, "when": now},
            status="current",
            created_at=now,
            updated_at=now,
        )
        assert page.id == "123"
        assert page.title == "Test Page"
        assert page.content_type == ContentType.BLOGPOST
        assert page.space == {"key": "TEST", "name": "Test Space"}
        assert isinstance(page.body, BodyContent)
        assert page.body.view == {"value": "<p>HTML content</p>"}
        assert page.body.storage == {"value": "<p>Storage content</p>"}
        assert page.body.plain == {"value": "Plain content"}
        assert isinstance(page.version, Version)
        assert page.version.number == 5
        assert page.version.when == now
        assert page.status == "current"
        assert page.created_at == now
        assert page.updated_at == now

    def test_confluence_page_type_mapping(self):
        page = ConfluencePage(id="123", title="Test Page", type="blogpost")
        assert page.content_type == ContentType.BLOGPOST

    def test_confluence_page_body_property_methods(self):
        page = ConfluencePage(
            id="123",
            title="Test Page",
            body={
                "view": {"value": "<p>HTML content</p>"},
                "storage": {"value": "<p>Storage content</p>"},
                "plain": {"value": "Plain content"},
            },
        )
        assert page.html_content == "<p>HTML content</p>"
        assert page.storage_content == "<p>Storage content</p>"
        assert page.plain_content == "Plain content"

    def test_confluence_page_missing_content(self):
        page = ConfluencePage(id="123", title="Test Page")
        assert page.html_content is None
        assert page.storage_content is None
        assert page.plain_content is None


class TestSearchResult:
    def test_search_result_empty(self):
        result = SearchResult()
        assert result.total_size == 0
        assert result.start == 0
        assert result.limit == 0
        assert result.results == []

    def test_search_result_with_values(self):
        result = SearchResult(
            total_size=100,
            start=0,
            limit=25,
            results=[
                {"id": "123", "title": "Page 1", "type": "page"},
                {"id": "456", "title": "Page 2", "type": "blogpost"},
            ],
        )
        assert result.total_size == 100
        assert result.start == 0
        assert result.limit == 25
        assert len(result.results) == 2
        assert all(isinstance(item, ConfluencePage) for item in result.results)
        assert result.results[0].id == "123"
        assert result.results[0].title == "Page 1"
        assert result.results[0].content_type == ContentType.PAGE
        assert result.results[1].id == "456"
        assert result.results[1].title == "Page 2"
        assert result.results[1].content_type == ContentType.BLOGPOST

    def test_search_result_size_mapping(self):
        result = SearchResult(size=100, start=0, limit=25)
        assert result.total_size == 100

    def test_search_result_with_page_objects(self):
        pages = [
            ConfluencePage(id="123", title="Page 1"),
            ConfluencePage(id="456", title="Page 2"),
        ]
        result = SearchResult(total_size=100, start=0, limit=25, results=pages)
        assert len(result.results) == 2
        assert all(isinstance(item, ConfluencePage) for item in result.results)
        assert result.results[0].id == "123"
        assert result.results[1].id == "456"


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"),
    reason=REAL_CONFIG_SKIP_REASON,
)
class TestListAttachments:
    @pytest.fixture(scope="class")
    def page_attachments_info(self, confluence_client, existing_page_id):
        try:
            attachments = confluence_client.list_attachments(existing_page_id)
            return {"page_id": existing_page_id, "attachments": attachments}
        except Exception as e:
            pytest.skip(f"Could not list attachments for page {existing_page_id}: {e}")
            return None

    def test_list_attachments_success(self, page_attachments_info):
        attachments = page_attachments_info["attachments"]
        if not attachments:
            pytest.skip(
                f"Page {page_attachments_info['page_id']} has no attachments to test."
            )

        assert isinstance(attachments, list)
        assert all(isinstance(att, ConfluenceAttachment) for att in attachments)
        first_att = attachments[0]
        assert isinstance(first_att.id, str)
        assert isinstance(first_att.title, str)
        assert first_att.content_type == ContentType.ATTACHMENT

    def test_list_attachments_empty(self, page_attachments_info):
        attachments = page_attachments_info["attachments"]
        if attachments:
            pytest.skip(
                f"Page {page_attachments_info['page_id']} has attachments, skipping empty test."
            )

        assert attachments == []

    def test_list_attachments_pagination(
        self, confluence_client, page_attachments_info
    ):
        page_id = page_attachments_info["page_id"]
        attachments = page_attachments_info["attachments"]

        if len(attachments) < 2:
            pytest.skip(
                f"Page {page_id} has fewer than 2 attachments, cannot test pagination."
            )

        results_limit_1_start_0 = confluence_client.list_attachments(
            page_id, limit=1, start=0
        )
        assert len(results_limit_1_start_0) == 1
        first_attachment_id = results_limit_1_start_0[0].id

        results_limit_1_start_1 = confluence_client.list_attachments(
            page_id, limit=1, start=1
        )
        assert len(results_limit_1_start_1) == 1
        second_attachment_id = results_limit_1_start_1[0].id

        assert first_attachment_id != second_attachment_id
        assert first_attachment_id == attachments[0].id
        assert second_attachment_id == attachments[1].id

    def test_list_attachments_non_existent_page(self, confluence_client):
        nonexistent_page_id = "0"
        with pytest.raises(ConfluenceAPIError) as excinfo:
            confluence_client.list_attachments(nonexistent_page_id)
        error_message = str(excinfo.value).lower()
        assert (
            excinfo.value.status_code == 404
            or "permission" in error_message
            or "not found" in error_message
        )


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"),
    reason=REAL_CONFIG_SKIP_REASON,
)
class TestDownloadAttachment:
    @pytest.fixture(scope="class")
    def attachment_to_download(self, confluence_client, existing_page_id):
        try:
            attachments = confluence_client.list_attachments(existing_page_id, limit=5)
            if not attachments:
                pytest.skip(
                    f"No attachments found on page {existing_page_id} to test download."
                )
                return None
            return attachments[0]
        except Exception as e:
            pytest.skip(
                f"Could not list attachments for page {existing_page_id} to find one for download: {e}"
            )
            return None

    def test_download_attachment_success(
        self, confluence_client, attachment_to_download
    ):
        if attachment_to_download is None:
            pytest.skip("No attachment fixture available for download test.")

        attachment_id = attachment_to_download.id
        try:
            content = confluence_client.download_attachment(attachment_id)
            assert isinstance(content, bytes)
            assert len(content) > 0
            if (
                attachment_to_download.file_size is not None
                and attachment_to_download.file_size > 0
            ):
                assert len(content) == attachment_to_download.file_size

        except ConfluenceAPIError as e:
            pytest.fail(
                f"Downloading attachment {attachment_id} failed unexpectedly: {e}"
            )

    def test_download_attachment_not_found(self, confluence_client):
        nonexistent_attachment_id = "0"
        with pytest.raises(ConfluenceAPIError) as excinfo:
            confluence_client.download_attachment(nonexistent_attachment_id)
        error_message = str(excinfo.value).lower()
        assert (
            excinfo.value.status_code == 404
            or "permission" in error_message
            or "not found" in error_message
        )


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"),
    reason=REAL_CONFIG_SKIP_REASON,
)
class TestListAttachments:
    @pytest.fixture(scope="class")
    def page_attachments_info(self, confluence_client, existing_page_id):
        try:
            attachments = confluence_client.list_attachments(existing_page_id)
            return {"page_id": existing_page_id, "attachments": attachments}
        except Exception as e:
            pytest.skip(f"Could not list attachments for page {existing_page_id}: {e}")
            return None

    def test_list_attachments_success(self, page_attachments_info):
        attachments = page_attachments_info["attachments"]
        if not attachments:
            pytest.skip(
                f"Page {page_attachments_info['page_id']} has no attachments to test."
            )

        assert isinstance(attachments, list)
        assert all(isinstance(att, ConfluenceAttachment) for att in attachments)
        first_att = attachments[0]
        assert isinstance(first_att.id, str)
        assert isinstance(first_att.title, str)
        assert first_att.content_type == ContentType.ATTACHMENT

    def test_list_attachments_empty(self, page_attachments_info):
        attachments = page_attachments_info["attachments"]
        if attachments:
            pytest.skip(
                f"Page {page_attachments_info['page_id']} has attachments, skipping empty test."
            )

        assert attachments == []

    def test_list_attachments_pagination(
        self, confluence_client, page_attachments_info
    ):
        page_id = page_attachments_info["page_id"]
        attachments = page_attachments_info["attachments"]

        if len(attachments) < 2:
            pytest.skip(
                f"Page {page_id} has fewer than 2 attachments, cannot test pagination."
            )

        results_limit_1_start_0 = confluence_client.list_attachments(
            page_id, limit=1, start=0
        )
        assert len(results_limit_1_start_0) == 1
        first_attachment_id = results_limit_1_start_0[0].id

        results_limit_1_start_1 = confluence_client.list_attachments(
            page_id, limit=1, start=1
        )
        assert len(results_limit_1_start_1) == 1
        second_attachment_id = results_limit_1_start_1[0].id

        assert first_attachment_id != second_attachment_id
        assert first_attachment_id == attachments[0].id
        assert second_attachment_id == attachments[1].id

    def test_list_attachments_non_existent_page(self, confluence_client):
        nonexistent_page_id = "0"
        with pytest.raises(ConfluenceAPIError) as excinfo:
            confluence_client.list_attachments(nonexistent_page_id)
        error_message = str(excinfo.value).lower()
        assert (
            excinfo.value.status_code == 404
            or "permission" in error_message
            or "not found" in error_message
        )


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.lazy_fixture("is_real_config_available"),
    reason=REAL_CONFIG_SKIP_REASON,
)
class TestDownloadAttachment:
    @pytest.fixture(scope="class")
    def attachment_to_download(self, confluence_client, existing_page_id):
        try:
            attachments = confluence_client.list_attachments(existing_page_id, limit=5)
            if not attachments:
                pytest.skip(
                    f"No attachments found on page {existing_page_id} to test download."
                )
                return None
            return attachments[0]
        except Exception as e:
            pytest.skip(
                f"Could not list attachments for page {existing_page_id} to find one for download: {e}"
            )
            return None

    def test_download_attachment_success(
        self, confluence_client, attachment_to_download
    ):
        if attachment_to_download is None:
            pytest.skip("No attachment fixture available for download test.")

        attachment_id = attachment_to_download.id
        try:
            content = confluence_client.download_attachment(attachment_id)
            assert isinstance(content, bytes)
            assert len(content) > 0
            if (
                attachment_to_download.file_size is not None
                and attachment_to_download.file_size > 0
            ):
                assert len(content) == attachment_to_download.file_size

        except ConfluenceAPIError as e:
            pytest.fail(
                f"Downloading attachment {attachment_id} failed unexpectedly: {e}"
            )

    def test_download_attachment_not_found(self, confluence_client):
        nonexistent_attachment_id = "0"
        with pytest.raises(ConfluenceAPIError) as excinfo:
            confluence_client.download_attachment(nonexistent_attachment_id)
        error_message = str(excinfo.value).lower()
        assert (
            excinfo.value.status_code == 404
            or "permission" in error_message
            or "not found" in error_message
        )
