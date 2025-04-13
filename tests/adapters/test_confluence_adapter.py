import time

import pytest
from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import (
    ConfluenceAttachment,
    ConfluencePage,
    ConfluenceSpace,
    ContentType,
    SearchResult,
)
from confluence_gateway.core.exceptions import ConfluenceAPIError

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client(confluence_client: ConfluenceClient) -> ConfluenceClient:
    if not confluence_client:
        pytest.skip("Confluence client fixture not available.")
    return confluence_client


def test_confluence_connection(client: ConfluenceClient):
    assert client.test_connection() is True


def test_list_all_spaces(client: ConfluenceClient):
    spaces = client.list_all_spaces(limit=5)
    assert isinstance(spaces, list)
    if spaces:
        assert all(isinstance(s, ConfluenceSpace) for s in spaces)
        assert spaces[0].id is not None
        assert spaces[0].key is not None


def test_get_space(client: ConfluenceClient):
    spaces = client.list_all_spaces(limit=1)
    if not spaces:
        pytest.skip("No spaces found to test get_space.")

    space_key = spaces[0].key
    space = client.get_space(space_key)
    assert isinstance(space, ConfluenceSpace)
    assert space.key == space_key
    assert space.id is not None


def test_search_basic(client: ConfluenceClient, real_search_term: str):
    search_result = client.search(query=real_search_term, limit=5)
    assert isinstance(search_result, SearchResult)
    assert isinstance(search_result.results, list)
    assert search_result.limit is not None
    assert search_result.start is not None
    assert search_result.total_size is not None
    if search_result.results:
        assert isinstance(search_result.results[0], ConfluencePage)


def test_search_by_cql(client: ConfluenceClient, real_search_term: str):
    cql = f'text ~ "{client._escape_cql(real_search_term)}" and type = page'
    search_result = client.search_by_cql(cql=cql, limit=5)
    assert isinstance(search_result, SearchResult)
    assert isinstance(search_result.results, list)
    if search_result.results:
        assert all(p.content_type == ContentType.PAGE for p in search_result.results)


def test_get_page(client: ConfluenceClient, real_search_term: str):
    search_result = client.search(query=real_search_term, limit=1)
    if not search_result.results:
        pytest.skip(
            f"No page found for search term '{real_search_term}' to test get_page."
        )

    page_id = search_result.results[0].id
    page = client.get_page(page_id)
    assert isinstance(page, ConfluencePage)
    assert page.id == page_id
    assert page.title is not None
    assert page.space is not None


def test_get_non_existent_page(client: ConfluenceClient):
    non_existent_id = "0000000000"
    with pytest.raises(ConfluenceAPIError) as excinfo:
        client.get_page(non_existent_id)
    assert excinfo.value.status_code == 404 or "not found" in str(excinfo.value).lower()


def test_list_attachments(client: ConfluenceClient, real_search_term: str):
    search_result = client.search(query=real_search_term, limit=1, content_type="page")
    if not search_result.results:
        pytest.skip(
            f"No page found for search term '{real_search_term}' to test list_attachments."
        )

    page_id = search_result.results[0].id
    attachments = client.list_attachments(page_id=page_id, limit=5)
    assert isinstance(attachments, list)
    if attachments:
        assert all(isinstance(a, ConfluenceAttachment) for a in attachments)
        assert attachments[0].id is not None
        assert attachments[0].title is not None


@pytest.mark.skipif(
    True, reason="Downloading can be slow and depends on finding attachments"
)
def test_download_attachment(client: ConfluenceClient, real_search_term: str):
    common_terms = [real_search_term, "template", "report", "meeting notes"]
    page_id_with_attachment = None
    attachment_id = None
    attachment_title = None

    for term in common_terms:
        search_result = client.search(query=term, limit=5, content_type="page")
        if not search_result.results:
            continue

        for page in search_result.results:
            try:
                time.sleep(0.2)
                attachments = client.list_attachments(page_id=page.id, limit=1)
                if attachments:
                    page_id_with_attachment = page.id
                    attachment_id = attachments[0].id
                    attachment_title = attachments[0].title
                    print(
                        f"\nINFO: Found attachment '{attachment_title}' ({attachment_id}) on page {page_id_with_attachment} for download test."
                    )
                    break
            except Exception as e:
                print(f"\nWARN: Error listing attachments for page {page.id}: {e}")
                continue
        if page_id_with_attachment:
            break

    if not attachment_id:
        pytest.skip(
            "Could not find any attachments on searched pages to test download."
        )

    downloaded_content = client.download_attachment(attachment_id)
    assert isinstance(downloaded_content, bytes)
    assert len(downloaded_content) > 0
