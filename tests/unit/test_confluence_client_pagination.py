from unittest.mock import MagicMock, patch

import pytest
from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import ConfluenceSpace, SpaceType
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
)


class TestConfluenceClientPagination:
    """Test the pagination functionality in ConfluenceClient."""

    @pytest.fixture
    def mock_confluence_api(self):
        """Create a mock Confluence API instance."""
        return MagicMock()

    @pytest.fixture
    def client(self, mock_confluence_api):
        """Create a ConfluenceClient instance with mocked API."""
        with patch(
            "confluence_gateway.adapters.confluence.client.Confluence"
        ) as mock_conf:
            mock_conf.return_value = mock_confluence_api
            client = ConfluenceClient()
            client.atlassian_api = mock_confluence_api
            return client

    def test_list_spaces_paginated_success(self, client, mock_confluence_api):
        """Test successful paginated space listing."""
        mock_response = {
            "results": [
                {
                    "id": "1",
                    "key": "SPACE1",
                    "name": "Space 1",
                    "type": "global",
                },
                {
                    "id": "2",
                    "key": "SPACE2",
                    "name": "Space 2",
                    "type": "personal",
                },
            ],
            "size": 10,  # Total count
            "_links": {"next": "/rest/api/space?start=2&limit=2"},
        }

        mock_confluence_api.get_all_spaces.return_value = mock_response

        spaces, total_count = client.list_spaces_paginated(start=0, limit=2)

        assert len(spaces) == 2
        assert total_count == 10
        assert spaces[0].key == "SPACE1"
        assert spaces[1].key == "SPACE2"

        mock_confluence_api.get_all_spaces.assert_called_once_with(
            start=0,
            limit=2,
            expand="description.plain",
            space_type=None,
            space_status=None,
        )

    def test_list_spaces_paginated_with_descriptions(self, client, mock_confluence_api):
        """Test paginated listing with space descriptions."""
        mock_response = {
            "results": [
                {
                    "id": "1",
                    "key": "SPACE1",
                    "name": "Space 1",
                    "type": "global",
                    "description": {"plain": {"value": "This is space 1 description"}},
                },
            ],
            "size": 1,
            "_links": {},
        }

        mock_confluence_api.get_all_spaces.return_value = mock_response

        spaces, total_count = client.list_spaces_paginated(start=0, limit=10)

        assert len(spaces) == 1
        assert total_count == 1
        assert spaces[0].description_text == "This is space 1 description"

    def test_list_spaces_paginated_empty_results(self, client, mock_confluence_api):
        """Test paginated listing with no results."""
        mock_response = {"results": [], "size": 0, "_links": {}}

        mock_confluence_api.get_all_spaces.return_value = mock_response

        spaces, total_count = client.list_spaces_paginated(start=50, limit=25)

        assert len(spaces) == 0
        assert total_count == 0

    def test_list_spaces_paginated_no_size_field(self, client, mock_confluence_api):
        """Test handling when API doesn't return size field."""
        mock_response = {
            "results": [
                {
                    "id": "1",
                    "key": "SPACE1",
                    "name": "Space 1",
                    "type": "global",
                },
            ],
            "_links": {"next": "/rest/api/space?start=1&limit=1"},
        }

        mock_confluence_api.get_all_spaces.return_value = mock_response

        spaces, total_count = client.list_spaces_paginated(start=0, limit=1)

        assert len(spaces) == 1
        # Should estimate total based on presence of next link
        assert total_count == 2  # start + len(spaces) + 1 (for next)

    def test_list_spaces_paginated_parse_error(self, client, mock_confluence_api):
        """Test handling of parse errors for individual spaces."""
        mock_response = {
            "results": [
                {
                    "id": "1",
                    "key": "SPACE1",
                    "name": "Space 1",
                    "type": "global",
                },
                {
                    # Invalid space data - missing required fields
                    "invalid": "data",
                },
                {
                    "id": "3",
                    "key": "SPACE3",
                    "name": "Space 3",
                    "type": "global",
                },
            ],
            "size": 3,
            "_links": {},
        }

        mock_confluence_api.get_all_spaces.return_value = mock_response

        spaces, total_count = client.list_spaces_paginated(start=0, limit=10)

        # Should parse 2 valid spaces and skip the invalid one
        assert len(spaces) == 2
        assert total_count == 3
        assert spaces[0].key == "SPACE1"
        assert spaces[1].key == "SPACE3"

    def test_list_spaces_paginated_authentication_error(
        self, client, mock_confluence_api
    ):
        """Test handling of authentication errors."""
        mock_confluence_api.get_all_spaces.side_effect = Exception("401 Unauthorized")

        with pytest.raises(ConfluenceAuthenticationError) as exc_info:
            client.list_spaces_paginated(start=0, limit=25)

        assert "Authentication failed" in str(exc_info.value)

    def test_list_spaces_paginated_connection_error(self, client, mock_confluence_api):
        """Test handling of connection errors."""
        import requests

        mock_confluence_api.get_all_spaces.side_effect = (
            requests.exceptions.ConnectionError("Connection refused")
        )

        with pytest.raises(ConfluenceConnectionError):
            client.list_spaces_paginated(start=0, limit=25)

    def test_list_spaces_paginated_api_error(self, client, mock_confluence_api):
        """Test handling of general API errors."""
        mock_confluence_api.get_all_spaces.side_effect = Exception("Some API error")

        with pytest.raises(ConfluenceAPIError) as exc_info:
            client.list_spaces_paginated(start=0, limit=25)

        assert "Some API error" in str(exc_info.value)

    def test_list_spaces_paginated_different_page_sizes(
        self, client, mock_confluence_api
    ):
        """Test pagination with different page sizes."""
        # Test small page size
        mock_response = {
            "results": [
                {
                    "id": str(i),
                    "key": f"SPACE{i}",
                    "name": f"Space {i}",
                    "type": "global",
                }
                for i in range(1, 6)
            ],
            "size": 50,
            "_links": {"next": "/rest/api/space?start=5&limit=5"},
        }

        mock_confluence_api.get_all_spaces.return_value = mock_response

        spaces, total_count = client.list_spaces_paginated(start=0, limit=5)
        assert len(spaces) == 5
        assert total_count == 50

        # Test large page size
        mock_response["results"] = [
            {"id": str(i), "key": f"SPACE{i}", "name": f"Space {i}", "type": "global"}
            for i in range(1, 101)
        ]
        mock_response["_links"] = {}

        mock_confluence_api.get_all_spaces.return_value = mock_response

        spaces, total_count = client.list_spaces_paginated(start=0, limit=100)
        assert len(spaces) == 100
        assert total_count == 50  # Still returns the size from response
