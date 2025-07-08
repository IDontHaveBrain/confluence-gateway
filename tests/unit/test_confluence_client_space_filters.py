from unittest.mock import MagicMock, patch

import pytest
from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.adapters.confluence.models import ConfluenceSpace, SpaceType


class TestConfluenceClientSpaceFilters:
    """Test the space filtering functionality in ConfluenceClient."""

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

    def test_list_all_spaces_with_type_filter(self, client, mock_confluence_api):
        """Test listing all spaces with space_type filter."""
        mock_response = {
            "results": [
                {
                    "id": "1",
                    "key": "GLOBAL1",
                    "name": "Global Space 1",
                    "type": "global",
                },
                {
                    "id": "2",
                    "key": "GLOBAL2",
                    "name": "Global Space 2",
                    "type": "global",
                },
            ],
            "size": 2,
            "_links": {},
        }

        def mock_get_all_spaces(**kwargs):
            # Simulate API behavior - return filtered results
            if kwargs.get("space_type") == "global":
                return mock_response
            return {"results": [], "size": 0, "_links": {}}

        mock_confluence_api.get_all_spaces.side_effect = mock_get_all_spaces

        spaces = client.list_all_spaces(limit=50, space_type="global", space_status=None)

        assert len(spaces) == 2
        assert all(s.type == SpaceType.GLOBAL for s in spaces)
        assert spaces[0].key == "GLOBAL1"
        assert spaces[1].key == "GLOBAL2"

    def test_list_all_spaces_with_status_filter(self, client, mock_confluence_api):
        """Test listing all spaces with space_status filter."""
        mock_response = {
            "results": [
                {
                    "id": "1",
                    "key": "ARCH1",
                    "name": "Archived Space 1",
                    "type": "global",
                    "status": "archived",
                },
            ],
            "size": 1,
            "_links": {},
        }

        def mock_get_all_spaces(**kwargs):
            # Simulate API behavior - return filtered results
            if kwargs.get("space_status") == "archived":
                return mock_response
            return {"results": [], "size": 0, "_links": {}}

        mock_confluence_api.get_all_spaces.side_effect = mock_get_all_spaces

        spaces = client.list_all_spaces(limit=50, space_type=None, space_status="archived")

        assert len(spaces) == 1
        assert spaces[0].key == "ARCH1"

    def test_list_all_spaces_with_both_filters(self, client, mock_confluence_api):
        """Test listing all spaces with both space_type and space_status filters."""
        mock_response = {
            "results": [
                {
                    "id": "1",
                    "key": "PERS1",
                    "name": "Personal Current Space",
                    "type": "personal",
                    "status": "current",
                },
            ],
            "size": 1,
            "_links": {},
        }

        def mock_get_all_spaces(**kwargs):
            # Simulate API behavior - return filtered results
            if kwargs.get("space_type") == "personal" and kwargs.get("space_status") == "current":
                return mock_response
            return {"results": [], "size": 0, "_links": {}}

        mock_confluence_api.get_all_spaces.side_effect = mock_get_all_spaces

        spaces = client.list_all_spaces(limit=50, space_type="personal", space_status="current")

        assert len(spaces) == 1
        assert spaces[0].type == SpaceType.PERSONAL
        assert spaces[0].key == "PERS1"

    def test_list_spaces_paginated_with_filters(self, client, mock_confluence_api):
        """Test paginated space listing with filters."""
        mock_response = {
            "results": [
                {
                    "id": "1",
                    "key": "GLOBAL1",
                    "name": "Global Space 1",
                    "type": "global",
                },
                {
                    "id": "2",
                    "key": "GLOBAL2",
                    "name": "Global Space 2",
                    "type": "global",
                },
            ],
            "size": 10,  # Total count of global spaces
            "_links": {"next": "/rest/api/space?start=2&limit=2&type=global"},
        }

        mock_confluence_api.get_all_spaces.return_value = mock_response

        spaces, total_count = client.list_spaces_paginated(
            start=0, limit=2, space_type="global", space_status="current"
        )

        assert len(spaces) == 2
        assert total_count == 10
        assert all(s.type == SpaceType.GLOBAL for s in spaces)

        mock_confluence_api.get_all_spaces.assert_called_once_with(
            start=0, limit=2, expand="description.plain", space_type="global", space_status="current"
        )

    def test_list_all_spaces_no_filters_default(self, client, mock_confluence_api):
        """Test that None values for filters work correctly."""
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
            "size": 2,
            "_links": {},
        }

        mock_confluence_api.get_all_spaces.return_value = mock_response

        spaces = client.list_all_spaces(limit=50, space_type=None, space_status=None)

        assert len(spaces) == 2
        # Should return mixed types when no filter is applied
        assert any(s.type == SpaceType.GLOBAL for s in spaces)
        assert any(s.type == SpaceType.PERSONAL for s in spaces)
