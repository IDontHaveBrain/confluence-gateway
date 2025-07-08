import json
from unittest.mock import MagicMock, patch

import pytest
from confluence_gateway.adapters.confluence.models import ConfluenceSpace, SpaceType
from confluence_gateway.cli.main import app
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
    ConfluenceGatewayError,
)
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def mock_spaces():
    """Create mock ConfluenceSpace objects for testing."""
    spaces = []
    for i in range(1, 51):  # Create 50 mock spaces
        space = ConfluenceSpace(
            id=str(i),
            key=f"SPACE{i}",
            name=f"Space {i}",
            title=f"Space {i}",
            type=SpaceType.GLOBAL if i % 2 == 0 else SpaceType.PERSONAL,
        )
        if i % 3 == 0:
            space.description_text = f"Description for space {i}"
        spaces.append(space)
    return spaces


class TestSpacesListCommand:
    """Test the spaces list command with pagination."""

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_default_pagination(self, mock_get_client, mock_spaces):
        """Test listing spaces with default pagination."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Return first 25 spaces
        mock_client.list_spaces_paginated.return_value = (mock_spaces[:25], 50)

        result = runner.invoke(app, ["spaces", "list"])

        assert result.exit_code == 0
        assert "Page 1/2" in result.output
        assert "50 total" in result.output
        assert "SPACE1" in result.output
        assert "SPACE25" in result.output
        assert "SPACE26" not in result.output

        # Verify the client was called with correct parameters
        mock_client.list_spaces_paginated.assert_called_once_with(
            start=0, limit=25, space_type=None, space_status="current"
        )

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_specific_page(self, mock_get_client, mock_spaces):
        """Test listing spaces on a specific page."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Return second page of spaces
        mock_client.list_spaces_paginated.return_value = (mock_spaces[25:50], 50)

        result = runner.invoke(app, ["spaces", "list", "--page", "2"])

        assert result.exit_code == 0
        assert "Page 2/2" in result.output
        assert "50 total" in result.output
        assert "SPACE26" in result.output
        assert "SPACE50" in result.output
        assert "SPACE1" not in result.output

        mock_client.list_spaces_paginated.assert_called_once_with(
            start=25, limit=25, space_type=None, space_status="current"
        )

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_custom_page_size(self, mock_get_client, mock_spaces):
        """Test listing spaces with custom page size."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Return 10 spaces
        mock_client.list_spaces_paginated.return_value = (mock_spaces[:10], 50)

        result = runner.invoke(app, ["spaces", "list", "--page-size", "10"])

        assert result.exit_code == 0
        assert "Page 1/5" in result.output
        assert "50 total" in result.output

        mock_client.list_spaces_paginated.assert_called_once_with(
            start=0, limit=10, space_type=None, space_status="current"
        )

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_all_flag(self, mock_get_client, mock_spaces):
        """Test listing all spaces with --all flag."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Return all spaces
        mock_client.list_all_spaces.return_value = mock_spaces

        result = runner.invoke(app, ["spaces", "list", "--all"])

        assert result.exit_code == 0
        assert "50 total" in result.output
        assert "Page" not in result.output  # No pagination info
        assert "SPACE1" in result.output
        assert "SPACE50" in result.output

        mock_client.list_all_spaces.assert_called_once_with(
            space_type=None, space_status="current"
        )
        mock_client.list_spaces_paginated.assert_not_called()

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_json_format_with_pagination(
        self, mock_get_client, mock_spaces
    ):
        """Test JSON output format with pagination info."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.list_spaces_paginated.return_value = (mock_spaces[:25], 50)

        result = runner.invoke(app, ["spaces", "list", "--format", "json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        assert "spaces" in data
        assert "pagination" in data
        assert len(data["spaces"]) == 25
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["page_size"] == 25
        assert data["pagination"]["total_pages"] == 2
        assert data["pagination"]["total_count"] == 50

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_empty_page(self, mock_get_client):
        """Test listing spaces on an empty page."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.list_spaces_paginated.return_value = ([], 10)

        result = runner.invoke(app, ["spaces", "list", "--page", "5"])

        assert result.exit_code == 0
        assert "No spaces found on page 5" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_no_spaces(self, mock_get_client):
        """Test listing spaces when there are no spaces."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.list_spaces_paginated.return_value = ([], 0)

        result = runner.invoke(app, ["spaces", "list"])

        assert result.exit_code == 0
        assert "No spaces found or no access to any spaces" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_error_handling(self, mock_get_client):
        """Test error handling in list command."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.list_spaces_paginated.side_effect = ConfluenceAPIError(
            error_message="API Error"
        )

        result = runner.invoke(app, ["spaces", "list"])

        assert result.exit_code == 1
        assert "❌ Confluence API error during listing spaces" in result.output

    def test_list_spaces_invalid_format(self):
        """Test invalid format option."""
        result = runner.invoke(app, ["spaces", "list", "--format", "invalid"])

        assert result.exit_code == 1
        assert "Error: format must be 'table', 'json', or 'csv'" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_with_descriptions(self, mock_get_client, mock_spaces):
        """Test that descriptions are included in JSON output when present."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Get spaces that have descriptions (every 3rd space)
        spaces_with_desc = [
            s for s in mock_spaces[:25] if s.description_text is not None
        ]
        mock_client.list_spaces_paginated.return_value = (mock_spaces[:25], 50)

        result = runner.invoke(app, ["spaces", "list", "--format", "json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Check that some spaces have descriptions
        spaces_with_desc_in_output = [s for s in data["spaces"] if "description" in s]
        assert len(spaces_with_desc_in_output) > 0
        assert len(spaces_with_desc_in_output) == len(spaces_with_desc)

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_filter_by_type_global(self, mock_get_client, mock_spaces):
        """Test filtering spaces by type=global."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock the API to accept space_type parameter
        mock_client.list_spaces_paginated.return_value = (
            [s for s in mock_spaces[:25] if s.type == SpaceType.GLOBAL],
            25,
        )

        result = runner.invoke(app, ["spaces", "list", "--type", "global"])

        assert result.exit_code == 0
        # Should only show global spaces (even numbered ones)
        assert "SPACE2" in result.output
        assert "SPACE4" in result.output
        # Check that personal spaces are not in the table (watch out for SPACE10, SPACE11, etc)
        lines = result.output.split("\n")
        # Extract only the space keys from the table rows
        space_keys = []
        for line in lines:
            if "│" in line and "SPACE" in line:
                # Extract the key from the table row
                parts = line.split("│")
                if len(parts) > 1:
                    key = parts[1].strip()
                    if key.startswith("SPACE"):
                        space_keys.append(key)
        # Now check that SPACE1 and SPACE3 are not in the list
        assert "SPACE1" not in space_keys
        assert "SPACE3" not in space_keys

        mock_client.list_spaces_paginated.assert_called_once_with(
            start=0, limit=25, space_type="global", space_status="current"
        )

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_filter_by_type_personal(self, mock_get_client, mock_spaces):
        """Test filtering spaces by type=personal."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock the API to accept space_type parameter
        mock_client.list_spaces_paginated.return_value = (
            [s for s in mock_spaces[:25] if s.type == SpaceType.PERSONAL],
            25,
        )

        result = runner.invoke(app, ["spaces", "list", "--type", "personal"])

        assert result.exit_code == 0
        # Should only show personal spaces (odd numbered ones)
        assert "SPACE1" in result.output
        assert "SPACE3" in result.output
        # Check that global spaces are not in the table
        lines = result.output.split("\n")
        # Extract only the space keys from the table rows
        space_keys = []
        for line in lines:
            if "│" in line and "SPACE" in line:
                # Extract the key from the table row
                parts = line.split("│")
                if len(parts) > 1:
                    key = parts[1].strip()
                    if key.startswith("SPACE"):
                        space_keys.append(key)
        # Now check that SPACE2 and SPACE4 are not in the list
        assert "SPACE2" not in space_keys
        assert "SPACE4" not in space_keys

        mock_client.list_spaces_paginated.assert_called_once_with(
            start=0, limit=25, space_type="personal", space_status="current"
        )

    def test_list_spaces_invalid_type(self):
        """Test invalid type filter."""
        result = runner.invoke(app, ["spaces", "list", "--type", "invalid"])

        assert result.exit_code == 1
        assert "Error: type must be 'personal', 'global', or 'all'" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_with_search(self, mock_get_client, mock_spaces):
        """Test searching spaces by name or key."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Return all spaces for client-side filtering
        mock_client.list_all_spaces.return_value = mock_spaces

        result = runner.invoke(app, ["spaces", "list", "--search", "Space 1"])

        assert result.exit_code == 0
        # Should show spaces with "Space 1" in name (Space 1, Space 10-19)
        assert "SPACE1" in result.output
        assert "SPACE10" in result.output
        # Extract space keys to verify SPACE2 is not included
        lines = result.output.split("\n")
        space_keys = []
        for line in lines:
            if "│" in line and "SPACE" in line:
                parts = line.split("│")
                if len(parts) > 1:
                    key = parts[1].strip()
                    if key.startswith("SPACE"):
                        space_keys.append(key)
        assert "SPACE2" not in space_keys

        # Should have called list_all_spaces since we have client-side filtering
        mock_client.list_all_spaces.assert_called_once_with(
            space_type=None, space_status="current"
        )

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_with_key_prefix(self, mock_get_client, mock_spaces):
        """Test filtering spaces by key prefix."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create some spaces with different key prefixes
        special_spaces = [
            ConfluenceSpace(
                id="100",
                key="TEAM1",
                name="Team 1 Space",
                title="Team 1",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="101",
                key="TEAM2",
                name="Team 2 Space",
                title="Team 2",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="102",
                key="DEV1",
                name="Dev 1 Space",
                title="Dev 1",
                type=SpaceType.GLOBAL,
            ),
        ]

        mock_client.list_all_spaces.return_value = special_spaces

        result = runner.invoke(app, ["spaces", "list", "--key-prefix", "TEAM"])

        assert result.exit_code == 0
        assert "TEAM1" in result.output
        assert "TEAM2" in result.output
        assert "DEV1" not in result.output

        mock_client.list_all_spaces.assert_called_once_with(
            space_type=None, space_status="current"
        )

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_combined_filters(self, mock_get_client, mock_spaces):
        """Test combining multiple filters."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create test data
        test_spaces = [
            ConfluenceSpace(
                id="1",
                key="DEV1",
                name="Development Space 1",
                title="Dev 1",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="2",
                key="DEV2",
                name="Development Space 2",
                title="Dev 2",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="3",
                key="PROD1",
                name="Production Space",
                title="Prod",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="4",
                key="DEV3",
                name="Personal Dev",
                title="Personal",
                type=SpaceType.PERSONAL,
            ),
        ]

        # Mock list_all_spaces to return only global spaces when space_type="global"
        def mock_list_all_spaces(space_type=None, space_status=None):
            if space_type == "global":
                return [s for s in test_spaces if s.type == SpaceType.GLOBAL]
            return test_spaces

        mock_client.list_all_spaces.side_effect = mock_list_all_spaces

        result = runner.invoke(
            app, ["spaces", "list", "--type", "global", "--key-prefix", "DEV"]
        )

        assert result.exit_code == 0
        # Should only show global spaces with DEV prefix
        assert "DEV1" in result.output
        assert "DEV2" in result.output
        assert "PROD1" not in result.output  # Wrong prefix
        assert "DEV3" not in result.output  # Personal type

        mock_client.list_all_spaces.assert_called_once_with(
            space_type="global", space_status="current"
        )

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_search_case_insensitive(self, mock_get_client):
        """Test that search is case-insensitive."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        test_spaces = [
            ConfluenceSpace(
                id="1",
                key="TEST",
                name="Test Space",
                title="Test",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="2",
                key="test",
                name="test space",
                title="test",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="3",
                key="TeSt",
                name="TeSt SpAcE",
                title="TeSt",
                type=SpaceType.GLOBAL,
            ),
        ]

        mock_client.list_all_spaces.return_value = test_spaces

        result = runner.invoke(app, ["spaces", "list", "--search", "TEST"])

        assert result.exit_code == 0
        # All spaces should match regardless of case
        assert "TEST" in result.output
        assert "test" in result.output
        assert "TeSt" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_sort_by_name(self, mock_get_client):
        """Test sorting spaces by name."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create test spaces with different names
        test_spaces = [
            ConfluenceSpace(
                id="3",
                key="ZEBRA",
                name="Zebra Space",
                title="Zebra",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="1",
                key="ALPHA",
                name="Alpha Space",
                title="Alpha",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="2",
                key="BETA",
                name="Beta Space",
                title="Beta",
                type=SpaceType.GLOBAL,
            ),
        ]

        mock_client.list_all_spaces.return_value = test_spaces

        result = runner.invoke(app, ["spaces", "list", "--sort", "name"])

        assert result.exit_code == 0
        # Check order in output - Alpha should come before Beta before Zebra
        alpha_pos = result.output.find("ALPHA")
        beta_pos = result.output.find("BETA")
        zebra_pos = result.output.find("ZEBRA")
        assert alpha_pos < beta_pos < zebra_pos

        mock_client.list_all_spaces.assert_called_once()

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_sort_by_key(self, mock_get_client):
        """Test sorting spaces by key."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create test spaces with different keys
        test_spaces = [
            ConfluenceSpace(
                id="3",
                key="CCC",
                name="Third Space",
                title="Third",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="1",
                key="AAA",
                name="First Space",
                title="First",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="2",
                key="BBB",
                name="Second Space",
                title="Second",
                type=SpaceType.GLOBAL,
            ),
        ]

        mock_client.list_all_spaces.return_value = test_spaces

        result = runner.invoke(app, ["spaces", "list", "--sort", "key"])

        assert result.exit_code == 0
        # Check order in output
        aaa_pos = result.output.find("AAA")
        bbb_pos = result.output.find("BBB")
        ccc_pos = result.output.find("CCC")
        assert aaa_pos < bbb_pos < ccc_pos

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_sort_by_type(self, mock_get_client):
        """Test sorting spaces by type."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create test spaces with different types
        test_spaces = [
            ConfluenceSpace(
                id="1",
                key="PERS1",
                name="Personal Space 1",
                title="Personal 1",
                type=SpaceType.PERSONAL,
            ),
            ConfluenceSpace(
                id="2",
                key="GLOB1",
                name="Global Space 1",
                title="Global 1",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="3",
                key="PERS2",
                name="Personal Space 2",
                title="Personal 2",
                type=SpaceType.PERSONAL,
            ),
            ConfluenceSpace(
                id="4",
                key="GLOB2",
                name="Global Space 2",
                title="Global 2",
                type=SpaceType.GLOBAL,
            ),
        ]

        mock_client.list_all_spaces.return_value = test_spaces

        result = runner.invoke(app, ["spaces", "list", "--sort", "type"])

        assert result.exit_code == 0
        # Extract table rows to check order
        lines = result.output.split("\n")
        space_keys = []
        for line in lines:
            if "│" in line and ("GLOB" in line or "PERS" in line):
                parts = line.split("│")
                if len(parts) > 1:
                    key = parts[1].strip()
                    if key.startswith(("GLOB", "PERS")):
                        space_keys.append(key)

        # Global spaces should come before personal spaces (alphabetically)
        assert space_keys.index("GLOB1") < space_keys.index("PERS1")
        assert space_keys.index("GLOB2") < space_keys.index("PERS2")

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_sort_by_id(self, mock_get_client):
        """Test sorting spaces by ID."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create test spaces with different IDs
        test_spaces = [
            ConfluenceSpace(
                id="300",
                key="THIRD",
                name="Third Space",
                title="Third",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="100",
                key="FIRST",
                name="First Space",
                title="First",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="200",
                key="SECOND",
                name="Second Space",
                title="Second",
                type=SpaceType.GLOBAL,
            ),
        ]

        mock_client.list_all_spaces.return_value = test_spaces

        result = runner.invoke(app, ["spaces", "list", "--sort", "id"])

        assert result.exit_code == 0
        # Check order in output
        first_pos = result.output.find("FIRST")
        second_pos = result.output.find("SECOND")
        third_pos = result.output.find("THIRD")
        assert first_pos < second_pos < third_pos

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_sort_reverse(self, mock_get_client):
        """Test reverse sorting."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create test spaces
        test_spaces = [
            ConfluenceSpace(
                id="1",
                key="AAA",
                name="AAA Space",
                title="AAA",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="2",
                key="BBB",
                name="BBB Space",
                title="BBB",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="3",
                key="CCC",
                name="CCC Space",
                title="CCC",
                type=SpaceType.GLOBAL,
            ),
        ]

        mock_client.list_all_spaces.return_value = test_spaces

        result = runner.invoke(app, ["spaces", "list", "--sort", "key", "--reverse"])

        assert result.exit_code == 0
        # Check reverse order - CCC should come before BBB before AAA
        ccc_pos = result.output.find("CCC")
        bbb_pos = result.output.find("BBB")
        aaa_pos = result.output.find("AAA")
        assert ccc_pos < bbb_pos < aaa_pos

    def test_list_spaces_invalid_sort(self):
        """Test invalid sort option."""
        result = runner.invoke(app, ["spaces", "list", "--sort", "invalid"])

        assert result.exit_code == 1
        assert "Error: sort must be 'name', 'key', 'type', or 'id'" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_sort_with_pagination(self, mock_get_client):
        """Test sorting with pagination."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create 10 test spaces
        test_spaces = []
        for i in range(10, 0, -1):  # Create in reverse order
            test_spaces.append(
                ConfluenceSpace(
                    id=str(i),
                    key=f"SPACE{i:02d}",
                    name=f"Space {i:02d}",
                    title=f"Space {i:02d}",
                    type=SpaceType.GLOBAL,
                )
            )

        mock_client.list_all_spaces.return_value = test_spaces

        # Get first page with sorting
        result = runner.invoke(
            app, ["spaces", "list", "--sort", "key", "--page-size", "5"]
        )

        assert result.exit_code == 0
        assert "Page 1/2" in result.output
        # First page should have SPACE01 through SPACE05
        assert "SPACE01" in result.output
        assert "SPACE05" in result.output
        assert "SPACE06" not in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_sort_with_filters(self, mock_get_client):
        """Test sorting combined with filters."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create test spaces
        test_spaces = [
            ConfluenceSpace(
                id="1",
                key="DEVZ",
                name="Z Development",
                title="Z Dev",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="2",
                key="DEVA",
                name="A Development",
                title="A Dev",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="3",
                key="PROD",
                name="Production",
                title="Prod",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="4",
                key="DEVB",
                name="B Development",
                title="B Dev",
                type=SpaceType.PERSONAL,
            ),
        ]

        # Mock list_all_spaces to return only global spaces when space_type="global"
        def mock_list_all_spaces(**kwargs):
            space_type = kwargs.get('space_type')
            if space_type == "global":
                return [s for s in test_spaces if s.type == SpaceType.GLOBAL]
            return test_spaces

        mock_client.list_all_spaces.side_effect = mock_list_all_spaces

        result = runner.invoke(
            app,
            ["spaces", "list", "--type", "global", "--key-prefix", "DEV", "--sort", "key"],
        )

        assert result.exit_code == 0
        # Should only show global DEV spaces sorted by key
        deva_pos = result.output.find("DEVA")
        devz_pos = result.output.find("DEVZ")
        assert deva_pos > -1, "DEVA not found in output"
        assert devz_pos > -1, "DEVZ not found in output"
        assert deva_pos < devz_pos  # DEVA should come before DEVZ when sorted by key
        assert "PROD" not in result.output  # Wrong prefix
        assert "DEVB" not in result.output  # Personal type

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_sort_json_format(self, mock_get_client):
        """Test sorting with JSON output format."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create test spaces
        test_spaces = [
            ConfluenceSpace(
                id="3",
                key="CCC",
                name="CCC Space",
                title="CCC",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="1",
                key="AAA",
                name="AAA Space",
                title="AAA",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="2",
                key="BBB",
                name="BBB Space",
                title="BBB",
                type=SpaceType.GLOBAL,
            ),
        ]

        mock_client.list_all_spaces.return_value = test_spaces

        result = runner.invoke(
            app, ["spaces", "list", "--sort", "key", "--format", "json"]
        )

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Check that spaces are sorted by key
        assert len(data["spaces"]) == 3
        assert data["spaces"][0]["key"] == "AAA"
        assert data["spaces"][1]["key"] == "BBB"
        assert data["spaces"][2]["key"] == "CCC"

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_sort_handles_none_values(self, mock_get_client):
        """Test sorting handles None values gracefully."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create test spaces with some None types
        test_spaces = [
            ConfluenceSpace(
                id="1",
                key="AAA",
                name="AAA Space",
                title="AAA",
                type=SpaceType.GLOBAL,
            ),
            ConfluenceSpace(
                id="2",
                key="BBB",
                name="BBB Space",
                title="BBB",
                type=None,  # No type
            ),
            ConfluenceSpace(
                id="3",
                key="CCC",
                name="CCC Space",
                title="CCC",
                type=SpaceType.PERSONAL,
            ),
        ]

        mock_client.list_all_spaces.return_value = test_spaces

        # Should not crash when sorting by type with None values
        result = runner.invoke(app, ["spaces", "list", "--sort", "type"])

        assert result.exit_code == 0
        assert "AAA" in result.output
        assert "BBB" in result.output
        assert "CCC" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_csv_format(self, mock_get_client, mock_spaces):
        """Test CSV output format."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.list_spaces_paginated.return_value = (mock_spaces[:5], 50)

        result = runner.invoke(app, ["spaces", "list", "--format", "csv"])

        assert result.exit_code == 0
        # Check CSV headers
        assert "Key,Name,Type,ID,Description" in result.output
        # Check some data rows
        assert "SPACE1,Space 1,personal,1," in result.output
        assert "SPACE2,Space 2,global,2," in result.output
        assert "SPACE3,Space 3,personal,3,Description for space 3" in result.output
        # Check pagination comment
        assert "# Page 1 of 2, 50 total spaces" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_csv_format_all(self, mock_get_client, mock_spaces):
        """Test CSV output format with --all flag."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.list_all_spaces.return_value = mock_spaces[:10]

        result = runner.invoke(app, ["spaces", "list", "--format", "csv", "--all"])

        assert result.exit_code == 0
        # Check CSV headers
        assert "Key,Name,Type,ID,Description" in result.output
        # Should have 10 data rows
        lines = result.output.strip().split('\n')
        csv_lines = [line for line in lines if line and not line.startswith('#')]
        assert len(csv_lines) == 11  # 1 header + 10 data rows
        # No pagination comment when using --all
        assert "# Page" not in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_csv_format_with_filter(self, mock_get_client, mock_spaces):
        """Test CSV output format with filtering."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # For type filtering without other client-side filters, list_spaces_paginated is used
        global_spaces = [s for s in mock_spaces[:20] if s.type == SpaceType.GLOBAL]
        mock_client.list_spaces_paginated.return_value = (global_spaces[:10], len(global_spaces))

        result = runner.invoke(app, ["spaces", "list", "--format", "csv", "--type", "global"])

        assert result.exit_code == 0
        # Check CSV headers
        assert "Key,Name,Type,ID,Description" in result.output
        # Should only have global spaces (even numbered)
        assert "SPACE2,Space 2,global,2," in result.output
        assert "SPACE4,Space 4,global,4," in result.output
        # Personal spaces should not be in output
        assert "SPACE1,Space 1,personal,1," not in result.output
        assert "SPACE3,Space 3,personal,3," not in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_csv_format_with_special_characters(self, mock_get_client):
        """Test CSV output handles special characters properly."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create spaces with special characters
        special_spaces = [
            ConfluenceSpace(
                id="1",
                key="TEST",
                name='Space with "quotes"',
                title='Space with "quotes"',
                type=SpaceType.GLOBAL,
                description_text='Description with, comma and "quotes"'
            ),
            ConfluenceSpace(
                id="2",
                key="TEST2",
                name="Space with\nnewline",
                title="Space with\nnewline",
                type=SpaceType.GLOBAL,
            ),
        ]

        mock_client.list_spaces_paginated.return_value = (special_spaces, 2)

        result = runner.invoke(app, ["spaces", "list", "--format", "csv"])

        assert result.exit_code == 0
        # Check that special characters are properly escaped
        assert '"Space with ""quotes"""' in result.output
        assert '"Description with, comma and ""quotes"""' in result.output
        assert '"Space with\nnewline"' in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_no_truncate_option(self, mock_get_client):
        """Test --no-truncate option for table format."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create space with very long name
        long_name_space = ConfluenceSpace(
            id="1",
            key="LONG",
            name="This is a very long space name that would normally be truncated in the table output but should show fully with --no-truncate option",
            title="Long title",
            type=SpaceType.GLOBAL,
        )

        mock_client.list_spaces_paginated.return_value = ([long_name_space], 1)

        result = runner.invoke(app, ["spaces", "list", "--no-truncate"])

        assert result.exit_code == 0
        # The full name should be visible (Rich will wrap it instead of truncating)
        assert "This is a very long space name" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_csv_no_progress_indicator(self, mock_get_client, mock_spaces):
        """Test that CSV format does not show progress indicators."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.list_all_spaces.return_value = mock_spaces[:5]

        result = runner.invoke(app, ["spaces", "list", "--format", "csv", "--all"])

        assert result.exit_code == 0
        # Should not contain progress indicator text
        assert "Fetching" not in result.output
        assert "✓" not in result.output
        # Should contain CSV data
        assert "Key,Name,Type,ID,Description" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_csv_with_sort(self, mock_get_client):
        """Test CSV output with sorting."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create unsorted spaces
        test_spaces = [
            ConfluenceSpace(id="3", key="CCC", name="C Space", title="C", type=SpaceType.GLOBAL),
            ConfluenceSpace(id="1", key="AAA", name="A Space", title="A", type=SpaceType.GLOBAL),
            ConfluenceSpace(id="2", key="BBB", name="B Space", title="B", type=SpaceType.GLOBAL),
        ]

        mock_client.list_all_spaces.return_value = test_spaces

        result = runner.invoke(app, ["spaces", "list", "--format", "csv", "--sort", "key"])

        assert result.exit_code == 0
        lines = result.output.strip().split('\n')
        # Find data rows (skip header and comments)
        data_lines = [line for line in lines if line and not line.startswith('#') and not line.startswith('Key,')]

        # Check order - AAA should come first, CCC last
        assert "AAA" in data_lines[0]
        assert "BBB" in data_lines[1]
        assert "CCC" in data_lines[2]

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_list_spaces_table_auto_width(self, mock_get_client):
        """Test that table columns auto-adjust width based on content."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create spaces with varying key lengths
        test_spaces = [
            ConfluenceSpace(id="1", key="A", name="Short", title="Short", type=SpaceType.GLOBAL),
            ConfluenceSpace(id="2", key="VERYLONGKEY", name="Very Long Space Name Here", title="Long", type=SpaceType.GLOBAL),
        ]

        mock_client.list_spaces_paginated.return_value = (test_spaces, 2)

        result = runner.invoke(app, ["spaces", "list"])

        assert result.exit_code == 0
        # The table should accommodate both short and long content
        assert "A" in result.output
        assert "VERYLONGKEY" in result.output
        assert "Very Long Space Name" in result.output


class TestSpaceInfoCommand:
    """Test the spaces info command."""

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_space_info_success(self, mock_get_client):
        """Test successful space info retrieval."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create a mock space with all attributes
        mock_space = ConfluenceSpace(
            id="123456",
            key="DEV",
            name="Development Team",
            title="Development Team",
            type=SpaceType.GLOBAL,
            description_text="Main development team documentation space",
            created_at="2023-01-15",
            updated_at="2025-01-07"
        )

        mock_client.get_space.return_value = mock_space

        result = runner.invoke(app, ["spaces", "info", "DEV"])

        assert result.exit_code == 0
        assert "Space Information" in result.output
        assert "Key: DEV" in result.output
        assert "Name: Development Team" in result.output
        assert "ID: 123456" in result.output
        assert "Type: global" in result.output
        assert "Description: Main development team documentation space" in result.output
        assert "Created: 2023-01-15" in result.output
        assert "Updated: 2025-01-07" in result.output

        mock_client.get_space.assert_called_once_with("DEV")

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_space_info_minimal_data(self, mock_get_client):
        """Test space info with minimal data (no description, dates)."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create a mock space with minimal attributes
        mock_space = ConfluenceSpace(
            id="789",
            key="MIN",
            name="Minimal Space",
            title="Minimal Space",
            type=SpaceType.PERSONAL
        )

        mock_client.get_space.return_value = mock_space

        result = runner.invoke(app, ["spaces", "info", "MIN"])

        assert result.exit_code == 0
        assert "Space Information" in result.output
        assert "Key: MIN" in result.output
        assert "Name: Minimal Space" in result.output
        assert "ID: 789" in result.output
        assert "Type: personal" in result.output
        # These should not appear since they're not set
        assert "Description:" not in result.output
        assert "Created:" not in result.output
        assert "Updated:" not in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_space_info_error(self, mock_get_client):
        """Test error handling in space info command."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.get_space.side_effect = ConfluenceAPIError(
            error_message="Space not found"
        )

        result = runner.invoke(app, ["spaces", "info", "NOTFOUND"])

        assert result.exit_code == 1
        assert "❌ Confluence API error during getting information for space 'NOTFOUND'" in result.output


class TestSpacesErrorHandling:
    """Test error handling improvements in spaces commands."""

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_authentication_error_handling(self, mock_get_client):
        """Test authentication error provides helpful guidance."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.list_spaces_paginated.side_effect = ConfluenceAuthenticationError(
            "Authentication failed"
        )

        result = runner.invoke(app, ["spaces", "list"])

        assert result.exit_code == 1
        assert "❌ Authentication failed during listing spaces" in result.output
        assert "💡 Check your credentials:" in result.output
        assert "Verify your Confluence URL is correct" in result.output
        assert "Check that your API token is valid" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_connection_error_handling(self, mock_get_client):
        """Test network connection error provides helpful guidance."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.list_spaces_paginated.side_effect = ConfluenceConnectionError(
            "Network error"
        )

        result = runner.invoke(app, ["spaces", "list"])

        assert result.exit_code == 1
        assert "❌ Network connection failed during listing spaces" in result.output
        assert "💡 Troubleshooting steps:" in result.output
        assert "Check your internet connection" in result.output
        assert "Verify the Confluence URL is reachable" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_api_error_403_handling(self, mock_get_client):
        """Test API 403 error provides helpful guidance."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        api_error = ConfluenceAPIError(status_code=403, error_message="Forbidden")
        mock_client.list_spaces_paginated.side_effect = api_error

        result = runner.invoke(app, ["spaces", "list"])

        assert result.exit_code == 1
        assert "❌ Confluence API error during listing spaces" in result.output
        assert "💡 Permission denied:" in result.output
        assert "You may not have permission to view spaces" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_api_error_429_handling(self, mock_get_client):
        """Test API 429 rate limit error provides helpful guidance."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        api_error = ConfluenceAPIError(status_code=429, error_message="Rate limited")
        mock_client.list_spaces_paginated.side_effect = api_error

        result = runner.invoke(app, ["spaces", "list"])

        assert result.exit_code == 1
        assert "❌ Confluence API error during listing spaces" in result.output
        assert "💡 Rate limit exceeded:" in result.output
        assert "Too many requests sent to Confluence" in result.output
        assert "Wait a few minutes before trying again" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_verbose_option_shows_details(self, mock_get_client):
        """Test --verbose option shows detailed error information."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.list_spaces_paginated.side_effect = ConfluenceAuthenticationError(
            "Invalid API token"
        )

        result = runner.invoke(app, ["spaces", "list", "--verbose"])

        assert result.exit_code == 1
        assert "❌ Authentication failed during listing spaces" in result.output
        assert "Technical details: Invalid API token" in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    @patch("confluence_gateway.cli.spaces_commands.time.sleep")  # Mock sleep to speed up test
    def test_retry_logic_on_connection_error(self, mock_sleep, mock_get_client):
        """Test retry logic works for connection errors."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First two calls fail, third succeeds
        mock_client.list_spaces_paginated.side_effect = [
            ConfluenceConnectionError("Network error"),
            ConfluenceConnectionError("Network error"),
            ([], 0)  # Success on third try
        ]

        result = runner.invoke(app, ["spaces", "list"])

        assert result.exit_code == 0
        assert "Network error. Retrying in" in result.output
        assert mock_client.list_spaces_paginated.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between retries

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_no_retry_on_auth_error(self, mock_get_client):
        """Test that authentication errors are not retried."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.list_spaces_paginated.side_effect = ConfluenceAuthenticationError(
            "Auth failed"
        )

        result = runner.invoke(app, ["spaces", "list"])

        assert result.exit_code == 1
        assert mock_client.list_spaces_paginated.call_count == 1  # No retries
        assert "Retrying" not in result.output

    @patch("confluence_gateway.cli.spaces_commands._get_confluence_client")
    def test_space_info_verbose_error(self, mock_get_client):
        """Test space info command with verbose error handling."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        api_error = ConfluenceAPIError(status_code=404, error_message="Space not found")
        mock_client.get_space.side_effect = api_error

        result = runner.invoke(app, ["spaces", "info", "NOTFOUND", "--verbose"])

        assert result.exit_code == 1
        assert "❌ Confluence API error during getting information for space 'NOTFOUND'" in result.output
        assert "💡 Resource not found:" in result.output
        assert "API error details: Space not found" in result.output
