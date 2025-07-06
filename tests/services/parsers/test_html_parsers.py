import io
from unittest.mock import MagicMock, Mock, patch

import pytest
from confluence_gateway.services.parsers.factory import (
    ParserNotAvailableError,
    get_parser,
)
from confluence_gateway.services.parsers.html_parsers import (
    MarkitdownHtmlParser,
    UnstructuredHtmlParser,
)


class TestMarkitdownHtmlParser:
    @pytest.fixture
    def parser(self):
        """Create a MarkitdownHtmlParser instance."""
        return MarkitdownHtmlParser()

    @pytest.fixture
    def mock_markitdown(self):
        """Mock the MarkItDown class."""
        with patch(
            "confluence_gateway.services.parsers.html_parsers.MarkItDownClass"
        ) as mock:
            mock_converter = Mock()
            mock_result = Mock()
            mock_result.markdown = "This is extracted text"
            mock_converter.convert.return_value = mock_result
            mock.return_value = mock_converter
            yield mock

    def test_markitdown_parser_basic(self, parser, mock_markitdown):
        """Test basic HTML parsing with markitdown."""
        html_content = "<html><body><h1>Title</h1><p>Content</p></body></html>"

        result = parser.parse(html_content)

        assert result == "This is extracted text"
        assert mock_markitdown.called
        assert mock_markitdown.return_value.convert.called

    def test_markitdown_parser_complex_html(self, parser, mock_markitdown):
        """Test parsing complex HTML with nested elements."""
        complex_html = """
        <html>
            <body>
                <div class="main">
                    <h1>Main Title</h1>
                    <article>
                        <h2>Subtitle</h2>
                        <p>Paragraph with <strong>bold</strong> and <em>italic</em> text.</p>
                        <ul>
                            <li>Item 1</li>
                            <li>Item 2</li>
                        </ul>
                    </article>
                </div>
            </body>
        </html>
        """

        mock_markitdown.return_value.convert.return_value.markdown = (
            "Main Title Subtitle Paragraph with bold and italic text. Item 1 Item 2"
        )

        result = parser.parse(complex_html)

        assert (
            result
            == "Main Title Subtitle Paragraph with bold and italic text. Item 1 Item 2"
        )

    def test_markitdown_parser_malformed_html(self, parser, mock_markitdown):
        """Test parsing malformed HTML."""
        malformed_html = "<p>Unclosed paragraph <div>Nested without closing"

        result = parser.parse(malformed_html)

        assert result == "This is extracted text"
        assert mock_markitdown.return_value.convert.called

    def test_markitdown_parser_empty_content(self, parser):
        """Test parsing empty content."""
        result = parser.parse("")
        assert result is None

        result = parser.parse(None)
        assert result is None

    def test_markitdown_parser_non_string_content(self, parser):
        """Test parsing non-string content."""
        result = parser.parse(123)
        assert result is None

        result = parser.parse([1, 2, 3])
        assert result is None

    def test_markitdown_parser_unicode_content(self, parser, mock_markitdown):
        """Test parsing HTML with unicode characters."""
        unicode_html = """
        <html>
            <body>
                <h1>Unicode Test: 你好世界</h1>
                <p>Emojis: 🌍 🚀 ✨</p>
                <p>Special chars: é à ñ ü</p>
            </body>
        </html>
        """

        mock_markitdown.return_value.convert.return_value.markdown = (
            "Unicode Test: 你好世界 Emojis: 🌍 🚀 ✨ Special chars: é à ñ ü"
        )

        result = parser.parse(unicode_html)

        assert "你好世界" in result
        assert "🌍" in result
        assert "é à ñ ü" in result

    def test_markitdown_parser_whitespace_handling(self, parser, mock_markitdown):
        """Test that parser properly handles whitespace."""
        html_with_whitespace = """
        <p>Text    with     multiple     spaces</p>
        <p>Text
        with
        newlines</p>
        """

        mock_markitdown.return_value.convert.return_value.markdown = (
            "Text    with     multiple     spaces\n\nText\nwith\nnewlines"
        )

        result = parser.parse(html_with_whitespace)

        # Result should have normalized whitespace
        assert result == "Text with multiple spaces Text with newlines"

    def test_markitdown_parser_no_library(self, parser):
        """Test behavior when markitdown library is not available."""
        with patch(
            "confluence_gateway.services.parsers.html_parsers.MarkItDownClass", None
        ):
            result = parser.parse("<p>Test</p>")
            assert result is None

    def test_markitdown_parser_exception_handling(self, parser, mock_markitdown):
        """Test exception handling during parsing."""
        mock_markitdown.return_value.convert.side_effect = Exception("Parsing error")

        result = parser.parse("<p>Test</p>")

        assert result is None

    @patch("confluence_gateway.services.parsers.html_parsers.logger")
    def test_markitdown_parser_logging(self, mock_logger, parser, mock_markitdown):
        """Test proper logging in markitdown parser."""
        # Successful parsing
        parser.parse("<p>Test</p>")
        mock_logger.debug.assert_called_with(
            "Successfully extracted text using markitdown"
        )

        # Error during parsing
        mock_markitdown.return_value.convert.side_effect = Exception("Test error")
        parser.parse("<p>Test</p>")
        mock_logger.error.assert_called()


class TestUnstructuredHtmlParser:
    @pytest.fixture
    def parser(self):
        """Create an UnstructuredHtmlParser instance."""
        return UnstructuredHtmlParser()

    @pytest.fixture
    def mock_partition_html(self):
        """Mock the partition_html function."""
        with patch(
            "confluence_gateway.services.parsers.html_parsers.partition_html"
        ) as mock:
            yield mock

    @pytest.fixture
    def mock_clean_whitespace(self):
        """Mock the clean_extra_whitespace function."""
        with patch(
            "confluence_gateway.services.parsers.html_parsers.clean_extra_whitespace"
        ) as mock:
            mock.side_effect = lambda x: x  # Don't modify the text
            yield mock

    def test_unstructured_parser_basic(
        self, parser, mock_partition_html, mock_clean_whitespace
    ):
        """Test basic HTML parsing with unstructured."""
        html_content = "<html><body><h1>Title</h1><p>Content</p></body></html>"

        # Mock elements returned by partition_html
        mock_elements = [Mock(text="Title"), Mock(text="Content")]
        mock_partition_html.return_value = mock_elements

        result = parser.parse(html_content)

        assert result == "Title\n\nContent"
        mock_partition_html.assert_called_once_with(text=html_content)

    def test_unstructured_parser_complex_html(
        self, parser, mock_partition_html, mock_clean_whitespace
    ):
        """Test parsing complex HTML with multiple elements."""
        complex_html = """
        <html>
            <body>
                <h1>Main Title</h1>
                <h2>Subtitle</h2>
                <p>First paragraph</p>
                <p>Second paragraph</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </body>
        </html>
        """

        mock_elements = [
            Mock(text="Main Title"),
            Mock(text="Subtitle"),
            Mock(text="First paragraph"),
            Mock(text="Second paragraph"),
            Mock(text="Item 1"),
            Mock(text="Item 2"),
        ]
        mock_partition_html.return_value = mock_elements

        result = parser.parse(complex_html)

        expected = "Main Title\n\nSubtitle\n\nFirst paragraph\n\nSecond paragraph\n\nItem 1\n\nItem 2"
        assert result == expected

    def test_unstructured_parser_empty_content(self, parser):
        """Test parsing empty content."""
        result = parser.parse("")
        assert result is None

        result = parser.parse(None)
        assert result is None

    def test_unstructured_parser_non_string_content(self, parser):
        """Test parsing non-string content."""
        result = parser.parse(123)
        assert result is None

        result = parser.parse({"key": "value"})
        assert result is None

    def test_unstructured_parser_no_library(self, parser):
        """Test behavior when unstructured library is not available."""
        with patch(
            "confluence_gateway.services.parsers.html_parsers.partition_html", None
        ):
            result = parser.parse("<p>Test</p>")
            assert result is None

    def test_unstructured_parser_no_text_attribute(
        self, parser, mock_partition_html, mock_clean_whitespace
    ):
        """Test handling elements without text attribute."""
        # Mix of elements with and without text attribute
        mock_elements = [
            Mock(text="Has text"),
            Mock(spec=[]),  # No text attribute
            Mock(text="Also has text"),
        ]
        mock_partition_html.return_value = mock_elements

        result = parser.parse("<p>Test</p>")

        assert result == "Has text\n\nAlso has text"

    def test_unstructured_parser_exception_handling(self, parser, mock_partition_html):
        """Test exception handling during parsing."""
        mock_partition_html.side_effect = Exception("Parsing error")

        result = parser.parse("<p>Test</p>")

        assert result is None

    def test_unstructured_parser_empty_elements(self, parser, mock_partition_html):
        """Test handling empty elements list."""
        mock_partition_html.return_value = []

        result = parser.parse("<p>Test</p>")

        assert result is None

    def test_unstructured_parser_whitespace_cleaning(
        self, parser, mock_partition_html, mock_clean_whitespace
    ):
        """Test whitespace cleaning functionality."""
        mock_elements = [
            Mock(text="  Text with spaces  "),
            Mock(text="  Another text  "),
        ]
        mock_partition_html.return_value = mock_elements

        result = parser.parse("<p>Test</p>")

        # clean_extra_whitespace should be called
        assert mock_clean_whitespace.called
        assert result == "  Text with spaces  \n\n  Another text  "

    @patch("confluence_gateway.services.parsers.html_parsers.logger")
    def test_unstructured_parser_logging(
        self, mock_logger, parser, mock_partition_html
    ):
        """Test proper logging in unstructured parser."""
        # Successful parsing
        mock_elements = [Mock(text="Text1"), Mock(text="Text2")]
        mock_partition_html.return_value = mock_elements

        parser.parse("<p>Test</p>")
        mock_logger.debug.assert_called_with(
            "Successfully extracted text using unstructured (combined 2 elements)"
        )

        # Error during parsing
        mock_partition_html.side_effect = Exception("Test error")
        parser.parse("<p>Test</p>")
        mock_logger.error.assert_called()


class TestParserFactory:
    def test_parser_factory_selection(self):
        """Test parser factory correctly selects parsers."""
        # Test with mocked availability
        with patch(
            "confluence_gateway.services.parsers.html_parsers.MarkItDownClass", Mock()
        ):
            parser = get_parser("markitdown", "html")
            assert isinstance(parser, MarkitdownHtmlParser)

        with patch(
            "confluence_gateway.services.parsers.html_parsers.partition_html", Mock()
        ):
            parser = get_parser("unstructured", "html")
            assert isinstance(parser, UnstructuredHtmlParser)

    def test_parser_factory_unavailable(self):
        """Test parser factory when libraries are not available."""
        # Patch the imports in the factory module
        with patch(
            "confluence_gateway.services.parsers.factory.MarkItDownClassHtml", None
        ):
            with pytest.raises(
                ParserNotAvailableError, match="Markitdown library not installed"
            ):
                get_parser("markitdown", "html")

        with patch("confluence_gateway.services.parsers.factory.partition_html", None):
            with pytest.raises(ParserNotAvailableError, match="Unstructured library"):
                get_parser("unstructured", "html")

    def test_parser_factory_invalid_name(self):
        """Test parser factory with invalid parser name."""
        with pytest.raises(ValueError, match="Unsupported HTML parser name"):
            get_parser("invalid_parser", "html")

    def test_parser_factory_case_insensitive(self):
        """Test parser factory is case-insensitive."""
        with patch(
            "confluence_gateway.services.parsers.html_parsers.MarkItDownClass", Mock()
        ):
            parser1 = get_parser("MarkItDown", "html")
            parser2 = get_parser("MARKITDOWN", "html")
            parser3 = get_parser("markitdown", "html")

            assert all(
                isinstance(p, MarkitdownHtmlParser) for p in [parser1, parser2, parser3]
            )
