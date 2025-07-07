import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from confluence_gateway.services.parsers.attachment_parsers import (
    MarkitdownAttachmentParser,
    UnstructuredAttachmentParser,
)


class TestMarkitdownAttachmentParser:
    @pytest.fixture
    def parser(self):
        """Create a MarkitdownAttachmentParser instance."""
        return MarkitdownAttachmentParser()

    @pytest.fixture
    def mock_markitdown(self):
        """Mock the MarkItDown class."""
        with patch(
            "confluence_gateway.services.parsers.attachment_parsers.MarkItDownClass"
        ) as mock:
            mock_converter = Mock()
            mock_result = Mock()
            mock_result.markdown = "This is extracted text from attachment"
            mock_converter.convert.return_value = mock_result
            mock.return_value = mock_converter
            yield mock

    def test_pdf_parser_basic(self, parser, mock_markitdown):
        """Test basic PDF parsing."""
        pdf_content = b"%PDF-1.4 fake pdf content"

        result = parser.parse(pdf_content, filename="test.pdf")

        assert result == "This is extracted text from attachment"
        assert mock_markitdown.called
        assert mock_markitdown.return_value.convert.called

        # Verify temp file was created with correct suffix
        convert_call = mock_markitdown.return_value.convert.call_args[0][0]
        assert "_test.pdf" in convert_call

    def test_docx_parser_with_images(self, parser, mock_markitdown):
        """Test DOCX parsing with complex content."""
        docx_content = b"PK\x03\x04 fake docx content with images"

        mock_markitdown.return_value.convert.return_value.markdown = "Document Title\n\nParagraph with text and [Image: diagram.png]\n\nMore content"

        result = parser.parse(docx_content, filename="document.docx")

        assert (
            result
            == "Document Title Paragraph with text and [Image: diagram.png] More content"
        )
        assert "document.docx" in str(mock_markitdown.return_value.convert.call_args)

    def test_pptx_parser_slides(self, parser, mock_markitdown):
        """Test PPTX parsing with multiple slides."""
        pptx_content = b"PK\x03\x04 fake pptx content"

        mock_markitdown.return_value.convert.return_value.markdown = (
            "Slide 1: Title\n\nSlide 2: Content\n\nSlide 3: Conclusion"
        )

        result = parser.parse(pptx_content, filename="presentation.pptx")

        assert "Slide 1: Title" in result
        assert "Slide 2: Content" in result
        assert "Slide 3: Conclusion" in result

    def test_txt_parser_encoding(self, parser, mock_markitdown):
        """Test text file parsing with different encodings."""
        txt_content = "Hello world with unicode: 你好世界 🌍".encode()

        mock_markitdown.return_value.convert.return_value.markdown = (
            "Hello world with unicode: 你好世界 🌍"
        )

        result = parser.parse(txt_content, filename="unicode.txt")

        assert "你好世界" in result
        assert "🌍" in result

    def test_md_parser_formatting(self, parser, mock_markitdown):
        """Test markdown file parsing."""
        md_content = b"# Title\n\n## Subtitle\n\n- Item 1\n- Item 2"

        mock_markitdown.return_value.convert.return_value.markdown = (
            "Title Subtitle Item 1 Item 2"
        )

        result = parser.parse(md_content, filename="readme.md")

        assert result == "Title Subtitle Item 1 Item 2"

    def test_unsupported_file_type(self, parser, mock_markitdown):
        """Test parsing unsupported file types."""
        # Markitdown should attempt to parse any file type
        binary_content = b"\x00\x01\x02\x03 binary data"

        mock_markitdown.return_value.convert.return_value.markdown = ""

        result = parser.parse(binary_content, filename="unknown.xyz")

        assert result is None  # Empty text returns None

    def test_file_size_limits(self, parser, mock_markitdown):
        """Test parsing large files."""
        # Create 10MB of content
        large_content = b"A" * (10 * 1024 * 1024)

        mock_markitdown.return_value.convert.return_value.markdown = (
            "Large file content"
        )

        result = parser.parse(large_content, filename="large.txt")

        assert result == "Large file content"

    def test_corrupted_files(self, parser, mock_markitdown):
        """Test handling of corrupted files."""
        corrupted_content = b"corrupted pdf content without proper header"

        # Simulate markitdown failing on corrupted file
        mock_markitdown.return_value.convert.side_effect = Exception(
            "Cannot parse corrupted file"
        )

        result = parser.parse(corrupted_content, filename="corrupted.pdf")

        assert result is None

    def test_empty_content(self, parser):
        """Test parsing empty content."""
        result = parser.parse(b"", filename="empty.txt")
        assert result is None

        result = parser.parse(None, filename="null.txt")
        assert result is None

    def test_non_bytes_content(self, parser):
        """Test parsing non-bytes content."""
        result = parser.parse("string content", filename="test.txt")
        assert result is None

        result = parser.parse(123, filename="test.txt")
        assert result is None

    def test_no_filename_provided(self, parser, mock_markitdown):
        """Test parsing without filename."""
        content = b"Some content"

        parser.parse(content)

        # Should use default filename
        convert_call = mock_markitdown.return_value.convert.call_args[0][0]
        assert "_unknown_attachment" in convert_call

    def test_no_library_available(self, parser):
        """Test behavior when markitdown is not available."""
        with patch(
            "confluence_gateway.services.parsers.attachment_parsers.MarkItDownClass",
            None,
        ):
            result = parser.parse(b"content", filename="test.pdf")
            assert result is None

    def test_temp_file_cleanup(self, parser, mock_markitdown):
        """Test that temporary files are cleaned up."""
        content = b"Test content"

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test_file.pdf"
            mock_temp.return_value.__enter__.return_value = mock_file

            with patch("pathlib.Path.unlink") as mock_unlink:
                parser.parse(content, filename="test.pdf")

                # Verify temp file was deleted
                mock_unlink.assert_called_once()

    def test_temp_file_cleanup_failure(self, parser, mock_markitdown):
        """Test handling of temp file cleanup failure."""
        content = b"Test content"

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test_file.pdf"
            mock_temp.return_value.__enter__.return_value = mock_file

            with patch("pathlib.Path.unlink", side_effect=OSError("Permission denied")):
                with patch(
                    "confluence_gateway.services.parsers.attachment_parsers.logger"
                ) as mock_logger:
                    parser.parse(content, filename="test.pdf")

                    # Should log warning but not raise
                    mock_logger.warning.assert_called_once()
                    assert (
                        "Could not delete temporary file"
                        in mock_logger.warning.call_args[0][0]
                    )

    @patch("confluence_gateway.services.parsers.attachment_parsers.logger")
    def test_logging(self, mock_logger, parser, mock_markitdown):
        """Test proper logging."""
        # Successful parsing
        parser.parse(b"content", filename="test.pdf")
        mock_logger.debug.assert_called_with(
            "Successfully extracted text from attachment 'test.pdf' using markitdown"
        )

        # Error during parsing
        mock_markitdown.return_value.convert.side_effect = Exception("Test error")
        parser.parse(b"content", filename="error.pdf")
        mock_logger.error.assert_called()
        assert (
            "Markitdown failed to parse attachment 'error.pdf'"
            in mock_logger.error.call_args[0][0]
        )


class TestUnstructuredAttachmentParser:
    @pytest.fixture
    def parser(self):
        """Create an UnstructuredAttachmentParser instance."""
        return UnstructuredAttachmentParser()

    @pytest.fixture
    def mock_partition(self):
        """Mock the unstructured partition function."""
        with patch(
            "confluence_gateway.services.parsers.attachment_parsers.unstructured_partition"
        ) as mock:
            yield mock

    @pytest.fixture
    def mock_clean_whitespace(self):
        """Mock the clean_extra_whitespace function."""
        with patch(
            "confluence_gateway.services.parsers.attachment_parsers.clean_extra_whitespace"
        ) as mock:
            mock.side_effect = lambda x: x  # Don't modify text
            yield mock

    def test_pdf_parser_basic(self, parser, mock_partition, mock_clean_whitespace):
        """Test basic PDF parsing with unstructured."""
        pdf_content = b"%PDF-1.4 fake pdf content"

        mock_elements = [Mock(text="Page 1 content"), Mock(text="Page 2 content")]
        mock_partition.return_value = mock_elements

        result = parser.parse(
            pdf_content, filename="test.pdf", content_type="application/pdf"
        )

        assert result == "Page 1 content\n\nPage 2 content"
        mock_partition.assert_called_once()

        # Verify BytesIO was used
        call_args = mock_partition.call_args
        assert isinstance(call_args.kwargs["file"], io.BytesIO)
        assert call_args.kwargs["file_filename"] == "test.pdf"
        assert call_args.kwargs["content_type"] == "application/pdf"

    def test_docx_parser_complex(self, parser, mock_partition, mock_clean_whitespace):
        """Test DOCX parsing with tables and lists."""
        docx_content = b"PK\x03\x04 fake docx content"

        mock_elements = [
            Mock(text="Title"),
            Mock(text="Table cell 1"),
            Mock(text="Table cell 2"),
            Mock(text="List item 1"),
            Mock(text="List item 2"),
        ]
        mock_partition.return_value = mock_elements

        result = parser.parse(docx_content, filename="document.docx")

        expected = "Title\n\nTable cell 1\n\nTable cell 2\n\nList item 1\n\nList item 2"
        assert result == expected

    def test_xlsx_parser(self, parser, mock_partition, mock_clean_whitespace):
        """Test Excel file parsing."""
        xlsx_content = b"PK\x03\x04 fake xlsx content"

        mock_elements = [
            Mock(text="Header 1,Header 2"),
            Mock(text="Data 1,Data 2"),
            Mock(text="Data 3,Data 4"),
        ]
        mock_partition.return_value = mock_elements

        result = parser.parse(
            xlsx_content,
            filename="spreadsheet.xlsx",
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        assert "Header 1,Header 2" in result
        assert "Data 1,Data 2" in result

    def test_empty_content(self, parser):
        """Test parsing empty content."""
        result = parser.parse(b"", filename="empty.pdf")
        assert result is None

        result = parser.parse(None, filename="null.pdf")
        assert result is None

    def test_non_bytes_content(self, parser):
        """Test parsing non-bytes content."""
        result = parser.parse("string content", filename="test.pdf")
        assert result is None

        result = parser.parse({"data": "value"}, filename="test.pdf")
        assert result is None

    def test_no_library_available(self, parser):
        """Test behavior when unstructured is not available."""
        with patch(
            "confluence_gateway.services.parsers.attachment_parsers.unstructured_partition",
            None,
        ):
            result = parser.parse(b"content", filename="test.pdf")
            assert result is None

    def test_no_text_attribute(self, parser, mock_partition, mock_clean_whitespace):
        """Test handling elements without text attribute."""
        mock_elements = [
            Mock(text="Has text"),
            Mock(spec=[]),  # No text attribute
            Mock(text="Also has text"),
        ]
        mock_partition.return_value = mock_elements

        result = parser.parse(b"content", filename="test.pdf")

        assert result == "Has text\n\nAlso has text"

    def test_exception_handling(self, parser, mock_partition):
        """Test exception handling during parsing."""
        mock_partition.side_effect = Exception("Parsing failed")

        result = parser.parse(b"content", filename="error.pdf")

        assert result is None

    def test_empty_elements(self, parser, mock_partition):
        """Test handling empty elements list."""
        mock_partition.return_value = []

        result = parser.parse(b"content", filename="empty.pdf")

        assert result is None

    def test_no_filename_provided(self, parser, mock_partition, mock_clean_whitespace):
        """Test parsing without filename."""
        mock_elements = [Mock(text="Content")]
        mock_partition.return_value = mock_elements

        parser.parse(b"content")

        # Should use default filename
        call_args = mock_partition.call_args
        assert call_args.kwargs["file_filename"] == "unknown_attachment"

    def test_content_type_handling(self, parser, mock_partition, mock_clean_whitespace):
        """Test content type is properly passed."""
        mock_elements = [Mock(text="Content")]
        mock_partition.return_value = mock_elements

        # With content type
        parser.parse(b"content", filename="test.pdf", content_type="application/pdf")
        assert mock_partition.call_args.kwargs["content_type"] == "application/pdf"

        # Without content type
        parser.parse(b"content", filename="test.pdf")
        assert mock_partition.call_args.kwargs["content_type"] is None

    @patch("confluence_gateway.services.parsers.attachment_parsers.logger")
    def test_logging(self, mock_logger, parser, mock_partition):
        """Test proper logging."""
        # Successful parsing
        mock_elements = [Mock(text="Text1"), Mock(text="Text2")]
        mock_partition.return_value = mock_elements

        parser.parse(b"content", filename="test.pdf")
        mock_logger.debug.assert_called_with(
            "Successfully extracted text from attachment 'test.pdf' using unstructured (combined 2 elements)"
        )

        # Error during parsing
        mock_partition.side_effect = Exception("Test error")
        parser.parse(b"content", filename="error.pdf")
        mock_logger.error.assert_called()
        assert (
            "Unstructured failed to parse attachment 'error.pdf'"
            in mock_logger.error.call_args[0][0]
        )

    def test_large_files(self, parser, mock_partition, mock_clean_whitespace):
        """Test parsing large files."""
        # Create 50MB of content
        large_content = b"A" * (50 * 1024 * 1024)

        mock_elements = [Mock(text=f"Section {i}") for i in range(100)]
        mock_partition.return_value = mock_elements

        result = parser.parse(large_content, filename="large.pdf")

        assert result is not None
        assert "Section 0" in result
        assert "Section 99" in result
