#!/usr/bin/env python3

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ContentType(Enum):
    API_DOCUMENTATION = "api_docs"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    TROUBLESHOOTING = "troubleshooting"
    CODE_EXAMPLE = "code_example"
    GENERIC = "generic"


@dataclass
class ContentBlock:
    type: str
    content: str
    metadata: dict[str, Any]
    start_pos: int
    end_pos: int


@dataclass
class StructuredContent:
    title: str
    content_type: ContentType
    blocks: list[ContentBlock]
    headers: list[dict[str, Any]]
    code_blocks: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    links: list[dict[str, Any]]
    api_endpoints: list[dict[str, Any]]


class ContentAnalyzer:
    def __init__(self):
        self.api_patterns = {
            "endpoint": re.compile(
                r"(GET|POST|PUT|DELETE|PATCH)\s+(/[^\s]+)", re.IGNORECASE
            ),
            "http_status": re.compile(r"\b(2\d{2}|4\d{2}|5\d{2})\b"),
            "json_response": re.compile(r'{\s*"[^"]+"\s*:', re.MULTILINE),
            "parameter": re.compile(r"\b(Parameters?|Attributes?)\b", re.IGNORECASE),
        }

        self.code_patterns = {
            "code_block": re.compile(r"```(\w+)?\s*\n?(.*?)\n?```", re.DOTALL),
            "inline_code": re.compile(r"`([^`]+)`"),
            "curl_command": re.compile(r"curl\s+[^\n]+", re.IGNORECASE),
        }

        self.structure_patterns = {
            "headers": re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE),
            "list_items": re.compile(r"^[\s]*[-*+]\s+(.+)$", re.MULTILINE),
            "numbered_list": re.compile(r"^[\s]*\d+\.\s+(.+)$", re.MULTILINE),
            "table_row": re.compile(r"\|[^|\n]+\|", re.MULTILINE),
        }

    def analyze_markdown_file(self, markdown_path: str) -> StructuredContent:
        """Analyze a markdown file directly"""
        from pathlib import Path

        from markdown_utils import load_markdown_with_metadata

        content, metadata = load_markdown_with_metadata(markdown_path)

        source_info = {
            "title": metadata.get("title", Path(markdown_path).stem),
            "name": metadata.get("source", "Unknown Source"),
        }

        return self.analyze_content(content, source_info)

    def analyze_content(
        self, content: str, source_info: dict[str, Any]
    ) -> StructuredContent:
        content_type = self._classify_content_type(content, source_info)

        blocks = self._extract_content_blocks(content)
        headers = self._extract_headers(content)
        code_blocks = self._extract_code_blocks(content)
        tables = self._extract_tables(content)
        links = self._extract_links(content)
        api_endpoints = self._extract_api_endpoints(content)

        return StructuredContent(
            title=source_info.get("title", "Untitled"),
            content_type=content_type,
            blocks=blocks,
            headers=headers,
            code_blocks=code_blocks,
            tables=tables,
            links=links,
            api_endpoints=api_endpoints,
        )

    def _classify_content_type(
        self, content: str, source_info: dict[str, Any]
    ) -> ContentType:
        content_lower = content.lower()

        # Check for API documentation patterns
        if (
            self.api_patterns["endpoint"].search(content)
            or "api" in source_info.get("name", "").lower()
            or any(
                word in content_lower
                for word in ["endpoint", "parameters", "response", "request"]
            )
        ):
            return ContentType.API_DOCUMENTATION

        # Check for tutorial patterns
        if any(
            word in content_lower
            for word in ["tutorial", "getting started", "quickstart", "guide"]
        ):
            return ContentType.TUTORIAL

        # Check for troubleshooting patterns
        if any(
            word in content_lower
            for word in ["error", "troubleshoot", "problem", "issue", "fix"]
        ):
            return ContentType.TROUBLESHOOTING

        # Check for reference patterns
        if any(
            word in content_lower
            for word in ["reference", "documentation", "spec", "manual"]
        ):
            return ContentType.REFERENCE

        return ContentType.GENERIC

    def _extract_content_blocks(self, content: str) -> list[ContentBlock]:
        blocks = []

        # Split content by major sections (headers)
        sections = re.split(r"\n(?=#{1,6}\s)", content)

        current_pos = 0
        for section in sections:
            if not section.strip():
                continue

            block_type = self._determine_block_type(section)
            start_pos = current_pos
            end_pos = current_pos + len(section)

            blocks.append(
                ContentBlock(
                    type=block_type,
                    content=section.strip(),
                    metadata=self._extract_block_metadata(section),
                    start_pos=start_pos,
                    end_pos=end_pos,
                )
            )

            current_pos = end_pos

        return blocks

    def _determine_block_type(self, section: str) -> str:
        section_lower = section.lower()

        if self.code_patterns["code_block"].search(section):
            return "code_section"
        elif self.api_patterns["endpoint"].search(section):
            return "api_section"
        elif "parameter" in section_lower or "attribute" in section_lower:
            return "parameter_section"
        elif self.structure_patterns["table_row"].search(section):
            return "table_section"
        elif any(word in section_lower for word in ["example", "sample"]):
            return "example_section"
        else:
            return "text_section"

    def _extract_block_metadata(self, section: str) -> dict[str, Any]:
        metadata = {}

        # Extract header if present
        header_match = re.match(r"^(#{1,6})\s+(.+)$", section, re.MULTILINE)
        if header_match:
            metadata["header_level"] = len(header_match.group(1))
            metadata["header_text"] = header_match.group(2).strip()

        # Count code blocks
        code_blocks = self.code_patterns["code_block"].findall(section)
        if code_blocks:
            metadata["code_block_count"] = len(code_blocks)
            metadata["languages"] = [lang for lang, _ in code_blocks if lang]

        # Check for API endpoints
        endpoints = self.api_patterns["endpoint"].findall(section)
        if endpoints:
            metadata["endpoints"] = endpoints

        return metadata

    def _extract_headers(self, content: str) -> list[dict[str, Any]]:
        headers = []

        for match in self.structure_patterns["headers"].finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()

            headers.append(
                {
                    "level": level,
                    "text": text,
                    "position": match.start(),
                    "anchor": self._generate_anchor(text),
                }
            )

        return headers

    def _extract_code_blocks(self, content: str) -> list[dict[str, Any]]:
        code_blocks = []

        for match in self.code_patterns["code_block"].finditer(content):
            language = match.group(1) or "text"
            code = match.group(2).strip()

            code_blocks.append(
                {
                    "language": language,
                    "code": code,
                    "position": match.start(),
                    "is_curl": bool(self.code_patterns["curl_command"].search(code)),
                    "line_count": len(code.split("\n")),
                }
            )

        return code_blocks

    def _extract_tables(self, content: str) -> list[dict[str, Any]]:
        tables = []

        # Simple table detection (markdown-style)
        table_sections = re.split(r"\n\s*\n", content)

        for i, section in enumerate(table_sections):
            if self.structure_patterns["table_row"].search(section):
                rows = [
                    line.strip()
                    for line in section.split("\n")
                    if self.structure_patterns["table_row"].search(line)
                ]

                if len(rows) > 1:  # At least header + one data row
                    tables.append(
                        {
                            "rows": rows,
                            "row_count": len(rows),
                            "position": i,
                            "has_header": True,
                        }
                    )

        return tables

    def _extract_links(self, content: str) -> list[dict[str, Any]]:
        links = []

        # Markdown links
        md_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
        for text, url in md_links:
            links.append({"text": text, "url": url, "type": "markdown"})

        # Plain URLs
        url_pattern = re.compile(r'https?://[^\s<>"\']+')
        for match in url_pattern.finditer(content):
            url = match.group()
            links.append(
                {"text": url, "url": url, "type": "plain", "position": match.start()}
            )

        return links

    def _extract_api_endpoints(self, content: str) -> list[dict[str, Any]]:
        endpoints = []

        for match in self.api_patterns["endpoint"].finditer(content):
            method = match.group(1).upper()
            path = match.group(2)

            endpoints.append(
                {
                    "method": method,
                    "path": path,
                    "position": match.start(),
                    "context": self._get_context_around_match(content, match, 100),
                }
            )

        return endpoints

    def _generate_anchor(self, text: str) -> str:
        # Convert to lowercase, replace spaces with hyphens, remove special chars
        anchor = re.sub(r"[^\w\s-]", "", text.lower())
        anchor = re.sub(r"\s+", "-", anchor)
        return anchor.strip("-")

    def _get_context_around_match(
        self, content: str, match, context_length: int
    ) -> str:
        start = max(0, match.start() - context_length)
        end = min(len(content), match.end() + context_length)
        return content[start:end].strip()


def analyze_collected_data(
    index_file: str, markdown_base_dir: str | None = None
) -> dict[str, Any]:
    """Analyze all collected content and return structure summary"""
    from pathlib import Path

    from markdown_utils import load_entry_content

    with open(index_file) as f:
        data = json.load(f)

    analyzer = ContentAnalyzer()
    analysis_results = {
        "total_entries": len(data["entries"]),
        "content_types": {},
        "structure_stats": {
            "total_headers": 0,
            "total_code_blocks": 0,
            "total_tables": 0,
            "total_api_endpoints": 0,
            "languages_found": set(),
        },
        "entries_analysis": [],
    }

    # Use provided base dir or default
    if markdown_base_dir:
        base_dir = Path(markdown_base_dir)
    else:
        base_dir = Path("config/real_data")

    for entry in data["entries"]:
        # Load content from markdown file if needed
        content = load_entry_content(entry, base_dir)

        structured = analyzer.analyze_content(
            content, {"title": entry["title"], "name": entry["source"]["name"]}
        )

        # Update statistics
        content_type = structured.content_type.value
        analysis_results["content_types"][content_type] = (
            analysis_results["content_types"].get(content_type, 0) + 1
        )

        analysis_results["structure_stats"]["total_headers"] += len(structured.headers)
        analysis_results["structure_stats"]["total_code_blocks"] += len(
            structured.code_blocks
        )
        analysis_results["structure_stats"]["total_tables"] += len(structured.tables)
        analysis_results["structure_stats"]["total_api_endpoints"] += len(
            structured.api_endpoints
        )

        for code_block in structured.code_blocks:
            analysis_results["structure_stats"]["languages_found"].add(
                code_block["language"]
            )

        analysis_results["entries_analysis"].append(
            {
                "id": entry["id"],
                "title": entry["title"],
                "content_type": content_type,
                "stats": {
                    "headers": len(structured.headers),
                    "code_blocks": len(structured.code_blocks),
                    "tables": len(structured.tables),
                    "api_endpoints": len(structured.api_endpoints),
                },
            }
        )

    # Convert set to list for JSON serialization
    analysis_results["structure_stats"]["languages_found"] = list(
        analysis_results["structure_stats"]["languages_found"]
    )

    return analysis_results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        index_file = sys.argv[1]
    else:
        index_file = "config/real_data/content_index.json"

    if not Path(index_file).exists():
        print(f"Error: {index_file} not found")
        sys.exit(1)

    results = analyze_collected_data(index_file)

    print("Content Analysis Results:")
    print(f"Total entries: {results['total_entries']}")
    print(f"Content types: {results['content_types']}")
    print(f"Structure statistics: {results['structure_stats']}")

    # Save detailed results
    output_file = "config/content_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Detailed analysis saved to: {output_file}")
