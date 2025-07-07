#!/usr/bin/env python3

import html
import uuid
from pathlib import Path
from typing import Any

from content_analyzer import ContentAnalyzer, StructuredContent
from markdown_utils import (
    convert_entry_to_clean_confluence,
    load_entry_content,
    prepare_markdown_for_plugin,
)


class ConfluenceFormatter:
    def __init__(self, use_markdown_plugin: bool = True):
        self.use_markdown_plugin = use_markdown_plugin
        self.analyzer = ContentAnalyzer()
        self.emoji_map = {
            "api_docs": "🔌",
            "api_documentation": "🔌",
            "tutorial": "📚",
            "troubleshooting": "🚨",
            "reference": "📖",
            "code_example": "💻",
            "generic": "📄",
            "info": "ℹ️",
            "warning": "⚠️",
            "tip": "💡",
            "note": "📝",
            "success": "✅",
            "error": "❌",
            "link": "🔗",
            "document": "📄",
            "stats": "📊",
            "prerequisites": "📋",
            "quick": "⚡",
            "navigation": "🧭",
            "time": "⏱️",
            "target": "🎯",
        }

    def format_entry_for_confluence(
        self, entry: dict[str, Any], markdown_base_dir: Path | None = None
    ) -> str:
        """Format entry for Confluence using Just Add+ Markdown plugin or fallback to clean format"""
        if self.use_markdown_plugin:
            return self._format_with_markdown_plugin(entry, markdown_base_dir)
        else:
            # Fallback to clean format for compatibility
            return convert_entry_to_clean_confluence(entry, markdown_base_dir)

    def _format_with_markdown_plugin(
        self, entry: dict[str, Any], markdown_base_dir: Path | None = None
    ) -> str:
        """Format entry using Just Add+ Markdown plugin macro"""
        # Load and prepare markdown content for the plugin
        content = prepare_markdown_for_plugin(entry, markdown_base_dir)

        # Analyze content for metadata (using raw content for analysis)
        raw_content = load_entry_content(entry, markdown_base_dir)
        structured = self.analyzer.analyze_content(
            raw_content, {"title": entry["title"], "name": entry["source"]["name"]}
        )

        # Create header section with metadata
        header_section = self._create_simple_header(entry, structured)

        # Create the markdown macro with the cleaned content
        markdown_macro = self._create_markdown_macro(content)

        # Create footer with document information
        footer_section = self._create_simple_footer(entry)

        return f"{header_section}\n\n{markdown_macro}\n\n{footer_section}"

    def _create_markdown_macro(self, content: str) -> str:
        """Create Just Add+ Markdown macro with inline content"""
        # Escape CDATA content properly
        escaped_content = self._safe_cdata(content)

        return f"""<ac:structured-macro ac:name="markdown" ac:schema-version="1" ac:macro-id="{self._generate_macro_id()}">
<ac:parameter ac:name="source">inline</ac:parameter>
<ac:plain-text-body>{escaped_content}</ac:plain-text-body>
</ac:structured-macro>"""

    def _create_simple_header(
        self, entry: dict[str, Any], structured: StructuredContent
    ) -> str:
        """Create a simple header with basic metadata"""
        content_type_display = structured.content_type.value.replace("_", " ").title()
        content_icon = self.emoji_map.get(
            structured.content_type.value, self.emoji_map["document"]
        )

        # Create basic stats
        stats = []
        if len(structured.headers) > 0:
            stats.append(f"{len(structured.headers)} sections")
        if len(structured.code_blocks) > 0:
            stats.append(f"{len(structured.code_blocks)} code examples")
        if len(structured.api_endpoints) > 0:
            stats.append(f"{len(structured.api_endpoints)} API endpoints")

        stats_text = " | ".join(stats) if stats else "No structured content detected"

        return f"""<ac:structured-macro ac:name="panel">
<ac:parameter ac:name="borderStyle">solid</ac:parameter>
<ac:parameter ac:name="borderColor">#0052cc</ac:parameter>
<ac:parameter ac:name="bgColor">#f7f8f9</ac:parameter>
<ac:rich-text-body>
<h1>{content_icon} {self._safe_xml_escape(entry["title"])}</h1>
<p><strong>Source:</strong> <a href="{self._safe_xml_escape(entry["url"])}" class="external-link" rel="nofollow">{self._safe_xml_escape(entry["source"]["name"])}</a></p>
<p><strong>Type:</strong> {content_type_display} | <strong>Categories:</strong> {", ".join(entry["categorization"]["categories"])} | <strong>Updated:</strong> {entry["collection_time"][:10]}</p>
<p><strong>Content:</strong> {stats_text}</p>
</ac:rich-text-body>
</ac:structured-macro>"""

    def _create_simple_footer(self, entry: dict[str, Any]) -> str:
        """Create a simple footer with document metadata"""
        return f"""<hr/>
<ac:structured-macro ac:name="note">
<ac:parameter ac:name="icon">false</ac:parameter>
<ac:parameter ac:name="title">{self.emoji_map["info"]} Document Information</ac:parameter>
<ac:rich-text-body>
<p><strong>Document ID:</strong> <code>{entry["id"]}</code> | <strong>Word Count:</strong> {entry["content_info"]["word_count"]:,} words | <strong>Reading Time:</strong> ~{self._estimate_reading_time(entry["content_info"]["word_count"])} minutes</p>
<p><strong>Original Source:</strong> <a href="{self._safe_xml_escape(entry["url"])}" class="external-link" rel="nofollow">{self.emoji_map["link"]} View Original</a></p>
</ac:rich-text-body>
</ac:structured-macro>"""

    def _generate_macro_id(self) -> str:
        """Generate a unique ID for the macro"""
        return str(uuid.uuid4())

    def _estimate_reading_time(self, word_count: int) -> int:
        """Estimate reading time based on word count (200 words per minute)"""
        return max(1, word_count // 200)

    def _safe_xml_escape(self, text: str) -> str:
        """Safely escape text for XML content"""
        if not text:
            return ""
        # Use HTML escape for basic entities
        text = html.escape(text, quote=True)
        return text

    def _safe_cdata(self, code: str) -> str:
        """Safely wrap content in CDATA, handling nested CDATA sequences"""
        if not code:
            return ""
        # Replace any ]]> sequences in the code to prevent CDATA breaking
        code = code.replace("]]>", "]]]]><![CDATA[>")
        return f"<![CDATA[{code}]]>"
