#!/usr/bin/env python3

import re
from pathlib import Path
from typing import Any

import yaml


def load_markdown_with_metadata(markdown_path: str) -> tuple[str, dict[str, Any]]:
    """
    Load markdown content and extract YAML front matter metadata.

    Args:
        markdown_path: Path to the markdown file

    Returns:
        Tuple of (content, metadata) where:
        - content: The markdown content without front matter
        - metadata: Dictionary of metadata from YAML front matter
    """
    path = Path(markdown_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Extract YAML front matter if present
    metadata = {}
    if content.startswith("---\n"):
        try:
            # Find the closing ---
            end_index = content.find("\n---\n", 4)
            if end_index > 0:
                yaml_content = content[4:end_index]
                metadata = yaml.safe_load(yaml_content) or {}
                # Remove front matter from content
                content = content[end_index + 5 :]  # Skip past \n---\n
        except yaml.YAMLError:
            # If YAML parsing fails, treat entire content as markdown
            pass

    return content.strip(), metadata


def load_entry_content(
    entry: dict[str, Any], markdown_base_dir: Path | None = None
) -> str:
    """
    Load content for an entry, either from the entry dict or from markdown file.

    Args:
        entry: Entry dictionary from content_index.json
        markdown_base_dir: Base directory for markdown files (default: config/real_data)

    Returns:
        The content as a string
    """
    # First check if entry has markdown_path
    if "markdown_path" in entry and entry["markdown_path"]:
        content, _ = load_markdown_with_metadata(entry["markdown_path"])
        return content

    # Fallback to content field if present
    if "content" in entry and entry["content"]:
        return entry["content"]

    # If neither is available, try to construct markdown path
    if markdown_base_dir is None:
        markdown_base_dir = Path("config/real_data")

    # Try to find markdown file based on entry ID and category
    if "id" in entry and "categorization" in entry:
        category = entry["categorization"].get("primary_category", "unknown")
        pattern = f"**/​*{entry['id'][:8]}*.md"

        markdown_dir = markdown_base_dir / "markdown_content" / category
        if markdown_dir.exists():
            files = list(markdown_dir.glob(pattern))
            if files:
                content, _ = load_markdown_with_metadata(str(files[0]))
                return content

    return "Content not available"


def get_markdown_files_for_category(
    category: str, markdown_base_dir: Path | None = None
) -> list[Path]:
    """
    Get all markdown files for a specific category.

    Args:
        category: Category name (e.g., 'api_docs', 'technical', 'knowledge_base')
        markdown_base_dir: Base directory for markdown files

    Returns:
        List of Path objects for markdown files in the category
    """
    if markdown_base_dir is None:
        markdown_base_dir = Path("config/real_data")

    markdown_dir = markdown_base_dir / "markdown_content" / category
    if not markdown_dir.exists():
        return []

    return list(markdown_dir.rglob("*.md"))


def convert_markdown_to_clean_confluence(content: str, title: str = None) -> str:
    """
    Convert markdown content to clean Confluence markup without extra formatting.

    Args:
        content: Markdown content string
        title: Optional title to use as H1 header

    Returns:
        Clean Confluence markup string
    """
    if not content:
        return ""

    # Remove YAML frontmatter if present
    if content.startswith("---\n"):
        end_index = content.find("\n---\n", 4)
        if end_index > 0:
            content = content[end_index + 5 :].strip()

    # Remove duplicate header section added by collector
    lines = content.split("\n")
    cleaned_lines = []
    skip_until_hr = False
    found_first_hr = False

    for line in lines:
        # Skip the duplicate header section until first horizontal rule
        if not found_first_hr and line.strip() == "---":
            found_first_hr = True
            skip_until_hr = False
            continue
        elif not found_first_hr:
            # Skip lines before first HR (duplicate header info)
            continue

        cleaned_lines.append(line)

    content = "\n".join(cleaned_lines).strip()

    # Add title as H1 if provided and not already present
    if title and not content.startswith("# "):
        content = f"# {title}\n\n{content}"

    # Convert markdown to clean Confluence markup
    # Headers
    content = re.sub(r"^#### (.+)$", r"h4. \1", content, flags=re.MULTILINE)
    content = re.sub(r"^### (.+)$", r"h3. \1", content, flags=re.MULTILINE)
    content = re.sub(r"^## (.+)$", r"h2. \1", content, flags=re.MULTILINE)
    content = re.sub(r"^# (.+)$", r"h1. \1", content, flags=re.MULTILINE)

    # Bold and italic
    content = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", content)
    content = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"_\1_", content)

    # Inline code
    content = re.sub(r"`([^`]+)`", r"{{{\1}}}", content)

    # Code blocks - fix the format
    def format_code_block(match):
        lang = match.group(1) or "none"
        code = match.group(2).strip()
        return f"{{code:{lang}}}\n{code}\n{{code}}"

    content = re.sub(
        r"```(\w*)\n?(.*?)\n?```", format_code_block, content, flags=re.DOTALL
    )

    # Links - convert [text](url) to [text|url]
    content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"[\1|\2]", content)

    # Lists - Confluence uses * for unordered and # for ordered
    lines = content.split("\n")
    for i, line in enumerate(lines):
        # Unordered lists
        if re.match(r"^[-*+]\s+", line):
            lines[i] = re.sub(r"^[-*+]\s+", "* ", line)
        # Ordered lists
        elif re.match(r"^\d+\.\s+", line):
            lines[i] = re.sub(r"^\d+\.\s+", "# ", line)

    content = "\n".join(lines)

    # Clean up excessive whitespace
    content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)
    content = content.strip()

    return content


def convert_entry_to_clean_confluence(
    entry: dict[str, Any], markdown_base_dir: Path | None = None
) -> str:
    """
    Convert an entry to clean Confluence markup.

    Args:
        entry: Entry dictionary from content_index.json
        markdown_base_dir: Base directory for markdown files

    Returns:
        Clean Confluence markup string
    """
    # Load the content
    content = load_entry_content(entry, markdown_base_dir)

    # Get title from entry
    title = entry.get("title", "Untitled Document")

    # Convert to clean Confluence format
    return convert_markdown_to_clean_confluence(content, title)


def prepare_markdown_for_plugin(
    entry: dict[str, Any], markdown_base_dir: Path | None = None
) -> str:
    """
    Prepare markdown content specifically for Just Add+ Markdown plugin.

    This function loads the raw markdown content and cleans it for optimal
    rendering by the Just Add+ Markdown plugin macro.

    Args:
        entry: Entry dictionary from content_index.json
        markdown_base_dir: Base directory for markdown files

    Returns:
        Clean markdown content ready for plugin rendering
    """
    # Load the raw content
    content = load_entry_content(entry, markdown_base_dir)

    if not content:
        return ""

    # Remove YAML frontmatter if present since the plugin will handle markdown rendering
    if content.startswith("---\n"):
        end_index = content.find("\n---\n", 4)
        if end_index > 0:
            content = content[end_index + 5 :].strip()

    # Remove duplicate header section added by collector (everything before first HR)
    lines = content.split("\n")
    cleaned_lines = []
    found_first_hr = False

    for line in lines:
        if not found_first_hr and line.strip() == "---":
            found_first_hr = True
            continue
        elif not found_first_hr:
            # Skip lines before first HR (duplicate header info)
            continue

        cleaned_lines.append(line)

    content = "\n".join(cleaned_lines).strip()

    # Ensure the title is present as H1 if not already there
    title = entry.get("title", "Untitled Document")
    if content and not content.startswith("# "):
        content = f"# {title}\n\n{content}"

    return content
