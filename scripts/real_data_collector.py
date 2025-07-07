#!/usr/bin/env python3

import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp
import yaml
from bs4 import BeautifulSoup
from markitdown import MarkItDown
from pydantic import BaseModel
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

console = Console()


class ContentEntry(BaseModel):
    id: str
    url: str
    title: str
    content: str
    content_hash: str
    source: dict[str, Any]
    metadata: dict[str, Any]
    categorization: dict[str, Any]
    content_info: dict[str, Any]
    collection_time: str
    attachments: list[dict[str, Any]] = []
    markdown_path: str | None = None  # Path to saved markdown file


class DuplicateDetector:
    def __init__(self, method: str = "content_hash", threshold: float = 0.95):
        self.method = method
        self.threshold = threshold
        self.seen_hashes: set[str] = set()
        self.seen_urls: set[str] = set()

    def is_duplicate(self, entry: ContentEntry) -> bool:
        if self.method == "content_hash":
            if entry.content_hash in self.seen_hashes:
                return True
        elif self.method == "url_hash":
            url_hash = hashlib.sha256(entry.url.encode()).hexdigest()
            if url_hash in self.seen_urls:
                return True
        return False

    def add_entry(self, entry: ContentEntry):
        self.seen_hashes.add(entry.content_hash)
        if self.method == "url_hash":
            url_hash = hashlib.sha256(entry.url.encode()).hexdigest()
            self.seen_urls.add(url_hash)
        else:
            self.seen_urls.add(entry.url)


class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(min(requests_per_minute // 4, 30))

    async def wait_if_needed(self):
        async with self._semaphore:
            async with self._lock:
                current_time = time.time()
                time_since_last = current_time - self.last_request_time

                if time_since_last < self.min_interval:
                    wait_time = self.min_interval - time_since_last
                    await asyncio.sleep(wait_time)

                self.last_request_time = time.time()


class URLValidator:
    """Validates URLs to avoid error pages and invalid content"""

    # Common error page patterns
    ERROR_PATTERNS = [
        r"/404(\.|/|$)",
        r"/error(\.|/|$)",
        r"/not[_-]?found",
        r"/page[_-]?not[_-]?found",
        r"/troubleshooting/.*not[_-]?found",
        r"/troubleshooting/.*resource[_-]?not[_-]?found",
        r"/(400|401|403|404|500|502|503)(\.|/|$)",
        r"/access[_-]?denied",
        r"/forbidden",
        r"/unauthorized",
    ]

    # Common authentication/login patterns to skip
    AUTH_PATTERNS = [
        r"/login(\.|/|$)",
        r"/signin(\.|/|$)",
        r"/auth(\.|/|$)",
        r"/oauth(\.|/|$)",
        r"/register(\.|/|$)",
        r"/signup(\.|/|$)",
    ]

    # File patterns to skip (non-documentation)
    SKIP_FILE_PATTERNS = [
        r"\.(zip|tar|gz|rar|7z|exe|dmg|pkg|deb|rpm)$",
        r"\.(mp3|mp4|avi|mkv|mov|wmv|flv)$",
        r"\.(jpg|jpeg|png|gif|bmp|svg|ico)$",
    ]

    @classmethod
    def is_valid_documentation_url(cls, url: str) -> bool:
        """Check if URL is likely to be valid documentation"""
        url_lower = url.lower()

        # Check for error patterns
        for pattern in cls.ERROR_PATTERNS:
            if re.search(pattern, url_lower):
                console.print(f"[yellow]Skipping error page URL: {url}[/yellow]")
                return False

        # Check for auth patterns
        for pattern in cls.AUTH_PATTERNS:
            if re.search(pattern, url_lower):
                console.print(f"[yellow]Skipping auth page URL: {url}[/yellow]")
                return False

        # Check for file patterns to skip
        for pattern in cls.SKIP_FILE_PATTERNS:
            if re.search(pattern, url_lower):
                console.print(
                    f"[yellow]Skipping non-documentation file: {url}[/yellow]"
                )
                return False

        return True

    @classmethod
    def is_error_page_content(cls, html_content: str, url: str) -> bool:
        """Check if the page content indicates an error page"""
        if not html_content:
            return True

        soup = BeautifulSoup(html_content, "html.parser")

        # Check title for error indicators
        title = soup.find("title")
        if title:
            title_text = title.get_text().lower()
            error_keywords = [
                "404",
                "not found",
                "error",
                "page not found",
                "access denied",
                "forbidden",
            ]
            if any(keyword in title_text for keyword in error_keywords):
                console.print(
                    f"[yellow]Detected error page by title: {title_text} - {url}[/yellow]"
                )
                return True

        # Check for common error page elements
        error_selectors = [
            "div.error-404",
            "div.not-found",
            "div.error-page",
            'h1:contains("404")',
            'h1:contains("Not Found")',
        ]

        for selector in error_selectors:
            try:
                if soup.select(selector):
                    console.print(
                        f"[yellow]Detected error page by selector: {selector} - {url}[/yellow]"
                    )
                    return True
            except:
                pass

        # Check body text for error indicators
        body_text = soup.get_text().lower()
        if len(body_text) < 200:  # Very short pages are often errors
            error_phrases = [
                "404",
                "page not found",
                "resource not found",
                "does not exist",
            ]
            if any(phrase in body_text for phrase in error_phrases):
                console.print(
                    f"[yellow]Detected error page by content - {url}[/yellow]"
                )
                return True

        return False


class BaseCollector(ABC):
    def __init__(self, config: dict[str, Any], production_config: dict[str, Any]):
        self.config = config
        self.production_config = production_config
        self.collection_settings = config.get("collection_settings", {})
        self.rate_limiter = RateLimiter(
            production_config.get("collection", {})
            .get("rate_limits", {})
            .get("requests_per_minute", 30)
        )
        self._session = None
        self._connector = None
        self.url_validator = URLValidator()

    @abstractmethod
    async def collect(self, source: dict[str, Any]) -> list[ContentEntry]:
        """Collect content from the source"""
        pass

    async def _save_attachment(
        self, content: bytes, filename: str, source_name: str
    ) -> Path:
        """Save attachment to disk"""
        source_dir = Path("config/real_data/attachments") / source_name.replace(
            " ", "_"
        )
        source_dir.mkdir(parents=True, exist_ok=True)

        file_path = source_dir / filename
        if file_path.exists():
            base, ext = os.path.splitext(filename)
            counter = 1
            while file_path.exists():
                file_path = source_dir / f"{base}_{counter}{ext}"
                counter += 1

        with open(file_path, "wb") as f:
            f.write(content)

        return file_path

    async def _save_markdown_content(
        self, entry: ContentEntry, markdown_dir: Path
    ) -> Path:
        """Save content as markdown file"""
        # Create category-specific directory
        category_dir = markdown_dir / entry.categorization["primary_category"]
        source_dir = category_dir / entry.source["name"].replace(" ", "_").replace(
            "/", "_"
        )
        source_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from title
        safe_title = re.sub(r"[^\w\s-]", "", entry.title)
        safe_title = re.sub(r"[-\s]+", "-", safe_title)
        safe_title = safe_title[:100]  # Limit length

        filename = f"{safe_title}_{entry.id[:8]}.md"
        file_path = source_dir / filename

        # Create markdown content with metadata header
        markdown_content = f"""---
title: {entry.title}
url: {entry.url}
source: {entry.source["name"]}
category: {entry.categorization["primary_category"]}
collected_at: {entry.collection_time}
---

# {entry.title}

**Source:** [{entry.source["name"]}]({entry.url})  
**Category:** {entry.categorization["primary_category"]}  
**Collected:** {entry.collection_time}

---

{entry.content}
"""

        # Add attachment links if any
        if entry.attachments:
            markdown_content += "\n\n## Attachments\n\n"
            for attachment in entry.attachments:
                markdown_content += (
                    f"- [{attachment['filename']}]({attachment['path']})\n"
                )

        # Save the markdown file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return file_path

    def _generate_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def _extract_text_from_html(
        self, html_content: str, selector: str | None = None
    ) -> str:
        """Extract text from HTML using markitdown"""
        import re
        import tempfile

        try:
            # If selector is provided, extract specific content first
            if selector:
                soup = BeautifulSoup(html_content, "html.parser")
                element = soup.select_one(selector)
                if element:
                    html_content = str(element)

            # Create MarkItDown instance
            md = MarkItDown()

            # Create a temporary HTML file for markitdown to process
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", delete=False
            ) as temp_file:
                temp_file.write(html_content)
                temp_file_path = temp_file.name

            try:
                # Convert to markdown
                result = md.convert(temp_file_path)
                text = result.text_content

                # Basic cleanup - remove excessive whitespace
                if text:
                    # Remove multiple consecutive empty lines
                    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
                    text = text.strip()

                return text or ""

            finally:
                # Always clean up the temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

        except Exception:
            # Fallback to basic BeautifulSoup text extraction
            try:
                soup = BeautifulSoup(html_content, "html.parser")
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                # Get text and clean it up
                text = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                text = "\n".join(chunk for chunk in chunks if chunk)
                return text
            except Exception:
                return ""

    async def _get_session(self):
        """Get or create shared aiohttp session with connection pooling"""
        if self._session is None or self._session.closed:
            self._connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(connector=self._connector)
        return self._session

    async def _close_session(self):
        """Close the shared session and connector"""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._connector and not self._connector.closed:
            await self._connector.close()

    def _extract_title_from_html(self, html_content: str) -> str:
        """Extract title from HTML"""
        soup = BeautifulSoup(html_content, "html.parser")

        title = None

        # Try h1 tag first
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text().strip()

        # Try title tag
        if not title:
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text().strip()

        if not title:
            og_title = soup.find("meta", property="og:title")
            if og_title:
                title = og_title.get("content", "").strip()

        return title or "Untitled Document"

    async def _fetch_url(
        self, session: aiohttp.ClientSession, url: str
    ) -> str | tuple[bytes, str, str] | None:
        """Fetch content from URL with error handling"""
        # Validate URL before fetching
        if not self.url_validator.is_valid_documentation_url(url):
            return None

        try:
            await self.rate_limiter.wait_if_needed()

            timeout_config = self.production_config.get("collection", {}).get(
                "timeout_settings", {}
            )
            default_timeout = timeout_config.get("default_timeout_seconds", 10)
            pdf_timeout = timeout_config.get("pdf_timeout_seconds", 60)
            large_file_timeout = timeout_config.get("large_file_timeout_seconds", 30)

            is_likely_pdf = any(url.lower().endswith(ext) for ext in [".pdf", ".PDF"])
            is_likely_large_file = any(
                url.lower().endswith(ext)
                for ext in [".pdf", ".PDF", ".ppt", ".pptx", ".doc", ".docx"]
            )

            if is_likely_pdf:
                timeout = aiohttp.ClientTimeout(total=pdf_timeout)
            elif is_likely_large_file:
                timeout = aiohttp.ClientTimeout(total=large_file_timeout)
            else:
                timeout = aiohttp.ClientTimeout(total=default_timeout)

            headers = {
                "User-Agent": self.collection_settings.get(
                    "user_agent", "Mozilla/5.0 (compatible; ConfluenceGatewayBot/1.0)"
                ),
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }

            # First, do a HEAD request to check content type and size
            try:
                async with session.head(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
                ) as head_response:
                    if head_response.status == 200:
                        content_type = head_response.headers.get("Content-Type", "")
                        content_length = head_response.headers.get("Content-Length", "")

                        if "pdf" in content_type.lower() and not is_likely_pdf:
                            timeout = aiohttp.ClientTimeout(total=pdf_timeout)

                        if content_length:
                            try:
                                file_size_mb = int(content_length) / (1024 * 1024)
                                very_large_threshold = timeout_config.get(
                                    "very_large_pdf_threshold_mb", 50
                                )

                                if file_size_mb > very_large_threshold:
                                    extended_timeout = min(300, pdf_timeout * 3)
                                    timeout = aiohttp.ClientTimeout(
                                        total=extended_timeout
                                    )
                            except (ValueError, TypeError):
                                pass
            except:
                pass

            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    content_type = response.headers.get("Content-Type", "")

                    # Check if it's a downloadable file
                    if any(
                        ct in content_type.lower()
                        for ct in [
                            "application/pdf",
                            "application/msword",
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            "application/vnd.ms-excel",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            "application/vnd.ms-powerpoint",
                            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        ]
                    ):
                        content_length = response.headers.get("Content-Length")
                        file_size_mb = 0
                        if content_length:
                            try:
                                file_size_mb = int(content_length) / (1024 * 1024)
                                console.print(
                                    f"[blue]Downloading {content_type} file: {file_size_mb:.1f}MB[/blue]"
                                )
                            except (ValueError, TypeError):
                                pass

                        # Download as attachment
                        if file_size_mb > 50:
                            chunks = []
                            async for chunk in response.content.iter_chunked(
                                1024 * 1024
                            ):
                                chunks.append(chunk)
                            content = b"".join(chunks)
                        else:
                            content = await response.read()

                        filename = url.split("/")[-1]
                        if not any(
                            filename.endswith(ext)
                            for ext in [
                                ".pdf",
                                ".doc",
                                ".docx",
                                ".xls",
                                ".xlsx",
                                ".ppt",
                                ".pptx",
                            ]
                        ):
                            ext_map = {
                                "application/pdf": ".pdf",
                                "application/msword": ".doc",
                                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                                "application/vnd.ms-excel": ".xls",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                                "application/vnd.ms-powerpoint": ".ppt",
                                "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
                            }
                            for ct, ext in ext_map.items():
                                if ct in content_type:
                                    filename = f"document_{hashlib.md5(url.encode()).hexdigest()[:8]}{ext}"
                                    break

                        console.print(
                            f"[blue]Downloaded attachment: {filename} ({len(content)} bytes)[/blue]"
                        )
                        return (content, filename, content_type)
                    else:
                        html_content = await response.text()

                        # Check if it's an error page
                        if self.url_validator.is_error_page_content(html_content, url):
                            return None

                        return html_content
                else:
                    console.print(
                        f"[yellow]Failed to fetch {url}: Status {response.status}[/yellow]"
                    )
                    return None

        except asyncio.TimeoutError:
            console.print(f"[red]Timeout error fetching {url}[/red]")
            return None
        except aiohttp.ClientResponseError as e:
            console.print(
                f"[red]HTTP error fetching {url}: {e.status} - {e.message}[/red]"
            )
            return None
        except aiohttp.ClientConnectionError as e:
            console.print(f"[red]Connection error fetching {url}: {str(e)}[/red]")
            return None
        except aiohttp.ClientPayloadError as e:
            console.print(f"[red]Payload error fetching {url}: {str(e)}[/red]")
            return None
        except Exception as e:
            console.print(
                f"[red]Unexpected error fetching {url}: {type(e).__name__}: {str(e)}[/red]"
            )
            return None

    def _should_follow_link(
        self, url: str, base_url: str, max_depth: int, current_depth: int
    ) -> bool:
        """Check if a link should be followed"""
        if current_depth >= max_depth:
            return False

        # Only follow links within the same domain
        url_domain = urlparse(url).netloc
        base_domain = urlparse(base_url).netloc

        if url_domain != base_domain:
            return False

        # Additional validation
        return self.url_validator.is_valid_documentation_url(url)


class WebDocumentationCollector(BaseCollector):
    async def collect(
        self,
        source: dict[str, Any],
        max_docs_per_source: int = 20,
        progress_callback=None,
        markdown_dir: Path = None,
    ) -> list[ContentEntry]:
        """Collect documentation from web pages with concurrent processing"""
        entries = []
        visited_urls = set()
        url_queue = asyncio.Queue()
        attachment_count = 0

        # Add all start URLs to the queue with depth 0
        for start_url in source.get("start_urls", []):
            if self.url_validator.is_valid_documentation_url(start_url):
                await url_queue.put((start_url, 0))

        concurrent_limit = (
            self.production_config.get("collection", {})
            .get("rate_limits", {})
            .get("concurrent_requests", 20)
        )
        semaphore = asyncio.Semaphore(concurrent_limit)
        entries_lock = asyncio.Lock()

        async with await self._get_session() as session:
            num_workers = min(concurrent_limit * 2, 40)
            workers = []
            for _ in range(num_workers):
                worker = asyncio.create_task(
                    self._url_worker(
                        session,
                        source,
                        url_queue,
                        entries,
                        visited_urls,
                        semaphore,
                        max_docs_per_source,
                        progress_callback,
                        markdown_dir,
                        entries_lock,
                    )
                )
                workers.append(worker)

            await url_queue.join()

            for worker in workers:
                worker.cancel()

            await asyncio.gather(*workers, return_exceptions=True)

        console.print(
            f"[green]✓ Collected {len(entries)} documents from {source['name']}[/green]"
        )

        return entries

    async def _url_worker(
        self,
        session: aiohttp.ClientSession,
        source: dict[str, Any],
        url_queue: asyncio.Queue,
        entries: list[ContentEntry],
        visited_urls: set[str],
        semaphore: asyncio.Semaphore,
        max_docs_per_source: int,
        progress_callback=None,
        markdown_dir: Path = None,
        entries_lock: asyncio.Lock = None,
    ):
        """Worker to process URLs from the queue"""
        while True:
            try:
                url, depth = await asyncio.wait_for(url_queue.get(), timeout=0.5)

                # Check if we've reached the max documents per source with lock
                if entries_lock:
                    async with entries_lock:
                        if len(entries) >= max_docs_per_source:
                            url_queue.task_done()
                            continue
                else:
                    if len(entries) >= max_docs_per_source:
                        url_queue.task_done()
                        continue

                async with semaphore:
                    await self._collect_page(
                        session,
                        source,
                        url,
                        depth,
                        entries,
                        visited_urls,
                        url_queue,
                        max_docs_per_source,
                        progress_callback,
                        markdown_dir,
                        entries_lock,
                    )

                url_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                console.print(f"[red]Worker error: {e}[/red]")
                url_queue.task_done()

    async def _collect_page(
        self,
        session: aiohttp.ClientSession,
        source: dict[str, Any],
        url: str,
        depth: int,
        entries: list[ContentEntry],
        visited_urls: set[str],
        url_queue: asyncio.Queue,
        max_docs_per_source: int,
        progress_callback=None,
        markdown_dir: Path = None,
        entries_lock: asyncio.Lock = None,
    ):
        """Collect a single page and add discovered links to the queue"""
        if url in visited_urls:
            return

        visited_urls.add(url)

        max_depth = source.get("max_depth", 2)
        if depth > max_depth:
            return

        try:
            result = await self._fetch_url(session, url)
            if not result:
                return

            # Check if it's an attachment
            if isinstance(result, tuple):
                content_bytes, filename, content_type = result
                attachment_path = await self._save_attachment(
                    content_bytes, filename, source["name"]
                )

                entry = ContentEntry(
                    id=hashlib.sha256(url.encode()).hexdigest()[:16],
                    url=url,
                    title=f"Attachment: {filename}",
                    content=f"This is an attachment file: {filename}\nType: {content_type}\nSize: {len(content_bytes)} bytes",
                    content_hash=hashlib.sha256(content_bytes).hexdigest(),
                    source={
                        "name": source["name"],
                        "type": source["type"],
                        "base_url": source["base_url"],
                    },
                    metadata={
                        "description": f"Attachment from {source['name']}",
                        "author": source["name"],
                        "language": "en",
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    },
                    categorization={
                        "primary_category": source["categories"][0],
                        "categories": source["categories"],
                    },
                    content_info={
                        "format": "attachment",
                        "attachment_type": content_type,
                        "file_size": len(content_bytes),
                    },
                    collection_time=datetime.now(timezone.utc).isoformat(),
                    attachments=[
                        {
                            "filename": filename,
                            "path": str(attachment_path),
                            "content_type": content_type,
                            "size": len(content_bytes),
                            "url": url,
                        }
                    ],
                )

                # Save as markdown if directory provided
                if markdown_dir:
                    markdown_path = await self._save_markdown_content(
                        entry, markdown_dir
                    )
                    entry.markdown_path = str(markdown_path)

                # Add entry with lock if provided
                if entries_lock:
                    async with entries_lock:
                        entries.append(entry)
                else:
                    entries.append(entry)
                return

            html_content = result
        except Exception as e:
            console.print(f"[yellow]Skipping {url} due to error: {e}[/yellow]")
            return

        # Extract content
        selector = source.get("selector", "main")
        text_content = self._extract_text_from_html(html_content, selector)

        # Check content length
        min_length = (
            self.production_config.get("collection", {})
            .get("content_filters", {})
            .get("min_length", 500)
        )
        max_length = (
            self.production_config.get("collection", {})
            .get("content_filters", {})
            .get("max_length", 100000)
        )

        if len(text_content) < min_length or len(text_content) > max_length:
            return

        # Extract title
        title = self._extract_title_from_html(html_content)

        # Create entry
        content_hash = self._generate_content_hash(text_content)
        entry = ContentEntry(
            id=hashlib.sha256(url.encode()).hexdigest()[:16],
            url=url,
            title=title,
            content=text_content,
            content_hash=content_hash,
            source={
                "name": source["name"],
                "type": source["type"],
                "base_url": source["base_url"],
            },
            metadata={
                "description": f"Documentation from {source['name']}",
                "author": source["name"],
                "language": "en",
                "collected_at": datetime.now(timezone.utc).isoformat(),
            },
            categorization={
                "primary_category": source["categories"][0],
                "categories": source["categories"],
            },
            content_info={
                "format": "html",
                "word_count": len(text_content.split()),
                "char_count": len(text_content),
            },
            collection_time=datetime.now(timezone.utc).isoformat(),
            attachments=[],
        )

        # Extract and queue links if not at max depth
        if depth < max_depth:
            soup = BeautifulSoup(html_content, "html.parser")
            links = soup.find_all("a", href=True)

            attachments_found = []

            for link in links:
                href = link["href"]
                absolute_url = urljoin(url, href)

                # Check if it's a document link
                if any(
                    absolute_url.lower().endswith(ext)
                    for ext in [
                        ".pdf",
                        ".doc",
                        ".docx",
                        ".xls",
                        ".xlsx",
                        ".ppt",
                        ".pptx",
                    ]
                ):
                    if absolute_url not in visited_urls:
                        # Apply rate limiting for attachment downloads
                        await self.rate_limiter.wait_if_needed()
                        attachment_result = await self._fetch_url(session, absolute_url)
                        if isinstance(attachment_result, tuple):
                            content_bytes, filename, content_type = attachment_result
                            attachment_path = await self._save_attachment(
                                content_bytes, filename, source["name"]
                            )
                            attachments_found.append(
                                {
                                    "filename": filename,
                                    "path": str(attachment_path),
                                    "content_type": content_type,
                                    "size": len(content_bytes),
                                    "url": absolute_url,
                                }
                            )
                            visited_urls.add(absolute_url)

                # Check if we should follow this link
                elif self._should_follow_link(
                    absolute_url, source["base_url"], max_depth, depth
                ):
                    if absolute_url not in visited_urls:
                        await url_queue.put((absolute_url, depth + 1))

            # Update entry with found attachments
            if attachments_found:
                entry.attachments.extend(attachments_found)

        # Save as markdown if directory provided
        if markdown_dir:
            markdown_path = await self._save_markdown_content(entry, markdown_dir)
            entry.markdown_path = str(markdown_path)

        # Add entry with lock if provided
        if entries_lock:
            async with entries_lock:
                entries.append(entry)
        else:
            entries.append(entry)

        # Update progress if callback provided
        if progress_callback:
            progress_percent = min(
                90, int((len(entries) / max(max_docs_per_source, 1)) * 90)
            )
            await progress_callback(progress_percent, f"Collected {len(entries)} docs")


class APIDocumentationCollector(BaseCollector):
    async def collect(
        self,
        source: dict[str, Any],
        max_docs_per_source: int = 20,
        progress_callback=None,
        markdown_dir: Path = None,
    ) -> list[ContentEntry]:
        """Collect API documentation"""
        if source.get("use_api"):
            return await self._collect_via_api(
                source, max_docs_per_source, markdown_dir
            )
        else:
            collector = WebDocumentationCollector(self.config, self.production_config)
            return await collector.collect(
                source, max_docs_per_source, progress_callback, markdown_dir
            )

    async def _collect_via_api(
        self,
        source: dict[str, Any],
        max_docs_per_source: int = 20,
        markdown_dir: Path = None,
    ) -> list[ContentEntry]:
        entries = []

        # API collection implementation (simplified for brevity)
        console.print(
            f"[green]✓ Collected {len(entries)} documents via API from {source['name']}[/green]"
        )
        return entries


class ImprovedRealDataCollector:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.cache_dir = config_path / "real_data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create markdown directory
        self.markdown_dir = self.cache_dir / "markdown_content"
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        # Create attachments directory
        self.attachments_dir = self.cache_dir / "attachments"
        self.attachments_dir.mkdir(parents=True, exist_ok=True)

        sources_file = config_path / "real_data" / "sources.yaml"
        with open(sources_file) as f:
            self.sources_config = yaml.safe_load(f)

        with open(config_path / "production_config.yaml") as f:
            self.production_config = yaml.safe_load(f)

        dup_config = self.production_config.get("collection", {}).get(
            "duplicate_detection", {}
        )
        self.duplicate_detector = DuplicateDetector(
            method=dup_config.get("method", "content_hash"),
            threshold=dup_config.get("similarity_threshold", 0.95),
        )

        self.shared_rate_limiter = RateLimiter(
            self.production_config.get("collection", {})
            .get("rate_limits", {})
            .get("requests_per_minute", 120)
        )

        self.index_path = self.cache_dir / "content_index.json"
        self.load_index()

        self._entries_lock = asyncio.Lock()
        self._duplicate_lock = asyncio.Lock()

        self._session = None
        self._connector = None

        # Initialize category counts for efficiency
        self._category_counts = defaultdict(int)
        for entry in self.index.get("entries", []):
            cat = entry["categorization"]["primary_category"]
            self._category_counts[cat] += 1

    def load_index(self):
        if self.index_path.exists():
            with open(self.index_path) as f:
                self.index = json.load(f)
        else:
            self.index = {
                "version": "3.0.0",  # Updated version for markdown support
                "last_updated": None,
                "total_entries": 0,
                "categories": {"api_docs": [], "technical": [], "knowledge_base": []},
                "sources": {},
                "entries": [],
                "markdown_files": [],  # Track markdown files
            }

    def clear_existing_data(self):
        """Clear all existing collected data"""
        console.print("[yellow]Clearing existing data...[/yellow]")

        self.index = {
            "version": "3.0.0",
            "last_updated": None,
            "total_entries": 0,
            "categories": {"api_docs": [], "technical": [], "knowledge_base": []},
            "sources": {},
            "entries": [],
            "markdown_files": [],
        }

        self.duplicate_detector.seen_hashes.clear()
        self.duplicate_detector.seen_urls.clear()

        # Clear category counts
        self._category_counts.clear()

        if self.index_path.exists():
            self.index_path.unlink()

        # Clear markdown directory
        import shutil

        if self.markdown_dir.exists():
            try:
                shutil.rmtree(self.markdown_dir)
            except PermissionError:
                # If rmtree fails, try removing files individually
                console.print(
                    "[yellow]Permission error with rmtree, trying alternative cleanup...[/yellow]"
                )
                import os

                for root, dirs, files in os.walk(self.markdown_dir, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception as e:
                            console.print(
                                f"[yellow]Could not remove file {name}: {e}[/yellow]"
                            )
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception as e:
                            console.print(
                                f"[yellow]Could not remove directory {name}: {e}[/yellow]"
                            )
                try:
                    os.rmdir(self.markdown_dir)
                except:
                    pass
            self.markdown_dir.mkdir(parents=True, exist_ok=True)

        # Clear attachments directory
        if self.attachments_dir.exists():
            try:
                shutil.rmtree(self.attachments_dir)
            except PermissionError:
                console.print(
                    "[yellow]Permission error clearing attachments, trying alternative cleanup...[/yellow]"
                )
                import os

                for root, dirs, files in os.walk(self.attachments_dir, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception as e:
                            console.print(
                                f"[yellow]Could not remove attachment {name}: {e}[/yellow]"
                            )
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception as e:
                            console.print(
                                f"[yellow]Could not remove attachment directory {name}: {e}[/yellow]"
                            )
                try:
                    os.rmdir(self.attachments_dir)
                except:
                    pass
            self.attachments_dir.mkdir(parents=True, exist_ok=True)

        console.print(
            "[green]✓ Existing data, markdown files, and attachments cleared[/green]"
        )

    async def _get_session(self):
        """Get or create shared aiohttp session with connection pooling"""
        if self._session is None or self._session.closed:
            self._connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(connector=self._connector)
        return self._session

    async def _close_session(self):
        """Close the shared session and connector"""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._connector and not self._connector.closed:
            await self._connector.close()

    def save_index(self):
        self.index["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.index["total_entries"] = len(self.index["entries"])

        console.print(
            f"[blue]Saving {len(self.index['entries'])} entries to {self.index_path}[/blue]"
        )
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)
        console.print("[green]✓ Index saved successfully[/green]")

    async def collect_all(self, clear_existing: bool = True):
        console.print(
            "[bold blue]Starting Improved Real Data Collection (with URL validation and Markdown output)[/bold blue]\n"
        )

        if clear_existing:
            self.clear_existing_data()

        enabled_sources = self.production_config.get("collection", {}).get(
            "enabled_sources", []
        )
        targets = self.production_config.get("collection", {}).get("targets", {})

        category_counts = defaultdict(int)
        source_progress = {}

        iteration = 0
        while True:
            iteration += 1
            console.print(f"\n[blue]Collection iteration {iteration}[/blue]")

            # Use the cached category counts
            current_counts = dict(self._category_counts)

            all_satisfied = True
            for cat, target in targets.items():
                min_docs = target.get("min_documents", 20)
                max_docs = target.get("max_documents", 50)
                current = current_counts.get(cat, 0)

                if current < min_docs:
                    all_satisfied = False
                    console.print(
                        f"[yellow]{cat}: {current}/{min_docs} documents (min: {min_docs}, max: {max_docs})[/yellow]"
                    )
                elif current >= max_docs:
                    console.print(
                        f"[green]{cat}: {current} documents (REACHED MAX: {max_docs})[/green]"
                    )
                else:
                    console.print(
                        f"[blue]{cat}: {current} documents (min: {min_docs}, max: {max_docs})[/blue]"
                    )

            if all_satisfied:
                console.print(
                    "\n[green]All categories have reached minimum document requirements![/green]"
                )
                break

            all_sources = []

            for source_type in enabled_sources:
                if source_type in self.sources_config["sources"]:
                    sources = self.sources_config["sources"][source_type]
                    for source in sources:
                        primary_cat = source["categories"][0]
                        current = current_counts.get(primary_cat, 0)
                        max_docs = targets.get(primary_cat, {}).get("max_documents", 50)

                        if current >= max_docs:
                            console.print(
                                f"[dim]Skipping sources for {primary_cat} - already at maximum ({max_docs} docs)[/dim]"
                            )
                            continue

                        all_sources.append((source_type, source))

            random.shuffle(all_sources)
            console.print(
                f"\n[blue]Randomized {len(all_sources)} sources for collection[/blue]"
            )

            if not all_sources:
                console.print("[red]No more sources available to collect from[/red]")
                break

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                main_task = progress.add_task(
                    "Overall Progress", total=len(all_sources)
                )

                for source_type, source in all_sources:
                    task_id = progress.add_task(
                        f"[cyan]{source['name']}[/cyan]", total=100, visible=False
                    )
                    source_progress[source["name"]] = task_id

                concurrent_sources = (
                    self.production_config.get("collection", {})
                    .get("rate_limits", {})
                    .get("concurrent_sources", 8)
                )

                source_queue = asyncio.Queue()
                for item in all_sources:
                    await source_queue.put(item)

                results = []
                completed_count = 0

                async def process_source_worker():
                    """Worker that continuously processes sources from the queue"""
                    nonlocal completed_count
                    while True:
                        try:
                            source_type, source = await source_queue.get()

                            progress.update(
                                source_progress[source["name"]], visible=True
                            )

                            primary_cat = source["categories"][0]
                            current = self._category_counts.get(primary_cat, 0)
                            min_needed = targets.get(primary_cat, {}).get(
                                "min_documents", 20
                            )
                            max_allowed = targets.get(primary_cat, {}).get(
                                "max_documents", 50
                            )

                            remaining_to_min = max(0, min_needed - current)
                            remaining_to_max = max(0, max_allowed - current)

                            # Limit to 40 documents per source to prevent domain bias
                            max_docs_per_source = min(40, remaining_to_max)

                            if max_docs_per_source <= 0:
                                console.print(
                                    f"[dim]Skipping {source['name']} - category {primary_cat} at maximum[/dim]"
                                )
                                source_queue.task_done()
                                completed_count += 1
                                progress.update(main_task, completed=completed_count)
                                continue

                            result = await self._collect_source_with_progress(
                                source_type,
                                source,
                                category_counts,
                                targets,
                                progress,
                                source_progress[source["name"]],
                                max_docs_per_source,
                            )
                            results.append(result)

                            progress.update(
                                source_progress[source["name"]], visible=False
                            )

                            completed_count += 1
                            progress.update(main_task, completed=completed_count)

                            source_queue.task_done()

                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            console.print(f"[red]Worker error: {e}[/red]")
                            source_queue.task_done()

                workers = [
                    asyncio.create_task(process_source_worker())
                    for _ in range(concurrent_sources)
                ]

                await source_queue.join()

                for worker in workers:
                    worker.cancel()

                await asyncio.gather(*workers, return_exceptions=True)

            self.save_index()

        await self._close_session()

        self._display_summary(dict(current_counts))

    async def _collect_source_with_progress(
        self,
        source_type: str,
        source: dict[str, Any],
        category_counts: dict[str, int],
        targets: dict[str, Any],
        progress: Progress,
        task_id: TaskID,
        max_docs_per_source: int = 20,
    ) -> int:
        """Collect from a single source with progress tracking"""
        try:
            progress.update(
                task_id,
                description=f"[cyan]{source['name']}[/cyan] - Starting...",
                completed=0,
            )

            if source["type"] == "web_api_docs":
                collector = APIDocumentationCollector(
                    self.sources_config, self.production_config
                )
            else:
                collector = WebDocumentationCollector(
                    self.sources_config, self.production_config
                )

            collector.rate_limiter = self.shared_rate_limiter

            progress.update(
                task_id,
                description=f"[cyan]{source['name']}[/cyan] - Collecting...",
                completed=10,
            )

            async def update_progress(percent, msg):
                progress.update(
                    task_id,
                    description=f"[cyan]{source['name']}[/cyan] - {msg}",
                    completed=10 + int(percent * 0.8),
                )

            # Collect entries with markdown directory
            entries = await collector.collect(
                source, max_docs_per_source, update_progress, self.markdown_dir
            )

            progress.update(
                task_id,
                description=f"[cyan]{source['name']}[/cyan] - Processing {len(entries)} docs...",
                completed=90,
            )

            new_entries = 0
            skipped_due_to_max = False
            async with self._entries_lock:
                for i, entry in enumerate(entries):
                    primary_cat = entry.categorization["primary_category"]

                    if primary_cat in targets:
                        max_docs = targets[primary_cat].get("max_documents", 50)
                        current_cat_count = self._category_counts.get(primary_cat, 0)

                        if current_cat_count >= max_docs:
                            console.print(
                                f"[yellow]Category {primary_cat} already at max ({max_docs}), skipping remaining entries from {source['name']}[/yellow]"
                            )
                            skipped_due_to_max = True
                            break

                    async with self._duplicate_lock:
                        if not self.duplicate_detector.is_duplicate(entry):
                            entry_dict = entry.model_dump()
                            self.index["entries"].append(entry_dict)

                            # Track markdown file if saved
                            if entry.markdown_path:
                                self.index["markdown_files"].append(
                                    {
                                        "entry_id": entry.id,
                                        "path": entry.markdown_path,
                                        "category": primary_cat,
                                    }
                                )

                            self.duplicate_detector.add_entry(entry)
                            new_entries += 1

                            # Update both local and instance category counts
                            category_counts[primary_cat] = (
                                category_counts.get(primary_cat, 0) + 1
                            )
                            self._category_counts[primary_cat] += 1

                            progress.update(
                                task_id,
                                completed=90 + int((i + 1) / max(len(entries), 1) * 10),
                                description=f"[cyan]{source['name']}[/cyan] - Processing... {new_entries} new docs",
                            )

            progress.update(
                task_id,
                completed=100,
                description=f"[green]{source['name']}[/green] - ✓ {new_entries} new docs",
            )
            console.print(
                f"[green]✓ Added {new_entries} new unique documents from {source['name']}[/green]"
            )
            return new_entries

        except Exception as e:
            progress.update(
                task_id,
                completed=100,
                description=f"[red]{source['name']}[/red] - ✗ Error",
            )
            console.print(f"[red]✗ Error collecting from {source['name']}: {e}[/red]")
            return 0
        finally:
            # Close collector session if exists
            if hasattr(collector, "_session") and collector._session:
                await collector._close_session()

    def _display_summary(self, category_counts: dict[str, int]):
        table = Table(title="Collection Summary")
        table.add_column("Category", style="cyan")
        table.add_column("Documents Collected", justify="right", style="green")
        table.add_column("Markdown Files", justify="right", style="blue")
        table.add_column("Target Range", justify="right")
        table.add_column("Status", justify="center")

        targets = self.production_config.get("collection", {}).get("targets", {})

        # Count markdown files per category
        markdown_counts = defaultdict(int)
        for md_file in self.index.get("markdown_files", []):
            markdown_counts[md_file["category"]] += 1

        for category, count in category_counts.items():
            if category in targets:
                target = targets[category]
                min_docs = target.get("min_documents", 20)
                max_docs = target.get("max_documents", 50)
                target_range = f"{min_docs}-{max_docs}"

                if count >= min_docs:
                    status = "[green]✓ Complete[/green]"
                else:
                    status = f"[red]✗ Need {min_docs - count} more[/red]"
            else:
                target_range = "N/A"
                status = "[yellow]No target[/yellow]"

            table.add_row(
                category,
                str(count),
                str(markdown_counts[category]),
                target_range,
                status,
            )

        console.print("\n")
        console.print(table)
        console.print(
            f"\n[bold green]Total unique documents collected: {len(self.index['entries'])}[/bold green]"
        )
        console.print(
            f"[bold blue]Total markdown files created: {len(self.index.get('markdown_files', []))}[/bold blue]"
        )


def main():
    import typer

    app = typer.Typer(
        help="Collect real documentation from live sources with improved URL validation and Markdown output"
    )

    @app.command()
    def collect(
        config_dir: Path = typer.Option(
            Path("config"), "--config-dir", "-c", help="Configuration directory"
        ),
        no_clear: bool = typer.Option(
            False, "--no-clear", help="Do not clear existing data before collection"
        ),
    ):
        """Collect documentation from configured sources with URL validation and save as Markdown"""
        collector = ImprovedRealDataCollector(config_dir)
        asyncio.run(collector.collect_all(clear_existing=not no_clear))

    @app.command()
    def stats(
        config_dir: Path = typer.Option(
            Path("config"), "--config-dir", "-c", help="Configuration directory"
        ),
    ):
        collector = ImprovedRealDataCollector(config_dir)

        category_counts = {}
        source_counts = {}

        for entry in collector.index["entries"]:
            cat = entry["categorization"]["primary_category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

            source = entry["source"]["name"]
            source_counts[source] = source_counts.get(source, 0) + 1

        console.print("[bold]Collection Statistics[/bold]\n")
        console.print(f"Total documents: {len(collector.index['entries'])}")
        console.print(
            f"Total markdown files: {len(collector.index.get('markdown_files', []))}"
        )
        console.print(f"Last updated: {collector.index.get('last_updated', 'Never')}\n")

        console.print("[bold]By Category:[/bold]")
        for cat, count in category_counts.items():
            console.print(f"  {cat}: {count}")

        console.print("\n[bold]By Source:[/bold]")
        for source, count in sorted(
            source_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            console.print(f"  {source}: {count}")

    app()


if __name__ == "__main__":
    main()
