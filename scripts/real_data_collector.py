#!/usr/bin/env python3

import asyncio
import hashlib
import json
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from urllib.parse import urljoin, urlparse

import aiohttp
import requests
import yaml
from markitdown import MarkItDown
from bs4 import BeautifulSoup
from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

console = Console()


class ContentEntry(BaseModel):
    id: str
    url: str
    title: str
    content: str
    content_hash: str
    source: Dict[str, Any]
    metadata: Dict[str, Any]
    categorization: Dict[str, Any]
    content_info: Dict[str, Any]
    collection_time: str
    

class DuplicateDetector:
    def __init__(self, method: str = "content_hash", threshold: float = 0.95):
        self.method = method
        self.threshold = threshold
        self.seen_hashes: Set[str] = set()
        self.seen_urls: Set[str] = set()
        
    def is_duplicate(self, entry: ContentEntry) -> bool:
        if self.method == "content_hash":
            if entry.content_hash in self.seen_hashes:
                return True
            self.seen_hashes.add(entry.content_hash)
        elif self.method == "url_hash":
            url_hash = hashlib.sha256(entry.url.encode()).hexdigest()
            if url_hash in self.seen_urls:
                return True
            self.seen_urls.add(url_hash)
        return False
    
    def add_entry(self, entry: ContentEntry):
        self.seen_hashes.add(entry.content_hash)
        self.seen_urls.add(entry.url)


class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        
    async def wait_if_needed(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
            
        self.last_request_time = time.time()


class BaseCollector(ABC):
    
    def __init__(self, config: Dict[str, Any], production_config: Dict[str, Any]):
        self.config = config
        self.production_config = production_config
        self.collection_settings = config.get('collection_settings', {})
        self.rate_limiter = RateLimiter(
            production_config.get('collection', {}).get('rate_limits', {}).get('requests_per_minute', 30)
        )
        
    @abstractmethod
    async def collect(self, source: Dict[str, Any]) -> List[ContentEntry]:
        """Collect content from the source"""
        pass
    
    def _generate_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _extract_text_from_html(self, html_content: str, selector: Optional[str] = None) -> str:
        """Extract text from HTML using markitdown"""
        import tempfile
        import re
        
        try:
            # If selector is provided, extract specific content first
            if selector:
                soup = BeautifulSoup(html_content, 'html.parser')
                element = soup.select_one(selector)
                if element:
                    html_content = str(element)
            
            # Create MarkItDown instance
            md = MarkItDown()
            
            # Create a temporary HTML file for markitdown to process
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as temp_file:
                temp_file.write(html_content)
                temp_file_path = temp_file.name
            
            try:
                # Convert to markdown
                result = md.convert(temp_file_path)
                text = result.text_content
                
                # Basic cleanup - remove excessive whitespace
                if text:
                    # Remove multiple consecutive empty lines
                    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
                    text = text.strip()
                
                return text or ""
                
            finally:
                # Always clean up the temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
        except Exception as e:
            # Fallback to basic BeautifulSoup text extraction
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                # Get text and clean it up
                text = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                return text
            except Exception:
                return ""
    
    def _extract_title_from_html(self, html_content: str) -> str:
        """Extract title from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Try different title sources
        title = None
        
        # Try h1 tag first
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text().strip()
            
        # Try title tag
        if not title:
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
                
        if not title:
            og_title = soup.find('meta', property='og:title')
            if og_title:
                title = og_title.get('content', '').strip()
                
        return title or "Untitled Document"
    
    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch content from URL with error handling"""
        try:
            # Skip PDF files
            if url.endswith('.pdf') or '#' in url and url.split('#')[0].endswith('.pdf'):
                console.print(f"[yellow]Skipping PDF file: {url}[/yellow]")
                return None
                
            await self.rate_limiter.wait_if_needed()
            
            timeout = aiohttp.ClientTimeout(total=self.collection_settings.get('timeout_seconds', 30))
            headers = {
                'User-Agent': self.collection_settings.get('user_agent', 
                    'Mozilla/5.0 (compatible; ConfluenceGatewayBot/1.0)')
            }
            
            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    # Check content type
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/pdf' in content_type:
                        console.print(f"[yellow]Skipping PDF content: {url}[/yellow]")
                        return None
                    return await response.text()
                else:
                    console.print(f"[yellow]Failed to fetch {url}: Status {response.status}[/yellow]")
                    return None
                    
        except Exception as e:
            console.print(f"[red]Error fetching {url}: {e}[/red]")
            return None
    
    def _should_follow_link(self, url: str, base_url: str, max_depth: int, current_depth: int) -> bool:
        """Check if a link should be followed"""
        if current_depth >= max_depth:
            return False
            
        # Only follow links within the same domain
        url_domain = urlparse(url).netloc
        base_domain = urlparse(base_url).netloc
        
        return url_domain == base_domain


class WebDocumentationCollector(BaseCollector):
    
    async def collect(self, source: Dict[str, Any]) -> List[ContentEntry]:
        """Collect documentation from web pages"""
        entries = []
        visited_urls = set()
        
        async with aiohttp.ClientSession() as session:
            # Process start URLs
            for start_url in source.get('start_urls', []):
                await self._collect_recursive(
                    session, source, start_url, entries, visited_urls, 0
                )
                # Limit documents per source
                if len(entries) >= 10:
                    break
                
        console.print(f"[green]✓ Collected {len(entries)} documents from {source['name']}[/green]")
        return entries
    
    async def _collect_recursive(self, session: aiohttp.ClientSession, source: Dict[str, Any], 
                                url: str, entries: List[ContentEntry], visited_urls: Set[str], 
                                depth: int):
        if url in visited_urls or len(entries) >= 10:
            return
            
        visited_urls.add(url)
        
        # Check depth limit
        max_depth = source.get('max_depth', 2)
        if depth > max_depth:
            return
            
        # Fetch page content
        try:
            html_content = await self._fetch_url(session, url)
            if not html_content:
                return
        except Exception as e:
            console.print(f"[yellow]Skipping {url} due to error: {e}[/yellow]")
            return
            
        # Extract content
        selector = source.get('selector', 'main')
        text_content = self._extract_text_from_html(html_content, selector)
        
        # Check content length
        min_length = self.production_config.get('collection', {}).get('content_filters', {}).get('min_length', 500)
        max_length = self.production_config.get('collection', {}).get('content_filters', {}).get('max_length', 100000)
        
        if len(text_content) < min_length or len(text_content) > max_length:
            console.print(f"[yellow]Skipping {url}: Content length {len(text_content)} outside range[/yellow]")
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
                "name": source['name'],
                "type": source['type'],
                "base_url": source['base_url']
            },
            metadata={
                "description": f"Documentation from {source['name']}",
                "author": source['name'],
                "language": "en",
                "collected_at": datetime.now(timezone.utc).isoformat()
            },
            categorization={
                "primary_category": source['categories'][0],
                "categories": source['categories']
            },
            content_info={
                "format": "html",
                "word_count": len(text_content.split()),
                "char_count": len(text_content)
            },
            collection_time=datetime.now(timezone.utc).isoformat()
        )
        
        entries.append(entry)
        
        # Extract and follow links if not at max depth
        if depth < max_depth:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link['href']
                absolute_url = urljoin(url, href)
                
                # Check if we should follow this link
                if self._should_follow_link(absolute_url, source['base_url'], max_depth, depth):
                    await self._collect_recursive(
                        session, source, absolute_url, entries, visited_urls, depth + 1
                    )


class APIDocumentationCollector(BaseCollector):
    
    async def collect(self, source: Dict[str, Any]) -> List[ContentEntry]:
        """Collect API documentation"""
        # If source has API endpoint, use that
        if source.get('use_api'):
            return await self._collect_via_api(source)
        else:
            # Otherwise use web scraping
            collector = WebDocumentationCollector(self.config, self.production_config)
            return await collector.collect(source)
    
    async def _collect_via_api(self, source: Dict[str, Any]) -> List[ContentEntry]:
        entries = []
        
        if source['name'].startswith("Dev.to"):
            api_endpoint = source['api_endpoint']
            params = source.get('api_params', {})
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_endpoint, params=params) as response:
                    if response.status == 200:
                        articles = await response.json()
                        
                        for article in articles[:30]:
                            content = article.get('description', '')
                            if not content:
                                content = f"# {article['title']}\n\n{article.get('tags', '')}\n\nRead more at: {article['url']}"
                            
                            entry = ContentEntry(
                                id=str(article['id']),
                                url=article['url'],
                                title=article['title'],
                                content=content,
                                content_hash=self._generate_content_hash(content),
                                source={
                                    "name": source['name'],
                                    "type": "api",
                                    "base_url": source['base_url']
                                },
                                metadata={
                                    "description": article.get('description', ''),
                                    "author": article['user']['username'],
                                    "language": "en",
                                    "tags": article.get('tags', '').split(', '),
                                    "collected_at": datetime.now(timezone.utc).isoformat()
                                },
                                categorization={
                                    "primary_category": source['categories'][0],
                                    "categories": source['categories']
                                },
                                content_info={
                                    "format": "markdown",
                                    "word_count": len(content.split()),
                                    "reading_time_minutes": article.get('reading_time_minutes', 0)
                                },
                                collection_time=datetime.now(timezone.utc).isoformat()
                            )
                            entries.append(entry)
                            
        elif source['name'] == "Stack Overflow Documentation":
            api_endpoint = source['api_endpoint']
            
            async with aiohttp.ClientSession() as session:
                params = {
                    'order': 'desc',
                    'sort': 'votes',
                    'tagged': 'python;javascript;java',
                    'site': 'stackoverflow',
                    'filter': 'withbody'
                }
                
                async with session.get(api_endpoint, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        questions = data.get('items', [])
                        
                        for q in questions[:20]:
                            content = f"# {q['title']}\n\n{q.get('body', '')}"
                            
                            if q.get('accepted_answer_id'):
                                content += "\n\n## Accepted Answer\n\n[Answer content would be fetched separately]"
                            
                            entry = ContentEntry(
                                id=str(q['question_id']),
                                url=q['link'],
                                title=q['title'],
                                content=content,
                                content_hash=self._generate_content_hash(content),
                                source={
                                    "name": source['name'],
                                    "type": "api",
                                    "base_url": source['base_url']
                                },
                                metadata={
                                    "description": f"Stack Overflow Q&A",
                                    "author": q['owner'].get('display_name', 'Anonymous'),
                                    "language": "en",
                                    "tags": q.get('tags', []),
                                    "score": q.get('score', 0),
                                    "collected_at": datetime.now(timezone.utc).isoformat()
                                },
                                categorization={
                                    "primary_category": source['categories'][0],
                                    "categories": source['categories']
                                },
                                content_info={
                                    "format": "html",
                                    "word_count": len(content.split()),
                                    "view_count": q.get('view_count', 0)
                                },
                                collection_time=datetime.now(timezone.utc).isoformat()
                            )
                            entries.append(entry)
        
        console.print(f"[green]✓ Collected {len(entries)} documents via API from {source['name']}[/green]")
        return entries


class RealDataCollector:
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.cache_dir = config_path / 'real_data'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        sources_file = config_path / 'real_data' / 'sources.yaml'
        with open(sources_file, 'r') as f:
            self.sources_config = yaml.safe_load(f)
            
        with open(config_path / 'production_config.yaml', 'r') as f:
            self.production_config = yaml.safe_load(f)
            
        dup_config = self.production_config.get('collection', {}).get('duplicate_detection', {})
        self.duplicate_detector = DuplicateDetector(
            method=dup_config.get('method', 'content_hash'),
            threshold=dup_config.get('similarity_threshold', 0.95)
        )
        
        self.collectors = {
            'web_api_docs': APIDocumentationCollector(self.sources_config, self.production_config),
            'web_tech_docs': WebDocumentationCollector(self.sources_config, self.production_config),
            'web_kb': WebDocumentationCollector(self.sources_config, self.production_config)
        }
        
        self.index_path = self.cache_dir / 'content_index.json'
        self.load_index()
        
    def load_index(self):
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                "version": "2.0.0",
                "last_updated": None,
                "total_entries": 0,
                "categories": {
                    "api_docs": [],
                    "technical": [],
                    "knowledge_base": []
                },
                "sources": {},
                "entries": []
            }
    
    def save_index(self):
        self.index['last_updated'] = datetime.now(timezone.utc).isoformat()
        self.index['total_entries'] = len(self.index['entries'])
        
        console.print(f"[blue]Saving {len(self.index['entries'])} entries to {self.index_path}[/blue]")
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
        console.print(f"[green]✓ Index saved successfully[/green]")
    
    async def collect_all(self):
        console.print("[bold blue]Starting Real Data Collection from Live Sources[/bold blue]\n")
        
        enabled_sources = self.production_config.get('collection', {}).get('enabled_sources', [])
        targets = self.production_config.get('collection', {}).get('targets', {})
        
        category_counts = {
            'api_docs': 0,
            'technical': 0,
            'knowledge_base': 0
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            for source_type in enabled_sources:
                if source_type not in self.sources_config['sources']:
                    continue
                    
                sources = self.sources_config['sources'][source_type]
                task = progress.add_task(f"Processing {source_type}", total=len(sources))
                
                for source in sources:
                    progress.update(task, description=f"Collecting from {source['name']}")
                    
                    collector = self.collectors.get(source['type'])
                    if not collector:
                        console.print(f"[red]No collector for type: {source['type']}[/red]")
                        progress.advance(task)
                        continue
                    
                    try:
                        entries = await collector.collect(source)
                        
                        new_entries = 0
                        for entry in entries:
                            if not self.duplicate_detector.is_duplicate(entry):
                                self.index['entries'].append(entry.model_dump())
                                self.duplicate_detector.add_entry(entry)
                                new_entries += 1
                                
                                primary_cat = entry.categorization['primary_category']
                                if primary_cat in category_counts:
                                    category_counts[primary_cat] += 1
                                    
                                if primary_cat in targets:
                                    max_docs = targets[primary_cat].get('max_documents', 50)
                                    if category_counts[primary_cat] >= max_docs:
                                        console.print(f"[yellow]Reached max documents ({max_docs}) for {primary_cat}[/yellow]")
                                        break
                        
                        console.print(f"[green]✓ Added {new_entries} new unique documents from {source['name']}[/green]")
                        
                    except Exception as e:
                        console.print(f"[red]✗ Error collecting from {source['name']}: {e}[/red]")
                    
                    progress.advance(task)
                    
                    all_satisfied = True
                    for cat, target in targets.items():
                        if category_counts.get(cat, 0) < target.get('min_documents', 20):
                            all_satisfied = False
                            break
                            
                    if all_satisfied:
                        console.print("[yellow]Collected minimum required documents for all categories[/yellow]")
                        break
        
        self.save_index()
        
        self._display_summary(category_counts)
    
    def _display_summary(self, category_counts: Dict[str, int]):
        table = Table(title="Collection Summary")
        table.add_column("Category", style="cyan")
        table.add_column("Documents Collected", justify="right", style="green")
        table.add_column("Target Range", justify="right")
        table.add_column("Status", justify="center")
        
        targets = self.production_config.get('collection', {}).get('targets', {})
        
        for category, count in category_counts.items():
            if category in targets:
                target = targets[category]
                min_docs = target.get('min_documents', 20)
                max_docs = target.get('max_documents', 50)
                target_range = f"{min_docs}-{max_docs}"
                
                if count >= min_docs:
                    status = "[green]✓ Complete[/green]"
                else:
                    status = f"[red]✗ Need {min_docs - count} more[/red]"
            else:
                target_range = "N/A"
                status = "[yellow]No target[/yellow]"
                
            table.add_row(category, str(count), target_range, status)
        
        console.print("\n")
        console.print(table)
        console.print(f"\n[bold green]Total unique documents collected: {len(self.index['entries'])}[/bold green]")
    
    def search_content(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search collected content by category"""
        results = []
        
        for entry in self.index['entries']:
            if category and entry['categorization']['primary_category'] != category:
                continue
            results.append(entry)
            
        return results


def main():
    import typer
    
    app = typer.Typer(help="Collect real documentation from live sources")
    
    @app.command()
    def collect(
        config_dir: Path = typer.Option(
            Path("config"),
            "--config-dir", "-c",
            help="Configuration directory"
        )
    ):
        collector = RealDataCollector(config_dir)
        asyncio.run(collector.collect_all())
    
    @app.command()
    def search(
        config_dir: Path = typer.Option(
            Path("config"),
            "--config-dir", "-c",
            help="Configuration directory"
        ),
        category: Optional[str] = typer.Option(
            None,
            "--category",
            help="Filter by category (api_docs, technical, knowledge_base)"
        )
    ):
        collector = RealDataCollector(config_dir)
        results = collector.search_content(category)
        
        console.print(f"\n[bold]Found {len(results)} documents[/bold]\n")
        
        for entry in results[:10]:
            console.print(f"• [cyan]{entry['title']}[/cyan]")
            console.print(f"  Category: {entry['categorization']['primary_category']}")
            console.print(f"  URL: {entry['url']}")
            console.print(f"  Words: {entry['content_info']['word_count']}")
            console.print()
    
    @app.command()
    def stats(
        config_dir: Path = typer.Option(
            Path("config"),
            "--config-dir", "-c",
            help="Configuration directory"
        )
    ):
        collector = RealDataCollector(config_dir)
        
        # Calculate stats
        category_counts = {}
        source_counts = {}
        
        for entry in collector.index['entries']:
            cat = entry['categorization']['primary_category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            source = entry['source']['name']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        console.print("[bold]Collection Statistics[/bold]\n")
        console.print(f"Total documents: {len(collector.index['entries'])}")
        console.print(f"Last updated: {collector.index.get('last_updated', 'Never')}\n")
        
        console.print("[bold]By Category:[/bold]")
        for cat, count in category_counts.items():
            console.print(f"  {cat}: {count}")
            
        console.print("\n[bold]By Source:[/bold]")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            console.print(f"  {source}: {count}")
    
    app()


if __name__ == "__main__":
    main()