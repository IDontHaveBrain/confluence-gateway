#!/usr/bin/env python3
"""
Real Data Collector for Confluence Gateway Testing

This script collects real technical documentation from various sources
to create realistic test data for Confluence Gateway.
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from pydantic import BaseModel
from rich.console import Console
from rich.progress import track

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

console = Console()
logger = logging.getLogger(__name__)


class ContentEntry(BaseModel):
    """Model for content metadata"""
    id: str
    source: Dict[str, Any]
    metadata: Dict[str, Any]
    categorization: Dict[str, Any]
    content: Dict[str, Any]
    quality_metrics: Dict[str, float]
    attachments: List[Dict[str, Any]] = []
    processing: Dict[str, Any]


class BaseCollector(ABC):
    """Base class for content collectors"""
    
    def __init__(self, config: Dict[str, Any], cache_dir: Path):
        self.config = config
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.get('collection_settings', {}).get(
                'user_agent', 'ConfluenceGatewayTestDataCollector/1.0'
            )
        })
        
    @abstractmethod
    def collect(self, source: Dict[str, Any]) -> List[ContentEntry]:
        """Collect content from the source"""
        pass
    
    def _generate_id(self, source_name: str, path: str) -> str:
        """Generate unique ID for content"""
        content = f"{source_name}:{path}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _calculate_quality_metrics(self, content: str) -> Dict[str, float]:
        """Calculate quality metrics for content"""
        # Simple metrics for now
        word_count = len(content.split())
        char_count = len(content)
        
        # Readability (simple approximation)
        avg_word_length = char_count / max(word_count, 1)
        readability = min(1.0, max(0.0, 1.0 - (avg_word_length - 5) / 10))
        
        # Technical depth (based on code blocks and technical terms)
        code_blocks = len(re.findall(r'```[\s\S]*?```', content))
        technical_terms = len(re.findall(
            r'\b(api|function|class|method|parameter|configuration|algorithm|database|server)\b', 
            content.lower()
        ))
        technical_depth = min(1.0, (code_blocks * 0.1 + technical_terms * 0.01))
        
        # Completeness (based on length)
        completeness = min(1.0, word_count / 1000)
        
        # Overall quality
        overall = (readability + technical_depth + completeness) / 3
        
        return {
            "readability_score": round(readability, 2),
            "technical_depth": round(technical_depth, 2),
            "completeness": round(completeness, 2),
            "overall_quality": round(overall, 2)
        }


class GitHubCollector(BaseCollector):
    """Collector for GitHub repositories"""
    
    def collect(self, source: Dict[str, Any]) -> List[ContentEntry]:
        """Collect documentation from GitHub repository"""
        entries = []
        repo_url = source['url']
        
        # Parse GitHub URL
        parts = urlparse(repo_url).path.strip('/').split('/')
        if len(parts) < 2:
            logger.error(f"Invalid GitHub URL: {repo_url}")
            return entries
            
        owner, repo = parts[0], parts[1]
        path = '/'.join(parts[4:]) if len(parts) > 4 else ''
        
        console.print(f"[yellow]Collecting from GitHub: {owner}/{repo}/{path}[/yellow]")
        
        # Use GitHub API to list files
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        
        try:
            response = self.session.get(api_url)
            response.raise_for_status()
            
            items = response.json()
            if not isinstance(items, list):
                items = [items]
                
            for item in track(items, description=f"Processing {source['name']}"):
                if item['type'] == 'file':
                    # Check file patterns
                    if any(item['name'].endswith(pattern.replace('*', '')) 
                          for pattern in source.get('file_patterns', ['*'])):
                        entry = self._process_github_file(source, owner, repo, item)
                        if entry:
                            entries.append(entry)
                            
                # Rate limiting
                time.sleep(self.config.get('collection_settings', {}).get('rate_limit_seconds', 2))
                
        except Exception as e:
            logger.error(f"Error collecting from GitHub: {e}")
            
        return entries
    
    def _process_github_file(self, source: Dict[str, Any], owner: str, repo: str, 
                            file_info: Dict[str, Any]) -> Optional[ContentEntry]:
        """Process a single GitHub file"""
        try:
            # Get file content
            response = self.session.get(file_info['download_url'])
            response.raise_for_status()
            
            content = response.text
            
            # Skip if too small or too large
            settings = self.config.get('collection_settings', {})
            if (len(content) < settings.get('min_content_length', 500) or
                len(content) > settings.get('max_content_length', 50000)):
                return None
                
            # Generate metadata
            entry_id = self._generate_id(source['name'], file_info['path'])
            
            # Determine format
            file_ext = Path(file_info['name']).suffix.lower()
            format_map = {
                '.md': 'markdown',
                '.rst': 'rst',
                '.txt': 'plain_text',
                '.html': 'html',
                '.ipynb': 'jupyter'
            }
            content_format = format_map.get(file_ext, 'plain_text')
            
            # Save raw content
            raw_path = self.cache_dir / 'raw_content' / f"{entry_id}_{file_info['name']}"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(content, encoding='utf-8')
            
            # Calculate metrics
            metrics = self._calculate_quality_metrics(content)
            
            # Create entry
            entry = ContentEntry(
                id=entry_id,
                source={
                    "name": source['name'],
                    "type": "github_repo",
                    "url": file_info['html_url'],
                    "fetch_date": datetime.now(timezone.utc).isoformat()
                },
                metadata={
                    "title": file_info['name'].replace('-', ' ').replace('_', ' ').title(),
                    "description": f"Documentation from {owner}/{repo}",
                    "author": owner,
                    "created_date": datetime.now(timezone.utc).isoformat(),
                    "modified_date": datetime.now(timezone.utc).isoformat(),
                    "language": source['languages'][0] if source.get('languages') else 'en',
                    "license": "Check repository"
                },
                categorization={
                    "primary_category": source['categories'][0],
                    "secondary_categories": source['categories'][1:],
                    "tags": self._extract_tags(content),
                    "topics": [repo, owner]
                },
                content={
                    "format": content_format,
                    "raw_file_path": str(raw_path.relative_to(self.cache_dir)),
                    "processed_file_path": "",
                    "word_count": len(content.split()),
                    "char_count": len(content),
                    "code_blocks_count": len(re.findall(r'```[\s\S]*?```', content)),
                    "images_count": len(re.findall(r'!\[.*?\]\(.*?\)', content)),
                    "tables_count": len(re.findall(r'\|.*\|.*\|', content))
                },
                quality_metrics=metrics,
                attachments=[],
                processing={
                    "preprocessed": False,
                    "validated": True,
                    "errors": [],
                    "warnings": []
                }
            )
            
            return entry
            
        except Exception as e:
            logger.error(f"Error processing file {file_info['name']}: {e}")
            return None
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract relevant tags from content"""
        # Simple keyword extraction
        keywords = []
        
        # Common technical terms
        tech_terms = ['api', 'database', 'server', 'client', 'function', 'class', 
                     'method', 'configuration', 'installation', 'deployment']
        
        content_lower = content.lower()
        for term in tech_terms:
            if term in content_lower:
                keywords.append(term)
                
        return keywords[:10]  # Limit to 10 tags


class WebScraperCollector(BaseCollector):
    """Collector for web documentation"""
    
    def collect(self, source: Dict[str, Any]) -> List[ContentEntry]:
        """Collect documentation from web pages"""
        entries = []
        base_url = source['base_url']
        max_pages = source.get('max_pages', 10)
        
        console.print(f"[yellow]Collecting from web: {base_url}[/yellow]")
        
        # Start with base URL
        urls_to_visit = [base_url]
        visited_urls = set()
        
        while urls_to_visit and len(entries) < max_pages:
            url = urls_to_visit.pop(0)
            
            if url in visited_urls:
                continue
                
            visited_urls.add(url)
            
            try:
                # Respect robots.txt
                if self.config.get('collection_settings', {}).get('respect_robots_txt', True):
                    # Simple check - in production, use robotparser
                    robots_url = urljoin(url, '/robots.txt')
                    # Skip complex robots.txt parsing for now
                    
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Parse content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract links for crawling
                for link in soup.find_all('a', href=True):
                    href = urljoin(url, link['href'])
                    if (href.startswith(base_url) and 
                        href not in visited_urls and 
                        href not in urls_to_visit):
                        urls_to_visit.append(href)
                
                # Process page content
                entry = self._process_web_page(source, url, response.text)
                if entry:
                    entries.append(entry)
                    
                # Rate limiting
                time.sleep(self.config.get('collection_settings', {}).get('rate_limit_seconds', 2))
                
            except Exception as e:
                logger.error(f"Error collecting from {url}: {e}")
                
        return entries
    
    def _process_web_page(self, source: Dict[str, Any], url: str, 
                         html_content: str) -> Optional[ContentEntry]:
        """Process a single web page"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.text.strip() if title else urlparse(url).path
            
            # Extract main content
            # Try common content containers
            content_tags = ['main', 'article', 'div.content', 'div.documentation']
            content_elem = None
            
            for tag in content_tags:
                if '.' in tag:
                    tag_name, class_name = tag.split('.')
                    content_elem = soup.find(tag_name, class_=class_name)
                else:
                    content_elem = soup.find(tag)
                    
                if content_elem:
                    break
                    
            if not content_elem:
                content_elem = soup.find('body')
                
            if not content_elem:
                return None
                
            # Extract text
            text_content = self._extract_text_from_html(str(content_elem))
            
            # Skip if too small
            settings = self.config.get('collection_settings', {})
            if len(text_content) < settings.get('min_content_length', 500):
                return None
                
            # Generate entry
            entry_id = self._generate_id(source['name'], url)
            
            # Save raw content
            raw_path = self.cache_dir / 'raw_content' / f"{entry_id}.html"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(html_content, encoding='utf-8')
            
            # Calculate metrics
            metrics = self._calculate_quality_metrics(text_content)
            
            # Create entry
            entry = ContentEntry(
                id=entry_id,
                source={
                    "name": source['name'],
                    "type": "web_scrape",
                    "url": url,
                    "fetch_date": datetime.now(timezone.utc).isoformat()
                },
                metadata={
                    "title": title_text,
                    "description": f"Documentation from {urlparse(url).netloc}",
                    "author": urlparse(url).netloc,
                    "created_date": datetime.now(timezone.utc).isoformat(),
                    "modified_date": datetime.now(timezone.utc).isoformat(),
                    "language": source['languages'][0] if source.get('languages') else 'en',
                    "license": "Check website"
                },
                categorization={
                    "primary_category": source['categories'][0],
                    "secondary_categories": source['categories'][1:],
                    "tags": self._extract_tags(text_content),
                    "topics": [urlparse(url).netloc]
                },
                content={
                    "format": "html",
                    "raw_file_path": str(raw_path.relative_to(self.cache_dir)),
                    "processed_file_path": "",
                    "word_count": len(text_content.split()),
                    "char_count": len(text_content),
                    "code_blocks_count": len(soup.find_all(['pre', 'code'])),
                    "images_count": len(soup.find_all('img')),
                    "tables_count": len(soup.find_all('table'))
                },
                quality_metrics=metrics,
                attachments=[],
                processing={
                    "preprocessed": False,
                    "validated": True,
                    "errors": [],
                    "warnings": []
                }
            )
            
            return entry
            
        except Exception as e:
            logger.error(f"Error processing web page {url}: {e}")
            return None
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract relevant tags from content"""
        # Simple keyword extraction
        keywords = []
        
        # Common technical terms
        tech_terms = ['api', 'database', 'server', 'client', 'function', 'class', 
                     'method', 'configuration', 'installation', 'deployment']
        
        content_lower = content.lower()
        for term in tech_terms:
            if term in content_lower:
                keywords.append(term)
                
        return keywords[:10]  # Limit to 10 tags


class RealDataCollector:
    """Main collector that orchestrates all source collectors"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.cache_dir = config_path / 'real_data'
        
        # Load configuration
        with open(config_path / 'real_data' / 'sources.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load existing index
        self.index_path = self.cache_dir / 'content_index.json'
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                "version": "1.0.0",
                "last_updated": None,
                "total_entries": 0,
                "categories": {
                    "technical": 0,
                    "api_docs": 0,
                    "knowledge_base": 0,
                    "project_docs": 0,
                    "multilingual": 0
                },
                "languages": {
                    "en": 0,
                    "ko": 0,
                    "mixed": 0,
                    "other": 0
                },
                "entries": []
            }
            
        # Initialize collectors
        self.collectors = {
            'github_repo': GitHubCollector(self.config, self.cache_dir),
            'web_scrape': WebScraperCollector(self.config, self.cache_dir)
        }
        
    def collect_all(self, source_types: Optional[List[str]] = None):
        """Collect content from all configured sources"""
        console.print("[bold blue]Starting Real Data Collection[/bold blue]\n")
        
        all_entries = []
        
        # Process each source type
        for source_type, sources in self.config['sources'].items():
            if source_types and source_type not in source_types:
                continue
                
            console.print(f"\n[bold]Processing {source_type} sources...[/bold]")
            
            for source in sources:
                collector = self.collectors.get(source['type'])
                if not collector:
                    console.print(f"[red]No collector for type: {source['type']}[/red]")
                    continue
                    
                try:
                    entries = collector.collect(source)
                    all_entries.extend(entries)
                    console.print(f"[green]✓ Collected {len(entries)} entries from {source['name']}[/green]")
                except Exception as e:
                    console.print(f"[red]✗ Error collecting from {source['name']}: {e}[/red]")
                    
        # Update index
        self._update_index(all_entries)
        
        console.print(f"\n[bold green]Collection complete! Total entries: {len(all_entries)}[/bold green]")
        
    def _update_index(self, new_entries: List[ContentEntry]):
        """Update the content index with new entries"""
        # Convert existing entries to dict for easier lookup
        existing_ids = {entry['id'] for entry in self.index['entries']}
        
        # Add new entries
        added_count = 0
        for entry in new_entries:
            if entry.id not in existing_ids:
                entry_dict = entry.model_dump()
                self.index['entries'].append(entry_dict)
                
                # Update counts
                primary_cat = entry.categorization['primary_category']
                if primary_cat in self.index['categories']:
                    self.index['categories'][primary_cat] += 1
                    
                lang = entry.metadata['language']
                if lang in self.index['languages']:
                    self.index['languages'][lang] += 1
                else:
                    self.index['languages']['other'] += 1
                    
                added_count += 1
                
        # Update metadata
        self.index['total_entries'] = len(self.index['entries'])
        self.index['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Save index
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
            
        console.print(f"[green]✓ Added {added_count} new entries to index[/green]")
        
    def search_content(self, category: Optional[str] = None, 
                      language: Optional[str] = None,
                      min_quality: float = 0.5) -> List[Dict[str, Any]]:
        """Search collected content by criteria"""
        results = []
        
        for entry in self.index['entries']:
            # Filter by category
            if category and entry['categorization']['primary_category'] != category:
                continue
                
            # Filter by language
            if language and entry['metadata']['language'] != language:
                continue
                
            # Filter by quality
            if entry['quality_metrics']['overall_quality'] < min_quality:
                continue
                
            results.append(entry)
            
        return results


def main():
    """Main entry point"""
    import typer
    
    app = typer.Typer(help="Collect real documentation for testing")
    
    @app.command()
    def collect(
        config_dir: Path = typer.Option(
            Path("scripts/config"),
            "--config-dir", "-c",
            help="Configuration directory"
        ),
        source_types: Optional[str] = typer.Option(
            None,
            "--sources", "-s",
            help="Comma-separated source types to collect (github,web_docs)"
        )
    ):
        """Collect real documentation from configured sources"""
        collector = RealDataCollector(config_dir)
        
        types = source_types.split(',') if source_types else None
        collector.collect_all(types)
        
    @app.command()
    def search(
        config_dir: Path = typer.Option(
            Path("scripts/config"),
            "--config-dir", "-c",
            help="Configuration directory"
        ),
        category: Optional[str] = typer.Option(
            None,
            "--category",
            help="Filter by category"
        ),
        language: Optional[str] = typer.Option(
            None,
            "--language",
            help="Filter by language"
        ),
        min_quality: float = typer.Option(
            0.5,
            "--min-quality",
            help="Minimum quality score"
        )
    ):
        """Search collected content"""
        collector = RealDataCollector(config_dir)
        results = collector.search_content(category, language, min_quality)
        
        console.print(f"\n[bold]Found {len(results)} entries[/bold]\n")
        
        for entry in results[:10]:  # Show first 10
            console.print(f"• {entry['metadata']['title']}")
            console.print(f"  Category: {entry['categorization']['primary_category']}")
            console.print(f"  Quality: {entry['quality_metrics']['overall_quality']}")
            console.print(f"  URL: {entry['source']['url']}")
            console.print()
            
    app()


if __name__ == "__main__":
    main()