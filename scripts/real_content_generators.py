#!/usr/bin/env python3
"""
Content Generators using Real Data

This module provides content generation using real collected data
instead of dummy/fake content.
"""

import json
import logging
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from confluence_gateway.adapters.confluence.client import ConfluenceClient

logger = logging.getLogger(__name__)


class RealDataContentGenerator:
    """Generate content using real collected data"""
    
    def __init__(self, client: ConfluenceClient, config: Any):
        self.client = client
        self.config = config
        self.config_dir = Path("scripts/config")
        self.real_data_dir = self.config_dir / "real_data"
        
        # Load content index
        self.content_index = self._load_content_index()
        
        # Cache for loaded content
        self._content_cache = {}
        
    def _load_content_index(self) -> Dict[str, Any]:
        """Load the content index"""
        index_path = self.real_data_dir / "content_index.json"
        if not index_path.exists():
            logger.warning("Content index not found. Run real_data_collector.py first.")
            return {
                "entries": [],
                "categories": {},
                "languages": {}
            }
            
        with open(index_path, 'r') as f:
            return json.load(f)
    
    def create_space(self, space_key: str, space_name: str) -> Optional[Dict[str, Any]]:
        """Create a new Confluence space"""
        try:
            # Check if space exists
            existing_spaces = self.client.atlassian_api.get_all_spaces()
            for space in existing_spaces.get('results', []):
                if space.get('key') == space_key:
                    logger.info(f"Space {space_key} already exists")
                    return space
            
            # Create new space
            space = self.client.atlassian_api.create_space(
                space_key=space_key,
                space_name=space_name
            )
            
            logger.info(f"Created space: {space_key}")
            return space
            
        except Exception as e:
            logger.error(f"Error creating space {space_key}: {e}")
            return None
    
    def generate_page_content(self, category: str, space_type: str) -> Dict[str, Any]:
        """Generate page content using real data"""
        # Find suitable content from index
        all_entries = self.content_index.get('entries', [])
        
        if not all_entries:
            # No real data available at all
            raise ValueError("No real content available. Please run real_data_collector.py to collect content.")
        
        # Try to find entries matching the category
        suitable_entries = [
            entry for entry in all_entries
            if (entry['categorization']['primary_category'] == category or 
                category in entry['categorization'].get('secondary_categories', []))
        ]
        
        # If no category match, use all available entries
        if not suitable_entries:
            suitable_entries = all_entries
            logger.info(f"No entries found for category '{category}', using all {len(all_entries)} available entries")
        
        # Select a random entry (will reuse entries if needed)
        entry = random.choice(suitable_entries)
        
        # Load actual content
        content = self._load_content(entry)
        
        # Generate title with timestamp and random suffix to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_title = entry['metadata']['title']
        source_name = entry['source']['name']
        # Add a random number to ensure uniqueness when reusing content
        unique_id = random.randint(1000, 9999)
        # Create a more descriptive title that relates to the actual content
        title = f"{source_name}: {original_title} [{timestamp}_{unique_id}]"
        
        # Prepare labels
        labels = [category, space_type]
        labels.extend(entry['categorization'].get('tags', [])[:3])  # Add some tags
        
        # Format content for Confluence
        formatted_content = self._format_for_confluence(content, entry, category)
        
        return {
            "title": title,
            "content": formatted_content,
            "labels": list(set(labels)),  # Remove duplicates
            "source_info": {
                "original_url": entry['source']['url'],
                "source_name": entry['source']['name'],
                "quality_score": entry['quality_metrics']['overall_quality']
            }
        }
    
    def _load_content(self, entry: Dict[str, Any]) -> str:
        """Load content from file"""
        entry_id = entry['id']
        
        # Check cache
        if entry_id in self._content_cache:
            return self._content_cache[entry_id]
        
        # Load from file
        raw_path = self.real_data_dir / entry['content']['raw_file_path']
        
        try:
            with open(raw_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Cache it
            self._content_cache[entry_id] = content
            return content
            
        except Exception as e:
            logger.error(f"Error loading content from {raw_path}: {e}")
            return f"Error loading content: {e}"
    
    def _format_for_confluence(self, content: str, entry: Dict[str, Any], category: str) -> str:
        """Format content for Confluence storage format"""
        content_format = entry['content']['format']
        
        # Add attribution header
        attribution = f"""
        <ac:structured-macro ac:name="info">
            <ac:rich-text-body>
                <p><strong>Source:</strong> {entry['source']['name']}</p>
                <p><strong>Original URL:</strong> <a href="{entry['source']['url']}">{entry['source']['url']}</a></p>
                <p><strong>Quality Score:</strong> {entry['quality_metrics']['overall_quality']}</p>
                <p><strong>Content Type:</strong> {content_format}</p>
                <p><strong>Original Category:</strong> {entry['categorization']['primary_category']}</p>
                <p><strong>Used in Space Category:</strong> {category}</p>
            </ac:rich-text-body>
        </ac:structured-macro>
        """
        
        # Convert content based on format
        if content_format == 'markdown':
            # Simple markdown to HTML conversion
            formatted = self._markdown_to_confluence(content)
        elif content_format == 'html':
            # Clean and adapt HTML for Confluence
            formatted = self._clean_html_for_confluence(content)
        elif content_format in ['rst', 'plain_text']:
            # Wrap in preformatted block
            formatted = f'<pre>{content}</pre>'
        else:
            formatted = f'<p>{content}</p>'
        
        # Add some structure
        final_content = f"""
        {attribution}
        
        <h2>Content</h2>
        {formatted}
        
        <hr/>
        <p><em>This content was imported from real documentation for testing purposes.</em></p>
        """
        
        return final_content
    
    def _markdown_to_confluence(self, markdown_content: str) -> str:
        """Simple markdown to Confluence HTML conversion"""
        # Very basic conversion - in production, use a proper markdown parser
        html = markdown_content
        
        # Headers
        html = html.replace('### ', '<h3>').replace('\n\n', '</h3>\n\n')
        html = html.replace('## ', '<h2>').replace('\n\n', '</h2>\n\n')
        html = html.replace('# ', '<h1>').replace('\n\n', '</h1>\n\n')
        
        # Code blocks
        html = html.replace('```', '<pre>').replace('</pre>\n<pre>', '</pre>\n\n<pre>')
        
        # Lists
        lines = html.split('\n')
        in_list = False
        new_lines = []
        
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    new_lines.append('<ul>')
                    in_list = True
                new_lines.append(f'<li>{line[2:].strip()}</li>')
            else:
                if in_list and not line.strip().startswith('- '):
                    new_lines.append('</ul>')
                    in_list = False
                new_lines.append(line)
        
        if in_list:
            new_lines.append('</ul>')
        
        html = '\n'.join(new_lines)
        
        # Paragraphs
        paragraphs = html.split('\n\n')
        html = '\n'.join(f'<p>{p}</p>' if not p.startswith('<') else p 
                        for p in paragraphs if p.strip())
        
        return html
    
    def _clean_html_for_confluence(self, html_content: str) -> str:
        """Clean HTML for Confluence compatibility"""
        # Remove problematic tags
        for tag in ['script', 'style', 'meta', 'link']:
            html_content = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', html_content, flags=re.DOTALL)
            html_content = re.sub(f'<{tag}[^>]*/?>', '', html_content)
        
        # Remove class and id attributes (Confluence doesn't like them)
        html_content = re.sub(r'\s*(class|id)="[^"]*"', '', html_content)
        
        return html_content
    
    def create_page(self, space_key: str, title: str, content: str, 
                   labels: List[str] = None) -> Optional[Dict[str, Any]]:
        """Create a page in Confluence"""
        try:
            # Create page
            page = self.client.atlassian_api.create_page(
                space=space_key,
                title=title,
                body=content,
                type='page',
                representation='storage'
            )
            
            # Add labels if provided
            if labels and page:
                page_id = page.get('id')
                for label in labels:
                    try:
                        self.client.atlassian_api.set_page_label(
                            page_id=page_id,
                            label=label.lower().replace(' ', '-')
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add label {label}: {e}")
            
            logger.info(f"Created page: {title}")
            return page
            
        except Exception as e:
            logger.error(f"Error creating page {title}: {e}")
            return None


class DummyContentGenerator:
    """Original dummy content generator for backward compatibility"""
    
    def __init__(self, client: ConfluenceClient, config: Any):
        self.client = client
        self.config = config
        self.templates = {
            'technical': [
                "Installation Guide for {topic}",
                "Architecture Overview: {topic}",
                "Troubleshooting {topic} Issues",
                "Performance Tuning for {topic}",
                "Security Best Practices: {topic}"
            ],
            'api_docs': [
                "{topic} REST API Reference",
                "{topic} GraphQL Schema",
                "Webhook Configuration for {topic}",
                "API Authentication: {topic}",
                "{topic} SDK Documentation"
            ],
            'knowledge_base': [
                "How to Configure {topic}",
                "FAQ: Common {topic} Questions",
                "Best Practices for {topic}",
                "{topic} Quick Start Guide",
                "Understanding {topic} Concepts"
            ],
            'project_docs': [
                "{topic} Project Planning",
                "Meeting Notes: {topic} Review",
                "{topic} Release Notes v{version}",
                "Sprint Planning: {topic}",
                "{topic} Roadmap Update"
            ],
            'multilingual': [
                "{topic} Documentation (EN/KO)",
                "多言語ガイド: {topic}",
                "Guía Multilingüe: {topic}",
                "Documentation Multilingue: {topic}",
                "다국어 문서: {topic}"
            ]
        }
        
        self.topics = [
            "Cloud Infrastructure", "Microservices", "Data Pipeline",
            "Machine Learning", "DevOps", "Security", "Analytics",
            "Mobile Development", "Web Services", "Database Management"
        ]
    
    def create_space(self, space_key: str, space_name: str) -> Optional[Dict[str, Any]]:
        """Create a new Confluence space"""
        try:
            # Check if space exists
            existing_spaces = self.client.atlassian_api.get_all_spaces()
            for space in existing_spaces.get('results', []):
                if space.get('key') == space_key:
                    logger.info(f"Space {space_key} already exists")
                    return space
            
            # Create new space
            space = self.client.atlassian_api.create_space(
                space_key=space_key,
                space_name=space_name
            )
            
            logger.info(f"Created space: {space_key}")
            return space
            
        except Exception as e:
            logger.error(f"Error creating space {space_key}: {e}")
            return None
    
    def generate_page_content(self, category: str, space_type: str) -> Dict[str, Any]:
        """Generate dummy page content"""
        topic = random.choice(self.topics)
        template = random.choice(self.templates.get(category, self.templates['technical']))
        
        # Generate title
        version = f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        title = template.format(topic=topic, version=version)
        
        # Generate content
        content = self._generate_dummy_content(category, topic)
        
        # Generate labels
        labels = [category, space_type, topic.lower().replace(' ', '-')]
        
        return {
            "title": title,
            "content": content,
            "labels": labels
        }
    
    def _generate_dummy_content(self, category: str, topic: str) -> str:
        """Generate dummy HTML content"""
        content_parts = []
        
        # Header
        content_parts.append(f"<h1>{topic} Documentation</h1>")
        content_parts.append(f"<p>This is comprehensive documentation for {topic} in the {category} category.</p>")
        
        # Table of Contents
        content_parts.append("<h2>Table of Contents</h2>")
        content_parts.append("<ul>")
        sections = ["Overview", "Getting Started", "Configuration", "Advanced Topics", "Troubleshooting"]
        for section in sections:
            content_parts.append(f"<li>{section}</li>")
        content_parts.append("</ul>")
        
        # Sections
        for i, section in enumerate(sections, 1):
            content_parts.append(f"<h2>{i}. {section}</h2>")
            content_parts.append(f"<p>This section covers important aspects of {topic} related to {section.lower()}.</p>")
            
            # Add some variety
            if section == "Configuration":
                content_parts.append("<pre>")
                content_parts.append("# Sample configuration")
                content_parts.append(f"{topic.lower().replace(' ', '_')}: {{")
                content_parts.append("  enabled: true")
                content_parts.append("  timeout: 30")
                content_parts.append("  max_connections: 100")
                content_parts.append("}")
                content_parts.append("</pre>")
            
            elif section == "Getting Started":
                content_parts.append("<ol>")
                content_parts.append(f"<li>Install {topic} dependencies</li>")
                content_parts.append("<li>Configure your environment</li>")
                content_parts.append("<li>Run the initialization script</li>")
                content_parts.append("<li>Verify the installation</li>")
                content_parts.append("</ol>")
            
            elif section == "Troubleshooting":
                content_parts.append("<table>")
                content_parts.append("<tr><th>Issue</th><th>Solution</th></tr>")
                content_parts.append(f"<tr><td>Connection timeout</td><td>Check network settings</td></tr>")
                content_parts.append(f"<tr><td>Authentication failed</td><td>Verify credentials</td></tr>")
                content_parts.append(f"<tr><td>Performance issues</td><td>Review resource allocation</td></tr>")
                content_parts.append("</table>")
        
        # Footer
        content_parts.append("<hr/>")
        content_parts.append(f"<p><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>")
        
        return '\n'.join(content_parts)
    
    def create_page(self, space_key: str, title: str, content: str, 
                   labels: List[str] = None) -> Optional[Dict[str, Any]]:
        """Create a page in Confluence"""
        try:
            # Create page
            page = self.client.atlassian_api.create_page(
                space=space_key,
                title=title,
                body=content,
                type='page',
                representation='storage'
            )
            
            # Add labels if provided
            if labels and page:
                page_id = page.get('id')
                for label in labels:
                    try:
                        self.client.atlassian_api.set_page_label(
                            page_id=page_id,
                            label=label.lower().replace(' ', '-')
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add label {label}: {e}")
            
            logger.info(f"Created page: {title}")
            return page
            
        except Exception as e:
            logger.error(f"Error creating page {title}: {e}")
            return None


# For backward compatibility and flexibility
ContentGenerator = DummyContentGenerator