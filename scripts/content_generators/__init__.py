"""Content generators for dummy data creation"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from atlassian import Confluence

logger = logging.getLogger(__name__)

class ContentGenerator:
    """Main content generator that orchestrates different content types"""
    
    def __init__(self, client, config):
        self.client = client
        self.config = config

        from .technical_docs import TechnicalDocsGenerator
        from .api_docs import APIDocsGenerator
        from .project_docs import ProjectDocsGenerator
        from .multilang_docs import MultilangDocsGenerator
        
        self.generators = {
            "installation": TechnicalDocsGenerator(config),
            "architecture": TechnicalDocsGenerator(config),
            "troubleshooting": TechnicalDocsGenerator(config),
            "rest": APIDocsGenerator(config),
            "graphql": APIDocsGenerator(config),
            "webhooks": APIDocsGenerator(config),
            "how-to": ProjectDocsGenerator(config),
            "faq": ProjectDocsGenerator(config),
            "best-practices": ProjectDocsGenerator(config),
            "planning": ProjectDocsGenerator(config),
            "meeting-notes": ProjectDocsGenerator(config),
            "releases": ProjectDocsGenerator(config),
            "english": MultilangDocsGenerator(config, language="en"),
            "korean": MultilangDocsGenerator(config, language="ko"),
            "mixed": MultilangDocsGenerator(config, language="mixed")
        }
    
    def create_space(self, space_key: str, space_name: str, description: str = None) -> Dict[str, Any]:
        """Create a new Confluence space using the atlassian library"""
        try:

            existing = self.client.atlassian_api.get_all_spaces(start=0, limit=500)
            for space in existing.get("results", []):
                if space["key"] == space_key:
                    logger.info(f"Space {space_key} already exists")
                    return space
            
            logger.info(f"Creating space with key: {space_key}, name: {space_name}")

            self.client.atlassian_api.create_space(space_key, space_name)

            created_space = self.client.atlassian_api.get_space(space_key)
            
            logger.info(f"Successfully created space: {space_key}")
            
            return created_space
            
        except Exception as e:
            logger.error(f"Failed to create space {space_key}: {e}")
            if "already exists" in str(e).lower():
                raise ValueError(f"Space with key '{space_key}' already exists")
            elif "permission" in str(e).lower():
                raise PermissionError(f"Permission denied to create space. You may need 'Create Space' global permission.")
            else:
                raise
    
    def generate_page_content(self, category: str, space_type: str) -> Dict[str, Any]:
        """Generate page content based on category"""
        generator = self.generators.get(category)
        if not generator:
            logger.warning(f"No generator for category: {category}")
            generator = self.generators.get("installation")  # Default
        
        return generator.generate(category, space_type)
    
    def create_page(
        self,
        space_key: str,
        title: str,
        content: str,
        parent_id: Optional[str] = None,
        labels: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a new Confluence page"""
        try:

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]
            unique_title = f"{self.config.prefix} - {title} - {timestamp}"

            page = self.client.atlassian_api.create_page(
                space_key,
                unique_title,
                content,
                parent_id=parent_id,
                type='page',
                representation='storage'
            )

            if labels and page:
                for label in labels:
                    try:
                        self.client.atlassian_api.set_page_label(page["id"], label)
                    except Exception as e:
                        logger.warning(f"Failed to add label '{label}': {e}")
            
            if page:
                logger.info(f"Created page: {unique_title}")
            
            return page
            
        except Exception as e:
            logger.error(f"Failed to create page '{title}': {e}")
            return None
    
    def create_nested_structure(self, space_key: str, structure: Dict[str, Any]) -> None:
        """Create a nested page structure"""
        def create_level(parent_id: Optional[str], items: List[Dict[str, Any]]):
            for item in items:
                page = self.create_page(
                    space_key,
                    item["title"],
                    item.get("content", f"<p>Content for {item['title']}</p>"),
                    parent_id=parent_id,
                    labels=item.get("labels", [])
                )
                
                if page and "children" in item:
                    create_level(page["id"], item["children"])
        
        create_level(None, structure.get("pages", []))

SEMANTIC_TEST_PAIRS = [
    {
        "content": "Create indexes for database optimization and improve query performance. Use proper indexing strategies like B-tree, hash, and full-text indexes to speed up data retrieval.",
        "similar_queries": ["DB performance enhancement", "SQL tuning", "query speed improvement", "database indexing best practices"]
    },
    {
        "content": "Microservice communication occurs through REST APIs and message queues. Implement service discovery, circuit breakers, and proper error handling for resilient inter-service communication.",
        "similar_queries": ["service integration", "MSA communication methods", "distributed system messaging", "microservices patterns"]
    },
    {
        "content": "Deploy applications using Docker containers and Kubernetes orchestration. Set up CI/CD pipelines with automated testing and rolling deployments for zero-downtime releases.",
        "similar_queries": ["container deployment", "K8s deployment strategy", "application release process", "DevOps automation"]
    },
    {
        "content": "Implement authentication using JWT tokens and OAuth2 flow. Secure your APIs with proper token validation, refresh mechanisms, and role-based access control.",
        "similar_queries": ["API security", "token-based auth", "secure authentication methods", "OAuth implementation"]
    },
    {
        "content": "Monitor application performance with distributed tracing and metrics collection. Use tools like Prometheus, Grafana, and Jaeger for comprehensive observability.",
        "similar_queries": ["APM tools", "application monitoring", "observability stack", "performance tracking"]
    }
]

CQL_METADATA_PATTERNS = [
    {"labels": ["deployment", "production", "v2.0"], "created_by": "devops_team"},
    {"labels": ["development", "setup", "local"], "created_by": "dev_team"},
    {"labels": ["api", "documentation", "swagger"], "created_by": "api_team"},
    {"labels": ["security", "compliance", "audit"], "created_by": "security_team"},
    {"labels": ["performance", "optimization", "metrics"], "created_by": "platform_team"}
]