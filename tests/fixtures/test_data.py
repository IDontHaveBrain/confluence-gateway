"""Test data fixtures for Confluence Gateway testing.

Provides representative Confluence content samples, mock API responses,
and test scenarios for comprehensive testing.
"""

from typing import Any

# Sample Confluence Page Content
SAMPLE_PAGES: list[dict[str, Any]] = [
    {
        "id": "12345",
        "title": "Software Development Best Practices",
        "content": """
<h1>Software Development Best Practices</h1>

<h2>Code Quality Standards</h2>
<p>Our engineering team follows strict code quality standards to ensure maintainable and reliable software:</p>
<ul>
    <li>All code must pass linting checks using our configured tools</li>
    <li>Test coverage must be above 80% for new features</li>
    <li>Code reviews are mandatory for all pull requests</li>
    <li>Documentation must be updated with new features</li>
</ul>

<h2>Development Workflow</h2>
<p>Our standard development workflow follows these key steps:</p>
<ol>
    <li>Create feature branch from main</li>
    <li>Implement feature with tests</li>
    <li>Submit pull request for review</li>
    <li>Address feedback and update code</li>
    <li>Merge after approval and CI success</li>
</ol>

<h2>Security Guidelines</h2>
<p>Security is paramount in our development process. All developers must:</p>
<ul>
    <li>Never commit secrets or API keys to version control</li>
    <li>Use environment variables for configuration</li>
    <li>Follow OWASP security guidelines</li>
    <li>Regularly update dependencies to patch vulnerabilities</li>
</ul>
""",
        "space": {"key": "DEV", "name": "Development Team"},
        "lastModified": "2024-01-15T10:30:00.000Z",
        "url": "https://company.atlassian.net/wiki/spaces/DEV/pages/12345",
        "type": "page",
    },
    {
        "id": "23456",
        "title": "API Documentation Guidelines",
        "content": """
<h1>API Documentation Guidelines</h1>

<h2>Overview</h2>
<p>This document outlines the standards for documenting our REST APIs.</p>

<h2>Required Documentation Elements</h2>
<ul>
    <li>Endpoint description and purpose</li>
    <li>HTTP method and URL pattern</li>
    <li>Request parameters and body schema</li>
    <li>Response codes and examples</li>
    <li>Authentication requirements</li>
    <li>Rate limiting information</li>
</ul>

<h2>OpenAPI Specification</h2>
<p>All APIs must include OpenAPI 3.0 specifications with:</p>
<ul>
    <li>Complete schema definitions</li>
    <li>Example request and response payloads</li>
    <li>Error response documentation</li>
</ul>
""",
        "space": {"key": "DOC", "name": "Documentation"},
        "lastModified": "2024-01-14T14:15:00.000Z",
        "url": "https://company.atlassian.net/wiki/spaces/DOC/pages/23456",
        "type": "page",
    },
    {
        "id": "34567",
        "title": "Deployment Process",
        "content": """
<h1>Deployment Process</h1>

<h2>Production Deployment</h2>
<p>Our production deployment follows a strict process to ensure system reliability:</p>

<h3>Pre-deployment Checklist</h3>
<ul>
    <li>All tests pass in CI/CD pipeline</li>
    <li>Security scan results reviewed</li>
    <li>Performance testing completed</li>
    <li>Database migration scripts validated</li>
    <li>Rollback plan prepared</li>
</ul>

<h3>Deployment Steps</h3>
<ol>
    <li>Schedule maintenance window if required</li>
    <li>Create deployment tag in version control</li>
    <li>Deploy to staging environment first</li>
    <li>Run smoke tests on staging</li>
    <li>Deploy to production using blue-green strategy</li>
    <li>Monitor system metrics for 30 minutes</li>
    <li>Notify stakeholders of successful deployment</li>
</ol>

<h2>Emergency Hotfixes</h2>
<p>Critical production issues may require emergency hotfixes:</p>
<ul>
    <li>Fast-track approval from tech lead required</li>
    <li>Minimal testing acceptable for critical fixes</li>
    <li>Immediate rollback plan must be ready</li>
    <li>Post-deployment retrospective mandatory</li>
</ul>
""",
        "space": {"key": "OPS", "name": "Operations"},
        "lastModified": "2024-01-13T09:45:00.000Z",
        "url": "https://company.atlassian.net/wiki/spaces/OPS/pages/34567",
        "type": "page",
    },
]

# Edge Case Content Samples
EDGE_CASE_PAGES: list[dict[str, Any]] = [
    {
        "id": "99999",
        "title": "Empty Page",
        "content": "<p></p>",
        "space": {"key": "EMPTY", "name": "Empty Space"},
        "lastModified": "2024-01-01T00:00:00.000Z",
        "url": "https://company.atlassian.net/wiki/spaces/EMPTY/pages/99999",
        "type": "page",
    },
    {
        "id": "88888",
        "title": "Large Document with Extensive Content",
        "content": """
<h1>Large Document with Extensive Content</h1>
"""
        + "\n".join(
            [
                f"<p>This is paragraph {i} with substantial content that includes detailed information about various aspects of our system architecture, implementation details, and operational procedures. This content is designed to test the handling of large documents in our search and indexing system.</p>"
                for i in range(1, 101)
            ]
        ),
        "space": {"key": "LARGE", "name": "Large Content Space"},
        "lastModified": "2024-01-16T12:00:00.000Z",
        "url": "https://company.atlassian.net/wiki/spaces/LARGE/pages/88888",
        "type": "page",
    },
]

# Sample Confluence Spaces
SAMPLE_SPACES: list[dict[str, Any]] = [
    {
        "key": "DEV",
        "name": "Development Team",
        "description": "Documentation and resources for the development team",
        "homepageId": "12345",
        "type": "global",
    },
    {
        "key": "DOC",
        "name": "Documentation",
        "description": "Technical documentation and guidelines",
        "homepageId": "23456",
        "type": "global",
    },
    {
        "key": "OPS",
        "name": "Operations",
        "description": "Operational procedures and deployment guides",
        "homepageId": "34567",
        "type": "global",
    },
    {
        "key": "EMPTY",
        "name": "Empty Space",
        "description": "Test space with minimal content",
        "homepageId": "99999",
        "type": "global",
    },
]

# Sample Search Queries and Expected Result Types
SAMPLE_SEARCH_QUERIES: list[dict[str, Any]] = [
    {
        "query": "deployment process",
        "expected_pages": ["34567"],  # Should find Deployment Process page
        "search_type": "text",
    },
    {
        "query": "code quality standards",
        "expected_pages": ["12345"],  # Should find Software Development Best Practices
        "search_type": "semantic",
    },
    {
        "query": "API documentation",
        "expected_pages": ["23456"],  # Should find API Documentation Guidelines
        "search_type": "hybrid",
    },
    {
        "query": "security guidelines OWASP",
        "expected_pages": ["12345"],  # Should find page with security content
        "search_type": "text",
    },
    {
        "query": "emergency hotfix procedures",
        "expected_pages": ["34567"],  # Should find deployment process with hotfix info
        "search_type": "semantic",
    },
]

# CQL (Confluence Query Language) Test Queries
SAMPLE_CQL_QUERIES: list[dict[str, Any]] = [
    {
        "cql": "space = DEV",
        "expected_pages": ["12345"],
        "description": "Find all pages in DEV space",
    },
    {
        "cql": 'title ~ "deployment"',
        "expected_pages": ["34567"],
        "description": "Find pages with 'deployment' in title",
    },
    {
        "cql": 'text ~ "API" AND space = DOC',
        "expected_pages": ["23456"],
        "description": "Find pages containing 'API' in DOC space",
    },
]

# Mock Confluence API Responses
MOCK_CONFLUENCE_RESPONSES: dict[str, Any] = {
    "spaces_list": {
        "results": SAMPLE_SPACES,
        "size": len(SAMPLE_SPACES),
        "start": 0,
        "limit": 50,
    },
    "search_results": {
        "results": [
            {
                "id": "12345",
                "title": "Software Development Best Practices",
                "excerpt": "Our engineering team follows strict code quality standards...",
                "url": "https://company.atlassian.net/wiki/spaces/DEV/pages/12345",
                "lastModified": "2024-01-15T10:30:00.000Z",
                "space": {"key": "DEV", "name": "Development Team"},
            }
        ],
        "size": 1,
        "totalSize": 1,
        "start": 0,
        "limit": 25,
    },
    "page_content": {
        "id": "12345",
        "title": "Software Development Best Practices",
        "body": {
            "storage": {
                "value": SAMPLE_PAGES[0]["content"],
                "representation": "storage",
            }
        },
        "space": {"key": "DEV", "name": "Development Team"},
        "version": {"when": "2024-01-15T10:30:00.000Z"},
        "_links": {
            "base": "https://company.atlassian.net/wiki",
            "webui": "/spaces/DEV/pages/12345",
        },
    },
}

# Test Configuration Templates (Essential configurations only)
TEST_CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "qdrant_memory": {
        "vector_db": {
            "type": "qdrant",
            "qdrant_url": ":memory:",
            "qdrant_local_path": None,
        },
        "embedding": {
            "provider": "sentence-transformers",
            "model_name": "all-MiniLM-L6-v2",
            "dimension": 384,
        },
    },
    "chroma_memory": {
        "vector_db": {
            "type": "chroma",
            "chroma_persist_path": None,
            "chroma_host": None,
            "chroma_port": None,
        },
        "embedding": {
            "provider": "sentence-transformers",
            "model_name": "all-MiniLM-L6-v2",
            "dimension": 384,
        },
    },
    "no_vector_db": {"vector_db": {"type": "none"}, "embedding": {"provider": "none"}},
}

# Sample RAG Generation Test Data
RAG_TEST_QUERIES: list[dict[str, Any]] = [
    {
        "query": "What are our code quality standards?",
        "expected_sources": [
            "12345"
        ],  # Should retrieve from Software Development Best Practices
        "context_should_contain": ["linting", "test coverage", "code reviews"],
    },
    {
        "query": "How do we handle emergency deployments?",
        "expected_sources": ["34567"],  # Should retrieve from Deployment Process
        "context_should_contain": ["hotfix", "emergency", "fast-track approval"],
    },
    {
        "query": "What should be included in API documentation?",
        "expected_sources": [
            "23456"
        ],  # Should retrieve from API Documentation Guidelines
        "context_should_contain": ["endpoint description", "OpenAPI", "schema"],
    },
]


def get_sample_page_by_id(page_id: str) -> dict[str, Any] | None:
    """Get a sample page by ID from test data."""
    all_pages = SAMPLE_PAGES + EDGE_CASE_PAGES
    for page in all_pages:
        if page["id"] == page_id:
            return page
    return None


def get_sample_pages_by_space(space_key: str) -> list[dict[str, Any]]:
    """Get all sample pages for a given space key."""
    all_pages = SAMPLE_PAGES + EDGE_CASE_PAGES
    return [page for page in all_pages if page["space"]["key"] == space_key]


def get_sample_space_by_key(space_key: str) -> dict[str, Any] | None:
    """Get a sample space by key from test data."""
    for space in SAMPLE_SPACES:
        if space["key"] == space_key:
            return space
    return None
