"""API documentation content generator"""

import random
import json
from typing import Dict, Any, List
from datetime import datetime

class APIDocsGenerator:
    """Generates API documentation content"""
    
    def __init__(self, config):
        self.config = config
        self.templates = {
            "rest": self._rest_api_templates(),
            "graphql": self._graphql_templates(),
            "webhooks": self._webhook_templates()
        }
    
    def generate(self, category: str, space_type: str) -> Dict[str, Any]:
        """Generate API documentation content"""
        templates = self.templates.get(category, self.templates["rest"])
        template = random.choice(templates)

        api_name = random.choice(["User", "Product", "Order", "Payment", "Inventory", "Analytics"])
        version = random.choice(["v1", "v2", "v3"])
        
        title = template["title_pattern"].format(
            api=api_name,
            version=version,
            type=category.upper()
        )
        
        content = self._generate_content(template, api_name, version)
        labels = template["labels"] + [category, "api", space_type.lower(), version]
        
        return {
            "title": title,
            "content": content,
            "labels": labels,
            "metadata": {
                "category": category,
                "type": "api",
                "api_version": version,
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def _generate_content(self, template: Dict[str, Any], api_name: str, version: str) -> str:
        """Generate HTML content for API documentation"""
        html_parts = [f"<h1>{api_name} API Documentation - {version}</h1>"]

        html_parts.append(f"""
            <ac:structured-macro ac:name="info">
                <ac:rich-text-body>
                    <p><strong>Base URL:</strong> https://api.example.com/{version}/{api_name.lower()}</p>
                    <p><strong>Authentication:</strong> Bearer token in Authorization header</p>
                    <p><strong>Content-Type:</strong> application/json</p>
                </ac:rich-text-body>
            </ac:structured-macro>
        """)

        for section in template["sections"]:
            html_parts.append(self._generate_section(section, api_name))

        if self.config.search_optimization["semantic_pairs"]:
            html_parts.append(self._add_semantic_content(template["category"]))
        
        return "\n".join(html_parts)
    
    def _generate_section(self, section: Dict[str, Any], api_name: str) -> str:
        """Generate a section of API documentation"""
        html = f"<h2>{section['heading']}</h2>"
        
        if section["type"] == "endpoint":
            html += self._generate_endpoint_doc(section["endpoint"], api_name)
        elif section["type"] == "schema":
            html += self._generate_schema_doc(section["schema"])
        elif section["type"] == "example":
            html += self._generate_example_doc(section["example"])
        elif section["type"] == "authentication":
            html += self._generate_auth_doc(section["auth"])
        
        return html
    
    def _generate_endpoint_doc(self, endpoint: Dict[str, Any], api_name: str) -> str:
        """Generate endpoint documentation"""
        method = endpoint["method"]
        path = endpoint["path"].format(resource=api_name.lower())
        
        html = f"""
            <h3>{method} {path}</h3>
            <p>{endpoint['description']}</p>
            
            <h4>Request Parameters</h4>
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Type</th>
                        <th>Required</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for param in endpoint.get("parameters", []):
            html += f"""
                    <tr>
                        <td>{param['name']}</td>
                        <td>{param['type']}</td>
                        <td>{param.get('required', False)}</td>
                        <td>{param['description']}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        """

        if "request_example" in endpoint:
            html += f"""
                <h4>Request Example</h4>
                <ac:structured-macro ac:name="code">
                    <ac:parameter ac:name="language">json</ac:parameter>
                    <ac:plain-text-body><![CDATA[{json.dumps(endpoint['request_example'], indent=2)}]]></ac:plain-text-body>
                </ac:structured-macro>
            """
        
        if "response_example" in endpoint:
            html += f"""
                <h4>Response Example</h4>
                <ac:structured-macro ac:name="code">
                    <ac:parameter ac:name="language">json</ac:parameter>
                    <ac:plain-text-body><![CDATA[{json.dumps(endpoint['response_example'], indent=2)}]]></ac:plain-text-body>
                </ac:structured-macro>
            """
        
        return html
    
    def _generate_schema_doc(self, schema: Dict[str, Any]) -> str:
        """Generate schema documentation"""
        return f"""
            <h3>{schema['name']} Schema</h3>
            <p>{schema['description']}</p>
            <ac:structured-macro ac:name="code">
                <ac:parameter ac:name="language">json</ac:parameter>
                <ac:plain-text-body><![CDATA[{json.dumps(schema['properties'], indent=2)}]]></ac:plain-text-body>
            </ac:structured-macro>
        """
    
    def _generate_example_doc(self, example: Dict[str, Any]) -> str:
        """Generate example documentation"""
        return f"""
            <h3>{example['title']}</h3>
            <p>{example['description']}</p>
            <ac:structured-macro ac:name="code">
                <ac:parameter ac:name="language">{example.get('language', 'bash')}</ac:parameter>
                <ac:plain-text-body><![CDATA[{example['code']}]]></ac:plain-text-body>
            </ac:structured-macro>
        """
    
    def _generate_auth_doc(self, auth: Dict[str, Any]) -> str:
        """Generate authentication documentation"""
        return f"""
            <h3>{auth['type']} Authentication</h3>
            <p>{auth['description']}</p>
            <h4>Implementation</h4>
            <ac:structured-macro ac:name="code">
                <ac:parameter ac:name="language">{auth.get('language', 'bash')}</ac:parameter>
                <ac:plain-text-body><![CDATA[{auth['example']}]]></ac:plain-text-body>
            </ac:structured-macro>
        """
    
    def _add_semantic_content(self, category: str) -> str:
        """Add content optimized for semantic search"""
        semantic_blocks = {
            "rest": """
                <h3>REST API Best Practices</h3>
                <p>This REST API follows RESTful principles including resource-based URLs, HTTP methods for CRUD operations, 
                stateless communication, and standard HTTP status codes. It supports pagination, filtering, sorting, 
                and field selection for optimal data retrieval.</p>
            """,
            "graphql": """
                <h3>GraphQL Query Optimization</h3>
                <p>This GraphQL API enables flexible data fetching with query batching, field selection, 
                nested resource resolution, and real-time subscriptions. It includes query complexity analysis, 
                depth limiting, and caching strategies for performance optimization.</p>
            """,
            "webhooks": """
                <h3>Webhook Integration Guide</h3>
                <p>This webhook system provides real-time event notifications with retry logic, signature verification, 
                event filtering, and payload customization. It supports multiple delivery methods, 
                event aggregation, and comprehensive error handling.</p>
            """
        }
        return semantic_blocks.get(category, "")
    
    def _rest_api_templates(self) -> List[Dict[str, Any]]:
        """REST API documentation templates"""
        return [
            {
                "title_pattern": "{api} {type} API Reference - {version}",
                "category": "rest",
                "labels": ["rest-api", "documentation", "reference"],
                "sections": [
                    {
                        "heading": "Endpoints",
                        "type": "endpoint",
                        "endpoint": {
                            "method": "GET",
                            "path": "/{resource}",
                            "description": "Retrieve a list of resources with pagination support",
                            "parameters": [
                                {"name": "page", "type": "integer", "required": False, "description": "Page number (default: 1)"},
                                {"name": "limit", "type": "integer", "required": False, "description": "Items per page (default: 20, max: 100)"},
                                {"name": "sort", "type": "string", "required": False, "description": "Sort field and order (e.g., 'name:asc')"},
                                {"name": "filter", "type": "string", "required": False, "description": "Filter expression"}
                            ],
                            "response_example": {
                                "data": [
                                    {"id": 1, "name": "Item 1", "created_at": "2024-01-15T10:00:00Z"},
                                    {"id": 2, "name": "Item 2", "created_at": "2024-01-15T11:00:00Z"}
                                ],
                                "meta": {
                                    "page": 1,
                                    "limit": 20,
                                    "total": 42
                                }
                            }
                        }
                    },
                    {
                        "heading": "Create Resource",
                        "type": "endpoint",
                        "endpoint": {
                            "method": "POST",
                            "path": "/{resource}",
                            "description": "Create a new resource",
                            "request_example": {
                                "name": "New Item",
                                "description": "Item description",
                                "tags": ["tag1", "tag2"]
                            },
                            "response_example": {
                                "id": 3,
                                "name": "New Item",
                                "created_at": "2024-01-15T12:00:00Z"
                            }
                        }
                    },
                    {
                        "heading": "Error Responses",
                        "type": "schema",
                        "schema": {
                            "name": "Error",
                            "description": "Standard error response format",
                            "properties": {
                                "error": {
                                    "code": "VALIDATION_ERROR",
                                    "message": "Validation failed",
                                    "details": [
                                        {"field": "name", "message": "Name is required"}
                                    ]
                                }
                            }
                        }
                    }
                ]
            }
        ]
    
    def _graphql_templates(self) -> List[Dict[str, Any]]:
        """GraphQL API documentation templates"""
        return [
            {
                "title_pattern": "{api} GraphQL Schema - {version}",
                "category": "graphql",
                "labels": ["graphql", "schema", "api"],
                "sections": [
                    {
                        "heading": "Query Examples",
                        "type": "example",
                        "example": {
                            "title": "Fetch User with Posts",
                            "description": "Query to fetch user details along with their recent posts",
                            "language": "graphql",
                            "code": """query GetUserWithPosts($userId: ID!, $postLimit: Int = 10) {
  user(id: $userId) {
    id
    name
    email
    posts(limit: $postLimit) {
      edges {
        node {
          id
          title
          content
          createdAt
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}"""
                        }
                    },
                    {
                        "heading": "Mutations",
                        "type": "example",
                        "example": {
                            "title": "Create User Mutation",
                            "description": "Mutation to create a new user",
                            "language": "graphql",
                            "code": """mutation CreateUser($input: CreateUserInput!) {
  createUser(input: $input) {
    user {
      id
      name
      email
    }
    errors {
      field
      message
    }
  }
}"""
                        }
                    }
                ]
            }
        ]
    
    def _webhook_templates(self) -> List[Dict[str, Any]]:
        """Webhook documentation templates"""
        return [
            {
                "title_pattern": "{api} Webhook Events - {version}",
                "category": "webhooks",
                "labels": ["webhooks", "events", "integration"],
                "sections": [
                    {
                        "heading": "Webhook Configuration",
                        "type": "example",
                        "example": {
                            "title": "Register Webhook Endpoint",
                            "description": "Register a new webhook endpoint to receive events",
                            "language": "bash",
                            "code": """curl -X POST https://api.example.com/v1/webhooks \\
  -H "Authorization: Bearer YOUR_API_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "url": "https://your-app.com/webhook",
    "events": ["user.created", "user.updated", "order.completed"],
    "secret": "your-webhook-secret"
  }'"""
                        }
                    },
                    {
                        "heading": "Event Payload",
                        "type": "schema",
                        "schema": {
                            "name": "WebhookEvent",
                            "description": "Standard webhook event payload structure",
                            "properties": {
                                "id": "evt_1234567890",
                                "type": "user.created",
                                "created": "2024-01-15T12:00:00Z",
                                "data": {
                                    "object": {
                                        "id": "usr_abc123",
                                        "email": "user@example.com",
                                        "name": "John Doe"
                                    }
                                }
                            }
                        }
                    },
                    {
                        "heading": "Signature Verification",
                        "type": "authentication",
                        "auth": {
                            "type": "Webhook",
                            "description": "Verify webhook signatures to ensure authenticity",
                            "language": "python",
                            "example": """import hmac
import hashlib

def verify_webhook_signature(payload, signature, secret):
    expected = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)"""
                        }
                    }
                ]
            }
        ]