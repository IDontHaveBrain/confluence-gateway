"""Technical documentation content generator"""

import random
from typing import Dict, Any, List
from datetime import datetime

class TechnicalDocsGenerator:
    """Generates technical documentation content"""
    
    def __init__(self, config):
        self.config = config
        self.templates = {
            "installation": self._installation_templates(),
            "architecture": self._architecture_templates(),
            "troubleshooting": self._troubleshooting_templates()
        }
    
    def generate(self, category: str, space_type: str) -> Dict[str, Any]:
        """Generate technical documentation content"""
        templates = self.templates.get(category, self.templates["installation"])
        template = random.choice(templates)

        title = template["title_pattern"].format(
            tech=random.choice(["Docker", "Kubernetes", "Python", "Node.js", "PostgreSQL", "MongoDB", "Redis", "Nginx"]),
            version=random.choice(["v1.0", "v2.0", "v3.0", "latest"]),
            env=random.choice(["Development", "Production", "Staging", "Testing"])
        )
        
        content = self._generate_content(template, category, title)
        labels = template["labels"] + [category, "technical", space_type.lower()]
        
        return {
            "title": title,
            "content": content,
            "labels": labels,
            "metadata": {
                "category": category,
                "type": "technical",
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def _generate_content(self, template: Dict[str, Any], category: str, title: str) -> str:
        """Generate HTML content based on template"""
        sections = template["sections"]
        html_parts = [f"<h1>{title}</h1>"]
        
        for section in sections:
            html_parts.append(f"<h2>{section['heading']}</h2>")
            
            if section["type"] == "paragraph":
                html_parts.append(f"<p>{section['content']}</p>")
            
            elif section["type"] == "code":
                code = section['content']
                html_parts.append(f'<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">{section.get("language", "bash")}</ac:parameter><ac:plain-text-body><![CDATA[{code}]]></ac:plain-text-body></ac:structured-macro>')
            
            elif section["type"] == "list":
                html_parts.append("<ul>")
                for item in section['items']:
                    html_parts.append(f"<li>{item}</li>")
                html_parts.append("</ul>")
            
            elif section["type"] == "table":
                html_parts.append('<table><tbody>')
                for row in section['rows']:
                    html_parts.append('<tr>')
                    for cell in row:
                        html_parts.append(f'<td>{cell}</td>')
                    html_parts.append('</tr>')
                html_parts.append('</tbody></table>')

        if self.config.search_optimization["semantic_pairs"]:
            html_parts.append(self._add_semantic_content(category))
        
        return "\n".join(html_parts)
    
    def _add_semantic_content(self, category: str) -> str:
        """Add content optimized for semantic search"""
        semantic_blocks = {
            "installation": """
                <h3>Related Topics</h3>
                <p>This installation guide covers environment setup, dependency management, configuration steps, 
                initial deployment, and verification procedures. It includes best practices for container deployment, 
                package installation, system requirements validation, and post-installation testing.</p>
            """,
            "architecture": """
                <h3>Architectural Considerations</h3>
                <p>This architecture document describes system design patterns, component interactions, 
                scalability strategies, fault tolerance mechanisms, and integration points. It covers 
                microservices communication, data flow diagrams, deployment topology, and performance characteristics.</p>
            """,
            "troubleshooting": """
                <h3>Common Issues and Solutions</h3>
                <p>This troubleshooting guide addresses frequent problems, error diagnostics, debugging techniques, 
                log analysis methods, and resolution procedures. It includes performance bottlenecks, 
                configuration errors, connectivity issues, and system recovery steps.</p>
            """
        }
        return semantic_blocks.get(category, "")
    
    def _installation_templates(self) -> List[Dict[str, Any]]:
        """Installation guide templates"""
        return [
            {
                "title_pattern": "{tech} {version} Installation Guide - {env} Environment",
                "labels": ["installation", "setup", "guide"],
                "sections": [
                    {
                        "heading": "Prerequisites",
                        "type": "list",
                        "items": [
                            "Operating System: Ubuntu 20.04 LTS or later",
                            "Memory: Minimum 4GB RAM (8GB recommended)",
                            "Storage: At least 20GB free disk space",
                            "Network: Stable internet connection for package downloads"
                        ]
                    },
                    {
                        "heading": "Installation Steps",
                        "type": "code",
                        "language": "bash",
                        "content": """# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

sudo apt-get install -y curl wget git build-essential

curl -fsSL https://get.example.com/install.sh | bash

example --version"""
                    },
                    {
                        "heading": "Configuration",
                        "type": "paragraph",
                        "content": "After installation, configure the application by editing the configuration file located at /etc/example/config.yaml. Ensure all required parameters are set according to your environment."
                    },
                    {
                        "heading": "Environment Variables",
                        "type": "table",
                        "rows": [
                            ["Variable", "Description", "Default"],
                            ["APP_PORT", "Application port", "8080"],
                            ["DB_HOST", "Database hostname", "localhost"],
                            ["LOG_LEVEL", "Logging level", "info"]
                        ]
                    }
                ]
            }
        ]
    
    def _architecture_templates(self) -> List[Dict[str, Any]]:
        """Architecture documentation templates"""
        return [
            {
                "title_pattern": "{tech} Microservices Architecture - {version}",
                "labels": ["architecture", "design", "microservices"],
                "sections": [
                    {
                        "heading": "System Overview",
                        "type": "paragraph",
                        "content": "This document describes the microservices architecture implementation using modern cloud-native technologies. The system is designed for high availability, scalability, and maintainability."
                    },
                    {
                        "heading": "Component Architecture",
                        "type": "list",
                        "items": [
                            "API Gateway: Routes requests to appropriate microservices",
                            "Authentication Service: Handles user authentication and JWT token generation",
                            "Data Service: Manages database operations and caching",
                            "Notification Service: Sends emails, SMS, and push notifications",
                            "Analytics Service: Processes and aggregates system metrics"
                        ]
                    },
                    {
                        "heading": "Communication Patterns",
                        "type": "code",
                        "language": "yaml",
                        "content": """services:
  api-gateway:
    type: REST
    protocol: HTTP/2
    load-balancer: round-robin
    
  message-bus:
    type: async
    broker: RabbitMQ
    pattern: publish-subscribe"""
                    }
                ]
            }
        ]
    
    def _troubleshooting_templates(self) -> List[Dict[str, Any]]:
        """Troubleshooting guide templates"""
        return [
            {
                "title_pattern": "{tech} Troubleshooting Guide - Common Issues",
                "labels": ["troubleshooting", "debugging", "errors"],
                "sections": [
                    {
                        "heading": "Connection Timeout Errors",
                        "type": "paragraph",
                        "content": "If you encounter connection timeout errors, check network connectivity, firewall rules, and service availability. Verify that all required ports are open and services are running."
                    },
                    {
                        "heading": "Debug Commands",
                        "type": "code",
                        "language": "bash",
                        "content": """# Check service status
systemctl status example-service

journalctl -u example-service -n 100

curl -v http://localhost:8080/health

htop"""
                    },
                    {
                        "heading": "Common Error Codes",
                        "type": "table",
                        "rows": [
                            ["Error Code", "Description", "Solution"],
                            ["ERR_001", "Database connection failed", "Check database credentials and connectivity"],
                            ["ERR_002", "Authentication failed", "Verify API keys and permissions"],
                            ["ERR_003", "Rate limit exceeded", "Implement backoff strategy or increase limits"]
                        ]
                    }
                ]
            }
        ]