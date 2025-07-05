"""Project documentation content generator"""

import random
from typing import Dict, Any, List
from datetime import datetime, timedelta

class ProjectDocsGenerator:
    """Generates project documentation content"""
    
    def __init__(self, config):
        self.config = config
        self.templates = {
            "planning": self._planning_templates(),
            "meeting-notes": self._meeting_templates(),
            "releases": self._release_templates(),
            "how-to": self._howto_templates(),
            "faq": self._faq_templates(),
            "best-practices": self._best_practices_templates()
        }
    
    def generate(self, category: str, space_type: str) -> Dict[str, Any]:
        """Generate project documentation content"""
        templates = self.templates.get(category, self.templates["planning"])
        template = random.choice(templates)

        project_name = random.choice(["Phoenix", "Atlas", "Nexus", "Horizon", "Quantum", "Velocity"])
        sprint_num = random.randint(1, 20)
        version = f"{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 20)}"
        
        title = template["title_pattern"].format(
            project=project_name,
            sprint=sprint_num,
            version=version,
            date=datetime.now().strftime("%Y-%m-%d")
        )
        
        content = self._generate_content(template, project_name, sprint_num, version)
        labels = template["labels"] + [category, "project", space_type.lower()]
        
        return {
            "title": title,
            "content": content,
            "labels": labels,
            "metadata": {
                "category": category,
                "type": "project",
                "project": project_name,
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def _generate_content(self, template: Dict[str, Any], project: str, sprint: int, version: str) -> str:
        """Generate HTML content for project documentation"""
        html_parts = []
        
        for section in template["sections"]:
            html_parts.append(self._generate_section(section, project, sprint, version))

        if self.config.search_optimization["semantic_pairs"]:
            html_parts.append(self._add_semantic_content(template["category"]))

        if self.config.search_optimization["cql_friendly_metadata"]:
            html_parts.append(self._add_cql_metadata(project, version))
        
        return "\n".join(html_parts)
    
    def _generate_section(self, section: Dict[str, Any], project: str, sprint: int, version: str) -> str:
        """Generate a section of project documentation"""
        html = f"<h2>{section['heading']}</h2>"
        
        if section["type"] == "timeline":
            html += self._generate_timeline(section.get("items", []), sprint)
        elif section["type"] == "checklist":
            html += self._generate_checklist(section.get("items", []))
        elif section["type"] == "status_table":
            html += self._generate_status_table(section.get("rows", []), project)
        elif section["type"] == "notes":
            html += self._generate_notes(section.get("content", []))
        elif section["type"] == "action_items":
            html += self._generate_action_items(section.get("items", []))
        elif section["type"] == "metrics":
            html += self._generate_metrics(section.get("data", {}))
        
        return html
    
    def _generate_timeline(self, items: List[str], sprint: int) -> str:
        """Generate timeline section"""
        html = "<ul>"
        base_date = datetime.now()
        
        for i, item in enumerate(items):
            date = (base_date + timedelta(days=i*7)).strftime("%Y-%m-%d")
            html += f"<li><strong>{date}</strong>: {item.format(sprint=sprint)}</li>"
        
        html += "</ul>"
        return html
    
    def _generate_checklist(self, items: List[str]) -> str:
        """Generate checklist with task list macro"""
        html = '<ac:structured-macro ac:name="tasklist"><ac:rich-text-body><ul>'
        
        for item in items:
            status = random.choice(["complete", "incomplete"])
            html += f'<li><ac:task><ac:task-status>{status}</ac:task-status><ac:task-body>{item}</ac:task-body></ac:task></li>'
        
        html += '</ul></ac:rich-text-body></ac:structured-macro>'
        return html
    
    def _generate_status_table(self, rows: List[Dict[str, Any]], project: str) -> str:
        """Generate status table"""
        html = """
            <table>
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Status</th>
                        <th>Progress</th>
                        <th>Owner</th>
                        <th>Notes</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for row in rows:
            status_color = {
                "On Track": "green",
                "At Risk": "yellow", 
                "Blocked": "red",
                "Complete": "blue"
            }
            status = random.choice(list(status_color.keys()))
            progress = random.randint(0, 100)
            owner = random.choice(["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown"])
            
            html += f"""
                <tr>
                    <td>{row['component'].format(project=project)}</td>
                    <td><ac:structured-macro ac:name="status"><ac:parameter ac:name="colour">{status_color[status]}</ac:parameter><ac:parameter ac:name="title">{status}</ac:parameter></ac:structured-macro></td>
                    <td>{progress}%</td>
                    <td>{owner}</td>
                    <td>{row.get('notes', 'On schedule')}</td>
                </tr>
            """
        
        html += "</tbody></table>"
        return html
    
    def _generate_notes(self, content_items: List[str]) -> str:
        """Generate meeting notes content"""
        html = ""
        for item in content_items:
            if isinstance(item, dict):
                if item["type"] == "decision":
                    html += f"""
                        <ac:structured-macro ac:name="info">
                            <ac:parameter ac:name="title">Decision</ac:parameter>
                            <ac:rich-text-body><p>{item['content']}</p></ac:rich-text-body>
                        </ac:structured-macro>
                    """
                elif item["type"] == "discussion":
                    html += f"<p><strong>Discussion:</strong> {item['content']}</p>"
            else:
                html += f"<p>{item}</p>"
        
        return html
    
    def _generate_action_items(self, items: List[Dict[str, str]]) -> str:
        """Generate action items table"""
        html = """
            <table>
                <thead>
                    <tr>
                        <th>Action Item</th>
                        <th>Owner</th>
                        <th>Due Date</th>
                        <th>Priority</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for item in items:
            due_date = (datetime.now() + timedelta(days=random.randint(1, 14))).strftime("%Y-%m-%d")
            priority = random.choice(["High", "Medium", "Low"])
            owner = random.choice(["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown"])
            
            html += f"""
                <tr>
                    <td>{item['action']}</td>
                    <td>{owner}</td>
                    <td>{due_date}</td>
                    <td>{priority}</td>
                </tr>
            """
        
        html += "</tbody></table>"
        return html
    
    def _generate_metrics(self, data: Dict[str, Any]) -> str:
        """Generate metrics section"""
        velocity = random.randint(20, 50)
        bugs_fixed = random.randint(5, 20)
        features_delivered = random.randint(3, 10)
        
        html = f"""
            <h3>Sprint Metrics</h3>
            <ul>
                <li>Velocity: {velocity} story points</li>
                <li>Features Delivered: {features_delivered}</li>
                <li>Bugs Fixed: {bugs_fixed}</li>
                <li>Team Satisfaction: {random.randint(7, 10)}/10</li>
            </ul>
        """
        return html
    
    def _add_semantic_content(self, category: str) -> str:
        """Add content optimized for semantic search"""
        semantic_blocks = {
            "planning": """
                <p>This project planning document outlines milestones, deliverables, resource allocation, 
                risk assessment, and timeline management. It includes stakeholder requirements, 
                technical specifications, budget constraints, and success criteria.</p>
            """,
            "meeting-notes": """
                <p>These meeting notes capture key decisions, action items, discussion points, 
                and follow-up tasks. They document participant contributions, resolved issues, 
                pending questions, and strategic alignments.</p>
            """,
            "releases": """
                <p>This release documentation covers version changes, feature additions, bug fixes, 
                breaking changes, migration guides, and deployment procedures. It includes 
                compatibility notes, performance improvements, and known issues.</p>
            """
        }
        return semantic_blocks.get(category, "")
    
    def _add_cql_metadata(self, project: str, version: str) -> str:
        """Add metadata optimized for CQL searches"""
        return f"""
            <!-- CQL Metadata -->
            <!-- Project: {project} -->
            <!-- Version: {version} -->
            <!-- Last Modified: {datetime.now().isoformat()} -->
            <!-- Status: Active -->
        """
    
    def _planning_templates(self) -> List[Dict[str, Any]]:
        """Project planning document templates"""
        return [
            {
                "title_pattern": "Project {project} - Sprint {sprint} Planning",
                "category": "planning",
                "labels": ["planning", "sprint", "roadmap"],
                "sections": [
                    {
                        "heading": "Sprint Goals",
                        "type": "checklist",
                        "items": [
                            "Complete user authentication module",
                            "Implement data validation layer",
                            "Set up automated testing pipeline",
                            "Deploy to staging environment",
                            "Conduct security review"
                        ]
                    },
                    {
                        "heading": "Timeline",
                        "type": "timeline",
                        "items": [
                            "Sprint {sprint} Kickoff",
                            "Development Phase",
                            "Testing and QA",
                            "Sprint Review",
                            "Sprint Retrospective"
                        ]
                    },
                    {
                        "heading": "Resource Allocation",
                        "type": "status_table",
                        "rows": [
                            {"component": "Frontend Development"},
                            {"component": "Backend Services"},
                            {"component": "Database Migration"},
                            {"component": "DevOps Setup"}
                        ]
                    }
                ]
            }
        ]
    
    def _meeting_templates(self) -> List[Dict[str, Any]]:
        """Meeting notes templates"""
        return [
            {
                "title_pattern": "{project} Team - Sprint {sprint} Retrospective - {date}",
                "category": "meeting-notes",
                "labels": ["meeting", "retrospective", "team"],
                "sections": [
                    {
                        "heading": "Attendees",
                        "type": "notes",
                        "content": ["Product Owner: Sarah Chen", "Scrum Master: Mike Wilson", "Development Team: 6 members"]
                    },
                    {
                        "heading": "What Went Well",
                        "type": "notes",
                        "content": [
                            {"type": "discussion", "content": "Automated testing reduced bug count by 40%"},
                            {"type": "discussion", "content": "Daily standups improved team communication"},
                            {"type": "discussion", "content": "Code review process caught critical issues early"}
                        ]
                    },
                    {
                        "heading": "Areas for Improvement",
                        "type": "notes",
                        "content": [
                            {"type": "discussion", "content": "Deployment process needs optimization"},
                            {"type": "discussion", "content": "Documentation falling behind development"},
                            {"type": "decision", "content": "Implement documentation-as-code approach"}
                        ]
                    },
                    {
                        "heading": "Action Items",
                        "type": "action_items",
                        "items": [
                            {"action": "Create deployment automation script"},
                            {"action": "Update API documentation"},
                            {"action": "Schedule knowledge sharing session"},
                            {"action": "Review and update coding standards"}
                        ]
                    }
                ]
            }
        ]
    
    def _release_templates(self) -> List[Dict[str, Any]]:
        """Release notes templates"""
        return [
            {
                "title_pattern": "{project} Release Notes - Version {version}",
                "category": "releases",
                "labels": ["release", "changelog", "version"],
                "sections": [
                    {
                        "heading": "Overview",
                        "type": "notes",
                        "content": [
                            f"This release includes significant performance improvements, new features, and bug fixes. All changes are backward compatible."
                        ]
                    },
                    {
                        "heading": "New Features",
                        "type": "notes",
                        "content": [
                            {"type": "discussion", "content": "Added real-time collaboration features"},
                            {"type": "discussion", "content": "Implemented advanced search functionality"},
                            {"type": "discussion", "content": "New dashboard with customizable widgets"}
                        ]
                    },
                    {
                        "heading": "Bug Fixes",
                        "type": "notes",
                        "content": [
                            "Fixed memory leak in data processing module",
                            "Resolved authentication timeout issues",
                            "Corrected timezone handling in reports"
                        ]
                    },
                    {
                        "heading": "Breaking Changes",
                        "type": "notes",
                        "content": [
                            {"type": "decision", "content": "API endpoint /v1/users deprecated, use /v2/users instead"},
                            {"type": "decision", "content": "Minimum Node.js version increased to 18.0"}
                        ]
                    },
                    {
                        "heading": "Migration Guide",
                        "type": "notes",
                        "content": [
                            "1. Update API endpoints in client applications",
                            "2. Run database migration script: migrate-v2.sql",
                            "3. Update environment variables as per new schema",
                            "4. Clear cache and restart services"
                        ]
                    }
                ]
            }
        ]
    
    def _howto_templates(self) -> List[Dict[str, Any]]:
        """How-to guide templates"""
        return [
            {
                "title_pattern": "How to Configure {project} for Production",
                "category": "how-to",
                "labels": ["how-to", "guide", "configuration"],
                "sections": [
                    {
                        "heading": "Prerequisites",
                        "type": "checklist",
                        "items": [
                            "Production server with minimum 8GB RAM",
                            "SSL certificate configured",
                            "Database backup strategy in place",
                            "Monitoring tools installed"
                        ]
                    },
                    {
                        "heading": "Step-by-Step Instructions",
                        "type": "notes",
                        "content": [
                            "1. Clone the repository to production server",
                            "2. Copy .env.production.example to .env",
                            "3. Configure environment variables",
                            "4. Run database migrations",
                            "5. Build production assets",
                            "6. Configure web server",
                            "7. Set up process monitoring",
                            "8. Verify deployment"
                        ]
                    }
                ]
            }
        ]
    
    def _faq_templates(self) -> List[Dict[str, Any]]:
        """FAQ templates"""
        return [
            {
                "title_pattern": "{project} Frequently Asked Questions",
                "category": "faq",
                "labels": ["faq", "support", "documentation"],
                "sections": [
                    {
                        "heading": "General Questions",
                        "type": "notes",
                        "content": [
                            {"type": "discussion", "content": "Q: What are the system requirements?\nA: Minimum 4GB RAM, 20GB storage, modern web browser"},
                            {"type": "discussion", "content": "Q: Is there a mobile app?\nA: Yes, available for iOS and Android"},
                            {"type": "discussion", "content": "Q: How often are updates released?\nA: Monthly feature updates, weekly security patches"} 
                        ]
                    }
                ]
            }
        ]
    
    def _best_practices_templates(self) -> List[Dict[str, Any]]:
        """Best practices templates"""
        return [
            {
                "title_pattern": "{project} Development Best Practices",
                "category": "best-practices",
                "labels": ["best-practices", "guidelines", "standards"],
                "sections": [
                    {
                        "heading": "Code Quality Standards",
                        "type": "checklist",
                        "items": [
                            "Follow agreed coding conventions",
                            "Write comprehensive unit tests",
                            "Document complex logic",
                            "Conduct code reviews",
                            "Use meaningful variable names"
                        ]
                    },
                    {
                        "heading": "Performance Guidelines",
                        "type": "notes",
                        "content": [
                            "Optimize database queries using indexes",
                            "Implement caching for frequently accessed data",
                            "Use lazy loading for heavy resources",
                            "Monitor and profile application performance"
                        ]
                    }
                ]
            }
        ]