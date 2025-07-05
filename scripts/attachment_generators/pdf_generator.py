"""PDF attachment generator"""

import io
import random
from typing import Dict, Any
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

from .metadata import ATTACHMENT_METADATA

class PDFGenerator:
    """Generates PDF attachments with various content types"""
    
    def __init__(self, config):
        self.config = config
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY
        ))
    
    def generate(self, category: str) -> Dict[str, Any]:
        """Generate PDF content based on category"""
        # Select appropriate template
        templates = {
            "api": self._generate_api_spec,
            "rest": self._generate_api_spec,
            "graphql": self._generate_api_spec,
            "technical": self._generate_technical_doc,
            "installation": self._generate_technical_doc,
            "architecture": self._generate_technical_doc,
            "project": self._generate_project_doc,
            "planning": self._generate_project_doc,
            "releases": self._generate_project_doc
        }
        
        generator_func = templates.get(category, self._generate_technical_doc)

        category_key = category if category in ["api", "technical", "project"] else "technical"
        filenames = ATTACHMENT_METADATA["pdf"].get(category_key, ["Document.pdf"])
        filename = f"{self.config.prefix}_{random.choice(filenames)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Generate PDF
        buffer = io.BytesIO()
        content = generator_func(buffer, filename, category)
        
        return {
            "filename": filename,
            "content": buffer.getvalue(),
            "content_type": "application/pdf",
            "size": buffer.tell()
        }
    
    def _generate_api_spec(self, buffer: io.BytesIO, filename: str, category: str) -> None:
        """Generate API specification PDF"""
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []

        story.append(Paragraph("API Specification Document", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Overview
        story.append(Paragraph("1. Overview", self.styles['Heading2']))
        story.append(Paragraph(
            "This document provides a comprehensive specification for the REST API endpoints, "
            "including authentication methods, request/response formats, and error handling procedures. "
            "The API follows RESTful principles and uses JSON for data exchange.",
            self.styles['CustomBody']
        ))
        story.append(Spacer(1, 0.3*inch))
        
        # Authentication
        story.append(Paragraph("2. Authentication", self.styles['Heading2']))
        story.append(Paragraph(
            "All API requests require authentication using Bearer tokens. Include the token in the "
            "Authorization header: <b>Authorization: Bearer YOUR_TOKEN</b>",
            self.styles['CustomBody']
        ))
        story.append(Spacer(1, 0.3*inch))
        
        # Endpoints table
        story.append(Paragraph("3. API Endpoints", self.styles['Heading2']))
        
        endpoints_data = [
            ['Method', 'Endpoint', 'Description', 'Auth Required'],
            ['GET', '/api/v1/users', 'List all users', 'Yes'],
            ['POST', '/api/v1/users', 'Create new user', 'Yes'],
            ['GET', '/api/v1/users/{id}', 'Get user details', 'Yes'],
            ['PUT', '/api/v1/users/{id}', 'Update user', 'Yes'],
            ['DELETE', '/api/v1/users/{id}', 'Delete user', 'Yes'],
            ['GET', '/api/v1/health', 'Health check', 'No'],
        ]
        
        table = Table(endpoints_data, colWidths=[1*inch, 2*inch, 2.5*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        # Response format
        story.append(Paragraph("4. Response Format", self.styles['Heading2']))
        story.append(Paragraph(
            "All successful responses follow a standard format with data wrapped in a 'data' field. "
            "Errors include an 'error' object with code and message fields.",
            self.styles['CustomBody']
        ))
        
        # Add some code examples
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Success Response Example:", self.styles['Heading3']))
        code_style = ParagraphStyle('Code', fontName='Courier', fontSize=10, leftIndent=20)
        story.append(Paragraph(
            '''<font name="Courier" size="10">{
    "data": {
        "id": 123,
        "name": "John Doe",
        "email": "john@example.com"
    },
    "meta": {
        "timestamp": "2024-01-15T10:00:00Z"
    }
}</font>''',
            code_style
        ))
        
        # Build PDF
        doc.build(story)
    
    def _generate_technical_doc(self, buffer: io.BytesIO, filename: str, category: str) -> None:
        """Generate technical documentation PDF"""
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []

        story.append(Paragraph("Technical Architecture Document", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # System overview
        story.append(Paragraph("System Architecture Overview", self.styles['Heading2']))
        story.append(Paragraph(
            "This document describes the technical architecture of our microservices-based system. "
            "The architecture is designed for scalability, reliability, and maintainability, "
            "utilizing cloud-native technologies and best practices.",
            self.styles['CustomBody']
        ))
        story.append(Spacer(1, 0.3*inch))
        
        # Components
        story.append(Paragraph("Core Components", self.styles['Heading2']))
        
        components = [
            ("API Gateway", "Routes and authenticates all external requests"),
            ("Service Mesh", "Handles inter-service communication and observability"),
            ("Message Queue", "Asynchronous communication between services"),
            ("Cache Layer", "Redis-based caching for performance optimization"),
            ("Database Cluster", "PostgreSQL with read replicas for data persistence"),
        ]
        
        for name, desc in components:
            story.append(Paragraph(f"<b>{name}</b>: {desc}", self.styles['CustomBody']))
            story.append(Spacer(1, 0.1*inch))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Technical specifications
        story.append(Paragraph("Technical Specifications", self.styles['Heading2']))
        
        specs_data = [
            ['Component', 'Technology', 'Version', 'Purpose'],
            ['Container Runtime', 'Docker', '24.0', 'Application containerization'],
            ['Orchestration', 'Kubernetes', '1.28', 'Container orchestration'],
            ['Service Mesh', 'Istio', '1.19', 'Service communication'],
            ['API Gateway', 'Kong', '3.4', 'API management'],
            ['Database', 'PostgreSQL', '15.0', 'Data persistence'],
            ['Cache', 'Redis', '7.2', 'Performance optimization'],
        ]
        
        table = Table(specs_data, colWidths=[2*inch, 1.5*inch, 1*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        
        # Build PDF
        doc.build(story)
    
    def _generate_project_doc(self, buffer: io.BytesIO, filename: str, category: str) -> None:
        """Generate project documentation PDF"""
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []

        story.append(Paragraph("Project Status Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive summary
        story.append(Paragraph("Executive Summary", self.styles['Heading2']))
        story.append(Paragraph(
            "This report provides an overview of the current project status, including completed milestones, "
            "ongoing activities, and upcoming deliverables. The project is currently on track to meet "
            "the planned delivery date with all critical path items progressing as expected.",
            self.styles['CustomBody']
        ))
        story.append(Spacer(1, 0.3*inch))
        
        # Project metrics
        story.append(Paragraph("Project Metrics", self.styles['Heading2']))
        
        metrics_data = [
            ['Metric', 'Target', 'Actual', 'Status'],
            ['Schedule', '100%', '85%', 'On Track'],
            ['Budget', '$500K', '$425K', 'Under Budget'],
            ['Scope', '100%', '90%', 'On Track'],
            ['Quality', '< 5 defects', '3 defects', 'Exceeding'],
            ['Team Velocity', '40 pts', '42 pts', 'Exceeding'],
        ]
        
        table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#eff6ff')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        # Milestones
        story.append(Paragraph("Completed Milestones", self.styles['Heading2']))
        milestones = [
            "✓ Requirements gathering and analysis phase completed",
            "✓ System architecture design approved by stakeholders",
            "✓ Development environment setup and CI/CD pipeline configured",
            "✓ Core API functionality implemented and tested",
            "✓ Security audit passed with no critical findings",
        ]
        
        for milestone in milestones:
            story.append(Paragraph(milestone, self.styles['CustomBody']))
            story.append(Spacer(1, 0.05*inch))
        
        story.append(PageBreak())
        
        # Risk assessment
        story.append(Paragraph("Risk Assessment", self.styles['Heading2']))
        story.append(Paragraph(
            "The following risks have been identified and mitigation strategies are in place:",
            self.styles['CustomBody']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        risks_data = [
            ['Risk', 'Probability', 'Impact', 'Mitigation'],
            ['Third-party API changes', 'Medium', 'High', 'Version locking and monitoring'],
            ['Key personnel availability', 'Low', 'High', 'Cross-training and documentation'],
            ['Performance requirements', 'Medium', 'Medium', 'Early load testing and optimization'],
            ['Security vulnerabilities', 'Low', 'High', 'Regular security scans and updates'],
        ]
        
        risk_table = Table(risks_data, colWidths=[2*inch, 1.2*inch, 1*inch, 2.8*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc2626')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fef2f2')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        
        story.append(risk_table)
        
        # Build PDF
        doc.build(story)