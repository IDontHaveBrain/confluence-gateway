#!/usr/bin/env python3

import re
import html
from typing import Dict, List, Any, Optional
from content_analyzer import ContentAnalyzer, ContentType, StructuredContent

class ConfluenceFormatter:
    def __init__(self):
        self.analyzer = ContentAnalyzer()
        
    def format_entry_for_confluence(self, entry: Dict[str, Any]) -> str:
        structured = self.analyzer.analyze_content(
            entry['content'], 
            {'title': entry['title'], 'name': entry['source']['name']}
        )
        
        if structured.content_type == ContentType.API_DOCUMENTATION:
            return self._format_api_documentation(entry, structured)
        elif structured.content_type == ContentType.TUTORIAL:
            return self._format_tutorial(entry, structured)
        elif structured.content_type == ContentType.TROUBLESHOOTING:
            return self._format_troubleshooting(entry, structured)
        else:
            return self._format_generic_content(entry, structured)

    def _format_api_documentation(self, entry: Dict[str, Any], structured: StructuredContent) -> str:
        sections = []
        
        sections.append(self._create_api_header(entry, structured))
        sections.append(self._create_toc())
        
        if structured.api_endpoints:
            sections.append(self._create_api_endpoints_summary(structured.api_endpoints))
        
        sections.append(self._format_main_content(entry, structured))
        
        if structured.code_blocks:
            sections.append(self._create_code_examples_section(structured.code_blocks))
        
        sections.append(self._create_footer(entry))
        
        return '\n\n'.join(sections)

    def _format_tutorial(self, entry: Dict[str, Any], structured: StructuredContent) -> str:
        sections = []
        
        # Tutorial Header
        sections.append(self._create_tutorial_header(entry, structured))
        
        # Progress Tracker
        sections.append(self._create_progress_tracker(structured.headers))
        
        # Table of Contents
        sections.append(self._create_toc())
        
        # Main Content with Step Numbers
        sections.append(self._format_tutorial_content(entry, structured))
        
        # Code Examples
        if structured.code_blocks:
            sections.append(self._create_code_examples_section(structured.code_blocks))
        
        # Footer
        sections.append(self._create_footer(entry))
        
        return '\n\n'.join(sections)

    def _format_troubleshooting(self, entry: Dict[str, Any], structured: StructuredContent) -> str:
        sections = []
        
        # Troubleshooting Header
        sections.append(self._create_troubleshooting_header(entry, structured))
        
        # Quick Solutions Panel
        sections.append(self._create_quick_solutions_panel(structured))
        
        # Table of Contents
        sections.append(self._create_toc())
        
        # Main Content
        sections.append(self._format_main_content(entry, structured))
        
        # Footer
        sections.append(self._create_footer(entry))
        
        return '\n\n'.join(sections)

    def _format_generic_content(self, entry: Dict[str, Any], structured: StructuredContent) -> str:
        sections = []
        
        # Standard Header
        sections.append(self._create_standard_header(entry, structured))
        
        # Table of Contents
        sections.append(self._create_toc())
        
        # Main Content
        sections.append(self._format_main_content(entry, structured))
        
        # Footer
        sections.append(self._create_footer(entry))
        
        return '\n\n'.join(sections)

    def _create_api_header(self, entry: Dict[str, Any], structured: StructuredContent) -> str:
        return f"""
<ac:layout>
<ac:layout-section ac:type="three_equal">
<ac:layout-cell>
<ac:structured-macro ac:name="info">
<ac:parameter ac:name="title">🔌 API Documentation</ac:parameter>
<ac:parameter ac:name="icon">true</ac:parameter>
<ac:rich-text-body>
<p><strong>Source:</strong> {html.escape(entry['source']['name'])}</p>
<p><strong>Endpoints:</strong> {len(structured.api_endpoints)}</p>
<p><strong>Code Examples:</strong> {len(structured.code_blocks)}</p>
</ac:rich-text-body>
</ac:structured-macro>
</ac:layout-cell>
<ac:layout-cell>
<ac:structured-macro ac:name="panel">
<ac:parameter ac:name="title">📊 Quick Stats</ac:parameter>
<ac:parameter ac:name="borderStyle">solid</ac:parameter>
<ac:parameter ac:name="borderColor">#0052cc</ac:parameter>
<ac:rich-text-body>
<p><strong>Word Count:</strong> {entry['content_info']['word_count']}</p>
<p><strong>Category:</strong> {', '.join(entry['categorization']['categories'])}</p>
<p><strong>Last Updated:</strong> {entry['collection_time'][:10]}</p>
</ac:rich-text-body>
</ac:structured-macro>
</ac:layout-cell>
<ac:layout-cell>
<ac:structured-macro ac:name="tip">
<ac:parameter ac:name="title">🔗 Original Source</ac:parameter>
<ac:rich-text-body>
<p><a href="{html.escape(entry['url'])}" class="external-link" rel="nofollow">View Original Documentation</a></p>
</ac:rich-text-body>
</ac:structured-macro>
</ac:layout-cell>
</ac:layout-section>
</ac:layout>
"""

    def _create_tutorial_header(self, entry: Dict[str, Any], structured: StructuredContent) -> str:
        return f"""
<ac:structured-macro ac:name="expand">
<ac:parameter ac:name="title">📚 Tutorial Overview</ac:parameter>
<ac:rich-text-body>
<ac:layout>
<ac:layout-section ac:type="two_equal">
<ac:layout-cell>
<ac:structured-macro ac:name="note">
<ac:parameter ac:name="title">📋 What You'll Learn</ac:parameter>
<ac:rich-text-body>
<p>This tutorial covers concepts from <strong>{html.escape(entry['source']['name'])}</strong></p>
<p><strong>Estimated Time:</strong> {self._estimate_reading_time(entry['content_info']['word_count'])} minutes</p>
<p><strong>Level:</strong> {self._detect_difficulty_level(entry['content'])}</p>
</ac:rich-text-body>
</ac:structured-macro>
</ac:layout-cell>
<ac:layout-cell>
<ac:structured-macro ac:name="info">
<ac:parameter ac:name="title">🎯 Prerequisites</ac:parameter>
<ac:rich-text-body>
<p>Review the <a href="{html.escape(entry['url'])}" class="external-link" rel="nofollow">original documentation</a> for prerequisites.</p>
</ac:rich-text-body>
</ac:structured-macro>
</ac:layout-cell>
</ac:layout-section>
</ac:layout>
</ac:rich-text-body>
</ac:structured-macro>
"""

    def _create_troubleshooting_header(self, entry: Dict[str, Any], structured: StructuredContent) -> str:
        return f"""
<ac:structured-macro ac:name="warning">
<ac:parameter ac:name="title">🚨 Troubleshooting Guide</ac:parameter>
<ac:rich-text-body>
<p>This guide helps resolve issues related to <strong>{html.escape(entry['source']['name'])}</strong></p>
<p>For additional help, visit the <a href="{html.escape(entry['url'])}" class="external-link" rel="nofollow">original documentation</a></p>
</ac:rich-text-body>
</ac:structured-macro>
"""

    def _create_standard_header(self, entry: Dict[str, Any], structured: StructuredContent) -> str:
        return f"""
<ac:structured-macro ac:name="panel">
<ac:parameter ac:name="title">📄 Document Information</ac:parameter>
<ac:parameter ac:name="borderStyle">solid</ac:parameter>
<ac:parameter ac:name="borderColor">#00875a</ac:parameter>
<ac:rich-text-body>
<p><strong>Source:</strong> {html.escape(entry['source']['name'])}</p>
<p><strong>Original URL:</strong> <a href="{html.escape(entry['url'])}" class="external-link" rel="nofollow">{html.escape(entry['url'])}</a></p>
<p><strong>Category:</strong> {', '.join(entry['categorization']['categories'])}</p>
<p><strong>Word Count:</strong> {entry['content_info']['word_count']}</p>
<p><strong>Collected:</strong> {entry['collection_time'][:10]}</p>
</ac:rich-text-body>
</ac:structured-macro>
"""

    def _create_toc(self) -> str:
        return """
<ac:structured-macro ac:name="toc">
<ac:parameter ac:name="printable">true</ac:parameter>
<ac:parameter ac:name="style">disc</ac:parameter>
<ac:parameter ac:name="maxLevel">4</ac:parameter>
<ac:parameter ac:name="minLevel">1</ac:parameter>
<ac:parameter ac:name="class">bigpanel</ac:parameter>
<ac:parameter ac:name="exclude">Table of Contents</ac:parameter>
<ac:parameter ac:name="type">list</ac:parameter>
<ac:parameter ac:name="outline">clear</ac:parameter>
<ac:parameter ac:name="include">.*</ac:parameter>
</ac:structured-macro>
"""

    def _create_api_endpoints_summary(self, endpoints: List[Dict[str, Any]]) -> str:
        if not endpoints:
            return ""
        
        endpoint_rows = []
        for endpoint in endpoints:
            method_color = {
                'GET': '#00875a',
                'POST': '#0052cc', 
                'PUT': '#ff8b00',
                'DELETE': '#de350b',
                'PATCH': '#6554c0'
            }.get(endpoint['method'], '#97a0af')
            
            endpoint_rows.append(f"""
<tr>
<td><ac:structured-macro ac:name="status">
<ac:parameter ac:name="colour">{method_color}</ac:parameter>
<ac:parameter ac:name="title">{endpoint['method']}</ac:parameter>
</ac:structured-macro></td>
<td><code>{html.escape(endpoint['path'])}</code></td>
</tr>
""")
        
        return f"""
<ac:structured-macro ac:name="expand">
<ac:parameter ac:name="title">🔌 API Endpoints ({len(endpoints)})</ac:parameter>
<ac:rich-text-body>
<table class="wrapped">
<tbody>
<tr>
<th>Method</th>
<th>Endpoint</th>
</tr>
{"".join(endpoint_rows)}
</tbody>
</table>
</ac:rich-text-body>
</ac:structured-macro>
"""

    def _create_progress_tracker(self, headers: List[Dict[str, Any]]) -> str:
        if len(headers) < 2:
            return ""
        
        steps = []
        for i, header in enumerate(headers[:5]):  # Limit to 5 steps
            steps.append(f"""
<ac:structured-macro ac:name="status">
<ac:parameter ac:name="colour">Grey</ac:parameter>
<ac:parameter ac:name="title">Step {i+1}: {html.escape(header['text'][:30])}</ac:parameter>
</ac:structured-macro>
""")
        
        return f"""
<ac:structured-macro ac:name="expand">
<ac:parameter ac:name="title">📋 Tutorial Progress</ac:parameter>
<ac:rich-text-body>
<p>{"".join(steps)}</p>
</ac:rich-text-body>
</ac:structured-macro>
"""

    def _create_quick_solutions_panel(self, structured: StructuredContent) -> str:
        # Extract quick solutions from headers or content
        solutions = []
        for header in structured.headers:
            if any(word in header['text'].lower() for word in ['fix', 'solve', 'solution']):
                solutions.append(f"• {header['text']}")
        
        if not solutions:
            solutions = ["• Check the original documentation for detailed solutions"]
        
        solutions_text = '\n'.join(solutions[:5])  # Limit to 5 solutions
        
        return f"""
<ac:structured-macro ac:name="tip">
<ac:parameter ac:name="title">⚡ Quick Solutions</ac:parameter>
<ac:rich-text-body>
{solutions_text}
</ac:rich-text-body>
</ac:structured-macro>
"""

    def _create_code_examples_section(self, code_blocks: List[Dict[str, Any]]) -> str:
        if not code_blocks:
            return ""
        
        code_examples = []
        for i, block in enumerate(code_blocks):
            title = f"Example {i+1}"
            if block['is_curl']:
                title += " (cURL Command)"
            elif block['language'] != 'text':
                title += f" ({block['language'].upper()})"
            
            code_examples.append(f"""
<ac:structured-macro ac:name="code">
<ac:parameter ac:name="language">{block['language']}</ac:parameter>
<ac:parameter ac:name="title">{title}</ac:parameter>
<ac:parameter ac:name="linenumbers">true</ac:parameter>
<ac:parameter ac:name="collapse">false</ac:parameter>
<ac:plain-text-body><![CDATA[{block['code']}]]></ac:plain-text-body>
</ac:structured-macro>
""")
        
        return f"""
<ac:structured-macro ac:name="expand">
<ac:parameter ac:name="title">💻 Code Examples ({len(code_blocks)})</ac:parameter>
<ac:rich-text-body>
{"".join(code_examples)}
</ac:rich-text-body>
</ac:structured-macro>
"""

    def _format_main_content(self, entry: Dict[str, Any], structured: StructuredContent) -> str:
        content = entry['content']
        
        # Since content is now in markdown format, process it more simply
        content = self._process_markdown_content(content)
        
        return f"""
<hr/>
{content}
"""

    def _format_tutorial_content(self, entry: Dict[str, Any], structured: StructuredContent) -> str:
        content = entry['content']
        
        # Add step numbers to headers for tutorials
        content = self._add_step_numbers_to_headers(content)
        # Process as markdown content
        content = self._process_markdown_content(content)
        
        return f"""
<hr/>
{content}
"""

    def _process_markdown_content(self, content: str) -> str:
        """Process markdown content for Confluence with clean conversion"""
        # Convert markdown headers to Confluence headers
        content = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', content, flags=re.MULTILINE)
        content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)  
        content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
        content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
        
        # Convert markdown bold/italic to HTML
        content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', content)
        
        # Convert inline code (avoid interfering with code blocks)
        content = re.sub(r'(?<!`)`([^`]+)`(?!`)', r'<code>\1</code>', content)
        
        # Convert markdown code blocks to Confluence code macros
        content = re.sub(
            r'```(\w*)\n?(.*?)\n?```',
            lambda m: f'<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">{m.group(1) or "text"}</ac:parameter><ac:plain-text-body><![CDATA[{m.group(2)}]]></ac:plain-text-body></ac:structured-macro>',
            content,
            flags=re.DOTALL
        )
        
        # Convert markdown links (avoid interfering with existing HTML links)
        content = re.sub(
            r'(?<!href=")\[([^\]]+)\]\(([^)]+)\)',
            r'<a href="\2" class="external-link" rel="nofollow">\1</a>',
            content
        )
        
        # Process lists - convert markdown to HTML
        lines = content.split('\n')
        processed_lines = []
        in_ul = False
        in_ol = False
        
        for line in lines:
            stripped = line.strip()
            
            # Unordered list
            if re.match(r'^[-*+]\s+', stripped):
                if not in_ul:
                    if in_ol:
                        processed_lines.append('</ol>')
                        in_ol = False
                    processed_lines.append('<ul>')
                    in_ul = True
                item_text = re.sub(r'^[-*+]\s+', '', stripped)
                processed_lines.append(f'<li>{item_text}</li>')
            
            # Ordered list  
            elif re.match(r'^\d+\.\s+', stripped):
                if not in_ol:
                    if in_ul:
                        processed_lines.append('</ul>')
                        in_ul = False
                    processed_lines.append('<ol>')
                    in_ol = True
                item_text = re.sub(r'^\d+\.\s+', '', stripped)
                processed_lines.append(f'<li>{item_text}</li>')
            
            else:
                # Close any open lists
                if in_ul:
                    processed_lines.append('</ul>')
                    in_ul = False
                if in_ol:
                    processed_lines.append('</ol>')
                    in_ol = False
                processed_lines.append(line)
        
        # Close any remaining open lists
        if in_ul:
            processed_lines.append('</ul>')
        if in_ol:
            processed_lines.append('</ol>')
        
        content = '\n'.join(processed_lines)
        
        # Highlight API terms
        api_terms = ['POST', 'GET', 'PUT', 'DELETE', 'PATCH', 'API', 'JSON', 'REST']
        for term in api_terms:
            # Only highlight if not already in tags
            pattern = f'\\b{term}\\b(?![^<]*>)'
            content = re.sub(pattern, f'<strong>{term}</strong>', content)
        
        return content

    def _add_step_numbers_to_headers(self, content: str) -> str:
        # Add step numbers to tutorial headers
        lines = content.split('\n')
        step_count = 0
        
        for i, line in enumerate(lines):
            if re.match(r'^##?\s+', line):
                step_count += 1
                lines[i] = re.sub(r'^(##?\s+)', f'\\1Step {step_count}: ', line)
        
        return '\n'.join(lines)

    def _create_footer(self, entry: Dict[str, Any]) -> str:
        return f"""
<hr/>
<ac:structured-macro ac:name="note">
<ac:parameter ac:name="title">📚 Reference</ac:parameter>
<ac:rich-text-body>
<p>This content was collected from: <a href="{html.escape(entry['url'])}" class="external-link" rel="nofollow">{html.escape(entry['source']['name'])}</a></p>
<p>Last collected: {entry['collection_time'][:10]}</p>
<p>Content ID: <code>{entry['id']}</code></p>
</ac:rich-text-body>
</ac:structured-macro>
"""

    def _estimate_reading_time(self, word_count: int) -> int:
        # Average reading speed: 200 words per minute
        return max(1, word_count // 200)

    def _detect_difficulty_level(self, content: str) -> str:
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['advanced', 'complex', 'enterprise']):
            return "Advanced"
        elif any(word in content_lower for word in ['intermediate', 'overview']):
            return "Intermediate"
        else:
            return "Beginner"