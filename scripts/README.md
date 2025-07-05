# Confluence Test Data Generator

Generate realistic test data in Confluence using real documentation from public sources or dummy content.

## Quick Start

```bash
# 1. Setup (from confluence-gateway root)
cd scripts && uv pip install -r requirements.txt

# 2. Configure Confluence credentials
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_USERNAME="your-email@example.com"  
export CONFLUENCE_API_TOKEN="YOUR_API_TOKEN"

# 3. Collect real documentation
python real_data_collector.py collect

# 4. Generate test data in Confluence (uses real data by default)
python generate_dummy_data.py create
```

## Core Commands

### Data Collection
```bash
# Collect from all configured sources
python real_data_collector.py collect

# Collect from specific sources
python real_data_collector.py collect --sources github,web_docs

# Search collected content
python real_data_collector.py search --category technical --min-quality 0.7
```

### Data Generation
```bash
# Use real documentation (default)
python generate_dummy_data.py create

# Force dummy content (not recommended)
python generate_dummy_data.py create --dummy-data

# Additional options
--recreate              # Clean up existing data first
--dry-run              # Preview without creating
--no-reuse-if-exists   # Don't reuse existing test data
```

### Cleanup
```bash
# Clean up test spaces by pattern
python generate_dummy_data.py cleanup --pattern TESTDUM --confirm

# Clean up specific space
python generate_dummy_data.py cleanup --space-key TESTDUMTECH01011234
```

## Configuration

### Real Data Sources (`config/real_data/sources.yaml`)
Define sources for collecting real documentation:
- GitHub repositories (Python, FastAPI, Django docs)
- Web documentation sites (MDN, PostgreSQL)
- Open datasets and tutorials

### Generation Settings (`config/dummy_data_config.yaml`)
Configure space creation and content generation:
- Space prefixes and categories
- Page counts and attachment settings
- Safety features and quality thresholds

## Generated Content Structure

| Space | Category | Content Type |
|-------|----------|--------------|
| TECH | Technical Documentation | Installation guides, architecture docs, troubleshooting |
| API | API Documentation | REST/GraphQL references, webhooks, authentication |
| KB | Knowledge Base | How-to guides, FAQs, best practices |
| PROJECT | Project Documentation | Planning docs, meeting notes, release notes |
| MULTILANG | Multilingual Content | Documentation in multiple languages |

## Features

- **Real Data Mode**: Uses actual documentation from public sources for realistic testing
- **Quality Scoring**: Filters content based on readability and technical depth
- **Smart Reuse**: Detects and reuses existing test data to save time
- **Safe Cleanup**: Pattern-based deletion with prefix protection
- **Attachment Support**: Generates PDF, DOCX, XLSX, and image files