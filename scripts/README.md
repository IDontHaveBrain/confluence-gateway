# Confluence Gateway Test Data Scripts

Collects real documentation and generates formatted Confluence test data.

## Quick Start

```bash
# Setup
cd scripts && uv pip install -r requirements.txt

# Configure credentials
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_USERNAME="your-email@example.com"  
export CONFLUENCE_API_TOKEN="YOUR_API_TOKEN"

# Collect and generate data
uv run python real_data_collector.py collect
uv run python generate_real_data.py create
```

## Scripts

### Real Data Collection (`real_data_collector.py`)
Collects technical documentation from live websites.

```bash
uv run python real_data_collector.py collect  # Collect from sources
uv run python real_data_collector.py stats    # Show statistics
```

### Enhanced Data Generation (`generate_real_data.py`)
Creates Confluence spaces with advanced formatting.

```bash
uv run python generate_real_data.py create --dry-run  # Preview
uv run python generate_real_data.py create           # Generate
uv run python generate_real_data.py cleanup --confirm # Clean up
```

### Content Analysis (`content_analyzer.py`)
Analyzes and classifies collected content structure.

```bash
uv run python content_analyzer.py  # Analyze collected data
```

## Configuration

### Main Files
- `config/real_data/sources.yaml` - Web documentation sources
- `config/production_config.yaml` - Collection and generation settings

### Key Settings
- Collection targets per category (20+ documents each)
- Duplicate detection via content hashing
- Rate limiting (30 requests/minute)
- Space configuration (TESTDUM prefix)

## Features

### Collection
- Live web scraping from documentation sites
- Automatic content classification (API docs, tutorials, troubleshooting)
- PDF detection and skipping
- Progress tracking with visual feedback

### Generation
- Content-aware Confluence formatting
- Rich macros (panels, code blocks, status indicators)
- Interactive elements (expandable sections, progress trackers)
- API endpoint detection and color-coded display
- Multi-column layouts optimized per content type

### Generated Spaces
- `TESTDUMAPI` - API Documentation (20 pages)
- `TESTDUMTECH` - Technical Documentation (25 pages)  
- `TESTDUMKB` - Knowledge Base (20 pages)

## Troubleshooting

**Collection Issues:**
- Reduce `max_depth` in sources.yaml for timeouts
- Verify URLs are accessible (403/404 errors)
- PDFs are automatically skipped

**Generation Issues:**  
- Set required environment variables
- Verify Confluence URL and token permissions
- Use `--dry-run` to preview before creation

**Cleanup:**
- Use `cleanup --confirm` for safe removal
- Check `confluence_generation_tracking.json` for records