# Confluence Dummy Data Generator

Generate test data in Confluence for testing search capabilities (keyword, semantic, CQL, hybrid) of Confluence Gateway.

## Installation

```bash
# From confluence-gateway root
uv pip install -e .

# Install script dependencies
cd scripts
uv pip install -r requirements.txt
```

## Configuration

Set Confluence credentials via environment variables or `~/.confluence_gateway_config.json`:

```bash
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_USERNAME="your-email@example.com"
export CONFLUENCE_API_TOKEN="YOUR_API_TOKEN"
```

## Usage

### Generate Test Data
```bash
# Create new test data
python generate_dummy_data.py create

# Force recreate (cleanup existing first)
python generate_dummy_data.py create --recreate

# Create alongside existing data
python generate_dummy_data.py create --no-reuse-if-exists

# Dry run mode
python generate_dummy_data.py create --dry-run
```

### Cleanup
```bash
# Clean up all spaces matching a pattern (queries Confluence directly)
python generate_dummy_data.py cleanup --pattern TESTDUM --confirm

# Clean up specific space
python generate_dummy_data.py cleanup --space-key TESTDUMTECH01011234
```

## Generated Content

- **Spaces**: TECH, API, KB, PROJECT, MULTILANG
- **Content Types**: Technical docs, API docs, Knowledge base, Project docs, Multilingual content
- **Attachments**: PDF, DOCX, XLSX, PNG/JPG files
- **Search Optimization**: Content optimized for keyword, semantic, CQL, and hybrid search

## Key Features

- **Smart Data Reuse**: Prevents duplicate generation, reuses existing valid test data
- **Pattern-Based Cleanup**: Query and clean up spaces directly from Confluence by pattern matching
- **Safety Features**: Unique prefix protection, timestamp suffixes for uniqueness
- **Flexible Configuration**: Customize page counts, content types, and attachment settings

## Configuration

Customize generation in `config/dummy_data_config.yaml` - adjust page counts, content settings, attachment types, and safety options.