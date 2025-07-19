# Confluence Gateway

**AI-powered search and knowledge retrieval for Atlassian Confluence**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status**: Beta (v0.1.0)

AI-powered search and knowledge retrieval for Atlassian Confluence with semantic search, hybrid algorithms, and RAG-powered Q&A.

## Features

- **Advanced Search**: Text, semantic, and hybrid search with Reciprocal Rank Fusion
- **RAG-Powered Q&A**: Generate contextual answers from your Confluence content  
- **Dual Interface**: CLI for automation + REST API for integration
- **GPU Acceleration**: Auto-detected GPU support with performance boost
- **Flexible Storage**: Qdrant, ChromaDB, or memory-only mode

## Quick Start

**Requirements**: Python 3.10+, Confluence API access

### 1. Install

```bash
git clone https://github.com/IDontHaveBrain/confluence-gateway.git
cd confluence-gateway
uv sync --dev
```

### 2. Configure

Set up your Confluence credentials:

```bash
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_USERNAME="your-email@example.com"
export CONFLUENCE_API_TOKEN="YOUR_API_TOKEN"
```

**Optional configuration:**
```bash
# AI features (requires API key)
export GENERATION_MODEL_NAME="openrouter/google/gemini-2.5-flash"
export GENERATION_LITELLM_API_KEY="YOUR_OPENROUTER_API_KEY"

# Vector storage (for semantic/hybrid search)
export VECTOR_DB_TYPE="chroma"  # or "qdrant"
export QDRANT_LOCAL_PATH="~/.confluence_gateway/qdrant_storage"

# GPU control (auto-detected by default)  
export EMBEDDING_DEVICE="cuda"    # Force GPU
```

### 3. Verify Connection

```bash
# Test installation and connection
uv run confluence-gateway --version
uv run confluence-gateway spaces list
```

You should see your Confluence spaces listed without errors.

### 4. Choose Your Interface

**CLI for automation and scripting:**
```bash
# Index content
uv run confluence-gateway index trigger --space DEV --space TECH

# Search with various modes  
uv run confluence-gateway search text "deployment guide"
uv run confluence-gateway search semantic "how to deploy"          # Requires vector DB
uv run confluence-gateway search hybrid "deployment guide"         # Combines keyword + semantic

# Get AI answers (requires AI configuration)
uv run confluence-gateway generate answer "What is our deployment process?"
```

**API for integration and web apps:**
```bash
# Start development server
uv run uvicorn confluence_gateway.api.app:app --reload

# Interactive docs: http://localhost:8000/docs
# Health check: curl "http://localhost:8000/health"
```

## API Reference

**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)

**Quick API test:**
```bash
# Text search (GET)
curl "http://localhost:8000/api/search?query=deployment&limit=5"

# Semantic search (POST)
curl -X POST "http://localhost:8000/api/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "deployment process", "top_k": 5}'

# Health check
curl "http://localhost:8000/health"
```

## Configuration

**Config file** (`~/.confluence_gateway_config.json`):
```json
{
  "confluence": {
    "url": "https://your-instance.atlassian.net",
    "username": "your-email@example.com",
    "api_token": "YOUR_API_TOKEN"
  },
  "vector_db": {"type": "qdrant"},
  "embedding": {"device": "cuda"}
}
```

**Priority**: Config file > Environment variables > Defaults

## Development

**Setup & Quality:**
```bash
# Install with development dependencies
uv sync --dev

# Code quality (run before commits)
uv run ruff check --fix && uv run ruff format && uv run mypy confluence_gateway/
uv run vulture confluence_gateway/ vulture_whitelist.py --min-confidence 90

# Or use pre-commit for automated quality checks
uv run pre-commit install
uv run pre-commit run --all-files

# Run tests (requires Confluence auth)
uv run pytest tests/ -v
```

**Development server:**
```bash
# Fast startup for development
export CONFLUENCE_GATEWAY_DEV_MODE="true"
uv run uvicorn confluence_gateway.api.app:app --reload
```

## Troubleshooting

**Common issues:**

1. **Vector DB Configuration Warning**: 
   ```
   Invalid Vector DB configuration: In-memory mode (:memory:) is only allowed during testing
   ```
   **Solution**: Set `VECTOR_DB_TYPE="chroma"` for semantic/hybrid features.

2. **Connection issues**:
   ```bash
   uv run confluence-gateway spaces list --verbose
   ```

3. **Slow startup**:
   ```bash
   export CONFLUENCE_GATEWAY_DEV_MODE="true"
   ```

## Architecture

**Hexagonal Architecture** with clean separation of concerns: core configuration, business services, external adapters, REST API, and CLI interface.

## License

MIT License - see [LICENSE](LICENSE) file