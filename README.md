# Confluence Gateway

**AI-powered search and knowledge retrieval for Atlassian Confluence**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status**: Beta (v0.1.0)

Transform your Confluence into a smart knowledge base with semantic search, hybrid algorithms, and AI-powered question answering.

## ✨ Key Features

- **🔍 Advanced Search**: Text, semantic, and hybrid search with Reciprocal Rank Fusion
- **🤖 RAG-Powered Q&A**: Generate contextual answers from your Confluence content  
- **⚡ Dual Interface**: CLI for automation + REST API for integration
- **🚀 GPU Acceleration**: Auto-detected GPU support with 5-10x performance boost
- **🗄️ Flexible Storage**: Qdrant, ChromaDB, or memory-only mode

## 🚀 Quick Start

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

# GPU control (auto-detected by default)  
export EMBEDDING_DEVICE="cuda"    # Force GPU

# Vector storage
export QDRANT_URL="http://localhost:6333"  # Persistent storage
```

### 3. Verify Connection

```bash
# Test installation and connection
uv run confluence-gateway --version
uv run confluence-gateway spaces list
```

✅ **Success**: You should see your Confluence spaces listed without errors.

## 🔧 Troubleshooting

**Common issues:**
```bash
# Connection issues
uv run confluence-gateway spaces list --verbose

# Test configuration
uv run python -c "from confluence_gateway.core.config import get_confluence_config; print(get_confluence_config())"

# Reset to development mode
export CONFLUENCE_GATEWAY_DEV_MODE="true"
```

### 4. Choose Your Interface

**CLI for automation and scripting:**
```bash
# Index content
uv run confluence-gateway index trigger --space-keys DEV,TECH

# Search with various modes  
uv run confluence-gateway search text "deployment guide"
uv run confluence-gateway search semantic "how to deploy"

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

## 🏗️ Architecture

**Hexagonal Architecture** with clean separation: `core/` (config), `services/` (business logic), `adapters/` (external integrations), `api/` (REST), `cli/` (commands).

**Key Services**: Multi-modal search, content indexing, RAG generation, vector embeddings, and RRF ranking.

## 🔧 Development

**Setup & Quality:**
```bash
# Install with development dependencies
uv sync --dev

# Code quality (run before commits)
uv run ruff check --fix && uv run ruff format && uv run mypy confluence_gateway/

# Run tests (requires Confluence auth)
uv run pytest tests/ -v
```

**Development server:**
```bash
# API with auto-reload + faster startup
export CONFLUENCE_GATEWAY_DEV_MODE="true"
uv run uvicorn confluence_gateway.api.app:app --reload
```

## 📚 API Reference

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

## ⚙️ Configuration

**Config file** (`~/.confluence_gateway_config.json`):
```json
{
  "confluence": {
    "url": "https://your-instance.atlassian.net",
    "username": "your-email@example.com",
    "api_token": "YOUR_API_TOKEN"
  },
  "vector_db": {
    "type": "qdrant",
    "qdrant_url": "http://localhost:6333"
  },
  "embedding": {
    "device": "cuda"  // "cpu" for CPU-only
  }
}
```

**Priority**: Config file > Environment variables > Defaults

**Storage options**: Qdrant (default), ChromaDB (`VECTOR_DB_TYPE=chroma`), or memory-only for testing

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

*Transform your Confluence documentation into an intelligent, searchable knowledge base with AI-powered insights.*