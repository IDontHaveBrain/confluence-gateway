# Confluence Gateway

**AI-powered search and knowledge retrieval for Atlassian Confluence**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status**: Beta (v0.1.0)

Transform your Confluence into a smart knowledge base with semantic search, hybrid algorithms, and AI-powered question answering.

## ✨ Key Features

- **🔍 Advanced Search**: Text, semantic, and hybrid search with Reciprocal Rank Fusion
- **🤖 RAG-Powered Q&A**: Generate contextual answers from your Confluence content  
- **⚡ Flexible Integration**: REST API + CLI interface with multiple vector database support

## 🚀 Quick Start

**Requirements**: Python 3.10+, Confluence API access

### 1. Install

```bash
# Clone repository and install dependencies
git clone https://github.com/IDontHaveBrain/confluence-gateway.git
cd confluence-gateway
uv sync --dev
uv run pre-commit install
```

### 2. Configure

**Environment variables (recommended):**
```bash
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_USERNAME="your-email@example.com"
export CONFLUENCE_API_TOKEN="YOUR_API_TOKEN"

# Optional: AI features (default: openrouter/google/gemini-2.5-flash)
export GENERATION_MODEL_NAME="openrouter/google/gemini-2.5-flash"
export GENERATION_LITELLM_API_KEY="YOUR_OPENROUTER_API_KEY"

# Optional: Vector database (default: memory mode)
export QDRANT_URL="http://localhost:6333"  # or ":memory:"
```

**Or config file at `~/.confluence_gateway_config.json`:**
```json
{
  "confluence": {
    "url": "https://your-instance.atlassian.net",
    "username": "your-email@example.com",
    "api_token": "YOUR_API_TOKEN"
  }
}
```

### 3. Use It

**CLI Interface:**
```bash
# Verify installation
uv run confluence-gateway --version

# List and manage spaces
uv run confluence-gateway spaces list --all
uv run confluence-gateway spaces list --search "dev"

# Index content
uv run confluence-gateway index trigger --space-keys DEV,TECH
uv run confluence-gateway index status

# Search with various modes
uv run confluence-gateway search text "deployment guide"
uv run confluence-gateway search semantic "how to deploy"
uv run confluence-gateway search text "process" --hybrid

# Get AI answers
uv run confluence-gateway generate answer "What is our deployment process?"
```

**API Server:**
```bash
# Start development server
uv run uvicorn confluence_gateway.api.app:app --reload
# API docs: http://localhost:8000/docs

# Health check
curl "http://localhost:8000/health"

# List spaces
curl "http://localhost:8000/api/spaces"

# Text search
curl "http://localhost:8000/api/search?query=deployment&limit=10"

# Semantic search (note the nested request structure)
curl -X POST "http://localhost:8000/api/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"search_request": {"query": "deployment process", "top_k": 5}}'

# Generate answers (note the nested request structure)
curl -X POST "http://localhost:8000/api/generate/answer" \
  -H "Content-Type: application/json" \
  -d '{"gen_request": {"query": "What is our deployment process?"}}'
```

## 🏗️ Architecture

**Hexagonal Architecture** (Ports and Adapters):

```
confluence_gateway/
├── core/              # Configuration, exceptions, schemas
├── services/          # Business logic (Search, Indexing, Generation, Ranking)
├── adapters/          # External integrations (Confluence, Vector DBs, Embeddings)
├── api/               # FastAPI REST interface
└── cli/               # Typer CLI interface
```

**Key Services:**
- `SearchService` - Multi-modal search with RRF hybrid ranking
- `IndexingService` - Content processing and vector storage  
- `GenerationService` - RAG answer generation
- `EmbeddingService` - Vector embedding management
- `RankingService` - Reciprocal Rank Fusion algorithm

## 🔧 Development & Testing

**Development workflow:**
```bash
# Start API server with auto-reload
uv run uvicorn confluence_gateway.api.app:app --reload

# Code quality pipeline (required before commits)  
uv run ruff check --fix && uv run ruff format && uv run mypy confluence_gateway/

# Run E2E tests (requires real Confluence instance)
echo '{"vector_db": {"qdrant_url": ":memory:", "qdrant_local_path": null}}' > ~/.confluence_gateway_config.json
uv run pytest tests/ -v
```

**Testing philosophy**: E2E only - test real functionality with actual Confluence instances. Requires `CONFLUENCE_URL`, `CONFLUENCE_USERNAME`, `CONFLUENCE_API_TOKEN` environment variables.

## 📚 API Reference

**API Docs**: `http://localhost:8000/docs` (interactive Swagger UI)

**Critical**: All POST endpoints require nested request objects:

```json
# Semantic Search
{"search_request": {"query": "deployment", "top_k": 10}}

# Answer Generation  
{"gen_request": {"query": "What is our process?", "top_k_retrieval": 5}}

# Advanced Search
{"request": {"query": "api", "space_key": "TECH", "limit": 20}}

# Indexing
{"space_keys": ["DEV", "TECH"]} 
{"index_all": true}
```

## ⚙️ Configuration

**Priority**: Environment Variables > `~/.confluence_gateway_config.json` > Defaults

### Vector Database Options

**Qdrant (Default):**
```bash
# Local storage (persistent)
export QDRANT_LOCAL_PATH="~/.confluence_gateway/qdrant_storage"

# Server mode  
export QDRANT_URL="http://localhost:6333"
```

**ChromaDB:**
```bash
export VECTOR_DB_TYPE="chroma"
export CHROMA_PERSIST_PATH="~/.confluence_gateway/chroma_storage"
```

### AI Generation Settings

```bash
# Default model (requires OpenRouter API key from https://openrouter.ai/)
export GENERATION_MODEL_NAME="openrouter/google/gemini-2.5-flash"
export GENERATION_LITELLM_API_KEY="YOUR_API_KEY"
```

## 🔒 Production Considerations

⚠️ **Security**: No built-in authentication - use reverse proxy with auth for production  
📦 **Package Manager**: Use `uv` (not pip) for all operations  
🗄️ **Vector Databases**: Qdrant (default), ChromaDB, or text-only mode  
🤖 **AI Providers**: LiteLLM with OpenAI, Anthropic, OpenRouter, and more

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

*Transform your Confluence documentation into an intelligent, searchable knowledge base with AI-powered insights.*