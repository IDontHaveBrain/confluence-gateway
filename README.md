# Confluence Gateway

**AI-powered search and knowledge retrieval for Atlassian Confluence**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status**: Beta (v0.1.0)

Transform your Confluence into a smart knowledge base with semantic search, hybrid algorithms, and AI-powered question answering.

## ✨ Key Features

- **🔍 Advanced Search**: Text, semantic, and hybrid search with Reciprocal Rank Fusion
- **🤖 RAG-Powered Q&A**: Generate contextual answers from your Confluence content  
- **⚡ Dual Interface**: CLI for automation + REST API for integration
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

**Optional**: Enable AI features (requires API key)
```bash
export GENERATION_MODEL_NAME="openrouter/google/gemini-2.5-flash"
export GENERATION_LITELLM_API_KEY="YOUR_OPENROUTER_API_KEY"
```

### 3. Verify Connection

```bash
# Test installation and connection
uv run confluence-gateway --version
uv run confluence-gateway spaces list
```

✅ **Success**: You should see your Confluence spaces listed without errors.

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

## 🔧 Development

**Essential commands:**
```bash
# Code quality (run before commits)
uv run ruff check --fix && uv run ruff format && uv run mypy confluence_gateway/

# Run tests
echo '{"vector_db": {"qdrant_url": ":memory:", "qdrant_local_path": null}}' > ~/.confluence_gateway_config.json
uv run pytest tests/ -v

# Start API with auto-reload
uv run uvicorn confluence_gateway.api.app:app --reload
```

**Note**: Tests use real Confluence instances and require authentication environment variables.

## 📚 API Reference

**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)

**Key endpoint request formats:**

```json
# Semantic Search
{"query": "deployment", "top_k": 10}

# Answer Generation  
{"query": "What is our process?", "top_k_retrieval": 5}

# Indexing
{"space_keys": ["DEV", "TECH"]} 
{"index_all": true}
```

## ⚙️ Advanced Configuration

**Config file alternative** (`~/.confluence_gateway_config.json`):
```json
{
  "confluence": {
    "url": "https://your-instance.atlassian.net",
    "username": "your-email@example.com",
    "api_token": "YOUR_API_TOKEN"
  },
  "vector_db": {
    "type": "qdrant",
    "qdrant_url": ":memory:"
  }
}
```

**Configuration Priority**: `~/.confluence_gateway_config.json` > Environment Variables > Defaults

**Storage options:**
```bash
# Persistent Qdrant (default)
export QDRANT_LOCAL_PATH="~/.confluence_gateway/qdrant_storage"

# ChromaDB alternative  
export VECTOR_DB_TYPE="chroma"
export CHROMA_PERSIST_PATH="~/.confluence_gateway/chroma_storage"

# Memory-only mode (testing)
export QDRANT_URL=":memory:"

# Development mode (faster startup)
export CONFLUENCE_GATEWAY_DEV_MODE="true"
```

## 🛠️ Troubleshooting

**Connection Issues:**
- `401 Unauthorized`: Check API token and permissions in Confluence
- `Connection refused`: Verify Confluence URL includes `https://`
- Test connection: `curl -u "email:token" "https://your-instance.atlassian.net/rest/api/space"`

**Performance:**
- Use development mode: `export CONFLUENCE_GATEWAY_DEV_MODE="true"`
- Memory mode for testing: `export QDRANT_URL=":memory:"`

**Common Errors:**
- `ModuleNotFoundError`: Run `uv sync --dev`
- Missing spaces: Check Confluence permissions for your API token

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

*Transform your Confluence documentation into an intelligent, searchable knowledge base with AI-powered insights.*