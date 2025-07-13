# Confluence Gateway

**AI-powered search and knowledge retrieval for Atlassian Confluence**

> **Status**: Beta (v0.1.0)

Transform your Confluence into a smart knowledge base with semantic search, hybrid algorithms, and AI-powered question answering.

## ✨ Key Features

- **🔍 Advanced Search Modes**
  - **Text Search**: Traditional keyword search with CQL support  
  - **Semantic Search**: Vector similarity using embeddings
  - **Hybrid Search**: Best of both worlds with Reciprocal Rank Fusion (RRF)

- **🤖 RAG-Powered Q&A**
  - Generate contextual answers from your Confluence content
  - Multiple LLM providers via LiteLLM (OpenAI, Anthropic, etc.)
  - Source attribution with direct links

- **⚡ Flexible Integration**
  - REST API with OpenAPI documentation
  - Full-featured CLI interface
  - Support for multiple vector databases and embedding providers

## 🚀 Quick Start

**Requirements**: Python 3.10+, Confluence API access

### 1. Install

```bash
# Clone and install dependencies
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

## 🧪 Testing

**Philosophy**: E2E testing only - no unit tests, no mocks, test real functionality

```bash
# Test environment setup (required for tests)
echo '{"vector_db": {"qdrant_url": ":memory:", "qdrant_local_path": null}}' > ~/.confluence_gateway_config.json
uv run python tests/setup_test_env.py

# Run all 22 E2E tests (10 CLI + 12 API)
uv run pytest tests/ -v

# Run by category
uv run pytest tests/cli/ -v     # CLI tests
uv run pytest tests/api/ -v     # API tests

# Test discovery
uv run pytest tests/ --collect-only
```

**Requirements for Testing:**
- Real Confluence instance with API access
- Environment variables: `CONFLUENCE_URL`, `CONFLUENCE_USERNAME`, `CONFLUENCE_API_TOKEN`
- Vector database automatically uses `:memory:` mode during tests

## 🔧 Development

**Essential Commands:**
```bash
# Setup development environment
uv sync --dev
uv run pre-commit install

# Development workflow
uv run confluence-gateway --help                                    # Test CLI
uv run uvicorn confluence_gateway.api.app:app --reload             # Test API
uv run ruff check --fix && uv run ruff format && uv run mypy confluence_gateway/  # Quality checks (MANDATORY before commits)
uv run pytest tests/ -v                                            # Run all tests

# Individual quality tools
uv run ruff check confluence_gateway/    # Linting
uv run ruff format confluence_gateway/   # Formatting  
uv run mypy confluence_gateway/          # Type checking
uv run pre-commit run --all-files        # All pre-commit hooks
```

**🚨 CRITICAL: Always run quality checks before commits:**
```bash
uv run ruff check --fix && uv run ruff format && uv run mypy confluence_gateway/
```

## 📚 API Reference

### Core Endpoints

#### Health Check
```http
GET /health
```

#### Spaces
```http
GET /api/spaces                    # List all spaces
GET /api/spaces/{space_key}        # Get space details
```

#### Search
```http
GET /api/search?query=text&limit=20                # Text search
POST /api/search/semantic                          # Semantic search
POST /api/search/advanced                          # Advanced search  
POST /api/search/cql                               # CQL search
```

#### Generation
```http
POST /api/generate/answer                          # RAG answer generation
```

#### Indexing
```http
POST /api/index/trigger                            # Trigger indexing
GET /api/index/status                              # Get indexing status
```

### Request Schemas

**Important**: All POST endpoints require nested request objects:

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

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, FastAPI, Typer
- **Package Manager**: UV (not pip)
- **Vector Databases**: Qdrant, ChromaDB
- **AI/ML**: LiteLLM, SentenceTransformers, LlamaIndex
- **Code Quality**: Ruff, MyPy, pre-commit hooks
- **Testing**: pytest E2E (22 tests)

## 🔒 Security

⚠️ **No built-in authentication** - Use reverse proxy (nginx/Apache) with auth
- Store API tokens in environment variables only
- Configure CORS appropriately for your environment
- Restrict network access to API server in production

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

*Transform your Confluence documentation into an intelligent, searchable knowledge base with AI-powered insights.*