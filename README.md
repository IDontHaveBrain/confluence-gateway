# Confluence Gateway

**AI-powered search and knowledge retrieval for Atlassian Confluence**

> **Status**: Beta (v0.1.0)

[English](README.md) | [한국어](README_ko.md)

---

## Confluence Gateway <a name="english"></a>

Transform your Confluence into a smart knowledge base with semantic search, hybrid search algorithms, and AI-powered question answering.

### ✨ Key Features

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

### 🎯 What's Implemented

**Core Features:**
- ✅ All search modes: Text, Semantic, CQL, Hybrid
- ✅ Content indexing with attachment support (PDF, DOCX, PPTX, TXT, MD)
- ✅ RAG answer generation with source attribution
- ✅ REST API + CLI with feature parity
- ✅ Multi-provider architecture (Qdrant/ChromaDB with local storage support, SentenceTransformers/LiteLLM)
- ✅ Hierarchical configuration system
- ✅ Comprehensive error handling and logging

**Planned Features:**
- 🔄 MCP (Model Context Protocol) server
- 🔄 Built-in authentication layer
- 🔄 Real-time indexing webhooks
- 🔄 Result caching layer

### 🏗️ Architecture

Clean service layer architecture with dependency injection:

```
├── adapters/      # External integrations (Confluence, Vector DBs, Embeddings)
├── services/      # Business logic (Search, Indexing, Generation, Ranking)
├── api/          # REST API with FastAPI
├── cli/          # CLI interface with Typer
└── core/         # Configuration & utilities
```

**Key Design Patterns:** Service Layer, Factory, Singleton, Dependency Injection

### 🚀 Quick Start

**Requirements:** Python 3.10+, Confluence API access

#### 1. Install
```bash
git clone https://github.com/yourusername/confluence-gateway.git
cd confluence-gateway
uv sync
```

#### 2. Configure
Environment variables (recommended):
```bash
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_USERNAME="your-email@example.com"
export CONFLUENCE_API_TOKEN="YOUR_API_TOKEN"
```

Or config file at `~/.confluence_gateway_config.json`:
```json
{
  "confluence": {
    "url": "https://your-instance.atlassian.net",
    "username": "your-email@example.com",
    "api_token": "YOUR_API_TOKEN"
  }
}
```

#### 3. Use It

**CLI:**
```bash
# List and manage spaces
uv run confluence-gateway spaces list
uv run confluence-gateway spaces list --all
uv run confluence-gateway spaces list --search "dev"
uv run confluence-gateway spaces list --key-prefix TEAM

# Index content
uv run confluence-gateway index trigger --space TECH
uv run confluence-gateway index trigger --all  # Index all accessible spaces
uv run confluence-gateway index status

# Search with various modes
uv run confluence-gateway search text "deployment guide"
uv run confluence-gateway search semantic "how to deploy"
uv run confluence-gateway search cql "space = TECH and text ~ deploy"

# Advanced text search options
uv run confluence-gateway search text "deployment" --type page --limit 10
uv run confluence-gateway search text "guide" --hybrid --top-n 5
uv run confluence-gateway search text "process" --sort-by updated_at --sort-dir desc

# Get AI answers
uv run confluence-gateway generate answer "What is our deployment process?" --top-k 5
```

**API Server:**
```bash
# Start server
uv run uvicorn confluence_gateway.api.app:app --reload
# API docs: http://localhost:8000/docs

# Health check
curl "http://localhost:8000/health"

# List spaces
curl "http://localhost:8000/api/spaces"

# Get space info
curl "http://localhost:8000/api/spaces/TECH"

# Text search (GET)
curl "http://localhost:8000/api/search?query=deployment&limit=20"

# Hybrid search (GET with use_hybrid=true)
curl "http://localhost:8000/api/search?query=deployment&use_hybrid=true"

# Semantic search (POST)
curl -X POST "http://localhost:8000/api/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "deployment process", "top_k": 5}'

# Advanced search (POST)
curl -X POST "http://localhost:8000/api/search/advanced" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deployment", 
    "space_key": "TECH",
    "content_type": "page",
    "sort_by": ["updated_at"],
    "sort_direction": ["desc"],
    "limit": 10
  }'

# CQL search (POST)
curl -X POST "http://localhost:8000/api/search/cql" \
  -H "Content-Type: application/json" \
  -d '{"cql": "space = TECH and text ~ deploy", "limit": 20}'

# Generate answers
curl -X POST "http://localhost:8000/api/generate/answer" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our deployment process?", "top_k_retrieval": 5}'

# Indexing
curl -X POST "http://localhost:8000/api/index/trigger" \
  -H "Content-Type: application/json" \
  -d '{"space_keys": ["TECH"]}'

# Index all accessible spaces
curl -X POST "http://localhost:8000/api/index/trigger" \
  -H "Content-Type: application/json" \
  -d '{"index_all": true}'

curl "http://localhost:8000/api/index/status"
```

### 📚 API Reference

<details>
<summary><strong>Complete API Endpoints Documentation</strong></summary>

#### Health Check
```http
GET /health
```
Returns service health status and Confluence connectivity.

**Response:**
```json
{
  "status": "ok|degraded",
  "version": "0.1.0",
  "timestamp": "2024-01-01T12:00:00Z",
  "confluence_connection": "ok|error|authentication_error|api_error",
  "confluence_error": "error message if any"
}
```

#### Spaces Endpoints

##### List All Spaces
```http
GET /api/spaces
```
Returns a list of all accessible Confluence spaces.

**Response:**
```json
[
  {
    "id": "12345",
    "key": "DEV",
    "name": "Development",
    "type": "global",
    "description": "Space for development documentation",
    "created_at": "2023-01-15T10:00:00Z",
    "updated_at": "2023-12-01T14:30:00Z"
  }
]
```

##### Get Space Details
```http
GET /api/spaces/{space_key}
```
Returns detailed information about a specific space.

**Parameters:**
- `space_key` (path, required): The unique key of the space

**Response:**
```json
{
  "id": "12345",
  "key": "DEV",
  "name": "Development",
  "type": "global",
  "description": "Space for development documentation",
  "created_at": "2023-01-15T10:00:00Z",
  "updated_at": "2023-12-01T14:30:00Z"
}
```

#### Search Endpoints

##### Text/Hybrid Search
```http
GET /api/search
```
**Query Parameters:**
- `query` (required, min 2 chars): Search text
- `space_key` (optional): Filter by space
- `content_type` (optional): Filter by type (page|blogpost|attachment|comment)
- `include_archived` (optional, default: false): Include archived content
- `limit` (optional, default: 20): Max results
- `start` (optional, default: 0): Pagination offset
- `expand` (optional, array): Fields to expand
- `use_hybrid` (optional, default: false): Enable hybrid search

##### Semantic Search
```http
POST /api/search/semantic
```
**Request Body:**
```json
{
  "query": "deployment process",
  "top_k": 10,
  "filters": {"space_key": "TECH"}
}
```

##### Advanced Search
```http
POST /api/search/advanced
```
**Request Body:**
```json
{
  "query": "deployment",
  "space_key": "TECH",
  "content_type": "page",
  "include_archived": false,
  "limit": 20,
  "start": 0,
  "expand": ["version", "history"],
  "get_all_results": false,
  "max_results": 100,
  "min_relevance": 0.5,
  "top_n": 10,
  "sort_by": ["updated_at"],
  "sort_direction": ["desc"],
  "use_hybrid": false
}
```

##### CQL Search
```http
POST /api/search/cql
```
**Request Body:**
```json
{
  "cql": "space = TECH AND type = page",
  "limit": 20,
  "start": 0,
  "expand": ["version"]
}
```

#### Indexing Endpoints

##### Trigger Indexing
```http
POST /api/index/trigger
```
**Request Body:**
```json
{
  "space_keys": ["TECH", "DEV"]
}
```
**Response:** 202 Accepted

##### Get Indexing Status
```http
GET /api/index/status
```
**Response:**
```json
{
  "status": "idle|running|success|failure",
  "last_run_start_time": "2024-01-01T10:00:00Z",
  "last_run_end_time": "2024-01-01T10:30:00Z",
  "last_error_message": null
}
```

#### Generation Endpoints

##### Generate Answer
```http
POST /api/generate/answer
```
**Request Body:**
```json
{
  "query": "What is our deployment process?",
  "top_k_retrieval": 5,
  "filters": {"space_key": "TECH"}
}
```
**Response:**
```json
{
  "answer": "Your deployment process involves...",
  "sources": [
    {
      "id": "12345_chunk_0",
      "score": 0.85,
      "title": "Deployment Guide",
      "url": "https://confluence.example.com/...",
      "space_key": "TECH"
    }
  ]
}
```

#### Response Formats

All endpoints return standardized responses:

**Success Response:**
```json
{
  "results": [...],
  "total": 42,
  "start": 0,
  "limit": 20,
  "took_ms": 123.45,
  "page_count": 3,
  "current_page": 1,
  "has_more": true,
  "links": {
    "next": "/api/search?query=api&start=20",
    "previous": null
  }
}
```

**Error Response:**
```json
{
  "status": "error",
  "code": 400,
  "message": "Invalid search parameters",
  "details": {
    "param": "query",
    "reason": "Query must be at least 2 characters long"
  }
}
```
</details>

### 📚 CLI Reference

<details>
<summary><strong>Complete CLI Commands and Options</strong></summary>

#### Spaces Commands
```bash
# List all Confluence spaces
uv run confluence-gateway spaces list [OPTIONS]
  --page, -p INTEGER      Page number (starts from 1, default: 1)
  --page-size, -s INTEGER Number of spaces per page (default: 25, max: 100)
  --all, -a              Fetch all spaces (ignore pagination)
  --type, -t TEXT        Filter by space type: personal, global, or all
  --search TEXT          Search spaces by name or key (case-insensitive)
  --key-prefix TEXT      Filter spaces by key prefix (case-insensitive)
  --sort TEXT            Sort spaces by: name, key, type, or id
  --reverse, -r          Reverse sort order
  --verbose, -v          Show detailed error messages and retry information
  --help                 Show help message

# Get detailed information about a specific space
uv run confluence-gateway spaces info SPACE_KEY [OPTIONS]
  --verbose, -v          Show detailed error messages and retry information
  --help                 Show help message
```

#### Index Commands
```bash
# Trigger indexing
uv run confluence-gateway index trigger [OPTIONS]
  --space, -s TEXT        Space keys to index (repeatable)
  --all, -a              Index all accessible spaces (ignores configuration filters)
  --help                  Show help message

# Check indexing status
uv run confluence-gateway index status
  --help                  Show help message
```

#### Search Commands
```bash
# Text search with advanced options
uv run confluence-gateway search text QUERY [OPTIONS]
  --space, -s TEXT        Filter by space key (repeatable)
  --type, -t TEXT         Filter by type: page, blogpost, attachment, comment
  --limit, -l INTEGER     Max results to return (default: 20)
  --archived              Include archived content
  --start INTEGER         Starting position for pagination
  --expand TEXT           Fields to expand (repeatable)
  --hybrid                Enable hybrid search (keyword + semantic)
  --sort-by TEXT          Sort by: title, created_at, updated_at, score, space_key
  --sort-dir TEXT         Sort direction: asc, desc
  --min-relevance FLOAT   Minimum relevance score (0.0-1.0)
  --top-n INTEGER         Return only top N results after fetching

# Semantic search
uv run confluence-gateway search semantic QUERY [OPTIONS]
  --space, -s TEXT        Filter by space key (repeatable)
  --type, -t TEXT         Filter by type
  --top-k, -k INTEGER     Number of results (default: 10)
  --min-relevance FLOAT   Minimum relevance score

# CQL search
uv run confluence-gateway search cql CQL_QUERY [OPTIONS]
  --limit, -l INTEGER     Max results (default: 20)
  --start INTEGER         Starting position
  --expand TEXT           Fields to expand (repeatable)
```

#### Generate Commands
```bash
# Generate AI answers
uv run confluence-gateway generate answer QUESTION [OPTIONS]
  --space, -s TEXT        Filter by space key (repeatable)
  --type, -t TEXT         Filter by type
  --top-k, -k INTEGER     Number of context chunks (default: 5)
  --search-mode TEXT      Search mode: hybrid, semantic, text (default: hybrid)
```
</details>

### ⚙️ Advanced Configuration

**Priority:** Environment Variables > `~/.confluence_gateway_config.json` > Defaults

**Default Generation Settings:**
- Model: `openrouter/google/gemini-2.5-flash`
- Max Context Tokens: `8000`
- Temperature: `0.1`
- Provider: `litellm`

Note: You'll need an OpenRouter API key to use the default model. Get one at https://openrouter.ai/

**Vector Database Options:**
- **Qdrant**: 
  - Local storage mode: Set `qdrant_local_path` (e.g., `~/.confluence_gateway/qdrant_storage`) and `qdrant_url` to `null`
  - Server mode: Set `qdrant_url` (e.g., `http://localhost:6333`) and `qdrant_local_path` to `null`
  - Default: Local storage with persistence enabled
- **ChromaDB**: 
  - Local storage: Set `chroma_persist_path` (e.g., `~/.confluence_gateway/chroma_storage`)
  - Remote server: Set `chroma_host` and `chroma_port`
- Both databases automatically create directories for local storage when configured

<details>
<summary><strong>Complete Configuration Example</strong></summary>

```json
{
  "confluence": {
    "url": "https://your-instance.atlassian.net",
    "username": "your-email@example.com",
    "api_token": "YOUR_API_TOKEN",
    "timeout": 15
  },
  "search": {
    "default_limit": 20,
    "max_limit": 100,
    "hybrid_search_enabled": true,
    "hybrid_rrf_k": 60
  },
  "embedding": {
    "provider": "sentence-transformers",
    "model_name": "all-MiniLM-L6-v2",
    "dimension": 384,
    "device": "cpu"
  },
  "vector_db": {
    "type": "qdrant",  // or "chroma"
    "collection_name": "confluence_embeddings",
    "embedding_dimension": 384,
    "chunk_size": 512,
    "chunk_overlap": 50,
    // Qdrant configuration (choose one mode)
    // For local storage (default):
    "qdrant_url": null,
    "qdrant_local_path": "~/.confluence_gateway/qdrant_storage",
    // For server mode:
    // "qdrant_url": "http://localhost:6333",
    // "qdrant_local_path": null,
    "qdrant_grpc_port": 6334,
    "qdrant_prefer_grpc": false,
    // ChromaDB configuration
    "chroma_persist_path": "~/.confluence_gateway/chroma_storage",
    "chroma_host": "localhost",
    "chroma_port": 8000
  },
  "indexing": {
    "include_spaces": null,
    "exclude_spaces": null,
    "include_attachments": true,
    "max_attachment_size_mb": 10,
    "allowed_attachment_extensions": ["pdf", "docx", "txt", "md"],
    "html_parser": "markitdown",
    "attachment_parser": "markitdown"
  },
  "generation": {
    "enable": true,
    "provider": "litellm",
    "model_name": "openrouter/google/gemini-2.5-flash",
    "litellm_api_key": "YOUR_API_KEY",
    "prompt_template": "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
    "max_context_tokens": 8000,
    "max_output_tokens": 500,
    "temperature": 0.1,
    "generation_timeout": 60
  }
}
```
</details>

**Common Environment Variables:**
```bash
# Required
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_API_TOKEN="YOUR_TOKEN"

# For AI features (defaults to openrouter/google/gemini-2.5-flash)
export GENERATION_MODEL_NAME="openrouter/google/gemini-2.5-flash"
export GENERATION_LITELLM_API_KEY="YOUR_OPENROUTER_API_KEY"

# Vector database (choose one)
# For Qdrant local storage (default, persistent):
export VECTOR_DB_TYPE="qdrant"
export VECTOR_DB_QDRANT_LOCAL_PATH="~/.confluence_gateway/qdrant_storage"
# For Qdrant server mode (unset QDRANT_LOCAL_PATH first):
# export VECTOR_DB_QDRANT_URL="http://localhost:6333"

# For ChromaDB:
export VECTOR_DB_TYPE="chroma"
export CHROMA_PERSIST_PATH="~/.confluence_gateway/chroma_storage"
# Or for remote ChromaDB:
export CHROMA_HOST="localhost"
export CHROMA_PORT="8000"
```

### 🧪 Testing

**Philosophy**: Simple E2E testing only. No unit tests, no mocks, no complex test scenarios.

```bash
# Setup testing environment
uv sync --dev

# Run all E2E tests
uv run pytest tests/ -v

# Run specific test categories
uv run pytest tests/cli/ -v     # CLI E2E tests
uv run pytest tests/api/ -v     # API E2E tests
```

**Requirements for Testing:**
- Real Confluence instance with API access
- Vector database automatically set to memory mode during tests
- Valid `CONFLUENCE_URL`, `CONFLUENCE_USERNAME`, `CONFLUENCE_API_TOKEN` environment variables

### 🔧 Development

```bash
# Setup
uv sync --group dev
uv run pre-commit install

# Code quality
uv run ruff format confluence_gateway tests # Format
uv run ruff check confluence_gateway tests  # Lint
uv run mypy confluence_gateway tests        # Type check
uv run pre-commit run --all-files           # All checks
```

### 🔒 Production Security

⚠️ **No built-in authentication** - Use reverse proxy (nginx/Apache) with auth
- Store API tokens in environment variables
- Configure CORS appropriately
- Restrict network access to API server

### 📄 License

MIT License - see [LICENSE](LICENSE) file

---




