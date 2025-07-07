# Confluence Gateway

**AI-powered search and knowledge retrieval for Atlassian Confluence**

> **Status**: Beta (v0.1.0)

[English](#english) | [한국어](#한국어)

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
- ✅ Multi-provider architecture (Qdrant/ChromaDB, SentenceTransformers/LiteLLM)
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
# Index content
uv run confluence-gateway index trigger --space TECH
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
curl -X POST "http://localhost:8000/api/indexing/trigger" \
  -H "Content-Type: application/json" \
  -d '{"space_keys": ["TECH"]}'

curl "http://localhost:8000/api/indexing/status"
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
POST /api/indexing/trigger
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
GET /api/indexing/status
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

#### Index Commands
```bash
# Trigger indexing
uv run confluence-gateway index trigger [OPTIONS]
  --space, -s TEXT        Space keys to index (repeatable)
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
    "type": "qdrant",
    "collection_name": "confluence_embeddings",
    "embedding_dimension": 384,
    "qdrant_url": "http://localhost:6333",
    "qdrant_grpc_port": 6334,
    "qdrant_prefer_grpc": false,
    "chunk_size": 512,
    "chunk_overlap": 50
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

# Vector database
export VECTOR_DB_TYPE="qdrant"
export VECTOR_DB_QDRANT_URL="http://localhost:6333"
```

### 🧪 Testing

```bash
# All tests
uv run pytest

# Fast unit tests only
uv run pytest -m unit

# Specific test categories
uv run pytest -m integration # Requires external services
uv run pytest -m api         # Requires Confluence API
uv run pytest -m semantic    # Requires vector DB + embeddings

# With coverage
uv run pytest --cov=confluence_gateway
```

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

## Confluence Gateway <a name="한국어"></a>

**Confluence를 위한 AI 기반 검색 및 지식 검색**

> **상태**: Beta (v0.1.0)

시맨틱 검색, 하이브리드 검색 알고리즘, AI 기반 질문 답변으로 Confluence를 스마트 지식 베이스로 변환하세요.

### ✨ 주요 기능

- **🔍 고급 검색 모드**
  - **텍스트 검색**: CQL 지원 전통적인 키워드 검색
  - **시맨틱 검색**: 임베딩을 사용한 벡터 유사성 검색
  - **하이브리드 검색**: Reciprocal Rank Fusion(RRF)으로 두 방식의 장점 결합

- **🤖 RAG 기반 Q&A**
  - Confluence 콘텐츠에서 맥락적 답변 생성
  - LiteLLM을 통한 다중 LLM 제공자 지원 (OpenAI, Anthropic 등)
  - 직접 링크가 포함된 출처 표시

- **⚡ 유연한 통합**
  - OpenAPI 문서가 포함된 REST API
  - 모든 기능을 갖춘 CLI 인터페이스
  - 다중 벡터 데이터베이스 및 임베딩 제공자 지원

### 🎯 구현 현황

**핵심 기능:**
- ✅ 모든 검색 모드: 텍스트, 시맨틱, CQL, 하이브리드
- ✅ 첨부 파일 지원 콘텐츠 인덱싱 (PDF, DOCX, PPTX, TXT, MD)
- ✅ 출처 표시가 포함된 RAG 답변 생성
- ✅ 기능 동등성을 갖춘 REST API + CLI
- ✅ 다중 제공자 아키텍처 (Qdrant/ChromaDB, SentenceTransformers/LiteLLM)
- ✅ 계층적 구성 시스템
- ✅ 포괄적인 오류 처리 및 로깅

**계획된 기능:**
- 🔄 MCP (Model Context Protocol) 서버
- 🔄 내장 인증 레이어
- 🔄 실시간 인덱싱 웹훅
- 🔄 결과 캐싱 레이어

### 🏗️ 아키텍처

의존성 주입이 포함된 깔끔한 서비스 레이어 아키텍처:

```
├── adapters/      # 외부 통합 (Confluence, Vector DB, Embeddings)
├── services/      # 비즈니스 로직 (Search, Indexing, Generation, Ranking)
├── api/          # FastAPI REST API
├── cli/          # Typer CLI 인터페이스
└── core/         # 구성 및 유틸리티
```

**주요 디자인 패턴:** Service Layer, Factory, Singleton, Dependency Injection

### 🚀 빠른 시작

**요구사항:** Python 3.10+, Confluence API 액세스

#### 1. 설치
```bash
git clone https://github.com/yourusername/confluence-gateway.git
cd confluence-gateway
uv sync
```

#### 2. 구성
환경 변수 (권장):
```bash
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_USERNAME="your-email@example.com"
export CONFLUENCE_API_TOKEN="YOUR_API_TOKEN"
```

또는 `~/.confluence_gateway_config.json` 설정 파일:
```json
{
  "confluence": {
    "url": "https://your-instance.atlassian.net",
    "username": "your-email@example.com",
    "api_token": "YOUR_API_TOKEN"
  }
}
```

#### 3. 사용하기

**CLI:**
```bash
# 콘텐츠 인덱싱
uv run confluence-gateway index trigger --space TECH
uv run confluence-gateway index status

# 다양한 검색 모드
uv run confluence-gateway search text "배포 가이드"
uv run confluence-gateway search semantic "배포하는 방법"
uv run confluence-gateway search cql "space = TECH and text ~ deploy"

# 고급 텍스트 검색 옵션
uv run confluence-gateway search text "배포" --type page --limit 10
uv run confluence-gateway search text "가이드" --hybrid --top-n 5
uv run confluence-gateway search text "프로세스" --sort-by updated_at --sort-dir desc

# AI 답변 얻기
uv run confluence-gateway generate answer "우리의 배포 프로세스는 무엇입니까?" --top-k 5
```

**API 서버:**
```bash
# 서버 시작
uv run uvicorn confluence_gateway.api.app:app --reload
# API 문서: http://localhost:8000/docs

# 상태 확인
curl "http://localhost:8000/health"

# 텍스트 검색 (GET)
curl "http://localhost:8000/api/search?query=배포&limit=20"

# 하이브리드 검색 (GET with use_hybrid=true)
curl "http://localhost:8000/api/search?query=배포&use_hybrid=true"

# 시맨틱 검색 (POST)
curl -X POST "http://localhost:8000/api/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "배포 프로세스", "top_k": 5}'

# 고급 검색 (POST)
curl -X POST "http://localhost:8000/api/search/advanced" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "배포", 
    "space_key": "TECH",
    "content_type": "page",
    "sort_by": ["updated_at"],
    "sort_direction": ["desc"],
    "limit": 10
  }'

# CQL 검색 (POST)
curl -X POST "http://localhost:8000/api/search/cql" \
  -H "Content-Type: application/json" \
  -d '{"cql": "space = TECH and text ~ deploy", "limit": 20}'

# 답변 생성
curl -X POST "http://localhost:8000/api/generate/answer" \
  -H "Content-Type: application/json" \
  -d '{"query": "우리의 배포 프로세스는 무엇입니까?", "top_k_retrieval": 5}'

# 인덱싱
curl -X POST "http://localhost:8000/api/indexing/trigger" \
  -H "Content-Type: application/json" \
  -d '{"space_keys": ["TECH"]}'

curl "http://localhost:8000/api/indexing/status"
```

### 📚 API 참조

<details>
<summary><strong>전체 API 엔드포인트 문서</strong></summary>

#### 상태 확인
```http
GET /health
```
서비스 상태 및 Confluence 연결 상태를 반환합니다.

**응답:**
```json
{
  "status": "ok|degraded",
  "version": "0.1.0",
  "timestamp": "2024-01-01T12:00:00Z",
  "confluence_connection": "ok|error|authentication_error|api_error",
  "confluence_error": "오류 메시지 (있는 경우)"
}
```

#### 검색 엔드포인트

##### 텍스트/하이브리드 검색
```http
GET /api/search
```
**쿼리 파라미터:**
- `query` (필수, 최소 2자): 검색 텍스트
- `space_key` (선택): 스페이스로 필터링
- `content_type` (선택): 타입으로 필터링 (page|blogpost|attachment|comment)
- `include_archived` (선택, 기본값: false): 보관된 콘텐츠 포함
- `limit` (선택, 기본값: 20): 최대 결과 수
- `start` (선택, 기본값: 0): 페이지네이션 오프셋
- `expand` (선택, 배열): 확장할 필드
- `use_hybrid` (선택, 기본값: false): 하이브리드 검색 활성화

##### 시맨틱 검색
```http
POST /api/search/semantic
```
**요청 본문:**
```json
{
  "query": "배포 프로세스",
  "top_k": 10,
  "filters": {"space_key": "TECH"}
}
```

##### 고급 검색
```http
POST /api/search/advanced
```
**요청 본문:**
```json
{
  "query": "배포",
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

##### CQL 검색
```http
POST /api/search/cql
```
**요청 본문:**
```json
{
  "cql": "space = TECH AND type = page",
  "limit": 20,
  "start": 0,
  "expand": ["version"]
}
```

#### 인덱싱 엔드포인트

##### 인덱싱 실행
```http
POST /api/indexing/trigger
```
**요청 본문:**
```json
{
  "space_keys": ["TECH", "DEV"]
}
```
**응답:** 202 Accepted

##### 인덱싱 상태 확인
```http
GET /api/indexing/status
```
**응답:**
```json
{
  "status": "idle|running|success|failure",
  "last_run_start_time": "2024-01-01T10:00:00Z",
  "last_run_end_time": "2024-01-01T10:30:00Z",
  "last_error_message": null
}
```

#### 생성 엔드포인트

##### 답변 생성
```http
POST /api/generate/answer
```
**요청 본문:**
```json
{
  "query": "우리의 배포 프로세스는 무엇입니까?",
  "top_k_retrieval": 5,
  "filters": {"space_key": "TECH"}
}
```
**응답:**
```json
{
  "answer": "배포 프로세스는 다음과 같습니다...",
  "sources": [
    {
      "id": "12345_chunk_0",
      "score": 0.85,
      "title": "배포 가이드",
      "url": "https://confluence.example.com/...",
      "space_key": "TECH"
    }
  ]
}
```

#### 응답 형식

모든 엔드포인트는 표준화된 응답을 반환합니다:

**성공 응답:**
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

**오류 응답:**
```json
{
  "status": "error",
  "code": 400,
  "message": "잘못된 검색 파라미터",
  "details": {
    "param": "query",
    "reason": "쿼리는 최소 2자 이상이어야 합니다"
  }
}
```
</details>

### 📚 CLI 참조

<details>
<summary><strong>전체 CLI 명령어 및 옵션</strong></summary>

#### 인덱싱 명령어
```bash
# 인덱싱 실행
uv run confluence-gateway index trigger [OPTIONS]
  --space, -s TEXT        인덱싱할 스페이스 키 (반복 가능)
  --help                  도움말 표시

# 인덱싱 상태 확인
uv run confluence-gateway index status
  --help                  도움말 표시
```

#### 검색 명령어
```bash
# 고급 옵션이 포함된 텍스트 검색
uv run confluence-gateway search text QUERY [OPTIONS]
  --space, -s TEXT        스페이스 키로 필터링 (반복 가능)
  --type, -t TEXT         타입으로 필터링: page, blogpost, attachment, comment
  --limit, -l INTEGER     최대 결과 수 (기본값: 20)
  --archived              보관된 콘텐츠 포함
  --start INTEGER         페이지네이션 시작 위치
  --expand TEXT           확장할 필드 (반복 가능)
  --hybrid                하이브리드 검색 활성화 (키워드 + 시맨틱)
  --sort-by TEXT          정렬 기준: title, created_at, updated_at, score, space_key
  --sort-dir TEXT         정렬 방향: asc, desc
  --min-relevance FLOAT   최소 관련성 점수 (0.0-1.0)
  --top-n INTEGER         페치 후 상위 N개 결과만 반환

# 시맨틱 검색
uv run confluence-gateway search semantic QUERY [OPTIONS]
  --space, -s TEXT        스페이스 키로 필터링 (반복 가능)
  --type, -t TEXT         타입으로 필터링
  --top-k, -k INTEGER     결과 수 (기본값: 10)
  --min-relevance FLOAT   최소 관련성 점수

# CQL 검색
uv run confluence-gateway search cql CQL_QUERY [OPTIONS]
  --limit, -l INTEGER     최대 결과 수 (기본값: 20)
  --start INTEGER         시작 위치
  --expand TEXT           확장할 필드 (반복 가능)
```

#### 생성 명령어
```bash
# AI 답변 생성
uv run confluence-gateway generate answer QUESTION [OPTIONS]
  --space, -s TEXT        스페이스 키로 필터링 (반복 가능)
  --type, -t TEXT         타입으로 필터링
  --top-k, -k INTEGER     컨텍스트 청크 수 (기본값: 5)
  --search-mode TEXT      검색 모드: hybrid, semantic, text (기본값: hybrid)
```
</details>

### ⚙️ 고급 구성

**우선순위:** 환경 변수 > `~/.confluence_gateway_config.json` > 기본값

**기본 생성 설정:**
- 모델: `openrouter/google/gemini-2.5-flash`
- 최대 컨텍스트 토큰: `8000`
- 온도: `0.1`
- 제공자: `litellm`

참고: 기본 모델을 사용하려면 OpenRouter API 키가 필요합니다. https://openrouter.ai/ 에서 받으세요.

<details>
<summary><strong>전체 구성 예제</strong></summary>

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
    "type": "qdrant",
    "collection_name": "confluence_embeddings",
    "embedding_dimension": 384,
    "qdrant_url": "http://localhost:6333",
    "qdrant_grpc_port": 6334,
    "qdrant_prefer_grpc": false,
    "chunk_size": 512,
    "chunk_overlap": 50
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

**주요 환경 변수:**
```bash
# 필수
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_API_TOKEN="YOUR_TOKEN"

# AI 기능용 (기본값: openrouter/google/gemini-2.5-flash)
export GENERATION_MODEL_NAME="openrouter/google/gemini-2.5-flash"
export GENERATION_LITELLM_API_KEY="YOUR_OPENROUTER_API_KEY"

# 벡터 데이터베이스
export VECTOR_DB_TYPE="qdrant"
export VECTOR_DB_QDRANT_URL="http://localhost:6333"
```

### 🧪 테스팅

```bash
# 모든 테스트
uv run pytest

# 빠른 단위 테스트만
uv run pytest -m unit

# 특정 테스트 카테고리
uv run pytest -m integration # 외부 서비스 필요
uv run pytest -m api         # Confluence API 필요
uv run pytest -m semantic    # 벡터 DB + 임베딩 필요

# 커버리지와 함께
uv run pytest --cov=confluence_gateway
```

### 🔧 개발

```bash
# 설정
uv sync --group dev
uv run pre-commit install

# 코드 품질
uv run ruff format confluence_gateway tests  # 포맷
uv run ruff check confluence_gateway tests   # 린트
uv run mypy confluence_gateway               # 타입 체크
uv run pre-commit run --all-files            # 모든 검사
```

### 🔒 프로덕션 보안

⚠️ **내장 인증 없음** - 인증이 포함된 리버스 프록시(nginx/Apache) 사용
- API 토큰을 환경 변수에 저장
- CORS를 적절히 구성
- API 서버에 대한 네트워크 액세스 제한

### 📄 라이선스

MIT 라이선스 - [LICENSE](LICENSE) 파일 참조
