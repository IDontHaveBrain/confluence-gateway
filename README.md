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
uv run confluence-gateway index trigger --space-keys TECH --sync

# Search
uv run confluence-gateway search text "deployment guide"
uv run confluence-gateway search semantic "how to deploy"

# Get AI answers
uv run confluence-gateway generate answer "What is our deployment process?"
```

**API Server:**
```bash
# Start server
uv run uvicorn confluence_gateway.api.app:app --reload
# API docs: http://localhost:8000/docs

# Search endpoints
curl "http://localhost:8000/api/search?query=deployment&mode=hybrid"
curl -X POST "http://localhost:8000/api/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "deployment process", "top_k": 5}'

# Generate answers
curl -X POST "http://localhost:8000/api/generate/answer" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our deployment process?"}'
```

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
    "qdrant_url": "http://localhost:6333",
    "chunk_size": 512,
    "chunk_overlap": 50
  },
  "indexing": {
    "include_attachments": true,
    "max_attachment_size_mb": 10,
    "allowed_attachment_extensions": ["pdf", "docx", "txt", "md"],
    "parser_type": "markitdown"
  },
  "generation": {
    "provider": "litellm",
    "model_name": "openrouter/google/gemini-2.5-flash",
    "litellm_api_key": "YOUR_API_KEY",
    "max_context_tokens": 8000,
    "temperature": 0.1
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
uv run confluence-gateway index trigger --space-keys TECH --sync

# 검색
uv run confluence-gateway search text "배포 가이드"
uv run confluence-gateway search semantic "배포하는 방법"

# AI 답변 얻기
uv run confluence-gateway generate answer "우리의 배포 프로세스는 무엇입니까?"
```

**API 서버:**
```bash
# 서버 시작
uv run uvicorn confluence_gateway.api.app:app --reload
# API 문서: http://localhost:8000/docs

# 검색 엔드포인트
curl "http://localhost:8000/api/search?query=배포&mode=hybrid"
curl -X POST "http://localhost:8000/api/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "배포 프로세스", "top_k": 5}'

# 답변 생성
curl -X POST "http://localhost:8000/api/generate/answer" \
  -H "Content-Type: application/json" \
  -d '{"question": "우리의 배포 프로세스는 무엇입니까?"}'
```

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
    "qdrant_url": "http://localhost:6333",
    "chunk_size": 512,
    "chunk_overlap": 50
  },
  "indexing": {
    "include_attachments": true,
    "max_attachment_size_mb": 10,
    "allowed_attachment_extensions": ["pdf", "docx", "txt", "md"],
    "parser_type": "markitdown"
  },
  "generation": {
    "provider": "litellm",
    "model_name": "openrouter/google/gemini-2.5-flash",
    "litellm_api_key": "YOUR_API_KEY",
    "max_context_tokens": 8000,
    "temperature": 0.1
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
