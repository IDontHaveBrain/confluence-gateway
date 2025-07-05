# Confluence Gateway

Open source tool to enhance search and knowledge retrieval from Confluence using RAG and LLMs

> **Project Status**: This project is in the early stages of development (Alpha). Core features are implemented but the API may change.

[English](#english) | [한국어](#한국어)

---

## Confluence Gateway <a name="english"></a>

**Enhance Your Confluence Search with RAG and LLMs**

Confluence Gateway provides enhanced search capabilities for Atlassian Confluence, enabling semantic search, hybrid search, and LLM-powered question answering through REST API and CLI interfaces.

### 🚀 Features

- **Advanced Search**
  - Traditional keyword search using Confluence Query Language (CQL)
  - Semantic search using vector embeddings
  - Hybrid search combining keyword and semantic approaches with RRF
  
- **RAG Integration**
  - Generate contextual answers from your Confluence content
  - Support for multiple LLM providers via LiteLLM
  - Source attribution for generated answers
  
- **Flexible Deployment**
  - REST API for integration
  - CLI for command-line usage
  - Configurable vector databases and embedding providers

### 📋 Implementation Status

Currently implemented:
- ✅ Keyword search with CQL support
- ✅ Semantic search with vector embeddings
- ✅ Hybrid search with Reciprocal Rank Fusion (RRF)
- ✅ Content indexing from Confluence (pages and attachments)
- ✅ RAG-based answer generation via LiteLLM
- ✅ REST API with FastAPI (OpenAPI documentation)
- ✅ Full CLI interface with all features
- ✅ Hierarchical configuration system
- ✅ Vector databases: Qdrant, ChromaDB
- ✅ Embedding providers: Sentence Transformers, LiteLLM
- ✅ HTML and attachment parsing (PDF, DOCX, etc.)

Not yet implemented:
- ❌ MCP (Model Context Protocol) server
- ❌ Built-in authentication (use reverse proxy)
- ❌ Real-time indexing webhooks
- ❌ Caching layer

### 🏗️ Architecture

```
confluence_gateway/
├── adapters/          # External integrations
│   ├── confluence/    # Confluence API client
│   ├── embedding/     # Embedding providers
│   └── vector_db/     # Vector databases
├── services/          # Business logic
│   ├── indexing.py    # Document indexing (singleton)
│   ├── search.py      # Search algorithms
│   ├── generation.py  # LLM generation
│   ├── ranking.py     # Result ranking (RRF)
│   └── parsers/       # Content parsing
├── api/              # REST API
│   └── routes/       # API endpoints
├── cli/              # CLI interface
└── core/             # Core utilities
    └── config.py     # Configuration management
```

### 🚀 Quick Start

#### Prerequisites

- Python 3.10+
- Confluence instance with API access
- (Optional) Vector database: Qdrant or ChromaDB
- (Optional) LLM API key for RAG features

#### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/confluence-gateway.git
cd confluence-gateway

# Install dependencies
uv pip install -e .
uv pip install -e ".[dev]"  # For development
```

#### Configuration

Create `~/.confluence_gateway_config.json`:

```json
{
  "confluence": {
    "url": "https://your-instance.atlassian.net",
    "username": "your-email@example.com",
    "api_token": "YOUR_API_TOKEN"
  }
}
```

Or use environment variables:
```bash
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_USERNAME="your-email@example.com"
export CONFLUENCE_API_TOKEN="YOUR_API_TOKEN"
```

#### Basic Usage

**CLI:**
```bash
# Index Confluence content
confluence-gateway index trigger --space-keys TECH --sync

# Check indexing status
confluence-gateway index status

# Search
confluence-gateway search text "deployment guide"
confluence-gateway search semantic "how to deploy"
confluence-gateway search cql "space = TECH and text ~ deploy"

# Generate answers
confluence-gateway generate answer "What is our deployment process?"
```

**API:**
```bash
# Start server
uvicorn confluence_gateway.api.app:app --reload

# API documentation available at http://localhost:8000/docs

# Search
curl "http://localhost:8000/search?query=deployment&mode=hybrid"

# Semantic search
curl -X POST "http://localhost:8000/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "deployment process", "top_k": 5}'

# Generate answer
curl -X POST "http://localhost:8000/generate/answer" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our deployment process?"}'
```

### ⚙️ Configuration

Configuration follows this priority: Environment Variables > `~/.confluence_gateway_config.json` > Defaults

#### Full Configuration Example

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
    "hybrid_keyword_fetch_limit": 50,
    "hybrid_semantic_fetch_limit": 50,
    "hybrid_rrf_k": 60
  },
  "embedding": {
    "provider": "sentence-transformers",
    "model_name": "all-MiniLM-L6-v2",
    "dimension": 384,
    "device": "cpu",
    "batch_size": 32
  },
  "vector_db": {
    "type": "qdrant",
    "collection_name": "confluence_embeddings",
    "embedding_dimension": 384,
    "qdrant_url": "http://localhost:6333",
    "chunk_size": 512,
    "chunk_overlap": 50
  },
  "indexing": {
    "batch_size": 10,
    "include_archived": false,
    "include_attachments": true,
    "max_attachment_size_mb": 10,
    "allowed_attachment_extensions": ["pdf", "docx", "txt", "md"],
    "parser_type": "markitdown"
  },
  "generation": {
    "enable": true,
    "provider": "litellm",
    "model_name": "openai/gpt-4",
    "litellm_api_key": "YOUR_API_KEY",
    "max_context_tokens": 3000,
    "temperature": 0.1,
    "top_k_results": 5
  }
}
```

#### Environment Variables

All configuration can be set via environment variables with the format `{SECTION}_{KEY}`:

```bash
# Confluence settings
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_API_TOKEN="YOUR_TOKEN"

# Embedding settings
export EMBEDDING_PROVIDER="sentence-transformers"
export EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"

# Vector DB settings
export VECTOR_DB_TYPE="qdrant"
export VECTOR_DB_QDRANT_URL="http://localhost:6333"

# Generation settings
export GENERATION_MODEL_NAME="openai/gpt-4"
export GENERATION_LITELLM_API_KEY="YOUR_API_KEY"
```

### 🧪 Testing

```bash
# Run all tests
pytest

# Unit tests only
pytest -m "not integration"

# Integration tests (requires Confluence)
export CONFLUENCE_URL="https://test.atlassian.net"
export CONFLUENCE_API_TOKEN="test-token"
pytest -m integration

# With coverage
pytest --cov=confluence_gateway
```

### 🔧 Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"
pre-commit install

# Code formatting
ruff format confluence_gateway tests

# Linting
ruff check confluence_gateway tests

# Type checking
mypy confluence_gateway

# Run all checks
pre-commit run --all-files
```

### 🔒 Security Considerations

This tool does not include built-in authentication. For production use:

1. **Use a reverse proxy** (nginx, Apache) with authentication
2. **Configure CORS** appropriately in the API settings
3. **Use environment variables** for sensitive data (API tokens)
4. **Restrict network access** to the API server

Example nginx configuration:
```nginx
location /api/ {
    auth_basic "Restricted";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:8000/;
}
```

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Confluence Gateway <a name="한국어"></a>

**RAG와 LLM으로 Confluence 검색 향상하기**

Confluence Gateway는 Atlassian Confluence에 향상된 검색 기능을 제공하여 시맨틱 검색, 하이브리드 검색, LLM 기반 질문 답변을 REST API와 CLI 인터페이스를 통해 지원합니다.

### 🚀 주요 기능

- **고급 검색**
  - Confluence Query Language (CQL)을 사용한 전통적인 키워드 검색
  - 벡터 임베딩을 사용한 시맨틱 검색
  - RRF로 키워드와 시맨틱 접근법을 결합한 하이브리드 검색
  
- **RAG 통합**
  - Confluence 콘텐츠에서 맥락적 답변 생성
  - LiteLLM을 통한 다중 LLM 제공자 지원
  - 생성된 답변에 대한 출처 표시
  
- **유연한 배포**
  - 통합을 위한 REST API
  - 명령줄 사용을 위한 CLI
  - 구성 가능한 벡터 데이터베이스 및 임베딩 제공자

### 📋 구현 상태

구현 완료:
- ✅ CQL 지원 키워드 검색
- ✅ 벡터 임베딩을 사용한 시맨틱 검색
- ✅ Reciprocal Rank Fusion (RRF)을 사용한 하이브리드 검색
- ✅ Confluence 콘텐츠 인덱싱 (페이지 및 첨부 파일)
- ✅ LiteLLM을 통한 RAG 기반 답변 생성
- ✅ FastAPI를 사용한 REST API (OpenAPI 문서화)
- ✅ 모든 기능을 갖춘 CLI 인터페이스
- ✅ 계층적 구성 시스템
- ✅ 벡터 데이터베이스: Qdrant, ChromaDB
- ✅ 임베딩 제공자: Sentence Transformers, LiteLLM
- ✅ HTML 및 첨부 파일 파싱 (PDF, DOCX 등)

아직 구현되지 않음:
- ❌ MCP (Model Context Protocol) 서버
- ❌ 내장 인증 (리버스 프록시 사용)
- ❌ 실시간 인덱싱 웹훅
- ❌ 캐싱 레이어

### 🏗️ 아키텍처

```
confluence_gateway/
├── adapters/          # 외부 통합
│   ├── confluence/    # Confluence API 클라이언트
│   ├── embedding/     # 임베딩 제공자
│   └── vector_db/     # 벡터 데이터베이스
├── services/          # 비즈니스 로직
│   ├── indexing.py    # 문서 인덱싱 (싱글톤)
│   ├── search.py      # 검색 알고리즘
│   ├── generation.py  # LLM 생성
│   ├── ranking.py     # 결과 순위 지정 (RRF)
│   └── parsers/       # 콘텐츠 파싱
├── api/              # REST API
│   └── routes/       # API 엔드포인트
├── cli/              # CLI 인터페이스
└── core/             # 핵심 유틸리티
    └── config.py     # 구성 관리
```

### 🚀 빠른 시작

#### 사전 요구 사항

- Python 3.10+
- API 액세스가 있는 Confluence 인스턴스
- (선택 사항) 벡터 데이터베이스: Qdrant 또는 ChromaDB
- (선택 사항) RAG 기능을 위한 LLM API 키

#### 설치

```bash
# 저장소 복제
git clone https://github.com/yourusername/confluence-gateway.git
cd confluence-gateway

# 의존성 설치
uv pip install -e .
uv pip install -e ".[dev]"  # 개발용
```

#### 구성

`~/.confluence_gateway_config.json` 생성:

```json
{
  "confluence": {
    "url": "https://your-instance.atlassian.net",
    "username": "your-email@example.com",
    "api_token": "YOUR_API_TOKEN"
  }
}
```

또는 환경 변수 사용:
```bash
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_USERNAME="your-email@example.com"
export CONFLUENCE_API_TOKEN="YOUR_API_TOKEN"
```

#### 기본 사용법

**CLI:**
```bash
# Confluence 콘텐츠 인덱싱
confluence-gateway index trigger --space-keys TECH --sync

# 인덱싱 상태 확인
confluence-gateway index status

# 검색
confluence-gateway search text "배포 가이드"
confluence-gateway search semantic "배포하는 방법"
confluence-gateway search cql "space = TECH and text ~ 배포"

# 답변 생성
confluence-gateway generate answer "우리의 배포 프로세스는 무엇입니까?"
```

**API:**
```bash
# 서버 시작
uvicorn confluence_gateway.api.app:app --reload

# API 문서는 http://localhost:8000/docs에서 확인 가능

# 검색
curl "http://localhost:8000/search?query=배포&mode=hybrid"

# 시맨틱 검색
curl -X POST "http://localhost:8000/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "배포 프로세스", "top_k": 5}'

# 답변 생성
curl -X POST "http://localhost:8000/generate/answer" \
  -H "Content-Type: application/json" \
  -d '{"question": "우리의 배포 프로세스는 무엇입니까?"}'
```

### ⚙️ 구성

구성 우선순위: 환경 변수 > `~/.confluence_gateway_config.json` > 기본값

#### 전체 구성 예제

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
    "hybrid_keyword_fetch_limit": 50,
    "hybrid_semantic_fetch_limit": 50,
    "hybrid_rrf_k": 60
  },
  "embedding": {
    "provider": "sentence-transformers",
    "model_name": "all-MiniLM-L6-v2",
    "dimension": 384,
    "device": "cpu",
    "batch_size": 32
  },
  "vector_db": {
    "type": "qdrant",
    "collection_name": "confluence_embeddings",
    "embedding_dimension": 384,
    "qdrant_url": "http://localhost:6333",
    "chunk_size": 512,
    "chunk_overlap": 50
  },
  "indexing": {
    "batch_size": 10,
    "include_archived": false,
    "include_attachments": true,
    "max_attachment_size_mb": 10,
    "allowed_attachment_extensions": ["pdf", "docx", "txt", "md"],
    "parser_type": "markitdown"
  },
  "generation": {
    "enable": true,
    "provider": "litellm",
    "model_name": "openai/gpt-4",
    "litellm_api_key": "YOUR_API_KEY",
    "max_context_tokens": 3000,
    "temperature": 0.1,
    "top_k_results": 5
  }
}
```

#### 환경 변수

모든 구성은 `{SECTION}_{KEY}` 형식의 환경 변수로 설정할 수 있습니다:

```bash
# Confluence 설정
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_API_TOKEN="YOUR_TOKEN"

# 임베딩 설정
export EMBEDDING_PROVIDER="sentence-transformers"
export EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"

# 벡터 DB 설정
export VECTOR_DB_TYPE="qdrant"
export VECTOR_DB_QDRANT_URL="http://localhost:6333"

# 생성 설정
export GENERATION_MODEL_NAME="openai/gpt-4"
export GENERATION_LITELLM_API_KEY="YOUR_API_KEY"
```

### 🧪 테스팅

```bash
# 모든 테스트 실행
pytest

# 단위 테스트만
pytest -m "not integration"

# 통합 테스트 (Confluence 필요)
export CONFLUENCE_URL="https://test.atlassian.net"
export CONFLUENCE_API_TOKEN="test-token"
pytest -m integration

# 커버리지와 함께
pytest --cov=confluence_gateway
```

### 🔧 개발

```bash
# 개발 의존성 설치
uv pip install -e ".[dev]"
pre-commit install

# 코드 포맷팅
ruff format confluence_gateway tests

# 린팅
ruff check confluence_gateway tests

# 타입 체킹
mypy confluence_gateway

# 모든 검사 실행
pre-commit run --all-files
```

### 🔒 보안 고려사항

이 도구는 내장 인증을 포함하지 않습니다. 프로덕션 사용을 위해:

1. **리버스 프록시 사용** (nginx, Apache) 인증과 함께
2. **CORS를 적절히 구성** API 설정에서
3. **민감한 데이터에 환경 변수 사용** (API 토큰)
4. **API 서버에 대한 네트워크 액세스 제한**

nginx 구성 예제:
```nginx
location /api/ {
    auth_basic "Restricted";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:8000/;
}
```

### 📄 라이선스

이 프로젝트는 MIT 라이선스에 따라 라이선스가 부여됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.