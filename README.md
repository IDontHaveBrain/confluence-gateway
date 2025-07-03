# Confluence Gateway

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

[English](#english) | [한국어](#한국어)

---

## Confluence Gateway <a name="english"></a>

**AI-Powered Confluence Search with RAG Integration**

Confluence Gateway enhances your Atlassian Confluence knowledge base with modern AI capabilities, providing semantic search, hybrid search algorithms, and LLM-powered question answering through both REST API and CLI interfaces.

### 🚀 Key Features

- **🔍 Advanced Search Capabilities**
  - **Keyword Search**: Traditional text-based search with CQL support
  - **Semantic Search**: Vector-based similarity search using embeddings
  - **Hybrid Search**: Combines keyword and semantic search with Reciprocal Rank Fusion (RRF)
  
- **🤖 RAG Integration**
  - Generate contextual answers from your Confluence content
  - Support for multiple LLM providers via LiteLLM
  - Configurable context windows and token limits

- **💾 Flexible Storage Options**
  - **Vector Databases**: Qdrant, ChromaDB, or in-memory storage
  - **Embedding Providers**: Sentence Transformers (local) or LiteLLM (OpenAI, Ollama, etc.)
  - **Content Parsers**: HTML and attachment parsing with Markitdown or Unstructured

- **🛠️ Developer-Friendly**
  - **REST API**: FastAPI-based with automatic OpenAPI documentation
  - **CLI Tool**: Full-featured command-line interface
  - **Configurable**: Hierarchical configuration system (env vars > JSON > defaults)

### 📋 Project Status

The project is **production-ready** from a code quality perspective with all core features implemented. However, before deployment, consider implementing:
- API authentication/authorization
- Rate limiting for production use
- Connection pooling for high-load scenarios

### 🏗️ Architecture

```
confluence_gateway/
├── adapters/          # External system integrations
│   ├── confluence/    # Atlassian API client
│   ├── embedding/     # Embedding providers (sentence-transformers, litellm)
│   └── vector_db/     # Vector databases (qdrant, chroma)
├── services/          # Business logic layer
│   ├── embedding.py   # Embedding orchestration
│   ├── indexing.py    # Document indexing with chunking
│   ├── search.py      # Search algorithms (keyword, semantic, hybrid)
│   ├── generation.py  # LLM-based answer generation
│   ├── ranking.py     # Result ranking (RRF for hybrid search)
│   └── parsers/       # Content parsing (HTML, PDF, DOCX, etc.)
├── api/              # FastAPI REST interface
│   ├── routes/       # API endpoints
│   └── schemas/      # Request/response models
├── cli/              # Typer CLI interface
└── core/             # Core utilities
    ├── config.py     # Configuration management
    └── exceptions.py # Custom exceptions
```

### 🚀 Quick Start

#### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Confluence instance with API access
- (Optional) Vector database (Qdrant or ChromaDB)
- (Optional) LLM API key for RAG features

#### Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/confluence-gateway.git
cd confluence-gateway

# Install the package and dependencies
uv pip install -e .
uv pip install -e ".[dev]"  # For development

# Install pre-commit hooks (for development)
pre-commit install
```

#### Configuration

Create a configuration file at `~/.confluence_gateway_config.json`:

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

See [Configuration](#configuration) section for full options.

#### Basic Usage

**CLI Examples:**
```bash
# Search for content
confluence-gateway search text "deployment process"

# Index Confluence pages for semantic search
confluence-gateway index pages --space-keys TECH DOC

# Perform semantic search
confluence-gateway search semantic "how to deploy to production"

# Generate AI answer (requires LLM configuration)
confluence-gateway generate answer "What is our deployment process?"
```

**API Examples:**
```bash
# Start the API server
uvicorn confluence_gateway.api.app:app --reload

# Search via API
curl "http://localhost:8000/api/search?query=deployment&limit=10"

# Semantic search
curl -X POST "http://localhost:8000/api/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "deployment process", "top_k": 5}'
```

### 📚 API Documentation

When the API server is running, visit:
- Interactive API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### ⚙️ Configuration <a name="configuration"></a>

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
    "device": "cpu"
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
    "include_spaces": ["TECH", "DOC"],
    "include_attachments": true,
    "max_attachment_size_mb": 10,
    "allowed_attachment_extensions": ["pdf", "docx", "txt", "md"]
  },
  "generation": {
    "enable": true,
    "provider": "litellm",
    "model_name": "openai/gpt-4",
    "litellm_api_key": "YOUR_OPENAI_API_KEY",
    "max_context_tokens": 3000,
    "temperature": 0.1
  }
}
```

### 🧪 Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest -m "not integration"

# Run integration tests (requires Confluence access)
pytest -m integration

# Run with coverage
pytest --cov=confluence_gateway --cov-report=html
```

### 🔧 Development

```bash
# Run linting and formatting
ruff check confluence_gateway tests
ruff format confluence_gateway tests

# Type checking
mypy confluence_gateway

# Run all pre-commit checks
pre-commit run --all-files
```

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🔮 Roadmap

- [ ] API Authentication & Authorization
- [ ] Rate Limiting Middleware
- [ ] Connection Pooling for Confluence Client
- [ ] MCP (Model Context Protocol) Server Implementation
- [ ] OpenAPI/Swagger UI Enhancements
- [ ] Webhook Support for Real-time Indexing
- [ ] Multi-language Support
- [ ] Advanced Analytics Dashboard

---

## Confluence Gateway <a name="한국어"></a>

**AI 기반 Confluence 검색 및 RAG 통합**

Confluence Gateway는 Atlassian Confluence 지식 베이스에 최신 AI 기능을 추가하여 시맨틱 검색, 하이브리드 검색 알고리즘, LLM 기반 질문 답변을 REST API와 CLI 인터페이스로 제공합니다.

### 🚀 주요 기능

- **🔍 고급 검색 기능**
  - **키워드 검색**: CQL 지원 전통적인 텍스트 기반 검색
  - **시맨틱 검색**: 임베딩을 사용한 벡터 기반 유사도 검색
  - **하이브리드 검색**: RRF(Reciprocal Rank Fusion)로 키워드와 시맨틱 검색 결합
  
- **🤖 RAG 통합**
  - Confluence 콘텐츠에서 맥락적 답변 생성
  - LiteLLM을 통한 다중 LLM 제공자 지원
  - 설정 가능한 컨텍스트 윈도우 및 토큰 제한

- **💾 유연한 저장소 옵션**
  - **벡터 데이터베이스**: Qdrant, ChromaDB 또는 인메모리 저장소
  - **임베딩 제공자**: Sentence Transformers(로컬) 또는 LiteLLM(OpenAI, Ollama 등)
  - **콘텐츠 파서**: Markitdown 또는 Unstructured를 사용한 HTML 및 첨부 파일 파싱

- **🛠️ 개발자 친화적**
  - **REST API**: 자동 OpenAPI 문서화가 포함된 FastAPI 기반
  - **CLI 도구**: 모든 기능을 갖춘 명령줄 인터페이스
  - **구성 가능**: 계층적 구성 시스템(환경 변수 > JSON > 기본값)

### 📋 프로젝트 상태

프로젝트는 코드 품질 관점에서 **프로덕션 준비**가 되어 있으며 모든 핵심 기능이 구현되어 있습니다. 그러나 배포 전에 다음을 구현하는 것을 고려하세요:
- API 인증/권한 부여
- 프로덕션 사용을 위한 속도 제한
- 고부하 시나리오를 위한 연결 풀링

### 🏗️ 아키텍처

```
confluence_gateway/
├── adapters/          # 외부 시스템 통합
│   ├── confluence/    # Atlassian API 클라이언트
│   ├── embedding/     # 임베딩 제공자 (sentence-transformers, litellm)
│   └── vector_db/     # 벡터 데이터베이스 (qdrant, chroma)
├── services/          # 비즈니스 로직 계층
│   ├── embedding.py   # 임베딩 오케스트레이션
│   ├── indexing.py    # 청킹을 사용한 문서 인덱싱
│   ├── search.py      # 검색 알고리즘 (키워드, 시맨틱, 하이브리드)
│   ├── generation.py  # LLM 기반 답변 생성
│   ├── ranking.py     # 결과 순위 지정 (하이브리드 검색용 RRF)
│   └── parsers/       # 콘텐츠 파싱 (HTML, PDF, DOCX 등)
├── api/              # FastAPI REST 인터페이스
│   ├── routes/       # API 엔드포인트
│   └── schemas/      # 요청/응답 모델
├── cli/              # Typer CLI 인터페이스
└── core/             # 핵심 유틸리티
    ├── config.py     # 구성 관리
    └── exceptions.py # 사용자 정의 예외
```

### 🚀 빠른 시작

#### 사전 요구 사항

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) 패키지 관리자
- API 액세스가 있는 Confluence 인스턴스
- (선택 사항) 벡터 데이터베이스 (Qdrant 또는 ChromaDB)
- (선택 사항) RAG 기능을 위한 LLM API 키

#### 설치

```bash
# uv가 설치되지 않은 경우 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 저장소 복제
git clone https://github.com/yourusername/confluence-gateway.git
cd confluence-gateway

# 패키지 및 종속성 설치
uv pip install -e .
uv pip install -e ".[dev]"  # 개발용

# pre-commit 훅 설치 (개발용)
pre-commit install
```

#### 구성

`~/.confluence_gateway_config.json`에 구성 파일 생성:

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

전체 옵션은 [구성](#configuration) 섹션을 참조하세요.

#### 기본 사용법

**CLI 예제:**
```bash
# 콘텐츠 검색
confluence-gateway search text "배포 프로세스"

# 시맨틱 검색을 위한 Confluence 페이지 인덱싱
confluence-gateway index pages --space-keys TECH DOC

# 시맨틱 검색 수행
confluence-gateway search semantic "프로덕션에 배포하는 방법"

# AI 답변 생성 (LLM 구성 필요)
confluence-gateway generate answer "우리의 배포 프로세스는 무엇입니까?"
```

**API 예제:**
```bash
# API 서버 시작
uvicorn confluence_gateway.api.app:app --reload

# API를 통한 검색
curl "http://localhost:8000/api/search?query=배포&limit=10"

# 시맨틱 검색
curl -X POST "http://localhost:8000/api/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "배포 프로세스", "top_k": 5}'
```

### 📚 API 문서

API 서버가 실행 중일 때 방문:
- 대화형 API 문서: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 🧪 테스팅

```bash
# 모든 테스트 실행
pytest

# 단위 테스트만 실행
pytest -m "not integration"

# 통합 테스트 실행 (Confluence 액세스 필요)
pytest -m integration

# 커버리지와 함께 실행
pytest --cov=confluence_gateway --cov-report=html
```

### 🔧 개발

```bash
# 린팅 및 포맷팅 실행
ruff check confluence_gateway tests
ruff format confluence_gateway tests

# 타입 체킹
mypy confluence_gateway

# 모든 pre-commit 검사 실행
pre-commit run --all-files
```

### 🤝 기여

1. 저장소를 포크하세요
2. 기능 브랜치를 생성하세요 (`git checkout -b feature/amazing-feature`)
3. 변경 사항을 커밋하세요 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시하세요 (`git push origin feature/amazing-feature`)
5. Pull Request를 여세요

### 📄 라이선스

이 프로젝트는 MIT 라이선스에 따라 라이선스가 부여됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

### 🔮 로드맵

- [ ] API 인증 및 권한 부여
- [ ] 속도 제한 미들웨어
- [ ] Confluence 클라이언트용 연결 풀링
- [ ] MCP (Model Context Protocol) 서버 구현
- [ ] OpenAPI/Swagger UI 개선
- [ ] 실시간 인덱싱을 위한 웹훅 지원
- [ ] 다국어 지원
- [ ] 고급 분석 대시보드