# Confluence Gateway

[![Project Status: Beta – The project is in beta. Features are complete but not ready for production use.](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/yourusername/confluence-gateway)
[![Version: 0.1.0](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/yourusername/confluence-gateway)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

[English](#english) | [한국어](#한국어)

---

## Confluence Gateway (Beta) <a name="english"></a>

**AI-Powered Confluence Search with RAG Integration (Beta)**

Confluence Gateway enhances your Atlassian Confluence knowledge base with modern AI capabilities, providing semantic search, hybrid search algorithms, and LLM-powered question answering through both REST API and CLI interfaces.

> ⚠️ **Beta Software**: This project is currently in beta (v0.1.0). While core features are implemented and tested, it is not yet ready for production use due to missing security features and performance optimizations.

### 🚀 Key Features

> 🛠️ All features below are implemented and functional in the beta version

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

**Current Status: Beta (v0.1.0)**

The project has all core features implemented and tested, but is **NOT production-ready** due to critical missing components:

#### ✅ Implemented Features
- All search capabilities (keyword, semantic, hybrid)
- RAG generation with multiple LLM providers
- Vector database integrations (Qdrant, ChromaDB)
- Complete REST API and CLI interfaces
- Content parsing for HTML and various attachments
- Comprehensive test coverage

#### ❌ Missing for Production
- **No authentication/authorization** - All API endpoints are public
- **No rate limiting** - Vulnerable to DoS attacks
- **Wide-open CORS** - Security risk for web deployments
- **No connection pooling** - Performance limitations
- **No caching layer** - Every request hits the backend
- **Plain-text secrets** - API tokens stored insecurely

### 🏗️ Architecture

The project follows Clean Architecture principles with clear separation of concerns:

```
confluence_gateway/
├── adapters/          # External system integrations (Ports & Adapters pattern)
│   ├── confluence/    # Atlassian API client with retry logic
│   ├── embedding/     # Embedding providers (sentence-transformers, litellm)
│   └── vector_db/     # Vector databases (qdrant, chroma)
├── services/          # Business logic layer (Use Cases)
│   ├── embedding.py   # Embedding orchestration
│   ├── indexing.py    # Document indexing (singleton pattern)
│   ├── search.py      # Search algorithms (keyword, semantic, hybrid)
│   ├── generation.py  # LLM-based answer generation
│   ├── ranking.py     # Result ranking (RRF for hybrid search)
│   └── parsers/       # Content parsing with factory pattern
├── api/              # FastAPI REST interface (Delivery Mechanism)
│   ├── routes/       # API endpoints (NO authentication)
│   └── schemas/      # Pydantic request/response models
├── cli/              # Typer CLI interface (Delivery Mechanism)
└── core/             # Core utilities and domain
    ├── config.py     # Hierarchical configuration with validation
    └── exceptions.py # Custom exception hierarchy
```

### ⚠️ Security Warning

**DO NOT deploy this to a public server without implementing:**
1. Authentication & Authorization
2. Rate Limiting
3. Proper CORS configuration
4. Secure secret management

See the [Security Considerations](#security-considerations) section for details.

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

### 🔒 Security Considerations <a name="security-considerations"></a>

#### Current Security Issues

1. **No Authentication** - All endpoints are public:
   ```python
   # Current: No security
   @router.get("/search")
   async def search(query: str):
       return results
   ```

2. **Wide-open CORS**:
   ```python
   # Current: Allows all origins
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # Security risk!
       allow_methods=["*"],
       allow_headers=["*"]
   )
   ```

3. **No Rate Limiting** - Vulnerable to abuse

4. **Plain-text Secrets** - API tokens in JSON config files

#### Required Security Implementation

Before production deployment, implement:

```python
# 1. Add Authentication
from fastapi.security import HTTPBearer
security = HTTPBearer()

# 2. Add Rate Limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

# 3. Configure CORS Properly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"]
)

# 4. Use Environment Variables for Secrets
# Never commit API tokens to repository!
```

### 🏗️ Current Limitations

#### Performance
- **No Connection Pooling** - Creates new session for each request
- **Sequential Processing** - Indexing is not parallelized
- **No Caching** - Every search hits the vector database
- **Synchronous Operations** - Limited async support

#### Scalability
- **Singleton Indexing** - Only one indexing operation at a time
- **Memory Constraints** - Large indexing jobs may exhaust memory
- **No Horizontal Scaling** - Single instance only

#### Features
- **No Real-time Updates** - Manual re-indexing required
- **Limited File Format Support** - Some attachment types not supported
- **No Multi-tenancy** - Single Confluence instance only

### 🔮 Roadmap

#### Phase 1: Security & Production Readiness (v0.2.0)
- [ ] API Authentication & Authorization
- [ ] Rate Limiting Middleware
- [ ] Secure Secret Management
- [ ] Proper CORS Configuration
- [ ] Connection Pooling
- [ ] Basic Caching Layer

#### Phase 2: Performance & Scalability (v0.3.0)
- [ ] Async Operations Throughout
- [ ] Parallel Indexing
- [ ] Redis Cache Integration
- [ ] Horizontal Scaling Support
- [ ] Batch Processing

#### Phase 3: Advanced Features (v0.4.0)
- [ ] MCP (Model Context Protocol) Server
- [ ] Webhook Support for Real-time Indexing
- [ ] Multi-tenancy Support
- [ ] Advanced Analytics Dashboard
- [ ] Admin UI for Configuration

---

## Confluence Gateway (베타) <a name="한국어"></a>

**AI 기반 Confluence 검색 및 RAG 통합 (베타)**

Confluence Gateway는 Atlassian Confluence 지식 베이스에 최신 AI 기능을 추가하여 시맨틱 검색, 하이브리드 검색 알고리즘, LLM 기반 질문 답변을 REST API와 CLI 인터페이스로 제공합니다.

> ⚠️ **베타 소프트웨어**: 이 프로젝트는 현재 베타(v0.1.0)입니다. 핵심 기능은 구현되고 테스트되었지만, 보안 기능 및 성능 최적화가 누락되어 프로덕션 사용에는 아직 준비되지 않았습니다.

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

**현재 상태: 베타 (v0.1.0)**

프로젝트는 모든 핵심 기능이 구현되고 테스트되었지만, 중요한 구성 요소가 누락되어 **프로덕션 준비가 되지 않았습니다**:

#### ✅ 구현된 기능
- 모든 검색 기능 (키워드, 시맨틱, 하이브리드)
- 여러 LLM 제공자를 통한 RAG 생성
- 벡터 데이터베이스 통합 (Qdrant, ChromaDB)
- 완전한 REST API 및 CLI 인터페이스
- HTML 및 다양한 첨부 파일을 위한 콘텐츠 파싱
- 포괄적인 테스트 커버리지

#### ❌ 프로덕션을 위해 누락된 것들
- **인증/권한 부여 없음** - 모든 API 엔드포인트가 공개되어 있음
- **속도 제한 없음** - DoS 공격에 취약함
- **와이드 오픈 CORS** - 웹 배포에 대한 보안 위험
- **연결 풀링 없음** - 성능 제한
- **캐싱 계층 없음** - 모든 요청이 백엔드를 호출함
- **평문 비밀** - API 토큰이 안전하지 않게 저장됨

### 🏗️ 아키텍처

프로젝트는 명확한 관심사 분리를 통한 Clean Architecture 원칙을 따릅니다:

```
confluence_gateway/
├── adapters/          # 외부 시스템 통합 (Ports & Adapters 패턴)
│   ├── confluence/    # 재시도 로직이 있는 Atlassian API 클라이언트
│   ├── embedding/     # 임베딩 제공자 (sentence-transformers, litellm)
│   └── vector_db/     # 벡터 데이터베이스 (qdrant, chroma)
├── services/          # 비즈니스 로직 계층 (Use Cases)
│   ├── embedding.py   # 임베딩 오케스트레이션
│   ├── indexing.py    # 문서 인덱싱 (싱글톤 패턴)
│   ├── search.py      # 검색 알고리즘 (키워드, 시맨틱, 하이브리드)
│   ├── generation.py  # LLM 기반 답변 생성
│   ├── ranking.py     # 결과 순위 지정 (하이브리드 검색용 RRF)
│   └── parsers/       # 팩토리 패턴을 사용한 콘텐츠 파싱
├── api/              # FastAPI REST 인터페이스 (Delivery Mechanism)
│   ├── routes/       # API 엔드포인트 (인증 없음)
│   └── schemas/      # Pydantic 요청/응답 모델
├── cli/              # Typer CLI 인터페이스 (Delivery Mechanism)
└── core/             # 핵심 유틸리티 및 도메인
    ├── config.py     # 검증을 통한 계층적 구성
    └── exceptions.py # 사용자 정의 예외 계층 구조
```

### ⚠️ 보안 경고

**다음을 구현하지 않고 공개 서버에 배포하지 마세요:**
1. 인증 및 권한 부여
2. 속도 제한
3. 적절한 CORS 구성
4. 안전한 비밀 관리

자세한 내용은 [보안 고려사항](#security-considerations-kr) 섹션을 참조하세요.

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

### 🔒 보안 고려사항 <a name="security-considerations-kr"></a>

#### 현재 보안 문제

1. **인증 없음** - 모든 엔드포인트가 공개되어 있음:
   ```python
   # 현재: 보안 없음
   @router.get("/search")
   async def search(query: str):
       return results
   ```

2. **와이드 오픈 CORS**:
   ```python
   # 현재: 모든 원본 허용
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # 보안 위험!
       allow_methods=["*"],
       allow_headers=["*"]
   )
   ```

3. **속도 제한 없음** - 남용에 취약함

4. **평문 비밀** - JSON 구성 파일에 API 토큰 저장

#### 필수 보안 구현

프로덕션 배포 전에 구현:

```python
# 1. 인증 추가
from fastapi.security import HTTPBearer
security = HTTPBearer()

# 2. 속도 제한 추가
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

# 3. CORS 올바르게 구성
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"]
)

# 4. 비밀에 환경 변수 사용
# API 토큰을 저장소에 커밋하지 마세요!
```

### 🏗️ 현재 제한사항

#### 성능
- **연결 풀링 없음** - 각 요청에 대해 새 세션 생성
- **순차 처리** - 인덱싱이 병렬화되지 않음
- **캐싱 없음** - 모든 검색이 벡터 데이터베이스를 호출
- **동기 작업** - 제한된 비동기 지원

#### 확장성
- **싱글톤 인덱싱** - 한 번에 하나의 인덱싱 작업만 가능
- **메모리 제약** - 대규모 인덱싱 작업은 메모리를 소진할 수 있음
- **수평 확장 없음** - 단일 인스턴스만 가능

#### 기능
- **실시간 업데이트 없음** - 수동 재인덱싱 필요
- **제한된 파일 형식 지원** - 일부 첨부 파일 유형이 지원되지 않음
- **다중 테넌시 없음** - 단일 Confluence 인스턴스만 가능

### 🔮 로드맵

#### 1단계: 보안 및 프로덕션 준비 (v0.2.0)
- [ ] API 인증 및 권한 부여
- [ ] 속도 제한 미들웨어
- [ ] 안전한 비밀 관리
- [ ] 적절한 CORS 구성
- [ ] 연결 풀링
- [ ] 기본 캐싱 계층

#### 2단계: 성능 및 확장성 (v0.3.0)
- [ ] 전체 비동기 작업
- [ ] 병렬 인덱싱
- [ ] Redis 캐시 통합
- [ ] 수평 확장 지원
- [ ] 배치 처리

#### 3단계: 고급 기능 (v0.4.0)
- [ ] MCP (Model Context Protocol) 서버
- [ ] 실시간 인덱싱을 위한 웹훅 지원
- [ ] 다중 테넌시 지원
- [ ] 고급 분석 대시보드
- [ ] 구성을 위한 관리자 UI
- [ ] 고급 분석 대시보드