# Confluence Gateway

**Atlassian Confluence를 위한 AI 기반 검색 및 지식 검색**

> **상태**: Beta (v0.1.0)

[English](README.md) | [한국어](README_ko.md)

시맨틱 검색, 하이브리드 알고리즘, AI 기반 질문 답변으로 Confluence를 스마트 지식 베이스로 변환하세요.

## ✨ 주요 기능

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

## 🚀 빠른 시작

**요구사항**: Python 3.10+, Confluence API 액세스

### 1. 설치

```bash
# 복제 및 의존성 설치
cd confluence-gateway
uv sync --dev
uv run pre-commit install
```

### 2. 구성

**환경 변수 (권장):**
```bash
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_USERNAME="your-email@example.com"
export CONFLUENCE_API_TOKEN="YOUR_API_TOKEN"

# 선택사항: AI 기능 (기본값: openrouter/google/gemini-2.5-flash)
export GENERATION_MODEL_NAME="openrouter/google/gemini-2.5-flash"
export GENERATION_LITELLM_API_KEY="YOUR_OPENROUTER_API_KEY"

# 선택사항: 벡터 데이터베이스 (기본값: 메모리 모드)
export QDRANT_URL="http://localhost:6333"  # 또는 ":memory:"
```

**또는 `~/.confluence_gateway_config.json` 설정 파일:**
```json
{
  "confluence": {
    "url": "https://your-instance.atlassian.net",
    "username": "your-email@example.com",
    "api_token": "YOUR_API_TOKEN"
  }
}
```

### 3. 사용하기

**CLI 인터페이스:**
```bash
# 설치 확인
uv run confluence-gateway --version

# 스페이스 목록 조회 및 관리
uv run confluence-gateway spaces list --all
uv run confluence-gateway spaces list --search "dev"

# 콘텐츠 인덱싱
uv run confluence-gateway index trigger --space-keys DEV,TECH
uv run confluence-gateway index status

# 다양한 검색 모드
uv run confluence-gateway search text "배포 가이드"
uv run confluence-gateway search semantic "배포하는 방법"
uv run confluence-gateway search text "프로세스" --hybrid

# AI 답변 얻기
uv run confluence-gateway generate answer "우리의 배포 프로세스는 무엇입니까?"
```

**API 서버:**
```bash
# 개발 서버 시작
uv run uvicorn confluence_gateway.api.app:app --reload
# API 문서: http://localhost:8000/docs

# 상태 확인
curl "http://localhost:8000/health"

# 스페이스 목록 조회
curl "http://localhost:8000/api/spaces"

# 텍스트 검색
curl "http://localhost:8000/api/search?query=배포&limit=10"

# 시맨틱 검색 (중첩된 요청 구조 주의)
curl -X POST "http://localhost:8000/api/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"search_request": {"query": "배포 프로세스", "top_k": 5}}'

# 답변 생성 (중첩된 요청 구조 주의)
curl -X POST "http://localhost:8000/api/generate/answer" \
  -H "Content-Type: application/json" \
  -d '{"gen_request": {"query": "우리의 배포 프로세스는 무엇입니까?"}}'
```

## 🏗️ 아키텍처

**헥사고날 아키텍처** (포트 및 어댑터):

```
confluence_gateway/
├── core/              # 구성, 예외, 스키마
├── services/          # 비즈니스 로직 (Search, Indexing, Generation, Ranking)
├── adapters/          # 외부 통합 (Confluence, Vector DB, Embeddings)
├── api/               # FastAPI REST 인터페이스
└── cli/               # Typer CLI 인터페이스
```

**주요 서비스:**
- `SearchService` - RRF 하이브리드 랭킹을 사용한 다중 모드 검색
- `IndexingService` - 콘텐츠 처리 및 벡터 저장  
- `GenerationService` - RAG 답변 생성
- `EmbeddingService` - 벡터 임베딩 관리
- `RankingService` - Reciprocal Rank Fusion 알고리즘

## 🧪 테스팅

**철학**: E2E 테스팅만 - 단위 테스트, 모킹, 실제 기능 테스트 없음

```bash
# 테스트 환경 설정 (테스트 시 필수)
echo '{"vector_db": {"qdrant_url": ":memory:", "qdrant_local_path": null}}' > ~/.confluence_gateway_config.json
uv run python tests/setup_test_env.py

# 모든 E2E 테스트 실행
uv run pytest tests/ -v

# 카테고리별 실행
uv run pytest tests/cli/ -v     # CLI 테스트
uv run pytest tests/api/ -v     # API 테스트

# 테스트 탐색
uv run pytest tests/ --collect-only
```

**테스팅 요구사항:**
- API 액세스가 가능한 실제 Confluence 인스턴스
- 환경 변수: `CONFLUENCE_URL`, `CONFLUENCE_USERNAME`, `CONFLUENCE_API_TOKEN`
- 테스트 중 벡터 데이터베이스가 자동으로 `:memory:` 모드 사용

## 🔧 개발

**필수 명령어:**
```bash
# 개발 환경 설정
uv sync --dev
uv run pre-commit install

# 개발 워크플로우
uv run confluence-gateway --help                                    # CLI 테스트
uv run uvicorn confluence_gateway.api.app:app --reload             # API 테스트
uv run ruff check --fix && uv run ruff format && uv run mypy confluence_gateway/  # 품질 검사 (커밋 전 필수)
uv run pytest tests/ -v                                            # 모든 테스트 실행

# 개별 품질 도구
uv run ruff check confluence_gateway/    # 린팅
uv run ruff format confluence_gateway/   # 포맷팅  
uv run mypy confluence_gateway/          # 타입 검사
uv run pre-commit run --all-files        # 모든 pre-commit 훅
```

**🚨 중요: 커밋 전 항상 품질 검사 실행:**
```bash
uv run ruff check --fix && uv run ruff format && uv run mypy confluence_gateway/
```

## 📚 API 참조

### 핵심 엔드포인트

#### 상태 확인
```http
GET /health
```

#### 스페이스
```http
GET /api/spaces                    # 모든 스페이스 목록 조회
GET /api/spaces/{space_key}        # 스페이스 상세 정보 조회
```

#### 검색
```http
GET /api/search?query=text&limit=20                # 텍스트 검색
POST /api/search/semantic                          # 시맨틱 검색
POST /api/search/advanced                          # 고급 검색  
POST /api/search/cql                               # CQL 검색
```

#### 생성
```http
POST /api/generate/answer                          # RAG 답변 생성
```

#### 인덱싱
```http
POST /api/index/trigger                            # 인덱싱 실행
GET /api/index/status                              # 인덱싱 상태 확인
```

### 요청 스키마

**중요**: 모든 POST 엔드포인트는 중첩된 요청 객체가 필요합니다:

```json
# 시맨틱 검색
{"search_request": {"query": "배포", "top_k": 10}}

# 답변 생성  
{"gen_request": {"query": "우리의 프로세스는?", "top_k_retrieval": 5}}

# 고급 검색
{"request": {"query": "api", "space_key": "TECH", "limit": 20}}

# 인덱싱
{"space_keys": ["DEV", "TECH"]} 
{"index_all": true}
```

## ⚙️ 구성

**우선순위**: 환경 변수 > `~/.confluence_gateway_config.json` > 기본값

### 벡터 데이터베이스 옵션

**Qdrant (기본값):**
```bash
# 로컬 저장소 (영구)
export QDRANT_LOCAL_PATH="~/.confluence_gateway/qdrant_storage"

# 서버 모드  
export QDRANT_URL="http://localhost:6333"
```

**ChromaDB:**
```bash
export VECTOR_DB_TYPE="chroma"
export CHROMA_PERSIST_PATH="~/.confluence_gateway/chroma_storage"
```

### AI 생성 설정

```bash
# 기본 모델 (https://openrouter.ai/에서 OpenRouter API 키 필요)
export GENERATION_MODEL_NAME="openrouter/google/gemini-2.5-flash"
export GENERATION_LITELLM_API_KEY="YOUR_API_KEY"
```

## 🛠️ 기술 스택

- **백엔드**: Python 3.10+, FastAPI, Typer
- **패키지 매니저**: UV (pip 아님)
- **벡터 데이터베이스**: Qdrant, ChromaDB
- **AI/ML**: LiteLLM, SentenceTransformers, LlamaIndex
- **코드 품질**: Ruff, MyPy, pre-commit 훅
- **테스팅**: pytest E2E

## 🔒 보안

⚠️ **내장 인증 없음** - 인증이 포함된 리버스 프록시(nginx/Apache) 사용
- API 토큰은 환경 변수에만 저장
- 환경에 맞게 CORS 적절히 구성
- 프로덕션에서 API 서버 네트워크 액세스 제한

## 📄 라이선스

MIT 라이선스 - [LICENSE](LICENSE) 파일 참조

---

*Confluence 문서를 AI 기반 인사이트가 포함된 지능형 검색 가능한 지식 베이스로 변환하세요.*