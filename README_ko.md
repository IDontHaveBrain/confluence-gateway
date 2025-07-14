# Confluence Gateway

**Atlassian Confluence를 위한 AI 기반 검색 및 지식 검색**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **상태**: Beta (v0.1.0)

시맨틱 검색, 하이브리드 알고리즘, AI 기반 질문 답변으로 Confluence를 스마트 지식 베이스로 변환하세요.

## ✨ 주요 기능

- **🔍 고급 검색**: Reciprocal Rank Fusion을 사용한 텍스트, 시맨틱, 하이브리드 검색
- **🤖 RAG 기반 Q&A**: Confluence 콘텐츠에서 맥락적 답변 생성  
- **⚡ 이중 인터페이스**: 자동화를 위한 CLI + 통합을 위한 REST API
- **🚀 GPU 가속**: 5-10배 성능 향상을 위한 GPU 자동 감지 지원
- **🗄️ 유연한 저장소**: Qdrant, ChromaDB 또는 메모리 전용 모드

## 🚀 빠른 시작

**요구사항**: Python 3.10+, Confluence API 액세스

### 1. 설치

```bash
git clone https://github.com/IDontHaveBrain/confluence-gateway.git
cd confluence-gateway
uv sync --dev
```

### 2. 구성

Confluence 자격 증명을 설정하세요:

```bash
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_USERNAME="your-email@example.com"
export CONFLUENCE_API_TOKEN="YOUR_API_TOKEN"
```

**선택적 구성:**
```bash
# AI 기능 (API 키 필요)
export GENERATION_MODEL_NAME="openrouter/google/gemini-2.5-flash"
export GENERATION_LITELLM_API_KEY="YOUR_OPENROUTER_API_KEY"

# GPU 제어 (기본적으로 자동 감지)
export EMBEDDING_DEVICE="cuda"    # GPU 강제 사용
export EMBEDDING_DEVICE="cpu"     # CPU 폴백 강제 사용

# 벡터 저장소
export QDRANT_URL="http://localhost:6333"  # 영구 저장소
```

### 3. 연결 확인

```bash
# 설치 및 연결 테스트
uv run confluence-gateway --version
uv run confluence-gateway spaces list
```

✅ **성공**: 오류 없이 Confluence 스페이스 목록이 표시되어야 합니다.

### 4. 인터페이스 선택

**자동화 및 스크립팅을 위한 CLI:**
```bash
# 콘텐츠 인덱싱
uv run confluence-gateway index trigger --space-keys DEV,TECH

# 다양한 검색 모드  
uv run confluence-gateway search text "배포 가이드"
uv run confluence-gateway search semantic "배포하는 방법"

# AI 답변 (AI 구성 필요)
uv run confluence-gateway generate answer "우리의 배포 프로세스는 무엇입니까?"
```

**통합 및 웹 앱을 위한 API:**
```bash
# 개발 서버 시작
uv run uvicorn confluence_gateway.api.app:app --reload

# 인터랙티브 문서: http://localhost:8000/docs
# 상태 확인: curl "http://localhost:8000/health"
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

## 🔧 개발

**설정 및 품질:**
```bash
# 개발 종속성과 함께 설치
uv sync --dev

# 코드 품질 (커밋 전 실행)
uv run ruff check --fix && uv run ruff format && uv run mypy confluence_gateway/

# 테스트 실행 (Confluence 인증 필요)
uv run pytest tests/ -v
```

**개발 서버:**
```bash
# 자동 재로드 + 빠른 시작을 위한 API
export CONFLUENCE_GATEWAY_DEV_MODE="true"
uv run uvicorn confluence_gateway.api.app:app --reload
```

## 📚 API 참조

**인터랙티브 문서**: `http://localhost:8000/docs` (Swagger UI)

**빠른 API 테스트:**
```bash
# 텍스트 검색 (GET)
curl "http://localhost:8000/api/search?query=deployment&limit=5"

# 시맨틱 검색 (POST)
curl -X POST "http://localhost:8000/api/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{"query": "deployment process", "top_k": 5}'

# 상태 확인
curl "http://localhost:8000/health"
```

## ⚙️ 구성

**구성 파일** (`~/.confluence_gateway_config.json`):
```json
{
  "confluence": {
    "url": "https://your-instance.atlassian.net",
    "username": "your-email@example.com",
    "api_token": "YOUR_API_TOKEN"
  },
  "vector_db": {
    "type": "qdrant",
    "qdrant_url": "http://localhost:6333"
  },
  "embedding": {
    "device": "cuda"  // CPU 전용의 경우 "cpu"
  }
}
```

**우선순위**: 구성 파일 > 환경 변수 > 기본값

**저장소 옵션**: Qdrant (기본값), ChromaDB (`VECTOR_DB_TYPE=chroma`), 또는 테스트용 메모리 전용

## 📄 라이선스

MIT 라이선스 - [LICENSE](LICENSE) 파일 참조

---

*Confluence 문서를 AI 기반 인사이트가 포함된 지능형 검색 가능한 지식 베이스로 변환하세요.*