# Confluence Gateway

**Atlassian Confluence를 위한 AI 기반 검색 및 지식 검색**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **상태**: Beta (v0.1.0)

[English](README.md) | [한국어](README_ko.md)

시맨틱 검색, 하이브리드 알고리즘, AI 기반 질문 답변으로 Confluence를 스마트 지식 베이스로 변환하세요.

## ✨ 주요 기능

- **🔍 고급 검색**: Reciprocal Rank Fusion을 사용한 텍스트, 시맨틱, 하이브리드 검색
- **🤖 RAG 기반 Q&A**: Confluence 콘텐츠에서 맥락적 답변 생성  
- **⚡ 이중 인터페이스**: 자동화를 위한 CLI + 통합을 위한 REST API
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

Confluence 자격 증명 설정:

```bash
export CONFLUENCE_URL="https://your-instance.atlassian.net"
export CONFLUENCE_USERNAME="your-email@example.com"
export CONFLUENCE_API_TOKEN="YOUR_API_TOKEN"
```

**선택사항**: AI 기능 활성화 (API 키 필요)
```bash
export GENERATION_MODEL_NAME="openrouter/google/gemini-2.5-flash"
export GENERATION_LITELLM_API_KEY="YOUR_OPENROUTER_API_KEY"
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

**필수 명령어:**
```bash
# 코드 품질 (커밋 전 실행)
uv run ruff check --fix && uv run ruff format && uv run mypy confluence_gateway/

# 테스트 실행
echo '{"vector_db": {"qdrant_url": ":memory:", "qdrant_local_path": null}}' > ~/.confluence_gateway_config.json
uv run pytest tests/ -v

# API 서버 (자동 재로드)
uv run uvicorn confluence_gateway.api.app:app --reload
```

**참고**: 테스트는 실제 Confluence 인스턴스를 사용하며 인증 환경 변수가 필요합니다.

## 📚 API 참조

**인터랙티브 문서**: `http://localhost:8000/docs` (Swagger UI)

**주요 엔드포인트 요청 형식:**

```json
# 시맨틱 검색
{"query": "배포", "top_k": 10}

# 답변 생성  
{"query": "우리의 프로세스는?", "top_k_retrieval": 5}

# 인덱싱
{"space_keys": ["DEV", "TECH"]} 
{"index_all": true}
```

## ⚙️ 고급 구성

**설정 파일 대안** (`~/.confluence_gateway_config.json`):
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

**구성 우선순위**: `~/.confluence_gateway_config.json` > 환경 변수 > 기본값

**저장소 옵션:**
```bash
# 영구 Qdrant (기본값)
export QDRANT_LOCAL_PATH="~/.confluence_gateway/qdrant_storage"

# ChromaDB 대안  
export VECTOR_DB_TYPE="chroma"
export CHROMA_PERSIST_PATH="~/.confluence_gateway/chroma_storage"

# 메모리 전용 모드 (테스트)
export QDRANT_URL=":memory:"

# 개발 모드 (빠른 시작)
export CONFLUENCE_GATEWAY_DEV_MODE="true"
```

## 🛠️ 문제 해결

**연결 문제:**
- `401 Unauthorized`: Confluence에서 API 토큰과 권한 확인
- `Connection refused`: Confluence URL에 `https://` 포함 확인
- 연결 테스트: `curl -u "email:token" "https://your-instance.atlassian.net/rest/api/space"`

**성능:**
- 개발 모드 사용: `export CONFLUENCE_GATEWAY_DEV_MODE="true"`
- 테스트용 메모리 모드: `export QDRANT_URL=":memory:"`

**일반적인 오류:**
- `ModuleNotFoundError`: `uv sync --dev` 실행
- 스페이스 누락: API 토큰의 Confluence 권한 확인

## 📄 라이선스

MIT 라이선스 - [LICENSE](LICENSE) 파일 참조

---

*Confluence 문서를 AI 기반 인사이트가 포함된 지능형 검색 가능한 지식 베이스로 변환하세요.*