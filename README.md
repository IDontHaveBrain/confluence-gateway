# Confluence Gateway <a name="english"></a>

[![Project Status: WIP – Initial development is in progress.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

[English](#english) | [한국어](#한국어)
<!-- TODO: Add other badges like Build Status, Coverage, License, PyPI version when applicable -->
<!-- [![Build Status](...)](...) -->
<!-- [![Coverage Status](...)](...) -->
<!-- [![License](...)](...) -->
<!-- [![PyPI version](...)](...) -->

**Enhanced Confluence Search and Knowledge Retrieval with RAG and LLMs**

Confluence Gateway aims to bridge the gap between your Confluence knowledge base and modern AI capabilities. It provides enhanced search functionalities, Retrieval-Augmented Generation (RAG) integration, and Large Language Model (LLM) powered answers based on your Confluence documents.

## Overview

Many teams rely on Confluence as their central knowledge repository. However, finding the *right* information quickly can sometimes be challenging using standard search. Confluence Gateway enhances this experience by:

1.  **Indexing:** Processing and embedding Confluence pages.
2.  **Retrieval:** Using semantic search (RAG) to find the most relevant document chunks based on user queries.
3.  **Generation:** Leveraging LLMs to synthesize information from retrieved documents and provide direct, contextual answers.
4.  **Direct API Access:** Offering standard keyword search via the Confluence API.
5.  **MCP Server:** Providing specific MCP-related functionalities (Details TBD).

This project is designed for teams and developers looking to unlock deeper insights and improve information accessibility within their Confluence instances.

## Project Status

This project is currently in the **early stages of development (Alpha)**.

## Configuration

The application can be configured using a JSON file or environment variables.

**Priority:**

1. **User Configuration File:** Settings defined in `~/.confluence_gateway_config.json`.
2. **Environment Variables:** Variables prefixed with `CONFLUENCE_`, `SEARCH_`, `VECTOR_DB_`, etc.
3. **Default Values:** Built-in defaults within the application.

**Configuration File (`~/.confluence_gateway_config.json`):**

Create a JSON file in your home directory with the following structure (only include sections and keys you want to override):

```json
{
  "confluence": {
    "url": "https://your-confluence-instance.atlassian.net",
    "username": "your_email@example.com",
    "api_token": "YOUR_CONFLUENCE_API_TOKEN", // Use env var CONFLUENCE_API_TOKEN preferably
    "timeout": 15
  },
  "search": {
    "default_limit": 25,
    "max_limit": 150,
    "default_expand": ["body.view", "version", "space"] // Example expand fields
  },
  "embedding": {
    // --- Choose ONE provider ---

    // Option 1: Local Sentence Transformer (Default if model/dimension provided)
    "provider": "sentence-transformers",
    "model_name": "all-MiniLM-L6-v2", // Or another compatible model from HuggingFace
    "dimension": 384,                 // Must match the model's output dimension
    "device": "cpu",                  // Or "cuda" if GPU is available and torch is installed

    // Option 2: LiteLLM (e.g., OpenAI)
    // "provider": "litellm",
    // "model_name": "openai/text-embedding-ada-002",
    // "dimension": 1536,                // Must match the model's output dimension
    // "litellm_api_key": "YOUR_OPENAI_API_KEY", // Use env var LITELLM_API_KEY preferably

    // Option 3: LiteLLM (e.g., Ollama - requires Ollama server running)
    // "provider": "litellm",
    // "model_name": "ollama/nomic-embed-text", // Or other model served by Ollama
    // "dimension": 768,                 // Must match the model's output dimension
    // "litellm_api_base": "http://localhost:11434", // Your Ollama API endpoint (env: LITELLM_API_BASE)

    // Option 4: Disable Embeddings (Default if no other embedding config provided)
    // "provider": "none"
  },
  "vector_db": {
    // --- Choose ONE type (or "none") ---

    // Option 1: Qdrant (Example)
    "type": "qdrant",
    "collection_name": "confluence_prod",
    // IMPORTANT: This dimension MUST match the 'dimension' in the 'embedding' config above!
    "embedding_dimension": 384,
    "qdrant_url": "http://localhost:6333", // Env: QDRANT_URL
    "qdrant_api_key": "OPTIONAL_QDRANT_KEY", // Env: QDRANT_API_KEY
    "qdrant_prefer_grpc": false, // Env: QDRANT_PREFER_GRPC
    "qdrant_grpc_port": 6334, // Env: QDRANT_GRPC_PORT

    // Option 2: ChromaDB (Persistent Local Storage)
    // "type": "chroma",
    // "collection_name": "confluence_chroma_local",
    // "embedding_dimension": 384, // Must match the 'embedding' dimension
    // "chroma_persist_path": "/path/to/your/chroma/data", // Env: CHROMA_PERSIST_PATH

    // Option 3: ChromaDB (Client/Server Mode)
    // "type": "chroma",
    // "collection_name": "confluence_chroma_server",
    // "embedding_dimension": 384, // Must match the 'embedding' dimension
    // "chroma_host": "localhost", // Env: CHROMA_HOST
    // "chroma_port": 8000, // Env: CHROMA_PORT

    // Option 4: Disable Vector DB (Default if no other vector_db config provided)
    // "type": "none",

    // --- Chunking settings (used during indexing if vector_db type is not 'none') ---
    "chunk_size": 512, // Env: VECTOR_DB_CHUNK_SIZE
    "chunk_overlap": 50 // Env: VECTOR_DB_CHUNK_OVERLAP
  }
}
```

---

# Confluence Gateway <a name="한국어"></a>

**RAG 및 LLM을 활용한 향상된 Confluence 검색 및 지식 검색**

Confluence Gateway는 Confluence 지식 베이스와 최신 AI 기능 간의 격차를 해소하는 것을 목표로 합니다. Confluence 문서를 기반으로 향상된 검색 기능, RAG(Retrieval-Augmented Generation) 통합, LLM(Large Language Model) 기반 답변을 제공합니다.

## 설정

애플리케이션은 JSON 파일이나 환경 변수를 사용하여 구성할 수 있습니다.

**우선순위:**

1. **사용자 구성 파일:** `~/.confluence_gateway_config.json`에 정의된 설정.
2. **환경 변수:** `CONFLUENCE_`, `SEARCH_`, `VECTOR_DB_` 등으로 시작하는 변수.
3. **기본값:** 애플리케이션 내의 내장 기본값.

**구성 파일 (`~/.confluence_gateway_config.json`):**

다음과 같은 구조로 홈 디렉토리에 JSON 파일을 생성하세요 (재정의하려는 섹션 및 키만 포함):

```json
{
  "confluence": {
    "url": "https://your-confluence-instance.atlassian.net",
    "username": "your_email@example.com",
    "api_token": "YOUR_CONFLUENCE_API_TOKEN", // 환경 변수 CONFLUENCE_API_TOKEN 사용 권장
    "timeout": 15
  },
  "search": {
    "default_limit": 25,
    "max_limit": 150,
    "default_expand": ["body.view", "version", "space"] // 확장 필드 예시
  },
  "embedding": {
    // --- 제공자(Provider) 중 하나를 선택하세요 ---

    // 옵션 1: 로컬 Sentence Transformer (모델/차원 정보가 제공된 경우 기본값)
    "provider": "sentence-transformers",
    "model_name": "all-MiniLM-L6-v2", // 또는 HuggingFace의 다른 호환 모델
    "dimension": 384,                 // 모델의 출력 차원과 일치해야 함
    "device": "cpu",                  // GPU 사용 가능하고 torch가 설치된 경우 "cuda"

    // 옵션 2: LiteLLM (예: OpenAI)
    // "provider": "litellm",
    // "model_name": "openai/text-embedding-ada-002",
    // "dimension": 1536,                // 모델의 출력 차원과 일치해야 함
    // "litellm_api_key": "YOUR_OPENAI_API_KEY", // 환경 변수 LITELLM_API_KEY 사용 권장

    // 옵션 3: LiteLLM (예: Ollama - Ollama 서버 실행 필요)
    // "provider": "litellm",
    // "model_name": "ollama/nomic-embed-text", // 또는 Ollama에서 제공하는 다른 모델
    // "dimension": 768,                 // 모델의 출력 차원과 일치해야 함
    // "litellm_api_base": "http://localhost:11434", // Ollama API 엔드포인트 (환경 변수: LITELLM_API_BASE)

    // 옵션 4: 임베딩 비활성화 (다른 임베딩 설정이 없는 경우 기본값)
    // "provider": "none"
  },
  "vector_db": {
    // --- 데이터베이스 유형(Type) 중 하나를 선택하세요 (또는 "none") ---

    // 옵션 1: Qdrant (예시)
    "type": "qdrant",
    "collection_name": "confluence_prod",
    // 중요: 이 차원은 위의 'embedding' 설정의 'dimension'과 반드시 일치해야 합니다!
    "embedding_dimension": 384,
    "qdrant_url": "http://localhost:6333", // 환경 변수: QDRANT_URL
    "qdrant_api_key": "OPTIONAL_QDRANT_KEY", // 환경 변수: QDRANT_API_KEY
    "qdrant_prefer_grpc": false, // 환경 변수: QDRANT_PREFER_GRPC
    "qdrant_grpc_port": 6334, // 환경 변수: QDRANT_GRPC_PORT

    // 옵션 2: ChromaDB (영구 로컬 저장소)
    // "type": "chroma",
    // "collection_name": "confluence_chroma_local",
    // "embedding_dimension": 384, // 'embedding' 차원과 일치해야 함
    // "chroma_persist_path": "/path/to/your/chroma/data", // 환경 변수: CHROMA_PERSIST_PATH

    // 옵션 3: ChromaDB (클라이언트/서버 모드)
    // "type": "chroma",
    // "collection_name": "confluence_chroma_server",
    // "embedding_dimension": 384, // 'embedding' 차원과 일치해야 함
    // "chroma_host": "localhost", // 환경 변수: CHROMA_HOST
    // "chroma_port": 8000, // 환경 변수: CHROMA_PORT

    // 옵션 4: 벡터 DB 비활성화 (다른 vector_db 설정이 없는 경우 기본값)
    // "type": "none",

    // --- 청킹 설정 (vector_db type이 'none'이 아닐 경우 인덱싱 시 사용) ---
    "chunk_size": 512, // 환경 변수: VECTOR_DB_CHUNK_SIZE
    "chunk_overlap": 50 // 환경 변수: VECTOR_DB_CHUNK_OVERLAP
  }
}
```

## 개요

많은 팀이 Confluence를 중앙 지식 저장소로 사용합니다. 그러나 표준 검색만으로는 *정확한* 정보를 빠르게 찾는 것이 어려울 수 있습니다. Confluence Gateway는 다음을 통해 이러한 경험을 향상시킵니다:

1.  **인덱싱:** Confluence 페이지 처리 및 임베딩.
2.  **검색 (Retrieval):** 사용자 쿼리를 기반으로 가장 관련성 높은 문서 청크를 찾기 위해 시맨틱 검색(RAG) 사용.
3.  **생성 (Generation):** 검색된 문서의 정보를 종합하고 직접적이고 맥락에 맞는 답변을 제공하기 위해 LLM 활용.
4.  **직접 API 접근:** Confluence API를 통한 표준 키워드 검색 제공.
5.  **MCP 서버:** 특정 MCP 관련 기능 제공 (세부 사항 추후 결정).

이 프로젝트는 Confluence 인스턴스 내에서 더 깊은 통찰력을 얻고 정보 접근성을 개선하려는 팀과 개발자를 위해 설계되었습니다.

## 프로젝트 상태

이 프로젝트는 현재 **초기 개발 단계(알파)**입니다.
