# Confluence Gateway <a name="english"></a>

[![Project Status: WIP – Initial development is in progress.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

[English](#english) | [한국어](#한국어)


**Enhanced Confluence Search and Knowledge Retrieval with RAG and LLMs**

Confluence Gateway aims to bridge the gap between your Confluence knowledge base and modern AI capabilities. It provides enhanced search functionalities, Retrieval-Augmented Generation (RAG) integration, and Large Language Model (LLM) powered answers based on your Confluence documents.

## Overview

Many teams rely on Confluence as their central knowledge repository. However, finding the *right* information quickly can sometimes be challenging using standard search. Confluence Gateway enhances this experience by:

1.  **Indexing:** Processing and embedding Confluence pages and attachments for semantic search.
2.  **Retrieval:** Using keyword, semantic (vector), or hybrid search to find the most relevant documents or document chunks based on user queries.
3.  **Generation:** Leveraging LLMs to synthesize information from retrieved documents and provide direct, contextual answers (RAG).
4.  **API & CLI:** Offering access to all functionalities via a FastAPI interface and a Typer-based command-line tool.
5.  **MCP Server:** Providing specific MCP-related functionalities (Details TBD).

This project is designed for teams and developers looking to unlock deeper insights and improve information accessibility within their Confluence instances.

## Project Status

This project is currently in the **early stages of development (Alpha)**. Core functionalities are being built and refined.

## Configuration

The application can be configured using a JSON file or environment variables.

**Priority:**

1. **User Configuration File:** Settings defined in `~/.confluence_gateway_config.json`.
2. **Environment Variables:** Variables prefixed with `CONFLUENCE_`, `SEARCH_`, `VECTOR_DB_`, `EMBEDDING_`, `INDEXING_`, `GENERATION_`.
3. **Default Values:** Built-in defaults within the application.

**Configuration File (`~/.confluence_gateway_config.json`):**

Create a JSON file in your home directory with the following structure (only include sections and keys you want to override):

```json
{
  "confluence": {
    "url": "https://your-confluence-instance.atlassian.net",
    "username": "your_email@example.com",
    "api_token": "YOUR_CONFLUENCE_API_TOKEN", // Use env var CONFLUENCE_API_TOKEN preferably
    "timeout": 15 // Request timeout in seconds
  },
  "search": {
    "default_limit": 20, // Default results per page for keyword/CQL search
    "max_limit": 100,   // Max results per page allowed
    "default_expand": ["body.view", "space", "version"], // Default Confluence fields to expand
    "hybrid_search_enabled": false, // Enable/disable hybrid search globally (env: SEARCH_HYBRID_SEARCH_ENABLED)
    "hybrid_keyword_fetch_limit": 50, // How many keyword results to fetch for hybrid (env: SEARCH_HYBRID_KEYWORD_FETCH_LIMIT)
    "hybrid_semantic_fetch_limit": 50, // How many semantic results to fetch for hybrid (env: SEARCH_HYBRID_SEMANTIC_FETCH_LIMIT)
    "hybrid_rrf_k": 60 // Reciprocal Rank Fusion constant 'k' (env: SEARCH_HYBRID_RRF_K)
  },
  "embedding": {
    // --- Choose ONE provider ---

    // Option 1: Local Sentence Transformer (Default if model/dimension provided)
    "provider": "sentence-transformers", // (env: EMBEDDING_PROVIDER)
    "model_name": "all-MiniLM-L6-v2", // Or another compatible model from HuggingFace (env: EMBEDDING_MODEL_NAME)
    "dimension": 384,                 // Must match the model's output dimension (env: EMBEDDING_DIMENSION)
    "device": "cpu",                  // Or "cuda" if GPU is available and torch is installed (env: EMBEDDING_DEVICE)

    // Option 2: LiteLLM (e.g., OpenAI, Azure OpenAI, Cohere, etc.)
    // "provider": "litellm", // (env: EMBEDDING_PROVIDER)
    // "model_name": "openai/text-embedding-ada-002", // (env: EMBEDDING_MODEL_NAME)
    // "dimension": 1536,                // Must match the model's output dimension (env: EMBEDDING_DIMENSION)
    // "litellm_api_key": "YOUR_PROVIDER_API_KEY", // Use env var LITELLM_API_KEY preferably
    // "litellm_api_base": "YOUR_PROVIDER_API_BASE_IF_NEEDED" // (env: LITELLM_API_BASE)

    // Option 3: LiteLLM (e.g., Ollama - requires Ollama server running)
    // "provider": "litellm", // (env: EMBEDDING_PROVIDER)
    // "model_name": "ollama/nomic-embed-text", // Or other model served by Ollama (env: EMBEDDING_MODEL_NAME)
    // "dimension": 768,                 // Must match the model's output dimension (env: EMBEDDING_DIMENSION)
    // "litellm_api_base": "http://localhost:11434", // Your Ollama API endpoint (env: LITELLM_API_BASE)

    // Option 4: Disable Embeddings (Default if no other embedding config provided)
    // "provider": "none" // (env: EMBEDDING_PROVIDER)
  },
  "vector_db": {
    // --- Choose ONE type (or "none") ---

    // Option 1: Qdrant
    "type": "qdrant", // (env: VECTOR_DB_TYPE)
    "collection_name": "confluence_embeddings", // (env: VECTOR_DB_COLLECTION_NAME)
    // IMPORTANT: This dimension MUST match the 'dimension' in the 'embedding' config above!
    "embedding_dimension": 384, // (env: VECTOR_DB_EMBEDDING_DIMENSION)
    "qdrant_url": "http://localhost:6333", // Or ":memory:" for in-memory (env: QDRANT_URL)
    "qdrant_api_key": null, // Optional Qdrant API Key (env: QDRANT_API_KEY)
    "qdrant_prefer_grpc": false, // Use gRPC instead of REST (env: QDRANT_PREFER_GRPC)
    "qdrant_grpc_port": 6334, // gRPC port if prefer_grpc is true (env: QDRANT_GRPC_PORT)

    // Option 2: ChromaDB (Persistent Local Storage)
    // "type": "chroma", // (env: VECTOR_DB_TYPE)
    // "collection_name": "confluence_chroma_local", // (env: VECTOR_DB_COLLECTION_NAME)
    // "embedding_dimension": 384, // Must match the 'embedding' dimension (env: VECTOR_DB_EMBEDDING_DIMENSION)
    // "chroma_persist_path": "/path/to/your/chroma/data", // (env: CHROMA_PERSIST_PATH)

    // Option 3: ChromaDB (Client/Server Mode)
    // "type": "chroma", // (env: VECTOR_DB_TYPE)
    // "collection_name": "confluence_chroma_server", // (env: VECTOR_DB_COLLECTION_NAME)
    // "embedding_dimension": 384, // Must match the 'embedding' dimension (env: VECTOR_DB_EMBEDDING_DIMENSION)
    // "chroma_host": "localhost", // (env: CHROMA_HOST)
    // "chroma_port": 8000, // (env: CHROMA_PORT)

    // Option 4: Disable Vector DB (Default if no other vector_db config provided)
    // "type": "none", // (env: VECTOR_DB_TYPE)

    // --- Chunking settings (used during indexing if vector_db type is not 'none') ---
    "chunk_size": 512, // Target size of text chunks (env: VECTOR_DB_CHUNK_SIZE)
    "chunk_overlap": 50 // Overlap between consecutive chunks (env: VECTOR_DB_CHUNK_OVERLAP)
  },
  "indexing": {
    // Optional: Specify which spaces to index. If 'include_spaces' is set, only these are indexed.
    "include_spaces": null, // ["DEV", "PRODUCT"] (env: INDEXING_INCLUDE_SPACES comma-separated)
    // Optional: Specify spaces to exclude. Applied *after* 'include_spaces' if set.
    "exclude_spaces": null, // ["ARCHIVE", "TEST"] (env: INDEXING_EXCLUDE_SPACES comma-separated)

    // Choose parser for HTML content: "markitdown" (default) or "unstructured"
    // Requires installing optional dependencies for the chosen parser, e.g.:
    // pip install markitdown OR pip install unstructured
    "html_parser": "markitdown", // (env: INDEXING_HTML_PARSER)

    // --- Attachment Indexing Settings ---
    "include_attachments": false, // Set to true to index attachments (env: INDEXING_INCLUDE_ATTACHMENTS)
    "max_attachment_size_mb": 10, // Max size limit in MB (env: INDEXING_MAX_ATTACHMENT_SIZE_MB)
    // Allowed extensions (lowercase, no dot). Null or empty list means allow all.
    "allowed_attachment_extensions": ["pdf", "docx", "pptx", "txt", "md"], // (env: INDEXING_ALLOWED_ATTACHMENT_EXTENSIONS comma-separated)
    // Choose parser for attachments: "markitdown" (default) or "unstructured"
    // Requires installing optional dependencies, e.g.:
    // pip install "markitdown[pdf,docx]" OR pip install "unstructured[local-inference]"
    "attachment_parser": "markitdown" // (env: INDEXING_ATTACHMENT_PARSER)
  },
  "generation": {
    // --- RAG Generation Settings ---
    "enable": false, // Enable/disable the /generate/answer endpoint and CLI command (env: GENERATION_ENABLE)
    "provider": "litellm", // Currently only litellm is supported (env: GENERATION_PROVIDER)
    "model_name": null, // e.g., "openai/gpt-4o", "ollama/llama3" (env: GENERATION_MODEL_NAME)
    "litellm_api_key": null, // API key for the generation model provider (env: GENERATION_LITELLM_API_KEY)
    "litellm_api_base": null, // API base URL if needed (e.g., for Ollama) (env: GENERATION_LITELLM_API_BASE)
    "prompt_template": "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:", // (env: GENERATION_PROMPT_TEMPLATE)
    "max_context_tokens": 3000, // Max tokens from retrieved docs to feed into prompt (env: GENERATION_MAX_CONTEXT_TOKENS)
    "max_output_tokens": 500, // Max tokens the LLM should generate (env: GENERATION_MAX_OUTPUT_TOKENS)
    "temperature": 0.1, // Generation temperature (0.0-2.0) (env: GENERATION_TEMPERATURE)
    "generation_timeout": 60 // Timeout for the LLM call in seconds (env: GENERATION_GENERATION_TIMEOUT)
  }
}
```

## Testing

This project employs an **optimized testing strategy** focused on verifying essential functionality through integration tests while minimizing complex mocks.

**Key Principles:**

*   **Integration First:** Tests prioritize verifying interactions between components and external systems (Confluence, Vector DBs, local Embedding Models) when configurations allow. E2E tests (API & CLI) are highly valued.
*   **Test Public Interfaces:** Focus on testing the public APIs of services, adapters, CLI commands, and API endpoints. Private methods (`_method`) are not tested directly.
*   **Strategic Mocking Only:** Mocks are used sparingly, mainly for external API calls (like LLMs via LiteLLM) or pragmatically within `IndexingService` tests to isolate its complex orchestration logic (mocking only the `ConfluenceClient` calls within these specific tests).
*   **Focus on Core Scenarios:** Prioritize testing the main "happy path" and critical error handling for major features.

**Running Tests:**

*   **Prerequisites for Integration Tests:**
    *   Valid Confluence credentials must be configured via environment variables or the `~/.confluence_gateway_config.json` file (see [Configuration](#configuration)).
    *   The target Confluence instance should have some content (spaces, pages) for tests to interact with.
    *   For semantic search and RAG tests (including CLI commands like `search semantic`, `search text --hybrid`, `generate answer`), appropriate embedding/vector DB configurations (or defaults using local/in-memory options) are needed.
*   **Run only Unit Tests (no external dependencies):**
    ```bash
    pytest -m "not integration"
    ```
*   **Run only Integration Tests (requires configured external services):**
    ```bash
    pytest -m integration
    ```
*   **Run all tests (will skip integration tests if dependencies are not met):**
    ```bash
    pytest
    ```

---

# Confluence Gateway <a name="한국어"></a>

**RAG 및 LLM을 활용한 향상된 Confluence 검색 및 지식 검색**

Confluence Gateway는 Confluence 지식 베이스와 최신 AI 기능 간의 격차를 해소하는 것을 목표로 합니다. Confluence 문서를 기반으로 향상된 검색 기능, RAG(Retrieval-Augmented Generation) 통합, LLM(Large Language Model) 기반 답변을 제공합니다.

## 개요

많은 팀이 Confluence를 중앙 지식 저장소로 사용합니다. 그러나 표준 검색만으로는 *정확한* 정보를 빠르게 찾는 것이 어려울 수 있습니다. Confluence Gateway는 다음을 통해 이러한 경험을 향상시킵니다:

1.  **인덱싱:** 시맨틱 검색을 위해 Confluence 페이지 및 첨부 파일 처리 및 임베딩.
2.  **검색 (Retrieval):** 키워드, 시맨틱(벡터) 또는 하이브리드 검색을 사용하여 사용자 쿼리를 기반으로 가장 관련성 높은 문서 또는 문서 청크 찾기.
3.  **생성 (Generation):** 검색된 문서의 정보를 종합하고 직접적이고 맥락에 맞는 답변을 제공하기 위해 LLM 활용 (RAG).
4.  **API & CLI:** FastAPI 인터페이스 및 Typer 기반 명령줄 도구를 통해 모든 기능에 접근 제공.
5.  **MCP 서버:** 특정 MCP 관련 기능 제공 (세부 사항 추후 결정).

이 프로젝트는 Confluence 인스턴스 내에서 더 깊은 통찰력을 얻고 정보 접근성을 개선하려는 팀과 개발자를 위해 설계되었습니다.

## 프로젝트 상태

이 프로젝트는 현재 **초기 개발 단계(알파)**입니다. 핵심 기능들이 구축되고 개선되고 있습니다.

## 설정

애플리케이션은 JSON 파일이나 환경 변수를 사용하여 구성할 수 있습니다.

**우선순위:**

1. **사용자 구성 파일:** `~/.confluence_gateway_config.json`에 정의된 설정.
2. **환경 변수:** `CONFLUENCE_`, `SEARCH_`, `VECTOR_DB_`, `EMBEDDING_`, `INDEXING_`, `GENERATION_` 등으로 시작하는 변수.
3. **기본값:** 애플리케이션 내의 내장 기본값.

**구성 파일 (`~/.confluence_gateway_config.json`):**

다음과 같은 구조로 홈 디렉토리에 JSON 파일을 생성하세요 (재정의하려는 섹션 및 키만 포함):

```json
{
  "confluence": {
    "url": "https://your-confluence-instance.atlassian.net",
    "username": "your_email@example.com",
    "api_token": "YOUR_CONFLUENCE_API_TOKEN", // 환경 변수 CONFLUENCE_API_TOKEN 사용 권장
    "timeout": 15 // 요청 타임아웃 (초)
  },
  "search": {
    "default_limit": 20, // 키워드/CQL 검색 기본 페이지당 결과 수
    "max_limit": 100,   // 허용되는 최대 페이지당 결과 수
    "default_expand": ["body.view", "space", "version"], // 기본 확장 Confluence 필드
    "hybrid_search_enabled": false, // 하이브리드 검색 전역 활성화/비활성화 (환경 변수: SEARCH_HYBRID_SEARCH_ENABLED)
    "hybrid_keyword_fetch_limit": 50, // 하이브리드용 키워드 결과 가져올 개수 (환경 변수: SEARCH_HYBRID_KEYWORD_FETCH_LIMIT)
    "hybrid_semantic_fetch_limit": 50, // 하이브리드용 시맨틱 결과 가져올 개수 (환경 변수: SEARCH_HYBRID_SEMANTIC_FETCH_LIMIT)
    "hybrid_rrf_k": 60 // Reciprocal Rank Fusion 상수 'k' (환경 변수: SEARCH_HYBRID_RRF_K)
  },
  "embedding": {
    // --- 제공자(Provider) 중 하나를 선택하세요 ---

    // 옵션 1: 로컬 Sentence Transformer (모델/차원 정보가 제공된 경우 기본값)
    "provider": "sentence-transformers", // (환경 변수: EMBEDDING_PROVIDER)
    "model_name": "all-MiniLM-L6-v2", // 또는 HuggingFace의 다른 호환 모델 (환경 변수: EMBEDDING_MODEL_NAME)
    "dimension": 384,                 // 모델의 출력 차원과 일치해야 함 (환경 변수: EMBEDDING_DIMENSION)
    "device": "cpu",                  // GPU 사용 가능하고 torch가 설치된 경우 "cuda" (환경 변수: EMBEDDING_DEVICE)

    // 옵션 2: LiteLLM (예: OpenAI, Azure OpenAI, Cohere 등)
    // "provider": "litellm", // (환경 변수: EMBEDDING_PROVIDER)
    // "model_name": "openai/text-embedding-ada-002", // (환경 변수: EMBEDDING_MODEL_NAME)
    // "dimension": 1536,                // 모델의 출력 차원과 일치해야 함 (환경 변수: EMBEDDING_DIMENSION)
    // "litellm_api_key": "YOUR_PROVIDER_API_KEY", // 환경 변수 LITELLM_API_KEY 사용 권장
    // "litellm_api_base": "YOUR_PROVIDER_API_BASE_IF_NEEDED" // (환경 변수: LITELLM_API_BASE)

    // 옵션 3: LiteLLM (예: Ollama - Ollama 서버 실행 필요)
    // "provider": "litellm", // (환경 변수: EMBEDDING_PROVIDER)
    // "model_name": "ollama/nomic-embed-text", // 또는 Ollama에서 제공하는 다른 모델 (환경 변수: EMBEDDING_MODEL_NAME)
    // "dimension": 768,                 // 모델의 출력 차원과 일치해야 함 (환경 변수: EMBEDDING_DIMENSION)
    // "litellm_api_base": "http://localhost:11434", // Ollama API 엔드포인트 (환경 변수: LITELLM_API_BASE)

    // 옵션 4: 임베딩 비활성화 (다른 임베딩 설정이 없는 경우 기본값)
    // "provider": "none" // (환경 변수: EMBEDDING_PROVIDER)
  },
  "vector_db": {
    // --- 데이터베이스 유형(Type) 중 하나를 선택하세요 (또는 "none") ---

    // 옵션 1: Qdrant
    "type": "qdrant", // (환경 변수: VECTOR_DB_TYPE)
    "collection_name": "confluence_embeddings", // (환경 변수: VECTOR_DB_COLLECTION_NAME)
    // 중요: 이 차원은 위의 'embedding' 설정의 'dimension'과 반드시 일치해야 합니다!
    "embedding_dimension": 384, // (환경 변수: VECTOR_DB_EMBEDDING_DIMENSION)
    "qdrant_url": "http://localhost:6333", // 또는 인메모리용 ":memory:" (환경 변수: QDRANT_URL)
    "qdrant_api_key": null, // 선택적 Qdrant API 키 (환경 변수: QDRANT_API_KEY)
    "qdrant_prefer_grpc": false, // REST 대신 gRPC 사용 (환경 변수: QDRANT_PREFER_GRPC)
    "qdrant_grpc_port": 6334, // prefer_grpc가 true일 경우 gRPC 포트 (환경 변수: QDRANT_GRPC_PORT)

    // 옵션 2: ChromaDB (영구 로컬 저장소)
    // "type": "chroma", // (환경 변수: VECTOR_DB_TYPE)
    // "collection_name": "confluence_chroma_local", // (환경 변수: VECTOR_DB_COLLECTION_NAME)
    // "embedding_dimension": 384, // 'embedding' 차원과 일치해야 함 (환경 변수: VECTOR_DB_EMBEDDING_DIMENSION)
    // "chroma_persist_path": "/path/to/your/chroma/data", // (환경 변수: CHROMA_PERSIST_PATH)

    // 옵션 3: ChromaDB (클라이언트/서버 모드)
    // "type": "chroma", // (환경 변수: VECTOR_DB_TYPE)
    // "collection_name": "confluence_chroma_server", // (환경 변수: VECTOR_DB_COLLECTION_NAME)
    // "embedding_dimension": 384, // 'embedding' 차원과 일치해야 함 (환경 변수: VECTOR_DB_EMBEDDING_DIMENSION)
    // "chroma_host": "localhost", // (환경 변수: CHROMA_HOST)
    // "chroma_port": 8000, // (환경 변수: CHROMA_PORT)

    // 옵션 4: 벡터 DB 비활성화 (다른 vector_db 설정이 없는 경우 기본값)
    // "type": "none", // (환경 변수: VECTOR_DB_TYPE)

    // --- 청킹 설정 (vector_db type이 'none'이 아닐 경우 인덱싱 시 사용) ---
    "chunk_size": 512, // 텍스트 청크 목표 크기 (환경 변수: VECTOR_DB_CHUNK_SIZE)
    "chunk_overlap": 50 // 연속된 청크 간의 중첩 (환경 변수: VECTOR_DB_CHUNK_OVERLAP)
  },
  "indexing": {
    // 선택 사항: 인덱싱할 스페이스 지정. 'include_spaces'가 설정되면 이 스페이스들만 고려됩니다.
    "include_spaces": null, // ["DEV", "PRODUCT"] (환경 변수: INDEXING_INCLUDE_SPACES 쉼표로 구분)
    // 선택 사항: 인덱싱에서 제외할 스페이스 지정. 'include_spaces' 설정 후 적용됩니다.
    "exclude_spaces": null, // ["ARCHIVE", "TEST"] (환경 변수: INDEXING_EXCLUDE_SPACES 쉼표로 구분)

    // HTML 콘텐츠 파서 선택: "markitdown" (기본값) 또는 "unstructured"
    // 선택한 파서에 필요한 선택적 의존성 설치 필요, 예:
    // pip install markitdown 또는 pip install unstructured
    "html_parser": "markitdown", // (환경 변수: INDEXING_HTML_PARSER)

    // --- 첨부파일 인덱싱 설정 ---
    "include_attachments": false, // true로 설정 시 첨부파일 인덱싱 활성화 (환경 변수: INDEXING_INCLUDE_ATTACHMENTS)
    "max_attachment_size_mb": 10, // 최대 크기 제한 (MB) (환경 변수: INDEXING_MAX_ATTACHMENT_SIZE_MB)
    // 허용되는 확장자 (소문자, 점 제외). null 또는 빈 리스트는 모두 허용.
    "allowed_attachment_extensions": ["pdf", "docx", "pptx", "txt", "md"], // (환경 변수: INDEXING_ALLOWED_ATTACHMENT_EXTENSIONS 쉼표로 구분)
    // 첨부파일 파서 선택: "markitdown" (기본값) 또는 "unstructured"
    // 선택적 의존성 설치 필요, 예:
    // pip install "markitdown[pdf,docx]" 또는 pip install "unstructured[local-inference]"
    "attachment_parser": "markitdown" // (환경 변수: INDEXING_ATTACHMENT_PARSER)
  },
  "generation": {
    // --- RAG 생성 설정 ---
    "enable": false, // /generate/answer 엔드포인트 및 CLI 명령어 활성화/비활성화 (환경 변수: GENERATION_ENABLE)
    "provider": "litellm", // 현재 litellm만 지원 (환경 변수: GENERATION_PROVIDER)
    "model_name": null, // 예: "openai/gpt-4o", "ollama/llama3" (환경 변수: GENERATION_MODEL_NAME)
    "litellm_api_key": null, // 생성 모델 제공자 API 키 (환경 변수: GENERATION_LITELLM_API_KEY)
    "litellm_api_base": null, // 필요한 경우 API 기본 URL (예: Ollama용) (환경 변수: GENERATION_LITELLM_API_BASE)
    "prompt_template": "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:", // (환경 변수: GENERATION_PROMPT_TEMPLATE)
    "max_context_tokens": 3000, // 프롬프트에 입력할 검색된 문서의 최대 토큰 수 (환경 변수: GENERATION_MAX_CONTEXT_TOKENS)
    "max_output_tokens": 500, // LLM이 생성해야 하는 최대 토큰 수 (환경 변수: GENERATION_MAX_OUTPUT_TOKENS)
    "temperature": 0.1, // 생성 temperature (0.0-2.0) (환경 변수: GENERATION_TEMPERATURE)
    "generation_timeout": 60 // LLM 호출 타임아웃 (초) (환경 변수: GENERATION_GENERATION_TIMEOUT)
  }
}
```

## 테스팅

이 프로젝트는 복잡한 모의(mock) 사용을 최소화하면서 통합 테스트를 통해 필수 기능을 검증하는 데 초점을 맞춘 **최적화된 테스트 전략**을 사용합니다.

**핵심 원칙:**

*   **통합 우선:** 설정이 허용될 때 컴포넌트 간의 상호 작용과 외부 시스템(Confluence, Vector DB, 로컬 임베딩 모델)과의 연동을 검증하는 테스트를 우선시합니다. E2E 테스트(API & CLI)가 가장 중요합니다.
*   **공개 인터페이스 테스트:** 서비스, 어댑터, CLI 명령어, API 엔드포인트의 공개 API 테스트에 집중합니다. 비공개 메서드(`_method`)는 직접 테스트하지 않습니다.
*   **전략적 모의 사용:** 모의는 예외적으로 사용하며, 주로 외부 API 호출(예: LiteLLM을 통한 LLM 호출)이나 `IndexingService` 테스트 내에서 복잡한 오케스트레이션 로직을 분리하기 위해 실용적으로 사용합니다 (이 특정 테스트 내에서만 `ConfluenceClient` 호출을 모의).
*   **핵심 시나리오 집중:** 주요 기능의 기본 성공 경로("happy path")와 중요한 예상 오류(예: 연결 문제, 인증 실패, 잘못된 사용자 입력으로 인한 4xx 오류) 처리를 우선적으로 테스트합니다.

**테스트 실행:**

*   **통합 테스트 사전 요구 사항:**
    *   유효한 Confluence 자격 증명이 환경 변수 또는 `~/.confluence_gateway_config.json` 파일을 통해 구성되어야 합니다 ([설정](#설정) 참조).
    *   대상 Confluence 인스턴스에 테스트가 상호 작용할 콘텐츠(스페이스, 페이지)가 있어야 합니다.
    *   시맨틱 검색 및 RAG 테스트(CLI 명령어 `search semantic`, `search text --hybrid`, `generate answer` 포함)에는 적절한 임베딩/벡터 DB 구성(또는 로컬/인메모리 옵션을 사용하는 기본값)이 필요합니다.
*   **단위 테스트만 실행 (외부 의존성 없음):**
    ```bash
    pytest -m "not integration"
    ```
*   **통합 테스트만 실행 (구성된 외부 서비스 필요):**
    ```bash
    pytest -m integration
    ```
*   **모든 테스트 실행 (의존성이 충족되지 않으면 통합 테스트는 건너뜀):**
    ```bash
    pytest
    ```
