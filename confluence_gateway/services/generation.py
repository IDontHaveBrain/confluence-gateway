import logging
from typing import Any

from confluence_gateway.adapters.vector_db.models import VectorSearchResultItem
from confluence_gateway.core.config import (
    GenerationConfig,
    dev_mode_log_stub,
    is_dev_mode,
)
from confluence_gateway.core.exceptions import (
    GenerationError,
    SemanticSearchError,
)
from confluence_gateway.services.search import SearchService

try:
    import tiktoken

    _tiktoken_available = True
except ImportError:
    _tiktoken_available = False

# Lazy loading for LiteLLM
_litellm = None
_litellm_available = None
_litellm_exceptions = None


def _get_litellm() -> Any:
    global _litellm, _litellm_available
    if _litellm is None:
        try:
            import litellm

            _litellm = litellm
            _litellm_available = True
        except ImportError:
            _litellm_available = False
            raise ImportError("litellm is required for generation features")
    return _litellm


def _get_litellm_exceptions() -> dict[str, Any]:
    global _litellm_exceptions
    if _litellm_exceptions is None:
        try:
            from litellm.exceptions import (
                APIConnectionError,
                AuthenticationError,
                BadRequestError,
                RateLimitError,
                ServiceUnavailableError,
                Timeout,
            )

            _litellm_exceptions = {
                "APIConnectionError": APIConnectionError,
                "AuthenticationError": AuthenticationError,
                "BadRequestError": BadRequestError,
                "RateLimitError": RateLimitError,
                "ServiceUnavailableError": ServiceUnavailableError,
                "Timeout": Timeout,
            }
        except ImportError:
            raise ImportError("litellm is required for generation features")
    return _litellm_exceptions


logger = logging.getLogger(__name__)


class GenerationService:
    def __init__(
        self,
        search_service: SearchService,
        config: GenerationConfig | None,
    ):
        self.search_service = search_service
        self.config = config
        self.tokenizer = None
        self.dev_mode = is_dev_mode()

        if self.dev_mode:
            dev_mode_log_stub("GenerationService (LiteLLM)")
            logger.info(
                "GenerationService initialized in DEV MODE - stub implementation only."
            )
            return

        if not _tiktoken_available:
            logger.warning(
                "TikToken library not found. Token counting for context window management will be approximate. "
                "Install with 'pip install tiktoken'"
            )
        else:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info("TikToken tokenizer initialized ('cl100k_base').")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize TikToken tokenizer: {e}. Token counting will be approximate.",
                    exc_info=True,
                )
                self.tokenizer = None

        if self.config and self.config.enable:
            logger.info(
                f"GenerationService initialized. Provider: {self.config.provider}, Model: {self.config.model_name}"
            )
        elif self.config:
            logger.info(
                "GenerationService initialized, but RAG generation is disabled in config."
            )
        else:
            logger.warning(
                "GenerationService initialized WITHOUT configuration. RAG generation disabled."
            )

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text.split())

    def _truncate_context(self, context: str, max_tokens: int) -> str:
        if not context:
            return ""

        if not self.tokenizer:
            avg_chars_per_token = 4
            max_chars = max_tokens * avg_chars_per_token
            if len(context) > max_chars:
                logger.warning(
                    f"Approximated context length ({len(context)} chars) exceeds limit ({max_chars} chars). Truncating."
                )
                return context[:max_chars]
            return context

        tokens = self.tokenizer.encode(context)
        if len(tokens) > max_tokens:
            logger.warning(
                f"Context token count ({len(tokens)}) exceeds limit ({max_tokens}). Truncating."
            )
            truncated_tokens = tokens[:max_tokens]
            return str(self.tokenizer.decode(truncated_tokens))
        return context

    def _format_context(
        self, results: list[VectorSearchResultItem], max_tokens: int
    ) -> str:
        combined_text = "\n\n---\n\n".join([res.text for res in results if res.text])

        if not combined_text:
            logger.warning("No text content found in retrieved search results.")
            return ""

        truncated_context = self._truncate_context(combined_text, max_tokens)
        logger.debug(
            f"Formatted context length (tokens): {self._count_tokens(truncated_context)}"
        )
        return truncated_context

    def _construct_prompt(self, context: str, query: str) -> list[dict[str, str]]:
        if not self.config or not self.config.prompt_template:
            logger.error(
                "Cannot construct prompt: Generation config or prompt template missing."
            )
            raise GenerationError(
                "Generation configuration or prompt template is missing."
            )

        try:
            full_prompt_text = self.config.prompt_template.format(
                context=context, query=query
            )
            messages = [{"role": "user", "content": full_prompt_text}]
            logger.debug(f"Constructed prompt messages: {messages}")
            return messages
        except KeyError as e:
            logger.error(
                f"Failed to format prompt template. Missing key: {e}. Template: '{self.config.prompt_template}'"
            )
            raise GenerationError(f"Invalid prompt template format: Missing key {e}")

    async def generate_answer(
        self,
        query: str,
        top_k_retrieval: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> tuple[str, list[VectorSearchResultItem]]:
        if self.dev_mode:
            logger.info(
                f"DEV MODE: Generating stub answer for query: '{query[:50]}...'"
            )
            stub_answer = f"[DEV MODE STUB] This is a placeholder answer for the query: '{query}'. In production, this would be generated using LiteLLM and the configured model."
            return stub_answer, []

        if not self.config or not self.config.enable:
            logger.error(
                "Attempted to generate answer, but generation is disabled in configuration."
            )
            raise GenerationError("RAG generation is disabled in the configuration.")

        if not query or not query.strip():
            raise GenerationError("Query cannot be empty.")

        logger.info(f"Starting RAG generation for query: '{query[:50]}...'")

        try:
            logger.debug(
                f"Retrieving context with top_k={top_k_retrieval}, filters={filters}"
            )
            retrieved_results, search_took_ms = self.search_service.search_semantic(
                query=query, top_k=top_k_retrieval, filters=filters
            )
            logger.info(
                f"Retrieved {len(retrieved_results)} context documents in {search_took_ms:.2f} ms."
            )
            if not retrieved_results:
                logger.warning(
                    "No relevant context found for the query. Cannot generate answer."
                )
                return "Could not find relevant information to answer the question.", []

        except SemanticSearchError as e:
            logger.error(
                f"Semantic search failed during RAG context retrieval: {e}",
                exc_info=True,
            )
            raise GenerationError(f"Failed to retrieve context: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error during context retrieval: {e}", exc_info=True
            )
            raise GenerationError(
                f"An unexpected error occurred while retrieving context: {e}"
            ) from e

        try:
            formatted_context = self._format_context(
                retrieved_results, self.config.max_context_tokens
            )
        except Exception as e:
            logger.error(f"Failed to format context: {e}", exc_info=True)
            raise GenerationError(
                f"An unexpected error occurred while formatting context: {e}"
            ) from e

        try:
            messages = self._construct_prompt(formatted_context, query)
        except GenerationError:
            raise
        except Exception as e:
            logger.error(f"Failed to construct prompt: {e}", exc_info=True)
            raise GenerationError(
                f"An unexpected error occurred while constructing the prompt: {e}"
            ) from e

        llm_response = None
        try:
            logger.info(f"Calling LLM model '{self.config.model_name}'...")
            litellm = _get_litellm()
            llm_response = await litellm.acompletion(
                model=self.config.model_name,
                messages=messages,
                api_key=self.config.litellm_api_key,
                api_base=str(self.config.litellm_api_base)
                if self.config.litellm_api_base
                else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                timeout=self.config.generation_timeout,
            )
            logger.info(f"LLM call successful for model '{self.config.model_name}'.")

        except tuple(_get_litellm_exceptions().values()) as e:
            error_type = type(e).__name__
            logger.error(
                f"LiteLLM API error calling model '{self.config.model_name}': {error_type}: {e}",
                exc_info=True,
            )
            raise GenerationError(f"LLM API error ({error_type}): {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error calling LLM model '{self.config.model_name}': {e}",
                exc_info=True,
            )
            raise GenerationError(
                f"An unexpected error occurred during LLM call: {e}"
            ) from e

        try:
            if llm_response and llm_response.choices:
                answer = llm_response.choices[0].message.content
                if answer:
                    logger.info("Successfully extracted answer from LLM response.")
                    return answer.strip(), retrieved_results
                else:
                    logger.warning("LLM response choice content was empty.")
                    raise GenerationError("LLM returned an empty answer.")
            else:
                logger.error(f"Unexpected LLM response structure: {llm_response}")
                raise GenerationError("Invalid or empty response received from LLM.")
        except (AttributeError, IndexError, TypeError) as e:
            logger.error(
                f"Failed to extract answer from LLM response: {e}. Response: {llm_response}",
                exc_info=True,
            )
            raise GenerationError(f"Could not parse answer from LLM response: {e}")

        raise GenerationError(
            "Generation process completed without returning an answer."
        )
