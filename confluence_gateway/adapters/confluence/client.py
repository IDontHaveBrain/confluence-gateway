import logging
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import requests
from atlassian import Confluence
from pydantic import ValidationError
from requests.auth import HTTPBasicAuth

from confluence_gateway.adapters.confluence.models import (
    ConfluenceAttachment,
    ConfluencePage,
    ConfluenceSpace,
    ContentType,
    SearchResult,
)
from confluence_gateway.core.config import (
    ConfluenceConfig,
    confluence_config,
    search_config,
)
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
    SearchParameterError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_backoff(
    max_retries: int = 5,
    initial_delay: float = 1,
    backoff_factor: float = 2,
    jitter_factor: float = 0.3,
) -> Any:
    def decorator(func: Any) -> Any:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            delay = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except ConfluenceAPIError as e:
                    if getattr(e, "status_code", None) != 429 or retries >= max_retries:
                        raise

                    jitter = random.uniform(0, jitter_factor) * delay
                    time.sleep(delay + jitter)

                    retries += 1
                    delay *= backoff_factor

        return wrapper

    return decorator


class ConfluenceClient:
    API_PATHS = ["wiki/rest/api", "rest/api"]

    config: ConfluenceConfig
    _working_api_path: str | None = None

    def __init__(self, config: ConfluenceConfig | None = None):
        config_to_use = config or confluence_config
        if not config_to_use:
            raise ValueError("Confluence configuration is missing")
        self.config = config_to_use
        self.base_url = str(self.config.url).rstrip("/")

        self._working_api_path = None

        self.atlassian_api = Confluence(
            url=self.base_url,
            username=self.config.username,
            password=self.config.api_token,
            timeout=self.config.timeout,
        )
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.auth = HTTPBasicAuth(self.config.username, self.config.api_token)
        session.headers.update(self._get_auth_headers())
        return session

    def _get_auth_headers(self) -> dict[str, str]:
        return {"Accept": "application/json", "Content-Type": "application/json"}

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        model_class: type[T] | None = None,
        api_version: str | None = None,
        use_transformer: bool = True,
    ) -> dict[str, Any] | T | None:
        api_path_to_use = api_version or self._working_api_path

        if not api_path_to_use:
            api_path_to_use = self._discover_working_api_path(
                endpoint, params, data, method
            )

        if not api_path_to_use:
            raise ConfluenceAPIError(
                status_code=404, error_message="Failed to find working API endpoint"
            )

        url = f"{self.base_url}/{api_path_to_use}/{endpoint.lstrip('/')}"
        response = self._execute_request(method, url, params, data)

        if response.status_code == 204:
            return None

        response_data: dict[str, Any] = response.json()

        if model_class:
            if use_transformer:
                transformer_method = self._get_transformer_for_model(model_class)
                if transformer_method:
                    return transformer_method(response_data)

            return model_class(**response_data)

        return response_data

    def _discover_working_api_path(
        self,
        endpoint: str,
        params: dict[str, Any] | None,
        data: dict[str, Any] | None,
        method: str,
    ) -> str | None:
        last_exception: requests.exceptions.RequestException | None = None

        for api_path in self.API_PATHS:
            try:
                url = f"{self.base_url}/{api_path}/{endpoint.lstrip('/')}"
                self._execute_request(method, url, params, data)

                self._working_api_path = api_path
                return api_path

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    last_exception = e
                    continue
                raise
            except requests.exceptions.RequestException as e:
                last_exception = e
                raise ConfluenceConnectionError(cause=e)

        if last_exception:
            if isinstance(last_exception, requests.exceptions.HTTPError):
                raise self._create_api_error_from_http_error(last_exception)
            raise ConfluenceConnectionError(cause=last_exception)

        return None

    def _execute_request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        data: dict[str, Any] | None,
    ) -> requests.Response:
        try:
            response: requests.Response = getattr(self.session, method.lower())(
                url, params=params, json=data, timeout=self.config.timeout
            )

            if response.status_code == 401:
                raise ConfluenceAuthenticationError(
                    "Authentication failed. Check username and API token."
                )

            if response.status_code != 404:
                response.raise_for_status()

            return response

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise
            raise self._create_api_error_from_http_error(e)
        except requests.exceptions.RequestException as e:
            raise ConfluenceConnectionError(cause=e)

    def _get_transformer_for_model(
        self, model_class: type[T]
    ) -> Callable[[dict[str, Any]], T] | None:
        if model_class == ConfluenceSpace:
            return self._parse_space
        elif model_class == ConfluencePage:
            return self._parse_page
        elif model_class == SearchResult:
            return self._parse_search_result
        return None

    def _create_api_error_from_http_error(
        self, http_error: requests.exceptions.HTTPError
    ) -> ConfluenceAPIError:
        error_message = str(http_error)
        try:
            error_data = http_error.response.json()
            if "message" in error_data:
                error_message = error_data["message"]
        except (ValueError, AttributeError):
            pass

        return ConfluenceAPIError(
            status_code=http_error.response.status_code, error_message=error_message
        )

    def test_connection(self) -> bool:
        try:
            spaces = self.atlassian_api.get_all_spaces(limit=1)
            if not isinstance(spaces, dict) or "results" not in spaces:
                raise ConfluenceAPIError(
                    error_message="Unexpected response format from Confluence API"
                )
            return True
        except Exception as e:
            if "401" in str(e):
                raise ConfluenceAuthenticationError(
                    "Authentication failed. Check username and API token."
                ) from e
            if isinstance(e, requests.exceptions.RequestException):
                raise ConfluenceConnectionError(cause=e)
            raise ConfluenceAPIError(error_message=str(e)) from e

    @with_backoff()
    def list_all_spaces(self, limit: int = 50) -> list[ConfluenceSpace]:
        all_spaces = []
        start = 0
        while True:
            try:
                logger.debug(f"Fetching spaces, start={start}, limit={limit}")
                spaces_data = self.atlassian_api.get_all_spaces(
                    start=start, limit=limit, expand="description.plain"
                )

                if not spaces_data or "results" not in spaces_data:
                    logger.debug("No more spaces found or unexpected response format.")
                    break

                results = spaces_data.get("results", [])
                for space_dict in results:
                    try:
                        space = self._parse_space(space_dict)
                        all_spaces.append(space)
                    except (ValidationError, TypeError) as parse_err:
                        logger.warning(
                            f"Failed to parse space data: {space_dict.get('key', 'N/A')}. Error: {parse_err}"
                        )

                if "next" not in spaces_data.get("_links", {}):
                    logger.debug("No 'next' link found, assuming all spaces fetched.")
                    break

                # Simple increment based on limit, assuming API behaves predictably
                start += limit
                # Add a small delay to avoid hitting rate limits aggressively
                time.sleep(0.1)

            except Exception as e:
                if "401" in str(e):
                    raise ConfluenceAuthenticationError(
                        "Authentication failed. Check username and API token."
                    ) from e
                if isinstance(e, requests.exceptions.RequestException):
                    raise ConfluenceConnectionError(cause=e)
                logger.error(
                    f"Error fetching spaces at start={start}: {e}", exc_info=True
                )
                # Decide whether to break or raise depending on desired robustness
                raise ConfluenceAPIError(error_message=str(e)) from e

        logger.info(f"Successfully fetched {len(all_spaces)} spaces in total.")
        return all_spaces

    def get_space(self, space_key: str) -> ConfluenceSpace:
        try:
            space_data = self.atlassian_api.get_space(
                space_key, expand="description.plain,metadata.labels"
            )

            return self._parse_space(space_data)
        except Exception as e:
            if "401" in str(e):
                raise ConfluenceAuthenticationError(
                    "Authentication failed. Check username and API token."
                ) from e
            if isinstance(e, requests.exceptions.RequestException):
                raise ConfluenceConnectionError(cause=e)
            if "404" in str(e):
                raise ConfluenceAPIError(
                    status_code=404, error_message=f"Space {space_key} not found"
                ) from e
            raise ConfluenceAPIError(error_message=str(e)) from e

    def get_page(self, page_id: str, expand: list[str] | None = None) -> ConfluencePage:
        expand_str = "body.view,body.storage,space,version,metadata,children.page,children.attachment,history,ancestors"
        if expand:
            expand_str = ",".join(expand)

        try:
            page_data = self.atlassian_api.get_page_by_id(page_id, expand=expand_str)

            return self._parse_page(page_data)
        except Exception as e:
            if "401" in str(e):
                raise ConfluenceAuthenticationError(
                    "Authentication failed. Check username and API token."
                ) from e
            if isinstance(e, requests.exceptions.RequestException):
                raise ConfluenceConnectionError(cause=e)
            error_str_lower = str(e).lower()
            is_not_found_error = (
                "404" in error_str_lower
                or "no content with the given id" in error_str_lower
                or "page not found" in error_str_lower
                or "could not be found" in error_str_lower
            )
            if is_not_found_error:
                raise ConfluenceAPIError(
                    status_code=404,
                    error_message=f"Page {page_id} not found or permission denied. (Original error: {e})",
                ) from e
            raise ConfluenceAPIError(error_message=str(e)) from e

    def _parse_attachment(self, data: dict[str, Any]) -> ConfluenceAttachment:
        return ConfluenceAttachment(**data)

    def _escape_cql(self, value: str) -> str:
        return value.replace('"', '\\"')

    def _parse_space(self, data: dict[str, Any]) -> ConfluenceSpace:
        if "id" in data and not isinstance(data["id"], str):
            data["id"] = str(data["id"])

        if "name" in data and "title" not in data:
            data["title"] = data["name"]

        if "description" in data and isinstance(data["description"], dict):
            if "plain" in data["description"] and isinstance(
                data["description"]["plain"], dict
            ):
                plain_desc = data["description"]["plain"]
                if "value" in plain_desc:
                    data["description_text"] = plain_desc["value"]

        return ConfluenceSpace(**data)

    def _parse_page(self, data: dict[str, Any]) -> ConfluencePage:
        if "body" in data and isinstance(data["body"], dict):
            body_data = data["body"]

            if "view" in body_data and isinstance(body_data["view"], dict):
                data["html_content"] = body_data["view"].get("value", "")

            if "storage" in body_data and isinstance(body_data["storage"], dict):
                data["storage_content"] = body_data["storage"].get("value", "")

            if "plain" in body_data and isinstance(body_data["plain"], dict):
                data["plain_content"] = body_data["plain"].get("value", "")

        if "version" in data and isinstance(data["version"], dict):
            data["version_number"] = data["version"].get("number", 0)

        if "space" in data and isinstance(data["space"], dict):
            space_data = data["space"]
            data["space_key"] = space_data.get("key", "")
            data["space_name"] = space_data.get("name", "")

        return ConfluencePage(**data)

    def _parse_search_result(self, data: dict[str, Any]) -> SearchResult:
        for total_field in ["totalSize", "total", "size"]:
            if total_field in data and "total_size" not in data:
                data["total_size"] = data[total_field]
                break

        transformed_results = []

        if "results" in data and isinstance(data["results"], list):
            for item in data["results"]:
                try:
                    if isinstance(item, dict):
                        if "content" in item and isinstance(item["content"], dict):
                            page_data = item["content"].copy()
                            page_data.update(
                                {
                                    k: v
                                    for k, v in item.items()
                                    if k != "content" and k not in page_data
                                }
                            )
                            page = self._parse_page(page_data)
                        else:
                            page = self._parse_page(item)

                        transformed_results.append(page)
                    elif isinstance(item, ConfluencePage):
                        transformed_results.append(item)
                except Exception:
                    pass

            data["results"] = transformed_results

        return SearchResult(**data)

    def extract_content_fields(
        self, content: ConfluencePage | ConfluenceAttachment
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": content.id,
            "title": content.title,
            "type": content.content_type.value,
            "created_at": content.created_at,
            "updated_at": content.updated_at,
            "space_key": None,
            "space_name": None,
            "url": None,
            "version": None,
            "html_content": None,
            "plain_content": None,
            "parent_id": None,
            "file_name": None,
            "file_size": None,
            "media_type": None,
            "download_url": None,
        }

        # Extract space info consistently, ensuring strings
        space_obj = getattr(content, "space", None)
        if isinstance(space_obj, ConfluenceSpace):
            result["space_key"] = space_obj.key or ""
            result["space_name"] = (space_obj.name or space_obj.title) or ""
        elif isinstance(space_obj, dict):
            result["space_key"] = space_obj.get("key") or ""
            result["space_name"] = space_obj.get("name") or ""
        else:
            # Ensure empty strings if space is None or unexpected type
            result["space_key"] = ""
            result["space_name"] = ""

        # Extract type-specific fields using isinstance
        if isinstance(content, ConfluencePage):
            result["html_content"] = content.html_content
            result["plain_content"] = content.plain_content
            if content.version:
                result["version"] = content.version.number

            # Construct URL for pages/blogposts
            if result["space_key"] and content.id:
                result["url"] = (
                    f"{self.base_url}/wiki/spaces/{result['space_key']}/pages/{content.id}"
                )
            elif hasattr(content, "_links") and content._links and content._links.webui:
                result["url"] = f"{self.base_url}{content._links.webui}"

            # Handle comments specifically (they are often modeled as ConfluencePage by the lib)
            if content.content_type == ContentType.COMMENT:
                container = getattr(content, "container", None)
                if isinstance(container, dict):
                    result["parent_id"] = container.get("id")
                # Use getattr for safer access if container is not a dict but might have id
                elif container is not None:
                    result["parent_id"] = getattr(container, "id", None)

        elif isinstance(content, ConfluenceAttachment):  # type: ignore[unreachable]
            result["file_name"] = content.title
            result["file_size"] = content.file_size
            result["media_type"] = content.media_type
            result["download_url"] = content.download_url

            # Fallback download URL construction
            if not result["download_url"] and content.id:
                # This path might vary, relying on _links is safer
                # result["download_url"] = f"{self.base_url}/wiki/download/attachments/{content.id}"
                pass  # Prefer relying on the _links.download if available

            # Construct URL for attachments
            if content._links and content._links.webui:
                result["url"] = f"{self.base_url}{content._links.webui}"

        # Return only keys that have non-None values if needed,
        # but returning the full structure might be more consistent.
        # return {k: v for k, v in result.items() if v is not None}
        return result

    def _build_search_cql(
        self,
        query: str,
        content_type: ContentType | str | None = None,
        space_key: str | None = None,
        include_archived: bool = False,
    ) -> str:
        # Consistently use manual CQL building for robustness
        query_escaped = self._escape_cql(query)
        cql_parts = [f'text ~ "{query_escaped}"']

        if content_type:
            content_type_str = getattr(content_type, "value", content_type)
            cql_parts.append(f'type = "{content_type_str}"')

        if space_key:
            space_key_escaped = self._escape_cql(space_key)
            cql_parts.append(f'space = "{space_key_escaped}"')

        return " AND ".join(cql_parts)

    @with_backoff()
    def search_by_cql(
        self,
        cql: str,
        limit: int | None = None,
        start: int | None = 0,
        expand: list[str] | None = None,
        get_all_results: bool = False,
        max_results: int | None = None,
        include_archived: bool = False,
    ) -> SearchResult:
        if not cql:
            raise SearchParameterError("CQL query cannot be empty")

        actual_limit = limit or search_config.default_limit

        expand_param = None
        if expand:
            expand_param = ",".join(expand)
        elif search_config.default_expand:
            expand_param = ",".join(search_config.default_expand)

        try:
            if get_all_results:
                all_results = self.atlassian_api.cql(
                    cql,
                    limit=actual_limit,
                    start=start or 0,
                    expand=expand_param,
                    include_archived_spaces=include_archived,
                )

                if max_results and len(all_results.get("results", [])) > max_results:
                    all_results["results"] = all_results["results"][:max_results]

                return self._parse_search_result(all_results)
            else:
                results = self.atlassian_api.cql(
                    cql,
                    limit=actual_limit,
                    start=start or 0,
                    expand=expand_param,
                    include_archived_spaces=include_archived,
                )

                return self._parse_search_result(results)
        except Exception as e:
            if "401" in str(e):
                raise ConfluenceAuthenticationError(
                    "Authentication failed. Check username and API token."
                ) from e
            if isinstance(e, requests.exceptions.RequestException):
                raise ConfluenceConnectionError(cause=e)
            raise ConfluenceAPIError(error_message=str(e)) from e

    @with_backoff()
    def list_attachments(
        self, page_id: str, limit: int = 50, start: int = 0
    ) -> list[ConfluenceAttachment]:
        attachments = []
        current_start = start
        while True:
            try:
                response = self.atlassian_api.get_attachments_from_content(
                    page_id=page_id, limit=limit, start=current_start
                )
                if not response or "results" not in response:
                    break

                results = response.get("results", [])
                for attach_data in results:
                    try:
                        attachments.append(self._parse_attachment(attach_data))
                    except (ValidationError, TypeError) as parse_err:
                        attach_id = attach_data.get("id", "UNKNOWN")
                        logger.warning(
                            f"Failed to parse attachment data for ID {attach_id}: {parse_err}"
                        )

                if len(results) < limit or "next" not in response.get("_links", {}):
                    break

                current_start += limit

            except Exception as e:
                if "401" in str(e):
                    raise ConfluenceAuthenticationError("Authentication failed.") from e
                if isinstance(e, requests.exceptions.RequestException):
                    raise ConfluenceConnectionError(cause=e)
                if "404" in str(e):
                    raise ConfluenceAPIError(
                        status_code=404,
                        error_message=f"Page {page_id} not found when listing attachments",
                    ) from e
                raise ConfluenceAPIError(
                    error_message=f"Failed to list attachments for page {page_id}: {e}"
                ) from e
        return attachments

    @with_backoff()
    def download_attachment(self, attachment_id: str) -> bytes:
        try:
            endpoint = f"content/attachment/{attachment_id}"
            attach_data = self._make_request("GET", endpoint, api_version="rest/api")

            if not attach_data or not isinstance(attach_data, dict):
                raise ConfluenceAPIError(
                    error_message=f"Could not retrieve metadata for attachment {attachment_id}"
                )

            download_path = attach_data.get("_links", {}).get("download")
            if not download_path:
                raise ConfluenceAPIError(
                    error_message=f"Download URL not found for attachment {attachment_id}"
                )

            download_url = f"{self.base_url}{download_path}"

            response = self.session.get(download_url, timeout=self.config.timeout * 2)
            response.raise_for_status()
            return response.content

        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError) and hasattr(e, "response"):
                if e.response.status_code == 401:
                    raise ConfluenceAuthenticationError("Authentication failed.") from e
                if e.response.status_code == 404:
                    raise ConfluenceAPIError(
                        status_code=404,
                        error_message=f"Attachment {attachment_id} not found",
                    ) from e
                raise ConfluenceAPIError(
                    status_code=e.response.status_code, error_message=str(e)
                ) from e
            raise ConfluenceConnectionError(cause=e)
        except Exception as e:
            raise ConfluenceAPIError(
                error_message=f"Failed to download attachment {attachment_id}: {e}"
            ) from e

    @with_backoff()
    def search(
        self,
        query: str,
        content_type: ContentType | str | None = None,
        space_key: str | None = None,
        include_archived: bool = False,
        limit: int | None = None,
        start: int | None = 0,
        expand: list[str] | None = None,
        get_all_results: bool = False,
        max_results: int | None = None,
    ) -> SearchResult:
        if not query:
            raise SearchParameterError("Query cannot be empty")

        if content_type and isinstance(content_type, str):
            try:
                content_type = ContentType(content_type.lower())
            except ValueError:
                pass

        try:
            if not content_type and not space_key and not get_all_results:
                actual_limit = limit or search_config.default_limit
                expand_param = None
                if expand:
                    expand_param = ",".join(expand)
                elif search_config.default_expand:
                    expand_param = ",".join(search_config.default_expand)

                cql_query = f'text ~ "{self._escape_cql(query)}"'

                results = self.atlassian_api.cql(
                    cql_query,
                    start=start or 0,
                    limit=actual_limit,
                    expand=expand_param,
                    include_archived_spaces=include_archived,
                )
                return self._parse_search_result(results)
            else:
                cql_query = self._build_search_cql(
                    query, content_type, space_key, include_archived
                )

                result: SearchResult = self.search_by_cql(
                    cql=cql_query,
                    limit=limit,
                    start=start,
                    expand=expand,
                    get_all_results=get_all_results,
                    max_results=max_results,
                    include_archived=include_archived,
                )
                return result
        except Exception as e:
            if "401" in str(e):
                raise ConfluenceAuthenticationError(
                    "Authentication failed. Check username and API token."
                ) from e
            if isinstance(e, requests.exceptions.RequestException):
                raise ConfluenceConnectionError(cause=e)
            raise ConfluenceAPIError(error_message=str(e)) from e
