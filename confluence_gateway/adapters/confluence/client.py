import random
import time
from functools import wraps
from typing import Any, Optional, TypeVar, Union

import requests
from atlassian import Confluence
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

T = TypeVar("T")


def with_backoff(max_retries=5, initial_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay

            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except ConfluenceAPIError as e:
                    if getattr(e, "status_code", None) != 429:
                        raise

                    if retries == max_retries:
                        raise

                    jitter = random.uniform(0, 0.3) * delay
                    sleep_time = delay + jitter
                    time.sleep(sleep_time)

                    retries += 1
                    delay *= 2

            return func(*args, **kwargs)

        return wrapper

    return decorator


class ConfluenceClient:
    API_PATHS = ["wiki/rest/api", "rest/api"]

    config: ConfluenceConfig
    _working_api_path: Optional[str] = None

    def __init__(self, config: Optional[ConfluenceConfig] = None):
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
        params: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
        model_class: Optional[type[T]] = None,
        api_version: Optional[str] = None,
        use_transformer: bool = True,
    ) -> Union[dict[str, Any], T, None]:
        response = None
        if self._working_api_path and api_version is None:
            api_version = self._working_api_path
        elif api_version is not None:
            pass
        else:
            last_exception = None

            for api_path in self.API_PATHS:
                try:
                    url = f"{self.base_url}/{api_path}/{endpoint.lstrip('/')}"
                    response = getattr(self.session, method.lower())(
                        url, params=params, json=data, timeout=self.config.timeout
                    )

                    if response.status_code == 401:
                        raise ConfluenceAuthenticationError(
                            "Authentication failed. Check username and API token."
                        )

                    if response.status_code != 404:
                        response.raise_for_status()
                        self._working_api_path = api_path
                        break
                except requests.exceptions.RequestException as e:
                    if not (
                        isinstance(e, requests.exceptions.HTTPError)
                        and hasattr(e, "response")
                        and e.response.status_code == 404
                    ):
                        raise
                    last_exception = e
            else:
                if last_exception:
                    if isinstance(last_exception, requests.exceptions.HTTPError):
                        error_message = str(last_exception)
                        try:
                            error_data = last_exception.response.json()
                            if "message" in error_data:
                                error_message = error_data["message"]
                        except (ValueError, AttributeError):
                            pass
                        raise ConfluenceAPIError(
                            status_code=last_exception.response.status_code,
                            error_message=error_message,
                        ) from last_exception
                    raise ConfluenceConnectionError(cause=last_exception)

                raise ConfluenceAPIError(
                    status_code=404, error_message="Failed to find working API endpoint"
                )

        if response is None:
            url = f"{self.base_url}/{api_version}/{endpoint.lstrip('/')}"
            try:
                response = getattr(self.session, method.lower())(
                    url, params=params, json=data, timeout=self.config.timeout
                )

                if response.status_code == 401:
                    raise ConfluenceAuthenticationError(
                        "Authentication failed. Check username and API token."
                    )

                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                if isinstance(e, requests.exceptions.HTTPError) and hasattr(
                    e, "response"
                ):
                    error_message = str(e)
                    try:
                        error_data = e.response.json()
                        if "message" in error_data:
                            error_message = error_data["message"]
                    except (ValueError, AttributeError):
                        pass

                    raise ConfluenceAPIError(
                        status_code=e.response.status_code, error_message=error_message
                    ) from e
                raise ConfluenceConnectionError(cause=e)

        try:
            if response.status_code == 204:
                return None

            response_data = response.json()

            if model_class:
                if use_transformer:
                    transformer_method = None
                    if model_class == ConfluenceSpace:
                        transformer_method = self._parse_space
                    elif model_class == ConfluencePage:
                        transformer_method = self._parse_page
                    elif model_class == SearchResult:
                        transformer_method = self._parse_search_result

                    if transformer_method:
                        return transformer_method(response_data)

                return model_class(**response_data)

            return response_data

        except requests.exceptions.RequestException as e:
            if (
                isinstance(e, requests.exceptions.HTTPError)
                and hasattr(e, "response")
                and e.response.status_code != 401
            ):
                error_message = str(e)
                try:
                    error_data = e.response.json()
                    if "message" in error_data:
                        error_message = error_data["message"]
                except (ValueError, AttributeError):
                    pass

                raise ConfluenceAPIError(
                    status_code=e.response.status_code, error_message=error_message
                ) from e
            raise ConfluenceConnectionError(cause=e)

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

    def get_page(
        self, page_id: str, expand: Optional[list[str]] = None
    ) -> ConfluencePage:
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
            if "404" in str(e):
                raise ConfluenceAPIError(
                    status_code=404, error_message=f"Page {page_id} not found"
                ) from e
            raise ConfluenceAPIError(error_message=str(e)) from e

    def _parse_attachment(self, data: dict[str, Any]) -> ConfluenceAttachment:
        # Basic parsing, model handles more complex logic
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
        if "totalSize" in data:
            data["total_size"] = data["totalSize"]
        elif "total" in data:
            data["total_size"] = data["total"]
        elif "size" in data and "total_size" not in data:
            data["total_size"] = data["size"]
        transformed_results = []
        if "results" in data and isinstance(data["results"], list):
            for item in data["results"]:
                if isinstance(item, dict):
                    try:
                        if "content" in item and isinstance(item["content"], dict):
                            page_data = item["content"].copy()
                            for key, value in item.items():
                                if key != "content" and key not in page_data:
                                    page_data[key] = value
                            page = self._parse_page(page_data)
                        else:
                            page = self._parse_page(item)

                        transformed_results.append(page)
                    except Exception:
                        pass
                elif isinstance(item, ConfluencePage):
                    transformed_results.append(item)

            data["results"] = transformed_results

        return SearchResult(**data)

    def extract_content_fields(
        self, content: Union[ConfluencePage, ConfluenceAttachment]
    ) -> dict[str, Any]:
        result = {
            "id": content.id,
            "title": content.title,
            "type": getattr(content.content_type, "value", content.content_type),
            "created_at": content.created_at,
            "updated_at": content.updated_at,
        }

        space = getattr(content, "space", None)
        if space:
            if isinstance(space, dict):
                result["space_key"] = space.get("key", "")
                result["space_name"] = space.get("name", "")
            else:
                result["space_key"] = getattr(space, "key", "")
                result["space_name"] = getattr(space, "name", "")
        else:
            result["space_key"] = getattr(content, "space_key", "")
            result["space_name"] = getattr(content, "space_name", "")

        content_type = getattr(content.content_type, "value", content.content_type)

        if content_type in (ContentType.PAGE.value, ContentType.BLOGPOST.value):
            if isinstance(content, ConfluencePage):
                result["html_content"] = getattr(content, "html_content", None)
                result["plain_content"] = getattr(content, "plain_content", None)

                version = getattr(content, "version", None)
                if version:
                    result["version"] = int(getattr(version, "number", 0))

                # Construct URL robustly
                space_key = result.get("space_key")
                if space_key and content.id:
                    result["url"] = (
                        f"{self.base_url}/wiki/spaces/{space_key}/pages/{content.id}"
                    )
                elif hasattr(content, "_links") and getattr(
                    content._links, "webui", None
                ):
                    result["url"] = f"{self.base_url}{content._links.webui}"

        elif content_type == ContentType.ATTACHMENT.value:
            if isinstance(content, ConfluenceAttachment):
                result["file_name"] = content.title
                result["file_size"] = content.file_size
                result["media_type"] = content.media_type
                result["download_url"] = content.download_url
                if not result["download_url"] and content.id:
                    # Fallback construction if link missing
                    result["download_url"] = (
                        f"{self.base_url}/wiki/download/attachments/{content.id}"
                    )
                # Ensure _links exists and has a webui attribute before accessing it
                if content._links and content._links.webui:
                    result["url"] = f"{self.base_url}{content._links.webui}"

        elif content_type == ContentType.COMMENT.value:
            if isinstance(
                content, ConfluencePage
            ):  # Comments might be parsed as Pages initially
                result["plain_content"] = getattr(content, "plain_content", None)
            container = getattr(content, "container", None)
            if container:
                result["parent_id"] = (
                    container.get("id", "")
                    if isinstance(container, dict)
                    else getattr(container, "id", "")
                )

        return result

    def _build_search_cql(
        self,
        query: str,
        content_type: Optional[Union[ContentType, str]] = None,
        space_key: Optional[str] = None,
        include_archived: bool = False,
    ) -> str:
        if hasattr(self.atlassian_api, "cql_builder"):
            try:
                cql = self.atlassian_api.cql_builder()
                cql.text_contains(query)

                if content_type:
                    content_type_str = getattr(content_type, "value", content_type)
                    cql.content_type(content_type_str)

                if space_key:
                    cql.space(space_key)

                if not include_archived:
                    # Note: atlassian-python-api doesn't directly support 'archived' status in cql_builder easily
                    # We might need to append it manually or rely on the include_archived_spaces flag later
                    pass

                return str(cql)
            except (AttributeError, TypeError):
                pass

        query_escaped = self._escape_cql(query)
        cql_parts = [f'text ~ "{query_escaped}"']

        if content_type:
            content_type_str = getattr(content_type, "value", content_type)
            cql_parts.append(f'type = "{content_type_str}"')

        if space_key:
            space_key_escaped = self._escape_cql(space_key)
            cql_parts.append(f'space = "{space_key_escaped}"')

        # Add archived filter manually if needed and not handled by library flags
        # if not include_archived:
        #     cql_parts.append('label != "archived"') # Example, actual label might differ

        return " AND ".join(cql_parts)

    @with_backoff()
    def search_by_cql(
        self,
        cql: str,
        limit: Optional[int] = None,
        start: Optional[int] = 0,
        expand: Optional[list[str]] = None,
        get_all_results: bool = False,
        max_results: Optional[int] = None,
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
        """Lists attachments for a given page ID."""
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
                    except Exception as parse_err:
                        # Log parsing error but continue
                        print(f"Warning: Failed to parse attachment data: {parse_err}")

                if len(results) < limit or "next" not in response.get("_links", {}):
                    break  # No more pages

                current_start += limit

            except Exception as e:
                if "401" in str(e):
                    raise ConfluenceAuthenticationError("Authentication failed.") from e
                if isinstance(e, requests.exceptions.RequestException):
                    raise ConfluenceConnectionError(cause=e)
                if "404" in str(e):  # Page not found
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
        """Downloads the content of an attachment by its ID."""
        try:
            # First, get attachment details to find the download URL
            # Using a direct API call as atlassian-python-api download saves to file
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

            # Use the session to download the content
            response = self.session.get(
                download_url, timeout=self.config.timeout * 2
            )  # Longer timeout for downloads
            response.raise_for_status()  # Raise HTTP errors
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
            # Catch potential parsing errors or other issues
            raise ConfluenceAPIError(
                error_message=f"Failed to download attachment {attachment_id}: {e}"
            ) from e

    @with_backoff()
    def search(
        self,
        query: str,
        content_type: Optional[Union[ContentType, str]] = None,
        space_key: Optional[str] = None,
        include_archived: bool = False,
        limit: Optional[int] = None,
        start: Optional[int] = 0,
        expand: Optional[list[str]] = None,
        get_all_results: bool = False,
        max_results: Optional[int] = None,
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

                return self.search_by_cql(
                    cql=cql_query,
                    limit=limit,
                    start=start,
                    expand=expand,
                    get_all_results=get_all_results,
                    max_results=max_results,
                    include_archived=include_archived,
                )
        except Exception as e:
            if "401" in str(e):
                raise ConfluenceAuthenticationError(
                    "Authentication failed. Check username and API token."
                ) from e
            if isinstance(e, requests.exceptions.RequestException):
                raise ConfluenceConnectionError(cause=e)
            raise ConfluenceAPIError(error_message=str(e)) from e
