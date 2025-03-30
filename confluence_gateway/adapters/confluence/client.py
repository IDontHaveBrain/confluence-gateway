import random
import time
from functools import wraps
from typing import Any, Optional, TypeVar, Union

import requests
from atlassian import Confluence
from requests.auth import HTTPBasicAuth

from confluence_gateway.adapters.confluence.models import (
    ConfluencePage,
    ConfluenceSpace,
    ContentType,
    SearchResult,
)
from confluence_gateway.core.config import confluence_config, search_config
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
    SearchParameterError,
)

T = TypeVar("T")


def with_backoff(max_retries=5, initial_delay=1):
    """
    Decorator to handle rate limiting with exponential backoff.

    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds

    Returns:
        Decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay

            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except ConfluenceAPIError as e:
                    # If the error is not due to rate limiting, re-raise
                    if getattr(e, "status_code", None) != 429:
                        raise

                    # If we've hit max retries, re-raise
                    if retries == max_retries:
                        raise

                    # Exponential backoff with jitter
                    jitter = random.uniform(0, 0.3) * delay
                    sleep_time = delay + jitter
                    time.sleep(sleep_time)

                    # Increment retries and delay
                    retries += 1
                    delay *= 2

            # This should not be reached due to the max_retries check above
            return func(*args, **kwargs)

        return wrapper

    return decorator


class ConfluenceClient:
    """Client for interacting with Confluence API."""

    API_VERSION = "rest/api"

    def __init__(self, config=None):
        """
        Initialize the Confluence client.

        Args:
            config: Configuration object. If None, uses default from environment.
        """
        self.config = config or confluence_config
        self.base_url = str(self.config.url).rstrip("/")
        self.api_url = f"{self.base_url}/{self.API_VERSION}"

        # Initialize both the atlassian-python-api client and our custom session
        self.atlassian_api = Confluence(
            url=self.base_url,
            username=self.config.username,
            password=self.config.api_token,
            timeout=self.config.timeout,
        )
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """
        Create and configure an HTTP session for Confluence API calls.

        Returns:
            Configured requests.Session object
        """
        # For custom API calls that aren't covered by atlassian-python-api
        session = requests.Session()
        session.auth = HTTPBasicAuth(self.config.username, self.config.api_token)
        session.headers.update(self._get_auth_headers())
        return session

    def _get_auth_headers(self) -> dict[str, str]:
        """
        Get authentication headers for Confluence API calls.

        Returns:
            Dictionary of headers
        """
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
        """
        Make a request to the Confluence API with error handling.

        Args:
            method: HTTP method (get, post, etc.)
            endpoint: API endpoint (will be appended to api_url)
            params: Query parameters
            data: Request body data
            model_class: Optional class to instantiate with the response data
            api_version: API version to use, defaults to self.API_VERSION
            use_transformer: Whether to use model-specific transformer if available

        Returns:
            Response data as dict, as specified model instance, or None if no content

        Raises:
            ConfluenceConnectionError: If connection fails
            ConfluenceAuthenticationError: If authentication fails
            ConfluenceAPIError: If API returns an error
        """
        if api_version is None:
            api_version = self.API_VERSION

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

            # Handle 204 No Content responses
            if response.status_code == 204:
                return None

            response_data = response.json()

            if model_class:
                # Use specific transformer methods if available and requested
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

                # Fall back to direct model instantiation
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
                    # Try to extract more detailed error message from response
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
        """
        Test the connection to Confluence API.

        Returns:
            True if connection successful, raises exception otherwise

        Raises:
            ConfluenceConnectionError: If connection fails
            ConfluenceAuthenticationError: If authentication fails
            ConfluenceAPIError: If API returns an error
        """
        try:
            # Use a lightweight operation to test connection
            self.atlassian_api.get_all_spaces(limit=1)
            return True
        except Exception as e:
            # Translate exceptions to our custom exception types
            if "401" in str(e):
                raise ConfluenceAuthenticationError(
                    "Authentication failed. Check username and API token."
                ) from e
            if isinstance(e, requests.exceptions.RequestException):
                raise ConfluenceConnectionError(cause=e)
            raise ConfluenceAPIError(error_message=str(e)) from e

    def get_space(self, space_key: str) -> ConfluenceSpace:
        """
        Get details about a specific Confluence space.

        Args:
            space_key: The key of the space to retrieve

        Returns:
            ConfluenceSpace object

        Raises:
            ConfluenceConnectionError: If connection fails
            ConfluenceAuthenticationError: If authentication fails
            ConfluenceAPIError: If API returns an error
        """
        try:
            # Use atlassian-python-api for getting space details
            # The get_space method already handles all the logic we need
            space_data = self.atlassian_api.get_space(
                space_key, expand="description.plain,metadata.labels"
            )

            # Transform the response to our model format
            return self._parse_space(space_data)
        except Exception as e:
            # Translate exceptions to our custom exception types
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
        """
        Get details about a specific Confluence page or other content.

        Args:
            page_id: The ID of the page to retrieve
            expand: List of fields to expand in the response

        Returns:
            ConfluencePage object

        Raises:
            ConfluenceConnectionError: If connection fails
            ConfluenceAuthenticationError: If authentication fails
            ConfluenceAPIError: If API returns an error
        """
        # Default expand fields if none provided
        expand_str = "body.view,body.storage,space,version,metadata,children.page,children.attachment,history,ancestors"
        if expand:
            expand_str = ",".join(expand)

        try:
            # Use atlassian-python-api to get page details
            # The get_page_by_id method fetches full page content with all requested expansions
            page_data = self.atlassian_api.get_page_by_id(page_id, expand=expand_str)

            # Transform the response to our model format
            return self._parse_page(page_data)
        except Exception as e:
            # Translate exceptions to our custom exception types
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

    def _escape_cql(self, value: str) -> str:
        """
        Escape special characters in a string for CQL.

        Args:
            value: The string to escape

        Returns:
            Escaped string
        """
        # Replace double quotes with escaped double quotes
        return value.replace('"', '\\"')

    def _parse_space(self, data: dict[str, Any]) -> ConfluenceSpace:
        """
        Transform Confluence API space response into a ConfluenceSpace object.

        Args:
            data: Raw API response data

        Returns:
            ConfluenceSpace object
        """
        # Handle description structure transformation
        if "description" in data and isinstance(data["description"], dict):
            if "plain" in data["description"] and isinstance(
                data["description"]["plain"], dict
            ):
                plain_desc = data["description"]["plain"]
                if "value" in plain_desc:
                    # Store the plain text value directly for easier access
                    data["description_text"] = plain_desc["value"]

        return ConfluenceSpace(**data)

    def _parse_page(self, data: dict[str, Any]) -> ConfluencePage:
        """
        Transform Confluence API content response into a ConfluencePage object.

        Args:
            data: Raw API response data

        Returns:
            ConfluencePage object
        """
        # Handle body structure
        if "body" in data and isinstance(data["body"], dict):
            body_data = data["body"]

            # Extract text content for easier access
            if "view" in body_data and isinstance(body_data["view"], dict):
                data["html_content"] = body_data["view"].get("value", "")

            if "storage" in body_data and isinstance(body_data["storage"], dict):
                data["storage_content"] = body_data["storage"].get("value", "")

            if "plain" in body_data and isinstance(body_data["plain"], dict):
                data["plain_content"] = body_data["plain"].get("value", "")

        # Handle version information
        if "version" in data and isinstance(data["version"], dict):
            data["version_number"] = data["version"].get("number", 0)

        # Handle space reference
        if "space" in data and isinstance(data["space"], dict):
            space_data = data["space"]
            data["space_key"] = space_data.get("key", "")
            data["space_name"] = space_data.get("name", "")

        return ConfluencePage(**data)

    def _parse_search_result(self, data: dict[str, Any]) -> SearchResult:
        """
        Transform Confluence API search response into a SearchResult object.

        Args:
            data: Raw API response data

        Returns:
            SearchResult object
        """
        # Map size to total_size
        if "size" in data and "total_size" not in data:
            data["total_size"] = data["size"]

        # Transform result items
        transformed_results = []
        if "results" in data and isinstance(data["results"], list):
            for item in data["results"]:
                if isinstance(item, dict):
                    # Process each result item as a page
                    page = self._parse_page(item)
                    transformed_results.append(page)
                elif isinstance(item, ConfluencePage):
                    transformed_results.append(item)

            data["results"] = transformed_results

        return SearchResult(**data)

    def extract_content_fields(self, content: ConfluencePage) -> dict[str, Any]:
        """
        Extract relevant fields from content based on content type.

        Args:
            content: ConfluencePage object

        Returns:
            Dictionary with extracted fields
        """
        result = {
            "id": content.id,
            "title": content.title,
            "type": getattr(content.content_type, "value", content.content_type),
            "created_at": content.created_at,
            "updated_at": content.updated_at,
        }

        # Add space information if available
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

        # Add type-specific fields
        content_type = content.content_type
        if content_type in (ContentType.PAGE, ContentType.BLOGPOST):
            # For pages and blog posts
            result["html_content"] = getattr(content, "html_content", None)
            result["plain_content"] = getattr(content, "plain_content", None)

            version = getattr(content, "version", None)
            if version:
                result["version"] = int(getattr(version, "number", version))

            # Use atlassian-python-api's URL construction via get_page_url if available
            if hasattr(self.atlassian_api, "get_page_url") and content.id:
                try:
                    result["url"] = self.atlassian_api.get_page_url(
                        result.get("space_key"), content.id
                    )
                except Exception:
                    # Fall back to our URL construction if the API method fails
                    if result.get("space_key"):
                        result["url"] = (
                            f"{self.base_url}/wiki/spaces/{result['space_key']}/pages/{content.id}"
                        )

        elif content_type == ContentType.ATTACHMENT:
            # For attachments
            result["file_name"] = content.title
            extensions = getattr(content, "extensions", {})
            if extensions:
                result["file_size"] = extensions.get("fileSize")
                result["media_type"] = extensions.get("mediaType")

            # Use atlassian-python-api's URL construction for attachments if available
            if hasattr(self.atlassian_api, "get_attachment_url") and content.id:
                try:
                    # According to docs, we need both content_id and filename
                    filename = result.get("file_name", content.title)
                    result["download_url"] = self.atlassian_api.get_attachment_url(
                        content.id, filename=filename
                    )
                except Exception:
                    # Fall back to our URL construction if the API method fails
                    if result.get("space_key"):
                        result["download_url"] = (
                            f"{self.base_url}/wiki/download/attachments/{content.id}"
                        )

        elif content_type == ContentType.COMMENT:
            # For comments
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
        """
        Build CQL query for searching Confluence content.

        Args:
            query: Text to search for
            content_type: Filter by content type (page, blogpost, etc.)
            space_key: Filter by space key
            include_archived: Include archived content

        Returns:
            CQL query string
        """
        # Use the atlassian-python-api's CQL builder functionality if available
        if hasattr(self.atlassian_api, "cql_builder"):
            try:
                cql = self.atlassian_api.cql_builder()
                # Add text search
                cql.text_contains(query)

                # Add content type filter
                if content_type:
                    content_type_str = getattr(content_type, "value", content_type)
                    cql.content_type(content_type_str)

                # Add space filter
                if space_key:
                    cql.space(space_key)

                # Add archived filter
                if not include_archived:
                    cql.status_not("archived")

                return str(cql)
            except (AttributeError, TypeError):
                # Fall back to manual CQL building if cql_builder isn't available or doesn't work
                pass

        # Manual CQL building as fallback
        query_escaped = self._escape_cql(query)
        cql_parts = [f'text ~ "{query_escaped}"']

        # Add content type filter
        if content_type:
            content_type_str = getattr(content_type, "value", content_type)
            cql_parts.append(f'type = "{content_type_str}"')

        # Add space filter
        if space_key:
            space_key_escaped = self._escape_cql(space_key)
            cql_parts.append(f'space = "{space_key_escaped}"')

        # Add archived filter if include_archived is False
        if not include_archived:
            cql_parts.append('status != "archived"')

        # Combine CQL parts with AND operator
        return " AND ".join(cql_parts)

    # The previous custom pagination methods have been removed as we now use
    # the built-in pagination capabilities of atlassian-python-api

    @with_backoff()
    def search_by_cql(
        self,
        cql: str,
        limit: Optional[int] = None,
        start: Optional[int] = 0,
        expand: Optional[list[str]] = None,
        get_all_results: bool = False,
        max_results: Optional[int] = None,
    ) -> SearchResult:
        """
        Search Confluence content using CQL (Confluence Query Language).

        Args:
            cql: CQL query string
            limit: Maximum number of results per page (default: from config)
            start: Starting position (default: 0)
            expand: Fields to expand in the response (default: from config)
            get_all_results: Whether to fetch all pages of results (default: False)
            max_results: Maximum total results to fetch when get_all_results is True

        Returns:
            SearchResult object containing the search results

        Raises:
            ConfluenceConnectionError: If connection fails
            ConfluenceAuthenticationError: If authentication fails
            ConfluenceAPIError: If API returns an error
            SearchParameterError: If invalid search parameters are provided
        """
        # Minimal validation
        if not cql:
            raise SearchParameterError("CQL query cannot be empty")

        # Set actual limit value to use
        actual_limit = limit or search_config.default_limit

        # Prepare expand parameter
        expand_param = None
        if expand:
            expand_param = ",".join(expand)
        elif search_config.default_expand:
            expand_param = ",".join(search_config.default_expand)

        try:
            # atlassian-python-api's cql method can handle pagination internally when we need all results
            if get_all_results:
                # If max_results is specified, we'll fetch all and truncate later
                all_results = self.atlassian_api.cql(
                    cql,
                    limit=actual_limit,
                    start=start or 0,
                    expand=expand_param,
                    include_archived_spaces=True,  # We'll get everything and let the CQL filter handle archives
                )

                # Truncate results if max_results is specified
                if max_results and len(all_results.get("results", [])) > max_results:
                    all_results["results"] = all_results["results"][:max_results]

                return self._parse_search_result(all_results)
            else:
                # Simple case: use atlassian-python-api for CQL search
                results = self.atlassian_api.cql(
                    cql,
                    limit=actual_limit,
                    start=start or 0,
                    expand=expand_param,
                )

                # Transform the response to our model format
                return self._parse_search_result(results)
        except Exception as e:
            # Translate exceptions to our custom exception types
            if "401" in str(e):
                raise ConfluenceAuthenticationError(
                    "Authentication failed. Check username and API token."
                ) from e
            if isinstance(e, requests.exceptions.RequestException):
                raise ConfluenceConnectionError(cause=e)
            raise ConfluenceAPIError(error_message=str(e)) from e

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
        """
        Search Confluence content using a text query.

        Args:
            query: Text to search for
            content_type: Filter by content type (page, blogpost, etc.)
            space_key: Filter by space key
            include_archived: Include archived content (default: False)
            limit: Maximum number of results per page (default: from config)
            start: Starting position (default: 0)
            expand: Fields to expand in the response (default: from config)
            get_all_results: Whether to fetch all pages of results (default: False)
            max_results: Maximum total results to fetch when get_all_results is True

        Returns:
            SearchResult object containing the search results

        Raises:
            ConfluenceConnectionError: If connection fails
            ConfluenceAuthenticationError: If authentication fails
            ConfluenceAPIError: If API returns an error
            SearchParameterError: If invalid search parameters are provided
        """
        # Minimal validation
        if not query:
            raise SearchParameterError("Query cannot be empty")

        # Convert string content_type to enum if needed
        if content_type and isinstance(content_type, str):
            try:
                content_type = ContentType(content_type.lower())
            except ValueError:
                pass  # Use as is, API will handle invalid values

        try:
            # For pure text searches without other filters, we can use a simple CQL query
            if not content_type and not space_key and not get_all_results:
                actual_limit = limit or search_config.default_limit
                expand_param = None
                if expand:
                    expand_param = ",".join(expand)
                elif search_config.default_expand:
                    expand_param = ",".join(search_config.default_expand)

                # Create basic text search CQL
                cql_query = f'text ~ "{self._escape_cql(query)}"'
                if not include_archived:
                    cql_query += ' AND status != "archived"'

                results = self.atlassian_api.cql(
                    cql_query,
                    start=start or 0,
                    limit=actual_limit,
                    expand=expand_param,
                )
                return self._parse_search_result(results)
            else:
                # Build CQL query for more complex searches
                cql_query = self._build_search_cql(
                    query, content_type, space_key, include_archived
                )

                # Use the search_by_cql method which now uses atlassian-python-api
                return self.search_by_cql(
                    cql=cql_query,
                    limit=limit,
                    start=start,
                    expand=expand,
                    get_all_results=get_all_results,
                    max_results=max_results,
                )
        except Exception as e:
            # Translate exceptions to our custom exception types
            if "401" in str(e):
                raise ConfluenceAuthenticationError(
                    "Authentication failed. Check username and API token."
                ) from e
            if isinstance(e, requests.exceptions.RequestException):
                raise ConfluenceConnectionError(cause=e)
            raise ConfluenceAPIError(error_message=str(e)) from e
