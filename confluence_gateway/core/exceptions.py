from typing import Optional


class ConfluenceGatewayError(Exception):
    pass


class ConfluenceConnectionError(ConfluenceGatewayError):
    def __init__(
        self,
        message: str = "Failed to connect to Confluence API",
        cause: Optional[Exception] = None,
    ):
        self.cause = cause
        super().__init__(f"{message}: {str(cause)}" if cause else message)


class ConfluenceAuthenticationError(ConfluenceGatewayError):
    def __init__(self, message: str = "Authentication failed with Confluence API"):
        super().__init__(message)


class ConfluenceAPIError(ConfluenceGatewayError):
    def __init__(
        self, status_code: Optional[int] = None, error_message: Optional[str] = None
    ):
        message = "Confluence API error"
        if status_code:
            message += f" (status code: {status_code})"
        if error_message:
            message += f": {error_message}"
        self.status_code = status_code
        self.error_message = error_message
        super().__init__(message)


class SearchParameterError(ConfluenceGatewayError):
    def __init__(self, message: str = "Invalid search parameters"):
        super().__init__(message)


class SemanticSearchError(ConfluenceGatewayError):
    """Error specific to semantic search operations."""

    pass
