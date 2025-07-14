class ConfluenceGatewayError(Exception):
    pass


class EmbeddingError(ConfluenceGatewayError):
    pass


class EmbeddingProviderError(ConfluenceGatewayError):
    pass


class ConfluenceConnectionError(ConfluenceGatewayError):
    def __init__(
        self,
        message: str = "Failed to connect to Confluence API",
        cause: Exception | None = None,
    ):
        self.cause = cause
        super().__init__(f"{message}: {str(cause)}" if cause else message)


class ConfluenceAuthenticationError(ConfluenceGatewayError):
    def __init__(self, message: str = "Authentication failed with Confluence API"):
        super().__init__(message)


class ConfluenceAPIError(ConfluenceGatewayError):
    def __init__(
        self, status_code: int | None = None, error_message: str | None = None
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
    pass


class GenerationError(ConfluenceGatewayError):
    pass


class EmbeddingCompatibilityError(ConfluenceGatewayError):
    """Raised when there's an incompatibility between embedding models."""

    pass
