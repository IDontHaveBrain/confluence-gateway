from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
    ConfluenceGatewayError,
    SearchParameterError,
)


class TestConfluenceGatewayError:
    def test_base_exception(self):
        error = ConfluenceGatewayError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


class TestConfluenceConnectionError:
    def test_default_message(self):
        error = ConfluenceConnectionError()
        assert str(error) == "Failed to connect to Confluence API"
        assert isinstance(error, ConfluenceGatewayError)

    def test_custom_message(self):
        error = ConfluenceConnectionError("Custom connection error")
        assert str(error) == "Custom connection error"

    def test_with_cause(self):
        cause = ValueError("Connection timeout")
        error = ConfluenceConnectionError(cause=cause)
        assert str(error) == "Failed to connect to Confluence API: Connection timeout"
        assert error.cause == cause

    def test_custom_message_with_cause(self):
        cause = ValueError("Connection timeout")
        error = ConfluenceConnectionError("Custom connection error", cause)
        assert str(error) == "Custom connection error: Connection timeout"
        assert error.cause == cause


class TestConfluenceAuthenticationError:
    def test_default_message(self):
        error = ConfluenceAuthenticationError()
        assert str(error) == "Authentication failed with Confluence API"
        assert isinstance(error, ConfluenceGatewayError)

    def test_custom_message(self):
        error = ConfluenceAuthenticationError("Invalid credentials")
        assert str(error) == "Invalid credentials"


class TestConfluenceAPIError:
    def test_default_message(self):
        error = ConfluenceAPIError()
        assert str(error) == "Confluence API error"
        assert isinstance(error, ConfluenceGatewayError)
        assert error.status_code is None
        assert error.error_message is None

    def test_with_status_code(self):
        error = ConfluenceAPIError(status_code=400)
        assert str(error) == "Confluence API error (status code: 400)"
        assert error.status_code == 400
        assert error.error_message is None

    def test_with_error_message(self):
        error = ConfluenceAPIError(error_message="Bad request")
        assert str(error) == "Confluence API error: Bad request"
        assert error.status_code is None
        assert error.error_message == "Bad request"

    def test_with_status_code_and_error_message(self):
        error = ConfluenceAPIError(status_code=404, error_message="Page not found")
        assert str(error) == "Confluence API error (status code: 404): Page not found"
        assert error.status_code == 404
        assert error.error_message == "Page not found"


class TestSearchParameterError:
    def test_default_message(self):
        error = SearchParameterError()
        assert str(error) == "Invalid search parameters"
        assert isinstance(error, ConfluenceGatewayError)

    def test_custom_message(self):
        error = SearchParameterError("Invalid query format")
        assert str(error) == "Invalid query format"
