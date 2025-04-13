import pytest
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
    ConfluenceGatewayError,
    EmbeddingError,
    EmbeddingProviderError,
    GenerationError,
    SearchParameterError,
    SemanticSearchError,
)


def test_confluence_gateway_error_instantiation():
    msg = "Base gateway error"
    err = ConfluenceGatewayError(msg)
    assert isinstance(err, Exception)
    assert str(err) == msg


def test_embedding_error_instantiation():
    msg = "Embedding failed"
    err = EmbeddingError(msg)
    assert isinstance(err, ConfluenceGatewayError)
    assert str(err) == msg


def test_embedding_provider_error_instantiation():
    msg = "Provider setup failed"
    err = EmbeddingProviderError(msg)
    assert isinstance(err, ConfluenceGatewayError)
    assert str(err) == msg


def test_confluence_connection_error_instantiation():
    msg = "Connection refused"
    err1 = ConfluenceConnectionError(msg)
    assert isinstance(err1, ConfluenceGatewayError)
    assert str(err1) == msg

    cause = ValueError("Underlying network issue")
    err2 = ConfluenceConnectionError(msg, cause=cause)
    assert isinstance(err2, ConfluenceGatewayError)
    assert msg in str(err2)
    assert str(cause) in str(err2)
    assert err2.cause is cause


def test_confluence_authentication_error_instantiation():
    msg = "Invalid API token"
    err = ConfluenceAuthenticationError(msg)
    assert isinstance(err, ConfluenceGatewayError)
    assert str(err) == msg
    err_default = ConfluenceAuthenticationError()
    assert "Authentication failed" in str(err_default)


def test_confluence_api_error_instantiation():
    err_base = ConfluenceAPIError()
    assert isinstance(err_base, ConfluenceGatewayError)
    assert "Confluence API error" == str(err_base)
    assert err_base.status_code is None
    assert err_base.error_message is None

    err_status = ConfluenceAPIError(status_code=404)
    assert "status code: 404" in str(err_status)
    assert err_status.status_code == 404

    err_msg = ConfluenceAPIError(error_message="Resource not found")
    assert ": Resource not found" in str(err_msg)
    assert err_msg.error_message == "Resource not found"

    err_full = ConfluenceAPIError(status_code=500, error_message="Server fault")
    assert "(status code: 500): Server fault" in str(err_full)
    assert err_full.status_code == 500
    assert err_full.error_message == "Server fault"


def test_search_parameter_error_instantiation():
    msg = "Invalid limit value"
    err = SearchParameterError(msg)
    assert isinstance(err, ConfluenceGatewayError)
    assert str(err) == msg
    err_default = SearchParameterError()
    assert "Invalid search parameters" in str(err_default)


def test_semantic_search_error_instantiation():
    msg = "Vector DB unavailable"
    err = SemanticSearchError(msg)
    assert isinstance(err, ConfluenceGatewayError)
    assert str(err) == msg


def test_generation_error_instantiation():
    msg = "LLM timed out"
    err = GenerationError(msg)
    assert isinstance(err, ConfluenceGatewayError)
    assert str(err) == msg
