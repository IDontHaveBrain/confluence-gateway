"""Parallel request test utilities and fixtures for concurrent API testing.

This module provides infrastructure for testing API endpoints under concurrent load:
- Async HTTP client fixtures
- Parallel request execution utilities
- Performance measurement and validation
- Error handling for concurrent requests

Usage:
    Tests can use the async_api_client fixture and utility functions to perform
    concurrent API requests and validate proper behavior under load.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest


@pytest.fixture
async def async_api_client(api_server):
    """Async API client fixture for concurrent testing with extended timeout for AI operations."""
    async with httpx.AsyncClient(
        base_url=api_server,
        timeout=httpx.Timeout(30.0),  # 30 second timeout for AI model operations
    ) as client:
        yield client


async def execute_parallel_requests(
    client: httpx.AsyncClient,
    requests: list[dict[str, Any]],
    max_concurrent: int = 5,
) -> list[httpx.Response]:
    """Execute multiple HTTP requests concurrently.

    Args:
        client: Async HTTP client instance
        requests: List of request dictionaries containing 'method', 'url', and optional 'json', 'params', 'headers'
        max_concurrent: Maximum number of concurrent requests (default: 5)

    Returns:
        List of responses in the same order as input requests

    Raises:
        ValueError: If requests list is empty or contains invalid request format
        httpx.HTTPError: If any HTTP error occurs during requests

    Example:
        requests = [
            {"method": "GET", "url": "/api/health"},
            {"method": "POST", "url": "/api/search/semantic", "json": {"query": "test", "top_k": 5}},
            {"method": "GET", "url": "/api/spaces", "params": {"limit": 10}},
        ]
        responses = await execute_parallel_requests(client, requests)
    """
    if not requests:
        raise ValueError("Requests list cannot be empty")

    # Validate request format
    for i, req in enumerate(requests):
        if not isinstance(req, dict) or "method" not in req or "url" not in req:
            raise ValueError(f"Request {i} must be a dict with 'method' and 'url' keys")

    async def make_request(request_config: dict[str, Any]) -> httpx.Response:
        """Make a single HTTP request with error handling."""
        method = request_config["method"].upper()
        url = request_config["url"]

        # Extract optional parameters
        json_data = request_config.get("json")
        params = request_config.get("params")
        headers = request_config.get("headers")

        # Make the request based on method
        if method == "GET":
            return await client.get(url, params=params, headers=headers)
        elif method == "POST":
            return await client.post(
                url, json=json_data, params=params, headers=headers
            )
        elif method == "PUT":
            return await client.put(url, json=json_data, params=params, headers=headers)
        elif method == "DELETE":
            return await client.delete(url, params=params, headers=headers)
        elif method == "PATCH":
            return await client.patch(
                url, json=json_data, params=params, headers=headers
            )
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

    # Use semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_request(request_config: dict[str, Any]) -> httpx.Response:
        """Make request with concurrency limiting."""
        async with semaphore:
            return await make_request(request_config)

    # Execute all requests concurrently
    tasks = [limited_request(req) for req in requests]
    responses = await asyncio.gather(*tasks)

    return responses


async def measure_response_times(
    client: httpx.AsyncClient,
    requests: list[dict[str, Any]],
    iterations: int = 1,
    max_concurrent: int = 5,
) -> dict[str, Any]:
    """Measure response times for concurrent API requests.

    Args:
        client: Async HTTP client instance
        requests: List of request dictionaries
        iterations: Number of times to repeat the measurement (default: 1)
        max_concurrent: Maximum number of concurrent requests (default: 5)

    Returns:
        Dictionary containing performance metrics:
        - total_time: Total time for all iterations
        - average_time: Average time per iteration
        - requests_per_second: Average requests per second
        - individual_times: List of response times for each request per iteration
        - status_codes: List of status codes for validation

    Example:
        requests = [{"method": "GET", "url": "/api/health"}] * 10
        metrics = await measure_response_times(client, requests, iterations=3)
        print(f"Average RPS: {metrics['requests_per_second']:.2f}")
    """
    if iterations < 1:
        raise ValueError("Iterations must be at least 1")

    all_times = []
    all_status_codes = []
    total_start_time = time.time()

    for iteration in range(iterations):
        # Execute requests with individual timing
        tasks = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def timed_request(
            request_config: dict[str, Any],
        ) -> tuple[float, httpx.Response]:
            """Execute request and measure its duration."""
            async with semaphore:
                start_time = time.time()

                method = request_config["method"].upper()
                url = request_config["url"]
                json_data = request_config.get("json")
                params = request_config.get("params")
                headers = request_config.get("headers")

                if method == "GET":
                    response = await client.get(url, params=params, headers=headers)
                elif method == "POST":
                    response = await client.post(
                        url, json=json_data, params=params, headers=headers
                    )
                elif method == "PUT":
                    response = await client.put(
                        url, json=json_data, params=params, headers=headers
                    )
                elif method == "DELETE":
                    response = await client.delete(url, params=params, headers=headers)
                elif method == "PATCH":
                    response = await client.patch(
                        url, json=json_data, params=params, headers=headers
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                end_time = time.time()
                return end_time - start_time, response

        # Execute all requests for this iteration
        tasks = [timed_request(req) for req in requests]
        results = await asyncio.gather(*tasks)

        # Extract timing and response data
        iteration_times = [duration for duration, _ in results]
        iteration_responses = [response for _, response in results]
        iteration_status_codes = [resp.status_code for resp in iteration_responses]

        all_times.extend(iteration_times)
        all_status_codes.extend(iteration_status_codes)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    total_requests = len(requests) * iterations

    # Calculate metrics
    average_time = total_time / iterations
    requests_per_second = total_requests / total_time if total_time > 0 else 0

    return {
        "total_time": total_time,
        "average_time": average_time,
        "requests_per_second": requests_per_second,
        "individual_times": all_times,
        "status_codes": all_status_codes,
        "total_requests": total_requests,
        "iterations": iterations,
        "max_response_time": max(all_times) if all_times else 0,
        "min_response_time": min(all_times) if all_times else 0,
        "avg_response_time": sum(all_times) / len(all_times) if all_times else 0,
    }


async def assert_concurrent_responses(
    responses: list[httpx.Response],
    expected_status: int = 200,
    expected_keys: list[str] | None = None,
    allow_failures: int = 0,
) -> None:
    """Validate that concurrent API responses meet expected criteria.

    Args:
        responses: List of HTTP responses to validate
        expected_status: Expected HTTP status code (default: 200)
        expected_keys: List of keys that must be present in JSON responses (optional)
        allow_failures: Number of failed responses to tolerate (default: 0)

    Raises:
        AssertionError: If validation fails

    Example:
        responses = await execute_parallel_requests(client, requests)
        await assert_concurrent_responses(
            responses,
            expected_status=200,
            expected_keys=["results", "total"]
        )
    """
    if not responses:
        raise AssertionError("No responses to validate")

    failed_responses = []
    status_failures = []
    json_failures = []
    key_failures = []

    for i, response in enumerate(responses):
        try:
            # Check status code
            if response.status_code != expected_status:
                status_failures.append(
                    f"Response {i}: expected {expected_status}, got {response.status_code}"
                )
                failed_responses.append(i)
                continue

            # Check JSON structure if expected_keys provided
            if expected_keys:
                try:
                    data = response.json()
                    missing_keys = [key for key in expected_keys if key not in data]
                    if missing_keys:
                        key_failures.append(
                            f"Response {i}: missing keys {missing_keys}"
                        )
                        failed_responses.append(i)
                except Exception as e:
                    json_failures.append(f"Response {i}: JSON parsing failed - {e}")
                    failed_responses.append(i)

        except Exception as e:
            failed_responses.append(i)
            json_failures.append(f"Response {i}: Validation error - {e}")

    # Count unique failures
    unique_failures = len(set(failed_responses))

    # Check if failures exceed tolerance
    if unique_failures > allow_failures:
        error_details = []
        if status_failures:
            error_details.extend(status_failures)
        if json_failures:
            error_details.extend(json_failures)
        if key_failures:
            error_details.extend(key_failures)

        error_message = (
            f"Concurrent response validation failed: {unique_failures} failures "
            f"(allowed: {allow_failures})\n" + "\n".join(error_details)
        )
        raise AssertionError(error_message)


# Utility function for common concurrent test patterns
async def run_concurrent_test(
    client: httpx.AsyncClient,
    endpoint_config: dict[str, Any],
    concurrent_count: int = 5,
    expected_status: int = 200,
    expected_keys: list[str] | None = None,
    performance_threshold_rps: float | None = None,
) -> dict[str, Any]:
    """Run a complete concurrent test for an API endpoint.

    This is a convenience function that combines request execution, timing,
    and validation for common concurrent testing scenarios.

    Args:
        client: Async HTTP client instance
        endpoint_config: Single request configuration to replicate
        concurrent_count: Number of concurrent requests to make
        expected_status: Expected HTTP status code
        expected_keys: Keys that must be present in JSON responses
        performance_threshold_rps: Minimum requests per second (optional)

    Returns:
        Dictionary containing test results and performance metrics

    Raises:
        AssertionError: If validation or performance criteria fail

    Example:
        config = {"method": "GET", "url": "/api/health"}
        results = await run_concurrent_test(
            client, config, concurrent_count=10, performance_threshold_rps=50
        )
    """
    # Create list of identical requests
    requests = [endpoint_config.copy() for _ in range(concurrent_count)]

    # Measure performance
    metrics = await measure_response_times(client, requests, iterations=1)

    # Execute requests for validation
    responses = await execute_parallel_requests(client, requests)

    # Validate responses
    await assert_concurrent_responses(
        responses, expected_status=expected_status, expected_keys=expected_keys
    )

    # Check performance threshold if specified
    if (
        performance_threshold_rps
        and metrics["requests_per_second"] < performance_threshold_rps
    ):
        raise AssertionError(
            f"Performance below threshold: {metrics['requests_per_second']:.2f} RPS "
            f"< {performance_threshold_rps} RPS"
        )

    return {
        "success": True,
        "concurrent_count": concurrent_count,
        "performance_metrics": metrics,
        "responses": responses,
        "endpoint": endpoint_config["url"],
    }


# Test examples using the parallel utilities
@pytest.mark.asyncio
async def test_parallel_health_checks(async_api_client):
    """Test multiple concurrent health check requests."""
    requests = [{"method": "GET", "url": "/health"}] * 10

    responses = await execute_parallel_requests(async_api_client, requests)
    await assert_concurrent_responses(responses, expected_status=200)

    # Verify all responses are identical except for timestamp fields
    response_data = [resp.json() for resp in responses]
    first_response = response_data[0]

    # Extract important fields to compare (excluding timestamp which varies between requests)
    def extract_comparable_fields(response_data):
        """Extract fields that should be identical across parallel health checks."""
        comparable = {}
        for key, value in response_data.items():
            if (
                key != "timestamp"
            ):  # Exclude timestamp as it naturally differs between requests
                comparable[key] = value
        return comparable

    first_comparable = extract_comparable_fields(first_response)

    for i, data in enumerate(response_data[1:], 1):
        comparable_data = extract_comparable_fields(data)
        assert comparable_data == first_comparable, (
            f"Health check response {i} differs in non-timestamp fields: {comparable_data} != {first_comparable}"
        )

    # Verify that essential fields are present in all responses
    essential_fields = ["status", "version", "confluence_connection"]
    for i, data in enumerate(response_data):
        for field in essential_fields:
            assert field in data, f"Response {i} missing essential field: {field}"


@pytest.mark.asyncio
async def test_parallel_search_requests(async_api_client):
    """Test concurrent semantic search requests with different queries."""
    base_request = {"method": "POST", "url": "/api/search/semantic"}

    requests = [
        {**base_request, "json": {"query": f"test query {i}", "top_k": 5}}
        for i in range(5)
    ]

    metrics = await measure_response_times(async_api_client, requests)
    responses = await execute_parallel_requests(async_api_client, requests)

    await assert_concurrent_responses(
        responses, expected_status=200, expected_keys=["results"]
    )

    # Log performance metrics
    print(f"Concurrent search performance: {metrics['requests_per_second']:.2f} RPS")
    print(f"Average response time: {metrics['avg_response_time']:.3f}s")


@pytest.mark.asyncio
async def test_mixed_concurrent_requests(async_api_client):
    """Test various API endpoints concurrently."""
    requests = [
        {"method": "GET", "url": "/health"},
        {"method": "GET", "url": "/api/spaces/"},
        {
            "method": "POST",
            "url": "/api/search/semantic",
            "json": {"query": "test", "top_k": 3},
        },
        {"method": "GET", "url": "/health"},  # Duplicate for load testing
        {
            "method": "POST",
            "url": "/api/search/cql",
            "json": {"cql": "text ~ test", "limit": 5},
        },
    ]

    start_time = time.time()
    responses = await execute_parallel_requests(
        async_api_client, requests, max_concurrent=3
    )
    end_time = time.time()

    # Validate responses with different expected status codes
    assert responses[0].status_code == 200  # Health check
    assert responses[1].status_code == 200  # Spaces
    assert responses[2].status_code == 200  # Semantic search
    assert responses[3].status_code == 200  # Health check duplicate
    assert responses[4].status_code == 200  # CQL search

    total_time = end_time - start_time
    rps = len(requests) / total_time

    print(f"Mixed concurrent requests: {rps:.2f} RPS, total time: {total_time:.3f}s")


@pytest.mark.asyncio
async def test_error_handling_in_parallel_requests(async_api_client):
    """Test error handling for concurrent requests."""
    # Mix valid and invalid requests
    requests = [
        {"method": "GET", "url": "/health"},  # Valid
        {"method": "GET", "url": "/nonexistent"},  # Should return 404
        {
            "method": "POST",
            "url": "/api/search/semantic",
            "json": {"invalid": "payload"},
        },  # Invalid payload
        {"method": "GET", "url": "/health"},  # Valid
    ]

    responses = await execute_parallel_requests(async_api_client, requests)

    # Validate mixed success/error responses
    assert responses[0].status_code == 200  # Valid health check
    assert responses[1].status_code == 404  # Non-existent endpoint
    assert responses[2].status_code in [400, 422]  # Invalid payload
    assert responses[3].status_code == 200  # Valid health check

    # Test with allow_failures parameter
    try:
        await assert_concurrent_responses(
            responses, expected_status=200, allow_failures=2
        )
        print("Error handling test passed with allowed failures")
    except AssertionError:
        pytest.fail("Should have passed with allow_failures=2")
