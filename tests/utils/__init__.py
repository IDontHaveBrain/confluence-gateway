"""Test utilities package for Confluence Gateway.

This package provides consolidated test utilities to eliminate duplication
and standardize test patterns across the testing suite.

Performance Testing Utilities:
    - performance_tracked: Decorator for automatic function timing
    - PerformanceTracker: Context manager for scoped timing with milestones
    - PerformanceBenchmark: Advanced benchmarking with statistical analysis
    - calculate_performance_stats: Statistical analysis for timing data
    - temporary_performance_threshold: Enforce performance requirements
    - parallel_execution_timer: Time multiple concurrent operations

Mock Testing Utilities:
    - MockContextFactory: Factory for creating standardized mock contexts
    - LiteLLMContext: Context manager for LiteLLM async mocking
    - EnvironmentContext: Context manager for temporary environment variables
    - SentenceTransformerContext: Context manager for sentence transformer mocking
    - ConfigurationMockHelper: Helper for configuration-related mocking
    - ResponseValidationHelper: Helper for response validation patterns
    - PerformanceMockHelper: Helper for performance-related mocking
    - MockMigrationHelper: Helper for migrating existing mock patterns
"""

# Performance testing utilities (consolidates 60+ manual timing blocks)
# Mock testing utilities (consolidates 8+ mocking patterns)
from .mock_helpers import (
    ConfigurationMockHelper,
    EnvironmentContext,
    LiteLLMContext,
    MockContextFactory,
    MockMigrationHelper,
    PerformanceMockHelper,
    ResponseValidationHelper,
    SentenceTransformerContext,
)
from .performance_helpers import (
    PerformanceBenchmark,
    PerformanceStats,
    PerformanceTracker,
    calculate_performance_stats,
    clear_performance_metrics,
    get_performance_metrics,
    log_embedding_operation,
    parallel_execution_timer,
    performance_tracked,
    temporary_performance_threshold,
)

__all__ = [
    # Performance testing utilities
    "PerformanceBenchmark",
    "PerformanceStats",
    "PerformanceTracker",
    "calculate_performance_stats",
    "clear_performance_metrics",
    "get_performance_metrics",
    "log_embedding_operation",
    "parallel_execution_timer",
    "performance_tracked",
    "temporary_performance_threshold",
    # Mock testing utilities
    "ConfigurationMockHelper",
    "EnvironmentContext",
    "LiteLLMContext",
    "MockContextFactory",
    "MockMigrationHelper",
    "PerformanceMockHelper",
    "ResponseValidationHelper",
    "SentenceTransformerContext",
]
