"""
Performance testing utilities for Confluence Gateway test suite.

This module consolidates 60+ manual timing blocks throughout the test suite into
reusable decorators, context managers, and benchmark classes. It provides:

- @performance_tracked decorator for automatic timing and logging
- PerformanceTracker context manager for scoped timing
- PerformanceBenchmark class for complex multi-iteration benchmarking
- Statistical analysis utilities (mean, median, percentiles)
- Integration with existing log_embedding_operation() patterns
- Support for parallel execution timing and comparison benchmarks

Usage Examples:

    # Decorator pattern (replaces simple start_time/end_time blocks)
    @performance_tracked("model_loading")
    def load_model():
        return SentenceTransformer("all-MiniLM-L6-v2")

    # Context manager pattern (replaces manual timing calculations)
    with PerformanceTracker("embedding_operation") as tracker:
        embeddings = model.encode(texts)
        tracker.log_milestone("encoding_complete")

    # Benchmark pattern (replaces multi-iteration timing loops)
    benchmark = PerformanceBenchmark("provider_comparison")
    benchmark.run_iterations("standard", lambda: create_standard_provider(), iterations=5)
    benchmark.run_iterations("optimized", lambda: create_optimized_provider(), iterations=5)
    benchmark.compare_results("standard", "optimized")

    # Statistical analysis
    times = [0.123, 0.145, 0.134, 0.129, 0.141]
    stats = calculate_performance_stats(times)
    print(f"Mean: {stats.mean:.3f}s, P95: {stats.p95:.3f}s")
"""

import asyncio
import functools
import statistics
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union

# Type aliases for clarity
TimingFunction = Callable[[], Any]
AsyncTimingFunction = Callable[[], Any]
F = TypeVar("F", bound=Callable[..., Any])

# Global performance metrics storage (thread-safe)
_performance_metrics: dict[str, list[float]] = {}
_metrics_lock = threading.Lock()


@dataclass
class PerformanceStats:
    """Statistical analysis results for performance measurements."""

    mean: float
    median: float
    min_time: float
    max_time: float
    std_dev: float
    p50: float  # 50th percentile (median)
    p95: float  # 95th percentile
    p99: float  # 99th percentile
    count: int
    total_time: float

    def __str__(self) -> str:
        """Human-readable summary of performance statistics."""
        return (
            f"Performance Stats - Count: {self.count}, "
            f"Mean: {self.mean:.3f}s, Median: {self.median:.3f}s, "
            f"Min: {self.min_time:.3f}s, Max: {self.max_time:.3f}s, "
            f"P95: {self.p95:.3f}s, StdDev: {self.std_dev:.3f}s"
        )


def calculate_performance_stats(times: list[float]) -> PerformanceStats:
    """
    Calculate comprehensive statistical analysis for performance measurements.

    Args:
        times: List of timing measurements in seconds

    Returns:
        PerformanceStats with comprehensive statistical analysis

    Raises:
        ValueError: If times list is empty
    """
    if not times:
        raise ValueError("Cannot calculate statistics for empty timing data")

    sorted_times = sorted(times)

    return PerformanceStats(
        mean=statistics.mean(times),
        median=statistics.median(times),
        min_time=min(times),
        max_time=max(times),
        std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
        p50=sorted_times[int(0.5 * len(sorted_times))],
        p95=sorted_times[int(0.95 * len(sorted_times))],
        p99=sorted_times[int(0.99 * len(sorted_times))],
        count=len(times),
        total_time=sum(times),
    )


def log_embedding_operation(operation_name: str, duration: float) -> None:
    """
    Log performance metrics for embedding operations (compatibility function).

    This maintains compatibility with existing test patterns while providing
    centralized performance tracking.

    Args:
        operation_name: Name of the embedding operation
        duration: Duration in seconds
    """
    with _metrics_lock:
        if operation_name not in _performance_metrics:
            _performance_metrics[operation_name] = []
        _performance_metrics[operation_name].append(duration)


def get_performance_metrics(
    operation_name: str | None = None,
) -> dict[str, list[float]]:
    """
    Retrieve stored performance metrics.

    Args:
        operation_name: Optional specific operation to retrieve. If None, returns all metrics.

    Returns:
        Dictionary of operation names to timing measurements
    """
    with _metrics_lock:
        if operation_name:
            return {operation_name: _performance_metrics.get(operation_name, [])}
        return _performance_metrics.copy()


def clear_performance_metrics(operation_name: str | None = None) -> None:
    """
    Clear stored performance metrics.

    Args:
        operation_name: Optional specific operation to clear. If None, clears all metrics.
    """
    with _metrics_lock:
        if operation_name:
            _performance_metrics.pop(operation_name, None)
        else:
            _performance_metrics.clear()


def performance_tracked(
    operation_name: str,
    log_result: bool = True,
    print_result: bool = True,
    threshold_warning: float | None = None,
) -> Callable[[F], F]:
    """
    Decorator to automatically track and log function execution time.

    Replaces the common pattern:
        start_time = time.time()
        result = function()
        duration = time.time() - start_time
        log_embedding_operation("operation_name", duration)
        print(f"Operation completed in {duration:.3f}s")

    Args:
        operation_name: Name for the operation (used in logging)
        log_result: Whether to log the result using log_embedding_operation()
        print_result: Whether to print timing information to stdout
        threshold_warning: Optional threshold in seconds to warn about slow operations

    Returns:
        Decorated function that automatically tracks performance

    Examples:
        @performance_tracked("model_loading")
        def load_model():
            return SentenceTransformer("all-MiniLM-L6-v2")

        @performance_tracked("embedding_batch", threshold_warning=1.0)
        def embed_texts(texts):
            return model.encode(texts)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time

                # Log performance metrics
                if log_result:
                    log_embedding_operation(operation_name, duration)

                # Print timing information
                if print_result:
                    print(f"{operation_name} completed in {duration:.3f}s")

                # Warn about slow operations
                if threshold_warning and duration > threshold_warning:
                    print(
                        f"WARNING: {operation_name} took {duration:.3f}s (threshold: {threshold_warning:.3f}s)"
                    )

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time

                # Log performance metrics
                if log_result:
                    log_embedding_operation(operation_name, duration)

                # Print timing information
                if print_result:
                    print(f"{operation_name} completed in {duration:.3f}s")

                # Warn about slow operations
                if threshold_warning and duration > threshold_warning:
                    print(
                        f"WARNING: {operation_name} took {duration:.3f}s (threshold: {threshold_warning:.3f}s)"
                    )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator


class PerformanceTracker:
    """
    Context manager for tracking performance of code blocks with milestone support.

    Replaces manual timing patterns and provides additional features like
    milestone tracking and detailed logging.

    Examples:
        # Simple timing (replaces start_time/end_time pattern)
        with PerformanceTracker("embedding_operation") as tracker:
            embeddings = model.encode(texts)

        # With milestones for detailed breakdown
        with PerformanceTracker("complex_operation") as tracker:
            data = load_data()
            tracker.log_milestone("data_loaded")

            processed = process_data(data)
            tracker.log_milestone("data_processed")

            result = analyze_data(processed)
            tracker.log_milestone("analysis_complete")
    """

    def __init__(
        self,
        operation_name: str,
        log_result: bool = True,
        print_result: bool = True,
        print_milestones: bool = False,
    ):
        """
        Initialize performance tracker.

        Args:
            operation_name: Name for the operation (used in logging)
            log_result: Whether to log final result using log_embedding_operation()
            print_result: Whether to print timing information to stdout
            print_milestones: Whether to print milestone information
        """
        self.operation_name = operation_name
        self.log_result = log_result
        self.print_result = print_result
        self.print_milestones = print_milestones
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.milestones: list[tuple[str, float]] = []

    def __enter__(self) -> "PerformanceTracker":
        """Start timing the operation."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Complete timing and log results."""
        self.end_time = time.time()

        if self.start_time is not None:
            duration = self.end_time - self.start_time

            # Log performance metrics
            if self.log_result:
                log_embedding_operation(self.operation_name, duration)

            # Print timing information
            if self.print_result:
                print(f"{self.operation_name} completed in {duration:.3f}s")

                # Print milestone breakdown if requested
                if self.print_milestones and self.milestones:
                    print(f"  Milestone breakdown for {self.operation_name}:")
                    prev_time = self.start_time
                    for milestone_name, milestone_time in self.milestones:
                        milestone_duration = milestone_time - prev_time
                        total_elapsed = milestone_time - self.start_time
                        print(
                            f"    {milestone_name}: +{milestone_duration:.3f}s (total: {total_elapsed:.3f}s)"
                        )
                        prev_time = milestone_time

    def log_milestone(self, milestone_name: str) -> None:
        """
        Log a milestone during operation execution.

        Args:
            milestone_name: Name of the milestone
        """
        milestone_time = time.time()
        self.milestones.append((milestone_name, milestone_time))

        if self.print_milestones and self.start_time is not None:
            elapsed = milestone_time - self.start_time
            print(f"  Milestone {milestone_name}: {elapsed:.3f}s elapsed")

    @property
    def elapsed_time(self) -> float | None:
        """Get current elapsed time if tracking is active."""
        if self.start_time is None:
            return None
        return time.time() - self.start_time

    @property
    def duration(self) -> float | None:
        """Get total duration if tracking is complete."""
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time


class PerformanceBenchmark:
    """
    Advanced benchmarking class for complex performance testing scenarios.

    Replaces multi-iteration timing loops and provides statistical analysis,
    comparison capabilities, and detailed reporting.

    Examples:
        # Compare different implementation approaches
        benchmark = PerformanceBenchmark("provider_comparison")

        # Run multiple iterations of each approach
        benchmark.run_iterations("standard", lambda: create_standard_provider(), iterations=5)
        benchmark.run_iterations("optimized", lambda: create_optimized_provider(), iterations=5)

        # Compare results and get improvement metrics
        improvement = benchmark.compare_results("standard", "optimized")
        print(f"Optimization provides {improvement:.2f}x improvement")

        # Get detailed statistics
        stats = benchmark.get_statistics("optimized")
        print(f"Optimized version: {stats}")
    """

    def __init__(self, benchmark_name: str, log_results: bool = True):
        """
        Initialize performance benchmark.

        Args:
            benchmark_name: Name for the benchmark suite
            log_results: Whether to log individual results using log_embedding_operation()
        """
        self.benchmark_name = benchmark_name
        self.log_results = log_results
        self.results: dict[str, list[float]] = {}
        self.metadata: dict[str, dict[str, Any]] = {}

    def run_iterations(
        self,
        variant_name: str,
        operation: TimingFunction,
        iterations: int = 5,
        warmup_iterations: int = 1,
        print_progress: bool = True,
    ) -> list[float]:
        """
        Run multiple iterations of an operation and collect timing data.

        Replaces patterns like:
            times = []
            for _ in range(iterations):
                start_time = time.time()
                operation()
                times.append(time.time() - start_time)

        Args:
            variant_name: Name for this variant (e.g., "standard", "optimized")
            operation: Function to execute and time
            iterations: Number of timing iterations to run
            warmup_iterations: Number of warmup iterations (not timed)
            print_progress: Whether to print progress information

        Returns:
            List of timing measurements in seconds
        """
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if print_progress:
            print(
                f"Running {self.benchmark_name}.{variant_name}: {iterations} iterations "
                f"(+{warmup_iterations} warmup)"
            )

        # Warmup iterations (not timed)
        for i in range(warmup_iterations):
            if print_progress and warmup_iterations > 0:
                print(f"  Warmup iteration {i + 1}/{warmup_iterations}")
            operation()

        # Timed iterations
        times = []
        for i in range(iterations):
            if print_progress:
                print(f"  Iteration {i + 1}/{iterations}")

            start_time = time.time()
            try:
                operation()
            except Exception as e:
                print(f"  ERROR in iteration {i + 1}: {e}")
                raise
            finally:
                duration = time.time() - start_time
                times.append(duration)

                # Log individual results
                if self.log_results:
                    log_embedding_operation(
                        f"{self.benchmark_name}_{variant_name}_iteration", duration
                    )

        # Store results
        self.results[variant_name] = times
        self.metadata[variant_name] = {
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "timestamp": time.time(),
        }

        # Log aggregate results
        if self.log_results:
            avg_time = sum(times) / len(times)
            log_embedding_operation(
                f"{self.benchmark_name}_{variant_name}_average", avg_time
            )

        if print_progress:
            avg_time = sum(times) / len(times)
            print(f"  {variant_name} average: {avg_time:.3f}s")

        return times

    async def run_async_iterations(
        self,
        variant_name: str,
        operation: AsyncTimingFunction,
        iterations: int = 5,
        warmup_iterations: int = 1,
        print_progress: bool = True,
    ) -> list[float]:
        """
        Run multiple iterations of an async operation and collect timing data.

        Args:
            variant_name: Name for this variant
            operation: Async function to execute and time
            iterations: Number of timing iterations to run
            warmup_iterations: Number of warmup iterations (not timed)
            print_progress: Whether to print progress information

        Returns:
            List of timing measurements in seconds
        """
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if print_progress:
            print(
                f"Running {self.benchmark_name}.{variant_name} (async): {iterations} iterations "
                f"(+{warmup_iterations} warmup)"
            )

        # Warmup iterations (not timed)
        for i in range(warmup_iterations):
            if print_progress and warmup_iterations > 0:
                print(f"  Warmup iteration {i + 1}/{warmup_iterations}")
            await operation()

        # Timed iterations
        times = []
        for i in range(iterations):
            if print_progress:
                print(f"  Iteration {i + 1}/{iterations}")

            start_time = time.time()
            try:
                await operation()
            except Exception as e:
                print(f"  ERROR in iteration {i + 1}: {e}")
                raise
            finally:
                duration = time.time() - start_time
                times.append(duration)

                # Log individual results
                if self.log_results:
                    log_embedding_operation(
                        f"{self.benchmark_name}_{variant_name}_iteration", duration
                    )

        # Store results
        self.results[variant_name] = times
        self.metadata[variant_name] = {
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "timestamp": time.time(),
            "async": True,
        }

        # Log aggregate results
        if self.log_results:
            avg_time = sum(times) / len(times)
            log_embedding_operation(
                f"{self.benchmark_name}_{variant_name}_average", avg_time
            )

        if print_progress:
            avg_time = sum(times) / len(times)
            print(f"  {variant_name} average: {avg_time:.3f}s")

        return times

    def get_statistics(self, variant_name: str) -> PerformanceStats:
        """
        Get comprehensive statistics for a benchmark variant.

        Args:
            variant_name: Name of the variant to analyze

        Returns:
            PerformanceStats with comprehensive analysis

        Raises:
            KeyError: If variant_name not found in results
        """
        if variant_name not in self.results:
            raise KeyError(f"Variant '{variant_name}' not found in benchmark results")

        return calculate_performance_stats(self.results[variant_name])

    def compare_results(
        self,
        baseline_variant: str,
        comparison_variant: str,
        print_comparison: bool = True,
    ) -> float:
        """
        Compare two benchmark variants and calculate improvement ratio.

        Replaces patterns like:
            if standard_avg_time > 0:
                improvement_ratio = standard_avg_time / optimized_avg_time
                print(f"Performance improvement: {improvement_ratio:.2f}x faster")

        Args:
            baseline_variant: Name of the baseline variant
            comparison_variant: Name of the variant to compare against baseline
            print_comparison: Whether to print detailed comparison

        Returns:
            Improvement ratio (>1.0 means comparison is faster than baseline)

        Raises:
            KeyError: If either variant not found in results
        """
        if baseline_variant not in self.results:
            raise KeyError(
                f"Baseline variant '{baseline_variant}' not found in benchmark results"
            )
        if comparison_variant not in self.results:
            raise KeyError(
                f"Comparison variant '{comparison_variant}' not found in benchmark results"
            )

        baseline_stats = self.get_statistics(baseline_variant)
        comparison_stats = self.get_statistics(comparison_variant)

        # Calculate improvement ratio based on mean times
        if baseline_stats.mean > 0:
            improvement_ratio = baseline_stats.mean / comparison_stats.mean
        else:
            improvement_ratio = 1.0

        if print_comparison:
            print(f"\n{self.benchmark_name} - Performance Comparison:")
            print(
                f"  {baseline_variant}: {baseline_stats.mean:.3f}s (±{baseline_stats.std_dev:.3f}s)"
            )
            print(
                f"  {comparison_variant}: {comparison_stats.mean:.3f}s (±{comparison_stats.std_dev:.3f}s)"
            )

            if improvement_ratio > 1.0:
                print(
                    f"  {comparison_variant} is {improvement_ratio:.2f}x faster than {baseline_variant}"
                )
                time_saved = baseline_stats.mean - comparison_stats.mean
                percent_improvement = (
                    (baseline_stats.mean - comparison_stats.mean) / baseline_stats.mean
                ) * 100
                print(
                    f"  Time saved: {time_saved:.3f}s ({percent_improvement:.1f}% improvement)"
                )
            elif improvement_ratio < 1.0:
                slowdown_ratio = 1.0 / improvement_ratio
                print(
                    f"  {comparison_variant} is {slowdown_ratio:.2f}x slower than {baseline_variant}"
                )
                time_lost = comparison_stats.mean - baseline_stats.mean
                percent_slowdown = (
                    (comparison_stats.mean - baseline_stats.mean) / baseline_stats.mean
                ) * 100
                print(
                    f"  Time lost: {time_lost:.3f}s ({percent_slowdown:.1f}% slowdown)"
                )
            else:
                print(
                    f"  {comparison_variant} and {baseline_variant} have similar performance"
                )

        return improvement_ratio

    def assert_performance_improvement(
        self,
        baseline_variant: str,
        comparison_variant: str,
        min_improvement: float = 1.1,
        error_message: str | None = None,
    ) -> None:
        """
        Assert that one variant shows meaningful performance improvement over another.

        Replaces patterns like:
            assert optimized_avg_time < standard_avg_time, (
                f"Optimized version should be faster: {optimized_avg_time:.3f}s vs {standard_avg_time:.3f}s"
            )

        Args:
            baseline_variant: Name of the baseline variant
            comparison_variant: Name of the variant that should be faster
            min_improvement: Minimum improvement ratio required (>1.0)
            error_message: Optional custom error message

        Raises:
            AssertionError: If performance improvement is not sufficient
        """
        improvement_ratio = self.compare_results(
            baseline_variant, comparison_variant, print_comparison=False
        )

        if improvement_ratio < min_improvement:
            baseline_stats = self.get_statistics(baseline_variant)
            comparison_stats = self.get_statistics(comparison_variant)

            if error_message is None:
                error_message = (
                    f"{comparison_variant} should be at least {min_improvement:.2f}x faster than {baseline_variant}, "
                    f"but was only {improvement_ratio:.2f}x faster "
                    f"({comparison_stats.mean:.3f}s vs {baseline_stats.mean:.3f}s)"
                )

            raise AssertionError(error_message)

    def get_all_results(self) -> dict[str, list[float]]:
        """Get all benchmark results."""
        return self.results.copy()

    def get_variants(self) -> list[str]:
        """Get list of all tested variants."""
        return list(self.results.keys())

    def print_summary(self) -> None:
        """Print a comprehensive summary of all benchmark results."""
        print(f"\n{self.benchmark_name} - Benchmark Summary:")
        print("=" * (len(self.benchmark_name) + 20))

        if not self.results:
            print("  No results available")
            return

        for variant_name in sorted(self.results.keys()):
            stats = self.get_statistics(variant_name)
            metadata = self.metadata.get(variant_name, {})

            print(f"\n  {variant_name}:")
            print(f"    Iterations: {metadata.get('iterations', 'unknown')}")
            print(f"    Mean: {stats.mean:.3f}s")
            print(f"    Median: {stats.median:.3f}s")
            print(f"    Min/Max: {stats.min_time:.3f}s / {stats.max_time:.3f}s")
            print(f"    Std Dev: {stats.std_dev:.3f}s")
            print(f"    P95: {stats.p95:.3f}s")


@contextmanager
def temporary_performance_threshold(
    operation_name: str, max_time: float, error_message: str | None = None
):
    """
    Context manager to enforce performance thresholds for operations.

    Replaces patterns like:
        start_time = time.time()
        operation()
        duration = time.time() - start_time
        assert duration < max_acceptable_time, f"Operation too slow: {duration:.3f}s"

    Args:
        operation_name: Name of the operation for error reporting
        max_time: Maximum acceptable time in seconds
        error_message: Optional custom error message

    Raises:
        AssertionError: If operation exceeds the time threshold

    Examples:
        with temporary_performance_threshold("embedding_batch", 1.0):
            embeddings = model.encode(large_text_batch)

        with temporary_performance_threshold("model_loading", 5.0, "Model loading too slow for CI"):
            model = SentenceTransformer("all-MiniLM-L6-v2")
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time

        if duration > max_time:
            if error_message is None:
                error_message = (
                    f"{operation_name} exceeded performance threshold: "
                    f"{duration:.3f}s > {max_time:.3f}s"
                )
            raise AssertionError(error_message)


def parallel_execution_timer(
    operations: dict[str, TimingFunction],
    max_concurrent: int = 10,
    print_results: bool = True,
) -> dict[str, float]:
    """
    Time multiple operations executing in parallel.

    Useful for testing concurrent performance and identifying bottlenecks
    in parallel execution scenarios.

    Args:
        operations: Dictionary mapping operation names to functions
        max_concurrent: Maximum number of concurrent operations
        print_results: Whether to print timing results

    Returns:
        Dictionary mapping operation names to execution times

    Examples:
        operations = {
            "search_text": lambda: search_service.search_text("query"),
            "search_semantic": lambda: search_service.search_semantic("query"),
            "get_spaces": lambda: confluence_service.get_spaces()
        }

        times = parallel_execution_timer(operations, max_concurrent=3)
        print(f"Text search: {times['search_text']:.3f}s")
    """
    import concurrent.futures
    import threading

    results = {}

    def time_operation(name: str, operation: TimingFunction) -> tuple[str, float]:
        start_time = time.time()
        try:
            operation()
        except Exception as e:
            print(f"ERROR in parallel operation '{name}': {e}")
            raise
        finally:
            duration = time.time() - start_time
        return name, duration

    # Execute operations in parallel with timing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_name = {
            executor.submit(time_operation, name, operation): name
            for name, operation in operations.items()
        }

        for future in concurrent.futures.as_completed(future_to_name):
            name, duration = future.result()
            results[name] = duration

            # Log each operation
            log_embedding_operation(f"parallel_{name}", duration)

            if print_results:
                print(f"Parallel operation '{name}' completed in {duration:.3f}s")

    if print_results:
        total_time = sum(results.values())
        max_time = max(results.values())
        print(
            f"Parallel execution summary: {len(operations)} operations, "
            f"total: {total_time:.3f}s, max: {max_time:.3f}s"
        )

    return results


# Compatibility aliases for existing test patterns
PerformanceTimer = PerformanceTracker  # Alias for backward compatibility


# Example usage and testing utilities
if __name__ == "__main__":
    # Example usage demonstrations
    print("Performance Helpers - Example Usage")
    print("=" * 40)

    # Example 1: Decorator pattern
    @performance_tracked("example_operation")
    def example_function():
        time.sleep(0.1)  # Simulate work
        return "result"

    result = example_function()

    # Example 2: Context manager pattern
    with PerformanceTracker("context_example", print_milestones=True) as tracker:
        time.sleep(0.05)
        tracker.log_milestone("step1_complete")
        time.sleep(0.05)
        tracker.log_milestone("step2_complete")

    # Example 3: Benchmark pattern
    benchmark = PerformanceBenchmark("comparison_example")
    benchmark.run_iterations("fast", lambda: time.sleep(0.01), iterations=3)
    benchmark.run_iterations("slow", lambda: time.sleep(0.02), iterations=3)

    improvement = benchmark.compare_results("slow", "fast")
    benchmark.print_summary()

    # Example 4: Statistical analysis
    times = [0.123, 0.145, 0.134, 0.129, 0.141]
    stats = calculate_performance_stats(times)
    print(f"\nStatistical analysis example: {stats}")

    # Example 5: Performance threshold
    try:
        with temporary_performance_threshold("threshold_example", 0.05):
            time.sleep(0.01)  # Should pass
        print("Threshold test passed")
    except AssertionError as e:
        print(f"Threshold test failed: {e}")

    print("\nStored performance metrics:")
    for operation, times in get_performance_metrics().items():
        if times:
            avg_time = sum(times) / len(times)
            print(f"  {operation}: {len(times)} measurements, avg {avg_time:.3f}s")
