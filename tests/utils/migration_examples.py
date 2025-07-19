"""
Migration examples for converting manual timing patterns to performance helpers.

This file demonstrates how to convert the 60+ manual timing blocks found throughout
the test suite to use the new consolidated performance testing utilities.

Run this file directly to see the examples in action:
    python -m tests.utils.migration_examples
"""

import time

from tests.utils.performance_helpers import (
    PerformanceBenchmark,
    PerformanceTracker,
    calculate_performance_stats,
    log_embedding_operation,
    performance_tracked,
    temporary_performance_threshold,
)


def demonstrate_migrations():
    """Demonstrate how to migrate existing timing patterns."""

    print("Performance Helpers - Migration Examples")
    print("=" * 50)

    # MIGRATION 1: Simple timing with logging
    print("\n1. Simple timing with logging:")
    print("-" * 30)

    print("BEFORE (manual pattern found 20+ times):")
    print("""
    start_time = time.time()
    operation()
    duration = time.time() - start_time
    log_embedding_operation("operation_name", duration)
    print(f"Operation completed in {duration:.3f}s")
    """)

    print("AFTER (using @performance_tracked decorator):")
    print("""
    @performance_tracked("operation_name")
    def operation():
        # Your operation here
        pass

    operation()  # Automatic timing and logging
    """)

    # Demonstrate the new pattern
    @performance_tracked("demo_operation")
    def demo_operation():
        time.sleep(0.05)  # Simulate work

    demo_operation()

    # MIGRATION 2: Multi-iteration benchmarking
    print("\n2. Multi-iteration benchmarking:")
    print("-" * 35)

    print("BEFORE (pattern found 8+ times):")
    print("""
    iterations = 5
    standard_times = []
    optimized_times = []

    for _ in range(iterations):
        start_time = time.time()
        standard_operation()
        standard_times.append(time.time() - start_time)

    for _ in range(iterations):
        start_time = time.time()
        optimized_operation()
        optimized_times.append(time.time() - start_time)

    standard_avg = sum(standard_times) / len(standard_times)
    optimized_avg = sum(optimized_times) / len(optimized_times)

    if standard_avg > 0:
        improvement_ratio = standard_avg / optimized_avg
        print(f"Performance improvement: {improvement_ratio:.2f}x faster")
    """)

    print("AFTER (using PerformanceBenchmark):")
    print("""
    benchmark = PerformanceBenchmark("operation_comparison")
    benchmark.run_iterations("standard", standard_operation, iterations=5)
    benchmark.run_iterations("optimized", optimized_operation, iterations=5)
    improvement = benchmark.compare_results("standard", "optimized")
    """)

    # Demonstrate the new pattern
    def standard_operation():
        time.sleep(0.02)

    def optimized_operation():
        time.sleep(0.01)

    benchmark = PerformanceBenchmark("demo_comparison")
    benchmark.run_iterations("standard", standard_operation, iterations=3)
    benchmark.run_iterations("optimized", optimized_operation, iterations=3)
    benchmark.compare_results("standard", "optimized")

    # MIGRATION 3: Context manager with milestones
    print("\n3. Complex operation with breakdown:")
    print("-" * 38)

    print("BEFORE (pattern found 5+ times):")
    print("""
    start_time = time.time()

    data = load_data()
    load_time = time.time() - start_time
    print(f"Data loading: {load_time:.3f}s")

    processed = process_data(data)
    process_time = time.time() - start_time - load_time
    print(f"Data processing: {process_time:.3f}s")

    total_time = time.time() - start_time
    log_embedding_operation("complex_operation", total_time)
    """)

    print("AFTER (using PerformanceTracker):")
    print("""
    with PerformanceTracker("complex_operation", print_milestones=True) as tracker:
        data = load_data()
        tracker.log_milestone("data_loaded")

        processed = process_data(data)
        tracker.log_milestone("data_processed")
    """)

    # Demonstrate the new pattern
    with PerformanceTracker("demo_complex_operation", print_milestones=True) as tracker:
        time.sleep(0.02)  # Simulate data loading
        tracker.log_milestone("data_loaded")

        time.sleep(0.02)  # Simulate data processing
        tracker.log_milestone("data_processed")

    # MIGRATION 4: Performance threshold validation
    print("\n4. Performance threshold validation:")
    print("-" * 37)

    print("BEFORE (pattern found 10+ times):")
    print("""
    start_time = time.time()
    operation()
    duration = time.time() - start_time

    max_acceptable_time = 1.0
    assert duration < max_acceptable_time, (
        f"Operation too slow: {duration:.3f}s"
    )
    """)

    print("AFTER (using temporary_performance_threshold):")
    print("""
    with temporary_performance_threshold("operation", 1.0):
        operation()
    """)

    # Demonstrate the new pattern
    try:
        with temporary_performance_threshold("demo_threshold", 0.1):
            time.sleep(0.02)  # Should pass
        print("✓ Threshold validation passed")
    except AssertionError as e:
        print(f"✗ Threshold validation failed: {e}")

    # MIGRATION 5: Statistical analysis
    print("\n5. Statistical analysis:")
    print("-" * 25)

    print("BEFORE (manual calculations found 5+ times):")
    print("""
    times = [0.123, 0.145, 0.134, 0.129, 0.141]
    avg_time = sum(times) / len(times)
    import statistics
    median_time = statistics.median(times)
    std_dev = statistics.stdev(times)
    print(f"Average: {avg_time:.3f}s, Median: {median_time:.3f}s")
    """)

    print("AFTER (using calculate_performance_stats):")
    print("""
    times = [0.123, 0.145, 0.134, 0.129, 0.141]
    stats = calculate_performance_stats(times)
    print(f"Stats: {stats}")
    """)

    # Demonstrate the new pattern
    times = [0.123, 0.145, 0.134, 0.129, 0.141]
    stats = calculate_performance_stats(times)
    print(f"Demo stats: {stats}")

    print("\n" + "=" * 50)
    print("MIGRATION BENEFITS:")
    print("✓ Consolidated 60+ manual timing blocks into reusable utilities")
    print("✓ Automatic logging integration with existing patterns")
    print("✓ Built-in statistical analysis and comparison capabilities")
    print("✓ Thread-safe performance metrics storage")
    print("✓ Support for both sync and async operations")
    print("✓ Milestone tracking for complex operation breakdowns")
    print("✓ Performance threshold enforcement with clear error messages")
    print("✓ Parallel execution timing capabilities")
    print("=" * 50)


def show_real_world_migration_examples():
    """Show specific examples from actual test files."""

    print("\nREAL-WORLD MIGRATION EXAMPLES")
    print("=" * 50)

    print("\nFrom test_performance_validation.py:")
    print("-" * 40)

    print("BEFORE (lines 35-48):")
    print("""
    start_time = time.time()
    model = shared_sentence_transformer_model
    access_time = time.time() - start_time

    assert model is not None
    assert access_time < 0.1, (
        f"Shared model access took too long: {access_time:.3f}s"
    )

    log_embedding_operation("shared_model_access", access_time)
    print(f"Shared model access completed in {access_time:.6f}s")
    """)

    print("AFTER (using performance utilities):")
    print("""
    with temporary_performance_threshold("shared_model_access", 0.1):
        with PerformanceTracker("shared_model_access") as tracker:
            model = shared_sentence_transformer_model
            assert model is not None
    """)

    print("\nFrom test_performance_validation.py:")
    print("-" * 40)

    print("BEFORE (lines 70-85, 87-104):")
    print("""
    iterations = 5
    standard_times = []

    for _ in range(iterations):
        start_time = time.time()
        standard_provider = SentenceTransformerProvider(config)
        standard_provider.initialize()
        standard_provider.embed_text("test text")
        standard_times.append(time.time() - start_time)

    optimized_times = []
    for _ in range(iterations):
        optimized_start = time.time()
        optimized_provider = SentenceTransformerProvider(config)
        inject_shared_model_into_provider(optimized_provider, shared_model)
        optimized_provider.embed_text("test text")
        optimized_times.append(time.time() - optimized_start)

    standard_avg_time = sum(standard_times) / len(standard_times)
    optimized_avg_time = sum(optimized_times) / len(optimized_times)

    # Log performance metrics
    log_embedding_operation("standard_provider_with_usage", standard_avg_time)
    log_embedding_operation("optimized_provider_with_usage", optimized_avg_time)

    if standard_avg_time > 0:
        improvement_ratio = standard_avg_time / optimized_avg_time
        print(f"Performance improvement: {improvement_ratio:.2f}x faster")
    """)

    print("AFTER (using PerformanceBenchmark):")
    print("""
    def create_standard_provider():
        provider = SentenceTransformerProvider(config)
        provider.initialize()
        provider.embed_text("test text")
        return provider

    def create_optimized_provider():
        provider = SentenceTransformerProvider(config)
        inject_shared_model_into_provider(provider, shared_model)
        provider.embed_text("test text")
        return provider

    benchmark = PerformanceBenchmark("provider_comparison")
    benchmark.run_iterations("standard", create_standard_provider, iterations=5)
    benchmark.run_iterations("optimized", create_optimized_provider, iterations=5)
    improvement = benchmark.compare_results("standard", "optimized")
    """)

    print("\nFrom test_shared_embedding_optimization.py:")
    print("-" * 45)

    print("BEFORE (lines 110-120):")
    print("""
    total_shared_time = 0.0
    for i, text in enumerate(test_texts, 1):
        start_time = time.time()
        _ = shared_sentence_transformer_model.encode([text])
        duration = time.time() - start_time
        total_shared_time += duration
        print(f"Text {i}: {duration:.4f}s")

    avg_time_per_text = total_shared_time / len(test_texts)
    print(f"Average time per text: {avg_time_per_text:.4f}s")
    """)

    print("AFTER (using PerformanceBenchmark):")
    print("""
    benchmark = PerformanceBenchmark("text_encoding")

    for i, text in enumerate(test_texts, 1):
        benchmark.run_iterations(f"text_{i}",
                               lambda t=text: shared_sentence_transformer_model.encode([t]),
                               iterations=1)

    # Get comprehensive statistics
    for variant in benchmark.get_variants():
        stats = benchmark.get_statistics(variant)
        print(f"{variant}: {stats.mean:.4f}s")
    """)


if __name__ == "__main__":
    # Run the migration demonstrations
    demonstrate_migrations()
    show_real_world_migration_examples()

    print("\nTo use these utilities in your tests:")
    print("from tests.utils import (")
    print("    performance_tracked,")
    print("    PerformanceTracker,")
    print("    PerformanceBenchmark,")
    print("    temporary_performance_threshold")
    print(")")
