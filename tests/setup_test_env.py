#!/usr/bin/env python3
"""
Test Environment Setup Verification Script

This script helps verify that the test environment is properly configured
for running Confluence Gateway E2E tests.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_environment_variables() -> list[tuple[str, bool, str]]:
    """Check required environment variables"""
    required_vars = [
        ("CONFLUENCE_URL", "https://your-instance.atlassian.net"),
        ("CONFLUENCE_USERNAME", "your-email@example.com"),
        ("CONFLUENCE_API_TOKEN", "your-api-token"),
    ]

    optional_vars = [
        ("GENERATION_MODEL_NAME", "openrouter/google/gemini-2.5-flash"),
        ("GENERATION_LITELLM_API_KEY", "your-openrouter-api-key"),
    ]

    results = []
    print("🔍 Checking Environment Variables...")

    for var, example in required_vars:
        value = os.getenv(var)
        is_set = value is not None and value.strip() != ""
        status = "✅" if is_set else "❌"
        print(f"  {status} {var}: {'SET' if is_set else 'NOT SET'}")
        if not is_set:
            print(f'     Example: export {var}="{example}"')
        results.append((var, is_set, "required"))

    print("\n🔍 Checking Optional Environment Variables...")
    for var, example in optional_vars:
        value = os.getenv(var)
        is_set = value is not None and value.strip() != ""
        status = "✅" if is_set else "⚠️"
        print(f"  {status} {var}: {'SET' if is_set else 'NOT SET (optional)'}")
        if not is_set:
            print(f'     Example: export {var}="{example}"')
        results.append((var, is_set, "optional"))

    return results


def check_dependencies() -> bool:
    """Check that required dependencies are installed"""
    print("\n🔍 Checking Dependencies...")

    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "--version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ pytest: {result.stdout.strip()}")
            return True
        else:
            print("  ❌ pytest: Not available")
            return False
    except Exception as e:
        print(f"  ❌ pytest: Error checking - {e}")
        return False


def check_test_imports() -> bool:
    """Check that test modules can be imported"""
    print("\n🔍 Checking Test Module Imports...")

    # Add current directory to sys.path for importing tests
    current_dir = Path(__file__).resolve().parent
    project_root = str(current_dir.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    test_modules = [
        "tests.conftest",
        "tests.cli.test_spaces_e2e",
        "tests.cli.test_search_e2e",
        "tests.cli.test_index_e2e",
        "tests.cli.test_generate_e2e",
        "tests.api.test_health_e2e",
        "tests.api.test_spaces_api_e2e",
        "tests.api.test_search_api_e2e",
        "tests.api.test_index_api_e2e",
        "tests.api.test_generate_api_e2e",
    ]

    all_good = True
    for module in test_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module}: {e}")
            all_good = False

    return all_good


def check_test_collection() -> bool:
    """Check that pytest can collect all tests"""
    print("\n🔍 Checking Test Collection...")

    try:
        result = subprocess.run(
            ["uv", "run", "pytest", "tests/", "--collect-only", "-q"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if "collected" in line:
                    print(f"  ✅ {line}")
                    return True
            print("  ⚠️ Tests collected but couldn't parse count")
            return True
        else:
            print(f"  ❌ Test collection failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Test collection error: {e}")
        return False


def main():
    """Main verification function"""
    print("🚀 Confluence Gateway E2E Test Environment Verification")
    print("=" * 60)

    # Check environment variables
    env_results = check_environment_variables()

    # Check dependencies
    deps_ok = check_dependencies()

    # Check test imports
    imports_ok = check_test_imports()

    # Check test collection
    collection_ok = check_test_collection()

    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)

    # Environment variables summary
    required_env_ok = all(
        is_set for var, is_set, req_type in env_results if req_type == "required"
    )
    print(
        f"Environment Variables: {'✅ READY' if required_env_ok else '❌ MISSING REQUIRED'}"
    )
    print(f"Dependencies: {'✅ READY' if deps_ok else '❌ FAILED'}")
    print(f"Test Imports: {'✅ READY' if imports_ok else '❌ FAILED'}")
    print(f"Test Collection: {'✅ READY' if collection_ok else '❌ FAILED'}")

    overall_ready = required_env_ok and deps_ok and imports_ok and collection_ok

    print(
        f"\nOverall Status: {'🎉 READY FOR TESTING' if overall_ready else '⚠️ SETUP REQUIRED'}"
    )

    if not overall_ready:
        print("\n📝 NEXT STEPS:")
        if not required_env_ok:
            print("1. Set required environment variables (see examples above)")
        if not deps_ok:
            print("2. Install dependencies: uv sync --dev")
        if not imports_ok:
            print("3. Fix test import issues")
        if not collection_ok:
            print("4. Fix test collection issues")

        print("\n📚 For more information, see:")
        print("   - docs/testing-implementation-plan.md")
        print("   - CLAUDE.md (Testing Strategy section)")

    return 0 if overall_ready else 1


if __name__ == "__main__":
    sys.exit(main())
