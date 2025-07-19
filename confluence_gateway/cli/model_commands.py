import typer

from confluence_gateway.cli.common import handle_cli_errors, print_status
from confluence_gateway.core.config import (
    get_embedding_config,
    get_vector_db_config,
)

app = typer.Typer(
    help="Check embedding model configuration and compatibility.",
    no_args_is_help=True,
)


@app.command("status")
@handle_cli_errors
def model_status() -> None:
    """Show current embedding model configuration."""
    embedding_config = get_embedding_config()
    vector_db_config = get_vector_db_config()

    if not embedding_config or not vector_db_config:
        print_status("Embedding or Vector DB not configured.", "error")
        return

    if embedding_config.provider == "none" or vector_db_config.type == "none":
        print_status("Embedding or Vector DB features are disabled.", "info")
        return

    print_status("Current Model Configuration:", "info")
    print(f"  Provider: {embedding_config.provider}")
    print(f"  Model: {embedding_config.model_name}")
    print(f"  Dimension: {embedding_config.dimension}")
    print(f"  Device: {embedding_config.device or 'auto'}")
    print(f"  Collection: {vector_db_config.get_effective_collection_name()}")


@app.command("info")
@handle_cli_errors
def model_info() -> None:
    """Show embedding model capabilities and configuration details."""
    embedding_config = get_embedding_config()
    vector_db_config = get_vector_db_config()

    if not embedding_config or not vector_db_config:
        print_status("Embedding or Vector DB not configured.", "error")
        return

    if embedding_config.provider == "none" or vector_db_config.type == "none":
        print_status("Embedding or Vector DB features are disabled.", "info")
        return

    print_status("Model Configuration Details:", "info")
    print(f"  Provider: {embedding_config.provider}")
    print(f"  Model: {embedding_config.model_name}")
    print(f"  Dimension: {embedding_config.dimension}")
    print(f"  Device: {embedding_config.device or 'auto'}")
    print(f"  Vector DB: {vector_db_config.type}")
    print(f"  Collection: {vector_db_config.get_effective_collection_name()}")

    # Basic capability info
    print("\nCapabilities:")
    if embedding_config.provider == "sentence-transformers":
        print("  - Local embedding generation")
        print("  - GPU acceleration (if available)")
    elif embedding_config.provider == "litellm":
        print("  - Remote API embedding generation")
        print("  - Multiple provider support")


@app.command("validate")
@handle_cli_errors
def validate_configuration() -> None:
    """Validate embedding model configuration for basic compatibility."""
    embedding_config = get_embedding_config()
    vector_db_config = get_vector_db_config()

    if not embedding_config or not vector_db_config:
        print_status("Embedding or Vector DB not configured.", "error")
        return

    if embedding_config.provider == "none" or vector_db_config.type == "none":
        print_status("Embedding or Vector DB features are disabled.", "info")
        return

    print_status("Validating model configuration...", "info")

    errors = []
    warnings = []

    # Basic required field validation
    if not embedding_config.model_name:
        errors.append(
            f"Model name is required for {embedding_config.provider} provider"
        )

    if embedding_config.dimension is None:
        errors.append("Embedding dimension must be specified")

    # Dimension compatibility check
    if (
        embedding_config.dimension
        and vector_db_config.embedding_dimension
        and embedding_config.dimension != vector_db_config.embedding_dimension
    ):
        warnings.append(
            f"Dimension mismatch: embedding ({embedding_config.dimension}) != "
            f"vector DB ({vector_db_config.embedding_dimension})"
        )

    # Provider-specific validation
    if embedding_config.provider == "litellm":
        if embedding_config.model_name and embedding_config.model_name.startswith(
            "ollama/"
        ):
            if not embedding_config.litellm_api_base:
                errors.append("LITELLM_API_BASE must be set for ollama models")

    # Report results
    if errors:
        print_status("Validation failed:", "error")
        for error in errors:
            print(f"  ✗ {error}")

    if warnings:
        print_status("Warnings:", "warning")
        for warning in warnings:
            print(f"  ⚠ {warning}")

    if not errors and not warnings:
        print_status("Configuration is valid.", "success")
    elif not errors:
        print_status("Configuration is valid with warnings.", "warning")
