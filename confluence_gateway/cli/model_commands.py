import json
import logging
from typing import Any

import typer

from confluence_gateway.cli.common import handle_cli_errors, print_status
from confluence_gateway.core.config import (
    get_embedding_config,
    get_model_change_info,
    get_vector_db_config,
    list_all_model_metadata,
    load_model_metadata,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Manage embedding models and track configuration changes.",
    no_args_is_help=True,
)


@app.command("list")
@handle_cli_errors
def list_models() -> None:
    """List all collections and their associated embedding models."""
    print_status("Loading model metadata...", "info")

    all_metadata = list_all_model_metadata()

    if not all_metadata:
        print_status("No model metadata found.", "info")
        print_status(
            "Models are tracked automatically when indexing is performed.", "dim"
        )
        return

    # Create output data
    output_data: dict[str, list[dict[str, Any]]] = {"collections": []}

    for collection_name, metadata in all_metadata.items():
        collection_info = {
            "collection_name": collection_name,
            "provider": metadata.provider,
            "model_name": metadata.model_name,
            "dimension": metadata.dimension,
            "device": metadata.device,
            "created_at": metadata.created_at.isoformat(),
            "configuration_hash": metadata.configuration_hash,
        }
        output_data["collections"].append(collection_info)

    # Sort by collection name for consistent output
    output_data["collections"].sort(key=lambda x: x["collection_name"])

    print(json.dumps(output_data, indent=2))


@app.command("status")
@handle_cli_errors
def model_status() -> None:
    """Check current model configuration and detect any changes."""
    print_status("Checking model configuration status...", "info")

    # Get current configurations
    embedding_config = get_embedding_config()
    vector_db_config = get_vector_db_config()
    model_change_info = get_model_change_info()

    if not embedding_config or not vector_db_config:
        print_status("Embedding or Vector DB not configured.", "error")
        return

    if embedding_config.provider == "none" or vector_db_config.type == "none":
        print_status("Embedding or Vector DB features are disabled.", "info")
        return

    # Load current metadata for the collection
    effective_collection_name = vector_db_config.get_effective_collection_name()
    current_metadata = load_model_metadata(effective_collection_name)

    output_data = {
        "current_configuration": {
            "collection_name": effective_collection_name,
            "provider": embedding_config.provider,
            "model_name": embedding_config.model_name,
            "dimension": embedding_config.dimension,
            "device": embedding_config.device,
        },
        "metadata_exists": current_metadata is not None,
        "changes_detected": model_change_info is not None,
    }

    if current_metadata:
        output_data["stored_metadata"] = {
            "provider": current_metadata.provider,
            "model_name": current_metadata.model_name,
            "dimension": current_metadata.dimension,
            "device": current_metadata.device,
            "created_at": current_metadata.created_at.isoformat(),
            "configuration_hash": current_metadata.configuration_hash,
        }

    if model_change_info:
        output_data["model_changes"] = {
            "change_type": model_change_info.change_type,
            "migration_required": model_change_info.migration_required,
            "warning_message": model_change_info.warning_message,
            "migration_guidance": model_change_info.migration_guidance,
        }

    print(json.dumps(output_data, indent=2))


@app.command("info")
@handle_cli_errors
def model_info(
    collection_name: str = typer.Argument(
        help="Name of the collection to get model information for"
    ),
) -> None:
    """Get detailed model information for a specific collection."""
    print_status(
        f"Loading model information for collection '{collection_name}'...", "info"
    )

    metadata = load_model_metadata(collection_name)

    if not metadata:
        print_status(
            f"No model metadata found for collection '{collection_name}'.", "error"
        )
        print_status(
            "Check that the collection name is correct and that indexing has been performed.",
            "dim",
        )
        return

    output_data = {
        "collection_name": metadata.collection_name,
        "provider": metadata.provider,
        "model_name": metadata.model_name,
        "dimension": metadata.dimension,
        "device": metadata.device,
        "created_at": metadata.created_at.isoformat(),
        "configuration_hash": metadata.configuration_hash,
    }

    print(json.dumps(output_data, indent=2))


@app.command("validate")
@handle_cli_errors
def validate_configuration() -> None:
    """Validate current embedding model configuration and check for potential issues."""
    print_status("Validating embedding model configuration...", "info")

    embedding_config = get_embedding_config()
    vector_db_config = get_vector_db_config()

    if not embedding_config or not vector_db_config:
        print_status("Embedding or Vector DB not configured.", "error")
        return

    if embedding_config.provider == "none" or vector_db_config.type == "none":
        print_status("Embedding or Vector DB features are disabled.", "info")
        return

    validation_results: dict[str, Any] = {
        "configuration_valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": [],
    }

    # Check provider-specific configurations
    if embedding_config.provider == "sentence-transformers":
        if not embedding_config.model_name:
            validation_results["errors"].append(
                "Model name is required for sentence-transformers provider"
            )
            validation_results["configuration_valid"] = False

        if embedding_config.dimension is None:
            validation_results["errors"].append("Embedding dimension must be specified")
            validation_results["configuration_valid"] = False

        if embedding_config.device is None:
            validation_results["recommendations"].append(
                "Consider specifying a device (cpu/cuda) for better performance"
            )

    elif embedding_config.provider == "litellm":
        if not embedding_config.model_name:
            validation_results["errors"].append(
                "Model name is required for litellm provider"
            )
            validation_results["configuration_valid"] = False

        if embedding_config.dimension is None:
            validation_results["errors"].append("Embedding dimension must be specified")
            validation_results["configuration_valid"] = False

        if embedding_config.model_name and embedding_config.model_name.startswith(
            "ollama/"
        ):
            if not embedding_config.litellm_api_base:
                validation_results["errors"].append(
                    "LITELLM_API_BASE must be set for ollama models"
                )
                validation_results["configuration_valid"] = False

    # Check vector DB dimension consistency
    if (
        embedding_config.dimension
        and vector_db_config.embedding_dimension
        and embedding_config.dimension != vector_db_config.embedding_dimension
    ):
        validation_results["warnings"].append(
            f"Embedding dimension ({embedding_config.dimension}) differs from "
            f"vector DB dimension ({vector_db_config.embedding_dimension})"
        )

    # Check for model changes
    model_change_info = get_model_change_info()
    if model_change_info:
        if model_change_info.migration_required:
            validation_results["warnings"].append(
                f"Model change detected: {model_change_info.warning_message}"
            )
            validation_results["recommendations"].append(
                model_change_info.migration_guidance
            )

    print(json.dumps(validation_results, indent=2))
