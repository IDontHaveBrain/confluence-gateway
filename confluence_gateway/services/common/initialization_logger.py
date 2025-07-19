"""
Standardized initialization logging utilities for Confluence Gateway services.

This module provides consistent logging patterns for service initialization,
component availability, configuration status, and dependency validation.
"""

import logging
from typing import Any


class InitializationLogger:
    """
    Standardized logging utility for service initialization patterns.

    This class consolidates the 11+ initialization logging patterns found across
    services like SearchService, GenerationService, and IndexingService.
    """

    @staticmethod
    def log_component_availability(
        service_name: str,
        component_name: str,
        is_available: bool,
        logger: logging.Logger,
        impact_message: str | None = None,
        success_message: str | None = None,
    ) -> None:
        """
        Log component availability in standardized "with/without" pattern.

        Used for patterns like:
        - "SearchService initialized with HybridSearchStrategy."
        - "SearchService initialized WITHOUT HybridSearchStrategy. Hybrid search will be disabled."

        Args:
            service_name: Name of the service being initialized
            component_name: Name of the component being checked
            is_available: Whether the component is available
            logger: Logger instance to use
            impact_message: Additional message about impact when component is missing
            success_message: Custom success message (defaults to standard pattern)
        """
        if is_available:
            message = (
                success_message or f"{service_name} initialized with {component_name}."
            )
            logger.info(message)
        else:
            base_message = f"{service_name} initialized WITHOUT {component_name}."
            if impact_message:
                message = f"{base_message} {impact_message}"
            else:
                message = base_message
            logger.warning(message)

    @staticmethod
    def log_service_configuration(
        service_name: str,
        config: Any,
        logger: logging.Logger,
        enabled_check_attr: str | None = "enable",
        config_details: dict[str, Any] | None = None,
        disabled_message: str | None = None,
        no_config_message: str | None = None,
    ) -> None:
        """
        Log service configuration status with optional details.

        Used for patterns like:
        - "GenerationService initialized. Provider: litellm, Model: gpt-4"
        - "GenerationService initialized, but RAG generation is disabled in config."
        - "GenerationService initialized WITHOUT configuration. RAG generation disabled."

        Args:
            service_name: Name of the service being initialized
            config: Configuration object to check
            logger: Logger instance to use
            enabled_check_attr: Attribute name to check if service is enabled
            config_details: Additional config details to include in success message
            disabled_message: Custom message when service is disabled
            no_config_message: Custom message when config is missing
        """
        if config is None:
            message = (
                no_config_message
                or f"{service_name} initialized WITHOUT configuration."
            )
            logger.warning(message)
            return

        # Check if service is enabled (if applicable)
        is_enabled = True
        if enabled_check_attr and hasattr(config, enabled_check_attr):
            is_enabled = getattr(config, enabled_check_attr, True)

        if is_enabled:
            if config_details:
                details_str = ", ".join(
                    [f"{k}: {v}" for k, v in config_details.items()]
                )
                message = f"{service_name} initialized. {details_str}"
            else:
                message = f"{service_name} initialized successfully."
            logger.info(message)
        else:
            message = (
                disabled_message
                or f"{service_name} initialized, but is disabled in config."
            )
            logger.info(message)

    @staticmethod
    def log_initialization_success(
        component_name: str,
        logger: logging.Logger,
        additional_details: str | None = None,
    ) -> None:
        """
        Log simple initialization success message.

        Used for patterns like:
        - "TextProcessor initialized successfully"
        - "EmbeddingManager initialized successfully"

        Args:
            component_name: Name of the component that was initialized
            logger: Logger instance to use
            additional_details: Optional additional details to append
        """
        message = f"{component_name} initialized successfully"
        if additional_details:
            message = f"{message}: {additional_details}"
        logger.info(message)

    @staticmethod
    def log_dependency_status(
        component_name: str,
        dependencies: dict[str, bool],
        logger: logging.Logger,
        success_message: str | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """
        Log detailed dependency status for component initialization.

        Used for patterns like:
        - "EmbeddingManager could not be initialized due to missing dependencies.
           embedding_service=True, vector_db_adapter=False, vector_db_config=True"

        Args:
            component_name: Name of the component being initialized
            dependencies: Dict mapping dependency names to availability status
            logger: Logger instance to use
            success_message: Custom success message when all dependencies available
            failure_reason: Additional context for why initialization failed
        """
        all_available = all(dependencies.values())

        if all_available:
            message = success_message or f"{component_name} initialized successfully"
            logger.info(message)
        else:
            dependency_status = ", ".join(
                [f"{name}={status}" for name, status in dependencies.items()]
            )
            base_message = f"{component_name} could not be initialized due to missing dependencies."
            if failure_reason:
                message = f"{base_message} {failure_reason} {dependency_status}"
            else:
                message = f"{base_message} {dependency_status}"
            logger.warning(message)

    @staticmethod
    def log_configuration_details(
        component_name: str,
        config_details: dict[str, Any],
        logger: logging.Logger,
        prefix_message: str | None = None,
    ) -> None:
        """
        Log detailed configuration information for a component.

        Used for patterns like:
        - "Initialized SentenceSplitter using adapter's config: chunk_size=512, chunk_overlap=50"
        - "IndexingService initialized with provided Vector DB Adapter: Type='qdrant'"

        Args:
            component_name: Name of the component being configured
            config_details: Dict of configuration key-value pairs
            logger: Logger instance to use
            prefix_message: Optional prefix for the log message
        """
        details_str = ", ".join([f"{k}={v}" for k, v in config_details.items()])

        if prefix_message:
            message = f"{prefix_message}: {details_str}"
        else:
            message = f"{component_name} configured with: {details_str}"

        logger.info(message)

    @staticmethod
    def log_initialization_summary(
        service_name: str,
        enabled_features: list[str],
        disabled_features: list[str],
        logger: logging.Logger,
        configuration_summary: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a comprehensive initialization summary for a service.

        Provides a high-level overview of what was successfully initialized
        and what features are disabled.

        Args:
            service_name: Name of the service
            enabled_features: List of successfully enabled features/components
            disabled_features: List of disabled or unavailable features/components
            logger: Logger instance to use
            configuration_summary: Optional dict of key configuration details
        """
        # Log main initialization
        logger.info(f"{service_name} initialization complete")

        # Log enabled features
        if enabled_features:
            features_str = ", ".join(enabled_features)
            logger.info(f"  Enabled features: {features_str}")

        # Log disabled features
        if disabled_features:
            features_str = ", ".join(disabled_features)
            logger.warning(f"  Disabled features: {features_str}")

        # Log configuration summary
        if configuration_summary:
            config_str = ", ".join(
                [f"{k}={v}" for k, v in configuration_summary.items()]
            )
            logger.info(f"  Configuration: {config_str}")
