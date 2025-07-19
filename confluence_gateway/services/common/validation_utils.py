"""
Consolidated dependency validation utilities for services.

Provides reusable functions to eliminate duplication across 24+ validation
checks found in services layer. Handles service availability checking,
logging, and dependency validation patterns.
"""

import logging
from collections.abc import Callable
from typing import Any, TypeVar

# Type variable for generic service types
T = TypeVar("T")
ServiceType = Any | None


class ValidationUtils:
    """
    Utility class for consolidating dependency validation patterns across services.

    Provides unified methods for:
    - Single service availability validation with logging
    - Multiple dependency validation with detailed reporting
    - Service initialization status logging
    - Conditional component validation
    """

    @staticmethod
    def validate_service_dependency(
        service: ServiceType,
        service_name: str,
        logger: logging.Logger,
        *,
        required: bool = False,
        operation_name: str | None = None,
        disable_message: str | None = None,
    ) -> bool:
        """
        Validate single service dependency with appropriate logging.

        Args:
            service: Service instance to validate (can be None)
            service_name: Human-readable name of the service for logging
            logger: Logger instance for output
            required: Whether service is required (affects log level)
            operation_name: Optional operation that will be affected if missing
            disable_message: Custom message when service is disabled

        Returns:
            True if service is available, False otherwise

        Examples:
            # Basic validation
            ValidationUtils.validate_service_dependency(
                self.embedding_service, "EmbeddingService", logger
            )

            # Required service with operation context
            ValidationUtils.validate_service_dependency(
                self.vector_db_adapter, "VectorDBAdapter", logger,
                required=True, operation_name="semantic search"
            )
        """
        if service is not None:
            info_msg = f"{service_name} initialized successfully."
            if operation_name:
                info_msg = (
                    f"{service_name} initialized. {operation_name.title()} enabled."
                )
            logger.info(info_msg)
            return True
        else:
            # Determine log level based on whether service is required
            log_level = logging.ERROR if required else logging.WARNING

            base_msg = f"{service_name} not available."
            if disable_message:
                warning_msg = f"{base_msg} {disable_message}"
            elif operation_name:
                warning_msg = f"{base_msg} {operation_name.title()} will be disabled."
            else:
                warning_msg = f"{base_msg} Related functionality will be disabled."

            logger.log(log_level, warning_msg)
            return False

    @staticmethod
    def log_service_availability(
        service: ServiceType,
        service_name: str,
        logger: logging.Logger,
        *,
        operation_name: str | None = None,
    ) -> bool:
        """
        Log service availability without validation (for informational logging).

        Args:
            service: Service instance to check
            service_name: Human-readable name of the service
            logger: Logger instance
            operation_name: Optional operation context

        Returns:
            True if service is available, False otherwise
        """
        return ValidationUtils.validate_service_dependency(
            service, service_name, logger, operation_name=operation_name
        )

    @staticmethod
    def check_multiple_dependencies(
        dependencies: dict[str, ServiceType],
        logger: logging.Logger,
        *,
        operation_name: str | None = None,
        all_required: bool = True,
    ) -> dict[str, bool]:
        """
        Validate multiple dependencies with detailed availability reporting.

        Args:
            dependencies: Dict mapping service names to service instances
            logger: Logger instance
            operation_name: Optional operation that requires these dependencies
            all_required: Whether all dependencies are required for the operation

        Returns:
            Dict mapping service names to availability status (True/False)

        Example:
            dependencies = {
                "embedding_service": self.embedding_service,
                "vector_db_adapter": self.vector_db_adapter,
                "vector_db_config": self.vector_db_config,
            }
            availability = ValidationUtils.check_multiple_dependencies(
                dependencies, logger, operation_name="embedding operations"
            )
        """
        availability = {}
        available_count = 0

        for service_name, service in dependencies.items():
            is_available = service is not None
            availability[service_name] = is_available
            if is_available:
                available_count += 1

        # Log overall status
        total_deps = len(dependencies)
        if available_count == total_deps:
            success_msg = (
                f"All dependencies available for {operation_name or 'operation'}"
            )
            logger.info(success_msg)
        elif available_count == 0:
            failure_msg = (
                f"No dependencies available for {operation_name or 'operation'}"
            )
            log_level = logging.ERROR if all_required else logging.WARNING
            logger.log(log_level, failure_msg)
        else:
            partial_msg = (
                f"Partial dependencies available for {operation_name or 'operation'}: "
                f"{available_count}/{total_deps} services available"
            )
            log_level = logging.WARNING if all_required else logging.INFO
            logger.log(log_level, partial_msg)

        # Log detailed availability status
        status_parts = []
        for service_name, is_available in availability.items():
            status_parts.append(f"{service_name}={is_available}")

        detailed_msg = f"Dependency status: {', '.join(status_parts)}"
        logger.debug(detailed_msg)

        return availability

    @staticmethod
    def validate_parser_initialization(
        parser_factory: Callable[..., T],
        parser_name: str,
        logger: logging.Logger,
        *,
        factory_args: tuple[Any, ...] | None = None,
        factory_kwargs: dict[str, Any] | None = None,
        content_type: str | None = None,
    ) -> T | None:
        """
        Safely initialize a parser with error handling and logging.

        Args:
            parser_factory: Factory function to create the parser
            parser_name: Name of the parser for logging
            logger: Logger instance
            factory_args: Positional arguments for factory function
            factory_kwargs: Keyword arguments for factory function
            content_type: Type of content parser handles (e.g., "HTML", "attachment")

        Returns:
            Parser instance if successful, None if failed

        Example:
            self.html_parser = ValidationUtils.validate_parser_initialization(
                get_parser,
                self.indexing_config.html_parser,
                logger,
                factory_kwargs={
                    "parser_name": self.indexing_config.html_parser,
                    "content_category": "html"
                },
                content_type="HTML"
            )
        """
        args = factory_args or ()
        kwargs = factory_kwargs or {}
        content_desc = f"{content_type} " if content_type else ""

        try:
            parser = parser_factory(*args, **kwargs)
            logger.info(f"Successfully initialized {content_desc}parser: {parser_name}")
            return parser
        except Exception as e:
            logger.warning(
                f"Could not initialize {content_desc}parser '{parser_name}': {e}. "
                f"{content_desc}content parsing will be disabled."
            )
            return None

    @staticmethod
    def validate_conditional_initialization(
        dependencies: dict[str, ServiceType],
        component_name: str,
        logger: logging.Logger,
        *,
        all_required: bool = True,
    ) -> bool:
        """
        Check if all required dependencies are available for component initialization.

        Args:
            dependencies: Dict mapping dependency names to service instances
            component_name: Name of component being initialized
            logger: Logger instance
            all_required: Whether all dependencies are required

        Returns:
            True if component can be initialized, False otherwise

        Example:
            can_init = ValidationUtils.validate_conditional_initialization(
                {
                    "embedding_service": self.embedding_service,
                    "vector_db_adapter": self.vector_db_adapter,
                    "vector_db_config": self.vector_db_config,
                },
                "EmbeddingManager",
                logger
            )
            if can_init:
                self.embedding_manager = EmbeddingManager(...)
        """
        availability = ValidationUtils.check_multiple_dependencies(
            dependencies, logger, operation_name=f"{component_name} initialization"
        )

        if all_required:
            can_initialize = all(availability.values())
        else:
            can_initialize = any(availability.values())

        if can_initialize:
            logger.info(
                f"{component_name} dependencies satisfied - initializing component"
            )
        else:
            missing_deps = [
                name for name, available in availability.items() if not available
            ]
            logger.warning(
                f"{component_name} cannot be initialized due to missing dependencies: "
                f"{', '.join(missing_deps)}"
            )

        return can_initialize

    @staticmethod
    def log_initialization_summary(
        service_name: str,
        dependencies: dict[str, ServiceType],
        logger: logging.Logger,
        *,
        additional_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a comprehensive initialization summary for a service.

        Args:
            service_name: Name of the service being initialized
            dependencies: Dict mapping dependency names to availability
            logger: Logger instance
            additional_info: Optional additional information to include

        Example:
            ValidationUtils.log_initialization_summary(
                "SearchService",
                {
                    "IndexingService": self.indexing_service,
                    "EmbeddingService": self.embedding_service,
                    "VectorDBAdapter": self.vector_db_adapter,
                },
                logger,
                additional_info={"hybrid_search_enabled": self.hybrid_search_strategy is not None}
            )
        """
        available_deps = [
            name for name, service in dependencies.items() if service is not None
        ]
        missing_deps = [
            name for name, service in dependencies.items() if service is None
        ]

        summary_parts = [f"{service_name} initialization complete."]

        if available_deps:
            summary_parts.append(f"Available: {', '.join(available_deps)}")

        if missing_deps:
            summary_parts.append(f"Missing: {', '.join(missing_deps)}")

        if additional_info:
            info_parts = [f"{k}={v}" for k, v in additional_info.items()]
            summary_parts.append(f"Status: {', '.join(info_parts)}")

        logger.info(" | ".join(summary_parts))
