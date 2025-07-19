import datetime
import importlib.metadata
import logging
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from confluence_gateway.api.dependencies import get_health_status
from confluence_gateway.api.routes import api_router

logger = logging.getLogger(__name__)


def _get_app_version() -> str:
    try:
        return importlib.metadata.version("confluence-gateway")
    except importlib.metadata.PackageNotFoundError:
        logger.warning(
            "Package 'confluence-gateway' not found in installed packages. "
            "Using fallback version. Consider installing the package with 'pip install -e .'."
        )
        return "0.0.0-dev"


APP_VERSION = _get_app_version()

app = FastAPI(
    title="Confluence Gateway",
    description="API for searching and retrieving Confluence content",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health_check(
    health_status: dict[str, Any] = Depends(get_health_status),
) -> dict[str, Any]:
    """
    Comprehensive health check endpoint.

    Uses the ServiceContainer to check the health of all services
    and provides detailed status information.
    """
    # Add version and timestamp to health status
    health_status["version"] = APP_VERSION
    health_status["timestamp"] = datetime.datetime.now().isoformat()

    return health_status


app.include_router(api_router, prefix="/api")
