import datetime
import importlib.metadata
import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.api.dependencies import get_confluence_client
from confluence_gateway.api.routes import api_router
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
)

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
def health_check(client: ConfluenceClient = Depends(get_confluence_client)):
    health_info = {
        "status": "ok",
        "version": APP_VERSION,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    try:
        client.test_connection()
        health_info["confluence_connection"] = "ok"
    except ConfluenceConnectionError as e:
        health_info["status"] = "degraded"
        health_info["confluence_connection"] = "error"
        health_info["confluence_error"] = str(e)
    except ConfluenceAuthenticationError as e:
        health_info["status"] = "degraded"
        health_info["confluence_connection"] = "authentication_error"
        health_info["confluence_error"] = str(e)
    except ConfluenceAPIError as e:
        health_info["status"] = "degraded"
        health_info["confluence_connection"] = "api_error"
        health_info["confluence_error"] = f"{e.error_message} (Status: {e.status_code})"
    except Exception as e:
        health_info["status"] = "degraded"
        health_info["confluence_connection"] = "unknown_error"
        health_info["confluence_error"] = str(e)

    return health_info


app.include_router(api_router, prefix="/api")
