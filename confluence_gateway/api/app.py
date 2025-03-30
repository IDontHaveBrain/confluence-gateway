import datetime

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

app = FastAPI(
    title="Confluence Gateway",
    description="API for searching and retrieving Confluence content",
    version="0.1.0",
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
        "version": app.version,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    try:
        # Test connection to Confluence
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
