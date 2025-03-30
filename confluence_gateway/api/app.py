import datetime

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.api.routes import api_router
from confluence_gateway.core.config import confluence_config
from confluence_gateway.core.exceptions import (
    ConfluenceAPIError,
    ConfluenceAuthenticationError,
    ConfluenceConnectionError,
)
from confluence_gateway.services.search import SearchService

# Create FastAPI instance
app = FastAPI(
    title="Confluence Gateway",
    description="API for searching and retrieving Confluence content",
    version="0.1.0",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (can be restricted in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Dependency injection functions
def get_confluence_client():
    """Dependency for ConfluenceClient."""
    return ConfluenceClient(config=confluence_config)


def get_search_service(client: ConfluenceClient = Depends(get_confluence_client)):
    """Dependency for SearchService."""
    return SearchService(client=client)


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check(client: ConfluenceClient = Depends(get_confluence_client)):
    """Check the health of the service and its connections."""
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
