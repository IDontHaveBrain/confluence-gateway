from fastapi import APIRouter

from .generation import router as generation_router
from .indexing import router as indexing_router
from .search import router as search_router
from .spaces import router as spaces_router

api_router = APIRouter()
api_router.include_router(search_router, prefix="/search", tags=["Search"])
api_router.include_router(indexing_router, prefix="/indexing", tags=["Indexing"])
api_router.include_router(generation_router, prefix="/generate", tags=["Generation"])
api_router.include_router(spaces_router, prefix="/spaces", tags=["Spaces"])
