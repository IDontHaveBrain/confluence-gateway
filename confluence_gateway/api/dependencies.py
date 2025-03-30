from fastapi import Depends

from confluence_gateway.adapters.confluence.client import ConfluenceClient
from confluence_gateway.core.config import confluence_config
from confluence_gateway.services.search import SearchService


def get_confluence_client():
    return ConfluenceClient(config=confluence_config)


def get_search_service(client: ConfluenceClient = Depends(get_confluence_client)):
    return SearchService(client=client)
