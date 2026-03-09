from __future__ import annotations

from typing import List, Dict, Any

from mcp.server.fastmcp import FastMCP

from . import config
from .indexer import (
    search_codebase as _search_codebase,
    list_indexed_files as _list_indexed_files,
    get_file_content as _get_file_content,
)


mcp = FastMCP(config.SERVER_NAME, json_response=True)


@mcp.tool()
def search_codebase(query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Search the indexed codebase for relevant code snippets."""
    return _search_codebase(query=query, top_k=top_k)


@mcp.tool()
def get_files() -> List[Dict[str, Any]]:
    """List all files that have been indexed in the codebase."""
    return _list_indexed_files()


@mcp.tool()
def get_file_content(path: str) -> str:
    """Return the full content of a given file path."""
    return _get_file_content(path)

