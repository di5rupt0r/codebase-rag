from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from . import config
from .indexer import (
    search_codebase as _search_codebase,
    list_indexed_files as _list_indexed_files,
    get_file_content as _get_file_content,
    index_codebase as _index_codebase,
    _get_client,
)

_host = os.environ.get("MCP_HOST", "127.0.0.1")
_port = int(os.environ.get("MCP_PORT", "8000"))
_allowed_host = os.environ.get("MCP_ALLOWED_HOST", "")

# Allow external hostname (e.g. Tailscale Funnel) when specified
_transport_security: TransportSecuritySettings | None = None
if _allowed_host and _host in ("127.0.0.1", "localhost", "::1"):
    _transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            f"{_host}:*", "localhost:*", "[::1]:*",
            _allowed_host, f"{_allowed_host}:*",
        ],
        allowed_origins=[
            f"http://{_host}:*", "http://localhost:*",
            f"https://{_allowed_host}", f"https://{_allowed_host}:*",
        ],
    )

mcp = FastMCP(
    config.SERVER_NAME,
    json_response=True,
    host=_host,
    port=_port,
    transport_security=_transport_security,
)


@mcp.tool()
def search_codebase(
    query: str, 
    top_k: int = config.DEFAULT_TOP_K, 
    project: Optional[str] = None,
    file_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Search the indexed codebase for relevant code snippets.
    
    Args:
        query: Search query text
        top_k: Number of results to return (default: 5)
        project: Project name to search within (optional)
        file_types: List of file extensions to filter (e.g., [".py", ".js"])
    
    Returns:
        Dictionary with search results, metadata, and timing info
    """
    import time
    start_time = time.time()
    
    # Use project name as collection name if provided
    collection_name = project if project else "codebase-rag"
    
    try:
        results = _search_codebase(query=query, top_k=top_k, collection_name=collection_name)
        
        # Filter by file types if specified
        if file_types:
            results = [
                result for result in results 
                if Path(result.get("path", "")).suffix in file_types
            ]
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return {
            "results": results,
            "total_indexed_chunks": len(_list_indexed_files(collection_name=collection_name)),
            "query_time_ms": round(query_time_ms, 2),
        }
    except Exception as e:
        return {
            "results": [],
            "error": str(e),
            "query_time_ms": round((time.time() - start_time) * 1000, 2),
        }


@mcp.tool()
def reindex_project(
    project_path: str, 
    project_name: str,
    force: bool = False
) -> Dict[str, Any]:
    """Re-index a project (useful after major code changes).
    
    Args:
        project_path: Absolute path to the project directory
        project_name: Name to use for the collection
        force: If True, deletes existing index before reindexing
    
    Returns:
        Dictionary with indexing status and statistics
    """
    import time
    import shutil
    start_time = time.time()
    
    try:
        project_path_obj = Path(project_path).resolve()
        if not project_path_obj.exists():
            return {
                "status": "error",
                "error": f"Project path does not exist: {project_path}",
                "time_seconds": 0,
            }
        
        collection_name = project_name
        
        # If force is True, delete existing collection before re-indexing
        if force:
            try:
                client = _get_client()
                client.delete_collection(name=collection_name)
            except Exception:
                # Collection might not exist — ignore
                pass
        
        # Count files before indexing
        files_before = len(list(_list_indexed_files(collection_name=collection_name)))
        
        # Perform indexing
        chunks_created = _index_codebase(project_path_obj, collection_name=collection_name)
        
        # Count files after indexing
        files_after = len(list(_list_indexed_files(collection_name=collection_name)))
        
        time_seconds = time.time() - start_time
        
        return {
            "status": "success",
            "files_indexed": files_after,
            "chunks_created": chunks_created,
            "time_seconds": round(time_seconds, 1),
            "force_used": force,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "time_seconds": round(time.time() - start_time, 1),
        }


@mcp.tool()
def list_indexed_projects() -> Dict[str, Any]:
    """List all projects that have been indexed.

    Returns:
        Dictionary with list of indexed projects and their metadata
    """
    try:
        client = _get_client()
        collections = client.list_collections()

        projects = []
        for col in collections:
            col_obj = client.get_collection(col.name)
            result = col_obj.get(include=["metadatas"])
            metadatas = result.get("metadatas") or []

            files: set = set()
            for meta in metadatas:
                if meta and meta.get("path"):
                    files.add(meta["path"])

            projects.append(
                {
                    "name": col.name,
                    "path": "N/A",
                    "files": len(files),
                    "chunks": len(metadatas),
                    "last_indexed": "Unknown",
                }
            )

        return {
            "projects": projects,
            "total_projects": len(projects),
        }
    except Exception as e:
        return {
            "projects": [],
            "error": str(e),
            "total_projects": 0,
        }


@mcp.tool()
def get_files(project: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all files that have been indexed in the codebase.
    
    Args:
        project: Project name to list files for (optional)
    
    Returns:
        List of file metadata dictionaries
    """
    collection_name = project if project else "codebase-rag"
    return _list_indexed_files(collection_name=collection_name)


@mcp.tool()
def get_file_content(path: str) -> str:
    """Return the full content of a given file path.
    
    Args:
        path: File path to read
        
    Returns:
        File content as string
    """
    return _get_file_content(path)


def main() -> None:
    """Entry point for the codebase-rag MCP server."""
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)  # type: ignore[arg-type]

