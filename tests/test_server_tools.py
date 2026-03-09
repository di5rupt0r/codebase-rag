from __future__ import annotations

from typing import List, Dict, Any

import pytest


def test_search_codebase_tool_delegates_to_indexer(mocker) -> None:
    """search_codebase tool should delegate to indexer.search_codebase."""
    from codebase_rag import server

    fake_result: List[Dict[str, Any]] = [
        {"path": "/tmp/example.py", "score": 0.9, "content": "def foo(): pass"}
    ]
    spy = mocker.patch(
        "codebase_rag.server._search_codebase",
        return_value=fake_result,
    )

    out = server.search_codebase("query", top_k=3)

    spy.assert_called_once_with(query="query", top_k=3)
    assert out == fake_result


def test_get_files_tool_delegates_to_indexer(mocker) -> None:
    """get_files tool should delegate to indexer.list_indexed_files."""
    from codebase_rag import server

    fake_files = [{"path": "/tmp/example.py"}]
    spy = mocker.patch(
        "codebase_rag.server._list_indexed_files",
        return_value=fake_files,
    )

    out = server.get_files()

    spy.assert_called_once_with()
    assert out == fake_files


def test_get_file_content_tool_delegates_to_indexer(mocker) -> None:
    """get_file_content tool should delegate to indexer.get_file_content."""
    from codebase_rag import server

    spy = mocker.patch(
        "codebase_rag.server._get_file_content",
        return_value="content",
    )

    out = server.get_file_content("/tmp/example.py")

    spy.assert_called_once_with("/tmp/example.py")
    assert out == "content"


def test_mcp_instance_name_matches_config() -> None:
    """Ensure MCP server name is consistent with config."""
    from codebase_rag import server, config

    assert server.mcp.name == config.SERVER_NAME

