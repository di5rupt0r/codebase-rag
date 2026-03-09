from __future__ import annotations

from codebase_rag.server import mcp


def main() -> None:
    """Entry point to run the MCP Codebase RAG server."""
    mcp.run()


if __name__ == "__main__":
    main()
