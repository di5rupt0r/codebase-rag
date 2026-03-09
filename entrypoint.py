#!/usr/bin/env python3
"""Entry point for MCP Codebase RAG Server."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Run the MCP server
if __name__ == "__main__":
    from src.codebase_rag.server import mcp
    mcp.run()
