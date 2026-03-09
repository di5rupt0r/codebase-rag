#!/usr/bin/env python3
"""Entry point for MCP Codebase RAG Server."""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import and run server
from codebase_rag.server import mcp

if __name__ == "__main__":
    mcp.run()
