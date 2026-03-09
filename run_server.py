#!/usr/bin/env python3
"""Wrapper script for MCP Codebase RAG Server with proper environment setup."""

import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import and run the server
try:
    from codebase_rag.server import mcp
    mcp.run()
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr)
    print("Installing dependencies with uvx...", file=sys.stderr)
    
    # Try to run with uvx as fallback
    try:
        server_path = project_root / "src" / "codebase_rag" / "server.py"
        cmd = [
            "uvx", "--with", "mcp", "--with", "chromadb", 
            "--with", "sentence-transformers", "--with", "numpy",
            "--with", "pydantic", "--with", "python-dotenv",
            "--with-editable", str(project_root),
            "python", str(server_path)
        ]
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Please install dependencies manually:", file=sys.stderr)
        print("cd /home/gabrielsb/mcp-servers/codebase-rag && pip install -e .", file=sys.stderr)
        sys.exit(1)
