#!/usr/bin/env python3
"""Self-contained MCP server that handles dependencies automatically."""

import subprocess
import sys
from pathlib import Path

def run_with_uvx():
    """Run the server using uvx with all dependencies."""
    project_root = Path(__file__).parent
    server_path = project_root / "src" / "codebase_rag" / "server.py"
    
    cmd = [
        "uvx",
        "--with", "mcp",
        "--with", "chromadb",
        "--with", "sentence-transformers", 
        "--with", "numpy",
        "--with", "pydantic",
        "--with", "python-dotenv",
        "--with-editable", str(project_root),
        "python", str(server_path)
    ]
    
    print("Starting MCP server with uvx...")
    subprocess.run(cmd, check=True)

def main():
    """Main entry point."""
    try:
        # Try uvx first (most reliable)
        run_with_uvx()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"uvx failed: {e}", file=sys.stderr)
        print("Please install uvx or run manually:", file=sys.stderr)
        print("uvx --with mcp --with chromadb --with sentence-transformers python /path/to/server.py", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
