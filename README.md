# MCP Codebase RAG Server

> Self-hosted MCP server that adds semantic vector search over your local codebases to any MCP-capable client (GitHub Copilot, Cline, Claude Desktop, etc.).  
> **Goal**: Robust RAG for Copilot (or any MCP client) without paying for Cursor/Windsurf.  
> **Zero cost. Zero limits. Full control.**

---

## 📋 Overview

### Problem Solved
- GitHub Copilot Pro has an excellent model but limited codebase RAG
- Cursor/Windsurf have good RAG but cost $15–20/month
- Continue.dev has RAG but doesn't integrate natively with MCP-aware agents

### Solution
This MCP server provides:
1. **Indexing** of local codebases using vector embeddings
2. **Hybrid semantic search** via `search_codebase` tool — combines dense (embeddings) + sparse (BM25) + RRF fusion
3. **Multi-project** support with isolated ChromaDB collections
4. **Universal integration** with any MCP client

### Tech Stack

| Component | Technology | Why |
|---|---|---|
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Fast, lightweight, 384-dim |
| Code embeddings | **microsoft/unixcoder-base** (optional) | Code-specific model, activated via `EMBEDDING_MODEL` |
| Vector DB | **ChromaDB** | Simple, persistent, zero config |
| Code parsing | **Tree-sitter** + BM25 + RRF | Universal language-agnostic chunking and hybrid search |
| MCP SDK | **modelcontextprotocol/python-sdk** | Official standard |
| Runtime | Python 3.11+ | — |

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- `pip` or `uv`

### Install

```bash
git clone https://github.com/di5rupt0r/codebase-rag.git
cd codebase-rag

# Install as a package (adds the `codebase-rag` command to ~/.local/bin)
pip install -e .
```

### Health Check

```bash
python scripts/health_check.py
```

Expected output:
```
🔍 MCP Codebase RAG Server Health Check
==================================================
Checking Embedding Provider... ✓ OK (3.03s)
Checking ChromaDB Connection... ✓ OK (0.14s)
Checking Search Functionality... ✓ OK (2.36s)
Checking Data Directory... ✓ OK (0.00s)
==================================================
Health Check Summary: 4/4 checks passed
🎉 All systems operational!
```

---

## 📖 Quick Start

### 1. Index a Project

```bash
# Index the current directory
python scripts/index_project.py . --name my-project

# Index a specific path
python scripts/index_project.py ~/projects/api --name api-backend

# Force full reindex
python scripts/index_project.py . --name my-project --force

# Dry run to preview what will be indexed
python scripts/index_project.py . --name my-project --dry-run
```

### 2. Start the MCP Server

#### stdio (default — for local clients)
```bash
codebase-rag
```

#### HTTP (for remote clients or always-on service)
```bash
MCP_TRANSPORT=streamable-http MCP_PORT=8080 codebase-rag
```

### 3. Configure Your MCP Client

#### VS Code (GitHub Copilot / Cline) — stdio mode

Add to your VS Code `mcp.json`:
```json
{
  "servers": {
    "codebase-rag": {
      "type": "stdio",
      "command": "codebase-rag"
    }
  }
}
```

#### VS Code — HTTP mode (when running as a service)
```json
{
  "servers": {
    "codebase-rag": {
      "type": "http",
      "url": "http://127.0.0.1:8080/mcp"
    }
  }
}
```

#### Claude Desktop
```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "codebase-rag"
    }
  }
}
```

---

## � Search Capabilities

### Hybrid Search Architecture

The server implements a **hybrid search** system that combines:

1. **Dense Search** (Vector Embeddings)
   - Semantic similarity using sentence-transformers
   - Finds conceptually similar code
   - Base: ChromaDB vector similarity

2. **Sparse Search** (BM25)
   - Exact lexical term matching
   - Finds precise identifiers and keywords
   - Base: rank-bm25 with regex tokenization

3. **Reciprocal Rank Fusion** (RRF)
   - Intelligent fusion of dense + sparse results
   - k=60 (standard literature value)
   - Improves both precision and recall

### Search Results

```json
{
  "results": [
    {
      "path": "src/auth.py",
      "content": "def authenticate_user(user, password): ...",
      "score": 0.0325,
      "type": "function",
      "name": "authenticate_user", 
      "line_start": 15,
      "line_end": 25
    }
  ],
  "total_indexed_chunks": 1247,
  "query_time_ms": 23.4,
  "search_type": "hybrid_rrf"
}
```

### Performance Characteristics

| Metric | Target | Description |
|---|---|---|
| Tree-sitter parsing | < 50ms/file | Universal language parsing |
| BM25 indexing | < 10ms/query | In-memory reconstruction |
| RRF fusion | < 1ms | In-memory score calculation |
| Total query time | < 100ms | End-to-end hybrid search |
| Memory overhead | < 50MB | For 5k chunks |

### Fallback Behavior

- **Tree-sitter unavailable** → Line-based chunking
- **BM25 unavailable** → Dense-only search  
- **Both unavailable** → Original dense search with keyword reranking

## �️ MCP Tools

### `search_codebase`
Hybrid semantic search over an indexed project using vector embeddings + BM25 + RRF fusion.

**Input**:
```json
{
  "query": "where is the authentication logic?",
  "top_k": 5,
  "project": "my-project",
  "file_types": [".py", ".js"]
}
```

**Output**:
```json
{
  "results": [
    {
      "path": "src/auth.py",
      "content": "def authenticate_user(user, password):\n    ...",
      "score": 0.89
    }
  ],
  "total_indexed_chunks": 1247,
  "query_time_ms": 23
}
```

### `reindex_project`
Re-index a project after large changes.

**Input**:
```json
{
  "project_path": "/path/to/your/project",
  "project_name": "my-project",
  "force": false
}
```

### `list_indexed_projects`
List all indexed projects.

### `get_files`
List indexed files in a project.

**Input**: `{ "project": "my-project" }`

### `get_file_content`
Return the full content of an indexed file.

**Input**: `{ "path": "src/main.py" }`

---

## ⚙️ Configuration

### Environment Variables

```bash
# ChromaDB path (default: ./data/chroma_db relative to install dir)
export CHROMA_DB_PATH="/custom/path/to/chroma"

# Embedding model (default: all-MiniLM-L6-v2)
# Use microsoft/unixcoder-base for better code-specific embeddings (~2GB, requires torch)
export EMBEDDING_MODEL="microsoft/unixcoder-base"

# HTTP transport settings (only needed in HTTP/service mode)
export MCP_TRANSPORT="streamable-http"
export MCP_HOST="127.0.0.1"
export MCP_PORT="8080"
# Set this when exposing via reverse proxy or Tailscale Funnel
export MCP_ALLOWED_HOST="your-hostname.example.com"

# Log level (default: INFO)
export LOG_LEVEL="DEBUG"
```

### Chunking (Advanced)

Edit `src/codebase_rag/config.py`:

```python
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # overlap between chunks
DEFAULT_TOP_K = 5         # default results per search
```

### Supported File Types

Python, JavaScript, TypeScript, JSX, TSX, Java, C, C++, Go, Rust, Ruby, PHP, C#, Shell, YAML, JSON.

### Ignored Patterns

`*.pyc`, `__pycache__`, `.git`, `node_modules`, `.venv`, `venv`, `*.egg-info`, `.pytest_cache`

---

## 📊 Benchmarks

| Operation | Expected Time | Notes |
|---|---|---|
| Index 20 .py files (~5k LOC) | ~5–8s | First run; incremental is much faster |
| Vector search (top_k=5) | ~20–50ms | ChromaDB in-process |
| Query embedding | ~10–20ms | sentence-transformers, CPU |
| Server cold start | ~2–3s | Model loaded into memory |

---

## 🤖 Automation Scripts

### Auto-discovery
Scan a directory for Git repositories and index them all automatically:
```bash
python scripts/auto_index.py ~/projects
```

### Watch Mode
Watch a project for file changes and reindex incrementally (debounced, 5s):
```bash
python scripts/watch.py /path/to/project --name my-project
```

### Git Hook (post-commit reindex)
Install a post-commit hook so changed files are reindexed automatically after every commit:
```bash
python scripts/setup_git_hook.py /path/to/your/repo my-project
```

---

## 🧪 Tests

```bash
# All tests (116 passing)
pytest -v

# Specific modules
pytest tests/test_config.py -v
pytest tests/test_embeddings.py -v
pytest tests/test_indexer.py -v
pytest tests/test_server.py -v

# With coverage
pytest --cov=codebase_rag --cov-report=html
```

---

## 🔧 Deploy as a systemd Service (Linux)

A template service file is provided at `systemd/codebase-rag-server.service`.  
Replace `YOUR_USERNAME` with your actual Linux username before installing:

```bash
# Substitute your username in-place
sed -i "s/YOUR_USERNAME/$USER/g" systemd/codebase-rag-server.service

# Install and start
sudo cp systemd/codebase-rag-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable codebase-rag-server
sudo systemctl start codebase-rag-server

# Check
sudo systemctl status codebase-rag-server
sudo journalctl -u codebase-rag-server -f
```

### Exposing Remotely via Tailscale Funnel (optional)

To use the server from a remote machine (Codespaces, company laptop, etc.):

```bash
# Expose port 8080 via Tailscale Funnel
tailscale funnel 8080

# Add to your service file:
# Environment="MCP_ALLOWED_HOST=your-machine.your-tailnet.ts.net"

# Then in your remote mcp.json:
# "url": "https://your-machine.your-tailnet.ts.net/mcp"
```

---

## 🐛 Troubleshooting

**Slow first start**: The embedding model (~100MB) is downloaded on first use. Run `health_check.py` to pre-load it.

**High memory usage**: The default model uses ~500MB RAM. If needed, use an even smaller model via `EMBEDDING_MODEL`.

**Permission errors**: Ensure the running user has write access to `data/chroma_db/`.

**Debug mode**:
```bash
LOG_LEVEL=DEBUG codebase-rag
```

---

## 📝 Contributing

1. Fork the project
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Follow strict TDD: RED → GREEN → REFACTOR
4. Atomic, descriptive commits
5. Open a pull request with tests

```bash
# Dev setup
pip install -e ".[dev]"
pytest -v --cov=codebase_rag
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🔗 References

- [MCP Protocol](https://modelcontextprotocol.io)
- [ChromaDB Docs](https://docs.trychroma.com)
- [sentence-transformers](https://www.sbert.net)
- [FastMCP](https://github.com/jlowin/fastmcp)

---
