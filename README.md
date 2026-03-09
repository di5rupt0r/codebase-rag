# MCP Codebase RAG Server

> Servidor MCP customizado que expõe busca semântica vetorial de codebase para qualquer client MCP (Copilot CLI, Cline, Claude Desktop, Perplexity, etc.).
> **Objetivo**: Adicionar RAG robusto ao Copilot Pro sem depender de ferramentas pagas.
> **Zero custo, zero limites, controle total.**

---

## 📋 Visão Geral

### Problema Resolvido
- GitHub Copilot Pro tem modelo excelente (Sonnet 4.6) mas RAG limitado
- Cursor/Windsurf têm RAG bom mas custam $15-20/mês  
- Continue.dev tem RAG mas não integra nativamente com agents MCP-aware

### Solução
Este servidor MCP oferece:
1. **Indexação** de codebase local usando embeddings vetoriais
2. **Busca semântica** via tool `search_codebase` 
3. **Multi-projeto** com coleções separadas no ChromaDB
4. **Integração total** com qualquer client MCP

### Stack Tecnológica

| Componente | Tecnologia | Motivo |
|---|---|---|
| Embeddings | **sentence-transformers** (all-MiniLM-L6-v2) | Rápido, leve, 384-dim |
| Vector DB | **ChromaDB** | Simples, persistente, zero config |
| Code parsing | Python AST + chunking por caractere | Funciona em qualquer linguagem |
| MCP SDK | **@modelcontextprotocol/sdk** | Padrão oficial |
| Runtime | Python 3.11+ | Ecosystem maduro |

---

## 🚀 Instalação e Setup

### Pré-requisitos
- Python 3.11+
- uv (recomendado) ou pip
- Acesso ao codebase que deseja indexar

### Instalação

```bash
# Clone o projeto
git clone <repository-url>
cd mcp-servers/codebase-rag

# Instale dependências com uv
uv install

# Ou com pip
pip install -e .
```

### Health Check

```bash
# Verifique se tudo está funcionando
python scripts/health_check.py
```

Saída esperada:
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

## 📖 Uso Rápido

### 1. Indexar um Projeto

```bash
# Indexe o projeto atual
python scripts/index_project.py . --name meu-projeto

# Indexe path específico
python scripts/index_project.py ~/projetos/api --name api-backend

# Force reindex (apaga índice existente)
python scripts/index_project.py . --name meu-projeto --force

# Dry run para ver o que será indexado
python scripts/index_project.py . --name meu-projeto --dry-run
```

### 2. Iniciar Servidor MCP

```bash
# Inicie o servidor
python src/codebase_rag/server.py

# Ou via main.py
python main.py
```

### 3. Configurar Client MCP

#### Copilot CLI
```bash
gh copilot config set mcp.servers.codebase-rag.command \
  "python /home/gabriel/mcp-servers/codebase-rag/src/codebase_rag/server.py"
```

#### Cline (VS Code)

**Configuração Testada e Funcional:**
```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "uvx",
      "args": [
        "--with", "mcp",
        "--with", "chromadb",
        "--with", "sentence-transformers",
        "--with", "numpy",
        "--with", "pydantic",
        "--with", "python-dotenv",
        "--with-editable", "/home/gabrielsb/mcp-servers/codebase-rag",
        "python", "entrypoint.py"
      ]
    }
  }
}
```

#### Claude Desktop

**Configuração Testada e Funcional:**
```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "uvx",
      "args": [
        "--with", "mcp",
        "--with", "chromadb",
        "--with", "sentence-transformers",
        "--with", "numpy",
        "--with", "pydantic",
        "--with", "python-dotenv",
        "--with-editable", "/home/gabrielsb/mcp-servers/codebase-rag",
        "python", "entrypoint.py"
      ]
    }
  }
}
```

---

## 🛠️ MCP Tools Disponíveis

### `search_codebase`
Busca semântica no codebase usando embeddings vetoriais.

**Input**:
```json
{
  "query": "onde está a lógica de autenticação?",
  "top_k": 5,
  "project": "meu-projeto",
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
Re-indexa um projeto após mudanças grandes.

**Input**:
```json
{
  "project_path": "/home/gabriel/projetos/api",
  "project_name": "api-backend",
  "force": false
}
```

**Output**:
```json
{
  "status": "success",
  "files_indexed": 23,
  "chunks_created": 1247,
  "time_seconds": 8.3
}
```

### `list_indexed_projects`
Lista todos os projetos indexados.

**Output**:
```json
{
  "projects": [
    {
      "name": "meu-projeto",
      "path": "/home/gabriel/meu-projeto",
      "files": 23,
      "chunks": 1247
    }
  ]
}
```

### `get_files`
Lista arquivos indexados em um projeto.

**Input**:
```json
{
  "project": "meu-projeto"
}
```

### `get_file_content`
Retorna conteúdo completo de um arquivo.

**Input**:
```json
{
  "path": "src/main.py"
}
```

---

## ⚙️ Configuração

### Variáveis de Ambiente

```bash
# Path do ChromaDB (default: ./data/chroma_db)
export CHROMA_DB_PATH="/custom/path/to/chroma"

# Modelo de embeddings (default: all-MiniLM-L6-v2)
export EMBEDDING_MODEL="multi-qa-mpnet-base-dot-v1"

# Nível de log (default: INFO)
export LOG_LEVEL="DEBUG"
```

### Configuração de Chunking

No arquivo `src/codebase_rag/config.py`:

```python
CHUNK_SIZE = 500          # caracteres por chunk
CHUNK_OVERLAP = 50        # overlap entre chunks
DEFAULT_TOP_K = 5         # resultados por busca
```

### Arquivos Suportados

- **Python**: `.py`
- **JavaScript/TypeScript**: `.js`, `.ts`, `.jsx`, `.tsx`
- **Java**: `.java`
- **C/C++**: `.c`, `.cpp`, `.h`
- **Go**: `.go`
- **Rust**: `.rs`
- **Outros**: `.rb`, `.php`, `.cs`, `.sh`, `.yaml`, `.yml`, `.json`

### Padrões Ignorados

- `*.pyc`, `__pycache__`, `.git`
- `node_modules`, `.venv`, `venv`
- `*.egg-info`, `.pytest_cache`

---

## 📊 Benchmarks

| Operação | Tempo Esperado | Notas |
|---|---|---|
| Indexar 20 arquivos .py (~5k LOC) | ~5-8s | Primeira vez; incremental é mais rápido |
| Busca vetorial (top_k=5) | ~20-50ms | ChromaDB é muito rápido |
| Embedding de query | ~10-20ms | sentence-transformers local |
| Cold start do server | ~2-3s | Carrega modelo na memória |

---

## 🧪 Testes

### Rodar Testes

```bash
# Todos os testes
uvx --with pytest --with numpy --with sentence-transformers --with-editable . pytest -v

# Testes específicos
pytest tests/test_config.py -v
pytest tests/test_embeddings.py -v  
pytest tests/test_indexer.py -v
pytest tests/test_server.py -v
```

### Cobertura

```bash
pytest --cov=codebase_rag --cov-report=html
```

---

## 🔧 Deploy (systemd)

### Criar Service File

`/etc/systemd/system/codebase-rag.service`:
```ini
[Unit]
Description=MCP Codebase RAG Server
After=network.target

[Service]
Type=simple
User=gabriel
WorkingDirectory=/home/gabriel/mcp-servers/codebase-rag
Environment="PATH=/home/gabriel/.local/bin:/usr/bin"
ExecStart=/home/gabriel/.local/bin/uv run python src/codebase_rag/server.py
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

### Comandos Deploy

```bash
# Instalar service
sudo cp systemd/codebase-rag.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable codebase-rag
sudo systemctl start codebase-rag

# Verificar status
sudo systemctl status codebase-rag
sudo journalctl -u codebase-rag -f
```

---

## 🎯 Exemplos de Uso

### Busca por Função Específica

```bash
# Via Copilot CLI
gh copilot suggest "use search_codebase para encontrar onde está a função score_job"

# Resultado esperado: trechos de código com a função, ordenados por relevância
```

### Análise de Arquitetura

```bash
gh copilot suggest "use search_codebase com query 'padrão de injeção de dependência' top_k=10"
```

### Debugging

```bash
gh copilot suggest "use search_codebase para encontrar onde ocorre o erro 'database connection failed'"
```

---

## 🔄 Roadmap Futuro

- [ ] **Indexação incremental** - só reindexa arquivos modificados
- [ ] **Auto-discovery** - detecta projetos Git automaticamente  
- [ ] **Watch mode** - reindexa automaticamente após mudanças
- [ ] **UI web** - interface para visualizar índice e testar buscas
- [ ] **Múltiplos modelos** - suporte a diferentes embedding models
- [ ] **Cache de queries** - acelera buscas frequentes
- [ ] **Integração LSP** - indexação awareness de linguagem

---

## 🐛 Troubleshooting

### Problemas Comuns

**Model loading lento**: Primeira execução baixa o modelo (~100MB). Execute `health_check.py` para pré-carregar.

**Memory usage**: Modelo usa ~500MB RAM. Se necessário, use modelo menor.

**Permissões**: Verifique se o usuário tem acesso de escrita ao `data/chroma_db`.

**Embeddings falhando**: Verifique conexão com internet para primeiro download do modelo.

### Debug Mode

```bash
# Ative logs detalhados
LOG_LEVEL=DEBUG python src/codebase_rag/server.py

# Verifique configuração
python -c "from codebase_rag import config; print(config.CHROMA_DB_PATH)"
```

---

## 📝 Contribuindo

1. Fork o projeto
2. Crie feature branch: `git checkout -b feature/nova-funcionalidade`
3. Siga TDD rigoroso: RED → GREEN → REFACTOR
4. Commits atômicos descritivos
5. Pull request com testes

### Desenvolvimento

```bash
# Ambiente dev
uv install --dev

# Formatação
black src/ tests/
ruff check src/ tests/

# Testes completos
pytest -v --cov=codebase_rag
```

---

## 📄 Licença

MIT License - ver arquivo LICENSE.

---

## 🔗 Referências

- [MCP Protocol](https://modelcontextprotocol.io)
- [ChromaDB Docs](https://docs.trychroma.com)  
- [sentence-transformers](https://www.sbert.net)
- [FastMCP](https://github.com/jlowin/fastmcp)

---

**Feito com ❤️ para dar poder ao Copilot com RAG local e ilimitado.**