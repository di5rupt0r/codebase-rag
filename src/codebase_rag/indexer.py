from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import numpy as np
import chromadb
from chromadb.api.models import Collection

try:
    from tree_sitter import Language, Parser
    from tree_sitter_languages import get_language
except ImportError:
    Language = None
    Parser = None
    get_language = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from . import config
from .embeddings import EmbeddingProvider


# Tree-sitter configuration
_EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript", 
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}

_CHUNK_NODE_TYPES = {
    "function_definition",
    "method_definition", 
    "class_definition",
    "constructor_definition",
    "async_function_definition",
    "async_method_definition",
}


@dataclass
class IndexStats:
    """Simple statistics about an indexing run."""
    documents_indexed: int


@dataclass
class BM25Index:
    """BM25 index for sparse search."""
    
    def __init__(self) -> None:
        if BM25Okapi is None:
            raise ImportError("rank-bm25 not available")
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def index(self, chunks: List[Dict[str, Any]]) -> None:
        """Index chunks for BM25 search."""
        self.chunks = chunks
        # Extract text content from chunks
        corpus = [chunk.get("text", "") for chunk in chunks]
        # Tokenize corpus
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(self, query: str, top_k: int = 10) -> List[tuple[int, float]]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        # Search
        doc_scores = self.bm25.get_scores(tokenized_query)
        # Get top-k results
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]
        return [(int(idx), float(doc_scores[idx])) for idx in top_indices if doc_scores[idx] > 0]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization using regex."""
        # Extract alphanumeric tokens, minimum length 2
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return [token for token in tokens if len(token) >= 2]


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[tuple[int, float]], 
    chunks: List[Dict[str, Any]], 
    k: int = 60, 
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Fuse dense and sparse search results using Reciprocal Rank Fusion."""
    # Create mapping from chunk index to chunk data
    chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
    
    # Create score dictionaries
    dense_scores = {}
    sparse_scores = {}
    
    # Process dense results (from ChromaDB)
    for i, result in enumerate(dense_results):
        chunk_idx = result.get("chunk_index", i)
        rank = i + 1  # 1-based ranking
        rrf_score = 1.0 / (k + rank)
        dense_scores[chunk_idx] = rrf_score
    
    # Process sparse results (from BM25)
    for chunk_idx, score in sparse_results:
        # Find rank of this chunk in sparse results
        sparse_rank = 1
        for i, (idx, _) in enumerate(sparse_results):
            if idx == chunk_idx:
                sparse_rank = i + 1
                break
        
        rrf_score = 1.0 / (k + sparse_rank)
        sparse_scores[chunk_idx] = rrf_score
    
    # Combine scores
    combined_scores = {}
    
    # Add dense scores
    for chunk_idx, score in dense_scores.items():
        combined_scores[chunk_idx] = score
    
    # Add sparse scores
    for chunk_idx, score in sparse_scores.items():
        if chunk_idx in combined_scores:
            combined_scores[chunk_idx] += score
        else:
            combined_scores[chunk_idx] = score
    
    # Sort by combined score (descending)
    sorted_results = sorted(
        combined_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Return top-k results with chunk data
    fused_results = []
    for chunk_idx, combined_score in sorted_results[:top_k]:
        chunk_data = chunk_map.get(chunk_idx)
        if chunk_data:
            result = {
                "path": chunk_data.get("path", ""),
                "content": chunk_data.get("text", ""),
                "score": combined_score,
                "type": chunk_data.get("type", ""),
                "name": chunk_data.get("name", ""),
                "line_start": chunk_data.get("line_start", 0),
                "line_end": chunk_data.get("line_end", 0),
            }
            fused_results.append(result)
    
    return fused_results


def _iter_source_files(root: Path) -> Iterable[Path]:
    """Yield all supported, non-ignored files under given root."""
    root = root.resolve()
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if not config.is_supported_file(path):
            continue
        if config.should_ignore_path(path):
            continue
        yield path


def _chunk_by_treesitter(content: str, file_extension: str) -> List[Dict[str, Any]]:
    """Chunk content using tree-sitter for universal language support."""
    if Language is None or Parser is None or get_language is None:
        return _fallback_chunk_by_lines(content)
    
    # Get language from file extension
    lang_name = _EXTENSION_MAP.get(file_extension.lower())
    if not lang_name:
        return _fallback_chunk_by_lines(content)
    
    try:
        # Get tree-sitter language
        language = get_language(lang_name)
        if language is None:
            return _fallback_chunk_by_lines(content)
        
        # Create parser
        parser = Parser()
        parser.set_language(language)
        
        # Parse content
        tree = parser.parse(bytes(content, "utf-8"))
        
        chunks = []
        # Walk the tree and extract chunk nodes
        def _extract_chunks(node, depth=0):
            if node.type in _CHUNK_NODE_TYPES:
                # Extract node text
                start_byte = node.start_byte
                end_byte = node.end_byte
                chunk_text = content[start_byte:end_byte]
                
                # Get line numbers
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                
                # Extract node name if available
                node_name = ""
                for child in node.children:
                    if child.type == "identifier":
                        node_name = content[child.start_byte:child.end_byte]
                        break
                
                chunk_type = "class" if "class" in node.type else "function"
                
                chunks.append({
                    "text": chunk_text,
                    "type": chunk_type,
                    "name": node_name,
                    "line_start": start_line,
                    "line_end": end_line,
                    "start": start_byte,
                    "end": end_byte,
                })
            
            # Recursively process children
            for child in node.children:
                _extract_chunks(child, depth + 1)
        
        _extract_chunks(tree.root_node)
        
        # If no chunks found, fallback to line-based
        if not chunks:
            return _fallback_chunk_by_lines(content)
        
        return chunks
        
    except Exception:
        # Any error, fallback to line-based chunking
        return _fallback_chunk_by_lines(content)


def _fallback_chunk_by_lines(content: str, chunk_size: int = 50, overlap: int = 5) -> List[Dict[str, Any]]:
    """Fallback line-based chunking for unsupported languages or errors."""
    lines = content.splitlines()
    chunks: List[Dict[str, Any]] = []
    step = max(1, chunk_size - overlap)
    
    for i in range(0, len(lines), step):
        end = min(i + chunk_size, len(lines))
        chunk_content = "\n".join(lines[i:end])
        
        chunks.append({
            "text": chunk_content,
            "type": "block",
            "name": "",
            "line_start": i + 1,
            "line_end": end,
            "start": i,
            "end": end,
        })
    
    return chunks


def _get_client(db_path: Optional[Path] = None) -> chromadb.PersistentClient:
    """Create or reuse a Chroma persistent client."""
    if db_path is None:
        db_path = config.get_chroma_db_path()
    else:
        db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def _get_collection(
    client: chromadb.PersistentClient,
    name: str,
) -> Collection:
    return client.get_or_create_collection(name=name)


def index_codebase(
    root: Path,
    db_path: Optional[Path] = None,
    *,
    embedding_provider: Optional[EmbeddingProvider] = None,
    collection_name: str = "codebase-rag",
) -> int:
    """Index all supported files under `root` into ChromaDB using tree-sitter chunking.

    Returns number of chunks indexed.
    """
    root = root.resolve()
    provider = embedding_provider or EmbeddingProvider()
    client = _get_client(db_path)
    collection = _get_collection(client, collection_name)

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for file_path in _iter_source_files(root):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # Use tree-sitter chunking
        file_extension = file_path.suffix
        chunks = _chunk_by_treesitter(text, file_extension)
        
        for idx, chunk in enumerate(chunks):
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "path": str(file_path),
                    "start": chunk.get("start", 0),
                    "end": chunk.get("end", 0),
                    "type": chunk.get("type", ""),
                    "name": chunk.get("name", ""),
                    "line_start": chunk.get("line_start", 0),
                    "line_end": chunk.get("line_end", 0),
                }
            )
            ids.append(f"{file_path}:{idx}")

    if not documents:
        return 0

    # Compute embeddings in batches to keep memory usage reasonable.
    embeddings: List[List[float]] = []
    batch_size = 64
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_embeddings = provider.encode(batch_docs)
        # Ensure 2D numpy array, then convert to list
        if not isinstance(batch_embeddings, np.ndarray):
            batch_embeddings = np.array(batch_embeddings)
        if batch_embeddings.ndim == 1:
            batch_embeddings = batch_embeddings.reshape(1, -1)
        embeddings.extend(batch_embeddings.astype("float32").tolist())

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )

    return len(documents)


def list_indexed_files(
    *,
    db_path: Optional[Path] = None,
    collection_name: str = "codebase-rag",
) -> List[Dict[str, Any]]:
    """Return a list of unique indexed files with basic metadata."""
    client = _get_client(db_path)
    collection = _get_collection(client, collection_name)

    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []

    files: Dict[str, Dict[str, Any]] = {}
    for meta in metadatas:
        if not meta:
            continue
        path = meta.get("path")
        if not path:
            continue
        if path not in files:
            files[path] = {"path": path}

    return list(files.values())


def search_codebase(
    *,
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    db_path: Optional[Path] = None,
    collection_name: str = "codebase-rag",
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> Dict[str, Any]:
    """Search indexed codebase using hybrid dense + sparse search with RRF."""
    start_time = time.time()
    
    provider = embedding_provider or EmbeddingProvider()
    client = _get_client(db_path)
    collection = _get_collection(client, collection_name)

    # 1. Dense search (ChromaDB)
    query_embedding = provider.encode([query])
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.array(query_embedding)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    dense_result = collection.query(
        query_embeddings=query_embedding.astype("float32").tolist(),
        n_results=top_k * 2,  # fetch extra candidates for RRF
        include=["documents", "metadatas", "distances"],
    )

    dense_documents = dense_result.get("documents", [[]])[0]
    dense_metadatas = dense_result.get("metadatas", [[]])[0]
    dense_distances = dense_result.get("distances", [[]])[0]

    # Prepare dense results for RRF
    dense_results = []
    for i, (doc, meta, dist) in enumerate(zip(dense_documents, dense_metadatas, dense_distances)):
        dense_results.append({
            "chunk_index": i,
            "path": (meta or {}).get("path", ""),
            "content": doc,
            "score": float(1.0 - float(dist)),  # convert distance to similarity
        })

    # 2. Sparse search (BM25)
    sparse_results = []
    try:
        # Get all chunks for BM25 indexing
        all_chunks = []
        for meta in collection.get(include=["metadatas"]).get("metadatas", []):
            if meta:
                chunk_text = collection.get(
                    ids=[meta.get("path", "unknown")],
                    include=["documents"]
                ).get("documents", [""])[0]
                all_chunks.append({
                    "text": chunk_text,
                    "path": meta.get("path", ""),
                })
        
        if all_chunks and BM25Okapi is not None:
            # Create BM25 index
            bm25_index = BM25Index()
            bm25_index.index(all_chunks)
            
            # Search BM25
            sparse_results = bm25_index.search(query, top_k * 2)
    except Exception:
        sparse_results = []

    # 3. RRF fusion
    # Reconstruct chunks list for RRF (simplified approach)
    chunks_for_rrf = []
    for i, (doc, meta) in enumerate(zip(dense_documents, dense_metadatas)):
        if meta:
            chunks_for_rrf.append({
                "text": doc,
                "path": meta.get("path", ""),
                "type": meta.get("type", ""),
                "name": meta.get("name", ""),
                "line_start": meta.get("line_start", 0),
                "line_end": meta.get("line_end", 0),
            })

    # Apply RRF
    if sparse_results:
        fused_results = reciprocal_rank_fusion(
            dense_results, sparse_results, chunks_for_rrf, k=60, top_k=top_k
        )
    else:
        # Fallback to dense-only if BM25 fails
        fused_results = dense_results[:top_k]

    query_time_ms = (time.time() - start_time) * 1000
    
    return {
        "results": fused_results,
        "total_indexed_chunks": len(collection.get(include=["metadatas"]).get("metadatas", [])),
        "query_time_ms": round(query_time_ms, 2),
        "search_type": "hybrid_rrf" if sparse_results else "dense_only",
    }


def get_file_content(path: str) -> str:
    """Read and return the full content of a file from disk."""
    file_path = Path(path)
    return file_path.read_text(encoding="utf-8", errors="ignore")
