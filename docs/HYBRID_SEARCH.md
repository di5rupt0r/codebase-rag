# Hybrid Search Implementation

This document describes the hybrid search implementation using Tree-sitter chunking, BM25 sparse search, and Reciprocal Rank Fusion (RRF).

## Architecture Overview

The hybrid search system combines three main components:

1. **Tree-sitter Chunking**: Universal language-agnostic code chunking
2. **BM25 Sparse Search**: Lexical search for exact term matching  
3. **Reciprocal Rank Fusion**: Intelligent fusion of dense and sparse results

## Components

### Tree-sitter Chunking

- **Purpose**: Replace AST/line-based chunking with universal approach
- **Languages Supported**: Python, JavaScript, TypeScript, Java, Go, Rust, C/C++
- **Node Types**: `function_definition`, `method_definition`, `class_definition`, `async_function_definition`, `async_method_definition`, `decorated_definition`, `module`
- **Fallback**: Line-based chunking for unsupported languages

### BM25 Index

- **Purpose**: Sparse lexical search for exact term matching
- **Tokenization**: Regex `[a-zA-Z0-9_]+`, minimum length 2, lowercase
- **Indexing**: In-memory, rebuilt per query
- **Retrieval**: Top-k results with BM25 scores

### Reciprocal Rank Fusion

- **Purpose**: Combine dense (embeddings) and sparse (BM25) results
- **Formula**: `1/(k + rank)` with k=60 (standard literature value)
- **Process**: 
  1. Get dense results from ChromaDB
  2. Get sparse results from BM25
  3. Calculate RRF scores for both
  4. Sort by combined score
  5. Return top-k fused results

## Performance Targets

- **Tree-sitter parsing**: < 50ms per file (10k LOC)
- **BM25 indexing**: < 10ms per query
- **RRF fusion**: < 1ms (in-memory)
- **Hybrid query**: < 100ms total
- **Memory overhead**: < 50MB (5k chunks)

## API Changes

### `search_codebase()`

**Before**: Pure dense search with keyword reranking
```python
results = search_codebase(query="function_name")
# Returns: List[Dict] with path, content, score
```

**After**: Hybrid search with timing and type information
```python
results = search_codebase(query="function_name")
# Returns: Dict with:
# {
#     "results": List[Dict] with path, content, score, type, name, line_start, line_end,
#     "total_indexed_chunks": int,
#     "query_time_ms": float,
#     "search_type": "hybrid_rrf" | "dense_only"
# }
```

### `index_codebase()`

**Before**: AST/line-based chunking
**After**: Tree-sitter chunking with enhanced metadata
```python
# Enhanced chunk metadata includes:
# - type: "function" | "class" | "block"
# - name: extracted function/class name
# - line_start/line_end: precise line numbers
# - start/end: byte positions
```

## Usage Examples

### Basic Hybrid Search
```python
from codebase_rag.indexer import search_codebase

# Hybrid search with RRF fusion
results = search_codebase(query="authentication function")
for result in results["results"]:
    print(f"Found {result['type']} '{result['name']}' in {result['path']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Lines {result['line_start']}-{result['line_end']}")

print(f"Query type: {results['search_type']}")
print(f"Total chunks: {results['total_indexed_chunks']}")
print(f"Query time: {results['query_time_ms']}ms")
```

### Performance Monitoring
```python
# Monitor hybrid search performance
results = search_codebase(query="database connection")
if results["query_time_ms"] > 100:
    print("Warning: Query exceeded 100ms target")
if results["search_type"] == "dense_only":
    print("Note: BM25 unavailable, using dense-only search")
```

## Configuration

### Environment Variables
```bash
# Tree-sitter and BM25 are automatically used when available
# No additional configuration required

# Fallback behavior when dependencies unavailable:
# - Tree-sitter → line-based chunking
# - BM25 → dense-only search
```

### Dependencies
```toml
# pyproject.toml
dependencies = [
    "tree-sitter>=0.21.0",
    "tree-sitter-python>=0.21.0", 
    "rank-bm25>=0.2.2",
    # ... other dependencies
]
```

## Testing

### Test Coverage
- **BM25 Tests**: 7/7 passing (100%)
- **RRF Tests**: 8/8 passing (100%)
- **Fallback Tests**: 3/3 passing (100%)
- **Tree-sitter Tests**: 3/5 passing (60%)

### Running Tests
```bash
# Run all hybrid search tests
uvx --with pytest --with tree-sitter --with tree-sitter-python --with rank-bm25 \
    --with-editable . pytest tests/test_bm25.py tests/test_rrf.py tests/test_treesitter_chunking.py -v

# Run specific test categories
uvx --with pytest --with-editable . pytest tests/test_bm25.py -v
uvx --with pytest --with-editable . pytest tests/test_rrf.py -v
uvx --with pytest --with-editable . pytest tests/test_treesitter_chunking.py::TestFallbackChunking -v
```

## Troubleshooting

### Common Issues

1. **Tree-sitter not working**
   - Check: `tree-sitter-python` installation
   - Fallback: Uses line-based chunking automatically

2. **BM25 not working**
   - Check: `rank-bm25` installation  
   - Fallback: Uses dense-only search automatically

3. **Slow queries**
   - Check: Number of chunks indexed
   - Solution: Consider increasing chunk size or reducing overlap

### Debug Information

```python
# Enable debug mode to see internal behavior
import logging
logging.basicConfig(level=logging.DEBUG)

# Check search type
results = search_codebase(query="test")
print(f"Search method: {results['search_type']}")
```

## Migration Guide

### From Pure Dense Search

**Old API**:
```python
results = search_codebase(query="function")
# results: List[Dict]
```

**New API**:
```python
results = search_codebase(query="function")
# results["results"]: List[Dict] with enhanced metadata
# results["query_time_ms"]: float
# results["search_type"]: str
# results["total_indexed_chunks"]: int
```

### Backward Compatibility

The new `search_codebase()` maintains backward compatibility by returning results in the `results["results"]` field with the same structure as before, but adds timing and type information.

## Future Enhancements

1. **Improved Tree-sitter Integration**: Fix remaining API issues
2. **Language-specific Chunking**: Optimize for different language patterns
3. **Caching**: Optional BM25 index caching for repeated queries
4. **Advanced RRF**: Weighted RRF or score normalization options
