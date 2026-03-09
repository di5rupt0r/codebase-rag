#!/usr/bin/env python3
"""CLI script to index a project for use with MCP Codebase RAG server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from codebase_rag.indexer import index_codebase


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Index a codebase for semantic search with MCP Codebase RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index current directory as project 'my-app'
  python index_project.py . --name my-app
  
  # Index specific path with custom collection name
  python index_project.py ~/projects/my-app --name my-app --force
  
  # Dry run to see what would be indexed
  python index_project.py . --name my-app --dry-run
        """
    )
    
    parser.add_argument(
        "path",
        type=str,
        help="Path to the project directory to index"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the project (used as collection name in ChromaDB)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing index before reindexing"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without actually indexing"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress information"
    )
    
    args = parser.parse_args()
    
    # Validate path
    project_path = Path(args.path).expanduser().resolve()
    if not project_path.exists():
        print(f"Error: Path does not exist: {project_path}", file=sys.stderr)
        sys.exit(1)
    
    if not project_path.is_dir():
        print(f"Error: Path is not a directory: {project_path}", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"Project path: {project_path}")
        print(f"Project name: {args.name}")
        print(f"Force reindex: {args.force}")
        print()
    
    if args.dry_run:
        print("DRY RUN - No actual indexing will be performed")
        print()
    
    try:
        if args.dry_run:
            # Just count files that would be indexed
            from codebase_rag.indexer import _iter_source_files
            files_to_index = list(_iter_source_files(project_path))
            
            print(f"Files to be indexed: {len(files_to_index)}")
            if args.verbose:
                print("\nFiles that would be indexed:")
                for file_path in sorted(files_to_index):
                    rel_path = file_path.relative_to(project_path)
                    print(f"  {rel_path}")
        else:
            # Perform actual indexing
            print(f"Indexing project '{args.name}' at {project_path}...")
            
            if args.force:
                print("Force mode: deleting existing collection...")
            
            chunks_created = index_codebase(
                project_path, 
                collection_name=args.name
            )
            
            print(f"✓ Successfully indexed project")
            print(f"  Chunks created: {chunks_created}")
            
            if args.verbose:
                from codebase_rag.indexer import list_indexed_files
                files = list_indexed_files(collection_name=args.name)
                print(f"  Files indexed: {len(files)}")
    
    except KeyboardInterrupt:
        print("\nIndexing interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during indexing: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
