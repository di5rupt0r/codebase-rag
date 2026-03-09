#!/usr/bin/env python3
"""Auto-discover and index Git repositories in specified scan paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from codebase_rag.indexer import _get_client, _get_collection, index_codebase


def discover_git_repos(scan_paths: list[str]) -> list[dict]:
    """Scan directories and return all Git repositories found."""
    repos = []
    for base in scan_paths:
        base_path = Path(base).expanduser().resolve()
        if not base_path.exists():
            print(f"Warning: scan path does not exist: {base_path}", file=sys.stderr)
            continue
        for git_dir in base_path.rglob(".git"):
            if git_dir.is_dir():
                repo_path = git_dir.parent
                repos.append({"name": repo_path.name, "path": str(repo_path)})
    return repos


def get_indexed_names() -> set[str]:
    """Return names of already-indexed projects from ChromaDB."""
    try:
        client = _get_client()
        return {col.name for col in client.list_collections()}
    except Exception:
        return set()


def auto_index(
    scan_paths: list[str],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Index all Git repos found in scan_paths that are not yet indexed."""
    repos = discover_git_repos(scan_paths)
    if not repos:
        print("No Git repositories found.")
        return

    indexed = set() if force else get_indexed_names()

    for repo in repos:
        name = repo["name"]
        path = repo["path"]
        if name in indexed:
            print(f"Skipping {name} (already indexed — use --force to reindex)")
            continue

        if dry_run:
            print(f"[dry-run] Would index: {name}  ({path})")
            continue

        print(f"Indexing {name} at {path}...")
        try:
            count = index_codebase(Path(path), collection_name=name)
            print(f"  ✓ {count} chunks indexed")
        except Exception as exc:
            print(f"  ✗ Failed to index {name}: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-discover and index Git repositories for codebase-rag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan two directories
  python auto_index.py --scan ~/projects ~/mcp-servers

  # Dry run: show what would be indexed without doing it
  python auto_index.py --scan ~/projects --dry-run

  # Force reindex even if already indexed
  python auto_index.py --scan ~/projects --force
        """,
    )
    parser.add_argument(
        "--scan",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Directories to scan for Git repositories",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reindex projects even if already indexed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without actually doing it",
    )

    args = parser.parse_args()
    auto_index(args.scan, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
