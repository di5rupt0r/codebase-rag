#!/usr/bin/env python3
"""Watch mode: monitor code changes and reindex files automatically with debouncing."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Add project root to path so imports work when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from codebase_rag import config
from codebase_rag.indexer import _get_client, _get_collection, chunk_file, _get_collection, EmbeddingProvider


_DEBOUNCE_SECONDS = 5


class _CodeChangeHandler(FileSystemEventHandler):
    """Handle filesystem events and debounce reindexing."""

    def __init__(
        self,
        project_name: str,
        debounce_seconds: float = _DEBOUNCE_SECONDS,
    ) -> None:
        self.project_name = project_name
        self.debounce = debounce_seconds
        self._pending: set[str] = set()
        self._deleted: set[str] = set()
        self._last_change: float = time.time()

    def _is_code_file(self, path: str) -> bool:
        return Path(path).suffix.lower() in config.SUPPORTED_EXTENSIONS

    def on_modified(self, event) -> None:
        if event.is_directory or not self._is_code_file(event.src_path):
            return
        self._pending.add(event.src_path)
        self._last_change = time.time()

    def on_created(self, event) -> None:
        self.on_modified(event)

    def on_deleted(self, event) -> None:
        if event.is_directory or not self._is_code_file(event.src_path):
            return
        self._deleted.add(event.src_path)
        self._pending.discard(event.src_path)
        self._last_change = time.time()

    def process_pending(self) -> None:
        if time.time() - self._last_change < self.debounce:
            return

        if self._deleted:
            self._flush_deletes()

        if self._pending:
            self._flush_pending()

    def _flush_deletes(self) -> None:
        client = _get_client()
        collection = _get_collection(client, self.project_name)
        for path in self._deleted:
            try:
                collection.delete(where={"path": path})
                print(f"[{_ts()}] Removed {path} from index")
            except Exception as exc:
                print(f"[{_ts()}] Failed to remove {path}: {exc}", file=sys.stderr)
        self._deleted.clear()

    def _flush_pending(self) -> None:
        provider = EmbeddingProvider()
        client = _get_client()
        collection = _get_collection(client, self.project_name)

        print(f"[{_ts()}] Reindexing {len(self._pending)} changed file(s)...")
        for path_str in self._pending:
            file_path = Path(path_str)
            if not file_path.exists():
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                # Remove stale chunks for this file
                collection.delete(where={"path": path_str})
                # Re-chunk and embed
                chunks = chunk_file(file_path, content)
                if not chunks:
                    continue
                docs = [c["content"] for c in chunks]
                import hashlib
                file_hash = hashlib.sha256(content.encode()).hexdigest()
                metas = [
                    {
                        "path": path_str,
                        "line_start": c["line_start"],
                        "line_end": c["line_end"],
                        "type": c.get("type", "block"),
                        "name": c.get("name", ""),
                        "docstring": c.get("docstring", ""),
                        "file_hash": file_hash,
                        "imports": ",".join(c.get("imports", [])),
                        "calls": ",".join(c.get("calls", [])),
                    }
                    for c in chunks
                ]
                import numpy as np
                embeddings = provider.encode(docs).astype("float32").tolist()
                ids = [f"{path_str}:{c['line_start']}:{i}" for i, c in enumerate(chunks)]
                collection.add(documents=docs, metadatas=metas, ids=ids, embeddings=embeddings)
                print(f"[{_ts()}]   ✓ {file_path.name} — {len(chunks)} chunk(s)")
            except Exception as exc:
                print(f"[{_ts()}]   ✗ {file_path.name}: {exc}", file=sys.stderr)

        self._pending.clear()


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def watch(project_path: str, project_name: str, debounce: float = _DEBOUNCE_SECONDS) -> None:
    """Start watching project_path and reindex changes for project_name."""
    path = Path(project_path).expanduser().resolve()
    if not path.exists():
        print(f"Error: path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    handler = _CodeChangeHandler(project_name, debounce_seconds=debounce)
    observer = Observer()
    observer.schedule(handler, str(path), recursive=True)
    observer.start()

    print(f"[{_ts()}] Watching {path} as '{project_name}' (debounce={debounce}s)")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
            handler.process_pending()
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        print(f"\n[{_ts()}] Stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch a project directory and auto-reindex changed files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python watch.py ~/my-project --name my-project
  python watch.py ~/my-project --name my-project --debounce 10
        """,
    )
    parser.add_argument("path", help="Project directory to watch")
    parser.add_argument("--name", required=True, help="Project name (ChromaDB collection)")
    parser.add_argument(
        "--debounce",
        type=float,
        default=_DEBOUNCE_SECONDS,
        metavar="SECONDS",
        help=f"Seconds to wait after last change before reindexing (default: {_DEBOUNCE_SECONDS})",
    )
    args = parser.parse_args()
    watch(args.path, args.name, debounce=args.debounce)


if __name__ == "__main__":
    main()
