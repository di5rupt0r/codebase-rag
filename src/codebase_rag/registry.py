"""Project registry — maps collection names to filesystem paths.

ChromaDB does not persist collection-level metadata reliably across versions,
so we use a JSON sidecar file (``project_registry.json``) stored alongside the
chroma_db directory to keep a durable record of which collection name maps to
which project root path, plus first/last indexed timestamps.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from . import config


def _registry_path() -> Path:
    return Path(config.get_chroma_db_path()) / "project_registry.json"


def load_registry() -> Dict[str, dict]:
    """Load the project registry from disk. Returns an empty dict if not found."""
    p = _registry_path()
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _save_registry(data: Dict[str, dict]) -> None:
    p = _registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def update_registry(project_name: str, project_path: str) -> None:
    """Upsert a project entry in the registry with the current timestamp."""
    reg = load_registry()
    now = datetime.now(timezone.utc).isoformat()
    if project_name not in reg:
        reg[project_name] = {"first_indexed": now}
    reg[project_name]["project_path"] = project_path
    reg[project_name]["last_indexed"] = now
    _save_registry(reg)


def get_project_path(project_name: str) -> Optional[str]:
    """Return the stored path for a project, or None if not found."""
    return load_registry().get(project_name, {}).get("project_path")
