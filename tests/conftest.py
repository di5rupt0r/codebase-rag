from __future__ import annotations

import sys
from pathlib import Path


def _add_src_to_sys_path() -> None:
    """Ensure the `src` directory is on `sys.path` for imports in tests."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if src_path.is_dir():
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_add_src_to_sys_path()

