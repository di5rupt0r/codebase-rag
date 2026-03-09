#!/usr/bin/env python3
"""Install a post-commit Git hook that auto-reindexes changed code files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REINDEX_SCRIPT = Path(__file__).parent.resolve() / "reindex_changed.py"

HOOK_TEMPLATE = """\
#!/bin/bash
# Auto-reindex changed code files after each commit (installed by setup_git_hook.py)
changed_files=$(git diff-tree --no-commit-id --name-only -r HEAD)
if echo "$changed_files" | grep -qE '\\.(py|js|ts|jsx|tsx|java|cpp|c|h|go|rs|rb|php|cs)$'; then
    {python} {reindex_script} --project {project_name} --files "$changed_files" &
fi
"""


def install_hook(repo_path: str, project_name: str) -> None:
    """Install post-commit hook in the given Git repository."""
    repo = Path(repo_path).expanduser().resolve()
    git_dir = repo / ".git"

    if not git_dir.is_dir():
        print(f"Error: {repo} is not a Git repository", file=sys.stderr)
        sys.exit(1)

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    hook_path = hooks_dir / "post-commit"

    hook_content = HOOK_TEMPLATE.format(
        python=sys.executable,
        reindex_script=REINDEX_SCRIPT,
        project_name=project_name,
    )

    hook_path.write_text(hook_content)
    hook_path.chmod(0o755)
    print(f"✓ post-commit hook installed at {hook_path}")
    print(f"  Project: {project_name}")
    print(f"  Reindex script: {REINDEX_SCRIPT}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install codebase-rag post-commit hook in a Git repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_git_hook.py ~/my-project --name my-project
  python setup_git_hook.py . --name my-project
        """,
    )
    parser.add_argument("repo_path", help="Path to the Git repository")
    parser.add_argument("--name", required=True, help="Project name (ChromaDB collection)")
    args = parser.parse_args()
    install_hook(args.repo_path, args.name)


if __name__ == "__main__":
    main()
