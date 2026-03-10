"""Tests for the project registry module.

TDD — these tests are written before the implementation.
All tests in this file should FAIL until registry.py is created.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from codebase_rag import registry


class TestLoadRegistry:
    """load_registry reads JSON sidecar; returns empty dict when missing."""

    def test_returns_empty_dict_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("codebase_rag.registry._registry_path", lambda: tmp_path / "project_registry.json")
        assert registry.load_registry() == {}

    def test_returns_parsed_dict_when_file_exists(self, tmp_path, monkeypatch):
        reg_file = tmp_path / "project_registry.json"
        reg_file.write_text(json.dumps({"my-project": {"project_path": "/some/path"}}))
        monkeypatch.setattr("codebase_rag.registry._registry_path", lambda: reg_file)

        result = registry.load_registry()
        assert result == {"my-project": {"project_path": "/some/path"}}


class TestUpdateRegistry:
    """update_registry upserts project info and persists to disk."""

    def test_creates_new_entry_with_timestamps(self, tmp_path, monkeypatch):
        reg_file = tmp_path / "project_registry.json"
        monkeypatch.setattr("codebase_rag.registry._registry_path", lambda: reg_file)

        registry.update_registry("my-project", "/home/user/my-project")

        data = json.loads(reg_file.read_text())
        assert "my-project" in data
        assert data["my-project"]["project_path"] == "/home/user/my-project"
        assert "first_indexed" in data["my-project"]
        assert "last_indexed" in data["my-project"]

    def test_updates_existing_entry_preserves_first_indexed(self, tmp_path, monkeypatch):
        reg_file = tmp_path / "project_registry.json"
        original = {
            "my-project": {
                "project_path": "/old/path",
                "first_indexed": "2026-01-01T00:00:00+00:00",
                "last_indexed": "2026-01-01T00:00:00+00:00",
            }
        }
        reg_file.write_text(json.dumps(original))
        monkeypatch.setattr("codebase_rag.registry._registry_path", lambda: reg_file)

        registry.update_registry("my-project", "/new/path")

        data = json.loads(reg_file.read_text())
        # first_indexed must NOT change on update
        assert data["my-project"]["first_indexed"] == "2026-01-01T00:00:00+00:00"
        # project_path and last_indexed are refreshed
        assert data["my-project"]["project_path"] == "/new/path"
        assert data["my-project"]["last_indexed"] != "2026-01-01T00:00:00+00:00"

    def test_creates_parent_dirs_if_missing(self, tmp_path, monkeypatch):
        reg_file = tmp_path / "nested" / "dir" / "project_registry.json"
        monkeypatch.setattr("codebase_rag.registry._registry_path", lambda: reg_file)

        registry.update_registry("proj", "/some/path")

        assert reg_file.exists()

    def test_registry_is_valid_json(self, tmp_path, monkeypatch):
        reg_file = tmp_path / "project_registry.json"
        monkeypatch.setattr("codebase_rag.registry._registry_path", lambda: reg_file)

        registry.update_registry("proj", "/some/path")

        # Must not raise
        json.loads(reg_file.read_text())


class TestGetProjectPath:
    """get_project_path returns stored path or None."""

    def test_returns_path_for_known_project(self, tmp_path, monkeypatch):
        reg_file = tmp_path / "project_registry.json"
        reg_file.write_text(json.dumps({"proj": {"project_path": "/a/b/c"}}))
        monkeypatch.setattr("codebase_rag.registry._registry_path", lambda: reg_file)

        assert registry.get_project_path("proj") == "/a/b/c"

    def test_returns_none_for_unknown_project(self, tmp_path, monkeypatch):
        reg_file = tmp_path / "project_registry.json"
        reg_file.write_text(json.dumps({}))
        monkeypatch.setattr("codebase_rag.registry._registry_path", lambda: reg_file)

        assert registry.get_project_path("unknown") is None

    def test_returns_none_when_registry_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("codebase_rag.registry._registry_path", lambda: tmp_path / "project_registry.json")

        assert registry.get_project_path("proj") is None
