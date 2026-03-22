"""Tests for MCP server tools."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from codebase_rag.server import (
    search_codebase,
    reindex_project,
    list_indexed_projects,
    get_files,
    get_file_content,
)


class TestSearchCodebase:
    """Test search_codebase MCP tool."""

    @patch("codebase_rag.server._search_codebase")
    @patch("codebase_rag.server._list_indexed_files")
    def test_search_codebase_basic(self, mock_list_files, mock_search):
        """Test basic search functionality."""
        mock_search.return_value = [
            {"path": "file1.py", "content": "def foo():", "score": 0.9}
        ]
        mock_list_files.return_value = [{"path": "file1.py"}]
        
        result = search_codebase(query="test query")
        
        assert "results" in result
        assert "total_indexed_chunks" in result
        assert "query_time_ms" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["path"] == "file1.py"

    @patch("codebase_rag.server._search_codebase")
    @patch("codebase_rag.server._list_indexed_files")
    def test_search_codebase_with_project(self, mock_list_files, mock_search):
        """Test search with project parameter."""
        mock_search.return_value = []
        mock_list_files.return_value = []
        
        search_codebase(query="test", project="my-project")
        
        mock_search.assert_called_once_with(query="test", top_k=5, collection_name="my-project")

    @patch("codebase_rag.server._search_codebase")
    @patch("codebase_rag.server._list_indexed_files")
    def test_search_codebase_with_file_types(self, mock_list_files, mock_search):
        """Test search with file type filtering."""
        mock_search.return_value = [
            {"path": "file1.py", "content": "def foo():", "score": 0.9},
            {"path": "file2.js", "content": "function bar() {}", "score": 0.8},
            {"path": "file3.txt", "content": "some text", "score": 0.7},
        ]
        mock_list_files.return_value = []
        
        result = search_codebase(query="test", file_types=[".py", ".js"])
        
        # Should filter out .txt file
        assert len(result["results"]) == 2
        assert all(Path(r["path"]).suffix in [".py", ".js"] for r in result["results"])

    @patch("codebase_rag.server._search_codebase")
    def test_search_codebase_error_handling(self, mock_search):
        """Test search error handling."""
        mock_search.side_effect = Exception("Search failed")
        
        result = search_codebase(query="test")
        
        assert "error" in result
        assert result["error"] == "Search failed"
        assert result["results"] == []

    @patch("codebase_rag.server._search_codebase")
    @patch("codebase_rag.server._list_indexed_files")
    def test_search_codebase_timing(self, mock_list_files, mock_search):
        """Test search timing is recorded."""
        mock_search.return_value = []
        mock_list_files.return_value = []
        
        result = search_codebase(query="test")
        
        assert "query_time_ms" in result
        assert isinstance(result["query_time_ms"], (int, float))
        assert result["query_time_ms"] >= 0


class TestReindexProject:
    """Test reindex_project MCP tool."""

    @patch("codebase_rag.server._index_codebase")
    @patch("codebase_rag.server._list_indexed_files")
    def test_reindex_project_basic(self, mock_list_files, mock_index):
        """Test basic reindexing."""
        mock_index.return_value = 10  # 10 chunks created
        mock_list_files.return_value = [{"path": "file1.py"}, {"path": "file2.py"}]
        
        with TemporaryDirectory() as tmp_dir:
            result = reindex_project(tmp_dir, "test-project")
            
            assert result["status"] == "success"
            assert result["files_indexed"] == 2
            assert result["chunks_created"] == 10
            assert "time_seconds" in result
            assert result["force_used"] is False

    def test_reindex_project_nonexistent_path(self):
        """Test reindexing with nonexistent path."""
        result = reindex_project("/nonexistent/path", "test-project")
        
        assert result["status"] == "error"
        assert "does not exist" in result["error"]
        assert result["time_seconds"] == 0

    @patch("codebase_rag.server._resolve_client")
    @patch("codebase_rag.server._index_codebase")
    @patch("codebase_rag.server._list_indexed_files")
    def test_reindex_project_with_force(self, mock_list_files, mock_index, mock_get_client):
        """Test reindexing with force=True deletes existing collection."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_index.return_value = 5
        mock_list_files.return_value = [{"path": "file1.py"}]

        with TemporaryDirectory() as tmp_dir:
            result = reindex_project(tmp_dir, "test-project", force=True)

            assert result["status"] == "success"
            assert result["force_used"] is True
            # delete_collection must have been attempted
            mock_client.delete_collection.assert_called_once_with(name="test-project")

    @patch("codebase_rag.server._index_codebase")
    def test_reindex_project_error_handling(self, mock_index):
        """Test reindexing error handling."""
        mock_index.side_effect = Exception("Indexing failed")
        
        with TemporaryDirectory() as tmp_dir:
            result = reindex_project(tmp_dir, "test-project")
            
            assert result["status"] == "error"
            assert result["error"] == "Indexing failed"
            assert "time_seconds" in result


class TestListIndexedProjects:
    """Test list_indexed_projects MCP tool."""

    @patch("codebase_rag.server._resolve_client")
    def test_list_indexed_projects_basic(self, mock_get_client):
        """Test project listing enumerates real ChromaDB collections."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        mock_col = Mock()
        mock_col.name = "my-project"
        mock_client.list_collections.return_value = [mock_col]

        mock_col_obj = Mock()
        mock_client.get_collection.return_value = mock_col_obj
        mock_col_obj.get.return_value = {
            "metadatas": [
                {"path": "/src/file1.py"},
                {"path": "/src/file2.py"},
                {"path": "/src/file1.py"},  # duplicate chunk of same file
            ]
        }

        result = list_indexed_projects()

        assert result["total_projects"] == 1
        project = result["projects"][0]
        assert project["name"] == "my-project"
        assert project["files"] == 2   # unique paths
        assert project["chunks"] == 3  # total metadata entries

    @patch("codebase_rag.server._resolve_client")
    def test_list_indexed_projects_empty(self, mock_get_client):
        """Test listing with no indexed collections."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.list_collections.return_value = []

        result = list_indexed_projects()

        assert result["total_projects"] == 0
        assert result["projects"] == []

    @patch("codebase_rag.server._resolve_client")
    def test_list_indexed_projects_error(self, mock_get_client):
        """Test error handling in project listing."""
        mock_get_client.side_effect = Exception("Connection failed")

        result = list_indexed_projects()

        assert "error" in result
        assert result["projects"] == []
        assert result["total_projects"] == 0


class TestGetFiles:
    """Test get_files MCP tool."""

    @patch("codebase_rag.server._list_indexed_files")
    def test_get_files_basic(self, mock_list_files):
        """get_files requires explicit project; results come from the right collection."""
        mock_list_files.return_value = [
            {"path": "file1.py"},
            {"path": "file2.js"},
        ]

        result = get_files(project="my-project")

        assert len(result) == 2
        assert result[0]["path"] == "file1.py"
        assert result[1]["path"] == "file2.js"
        mock_list_files.assert_called_once_with(collection_name="my-project")

    @patch("codebase_rag.server._list_indexed_files")
    def test_get_files_with_project(self, mock_list_files):
        """Test file listing with project parameter."""
        mock_list_files.return_value = [{"path": "project_file.py"}]
        
        get_files(project="my-project")
        
        mock_list_files.assert_called_once_with(collection_name="my-project")

    @patch("codebase_rag.server._list_indexed_files")
    def test_get_files_error_handling(self, mock_list_files):
        """When _list_indexed_files raises, the exception propagates."""
        mock_list_files.side_effect = Exception("List failed")

        with pytest.raises(Exception, match="List failed"):
            get_files(project="my-project")


class TestGetFileContent:
    """Test get_file_content MCP tool."""

    def test_get_file_content_basic(self):
        """Test basic file content reading."""
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.py"
            file_path.write_text("print('hello world')")
            
            content = get_file_content(str(file_path))
            
            assert content == "print('hello world')"

    def test_get_file_content_nonexistent(self):
        """Test reading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            get_file_content("/nonexistent/file.py")

    def test_get_file_content_encoding_errors(self):
        """Test handling encoding errors."""
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.py"
            # Write invalid UTF-8 bytes
            file_path.write_bytes(b'\xff\xfe\x00\x00')
            
            # Should not raise, but return empty or partial content
            content = get_file_content(str(file_path))
            assert isinstance(content, str)


class TestGetFilesRequiresProject:
    """Bug 1 — get_files must require an explicit project; old default 'codebase-rag' is wrong."""

    def test_get_files_without_project_returns_error(self):
        """Calling get_files() with no project must return an error dict, not silently
        fall back to the server's own 'codebase-rag' collection."""
        result = get_files()
        assert isinstance(result, dict)
        assert "error" in result

    @patch("codebase_rag.server._list_indexed_files")
    def test_get_files_with_project_uses_that_collection(self, mock_list_files):
        """get_files(project='x') must query collection 'x', not 'codebase-rag'."""
        mock_list_files.return_value = [{"path": "a.py"}]
        result = get_files(project="x")
        mock_list_files.assert_called_once_with(collection_name="x")
        assert isinstance(result, list)


class TestListIndexedProjectsShowsPath:
    """Bug 3 — list_indexed_projects must return the real path from the registry."""

    @patch("codebase_rag.server._resolve_client")
    @patch("codebase_rag.server.registry.get_project_path")
    def test_path_comes_from_registry(self, mock_get_path, mock_get_client):
        """list_indexed_projects must call registry.get_project_path and return the result."""
        mock_get_path.return_value = "/real/project/path"

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_col = Mock()
        mock_col.name = "my-project"
        mock_client.list_collections.return_value = [mock_col]
        mock_col_obj = Mock()
        mock_client.get_collection.return_value = mock_col_obj
        mock_col_obj.get.return_value = {"metadatas": [{"path": "/real/project/path/main.py"}]}

        result = list_indexed_projects()

        assert result["projects"][0]["path"] == "/real/project/path"
        mock_get_path.assert_called_once_with("my-project")

    @patch("codebase_rag.server._resolve_client")
    @patch("codebase_rag.server.registry.get_project_path")
    def test_path_is_na_when_not_in_registry(self, mock_get_path, mock_get_client):
        """When a collection is not in the registry, path falls back to 'N/A'."""
        mock_get_path.return_value = None

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_col = Mock()
        mock_col.name = "old-project"
        mock_client.list_collections.return_value = [mock_col]
        mock_col_obj = Mock()
        mock_client.get_collection.return_value = mock_col_obj
        mock_col_obj.get.return_value = {"metadatas": [{"path": "/some/file.py"}]}

        result = list_indexed_projects()

        assert result["projects"][0]["path"] == "N/A"


class TestReindexWritesRegistry:
    """reindex_project must persist the project_path to the registry."""

    @patch("codebase_rag.server.registry.update_registry")
    @patch("codebase_rag.server._index_codebase")
    @patch("codebase_rag.server._list_indexed_files")
    def test_reindex_calls_update_registry(self, mock_list, mock_index, mock_update_reg):
        mock_index.return_value = 5
        mock_list.return_value = [{"path": "f.py"}]

        with TemporaryDirectory() as tmp_dir:
            reindex_project(tmp_dir, "my-project")

        mock_update_reg.assert_called_once_with("my-project", str(Path(tmp_dir).resolve()))


class TestGetFileContentWithProject:
    """Bug 2 — get_file_content tool must accept optional project param for validation."""

    @patch("codebase_rag.server._get_file_content")
    def test_get_file_content_passes_project_to_indexer(self, mock_get_content):
        """The server tool must forward the project param to the indexer."""
        mock_get_content.return_value = "code"
        get_file_content("/some/file.py", project="my-project")
        mock_get_content.assert_called_once_with("/some/file.py", project="my-project")

    @patch("codebase_rag.server._get_file_content")
    def test_get_file_content_without_project_still_works(self, mock_get_content):
        mock_get_content.return_value = "code"
        get_file_content("/some/file.py")
        mock_get_content.assert_called_once_with("/some/file.py", project=None)


class TestServerIntegration:
    """Test server integration scenarios."""

    @patch("codebase_rag.server._search_codebase")
    @patch("codebase_rag.server._list_indexed_files")
    @patch("codebase_rag.server._index_codebase")
    def test_full_workflow(self, mock_index, mock_list_files, mock_search):
        """Test complete workflow: index -> list -> search."""
        # 1. Index a project
        mock_index.return_value = 5
        mock_list_files.return_value = [{"path": "file1.py"}]
        
        with TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "test.py").write_text("print('test')")
            
            index_result = reindex_project(tmp_dir, "test-project")
            assert index_result["status"] == "success"
        
        # 2. List files
        files = get_files(project="test-project")
        assert len(files) == 1
        
        # 3. Search
        mock_search.return_value = [
            {"path": "file1.py", "content": "print('test')", "score": 0.9}
        ]
        
        search_result = search_codebase(query="print", project="test-project")
        assert len(search_result["results"]) == 1
        assert search_result["results"][0]["path"] == "file1.py"
