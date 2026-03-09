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

    @patch("codebase_rag.server._index_codebase")
    @patch("codebase_rag.server._list_indexed_files")
    @patch("codebase_rag.server._search_codebase")
    def test_reindex_project_with_force(self, mock_search, mock_list_files, mock_index):
        """Test reindexing with force=True."""
        # Mock the client and collection for deletion
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        
        # Patch the _get_client function
        with patch("codebase_rag.server._search_codebase") as mock_search_func:
            mock_search_func.__globals__ = {"_get_client": lambda: mock_client}
            
            mock_index.return_value = 5
            mock_list_files.return_value = [{"path": "file1.py"}]
            
            with TemporaryDirectory() as tmp_dir:
                result = reindex_project(tmp_dir, "test-project", force=True)
                
                assert result["status"] == "success"
                assert result["force_used"] is True

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

    @patch("codebase_rag.server._list_indexed_files")
    def test_list_indexed_projects_basic(self, mock_list_files):
        """Test basic project listing."""
        mock_list_files.return_value = [
            {"path": "file1.py"},
            {"path": "file2.py"},
        ]
        
        result = list_indexed_projects()
        
        assert "projects" in result
        assert "total_projects" in result
        assert len(result["projects"]) == 1
        assert result["projects"][0]["name"] == "default"
        assert result["projects"][0]["files"] == 2
        assert result["total_projects"] == 1

    @patch("codebase_rag.server._list_indexed_files")
    def test_list_indexed_projects_empty(self, mock_list_files):
        """Test listing with no indexed files."""
        mock_list_files.return_value = []
        
        result = list_indexed_projects()
        
        assert len(result["projects"]) == 1
        assert result["projects"][0]["files"] == 0

    @patch("codebase_rag.server._list_indexed_files")
    def test_list_indexed_projects_error(self, mock_list_files):
        """Test error handling in project listing."""
        mock_list_files.side_effect = Exception("List failed")
        
        result = list_indexed_projects()
        
        assert "error" in result
        assert result["error"] == "List failed"
        assert result["projects"] == []
        assert result["total_projects"] == 0


class TestGetFiles:
    """Test get_files MCP tool."""

    @patch("codebase_rag.server._list_indexed_files")
    def test_get_files_basic(self, mock_list_files):
        """Test basic file listing."""
        mock_list_files.return_value = [
            {"path": "file1.py"},
            {"path": "file2.js"},
        ]
        
        result = get_files()
        
        assert len(result) == 2
        assert result[0]["path"] == "file1.py"
        assert result[1]["path"] == "file2.js"

    @patch("codebase_rag.server._list_indexed_files")
    def test_get_files_with_project(self, mock_list_files):
        """Test file listing with project parameter."""
        mock_list_files.return_value = [{"path": "project_file.py"}]
        
        get_files(project="my-project")
        
        mock_list_files.assert_called_once_with(collection_name="my-project")

    @patch("codebase_rag.server._list_indexed_files")
    def test_get_files_error_handling(self, mock_list_files):
        """Test error handling in file listing."""
        mock_list_files.side_effect = Exception("List failed")
        
        with pytest.raises(Exception, match="List failed"):
            get_files()


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
