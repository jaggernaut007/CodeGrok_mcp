import pytest
from unittest.mock import MagicMock, patch
from codegrok_mcp.indexing.embedding_service import embed
from codegrok_mcp.indexing.source_retriever import SourceRetriever
from codegrok_mcp.parsers.treesitter_parser import TreeSitterParser


class TestEmbeddingHelpers:
    @patch("codegrok_mcp.indexing.embedding_service.get_embedding_service")
    def test_embed_convenience(self, mock_get_service):
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        # Test string
        embed("text")
        mock_service.embed.assert_called_with("text", is_query=False)

        # Test list
        embed(["t1", "t2"])
        mock_service.embed_batch.assert_called_with(["t1", "t2"], is_query=False)


class TestSourceRetrieverExtras:
    def test_parallel_init(self, tmp_path):
        # Just check it initializes without error
        sr = SourceRetriever(str(tmp_path), parallel=True, max_workers=2)
        assert sr.parallel
        assert sr.max_workers == 2

    def test_logging(self, tmp_path, capsys):
        sr = SourceRetriever(str(tmp_path), verbose=True)
        sr._log("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_logging_disabled(self, tmp_path, capsys):
        sr = SourceRetriever(str(tmp_path), verbose=False)
        sr._log("test message")
        captured = capsys.readouterr()
        assert "test message" not in captured.out


class TestParserExtras:
    def test_clean_docstring(self):
        parser = TreeSitterParser()

        # Triple quotes
        assert parser._clean_docstring('"""doc"""') == "doc"
        assert parser._clean_docstring("'''doc'''") == "doc"

        # Single quotes
        assert parser._clean_docstring('"doc"') == "doc"
        assert parser._clean_docstring("'doc'") == "doc"

        # Multiline
        assert parser._clean_docstring('"""\n  Line 1\n  Line 2\n"""') == "Line 1"

        # Raw
        assert parser._clean_docstring("Simple doc") == "Simple doc"
