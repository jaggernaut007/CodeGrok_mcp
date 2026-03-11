"""Unit tests for TreeSitterParser."""

import pytest
from codegrok_mcp.parsers.treesitter_parser import TreeSitterParser, ThreadLocalParserFactory
from codegrok_mcp.core.models import SymbolType


class TestTreeSitterParser:

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    def test_parse_python_function(self, parser, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): pass")

        result = parser.parse_file(str(test_file))

        assert result.is_successful
        assert len(result.symbols) >= 1
        assert any(s.name == "hello" for s in result.symbols)

    def test_parse_python_class(self, parser, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("class MyClass:\n    def method(self): pass")

        result = parser.parse_file(str(test_file))

        assert result.is_successful
        symbols = {s.name: s for s in result.symbols}
        assert "MyClass" in symbols
        assert symbols["MyClass"].type == SymbolType.CLASS

    def test_parse_python_with_docstring(self, parser, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text('''
def documented_function():
    """This is a docstring."""
    pass
''')

        result = parser.parse_file(str(test_file))

        assert result.is_successful
        func = next(s for s in result.symbols if s.name == "documented_function")
        assert func is not None

    def test_supported_extensions(self, parser):
        extensions = parser.get_supported_extensions()

        assert ".py" in extensions
        assert ".js" in extensions
        assert ".ts" in extensions
        assert ".go" in extensions

    def test_handles_nonexistent_file(self, parser):
        """Parser raises FileNotFoundError for nonexistent files."""
        with pytest.raises(FileNotFoundError):
            parser.parse_file("/nonexistent/file.py")

    def test_parse_javascript_function(self, parser, tmp_path):
        test_file = tmp_path / "test.js"
        test_file.write_text("function hello() { return 'world'; }")

        result = parser.parse_file(str(test_file))

        assert result.is_successful
        assert any(s.name == "hello" for s in result.symbols)

    def test_parse_javascript_constants(self, parser, tmp_path):
        js_file = tmp_path / "test.js"
        js_file.write_text("const MAX_SIZE = 100;")
        result = parser.parse_file(str(js_file))
        assert any(s.name == "MAX_SIZE" for s in result.symbols)

    def test_parse_go_function(self, parser, tmp_path):
        test_file = tmp_path / "test.go"
        test_file.write_text("package main\n\nfunc main() {}")

        result = parser.parse_file(str(test_file))

        assert result.is_successful
        assert any(s.name == "main" for s in result.symbols)


class TestParserEdgeCases:
    """Test edge cases in parsing."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    def test_empty_file(self, parser, tmp_path):
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = parser.parse_file(str(test_file))

        assert result.is_successful
        assert len(result.symbols) == 0

    def test_syntax_error_file(self, parser, tmp_path):
        test_file = tmp_path / "broken.py"
        test_file.write_text("def broken(")

        result = parser.parse_file(str(test_file))

        # Parser should still work but may have errors
        assert result is not None

    def test_unicode_content(self, parser, tmp_path):
        test_file = tmp_path / "unicode.py"
        test_file.write_text('def greet(): return "Hello, 世界! 🌍"')

        result = parser.parse_file(str(test_file))

        assert result.is_successful
        assert any(s.name == "greet" for s in result.symbols)

    def test_parse_binary_file_returns_error(self, parser, tmp_path):
        binary_file = tmp_path / "test.pyc"
        binary_file.write_bytes(b"\x00\x01\x02\x03")
        result = parser.parse_file(str(binary_file))
        assert not result.is_successful
        # We expect some error message, though exact wording depends on implementation
        assert result.error is not None

    def test_parse_python_constant(self, parser, tmp_path):
        f = tmp_path / "consts.py"
        f.write_text("MAX_RETRIES = 5\nTIMEOUT = 10")
        result = parser.parse_file(str(f))
        consts = [s.name for s in result.symbols if s.type.value == "variable"]
        assert "MAX_RETRIES" in consts
        assert "TIMEOUT" in consts


class TestThreadLocalParserFactory:
    def test_thread_local_parser_factory(self):
        factory = ThreadLocalParserFactory()
        parser1 = factory.get_parser()
        parser2 = factory.get_parser()
        assert parser1 is parser2

    def test_factory_creates_parser(self):
        factory = ThreadLocalParserFactory()
        parser = factory.get_parser()
        assert isinstance(parser, TreeSitterParser)
