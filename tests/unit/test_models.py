"""Unit tests for core data models."""

import pytest
from codegrok_mcp.core.models import Symbol, SymbolType, ParsedFile, CodebaseIndex


class TestSymbol:
    """Test Symbol dataclass."""

    def test_create_function_symbol(self):
        symbol = Symbol(
            name="hello",
            type=SymbolType.FUNCTION,
            filepath="/path/file.py",
            line_start=1,
            line_end=3,
            language="python",
            signature="def hello():",
        )

        assert symbol.name == "hello"
        assert symbol.type == SymbolType.FUNCTION
        assert symbol.language == "python"

    def test_create_class_symbol(self):
        symbol = Symbol(
            name="MyClass",
            type=SymbolType.CLASS,
            filepath="/path/file.py",
            line_start=1,
            line_end=10,
            language="python",
            signature="class MyClass:",
        )

        assert symbol.name == "MyClass"
        assert symbol.type == SymbolType.CLASS

    def test_line_count_property(self):
        symbol = Symbol(
            name="func",
            type=SymbolType.FUNCTION,
            filepath="/path/file.py",
            line_start=5,
            line_end=10,
            language="python",
            signature="def func():",
        )

        assert symbol.line_count == 6  # 5,6,7,8,9,10

    def test_qualified_name_without_parent(self):
        symbol = Symbol(
            name="standalone",
            type=SymbolType.FUNCTION,
            filepath="/path/file.py",
            line_start=1,
            line_end=3,
            language="python",
            signature="def standalone():",
        )

        assert symbol.qualified_name == "standalone"

    def test_qualified_name_with_parent(self):
        symbol = Symbol(
            name="method",
            type=SymbolType.METHOD,
            filepath="/path/file.py",
            line_start=5,
            line_end=10,
            language="python",
            signature="def method(self):",
            parent="MyClass",
        )

        assert symbol.qualified_name == "MyClass.method"

    def test_symbol_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            Symbol(
                name="",
                type=SymbolType.FUNCTION,
                filepath="/path/file.py",
                line_start=1,
                line_end=3,
                language="python",
                signature="def foo():",
            )

    def test_symbol_invalid_lines_raises(self):
        with pytest.raises(ValueError, match="line_end.*must be >= line_start"):
            Symbol(
                name="foo",
                type=SymbolType.FUNCTION,
                filepath="/path/file.py",
                line_start=5,
                line_end=4,  # Invalid
                language="python",
                signature="def foo():",
            )

    def test_symbol_serialization_roundtrip(self):
        symbol = Symbol(
            name="complex_symbol",
            type=SymbolType.METHOD,
            filepath="/src/app.py",
            line_start=10,
            line_end=20,
            language="python",
            signature="def method(self):",
            parent="AppClass",
            docstring="Doing things",
            imports=["os", "sys"],
            calls=["print"],
            metadata={"complexity": 5},
        )

        data = symbol.to_dict()
        restored = Symbol.from_dict(data)

        assert restored == symbol
        assert restored.parent == "AppClass"
        assert restored.metadata["complexity"] == 5


class TestParsedFile:
    """Test ParsedFile dataclass."""

    def test_create_successful_parsed_file(self):
        symbols = [
            Symbol(
                name="func",
                type=SymbolType.FUNCTION,
                filepath="/path/file.py",
                line_start=1,
                line_end=3,
                language="python",
                signature="def func():",
            )
        ]

        parsed = ParsedFile(filepath="/path/file.py", language="python", symbols=symbols)

        assert parsed.is_successful
        assert parsed.symbol_count == 1
        assert parsed.language == "python"

    def test_create_failed_parsed_file(self):
        parsed = ParsedFile(
            filepath="/path/file.py", language="python", symbols=[], error="Failed to parse"
        )

        assert not parsed.is_successful
        assert parsed.error == "Failed to parse"

    def test_get_symbols_by_type(self):
        symbols = [
            Symbol(
                name="func1",
                type=SymbolType.FUNCTION,
                filepath="/path/file.py",
                line_start=1,
                line_end=3,
                language="python",
                signature="def func1():",
            ),
            Symbol(
                name="MyClass",
                type=SymbolType.CLASS,
                filepath="/path/file.py",
                line_start=5,
                line_end=10,
                language="python",
                signature="class MyClass:",
            ),
            Symbol(
                name="func2",
                type=SymbolType.FUNCTION,
                filepath="/path/file.py",
                line_start=12,
                line_end=15,
                language="python",
                signature="def func2():",
            ),
        ]

        parsed = ParsedFile(filepath="/path/file.py", language="python", symbols=symbols)

        functions = parsed.get_symbols_by_type(SymbolType.FUNCTION)
        classes = parsed.get_symbols_by_type(SymbolType.CLASS)

        assert len(functions) == 2
        assert len(classes) == 1

    def test_empty_file_is_successful(self):
        parsed = ParsedFile(filepath="/path/empty.py", language="python", symbols=[])

        assert parsed.is_successful

        assert parsed.symbol_count == 0


class TestCodebaseIndex:
    """Test CodebaseIndex dataclass."""

    def test_create_index(self):
        index = CodebaseIndex(root_path="/app")
        assert index.total_files == 0
        assert index.total_symbols == 0

    def test_index_validation(self):
        with pytest.raises(ValueError, match="root_path cannot be empty"):
            CodebaseIndex(root_path="")

        with pytest.raises(ValueError, match="total_files must be >= 0"):
            CodebaseIndex(root_path="/app", total_files=-1)

    def test_serialization_roundtrip(self):
        parsed_file = ParsedFile(
            filepath="/app/main.py",
            language="python",
            symbols=[
                Symbol(
                    name="main",
                    type=SymbolType.FUNCTION,
                    filepath="/app/main.py",
                    line_start=1,
                    line_end=5,
                    language="python",
                    signature="def main():",
                )
            ],
        )

        index = CodebaseIndex(
            root_path="/app", files={"/app/main.py": parsed_file}, total_files=1, total_symbols=1
        )

        data = index.to_dict()
        restored = CodebaseIndex.from_dict(data)

        assert restored == index
        assert "/app/main.py" in restored.files
        assert restored.files["/app/main.py"].symbol_count == 1

    def test_get_symbols_by_name(self):
        s1 = Symbol(
            name="target",
            type=SymbolType.FUNCTION,
            filepath="/a.py",
            line_start=1,
            line_end=1,
            language="py",
            signature="def target()",
        )
        s2 = Symbol(
            name="other",
            type=SymbolType.FUNCTION,
            filepath="/b.py",
            line_start=1,
            line_end=1,
            language="py",
            signature="def other()",
        )
        s3 = Symbol(
            name="target",
            type=SymbolType.VARIABLE,
            filepath="/c.py",
            line_start=1,
            line_end=1,
            language="py",
            signature="target = 1",
        )

        index = CodebaseIndex(
            root_path="/app",
            files={
                "/a.py": ParsedFile(filepath="/a.py", language="py", symbols=[s1]),
                "/b.py": ParsedFile(filepath="/b.py", language="py", symbols=[s2]),
                "/c.py": ParsedFile(filepath="/c.py", language="py", symbols=[s3]),
            },
        )

        assert index.get_symbols_by_name("NonExistent") == []

        results = index.get_symbols_by_name("target")
        assert len(results) == 2
        assert s1 in results
        assert s3 in results

    def test_get_symbols_by_type(self):
        file1 = ParsedFile(
            filepath="/root/file1.py",
            language="python",
            symbols=[
                Symbol(
                    name="func1",
                    type=SymbolType.FUNCTION,
                    filepath="/root/file1.py",
                    line_start=1,
                    line_end=5,
                    language="python",
                    signature="def func1()",
                ),
                Symbol(
                    name="Class1",
                    type=SymbolType.CLASS,
                    filepath="/root/file1.py",
                    line_start=10,
                    line_end=20,
                    language="python",
                    signature="class Class1",
                ),
            ],
        )
        index = CodebaseIndex(root_path="/root", files={"/root/file1.py": file1})

        funcs = index.get_symbols_by_type(SymbolType.FUNCTION)
        assert len(funcs) == 1
        assert funcs[0].name == "func1"

        classes = index.get_symbols_by_type(SymbolType.CLASS)
        assert len(classes) == 1
        assert classes[0].name == "Class1"

    def test_validation_total_symbols(self):
        with pytest.raises(ValueError):
            CodebaseIndex(root_path="/root", total_symbols=-1)

    def test_successful_failed_parses_properties(self):
        file1 = ParsedFile(filepath="f1", language="py", symbols=[])
        file2 = ParsedFile(filepath="f2", language="py", symbols=[], error="Error")

        index = CodebaseIndex(root_path="/root", files={"f1": file1, "f2": file2})
        assert index.successful_parses == 1
        assert index.failed_parses == 1

    def test_stats_properties(self):
        success = ParsedFile(filepath="/a.py", language="py", symbols=[])
        failed = ParsedFile(filepath="/b.py", language="py", symbols=[], error="Syntax error")

        index = CodebaseIndex(root_path="/app", files={"/a.py": success, "/b.py": failed})

        assert index.successful_parses == 1
        assert index.failed_parses == 1
