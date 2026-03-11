import pytest


def test_lazy_import_treesitter_parser():
    import codegrok_mcp

    assert hasattr(codegrok_mcp, "TreeSitterParser")


def test_lazy_import_symbol():
    import codegrok_mcp

    assert hasattr(codegrok_mcp, "Symbol")


def test_lazy_import_symbol_type():
    import codegrok_mcp

    assert hasattr(codegrok_mcp, "SymbolType")


def test_lazy_import_parsed_file():
    import codegrok_mcp

    assert hasattr(codegrok_mcp, "ParsedFile")


def test_lazy_import_codebase_index():
    import codegrok_mcp

    assert hasattr(codegrok_mcp, "CodebaseIndex")


def test_lazy_import_iparser():
    import codegrok_mcp

    assert hasattr(codegrok_mcp, "IParser")


def test_lazy_import_parser_factory():
    import codegrok_mcp

    assert hasattr(codegrok_mcp, "ThreadLocalParserFactory")


def test_lazy_import_invalid_name():
    import codegrok_mcp

    with pytest.raises(AttributeError):
        _ = codegrok_mcp.NonExistentClass


def test_all_exported_names_are_accessible():
    import codegrok_mcp

    for name in codegrok_mcp.__all__:
        assert hasattr(codegrok_mcp, name)
