"""
TreeSitterParser - Production-grade multi-language code parser.

This module implements the IParser interface using tree-sitter for parsing
source code files in multiple programming languages. It extracts symbols
(functions, classes, methods), imports, function calls, and metadata.

Supported Languages:
    - Python (.py, .pyi, .pyw)
    - JavaScript (.js, .jsx, .mjs, .cjs)
    - TypeScript (.ts, .tsx, .mts, .cts)
    - C (.c, .h)
    - C++ (.cpp, .cc, .cxx, .hpp, .hh)
    - Bash (.sh, .bash, .zsh)
    - Go (.go)
    - Java (.java)
    - Kotlin (.kt, .kts)

Performance:
    - Caches parsers per language for speed
    - Targets >200 files/second on typical codebases
    - Handles large files (up to 10MB with warnings)

Usage:
    >>> parser = TreeSitterParser()
    >>> result = parser.parse_file("example.py")
    >>> print(f"Found {len(result.symbols)} symbols")
"""

import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from tree_sitter_languages import get_parser

from codegrok_mcp.core.interfaces import IParser
from codegrok_mcp.core.models import Symbol, ParsedFile, SymbolType
from codegrok_mcp.parsers.language_configs import (
    get_language_for_file,
    get_config_for_language,
    get_supported_extensions,
)

# Configure logging
logger = logging.getLogger(__name__)


class TreeSitterParser(IParser):
    """
    Production-ready tree-sitter based code parser.

    This parser uses tree-sitter grammars to parse source code into ASTs,
    then traverses the AST to extract symbols, imports, and function calls.
    Parsers are cached per language for performance.

    Attributes:
        MAX_CODE_SNIPPET_CHARS: Maximum length of code snippets (500 chars)
        MAX_FILE_SIZE_MB: Maximum file size before warning (10 MB)
        _parsers: Cache of tree-sitter parsers by language

    Thread Safety:
        This class is NOT thread-safe. Create separate instances for
        concurrent parsing.
    """

    # Constants
    # 4000 chars ≈ 1000-1300 tokens, matching the research-optimal 1024 token sweet spot
    # for nomic-embed-code (trained on 2048 tokens, supports 8192 via NTK interpolation)
    MAX_CODE_SNIPPET_CHARS = 4000
    MAX_FILE_SIZE_MB = 10

    def __init__(self):
        """Initialize the parser with an empty parser cache."""
        self._parsers: Dict[str, Any] = {}
        logger.debug("TreeSitterParser initialized")

    def can_parse(self, filepath: str) -> bool:
        """
        Determine if this parser can handle the given file.

        Fast check based on file extension lookup.

        Args:
            filepath: Path to the file to check

        Returns:
            True if the file extension is supported, False otherwise

        Example:
            >>> parser = TreeSitterParser()
            >>> parser.can_parse("example.py")
            True
            >>> parser.can_parse("example.txt")
            False
        """
        language = get_language_for_file(filepath)
        return language is not None

    def parse_file(self, filepath: str) -> ParsedFile:
        """
        Parse a source file and extract all symbols.

        This is the main entry point for parsing. It:
        1. Validates the file exists and is supported
        2. Detects the programming language
        3. Reads and parses the file into an AST
        4. Extracts symbols, imports, and calls
        5. Returns a ParsedFile with all extracted data

        Args:
            filepath: Absolute path to the file to parse

        Returns:
            ParsedFile object containing symbols and metadata

        Raises:
            FileNotFoundError: If the file doesn't exist

        Example:
            >>> parser = TreeSitterParser()
            >>> result = parser.parse_file("/path/to/example.py")
            >>> print(f"Found {len(result.symbols)} symbols")
        """
        start_time = time.time()

        # Validate file exists
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Detect language
        language = get_language_for_file(filepath)
        if language is None:
            error_msg = f"Unsupported file type: {file_path.suffix}"
            logger.warning(f"{error_msg} - {filepath}")
            return ParsedFile(
                filepath=str(filepath),
                language="unknown",
                symbols=[],
                imports=[],
                parse_time=time.time() - start_time,
                error=error_msg,
            )

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            logger.warning(
                f"Large file ({file_size_mb:.2f}MB): {filepath}. " f"Parsing may be slow."
            )

        # Read file content
        try:
            content = file_path.read_bytes()

            # Check if file is binary (contains null bytes or non-text characters)
            if self._is_binary_file(content):
                error_msg = "Binary file detected - cannot parse"
                logger.error(f"{error_msg}: {filepath}")
                return ParsedFile(
                    filepath=str(filepath),
                    language=language,
                    symbols=[],
                    imports=[],
                    parse_time=time.time() - start_time,
                    error=error_msg,
                )
        except UnicodeDecodeError:
            # Binary file
            error_msg = "Binary file detected - cannot parse"
            logger.error(f"{error_msg}: {filepath}")
            return ParsedFile(
                filepath=str(filepath),
                language=language,
                symbols=[],
                imports=[],
                parse_time=time.time() - start_time,
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Error reading file: {str(e)}"
            logger.error(f"{error_msg}: {filepath}")
            return ParsedFile(
                filepath=str(filepath),
                language=language,
                symbols=[],
                imports=[],
                parse_time=time.time() - start_time,
                error=error_msg,
            )

        # Get or create parser for this language
        try:
            parser = self._get_parser(language)
            config = get_config_for_language(language)
        except Exception as e:  # pragma: no cover
            error_msg = f"Failed to initialize parser: {str(e)}"
            logger.error(f"{error_msg}: {filepath}")
            return ParsedFile(
                filepath=str(filepath),
                language=language,
                symbols=[],
                imports=[],
                parse_time=time.time() - start_time,
                error=error_msg,
            )

        # Parse the file
        try:
            tree = parser.parse(content)
            root_node = tree.root_node

            # Extract symbols and imports
            symbols = self._extract_symbols(root_node, content, filepath, language, config)
            imports = self._extract_imports(root_node, content, config)

            parse_time = time.time() - start_time
            logger.debug(
                f"Parsed {filepath} in {parse_time:.3f}s - "
                f"found {len(symbols)} symbols, {len(imports)} imports"
            )

            return ParsedFile(
                filepath=str(filepath),
                language=language,
                symbols=symbols,
                imports=imports,
                parse_time=parse_time,
                error=None,
            )

        except Exception as e:  # pragma: no cover
            error_msg = f"Parsing error: {str(e)}"
            logger.error(f"{error_msg}: {filepath}", exc_info=True)
            return ParsedFile(
                filepath=str(filepath),
                language=language,
                symbols=[],
                imports=[],
                parse_time=time.time() - start_time,
                error=error_msg,
            )

    def get_supported_extensions(self) -> List[str]:
        """
        Return list of file extensions this parser supports.

        Returns:
            Sorted list of file extensions (with leading dots)

        Example:
            >>> parser = TreeSitterParser()
            >>> extensions = parser.get_supported_extensions()
            >>> '.py' in extensions
            True
        """
        return sorted(list(get_supported_extensions()))

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _is_binary_file(self, content: bytes) -> bool:
        """
        Check if file content is binary (not text).

        Args:
            content: File content as bytes

        Returns:
            True if file appears to be binary, False otherwise
        """
        # Check first 8192 bytes for binary markers
        sample = content[:8192]

        # Check for null bytes (common in binary files)
        if b"\x00" in sample:
            return True

        # Check for high ratio of non-text bytes
        try:
            # Try to decode as UTF-8
            sample.decode("utf-8")
            return False
        except UnicodeDecodeError:
            # Count how many bytes fail to decode
            text_chars = sum(1 for b in sample if 32 <= b < 127 or b in (9, 10, 13))
            if len(sample) == 0:
                return False
            text_ratio = text_chars / len(sample)
            # If less than 70% are text characters, consider it binary
            return text_ratio < 0.7

    def _get_parser(self, language: str):
        """
        Get or create a tree-sitter parser for the given language.

        Parsers are cached to avoid repeated initialization.

        Args:
            language: Language name (e.g., 'python', 'javascript')

        Returns:
            Tree-sitter parser instance
        """
        if language not in self._parsers:
            logger.debug(f"Creating new parser for language: {language}")
            self._parsers[language] = get_parser(language)
        return self._parsers[language]

    def _extract_symbols(
        self,
        root_node,
        content: bytes,
        filepath: str,
        language: str,
        config: Dict[str, Any],
    ) -> List[Symbol]:
        """
        Extract all symbols (functions, classes, methods) from the AST.

        Args:
            root_node: Tree-sitter root node
            content: File content as bytes
            filepath: Path to the file being parsed
            language: Language name
            config: Language configuration

        Returns:
            List of extracted Symbol objects
        """
        symbols: List[Symbol] = []

        # Track classes for parent-child relationships
        current_class: Optional[str] = None
        class_stack: List[str] = []

        # Get node types from config
        function_types = config.get("function_types", [])
        class_types = config.get("class_types", [])
        method_types = config.get("method_types", [])
        constant_types = config.get("constant_types", [])

        def traverse(node, depth: int = 0):
            """Recursively traverse the AST and extract symbols."""
            nonlocal current_class

            node_type = node.type

            # Extract classes
            if node_type in class_types:
                symbol = self._extract_class_symbol(node, content, filepath, language, config)
                if symbol:
                    symbols.append(symbol)
                    # Push class onto stack for method extraction
                    class_stack.append(symbol.name)
                    current_class = symbol.name

            # Extract functions (top-level)
            elif node_type in function_types and not current_class:
                symbol = self._extract_function_symbol(
                    node, content, filepath, language, config, parent=None
                )
                if symbol:
                    symbols.append(symbol)

            # Extract methods (inside classes)
            elif node_type in method_types and current_class:
                symbol = self._extract_function_symbol(
                    node,
                    content,
                    filepath,
                    language,
                    config,
                    parent=current_class,
                    is_method=True,
                )
                if symbol:
                    symbols.append(symbol)

            # Extract constants (module-level only)
            elif node_type in constant_types and not current_class:
                symbol = self._extract_constant_symbol(node, content, filepath, language, config)
                if symbol:
                    symbols.append(symbol)

            # Recursively process children
            for child in node.children:
                traverse(child, depth + 1)

            # Pop class from stack when exiting class body
            if node_type in class_types and class_stack:
                class_stack.pop()
                current_class = class_stack[-1] if class_stack else None

        # Start traversal
        traverse(root_node)
        return symbols

    def _extract_class_symbol(
        self,
        node,
        content: bytes,
        filepath: str,
        language: str,
        config: Dict[str, Any],
    ) -> Optional[Symbol]:
        """
        Extract a class symbol from an AST node.

        Args:
            node: Tree-sitter class node
            content: File content as bytes
            filepath: Path to the file
            language: Language name
            config: Language configuration

        Returns:
            Symbol object or None if extraction fails
        """
        name = self._get_node_name(node, content, config)
        if not name:
            return None

        # Get line numbers (tree-sitter is 0-indexed, we use 1-indexed)
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1

        # Extract signature
        signature = self._get_node_text(node, content).split("\n")[0].strip()

        # Extract docstring
        docstring = self._extract_docstring(node, content, config)

        # Extract code snippet
        code_snippet = self._get_code_snippet(node, content)

        # Extract imports used within class scope
        imports = self._extract_imports_from_node(node, content, config)

        # Extract function calls
        calls = self._extract_calls_from_node(node, content, config)

        return Symbol(
            name=name,
            type=SymbolType.CLASS,
            filepath=filepath,
            line_start=line_start,
            line_end=line_end,
            language=language,
            signature=signature,
            docstring=docstring,
            parent=None,
            code_snippet=code_snippet,
            imports=imports,
            calls=calls,
        )

    def _extract_function_symbol(
        self,
        node,
        content: bytes,
        filepath: str,
        language: str,
        config: Dict[str, Any],
        parent: Optional[str] = None,
        is_method: bool = False,
    ) -> Optional[Symbol]:
        """
        Extract a function or method symbol from an AST node.

        Args:
            node: Tree-sitter function/method node
            content: File content as bytes
            filepath: Path to the file
            language: Language name
            config: Language configuration
            parent: Parent class name (for methods)
            is_method: True if this is a method, False if function

        Returns:
            Symbol object or None if extraction fails
        """
        name = self._get_node_name(node, content, config)
        if not name:
            return None

        # Get line numbers (tree-sitter is 0-indexed, we use 1-indexed)
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1

        # Extract signature
        signature = self._get_node_text(node, content).split("\n")[0].strip()

        # Extract docstring
        docstring = self._extract_docstring(node, content, config)

        # Extract code snippet
        code_snippet = self._get_code_snippet(node, content)

        # Extract imports used within function scope
        imports = self._extract_imports_from_node(node, content, config)

        # Extract function calls
        calls = self._extract_calls_from_node(node, content, config)

        symbol_type = SymbolType.METHOD if is_method else SymbolType.FUNCTION

        return Symbol(
            name=name,
            type=symbol_type,
            filepath=filepath,
            line_start=line_start,
            line_end=line_end,
            language=language,
            signature=signature,
            docstring=docstring,
            parent=parent,
            code_snippet=code_snippet,
            imports=imports,
            calls=calls,
        )

    def _extract_constant_symbol(
        self,
        node,
        content: bytes,
        filepath: str,
        language: str,
        config: Dict[str, Any],
    ) -> Optional[Symbol]:
        """
        Extract a constant/variable symbol from an AST node.

        Handles different patterns:
        - Python: expression_statement with assignment (CONST = value)
        - JavaScript: lexical_declaration (const NAME = value)
        - Go: const_declaration (const NAME = value)
        - Bash: variable_assignment or declaration_command (NAME=value)

        Args:
            node: Tree-sitter constant node
            content: File content as bytes
            filepath: Path to the file
            language: Language name
            config: Language configuration

        Returns:
            Symbol object or None if extraction fails
        """
        # Get the full text of the constant
        full_text = self._get_node_text(node, content).strip()

        # Extract the name based on language patterns
        name = None

        if language == "python":
            # Python: expression_statement -> assignment -> identifier
            # Look for UPPERCASE names (convention for constants)
            for child in node.children:
                if child.type == "assignment":
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            potential_name = self._get_node_text(subchild, content)
                            # Only extract UPPERCASE constants
                            if (
                                potential_name.isupper()
                                or "_" in potential_name
                                and potential_name.replace("_", "").isupper()
                            ):
                                name = potential_name
                            break
                    break

        elif language == "javascript" or language == "typescript":
            # JavaScript: lexical_declaration -> variable_declarator -> identifier
            # Only extract const with UPPERCASE names (convention for constants)
            if full_text.startswith("const "):
                for child in node.children:
                    if child.type == "variable_declarator":
                        for subchild in child.children:
                            if subchild.type == "identifier":
                                potential_name = self._get_node_text(subchild, content)
                                # Only UPPERCASE constants
                                if potential_name.isupper() or (
                                    "_" in potential_name
                                    and potential_name.replace("_", "").isupper()
                                ):
                                    name = potential_name
                                break
                        break

                        break

        # Add pragma for other languages not covered by tests yet
        elif language == "go":  # pragma: no cover
            # Go: const_declaration -> const_spec -> identifier
            # Return all const names as one symbol (grouped)
            const_names = []
            for child in node.children:
                if child.type == "const_spec":
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            const_names.append(self._get_node_text(subchild, content))
                            break
            if const_names:
                name = ", ".join(const_names)

        elif language == "bash":  # pragma: no cover
            # Bash: variable_assignment or declaration_command
            # Only extract UPPERCASE variables (constants by convention)
            if node.type == "declaration_command":
                # readonly VAR=value
                for child in node.children:
                    if child.type == "variable_assignment":
                        for subchild in child.children:
                            if subchild.type == "variable_name":
                                potential_name = self._get_node_text(subchild, content)
                                if potential_name.isupper() or (
                                    "_" in potential_name
                                    and potential_name.replace("_", "").isupper()
                                ):
                                    name = potential_name
                                break
                        break
            elif node.type == "variable_assignment":
                # VAR=value - only at top level with UPPERCASE
                for child in node.children:
                    if child.type == "variable_name":
                        potential_name = self._get_node_text(child, content)
                        if potential_name.isupper() or (
                            "_" in potential_name and potential_name.replace("_", "").isupper()
                        ):
                            name = potential_name
                        break

        if not name:
            return None

        # Get line numbers
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1

        return Symbol(
            name=name,
            type=SymbolType.VARIABLE,
            filepath=filepath,
            line_start=line_start,
            line_end=line_end,
            language=language,
            signature=full_text,
            docstring=None,
            parent=None,
            code_snippet=full_text,
            imports=[],
            calls=[],
        )

    def _extract_imports(self, root_node, content: bytes, config: Dict[str, Any]) -> List[str]:
        """
        Extract all import statements from the file.

        Args:
            root_node: Tree-sitter root node
            content: File content as bytes
            config: Language configuration

        Returns:
            List of import statement strings
        """
        imports: List[str] = []
        import_types = config.get("import_types", [])

        def traverse(node):
            if node.type in import_types:
                import_text = self._get_node_text(node, content).strip()
                if import_text and import_text not in imports:
                    imports.append(import_text)

            for child in node.children:
                traverse(child)

        traverse(root_node)
        return imports

    def _extract_imports_from_node(self, node, content: bytes, config: Dict[str, Any]) -> List[str]:
        """
        Extract imports from a specific node (for scoped imports).

        Args:
            node: Tree-sitter node
            content: File content as bytes
            config: Language configuration

        Returns:
            List of import statement strings
        """
        imports: List[str] = []
        import_types = config.get("import_types", [])

        def traverse(n):
            if n.type in import_types:
                import_text = self._get_node_text(n, content).strip()
                if import_text and import_text not in imports:
                    imports.append(import_text)

            for child in n.children:
                traverse(child)

        traverse(node)
        return imports

    def _extract_calls_from_node(self, node, content: bytes, config: Dict[str, Any]) -> List[str]:
        """
        Extract function calls from a specific node.

        Args:
            node: Tree-sitter node
            content: File content as bytes
            config: Language configuration

        Returns:
            List of function call names
        """
        calls: Set[str] = set()
        call_types = config.get("call_types", [])

        def traverse(n):
            if n.type in call_types:
                # Extract the function name being called
                call_name = self._get_call_name(n, content)
                if call_name:
                    calls.add(call_name)

            for child in n.children:
                traverse(child)

        traverse(node)
        return sorted(list(calls))

    def _get_call_name(self, call_node, content: bytes) -> Optional[str]:
        """
        Extract the function name from a call expression.

        Args:
            call_node: Tree-sitter call node
            content: File content as bytes

        Returns:
            Function name or None
        """
        # Try to find the function identifier
        for child in call_node.children:
            if child.type in ("identifier", "name", "word", "field_identifier"):
                return self._get_node_text(child, content).strip()
            elif child.type == "attribute":
                # For method calls like obj.method()
                for subchild in child.children:
                    if subchild.type in ("identifier", "property_identifier", "field_identifier"):
                        return self._get_node_text(subchild, content).strip()
            elif child.type == "member_expression":
                # JavaScript/TypeScript member expressions
                for subchild in child.children:
                    if subchild.type in ("property_identifier", "identifier"):
                        return self._get_node_text(subchild, content).strip()
            elif child.type == "selector_expression":
                # Go selector expressions
                for subchild in child.children:
                    if subchild.type in ("field_identifier", "identifier"):
                        return self._get_node_text(subchild, content).strip()

        # Fallback: try to get first identifier child
        if call_node.child_count > 0:
            first_child = call_node.children[0]
            return self._get_node_text(first_child, content).strip().split("(")[0]

        return None

    def _extract_docstring(self, node, content: bytes, config: Dict[str, Any]) -> str:
        """
        Extract docstring from a function or class node.

        Args:
            node: Tree-sitter node
            content: File content as bytes
            config: Language configuration

        Returns:
            Docstring text (empty string if none found)
        """
        # Look for the first string literal in the body
        body = self._get_body_node(node, config)
        if not body:
            return ""

        # Check first child of body for docstring
        for child in body.children:
            if child.type == "expression_statement":
                # Python: first expression statement might be docstring
                for subchild in child.children:
                    if subchild.type in ("string", "string_literal"):
                        return self._clean_docstring(self._get_node_text(subchild, content))
            elif child.type in ("string", "string_literal", "comment"):
                return self._clean_docstring(self._get_node_text(child, content))

        return ""

    def _get_body_node(self, node, config: Dict[str, Any]):
        """
        Get the body node of a function or class.

        Args:
            node: Tree-sitter node
            config: Language configuration

        Returns:
            Body node or None
        """
        body_field = config.get("body_field", "body")

        # Try named child first
        for child in node.children:
            if child.type in ("block", "body", "compound_statement", "statement_block"):
                return child

        # Try field access
        if hasattr(node, "child_by_field_name"):
            body = node.child_by_field_name(body_field)
            if body:
                return body

        return None

    def _get_node_name(self, node, content: bytes, config: Dict[str, Any]) -> Optional[str]:
        """
        Extract the name/identifier from a node.

        Args:
            node: Tree-sitter node
            content: File content as bytes
            config: Language configuration

        Returns:
            Name string or None
        """
        # Try field-based access first
        if hasattr(node, "child_by_field_name"):
            name_node = node.child_by_field_name("name")
            if name_node:
                return self._get_node_text(name_node, content).strip()

        # Try finding identifier child
        for child in node.children:
            if child.type in (
                "identifier",
                "name",
                "type_identifier",
                "field_identifier",
                "property_identifier",
            ):
                return self._get_node_text(child, content).strip()

        # For C/C++ functions with declarators
        for child in node.children:
            if child.type in ("declarator", "function_declarator"):
                return self._get_node_name(child, content, config)

        return None

    def _get_node_text(self, node, content: bytes) -> str:
        """
        Extract text content from a node.

        Args:
            node: Tree-sitter node
            content: File content as bytes

        Returns:
            Node text as string
        """
        try:
            return content[node.start_byte : node.end_byte].decode("utf-8")
        except UnicodeDecodeError:
            # Fallback for binary content
            return content[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

    def _get_code_snippet(self, node, content: bytes) -> str:
        """
        Extract code snippet from a node, truncating if necessary.

        Args:
            node: Tree-sitter node
            content: File content as bytes

        Returns:
            Code snippet (max MAX_CODE_SNIPPET_CHARS characters)
        """
        text = self._get_node_text(node, content)

        if len(text) > self.MAX_CODE_SNIPPET_CHARS:
            return text[: self.MAX_CODE_SNIPPET_CHARS] + "..."

        return text

    def _clean_docstring(self, raw_docstring: str) -> str:
        """
        Clean a raw docstring by removing quotes and extra whitespace.

        Args:
            raw_docstring: Raw docstring from source

        Returns:
            Cleaned docstring
        """
        # Remove quotes (""", ''', ", ')
        cleaned = raw_docstring.strip()

        # Remove triple quotes
        for quote in ('"""', "'''"):
            if cleaned.startswith(quote) and cleaned.endswith(quote):
                cleaned = cleaned[3:-3]
                break

        # Remove single quotes
        for quote in ('"', "'"):
            if cleaned.startswith(quote) and cleaned.endswith(quote):
                cleaned = cleaned[1:-1]
                break

        # Clean up whitespace
        cleaned = cleaned.strip()

        # For multi-line docstrings, extract just the first line/paragraph
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        if lines:
            return lines[0]

        return cleaned


# ===========================================================================
# Thread-Safe Parser Factory for Parallel Parsing
# ===========================================================================


class ThreadLocalParserFactory:
    """
    Factory that provides thread-local TreeSitterParser instances.

    This enables safe parallel parsing where each thread gets its own
    parser instance, avoiding contention on the internal parser cache.

    Thread Safety:
        This class IS thread-safe. It uses thread-local storage to ensure
        each thread gets its own independent TreeSitterParser instance.

    Usage:
        factory = ThreadLocalParserFactory()
        # In each thread:
        parser = factory.get_parser()
        result = parser.parse_file(filepath)

    Example:
        >>> from concurrent.futures import ThreadPoolExecutor
        >>> factory = ThreadLocalParserFactory()
        >>> def parse_file(filepath):
        ...     parser = factory.get_parser()
        ...     return parser.parse_file(filepath)
        >>> with ThreadPoolExecutor(max_workers=4) as executor:
        ...     results = list(executor.map(parse_file, file_list))
    """

    def __init__(self):
        """Initialize the factory with thread-local storage."""
        self._local = threading.local()

    def get_parser(self) -> TreeSitterParser:
        """
        Get or create a TreeSitterParser for the current thread.

        Each thread will get its own independent parser instance,
        which is cached for the lifetime of the thread.

        Returns:
            TreeSitterParser instance unique to the calling thread
        """
        if not hasattr(self._local, "parser"):
            self._local.parser = TreeSitterParser()
        return self._local.parser
