"""
Language-specific configurations for tree-sitter parsing.

This module provides comprehensive AST node type mappings for each supported language,
enabling accurate extraction of functions, classes, methods, imports, and function calls.

Supported Languages:
    - Python (.py)
    - JavaScript/TypeScript (.js, .jsx, .ts, .tsx)
    - C/C++ (.c, .cc, .cpp, .h, .hpp)
    - Bash (.sh, .bash)
    - Go (.go)
    - Java (.java)
    - Kotlin (.kt, .kts)

Usage:
    >>> language = get_language_for_file("example.py")
    >>> config = get_config_for_language(language)
    >>> function_types = config['function_types']

Adding New Languages:
    1. Add file extension mappings to EXTENSION_MAP
    2. Create language config dict with required node types
    3. Document the tree-sitter grammar node types
    4. Add tests for the new language
"""

from typing import Optional, Dict, List, Set
from pathlib import Path

# ==============================================================================
# Extension to Language Mapping
# ==============================================================================

EXTENSION_MAP: Dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",  # Type stub files
    ".pyw": "python",  # Windows Python GUI scripts
    # JavaScript/TypeScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",  # ES6 modules
    ".cjs": "javascript",  # CommonJS modules
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",  # TypeScript ES6 modules
    ".cts": "typescript",  # TypeScript CommonJS modules
    # C/C++
    ".c": "c",
    ".h": "c",  # Header files (may be C or C++)
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".h++": "cpp",
    # Bash
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",  # Zsh is similar enough to bash
    # Go
    ".go": "go",
    # Java
    ".java": "java",
    # Kotlin
    ".kt": "kotlin",
    ".kts": "kotlin",  # Kotlin script files
}


# ==============================================================================
# Language Configurations
# ==============================================================================

LANGUAGE_CONFIGS: Dict[str, Dict[str, any]] = {
    # --------------------------------------------------------------------------
    # Python Configuration
    # --------------------------------------------------------------------------
    "python": {
        "function_types": [
            "function_definition",  # def foo(): ...
        ],
        "class_types": [
            "class_definition",  # class MyClass: ...
        ],
        "method_types": [
            "function_definition",  # Methods are functions inside class bodies
        ],
        "constant_types": [
            "expression_statement",  # MODULE_CONST = value (at module level)
        ],
        "import_types": [
            "import_statement",  # import os
            "import_from_statement",  # from os import path
        ],
        "call_types": [
            "call",  # function_name(args)
        ],
        "decorator_types": [
            "decorator",  # @decorator
        ],
        "docstring_field": "string",  # First string literal in function/class body
        "identifier_field": "name",  # Field containing the symbol name
        "body_field": "body",  # Field containing the body block
        # Additional node types for comprehensive parsing
        "async_function_types": [
            "function_definition",  # async def foo(): ... (same node type)
        ],
        "lambda_types": [
            "lambda",  # lambda x: x + 1
        ],
        # Examples of AST nodes:
        # function_definition:
        #   name: identifier
        #   parameters: parameters
        #   body: block
        #   return_type: type (optional, for type hints)
        #
        # class_definition:
        #   name: identifier
        #   superclasses: argument_list (optional)
        #   body: block
        #
        # call:
        #   function: identifier | attribute
        #   arguments: argument_list
    },
    # --------------------------------------------------------------------------
    # JavaScript Configuration
    # --------------------------------------------------------------------------
    "javascript": {
        "function_types": [
            "function_declaration",  # function foo() {}
            "function",  # function foo() {} (alternate name in some versions)
            "generator_function_declaration",  # function* foo() {}
        ],
        "class_types": [
            "class_declaration",  # class MyClass {}
            # Note: 'class' is just the keyword, not the full class node
        ],
        "method_types": [
            "method_definition",  # Methods inside class bodies
            "function_expression",  # foo: function() {}
            "arrow_function",  # foo: () => {}
        ],
        "constant_types": [
            "lexical_declaration",  # const MAX_RETRIES = 3;
        ],
        "import_types": [
            "import_statement",  # import { x } from 'module'
            "import_clause",  # Part of import statement
        ],
        "call_types": [
            "call_expression",  # foo()
            "new_expression",  # new Foo()
        ],
        "export_types": [
            "export_statement",  # export { foo }
        ],
        "docstring_field": "comment",  # JSDoc comments (/** ... */)
        "identifier_field": "name",
        "body_field": "body",
        # Additional patterns
        "arrow_function_types": [
            "arrow_function",  # const foo = () => {}
        ],
        "variable_declaration_types": [
            "variable_declarator",  # const foo = ...
        ],
        # Examples of AST nodes:
        # function_declaration:
        #   name: identifier
        #   parameters: formal_parameters
        #   body: statement_block
        #
        # arrow_function:
        #   parameter: identifier | formal_parameters
        #   body: statement_block | expression
        #
        # method_definition:
        #   name: property_identifier
        #   parameters: formal_parameters
        #   body: statement_block
        #
        # call_expression:
        #   function: identifier | member_expression
        #   arguments: arguments
    },
    # --------------------------------------------------------------------------
    # TypeScript Configuration
    # --------------------------------------------------------------------------
    "typescript": {
        "function_types": [
            "function_declaration",
            "function_signature",  # TypeScript type declaration
            "generator_function_declaration",
        ],
        "class_types": [
            "class_declaration",
            "interface_declaration",  # TypeScript interfaces
            "type_alias_declaration",  # type MyType = ...
        ],
        "method_types": [
            "method_definition",
            "method_signature",  # TypeScript method signatures in interfaces
            "arrow_function",
            "function_expression",
        ],
        "import_types": [
            "import_statement",
            "import_clause",
        ],
        "call_types": [
            "call_expression",
            "new_expression",
        ],
        "export_types": [
            "export_statement",
        ],
        "docstring_field": "comment",  # TSDoc comments
        "identifier_field": "name",
        "body_field": "body",
        # TypeScript-specific
        "interface_types": [
            "interface_declaration",  # interface Foo { ... }
        ],
        "type_types": [
            "type_alias_declaration",  # type Foo = ...
        ],
        "enum_types": [
            "enum_declaration",  # enum Foo { ... }
        ],
        # Examples of AST nodes:
        # interface_declaration:
        #   name: type_identifier
        #   body: interface_body
        #
        # type_alias_declaration:
        #   name: type_identifier
        #   value: type
    },
    # --------------------------------------------------------------------------
    # C Configuration
    # --------------------------------------------------------------------------
    "c": {
        "function_types": [
            "function_definition",  # void foo() { ... }
            "function_declarator",  # Function declarations
        ],
        "class_types": [
            "struct_specifier",  # struct Foo { ... }
            "union_specifier",  # union Foo { ... }
        ],
        "method_types": [
            "function_definition",  # C doesn't have methods, but struct functions
        ],
        "import_types": [
            "preproc_include",  # #include <stdio.h>
        ],
        "call_types": [
            "call_expression",  # foo()
        ],
        "typedef_types": [
            "type_definition",  # typedef struct { ... } Foo;
        ],
        "docstring_field": "comment",  # /* ... */ or // ...
        "identifier_field": "declarator",
        "body_field": "body",
        # Additional C-specific nodes
        "enum_types": [
            "enum_specifier",  # enum Foo { ... }
        ],
        "macro_types": [
            "preproc_def",  # #define FOO 123
            "preproc_function_def",  # #define FOO(x) (x + 1)
        ],
        # Examples of AST nodes:
        # function_definition:
        # type: primitive_type | struct_specifier
        #   declarator: function_declarator
        #     declarator: identifier
        #     parameters: parameter_list
        #   body: compound_statement
        #
        # struct_specifier:
        #   name: type_identifier
        #   body: field_declaration_list
        #
        # preproc_include:
        #   path: string_literal | system_lib_string
    },
    # --------------------------------------------------------------------------
    # C++ Configuration
    # --------------------------------------------------------------------------
    "cpp": {
        "function_types": [
            "function_definition",
            # Note: function_declarator is a child of function_definition, not standalone
        ],
        "class_types": [
            "class_specifier",  # class Foo { ... }
            "struct_specifier",  # struct Foo { ... }
            "union_specifier",  # union Foo { ... }
        ],
        "method_types": [
            "function_definition",  # Methods inside class bodies
            # Note: field_declaration is for member variables, not methods
        ],
        "import_types": [
            "preproc_include",  # #include <iostream>
        ],
        "call_types": [
            "call_expression",  # foo()
        ],
        "namespace_types": [
            "namespace_definition",  # namespace foo { ... }
        ],
        "template_types": [
            "template_declaration",  # template<typename T> class Foo { ... }
        ],
        "typedef_types": [
            "type_definition",  # typedef or using
            "alias_declaration",  # using Foo = Bar;
        ],
        "docstring_field": "comment",  # Doxygen comments (/** ... */)
        "identifier_field": "declarator",
        "body_field": "body",
        # C++-specific features
        "enum_types": [
            "enum_specifier",  # enum class Foo { ... }
        ],
        "lambda_types": [
            "lambda_expression",  # [](int x) { return x + 1; }
        ],
        # Examples of AST nodes:
        # class_specifier:
        #   name: type_identifier
        #   body: field_declaration_list
        #   base_class_clause: (optional)
        #
        # namespace_definition:
        #   name: identifier
        #   body: declaration_list
        #
        # template_declaration:
        #   parameters: template_parameter_list
        #   declaration: class_specifier | function_definition
    },
    # --------------------------------------------------------------------------
    # Bash Configuration
    # --------------------------------------------------------------------------
    "bash": {
        "function_types": [
            "function_definition",  # foo() { ... } or function foo { ... }
        ],
        "class_types": [
            # Bash doesn't have classes
        ],
        "method_types": [
            # Bash doesn't have methods
        ],
        "constant_types": [
            "declaration_command",  # readonly VAR=value (true constants)
            # Note: variable_assignment excluded to avoid duplicates within declaration_command
        ],
        "import_types": [
            "command",  # Can detect source/. commands via command name
        ],
        "call_types": [
            "command",  # foo arg1 arg2
            "command_substitution",  # $(foo)
        ],
        "variable_types": [
            "variable_assignment",  # VAR=value
        ],
        "docstring_field": "comment",  # # ...
        "identifier_field": "name",
        "body_field": "body",
        # Bash-specific patterns
        "source_patterns": [
            "source",  # source script.sh
            ".",  # . script.sh
        ],
        # Examples of AST nodes:
        # function_definition:
        #   name: word
        #   body: compound_statement
        #
        # command:
        #   name: command_name
        #   arguments: (word)*
        #
        # variable_assignment:
        #   name: variable_name
        #   value: word | string | command_substitution
    },
    # --------------------------------------------------------------------------
    # Go Configuration
    # --------------------------------------------------------------------------
    "go": {
        "function_types": [
            "function_declaration",  # func foo() { ... }
        ],
        "class_types": [
            "type_declaration",  # type Foo struct { ... }
        ],
        "method_types": [
            "method_declaration",  # func (r Receiver) foo() { ... }
        ],
        "constant_types": [
            "const_declaration",  # const ( MaxRetries = 3 ... )
        ],
        "import_types": [
            "import_declaration",  # import "fmt" or import ( ... )
            "import_spec",  # Individual import within import block
        ],
        "call_types": [
            "call_expression",  # foo()
        ],
        "interface_types": [
            "interface_type",  # interface { ... } within type_declaration
        ],
        "struct_types": [
            "struct_type",  # struct { ... } within type_declaration
        ],
        "docstring_field": "comment",  # GoDoc comments (// ...)
        "identifier_field": "name",
        "body_field": "body",
        # Additional Go patterns
        "package_types": [
            "package_clause",  # package main
        ],
        "const_types": [
            "const_declaration",  # const Foo = 123
        ],
        "var_types": [
            "var_declaration",  # var foo int
        ],
        # Examples of AST nodes:
        # function_declaration:
        #   name: identifier
        #   parameters: parameter_list
        #   result: parameter_list | type_identifier (optional)
        #   body: block
        #
        # method_declaration:
        #   receiver: parameter_list
        #   name: field_identifier
        #   parameters: parameter_list
        #   result: parameter_list | type_identifier (optional)
        #   body: block
        #
        # type_declaration:
        #   name: type_identifier
        # type: struct_type | interface_type | ...
        #
        # import_declaration:
        #   import_spec: package_identifier string_literal
        #
        # call_expression:
        #   function: identifier | selector_expression
        #   arguments: argument_list
    },
    # --------------------------------------------------------------------------
    # Java Configuration
    # --------------------------------------------------------------------------
    "java": {
        "function_types": [
            "method_declaration",  # public void foo() { ... }
        ],
        "class_types": [
            "class_declaration",  # public class MyClass { ... }
            "interface_declaration",  # public interface MyInterface { ... }
            "enum_declaration",  # public enum MyEnum { ... }
        ],
        "method_types": [
            "method_declaration",  # Methods inside class bodies
            "constructor_declaration",  # public MyClass() { ... }
        ],
        "constant_types": [
            "field_declaration",  # private static final int MAX = 100;
        ],
        "import_types": [
            "import_declaration",  # import java.util.List;
        ],
        "call_types": [
            "method_invocation",  # obj.method() or method()
            "object_creation_expression",  # new MyClass()
        ],
        "package_types": [
            "package_declaration",  # package com.example;
        ],
        "annotation_types": [
            "marker_annotation",  # @Override
            "annotation",  # @SuppressWarnings("unchecked")
        ],
        "docstring_field": "comment",  # Javadoc comments (/** ... */)
        "identifier_field": "name",
        "body_field": "body",
        # Additional Java-specific nodes
        "interface_types": [
            "interface_declaration",  # interface Foo { ... }
        ],
        "enum_types": [
            "enum_declaration",  # enum Foo { A, B, C }
        ],
        # Examples of AST nodes:
        # class_declaration:
        #   modifiers: public, abstract, final, etc.
        #   name: identifier
        #   superclass: type_identifier (optional)
        #   super_interfaces: type_list (optional)
        #   body: class_body
        #
        # method_declaration:
        #   modifiers: public, static, etc.
        # type: type_identifier | void_type
        #   name: identifier
        #   parameters: formal_parameters
        #   body: block
        #
        # import_declaration:
        #   path: scoped_identifier
    },
    # --------------------------------------------------------------------------
    # Kotlin Configuration
    # --------------------------------------------------------------------------
    "kotlin": {
        "function_types": [
            "function_declaration",  # fun foo() { ... }
        ],
        "class_types": [
            "class_declaration",  # class MyClass { ... }
            "object_declaration",  # object Singleton { ... }
        ],
        "method_types": [
            "function_declaration",  # Methods inside class bodies (same node type)
        ],
        "constant_types": [
            "property_declaration",  # val/var declarations
        ],
        "import_types": [
            "import_header",  # import kotlin.collections.List
        ],
        "call_types": [
            "call_expression",  # foo() or obj.foo()
        ],
        "package_types": [
            "package_header",  # package com.example
        ],
        "annotation_types": [
            "annotation",  # @Annotation
        ],
        "docstring_field": "comment",  # KDoc comments (/** ... */)
        "identifier_field": "simple_identifier",
        "body_field": "class_body",
        # Kotlin-specific features
        "interface_types": [
            "class_declaration",  # interface in Kotlin uses class_declaration
        ],
        "enum_types": [
            "class_declaration",  # enum class uses class_declaration
        ],
        "object_types": [
            "object_declaration",  # object Singleton { ... }
            "companion_object",  # companion object { ... }
        ],
        "data_class_types": [
            "class_declaration",  # data class uses class_declaration with modifier
        ],
        "lambda_types": [
            "lambda_literal",  # { x -> x + 1 }
        ],
        # Examples of AST nodes:
        # class_declaration:
        #   modifiers: data, sealed, open, etc.
        #   name: type_identifier
        #   primary_constructor: (optional)
        #   delegation_specifiers: superclass/interfaces
        #   body: class_body
        #
        # function_declaration:
        #   modifiers: suspend, override, etc.
        #   name: simple_identifier
        #   parameters: function_value_parameters
        #   return_type: type (optional)
        #   body: function_body
        #
        # object_declaration:
        #   name: type_identifier
        #   body: class_body
        #
        # import_header:
        #   identifier: package path
    },
}


# ==============================================================================
# Helper Functions
# ==============================================================================


def get_language_for_file(filepath: str) -> Optional[str]:
    """
    Determine the programming language from a file path.

    Args:
        filepath: Path to the file (can be relative or absolute)

    Returns:
        Language name (e.g., 'python', 'javascript') or None if not supported

    Examples:
        >>> get_language_for_file('example.py')
        'python'
        >>> get_language_for_file('/path/to/script.js')
        'javascript'
        >>> get_language_for_file('unknown.txt')
        None
    """
    path = Path(filepath)
    extension = path.suffix.lower()
    return EXTENSION_MAP.get(extension)


def get_config_for_language(language: str) -> Dict[str, any]:
    """
    Retrieve the tree-sitter configuration for a given language.

    Args:
        language: Language name (e.g., 'python', 'javascript')

    Returns:
        Configuration dictionary with node type mappings

    Raises:
        KeyError: If the language is not supported

    Examples:
        >>> config = get_config_for_language('python')
        >>> config['function_types']
        ['function_definition']
    """
    if language not in LANGUAGE_CONFIGS:
        raise KeyError(
            f"Unsupported language: {language}. "
            f"Supported languages: {', '.join(LANGUAGE_CONFIGS.keys())}"
        )
    return LANGUAGE_CONFIGS[language]


def get_supported_extensions() -> Set[str]:
    """
    Get a set of all supported file extensions.

    Returns:
        Set of file extensions (including the dot)

    Examples:
        >>> extensions = get_supported_extensions()
        >>> '.py' in extensions
        True
    """
    return set(EXTENSION_MAP.keys())


def validate_config(language: str) -> bool:
    """
    Validate that a language configuration has all required fields.

    Args:
        language: Language name to validate

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is missing required fields
    """
    required_fields = [
        "function_types",
        "class_types",
        "method_types",
        "import_types",
        "call_types",
        "docstring_field",
        "identifier_field",
        "body_field",
    ]

    config = get_config_for_language(language)
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        raise ValueError(
            f"Configuration for {language} is missing required fields: {', '.join(missing_fields)}"
        )

    return True


# ==============================================================================
# Configuration Validation
# ==============================================================================

# Validate all configurations on module import
for lang in LANGUAGE_CONFIGS.keys():
    try:
        validate_config(lang)
    except ValueError as e:
        raise ValueError(f"Invalid configuration detected: {e}")
