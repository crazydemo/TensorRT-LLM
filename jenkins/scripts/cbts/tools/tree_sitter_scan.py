#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scan one Python source file with Tree-sitter and write a SQLite index."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    language TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    byte_count INTEGER NOT NULL,
    line_count INTEGER NOT NULL,
    parse_ms REAL NOT NULL,
    has_error INTEGER NOT NULL,
    error_node_count INTEGER NOT NULL
);

CREATE TABLE symbols (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    parent_symbol_id INTEGER REFERENCES symbols(id),
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    pytest_node_id TEXT,
    parameters TEXT,
    bases TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL
);

CREATE TABLE imports (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    caller_symbol_id INTEGER REFERENCES symbols(id),
    kind TEXT NOT NULL,
    statement TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL
);

CREATE TABLE calls (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    caller_symbol_id INTEGER REFERENCES symbols(id),
    caller_qualified_name TEXT NOT NULL,
    callee_text TEXT NOT NULL,
    is_torch_op INTEGER NOT NULL,
    is_tensorrt_llm_binding INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL
);

CREATE TABLE decorators (
    id INTEGER PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id),
    expression TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL
);

CREATE TABLE parse_errors (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    node_type TEXT NOT NULL,
    text TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL
);

CREATE INDEX idx_symbols_kind ON symbols(kind);
CREATE INDEX idx_symbols_pytest_node_id ON symbols(pytest_node_id);
CREATE INDEX idx_calls_caller ON calls(caller_symbol_id);
CREATE INDEX idx_calls_callee ON calls(callee_text);

CREATE VIEW test_cases AS
SELECT
    id,
    qualified_name,
    pytest_node_id,
    parameters,
    start_line,
    end_line
FROM symbols
WHERE kind IN ('test_function', 'test_method');

CREATE VIEW case_calls AS
SELECT
    s.pytest_node_id,
    s.qualified_name AS test_case,
    c.callee_text,
    c.is_torch_op,
    c.is_tensorrt_llm_binding,
    c.start_line
FROM calls AS c
JOIN symbols AS s ON s.id = c.caller_symbol_id
WHERE s.kind IN ('test_function', 'test_method');
"""


@dataclass(frozen=True)
class Scope:
    symbol_id: int | None
    qualified_name: str
    class_names: tuple[str, ...]


class PythonIndexer:
    """Extract Python syntax facts from a Tree-sitter parse tree."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        file_id: int,
        relative_path: str,
        source: bytes,
    ) -> None:
        self._connection = connection
        self._file_id = file_id
        self._relative_path = relative_path
        self._source = source

    def index(self, root: Node) -> None:
        """Index definitions and expressions reachable from the root node."""
        self._walk(root, Scope(None, "<module>", ()))

    def _walk(self, node: Node, scope: Scope) -> None:
        if node.type == "decorated_definition":
            definition = next(
                (
                    child
                    for child in node.named_children
                    if child.type in {"class_definition", "function_definition"}
                ),
                None,
            )
            if definition is not None:
                decorators = [child for child in node.named_children if child.type == "decorator"]
                self._index_definition(definition, scope, decorators)
            return

        if node.type in {"class_definition", "function_definition"}:
            self._index_definition(node, scope, [])
            return

        if node.type in {"import_statement", "import_from_statement"}:
            self._insert_import(node, scope)

        if node.type == "call":
            self._insert_call(node, scope)

        for child in node.named_children:
            self._walk(child, scope)

    def _index_definition(self, node: Node, scope: Scope, decorators: list[Node]) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = self._text(name_node)
        qualified_name = (
            name if scope.qualified_name == "<module>" else f"{scope.qualified_name}.{name}"
        )
        is_class = node.type == "class_definition"
        parameters_node = node.child_by_field_name("parameters")
        superclasses_node = node.child_by_field_name("superclasses")
        parameters = self._text(parameters_node) if parameters_node is not None else None
        bases = self._text(superclasses_node) if superclasses_node is not None else None

        if is_class:
            kind = "class"
        elif name.startswith("test_"):
            kind = "test_method" if scope.class_names else "test_function"
        else:
            kind = "method" if scope.class_names else "function"

        pytest_node_id = None
        if kind in {"test_function", "test_method"}:
            parts = [self._relative_path, *scope.class_names, name]
            pytest_node_id = "::".join(parts)

        cursor = self._connection.execute(
            """
            INSERT INTO symbols(
                file_id, parent_symbol_id, kind, name, qualified_name,
                pytest_node_id, parameters, bases, start_line, end_line
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._file_id,
                scope.symbol_id,
                kind,
                name,
                qualified_name,
                pytest_node_id,
                parameters,
                bases,
                self._start_line(node),
                self._end_line(node),
            ),
        )
        symbol_id = int(cursor.lastrowid)

        for decorator in decorators:
            expression = self._text(decorator).removeprefix("@").strip()
            self._connection.execute(
                """
                INSERT INTO decorators(symbol_id, expression, start_line, end_line)
                VALUES (?, ?, ?, ?)
                """,
                (
                    symbol_id,
                    expression,
                    self._start_line(decorator),
                    self._end_line(decorator),
                ),
            )

        class_names = (*scope.class_names, name) if is_class else scope.class_names
        child_scope = Scope(symbol_id, qualified_name, class_names)
        body = node.child_by_field_name("body")
        if body is not None:
            self._walk(body, child_scope)

    def _insert_import(self, node: Node, scope: Scope) -> None:
        self._connection.execute(
            """
            INSERT INTO imports(
                file_id, caller_symbol_id, kind, statement, start_line, end_line
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self._file_id,
                scope.symbol_id,
                node.type,
                self._text(node),
                self._start_line(node),
                self._end_line(node),
            ),
        )

    def _insert_call(self, node: Node, scope: Scope) -> None:
        function = node.child_by_field_name("function")
        if function is None:
            return
        callee = self._text(function)
        self._connection.execute(
            """
            INSERT INTO calls(
                file_id, caller_symbol_id, caller_qualified_name, callee_text,
                is_torch_op, is_tensorrt_llm_binding, start_line, end_line
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._file_id,
                scope.symbol_id,
                scope.qualified_name,
                callee,
                int(callee.startswith("torch.ops.")),
                int(callee.startswith("tensorrt_llm.bindings.")),
                self._start_line(node),
                self._end_line(node),
            ),
        )

    def _text(self, node: Node) -> str:
        return self._source[node.start_byte : node.end_byte].decode("utf-8")

    @staticmethod
    def _start_line(node: Node) -> int:
        return node.start_point.row + 1

    @staticmethod
    def _end_line(node: Node) -> int:
        return node.end_point.row + 1


def _collect_error_nodes(root: Node) -> list[Node]:
    errors: list[Node] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_error or node.is_missing:
            errors.append(node)
        stack.extend(node.named_children)
    return errors


def _relative_path(source_path: Path, repo_root: Path | None) -> str:
    if repo_root is None:
        return source_path.as_posix()
    try:
        return source_path.relative_to(repo_root).as_posix()
    except ValueError:
        return source_path.as_posix()


def scan(source_path: Path, output_path: Path, repo_root: Path | None) -> None:
    """Parse one Python file and replace the target SQLite database."""
    import tree_sitter_python
    from tree_sitter import Language, Parser

    source = source_path.read_bytes()
    language = Language(tree_sitter_python.language())
    parser = Parser(language)

    start = time.perf_counter()
    tree = parser.parse(source)
    parse_ms = (time.perf_counter() - start) * 1_000
    errors = _collect_error_nodes(tree.root_node)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.unlink(missing_ok=True)

    with sqlite3.connect(temporary_path) as connection:
        connection.executescript(SCHEMA)
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tree_sitter_version": importlib.metadata.version("tree-sitter"),
            "tree_sitter_python_version": importlib.metadata.version("tree-sitter-python"),
        }
        connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items())
        relative_path = _relative_path(source_path, repo_root)
        cursor = connection.execute(
            """
            INSERT INTO files(
                path, language, sha256, byte_count, line_count, parse_ms,
                has_error, error_node_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                relative_path,
                "python",
                hashlib.sha256(source).hexdigest(),
                len(source),
                len(source.splitlines()),
                parse_ms,
                int(tree.root_node.has_error),
                len(errors),
            ),
        )
        file_id = int(cursor.lastrowid)
        PythonIndexer(connection, file_id, relative_path, source).index(tree.root_node)
        for node in errors:
            connection.execute(
                """
                INSERT INTO parse_errors(
                    file_id, node_type, text, start_line, end_line
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    node.type,
                    source[node.start_byte : node.end_byte].decode("utf-8"),
                    node.start_point.row + 1,
                    node.end_point.row + 1,
                ),
            )

    temporary_path.replace(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan one Python source file with Tree-sitter into SQLite."
    )
    parser.add_argument("source", type=Path, help="Python source file to scan.")
    parser.add_argument("--output", type=Path, required=True, help="Output SQLite path.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root used to store a relative source path.",
    )
    args = parser.parse_args()

    source_path = args.source.resolve()
    if not source_path.is_file():
        parser.error(f"source file does not exist: {source_path}")
    if source_path.suffix != ".py":
        parser.error(f"only Python source is supported: {source_path}")

    repo_root = args.repo_root.resolve() if args.repo_root is not None else None
    scan(source_path, args.output.resolve(), repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
