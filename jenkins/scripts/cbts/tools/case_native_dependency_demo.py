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
"""Build an evidence-graded case-to-native dependency demo database.

The input is a compact CBTS PY_START coverage database. Runtime touches narrow
the Python functions that are parsed. Tree-sitter then extracts explicit
``torch.ops`` and nanobind calls and follows uniquely resolved native symbols.

This is deliberately a demo, not a compiler. Conditional calls and ambiguous
C++ dispatches remain candidates and carry lower confidence in the output.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sqlite3
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:
    from tree_sitter import Node, Parser


BRANCH_NODES = {
    "conditional_expression",
    "for_statement",
    "if_statement",
    "match_statement",
    "try_statement",
    "while_statement",
    "with_statement",
}
CPP_SUFFIXES = {".cc", ".cpp", ".cu", ".cuh", ".cxx", ".h", ".hpp"}
CPP_FUNCTION_SUFFIXES = {".cc", ".cpp", ".cu", ".cxx"}
PYTHON_CUSTOM_OP_RE = re.compile(r"@torch\.library\.custom_op\(\s*[\"'](?P<name>[^\"']+)[\"']")
NANOBIND_DEF_RE = re.compile(
    r"\bm\.def\(\s*[\"'](?P<name>[^\"']+)[\"']\s*,\s*&(?P<target>[A-Za-z_][\w:]*(?:\(\))?)"
)
NANOBIND_CLASS_RE = re.compile(r"\bnb::class_<\s*(?P<type>[A-Za-z_][\w:]*)")
NANOBIND_CLASS_METHOD_RE = re.compile(
    r"\.def(?:_prop_ro|_prop_rw|_ro|_rw)?\(\s*[\"'](?P<name>[^\"']+)[\"']"
    r"\s*,\s*&(?P<target>[A-Za-z_][\w:]*(?:\(\))?)"
)
TORCH_IMPL_RE = re.compile(
    r"\bm\.(?P<kind>def|impl)\(\s*[\"'](?P<name>[^\"'(]+)"
    r"(?:\([^\"']*)?[\"'](?:\s*,\s*&(?P<target>[A-Za-z_][\w:]*))?"
)


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE selected_case (
    test TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    outcome TEXT,
    expected_workers INTEGER NOT NULL,
    saved_procs INTEGER NOT NULL,
    touched_files INTEGER NOT NULL,
    touched_symbols INTEGER NOT NULL
);

CREATE TABLE python_touch (
    file TEXT NOT NULL,
    qualname TEXT NOT NULL,
    source_exists INTEGER NOT NULL,
    PRIMARY KEY(file, qualname)
) WITHOUT ROWID;

CREATE TABLE boundary (
    id INTEGER PRIMARY KEY,
    python_file TEXT NOT NULL,
    python_qualname TEXT NOT NULL,
    line INTEGER NOT NULL,
    expression TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    conditional_depth INTEGER NOT NULL,
    confidence TEXT NOT NULL,
    evidence TEXT NOT NULL,
    UNIQUE(python_file, python_qualname, line, expression)
);

CREATE TABLE implementation (
    id INTEGER PRIMARY KEY,
    boundary_id INTEGER NOT NULL REFERENCES boundary(id),
    kind TEXT NOT NULL,
    file TEXT NOT NULL,
    line INTEGER NOT NULL,
    symbol TEXT,
    confidence TEXT NOT NULL,
    evidence TEXT NOT NULL,
    UNIQUE(boundary_id, kind, file, line, symbol)
);

CREATE TABLE native_edge (
    id INTEGER PRIMARY KEY,
    boundary_id INTEGER NOT NULL REFERENCES boundary(id),
    hop INTEGER NOT NULL,
    caller_file TEXT NOT NULL,
    caller_line INTEGER NOT NULL,
    caller_symbol TEXT NOT NULL,
    callee_expression TEXT NOT NULL,
    callee_file TEXT NOT NULL,
    callee_line INTEGER NOT NULL,
    callee_symbol TEXT NOT NULL,
    confidence TEXT NOT NULL,
    evidence TEXT NOT NULL,
    UNIQUE(
        boundary_id, hop, caller_file, caller_line, caller_symbol,
        callee_expression, callee_file, callee_line, callee_symbol
    )
);

CREATE INDEX idx_boundary_name ON boundary(canonical_name);
CREATE INDEX idx_implementation_boundary ON implementation(boundary_id);
CREATE INDEX idx_native_edge_boundary ON native_edge(boundary_id);

CREATE VIEW boundary_summary AS
SELECT
    boundary.id,
    boundary.kind,
    boundary.canonical_name,
    boundary.confidence,
    boundary.conditional_depth,
    boundary.python_file,
    boundary.python_qualname,
    boundary.line,
    COUNT(DISTINCT implementation.id) AS implementations,
    COUNT(DISTINCT native_edge.id) AS native_edges
FROM boundary
LEFT JOIN implementation ON implementation.boundary_id = boundary.id
LEFT JOIN native_edge ON native_edge.boundary_id = boundary.id
GROUP BY boundary.id;

CREATE VIEW native_files AS
SELECT DISTINCT boundary_id, file, confidence, kind
FROM implementation
WHERE file GLOB '*.cpp' OR file GLOB '*.cc' OR file GLOB '*.cxx' OR file GLOB '*.cu'
UNION
SELECT DISTINCT boundary_id, callee_file, confidence, 'native_call'
FROM native_edge;
"""


@dataclass(frozen=True)
class Boundary:
    python_file: str
    python_qualname: str
    line: int
    expression: str
    canonical_name: str
    kind: str
    conditional_depth: int
    confidence: str
    evidence: str


@dataclass(frozen=True)
class FunctionDefinition:
    path: str
    line: int
    qualified_name: str
    base_name: str
    calls: tuple[tuple[str, int], ...]
    parameter_types: tuple[tuple[str, str], ...]


NativeEdge = tuple[int, FunctionDefinition, str, int, FunctionDefinition, str]


def _text(source: bytes, node: Node) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _line(node: Node) -> int:
    return node.start_point.row + 1


def _walk(node: Node) -> Iterator[Node]:
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.named_children))


def _load_parsers() -> tuple[Parser, Parser]:
    try:
        import tree_sitter_cpp
        import tree_sitter_python
        from tree_sitter import Language, Parser
    except ImportError as error:
        raise RuntimeError(
            "Tree-sitter dependencies are missing. Install tree-sitter, "
            "tree-sitter-python, and tree-sitter-cpp."
        ) from error

    return (
        Parser(Language(tree_sitter_python.language())),
        Parser(Language(tree_sitter_cpp.language())),
    )


def _load_case(
    coverage_db: Path, test: str
) -> tuple[tuple[str, str | None, int, int], list[tuple[str, str]]]:
    uri = f"file:{coverage_db.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            """
            SELECT stage, outcome, expected_workers, saved_procs
            FROM test_case_meta
            WHERE test = ?
            """,
            (test,),
        ).fetchone()
        if row is None:
            matches = connection.execute(
                "SELECT test FROM test_case_meta WHERE test LIKE ? ORDER BY test LIMIT 10",
                (f"%{test}%",),
            ).fetchall()
            suggestions = "\n".join(f"  {item[0]}" for item in matches)
            suffix = f"\nClose matches:\n{suggestions}" if suggestions else ""
            raise ValueError(f"case not found in coverage DB: {test}{suffix}")
        touches = connection.execute(
            "SELECT file, qualname FROM touch_rows WHERE test = ? ORDER BY file, qualname",
            (test,),
        ).fetchall()
    if not touches:
        raise ValueError(f"case has no saved Python touches: {test}")
    return (str(row[0]), row[1], int(row[2]), int(row[3])), touches


def _python_import_aliases(source: bytes, root: Node) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in _walk(root):
        if node.type not in {"import_statement", "import_from_statement"}:
            continue
        statement = _text(source, node).strip()
        match = re.match(r"from\s+([\w.]+)\s+import\s+(.+)$", statement, re.DOTALL)
        if match:
            module = match.group(1)
            for item in match.group(2).strip("() ").split(","):
                pieces = item.strip().split()
                if not pieces:
                    continue
                imported = pieces[0]
                local = pieces[2] if len(pieces) >= 3 and pieces[1] == "as" else imported
                aliases[local] = f"{module}.{imported}"
            continue
        match = re.match(r"import\s+([\w.]+)(?:\s+as\s+(\w+))?$", statement)
        if match:
            module = match.group(1)
            aliases[match.group(2) or module.split(".")[0]] = module
    return aliases


def _python_definition_name(source: bytes, node: Node) -> str | None:
    name = node.child_by_field_name("name")
    return _text(source, name) if name is not None else None


def _canonical_boundary(expression: str, aliases: dict[str, str]) -> tuple[str, str] | None:
    if expression.startswith("torch.ops."):
        parts = expression.split(".")
        if len(parts) >= 4:
            return "torch_op", f"{parts[2]}::{parts[3]}"
        return None

    first, separator, rest = expression.partition(".")
    resolved = aliases.get(first)
    if resolved and resolved.startswith("tensorrt_llm.bindings") and separator:
        return "nanobind", f"{resolved}.{rest}"
    if expression.startswith("tensorrt_llm.bindings."):
        return "nanobind", expression
    return None


def _extract_python_boundaries(
    repo_root: Path,
    touches: Iterable[tuple[str, str]],
    parser: Parser,
) -> tuple[list[Boundary], list[tuple[str, str, int]]]:
    by_file: dict[str, set[str]] = defaultdict(set)
    for path, qualname in touches:
        by_file[path].add(qualname)

    boundaries: list[Boundary] = []
    source_status: list[tuple[str, str, int]] = []
    for relative_path, touched_qualnames in by_file.items():
        path = repo_root / relative_path
        exists = path.is_file()
        source_status.extend(
            (relative_path, qualname, int(exists)) for qualname in touched_qualnames
        )
        if not exists or path.suffix != ".py":
            continue
        source = path.read_bytes()
        root = parser.parse(source).root_node
        aliases = _python_import_aliases(source, root)

        def visit(node: Node, scope: tuple[str, ...], conditional_depth: int) -> None:
            if node.type in {"class_definition", "function_definition"}:
                name = _python_definition_name(source, node)
                if name is None:
                    return
                child_scope = (*scope, name)
                qualified_name = ".".join(child_scope)
                body = node.child_by_field_name("body")
                if body is None:
                    return
                if node.type == "function_definition" and qualified_name in touched_qualnames:
                    visit(body, child_scope, 0)
                    return
                for child in body.named_children:
                    if child.type in {"class_definition", "function_definition"}:
                        visit(child, child_scope, 0)
                return

            next_depth = conditional_depth + int(node.type in BRANCH_NODES)
            if node.type == "call":
                function = node.child_by_field_name("function")
                if function is not None:
                    expression = _text(source, function)
                    canonical = _canonical_boundary(expression, aliases)
                    if canonical is not None:
                        kind, canonical_name = canonical
                        confidence = "high" if next_depth == 0 else "candidate"
                        evidence = (
                            "PY_START touched this function; boundary call is syntactically unconditional"
                            if next_depth == 0
                            else "PY_START touched this function; boundary call is inside control flow"
                        )
                        boundaries.append(
                            Boundary(
                                relative_path,
                                ".".join(scope),
                                _line(node),
                                expression,
                                canonical_name,
                                kind,
                                next_depth,
                                confidence,
                                evidence,
                            )
                        )
            for child in node.named_children:
                if child.type not in {"class_definition", "function_definition"}:
                    visit(child, scope, next_depth)

        for child in root.named_children:
            visit(child, (), 0)

    return sorted(
        set(boundaries),
        key=lambda item: (
            item.python_file,
            item.line,
            item.expression,
        ),
    ), source_status


def _source_line(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def _find_python_custom_ops(repo_root: Path, canonical_name: str) -> list[tuple[str, int, str]]:
    results: list[tuple[str, int, str]] = []
    for path in (repo_root / "tensorrt_llm").rglob("*.py"):
        source = path.read_text(errors="replace")
        for match in PYTHON_CUSTOM_OP_RE.finditer(source):
            if match.group("name") == canonical_name:
                results.append(
                    (
                        path.relative_to(repo_root).as_posix(),
                        _source_line(source, match.start()),
                        canonical_name,
                    )
                )
    return results


def _torch_op_parts(canonical_name: str) -> tuple[str, str]:
    namespace, separator, op_name = canonical_name.partition("::")
    if not separator:
        raise ValueError(f"invalid torch op name: {canonical_name}")
    return namespace, op_name


def _find_cpp_torch_registrations(
    repo_root: Path, canonical_name: str
) -> list[tuple[str, int, str, str | None]]:
    _namespace, op_name = _torch_op_parts(canonical_name)
    results: list[tuple[str, int, str, str | None]] = []
    for path in (repo_root / "cpp").rglob("*"):
        if path.suffix not in CPP_SUFFIXES:
            continue
        source = path.read_text(errors="replace")
        if op_name not in source:
            continue
        for match in TORCH_IMPL_RE.finditer(source):
            if match.group("name") != op_name:
                continue
            results.append(
                (
                    path.relative_to(repo_root).as_posix(),
                    _source_line(source, match.start()),
                    f"torch_library_{match.group('kind')}",
                    match.group("target"),
                )
            )
    return results


def _nanobind_leaf(canonical_name: str) -> str:
    return canonical_name.rsplit(".", 1)[-1]


def _nanobind_class_registrations(source: str) -> Iterator[tuple[str, int, str]]:
    """Yield (class leaf, source offset, method target) from chained class bindings."""
    classes = list(NANOBIND_CLASS_RE.finditer(source))
    for index, class_match in enumerate(classes):
        class_type = class_match.group("type")
        class_leaf = class_type.rsplit("::", 1)[-1]
        segment_end = classes[index + 1].start() if index + 1 < len(classes) else len(source)
        segment = source[class_match.end() : segment_end]
        for method_match in NANOBIND_CLASS_METHOD_RE.finditer(segment):
            target = method_match.group("target")
            if not (target.startswith(f"{class_type}::") or target.startswith(f"{class_leaf}::")):
                continue
            yield class_leaf, class_match.end() + method_match.start(), target


def _find_nanobind_registrations(
    repo_root: Path, canonical_name: str
) -> list[tuple[str, int, str]]:
    leaf = _nanobind_leaf(canonical_name)
    results: list[tuple[str, int, str]] = []
    native_root = repo_root / "cpp" / "tensorrt_llm" / "nanobind"
    for path in native_root.rglob("*.cpp"):
        source = path.read_text(errors="replace")
        if leaf not in source:
            continue
        for match in NANOBIND_DEF_RE.finditer(source):
            if match.group("name") == leaf:
                results.append(
                    (
                        path.relative_to(repo_root).as_posix(),
                        _source_line(source, match.start()),
                        match.group("target"),
                    )
                )
        for class_leaf, offset, target in _nanobind_class_registrations(source):
            if class_leaf == leaf:
                results.append(
                    (
                        path.relative_to(repo_root).as_posix(),
                        _source_line(source, offset),
                        target,
                    )
                )
    return results


def _cpp_declarator_name(source: bytes, declarator: Node | None) -> str | None:
    current = declarator
    while current is not None:
        if current.type in {
            "destructor_name",
            "field_identifier",
            "identifier",
            "operator_name",
            "qualified_identifier",
        }:
            return _text(source, current)
        nested = current.child_by_field_name("declarator")
        if nested is not None:
            current = nested
            continue
        candidates = [
            child
            for child in current.named_children
            if child.type
            in {
                "destructor_name",
                "field_identifier",
                "identifier",
                "qualified_identifier",
                "template_function",
            }
        ]
        return _text(source, candidates[0]) if candidates else None
    return None


def _call_base(expression: str) -> str | None:
    without_templates = re.sub(r"<[^<>]*>", "", expression)
    pieces = re.split(r"::|->|\.", without_templates)
    if not pieces:
        return None
    match = re.search(r"[A-Za-z_]\w*", pieces[-1])
    return match.group(0) if match else None


def _cpp_calls(source: bytes, body: Node | None) -> tuple[tuple[str, int], ...]:
    if body is None:
        return ()
    calls: list[tuple[str, int]] = []
    for node in _walk(body):
        if node.type != "call_expression":
            continue
        function = node.child_by_field_name("function")
        if function is not None:
            calls.append((_text(source, function), _line(node)))
    return tuple(calls)


def _cpp_parameter_types(source: bytes, declarator: Node | None) -> tuple[tuple[str, str], ...]:
    if declarator is None:
        return ()
    result: list[tuple[str, str]] = []
    for node in _walk(declarator):
        if node.type != "parameter_declaration":
            continue
        parameter_declarator = node.child_by_field_name("declarator")
        name = _cpp_declarator_name(source, parameter_declarator)
        if name:
            result.append((name, _text(source, node)))
    return tuple(result)


def _index_cpp_functions(
    repo_root: Path,
    parser: Parser,
    source_paths: Iterable[Path] | None = None,
) -> list[FunctionDefinition]:
    definitions: list[FunctionDefinition] = []
    native_root = repo_root / "cpp"
    paths = native_root.rglob("*") if source_paths is None else source_paths
    for path in paths:
        if path.suffix not in CPP_FUNCTION_SUFFIXES:
            continue
        try:
            relative_parts = path.relative_to(repo_root).parts
        except ValueError:
            continue
        if "tests" in relative_parts or relative_parts[:4] == (
            "cpp",
            "kernels",
            "xqa",
            "test",
        ):
            continue
        source = path.read_bytes()
        root = parser.parse(source).root_node

        def visit(node: Node, scopes: tuple[str, ...]) -> None:
            next_scopes = scopes
            if node.type in {"class_specifier", "namespace_definition", "struct_specifier"}:
                name = node.child_by_field_name("name")
                if name is not None:
                    next_scopes = (*scopes, _text(source, name))
            if node.type == "function_definition":
                declarator = node.child_by_field_name("declarator")
                name = _cpp_declarator_name(source, declarator)
                # Namespace-opening macros such as TRTLLM_NAMESPACE_BEGIN are
                # intentionally opaque to Tree-sitter.  The C++ grammar may
                # recover by treating the following namespace as one giant
                # function named "namespace".  Do not let that recovery node
                # hide all real function definitions nested below it.
                is_namespace_recovery = name is not None and name.startswith("namespace")
                has_recovery_error = is_namespace_recovery and any(
                    child.type == "ERROR" for child in _walk(node)
                )
                if is_namespace_recovery and has_recovery_error:
                    for child in node.named_children:
                        visit(child, next_scopes)
                    return
                if name:
                    qualified_name = name if "::" in name else "::".join((*scopes, name))
                    base_name = qualified_name.rsplit("::", 1)[-1]
                    definitions.append(
                        FunctionDefinition(
                            path.relative_to(repo_root).as_posix(),
                            _line(node),
                            qualified_name,
                            base_name,
                            _cpp_calls(source, node.child_by_field_name("body")),
                            _cpp_parameter_types(source, declarator),
                        )
                    )
                return
            for child in node.named_children:
                visit(child, next_scopes)

        visit(root, ())
    return definitions


def _matches_target(definition: FunctionDefinition, target: str) -> bool:
    normalized = target.removeprefix("&").lstrip(":")
    return definition.qualified_name.endswith(normalized)


def _find_target_definitions(
    definitions: list[FunctionDefinition], target: str
) -> tuple[list[FunctionDefinition], str]:
    exact = [definition for definition in definitions if _matches_target(definition, target)]
    if exact:
        return exact, "registration target suffix match"

    # Tree-sitter does not expand namespace-opening macros.  In files wrapped
    # by TRTLLM_NAMESPACE_BEGIN, its error recovery can therefore preserve the
    # function name while losing the enclosing namespace.  A globally unique
    # base name is still useful evidence, but record the fallback explicitly.
    base = _call_base(target)
    by_base = [definition for definition in definitions if definition.base_name == base]
    if len(by_base) == 1:
        return by_base, "globally unique base-name fallback after parser recovery"
    return [], "registration target did not resolve uniquely"


def _resolve_call(
    caller: FunctionDefinition,
    expression: str,
    by_base: dict[str, list[FunctionDefinition]],
) -> tuple[FunctionDefinition, str] | None:
    base = _call_base(expression)
    if base is None:
        return None
    candidates = by_base.get(base, [])
    if not candidates:
        return None
    receiver_match = re.match(r"\s*([A-Za-z_]\w*)\s*(?:->|\.)", expression)
    if receiver_match:
        receiver = receiver_match.group(1)
        parameter_type = dict(caller.parameter_types).get(receiver)
        if parameter_type:
            type_names = set(re.findall(r"[A-Za-z_]\w*", parameter_type))
            typed = [
                candidate
                for candidate in candidates
                if any(
                    f"{type_name}::{base}" in candidate.qualified_name for type_name in type_names
                )
            ]
            if len(typed) == 1:
                return typed[0], "receiver parameter type and method name match"
    same_file = [candidate for candidate in candidates if candidate.path == caller.path]
    if len(same_file) == 1:
        return same_file[0], "unique definition in caller translation unit"
    qualifier = re.sub(r"<[^<>]*>", "", expression).replace(".", "::").replace("->", "::")
    qualified = [
        candidate
        for candidate in candidates
        if qualifier.endswith(candidate.qualified_name)
        or candidate.qualified_name.endswith(qualifier)
    ]
    if len(qualified) == 1:
        return qualified[0], "unique qualified symbol definition"
    if receiver_match:
        return None
    if len(candidates) == 1:
        return candidates[0], "unique symbol definition in native source tree"
    return None


def _native_closure(
    starts: Iterable[FunctionDefinition],
    definitions: list[FunctionDefinition],
    phase: str,
    max_depth: int,
) -> list[NativeEdge]:
    by_base: dict[str, list[FunctionDefinition]] = defaultdict(list)
    for definition in definitions:
        by_base[definition.base_name].append(definition)

    edges: list[NativeEdge] = []
    queue = deque((definition, 0) for definition in starts)
    visited = {(definition.path, definition.line) for definition in starts}
    while queue:
        caller, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for expression, line in caller.calls:
            base = _call_base(expression)
            if phase == "context" and base == "enqueueGeneration":
                continue
            if phase == "generation" and base == "enqueueContext":
                continue
            resolved = _resolve_call(caller, expression, by_base)
            if resolved is None:
                continue
            callee, evidence = resolved
            edge = (depth + 1, caller, expression, line, callee, evidence)
            edges.append(edge)
            key = (callee.path, callee.line)
            if key not in visited:
                visited.add(key)
                queue.append((callee, depth + 1))
    return edges


def _insert_boundary(connection: sqlite3.Connection, boundary: Boundary) -> int:
    cursor = connection.execute(
        """
        INSERT INTO boundary(
            python_file, python_qualname, line, expression, canonical_name,
            kind, conditional_depth, confidence, evidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            boundary.python_file,
            boundary.python_qualname,
            boundary.line,
            boundary.expression,
            boundary.canonical_name,
            boundary.kind,
            boundary.conditional_depth,
            boundary.confidence,
            boundary.evidence,
        ),
    )
    return int(cursor.lastrowid)


def build_demo(
    coverage_db: Path,
    repo_root: Path,
    test: str,
    output: Path,
    phase: str,
    native_depth: int,
) -> None:
    python_parser, cpp_parser = _load_parsers()
    case_meta, touches = _load_case(coverage_db, test)
    boundaries, source_status = _extract_python_boundaries(repo_root, touches, python_parser)

    cpp_start = time.perf_counter()
    definitions = _index_cpp_functions(repo_root, cpp_parser)
    cpp_parse_ms = (time.perf_counter() - cpp_start) * 1_000

    definitions_by_target: dict[str, list[FunctionDefinition]] = defaultdict(list)
    definition_resolution: dict[str, str] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    with sqlite3.connect(temporary) as connection:
        connection.executescript(SCHEMA)
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "coverage_db": coverage_db.as_posix(),
            "repo_root": repo_root.as_posix(),
            "phase_hint": phase,
            "native_depth": str(native_depth),
            "tree_sitter_version": importlib.metadata.version("tree-sitter"),
            "tree_sitter_python_version": importlib.metadata.version("tree-sitter-python"),
            "tree_sitter_cpp_version": importlib.metadata.version("tree-sitter-cpp"),
            "cpp_parse_ms": f"{cpp_parse_ms:.3f}",
            "native_function_definitions": str(len(definitions)),
            "confidence_semantics": (
                "boundary confidence describes lexical execution likelihood; "
                "native-edge confidence describes symbol-resolution strength, not runtime execution"
            ),
        }
        connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items())
        stage, outcome, expected_workers, saved_procs = case_meta
        connection.execute(
            """
            INSERT INTO selected_case(
                test, stage, outcome, expected_workers, saved_procs,
                touched_files, touched_symbols
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                test,
                stage,
                outcome,
                expected_workers,
                saved_procs,
                len({path for path, _qualname in touches}),
                len(touches),
            ),
        )
        connection.executemany(
            "INSERT INTO python_touch(file, qualname, source_exists) VALUES (?, ?, ?)",
            source_status,
        )

        for boundary in boundaries:
            boundary_id = _insert_boundary(connection, boundary)
            native_targets: list[str] = []
            if boundary.kind == "torch_op":
                for path, line, symbol in _find_python_custom_ops(
                    repo_root, boundary.canonical_name
                ):
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO implementation(
                            boundary_id, kind, file, line, symbol, confidence, evidence
                        ) VALUES (?, 'python_custom_op', ?, ?, ?, 'high', ?)
                        """,
                        (
                            boundary_id,
                            path,
                            line,
                            symbol,
                            "exact torch.library.custom_op registration string",
                        ),
                    )
                for path, line, kind, target in _find_cpp_torch_registrations(
                    repo_root, boundary.canonical_name
                ):
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO implementation(
                            boundary_id, kind, file, line, symbol, confidence, evidence
                        ) VALUES (?, ?, ?, ?, ?, 'high', ?)
                        """,
                        (
                            boundary_id,
                            kind,
                            path,
                            line,
                            target,
                            "exact m.def/m.impl operator-name match",
                        ),
                    )
                    if target:
                        native_targets.append(target)
            else:
                for path, line, target in _find_nanobind_registrations(
                    repo_root, boundary.canonical_name
                ):
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO implementation(
                            boundary_id, kind, file, line, symbol, confidence, evidence
                        ) VALUES (?, 'nanobind_registration', ?, ?, ?, 'high', ?)
                        """,
                        (
                            boundary_id,
                            path,
                            line,
                            target,
                            "exact nanobind m.def name match",
                        ),
                    )
                    native_targets.append(target)

            starts: list[FunctionDefinition] = []
            for target in sorted(set(native_targets)):
                if target not in definitions_by_target:
                    (
                        definitions_by_target[target],
                        definition_resolution[target],
                    ) = _find_target_definitions(definitions, target)
                matches = definitions_by_target[target]
                resolution_evidence = definition_resolution[target]
                confidence = "high" if len(matches) == 1 else "candidate"
                for definition in matches:
                    starts.append(definition)
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO implementation(
                            boundary_id, kind, file, line, symbol, confidence, evidence
                        ) VALUES (?, 'native_definition', ?, ?, ?, ?, ?)
                        """,
                        (
                            boundary_id,
                            definition.path,
                            definition.line,
                            definition.qualified_name,
                            confidence,
                            f"{resolution_evidence}; resolved to {len(matches)} definition(s)",
                        ),
                    )

            for hop, caller, expression, line, callee, evidence in _native_closure(
                starts, definitions, phase, native_depth
            ):
                confidence = (
                    "high" if "qualified" in evidence or "unique symbol" in evidence else "medium"
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO native_edge(
                        boundary_id, hop, caller_file, caller_line, caller_symbol,
                        callee_expression, callee_file, callee_line, callee_symbol,
                        confidence, evidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        boundary_id,
                        hop,
                        caller.path,
                        line,
                        caller.qualified_name,
                        expression,
                        callee.path,
                        callee.line,
                        callee.qualified_name,
                        confidence,
                        evidence,
                    ),
                )
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an evidence-graded case-to-native demo SQLite database."
    )
    parser.add_argument("--coverage-db", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--case", required=True, help="Exact test value from test_case_meta.test.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("any", "context", "generation"),
        default="any",
        help="Optional workload hint used to prune enqueueContext/enqueueGeneration.",
    )
    parser.add_argument("--native-depth", type=int, default=4)
    args = parser.parse_args()

    coverage_db = args.coverage_db.resolve()
    repo_root = args.repo_root.resolve()
    if not coverage_db.is_file():
        parser.error(f"coverage DB does not exist: {coverage_db}")
    if not repo_root.is_dir():
        parser.error(f"repository root does not exist: {repo_root}")
    if args.native_depth < 0:
        parser.error("--native-depth must be non-negative")
    try:
        build_demo(
            coverage_db,
            repo_root,
            args.case,
            args.output.resolve(),
            args.phase,
            args.native_depth,
        )
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
