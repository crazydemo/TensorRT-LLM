#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build a reusable static Python-to-C++/CUDA graph for CBTS dry-runs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import re
import sqlite3
import subprocess
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import case_native_dependency_demo as matcher

SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE python_boundary(
    id INTEGER PRIMARY KEY,
    python_file TEXT NOT NULL,
    python_qualname TEXT NOT NULL,
    line INTEGER NOT NULL,
    kind TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    conditional_depth INTEGER NOT NULL,
    UNIQUE(python_file, python_qualname, line, canonical_name)
);
CREATE TABLE python_native_file(
    python_file TEXT NOT NULL,
    python_qualname TEXT NOT NULL,
    native_file TEXT NOT NULL,
    min_hops INTEGER NOT NULL,
    evidence TEXT NOT NULL,
    PRIMARY KEY(python_file, python_qualname, native_file)
) WITHOUT ROWID;
CREATE INDEX idx_native_reverse ON python_native_file(native_file);
CREATE VIEW native_file_summary AS
SELECT native_file, COUNT(DISTINCT python_file || '::' || python_qualname) AS python_functions,
       MIN(min_hops) AS min_hops
FROM python_native_file GROUP BY native_file;
"""


@dataclass(frozen=True)
class PythonBoundary:
    file: str
    qualname: str
    line: int
    kind: str
    canonical_name: str
    conditional_depth: int


@dataclass(frozen=True)
class CompilationDatabase:
    """Project sources and completeness counters from compile_commands.json."""

    path: Path
    entry_count: int
    unique_source_count: int
    available_sources: tuple[Path, ...]
    missing_source_count: int
    outside_repo_count: int
    compiler_counts: tuple[tuple[str, int], ...]

    def metadata(self) -> list[tuple[str, str]]:
        """Serialize compilation database provenance for the graph metadata table."""
        return [
            ("compile_commands", self.path.as_posix()),
            ("compile_commands_sha256", hashlib.sha256(self.path.read_bytes()).hexdigest()),
            ("compile_commands_entries", str(self.entry_count)),
            ("compile_commands_unique_sources", str(self.unique_source_count)),
            ("compile_commands_available_sources", str(len(self.available_sources))),
            ("compile_commands_missing_sources", str(self.missing_source_count)),
            ("compile_commands_outside_repo", str(self.outside_repo_count)),
            (
                "compile_commands_compilers",
                ",".join(f"{compiler}={count}" for compiler, count in self.compiler_counts),
            ),
        ]


def _mapped_source(
    repo_root: Path,
    source: Path,
    source_root: Path | None,
) -> Path | None:
    if source_root is not None:
        try:
            return repo_root / source.relative_to(source_root)
        except ValueError:
            return None

    try:
        return repo_root / source.relative_to(repo_root)
    except ValueError:
        pass

    parts = source.parts
    try:
        cpp_index = parts.index("cpp")
    except ValueError:
        return None
    return repo_root.joinpath(*parts[cpp_index:])


def _compiler(command: str) -> str:
    fields = command.lstrip().split(maxsplit=1)
    return fields[0] if fields else "<unknown>"


def _load_compilation_database(
    repo_root: Path,
    path: Path,
    source_root: Path | None,
) -> CompilationDatabase:
    """Load and remap a CMake compilation database to this source checkout."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: unable to read compilation database: {error}") from error
    if not isinstance(payload, list):
        raise ValueError(f"{path}: compilation database must contain a JSON array")

    mapped_sources: dict[str, Path | None] = {}
    compiler_counts: Counter[str] = Counter()
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry {index} is not a JSON object")
        source_value = entry.get("file")
        command_value = entry.get("command")
        if not isinstance(source_value, str) or not source_value:
            raise ValueError(f"{path}: entry {index} has no string file field")
        if not isinstance(command_value, str) or not command_value:
            raise ValueError(f"{path}: entry {index} has no string command field")
        source = Path(source_value)
        mapped_sources.setdefault(source_value, _mapped_source(repo_root, source, source_root))
        compiler_counts[_compiler(command_value)] += 1

    available_sources = tuple(
        sorted(
            {
                mapped
                for mapped in mapped_sources.values()
                if mapped is not None and mapped.is_file()
            }
        )
    )
    missing_source_count = sum(
        mapped is not None and not mapped.is_file() for mapped in mapped_sources.values()
    )
    outside_repo_count = sum(mapped is None for mapped in mapped_sources.values())
    return CompilationDatabase(
        path=path,
        entry_count=len(payload),
        unique_source_count=len(mapped_sources),
        available_sources=available_sources,
        missing_source_count=missing_source_count,
        outside_repo_count=outside_repo_count,
        compiler_counts=tuple(sorted(compiler_counts.items())),
    )


def _aliases(tree: ast.AST) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                result[item.asname or item.name] = f"{node.module}.{item.name}"
        elif isinstance(node, ast.Import):
            for item in node.names:
                result[item.asname or item.name.split(".")[0]] = item.name
    return result


class _CallVisitor(ast.NodeVisitor):
    def __init__(self, file: str, qualname: str, aliases: dict[str, str]) -> None:
        self.file = file
        self.qualname = qualname
        self.aliases = aliases
        self.depth = 0
        self.boundaries: list[PythonBoundary] = []

    def visit_Call(self, node: ast.Call) -> None:
        expression = ast.unparse(node.func)
        canonical = matcher._canonical_boundary(expression, self.aliases)
        if canonical is not None:
            kind, name = canonical
            self.boundaries.append(
                PythonBoundary(self.file, self.qualname, node.lineno, kind, name, self.depth)
            )
        self.generic_visit(node)

    def _branch(self, node: ast.AST) -> None:
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_If = _branch
    visit_For = _branch
    visit_AsyncFor = _branch
    visit_While = _branch
    visit_Try = _branch
    visit_Match = _branch
    visit_IfExp = _branch

    def visit_FunctionDef(self, _node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, _node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, _node: ast.ClassDef) -> None:
        return


def _python_boundaries(repo_root: Path) -> list[PythonBoundary]:
    result: list[PythonBoundary] = []
    for path in (repo_root / "tensorrt_llm").rglob("*.py"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        relative = path.relative_to(repo_root).as_posix()
        aliases = _aliases(tree)

        def collect(body: list[ast.stmt], prefix: tuple[str, ...]) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    collect(node.body, (*prefix, node.name))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = ".".join((*prefix, node.name))
                    visitor = _CallVisitor(relative, qualname, aliases)
                    for statement in node.body:
                        visitor.visit(statement)
                    result.extend(visitor.boundaries)

        collect(tree.body, ())
    return sorted(set(result), key=lambda row: (row.file, row.qualname, row.line))


def _include_headers(repo_root: Path, native_file: str) -> set[str]:
    path = repo_root / native_file
    try:
        source = path.read_text(errors="replace")
    except OSError:
        return set()
    headers = set()
    for include in re.findall(r'^\s*#\s*include\s+"([^"]+)"', source, re.MULTILINE):
        candidate = repo_root / "cpp" / include
        if candidate.is_file():
            headers.add(candidate.relative_to(repo_root).as_posix())
    return headers


def _registration_indexes(
    repo_root: Path,
    source_paths: tuple[Path, ...] | None = None,
) -> tuple[
    dict[str, list[tuple[str, int, str]]],
    dict[str, list[tuple[str, int, str | None]]],
]:
    """Index nanobind and torch registrations with one native-tree pass."""
    nanobind: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    torch_ops: dict[str, list[tuple[str, int, str | None]]] = defaultdict(list)
    paths = (repo_root / "cpp").rglob("*") if source_paths is None else source_paths
    for path in paths:
        if path.suffix not in matcher.CPP_SUFFIXES:
            continue
        source = path.read_text(errors="replace")
        relative = path.relative_to(repo_root).as_posix()
        for match in matcher.NANOBIND_DEF_RE.finditer(source):
            nanobind[match.group("name")].append(
                (relative, matcher._source_line(source, match.start()), match.group("target"))
            )
        for class_leaf, offset, target in matcher._nanobind_class_registrations(source):
            nanobind[class_leaf].append((relative, matcher._source_line(source, offset), target))
        for match in matcher.TORCH_IMPL_RE.finditer(source):
            torch_ops[match.group("name")].append(
                (relative, matcher._source_line(source, match.start()), match.group("target"))
            )
    return dict(nanobind), dict(torch_ops)


def build(
    repo_root: Path,
    output: Path,
    native_depth: int,
    compile_commands: Path | None = None,
    compile_commands_source_root: Path | None = None,
) -> None:
    _python_parser, cpp_parser = matcher._load_parsers()
    started = time.perf_counter()
    compilation_database = (
        _load_compilation_database(repo_root, compile_commands, compile_commands_source_root)
        if compile_commands is not None
        else None
    )
    native_sources = (
        compilation_database.available_sources if compilation_database is not None else None
    )
    boundaries = _python_boundaries(repo_root)
    definitions = matcher._index_cpp_functions(repo_root, cpp_parser, native_sources)
    nanobind_registrations, torch_registrations = _registration_indexes(repo_root, native_sources)
    rows: dict[tuple[str, str, str], tuple[int, str]] = {}
    boundaries_without_registration = 0
    unresolved_registration_targets = 0
    ambiguous_registration_targets = 0

    def add(boundary: PythonBoundary, file: str, hops: int, evidence: str) -> None:
        key = (boundary.file, boundary.qualname, file)
        old = rows.get(key)
        if old is None or hops < old[0]:
            rows[key] = (hops, evidence)

    for boundary in boundaries:
        targets: list[str] = []
        registrations: list[tuple[str, int, str]] = []
        if boundary.kind == "nanobind":
            registrations = nanobind_registrations.get(
                matcher._nanobind_leaf(boundary.canonical_name), []
            )
        else:
            _namespace, op_name = matcher._torch_op_parts(boundary.canonical_name)
            for file, line, target in torch_registrations.get(op_name, []):
                if target:
                    registrations.append((file, line, target))
        if not registrations:
            boundaries_without_registration += 1
        for file, _line, target in registrations:
            add(boundary, file, 0, "exact registration-name match")
            targets.append(target)
        starts = []
        for target in sorted(set(targets)):
            matches, evidence = matcher._find_target_definitions(definitions, target)
            if not matches:
                unresolved_registration_targets += 1
                continue
            if len(matches) > 1:
                ambiguous_registration_targets += 1
                continue
            definition = matches[0]
            starts.append(definition)
            add(boundary, definition.path, 0, evidence)
        for hop, _caller, _expression, _line, callee, evidence in matcher._native_closure(
            starts, definitions, "any", native_depth
        ):
            add(boundary, callee.path, hop, evidence)

    for (python_file, python_qualname, native_file), (hops, evidence) in list(rows.items()):
        boundary = PythonBoundary(python_file, python_qualname, 0, "", "", 0)
        for header in _include_headers(repo_root, native_file):
            add(boundary, header, hops + 1, f"included by {native_file}")

    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(temporary) as connection:
        connection.executescript(SCHEMA)
        metadata = [
            ("schema_version", "1"),
            (
                "source_commit",
                subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip(),
            ),
            ("native_depth", str(native_depth)),
            ("tree_sitter_version", importlib.metadata.version("tree-sitter")),
            ("tree_sitter_python_version", importlib.metadata.version("tree-sitter-python")),
            ("tree_sitter_cpp_version", importlib.metadata.version("tree-sitter-cpp")),
            ("python_boundaries", str(len(boundaries))),
            ("native_sources_indexed", str(len(native_sources or ()))),
            ("native_function_definitions", str(len(definitions))),
            ("python_native_mappings", str(len(rows))),
            (
                "mapped_python_boundaries",
                str(len({(python_file, qualname) for python_file, qualname, _file in rows})),
            ),
            ("boundaries_without_registration", str(boundaries_without_registration)),
            ("unresolved_registration_targets", str(unresolved_registration_targets)),
            ("ambiguous_registration_targets", str(ambiguous_registration_targets)),
            ("build_seconds", f"{time.perf_counter() - started:.3f}"),
        ]
        if compilation_database is not None:
            metadata.extend(compilation_database.metadata())
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            metadata,
        )
        connection.executemany(
            "INSERT INTO python_boundary(python_file, python_qualname, line, kind, "
            "canonical_name, conditional_depth) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    row.file,
                    row.qualname,
                    row.line,
                    row.kind,
                    row.canonical_name,
                    row.conditional_depth,
                )
                for row in boundaries
            ],
        )
        connection.executemany(
            "INSERT INTO python_native_file VALUES (?, ?, ?, ?, ?)",
            [(*key, hops, evidence) for key, (hops, evidence) in rows.items()],
        )
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-depth", type=int, default=5)
    parser.add_argument(
        "--compile-commands",
        type=Path,
        help="CMake compile_commands.json used to restrict native parsing to built sources",
    )
    parser.add_argument(
        "--compile-commands-source-root",
        type=Path,
        help="source root recorded in compile_commands.json, for example /code/tensorrt_llm",
    )
    args = parser.parse_args()
    build(
        args.repo_root.resolve(),
        args.output.resolve(),
        args.native_depth,
        args.compile_commands.resolve() if args.compile_commands else None,
        args.compile_commands_source_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
