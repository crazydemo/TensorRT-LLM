#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit every Python nanobind/torch.ops boundary and render its native call graph."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import case_native_dependency_demo as matcher
from build_native_codegraph import PythonBoundary, _load_compilation_database, _python_boundaries
from render_native_coverage_report import (
    REPORT_MARKER,
    SCRIPT,
    STYLES,
    _badge,
    _document,
    _escape,
    _metrics,
    _page_name,
)


@dataclass(frozen=True)
class Registration:
    """One native registration site discovered in a built source."""

    file: str
    line: int
    kind: str
    target: str | None


@dataclass(frozen=True)
class Entry:
    """A uniquely resolved C++ entry for a registration target."""

    target: str
    file: str
    line: int
    symbol: str
    evidence: str


@dataclass(frozen=True)
class Edge:
    """One statically resolved C++ call edge."""

    hop: int
    caller_file: str
    caller_line: int
    caller_symbol: str
    expression: str
    callee_file: str
    callee_line: int
    callee_symbol: str
    evidence: str


def _registration_indexes(
    repo_root: Path, source_paths: Iterable[Path]
) -> tuple[dict[str, list[Registration]], dict[str, list[Registration]]]:
    nanobind: dict[str, list[Registration]] = defaultdict(list)
    torch_ops: dict[str, list[Registration]] = defaultdict(list)
    for path in source_paths:
        if path.suffix not in matcher.CPP_SUFFIXES:
            continue
        source = path.read_text(errors="replace")
        relative = path.relative_to(repo_root).as_posix()
        for match in matcher.NANOBIND_DEF_RE.finditer(source):
            nanobind[match.group("name")].append(
                Registration(
                    relative,
                    matcher._source_line(source, match.start()),
                    "nanobind_def",
                    match.group("target"),
                )
            )
        for class_leaf, offset, target in matcher._nanobind_class_registrations(source):
            nanobind[class_leaf].append(
                Registration(
                    relative,
                    matcher._source_line(source, offset),
                    "nanobind_class_method",
                    target,
                )
            )
        for match in matcher.TORCH_IMPL_RE.finditer(source):
            torch_ops[match.group("name")].append(
                Registration(
                    relative,
                    matcher._source_line(source, match.start()),
                    f"torch_library_{match.group('kind')}",
                    match.group("target"),
                )
            )
    return dict(nanobind), dict(torch_ops)


def _caller_cases(
    coverage_db: Path, boundaries: list[PythonBoundary]
) -> dict[tuple[str, str], set[str]]:
    wanted = {(boundary.file, boundary.qualname) for boundary in boundaries}
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    uri = f"file:{coverage_db.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        for test, file, qualname in connection.execute(
            "SELECT test, file, qualname FROM touch_rows WHERE test!=''"
        ):
            key = (file, qualname)
            if key in wanted:
                result[key].add(test)
    return dict(result)


def _edge_dicts(native_edges: list[matcher.NativeEdge]) -> list[Edge]:
    return [
        Edge(
            hop,
            caller.path,
            line,
            caller.qualified_name,
            expression,
            callee.path,
            callee.line,
            callee.qualified_name,
            evidence,
        )
        for hop, caller, expression, line, callee, evidence in native_edges
    ]


def _representative_path(entries: list[Entry], edges: list[Edge]) -> list[Edge]:
    """Prefer the shortest resolved path ending in a CUDA source."""
    adjacency: dict[tuple[str, str], list[Edge]] = defaultdict(list)
    for edge in edges:
        adjacency[(edge.caller_file, edge.caller_symbol)].append(edge)
    queue = deque(((entry.file, entry.symbol), []) for entry in entries)
    seen = {(entry.file, entry.symbol) for entry in entries}
    fallback: list[Edge] = []
    while queue:
        node, path = queue.popleft()
        if len(path) > len(fallback):
            fallback = path
        if path and path[-1].callee_file.endswith((".cu", ".cuh")):
            return path
        for edge in adjacency.get(node, []):
            callee = (edge.callee_file, edge.callee_symbol)
            if callee in seen:
                continue
            seen.add(callee)
            queue.append((callee, [*path, edge]))
    return fallback


def _audit_ops(
    boundaries: list[PythonBoundary],
    definitions: list[matcher.FunctionDefinition],
    nanobind_registrations: dict[str, list[Registration]],
    torch_registrations: dict[str, list[Registration]],
    caller_cases: dict[tuple[str, str], set[str]],
    native_depth: int,
) -> list[dict[str, Any]]:
    by_op: dict[tuple[str, str], list[PythonBoundary]] = defaultdict(list)
    for boundary in boundaries:
        by_op[(boundary.kind, boundary.canonical_name)].append(boundary)

    audits = []
    for (kind, canonical_name), callsites in sorted(by_op.items()):
        registration_key = (
            matcher._nanobind_leaf(canonical_name)
            if kind == "nanobind"
            else matcher._torch_op_parts(canonical_name)[1]
        )
        registration_index = nanobind_registrations if kind == "nanobind" else torch_registrations
        registrations = registration_index.get(registration_key, [])
        entries: list[Entry] = []
        unresolved_targets: list[str] = []
        ambiguous_targets: list[str] = []
        starts: list[matcher.FunctionDefinition] = []
        for target in sorted({row.target for row in registrations if row.target}):
            matches, evidence = matcher._find_target_definitions(definitions, target)
            if not matches:
                unresolved_targets.append(target)
            elif len(matches) > 1:
                ambiguous_targets.append(target)
            else:
                definition = matches[0]
                starts.append(definition)
                entries.append(
                    Entry(
                        target,
                        definition.path,
                        definition.line,
                        definition.qualified_name,
                        evidence,
                    )
                )
        edges = _edge_dicts(matcher._native_closure(starts, definitions, "any", native_depth))
        cases = set()
        callers = []
        for callsite in callsites:
            callsite_cases = caller_cases.get((callsite.file, callsite.qualname), set())
            cases.update(callsite_cases)
            callers.append(
                {
                    "python_file": callsite.file,
                    "python_qualname": callsite.qualname,
                    "line": callsite.line,
                    "conditional_depth": callsite.conditional_depth,
                    "inferred_case_count": len(callsite_cases),
                }
            )
        native_files = (
            {row.file for row in registrations}
            | {row.file for row in entries}
            | {edge.callee_file for edge in edges}
        )
        cuda_files = {file for file in native_files if file.endswith((".cu", ".cuh"))}
        if not registrations:
            status = "no_registration"
        elif not entries:
            status = "registration_only"
        elif cuda_files:
            status = "cuda_reachable"
        else:
            status = "cpp_resolved"
        audits.append(
            {
                "kind": kind,
                "canonical_name": canonical_name,
                "registration_key": registration_key,
                "status": status,
                "callsites": callers,
                "python_functions": len({(row.file, row.qualname) for row in callsites}),
                "conditional_callsites": sum(row.conditional_depth > 0 for row in callsites),
                "inferred_case_count": len(cases),
                "registrations": [asdict(row) for row in registrations],
                "entries": [asdict(row) for row in entries],
                "unresolved_targets": unresolved_targets,
                "ambiguous_targets": ambiguous_targets,
                "edges": [asdict(row) for row in edges],
                "native_files": sorted(native_files),
                "cuda_files": sorted(cuda_files),
                "representative_path": [
                    asdict(row) for row in _representative_path(entries, edges)
                ],
            }
        )
    return audits


def _summary(
    audits: list[dict[str, Any]],
    boundaries: list[PythonBoundary],
    definitions: list[matcher.FunctionDefinition],
    nanobind_registrations: dict[str, list[Registration]],
    torch_registrations: dict[str, list[Registration]],
) -> dict[str, int]:
    return {
        "python_boundary_ops": len(audits),
        "nanobind_boundary_ops": sum(row["kind"] == "nanobind" for row in audits),
        "torch_op_boundary_ops": sum(row["kind"] == "torch_op" for row in audits),
        "python_callsites": len(boundaries),
        "ops_with_inferred_case_coverage": sum(row["inferred_case_count"] > 0 for row in audits),
        "ops_with_registration": sum(bool(row["registrations"]) for row in audits),
        "ops_with_cpp_entry": sum(bool(row["entries"]) for row in audits),
        "ops_reaching_cuda": sum(bool(row["cuda_files"]) for row in audits),
        "registered_nanobind_names": len(nanobind_registrations),
        "registered_torch_names": len(torch_registrations),
        "native_function_definitions": len(definitions),
    }


def _source(path: str, line: int | None = None) -> str:
    label = f"{path}:{line}" if line else path
    return f'<span class="mono">{_escape(label)}</span>'


def _render_chain(audit: dict[str, Any]) -> str:
    cards = [
        ("Python boundary", audit["canonical_name"], "runtime op hit not collected"),
    ]
    for registration in audit["registrations"][:1]:
        cards.append(
            (
                registration["kind"],
                registration["target"] or audit["registration_key"],
                f"{registration['file']}:{registration['line']}",
            )
        )
    for entry in audit["entries"][:1]:
        cards.append(("C++ entry", entry["symbol"], f"{entry['file']}:{entry['line']}"))
    for edge in audit["representative_path"]:
        cards.append(
            (
                f"native hop {edge['hop']}",
                edge["callee_symbol"],
                f"{edge['callee_file']}:{edge['callee_line']}",
            )
        )
    return (
        '<div class="chain">'
        + '<div class="arrow">↓</div>'.join(
            '<div class="chain-card">'
            f'<div class="muted">{_escape(kind)}</div><strong class="mono">{_escape(symbol)}</strong>'
            f'<div class="mono muted">{_escape(location)}</div></div>'
            for kind, symbol, location in cards
        )
        + "</div>"
    )


def _table(headers: list[str], rows: list[list[Any]], table_id: str) -> str:
    rendered_rows = []
    for row in rows:
        search = " ".join(str(value or "") for value in row).lower()
        rendered_rows.append(
            f'<tr data-search="{_escape(search)}">'
            + "".join(f"<td>{value}</td>" for value in row)
            + "</tr>"
        )
    return (
        f'<div class="table-wrap"><table id="{_escape(table_id)}"><thead><tr>'
        + "".join(f"<th>{_escape(header)}</th>" for header in headers)
        + f"</tr></thead><tbody>{''.join(rendered_rows)}</tbody></table></div>"
    )


def _render_op_page(output: Path, audit: dict[str, Any]) -> None:
    caller_rows = [
        [
            _source(row["python_file"], row["line"]),
            f'<span class="mono">{_escape(row["python_qualname"])}</span>',
            row["conditional_depth"],
            row["inferred_case_count"],
        ]
        for row in audit["callsites"]
    ]
    registration_rows = [
        [
            _badge(row["kind"]),
            _source(row["file"], row["line"]),
            f'<span class="mono">{_escape(row["target"] or "")}</span>',
        ]
        for row in audit["registrations"]
    ]
    edge_rows = [
        [
            row["hop"],
            _source(row["caller_file"], row["caller_line"]),
            f'<span class="mono">{_escape(row["caller_symbol"])}</span>',
            _source(row["callee_file"], row["callee_line"]),
            f'<span class="mono">{_escape(row["callee_symbol"])}</span>',
            _escape(row["evidence"]),
        ]
        for row in audit["edges"]
    ]
    body = f"""
<nav class="nav"><a href="../index.html">← Boundary ops</a><a href="../registrations.html">Registrations</a></nav>
<div class="eyebrow">{_escape(audit["kind"])} · {_badge(audit["status"])}</div>
<h1 class="mono case-name">{_escape(audit["canonical_name"])}</h1>
<div class="notice">Case counts are inferred from runtime entry into the enclosing Python function.
The current DB does not prove that this op callsite executed. Native edges are static candidates.</div>
{
        _metrics(
            [
                (len(audit["callsites"]), "Python callsites"),
                (audit["python_functions"], "Python functions"),
                (audit["inferred_case_count"], "inferred cases"),
                (len(audit["registrations"]), "registrations"),
                (len(audit["entries"]), "resolved C++ entries"),
                (len(audit["cuda_files"]), "reachable CUDA files"),
            ]
        )
    }
<section class="panel"><h2>Representative boundary → CUDA path</h2>{_render_chain(audit)}</section>
<section class="panel"><h2>Python callsites</h2>{
        _table(
            ["Source", "Qualname", "Conditional depth", "Inferred cases"], caller_rows, "callers"
        )
    }</section>
<section class="panel"><h2>Registrations</h2>{
        _table(["Kind", "Source", "Target"], registration_rows, "registrations")
    }</section>
<section class="panel"><h2>Native call edges</h2><div class="toolbar"><input class="filter"
data-filter="edges" placeholder="Search C++/.cu path, symbol, or evidence…"></div>
{
        _table(
            ["Hop", "Caller source", "Caller symbol", "Callee source", "Callee symbol", "Evidence"],
            edge_rows,
            "edges",
        )
    }</section>"""
    (output / "ops" / _page_name(f"{audit['kind']}:{audit['canonical_name']}")).write_text(
        _document(audit["canonical_name"], body, depth=1), encoding="utf-8"
    )


def _render_report(
    output: Path,
    audits: list[dict[str, Any]],
    summary: dict[str, int],
    nanobind_registrations: dict[str, list[Registration]],
    torch_registrations: dict[str, list[Registration]],
    source_commit: str,
    available_source_count: int,
    missing_source_count: int,
) -> None:
    if output.exists() and any(output.iterdir()) and not (output / REPORT_MARKER).is_file():
        raise ValueError(f"refusing to replace non-report directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for subdirectory in ("assets", "ops"):
        path = output / subdirectory
        path.mkdir(exist_ok=True)
        for old_file in path.iterdir():
            old_file.unlink()
    (output / REPORT_MARKER).write_text("generated by audit_native_boundaries.py\n")
    (output / "assets/style.css").write_text(
        STYLES + ".chain{display:flex;flex-direction:column;gap:0;max-width:900px;margin:auto}"
        ".chain-card{padding:12px 15px;background:var(--panel2);border:1px solid var(--line);border-radius:9px}"
        ".arrow{text-align:center;color:var(--green);font-size:20px;height:28px}",
        encoding="utf-8",
    )
    (output / "assets/report.js").write_text(SCRIPT, encoding="utf-8")

    index_rows = []
    for audit in audits:
        key = f"{audit['kind']}:{audit['canonical_name']}"
        index_rows.append(
            [
                _badge(audit["kind"]),
                f'<a class="mono" href="ops/{_page_name(key)}">{_escape(audit["canonical_name"])}</a>',
                _badge(audit["status"]),
                len(audit["callsites"]),
                audit["inferred_case_count"],
                len(audit["registrations"]),
                len(audit["entries"]),
                len(audit["cuda_files"]),
            ]
        )
        _render_op_page(output, audit)
    body = f"""
<nav class="nav"><a href="index.html">Boundary ops</a><a href="registrations.html">Registrations</a></nav>
<div class="eyebrow">Current-commit cross-language audit</div><h1>nanobind / torch.ops coverage</h1>
<p class="muted">Source commit: <span class="mono">{_escape(source_commit)}</span></p>
<div class="notice">“Inferred cases” means PY_START entered a Python function containing the boundary.
It is not runtime op coverage. Registration and C++/CUDA graphs are static and intentionally conservative.</div>
<div class="notice">The native index parsed {available_source_count:,} compilation-database sources present
in this checkout. {missing_source_count:,} generated/build sources referenced by compile_commands.json are not
present locally, so deeper generated CUDA paths can be incomplete.</div>
{
        _metrics(
            [
                (summary["python_boundary_ops"], "Python boundary ops"),
                (summary["ops_with_inferred_case_coverage"], "ops with inferred cases"),
                (summary["ops_with_registration"], "ops with registrations"),
                (summary["ops_with_cpp_entry"], "ops with resolved C++ entry"),
                (summary["ops_reaching_cuda"], "ops reaching CUDA"),
                (summary["python_callsites"], "Python callsites"),
            ]
        )
    }
<section class="panel"><div class="toolbar"><input class="filter" data-filter="ops"
placeholder="Search op, kind, or resolution status…"></div>
{
        _table(
            [
                "Kind",
                "Canonical op",
                "Status",
                "Callsites",
                "Inferred cases",
                "Registrations",
                "C++ entries",
                "CUDA files",
            ],
            index_rows,
            "ops",
        )
    }</section>"""
    (output / "index.html").write_text(
        _document("nanobind / torch.ops audit", body), encoding="utf-8"
    )

    referenced = {(audit["kind"], audit["registration_key"]) for audit in audits}
    registration_rows = []
    for kind, registrations in (
        ("nanobind", nanobind_registrations),
        ("torch_op", torch_registrations),
    ):
        for name, rows in sorted(registrations.items()):
            registration_rows.append(
                [
                    _badge(kind),
                    f'<span class="mono">{_escape(name)}</span>',
                    len(rows),
                    len({row.target for row in rows if row.target}),
                    "yes" if (kind, name) in referenced else "no",
                ]
            )
    registration_body = f"""
<nav class="nav"><a href="index.html">Boundary ops</a><a href="registrations.html">Registrations</a></nav>
<div class="eyebrow">Built-source registration inventory</div><h1>Registration names</h1>
<p class="muted">Nanobind names are leaf names because the current regex index does not reconstruct the
module hierarchy. Torch names are operator leaf names; namespace ownership is not yet retained.</p>
<section class="panel"><div class="toolbar"><input class="filter" data-filter="registration-index"
placeholder="Search registration name or kind…"></div>
{
        _table(
            ["Kind", "Registration name", "Sites", "Targets", "Python referenced"],
            registration_rows,
            "registration-index",
        )
    }</section>"""
    (output / "registrations.html").write_text(
        _document("Native registrations", registration_body), encoding="utf-8"
    )


def build_report(
    repo_root: Path,
    coverage_db: Path,
    compile_commands: Path,
    compile_commands_source_root: Path,
    output: Path,
    native_depth: int,
) -> dict[str, Any]:
    """Build the complete boundary audit and its HTML/JSON artifacts."""
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _python_parser, cpp_parser = matcher._load_parsers()
    compilation_database = _load_compilation_database(
        repo_root, compile_commands, compile_commands_source_root
    )
    boundaries = _python_boundaries(repo_root)
    definitions = matcher._index_cpp_functions(
        repo_root, cpp_parser, compilation_database.available_sources
    )
    nanobind_registrations, torch_registrations = _registration_indexes(
        repo_root, compilation_database.available_sources
    )
    caller_cases = _caller_cases(coverage_db, boundaries)
    audits = _audit_ops(
        boundaries,
        definitions,
        nanobind_registrations,
        torch_registrations,
        caller_cases,
        native_depth,
    )
    summary = _summary(
        audits,
        boundaries,
        definitions,
        nanobind_registrations,
        torch_registrations,
    )
    _render_report(
        output,
        audits,
        summary,
        nanobind_registrations,
        torch_registrations,
        source_commit,
        len(compilation_database.available_sources),
        compilation_database.missing_source_count,
    )
    payload = {
        "summary": summary,
        "provenance": {
            "source_commit": source_commit,
            "coverage_db": coverage_db.as_posix(),
            "compile_commands": compile_commands.as_posix(),
            "compile_commands_available_sources": len(compilation_database.available_sources),
            "compile_commands_missing_sources": compilation_database.missing_source_count,
            "native_depth": native_depth,
        },
        "ops": audits,
        "registration_inventory": {
            "nanobind": {
                name: [asdict(row) for row in rows]
                for name, rows in sorted(nanobind_registrations.items())
            },
            "torch_op": {
                name: [asdict(row) for row in rows]
                for name, rows in sorted(torch_registrations.items())
            },
        },
    }
    (output / "audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--coverage-db", type=Path, required=True)
    parser.add_argument("--compile-commands", type=Path, required=True)
    parser.add_argument("--compile-commands-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-depth", type=int, default=5)
    args = parser.parse_args()
    if args.native_depth < 0:
        parser.error("--native-depth must be non-negative")
    for label, path in (
        ("coverage DB", args.coverage_db),
        ("compilation database", args.compile_commands),
    ):
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    payload = build_report(
        args.repo_root.resolve(),
        args.coverage_db.resolve(),
        args.compile_commands.resolve(),
        args.compile_commands_source_root,
        args.output_dir.resolve(),
        args.native_depth,
    )
    print(json.dumps(payload["summary"], indent=2))
    print(f"Open {args.output_dir.resolve() / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
