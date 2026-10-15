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
"""Render a case-to-native dependency SQLite database as self-contained HTML."""

from __future__ import annotations

import argparse
import html
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

COPYRIGHT = """<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->"""


def _query(connection: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql)
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _source(file: str, line: int | None = None) -> str:
    label = f"{file}:{line}" if line else file
    target = f"../{file}"
    return f'<a class="source" href="{_escape(target)}">{_escape(label)}</a>'


def _badge(value: str) -> str:
    css_class = value.lower().replace("_", "-")
    return f'<span class="badge {css_class}">{_escape(value)}</span>'


def _table(
    table_id: str,
    columns: list[tuple[str, str]],
    rows: Iterable[dict[str, Any]],
    source_column: tuple[str, str] | None = None,
) -> str:
    header = "".join(f"<th>{_escape(label)}</th>" for _key, label in columns)
    body: list[str] = []
    for row in rows:
        cells: list[str] = []
        for key, _label in columns:
            value = row.get(key)
            if source_column and key == source_column[0]:
                rendered = _source(str(value), int(row[source_column[1]]))
            elif key in {"confidence", "kind", "outcome"}:
                rendered = _badge(str(value))
            else:
                rendered = _escape(value)
            cells.append(f"<td>{rendered}</td>")
        search_text = " ".join(str(value or "") for value in row.values()).lower()
        body.append(f'<tr data-search="{_escape(search_text)}">{"".join(cells)}</tr>')
    return (
        f'<div class="table-wrap"><table id="{_escape(table_id)}">'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _path_score(path: list[dict[str, Any]], phase: str) -> float:
    score = len(path) * 0.25
    for edge in path:
        score += 4 if edge["confidence"] == "high" else 2
        callee_symbol = str(edge["callee_symbol"])
        callee_file = str(edge["callee_file"])
        if callee_file.endswith(".cu"):
            score += 6
        if callee_symbol.startswith("invoke"):
            score += 3
        if phase == "context":
            score += 4 if "context" in callee_symbol.lower() else 0
            score -= 5 if "generation" in callee_symbol.lower() else 0
        elif phase == "generation":
            score += 4 if "generation" in callee_symbol.lower() else 0
            score -= 5 if "context" in callee_symbol.lower() else 0
    return score


def _representative_path(
    implementations: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    phase: str,
) -> list[dict[str, Any]]:
    starts = {
        (row["file"], row["symbol"])
        for row in implementations
        if row["kind"] == "native_definition"
    }
    adjacency: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        adjacency[(edge["caller_file"], edge["caller_symbol"])].append(edge)

    paths: list[list[dict[str, Any]]] = []

    def visit(
        node: tuple[str, str],
        path: list[dict[str, Any]],
        seen: set[tuple[str, str]],
    ) -> None:
        outgoing = adjacency.get(node, [])
        if not outgoing:
            paths.append(path)
            return
        extended = False
        for edge in outgoing:
            callee = (edge["callee_file"], edge["callee_symbol"])
            if callee in seen:
                continue
            extended = True
            visit(callee, [*path, edge], {*seen, callee})
        if not extended:
            paths.append(path)

    for start in starts:
        visit(start, [], {start})
    return max(paths, key=lambda path: _path_score(path, phase), default=[])


def _chain_html(
    boundary: dict[str, Any],
    implementations: list[dict[str, Any]],
    path: list[dict[str, Any]],
) -> str:
    cards = [
        (
            "Python boundary",
            boundary["canonical_name"],
            boundary["python_file"],
            boundary["line"],
            boundary["confidence"],
        )
    ]
    for implementation in implementations:
        cards.append(
            (
                implementation["kind"].replace("_", " "),
                implementation["symbol"],
                implementation["file"],
                implementation["line"],
                implementation["confidence"],
            )
        )
    seen = {(item[2], item[3], item[1]) for item in cards}
    for edge in path:
        key = (edge["callee_file"], edge["callee_line"], edge["callee_symbol"])
        if key in seen:
            continue
        seen.add(key)
        cards.append(
            (
                f"native hop {edge['hop']}",
                edge["callee_symbol"],
                edge["callee_file"],
                edge["callee_line"],
                edge["confidence"],
            )
        )
    rendered = []
    for index, (kind, symbol, file, line, confidence) in enumerate(cards):
        if index:
            rendered.append('<div class="arrow">↓</div>')
        rendered.append(
            '<div class="chain-card">'
            f'<div class="chain-kind">{_escape(kind)} {_badge(str(confidence))}</div>'
            f'<div class="chain-symbol">{_escape(symbol)}</div>'
            f'<div class="chain-file">{_source(str(file), int(line))}</div>'
            "</div>"
        )
    return "".join(rendered)


def render(database: Path, output: Path) -> None:
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        case = _query(connection, "SELECT * FROM selected_case")[0]
        metadata = {
            row["key"]: row["value"]
            for row in _query(connection, "SELECT key, value FROM metadata")
        }
        boundaries = _query(
            connection,
            "SELECT * FROM boundary_summary ORDER BY confidence, canonical_name, line",
        )
        implementations = _query(
            connection,
            """
            SELECT implementation.*, boundary.canonical_name
            FROM implementation
            JOIN boundary ON boundary.id = implementation.boundary_id
            ORDER BY boundary_id, implementation.id
            """,
        )
        edges = _query(
            connection,
            """
            SELECT native_edge.*, boundary.canonical_name
            FROM native_edge
            JOIN boundary ON boundary.id = native_edge.boundary_id
            ORDER BY boundary_id, hop, caller_file, caller_line
            """,
        )

    primary = max(
        boundaries,
        key=lambda row: (row["confidence"] == "high", row["native_edges"]),
    )
    primary_implementations = [
        row for row in implementations if row["boundary_id"] == primary["id"]
    ]
    primary_edges = [row for row in edges if row["boundary_id"] == primary["id"]]
    path = _representative_path(
        primary_implementations, primary_edges, metadata.get("phase_hint", "any")
    )
    native_files = {row["callee_file"] for row in primary_edges}
    cuda_edges = sum(str(row["callee_file"]).endswith(".cu") for row in primary_edges)
    high_boundaries = sum(row["confidence"] == "high" for row in boundaries)

    boundary_table = _table(
        "boundaries",
        [
            ("confidence", "Confidence"),
            ("kind", "Kind"),
            ("canonical_name", "Boundary"),
            ("python_file", "Python source"),
            ("conditional_depth", "Control depth"),
            ("implementations", "Implementations"),
            ("native_edges", "Native edges"),
        ],
        boundaries,
        ("python_file", "line"),
    )
    implementation_table = _table(
        "implementations",
        [
            ("confidence", "Confidence"),
            ("kind", "Kind"),
            ("canonical_name", "Boundary"),
            ("file", "Source"),
            ("symbol", "Symbol"),
            ("evidence", "Evidence"),
        ],
        implementations,
        ("file", "line"),
    )
    edge_table = _table(
        "edges",
        [
            ("hop", "Hop"),
            ("confidence", "Confidence"),
            ("caller_file", "Caller"),
            ("caller_symbol", "Caller symbol"),
            ("callee_file", "Callee"),
            ("callee_symbol", "Callee symbol"),
            ("evidence", "Resolution evidence"),
        ],
        primary_edges,
        ("caller_file", "caller_line"),
    )

    styles = """
:root{color-scheme:dark;--bg:#0b0f0d;--panel:#121814;--panel2:#18201b;--text:#edf4ef;
--muted:#91a097;--line:#2a352e;--green:#76b900;--amber:#f0b429;--blue:#55a7ff}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 80% 0,#17301e 0,transparent 32rem),var(--bg);
color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}
.page{max-width:1500px;margin:auto;padding:42px 28px 80px}
h1{font-size:30px;margin:0 0 8px;letter-spacing:-.03em}h2{font-size:19px;margin:0 0 16px}
p{color:var(--muted)}code,.chain-symbol,.source{font-family:"SFMono-Regular",Consolas,monospace}
.eyebrow{color:var(--green);font-weight:700;text-transform:uppercase;letter-spacing:.12em;font-size:12px}
.case{font-family:"SFMono-Regular",Consolas,monospace;color:#c8d3cb;overflow-wrap:anywhere}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px;margin:28px 0}
.metric,.section{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);
border-radius:14px;box-shadow:0 18px 60px #0004}
.metric{padding:16px}.metric strong{font-size:26px;display:block}.metric span{color:var(--muted);font-size:12px}
.section{padding:22px;margin:18px 0}
.notice{border-left:3px solid var(--amber);padding:10px 14px;background:#241d0d;color:#e7d9b5;border-radius:4px}
.chain{display:flex;flex-direction:column;align-items:stretch;max-width:980px;margin:auto}
.chain-card{background:var(--panel2);border:1px solid #344239;border-radius:10px;padding:13px 16px}
.chain-kind{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.07em}
.chain-symbol{font-size:15px;font-weight:700;margin:5px 0;overflow-wrap:anywhere}.chain-file{font-size:12px}
.arrow{text-align:center;color:var(--green);font-size:20px;height:28px}.source{color:#9dd25a;text-decoration:none}
.source:hover{text-decoration:underline}
.badge{display:inline-block;padding:2px 7px;border-radius:999px;border:1px solid #3b493f;color:#c5d0c8;
font-size:10px;text-transform:uppercase;letter-spacing:.04em}
.badge.high,.badge.passed{border-color:#4c7416;color:#a9dc64;background:#18250f}
.badge.medium{border-color:#315c80;color:#83c3ff;background:#102233}
.badge.candidate{border-color:#765a1b;color:#f4c964;background:#2a210d}
.toolbar{display:flex;gap:10px;align-items:center;margin-bottom:12px}
.filter{width:min(560px,100%);background:#0d120f;border:1px solid #354139;color:var(--text);
border-radius:8px;padding:10px 12px}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:9px}
table{border-collapse:collapse;width:100%;min-width:900px}
th{position:sticky;top:0;background:#1b251e;color:#b9c6bd;text-align:left;font-size:11px;
text-transform:uppercase;letter-spacing:.05em}
th,td{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:hover{background:#18211b}td{font-size:12px}
details summary{cursor:pointer;font-weight:700;font-size:16px;margin:-2px 0 14px}
.footer{color:#67736b;text-align:center;margin-top:32px;font-size:12px}
@media(max-width:700px){.page{padding:24px 14px}.section{padding:15px}h1{font-size:24px}}
"""
    script = """
document.querySelectorAll('[data-filter]').forEach(input => {
  input.addEventListener('input', () => {
    const needle = input.value.toLowerCase().trim();
    const table = document.getElementById(input.dataset.filter);
    table.querySelectorAll('tbody tr').forEach(row => {
      row.hidden = needle && !row.dataset.search.includes(needle);
    });
  });
});
"""
    metrics = [
        (case["touched_files"], "runtime-touched Python files"),
        (case["touched_symbols"], "runtime-touched symbols"),
        (len(boundaries), f"boundary callsites ({high_boundaries} high)"),
        (len(primary_edges), "native candidate edges"),
        (len(native_files), "native candidate files"),
        (cuda_edges, "edges entering .cu files"),
    ]
    metrics_html = "".join(
        f'<div class="metric"><strong>{_escape(value)}</strong><span>{_escape(label)}</span></div>'
        for value, label in metrics
    )
    phase = _escape(metadata.get("phase_hint", "any"))
    chain = _chain_html(primary, primary_implementations, path)
    parse_ms = _escape(metadata.get("cpp_parse_ms", "?"))
    created_at = _escape(metadata.get("created_at", ""))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CBTS native dependency report</title><style>{styles}</style></head><body>{COPYRIGHT}
<main class="page"><div class="eyebrow">CBTS · case-to-native exploration</div>
<h1>Accuracy case native dependency report</h1>
<div class="case">{_escape(case["test"])}</div>
<div class="metrics">{metrics_html}</div>
<section class="section"><h2>Representative top-down path</h2>
<p>Ranked for the <strong>{phase}</strong> phase. This is one useful static path,
not proof that every branch ran.</p><div class="chain">{chain}</div></section>
<section class="section"><div class="notice"><strong>How to read confidence:</strong>
boundary confidence estimates lexical execution likelihood. Native-edge confidence describes
symbol-resolution strength only; it does not establish runtime execution.</div></section>
<section class="section"><h2>Detected language boundaries</h2><div class="toolbar">
<input class="filter" data-filter="boundaries" placeholder="Filter by op, Python file, confidence…">
</div>{boundary_table}</section>
<section class="section"><details open>
<summary>Registrations and implementations ({len(implementations)})</summary><div class="toolbar">
<input class="filter" data-filter="implementations" placeholder="Filter registrations and definitions…">
</div>{implementation_table}</details></section>
<section class="section"><details><summary>Native edges for primary boundary ({len(primary_edges)})</summary>
<div class="toolbar"><input class="filter" data-filter="edges" placeholder="Filter by .cu, symbol, evidence…">
</div>{edge_table}</details></section>
<div class="footer">Generated from {_escape(database.name)} · Tree-sitter C++ parse {parse_ms} ms
· {created_at}</div>
</main><script>{script}</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error(f"database does not exist: {args.database}")
    render(args.database, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
