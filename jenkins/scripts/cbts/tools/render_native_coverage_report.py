#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render PY_START coverage and a native codegraph as a static HTML folder.

Native hits in this report are inferred: a case runtime-hit a Python function
that the static graph maps to a C++ or CUDA file. They are not native runtime
line coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COPYRIGHT = """<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->"""

STYLES = """
:root{color-scheme:dark;--bg:#0a0e0b;--panel:#121814;--panel2:#172019;--line:#2a362d;
--text:#edf5ef;--muted:#91a298;--green:#76b900;--blue:#64b5f6;--amber:#f3bd4b;--red:#ff7b72}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#17351f 0,
transparent 34rem),var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,
-apple-system,sans-serif}.page{max-width:1500px;margin:auto;padding:38px 28px 70px}a{color:#a5d866;
text-decoration:none}a:hover{text-decoration:underline}h1{font-size:29px;letter-spacing:-.03em;margin:3px 0 8px}
h2{font-size:18px;margin:0 0 14px}.mono,code{font-family:"SFMono-Regular",Consolas,monospace}
.muted{color:var(--muted)}.eyebrow{color:var(--green);font-size:11px;font-weight:800;letter-spacing:.13em;
text-transform:uppercase}.nav{display:flex;gap:16px;margin:0 0 25px}.metrics{display:grid;
grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:11px;margin:25px 0}.metric,.panel{background:#121814ee;
border:1px solid var(--line);border-radius:13px;box-shadow:0 16px 55px #0004}.metric{padding:15px}.metric strong{
display:block;font-size:25px}.metric span{color:var(--muted);font-size:11px}.panel{padding:20px;margin:16px 0}
.notice{border-left:3px solid var(--amber);background:#251d0d;color:#eadcb9;padding:11px 14px;border-radius:5px;
margin:18px 0}.toolbar{display:flex;gap:10px;align-items:center;margin:12px 0}.filter{width:min(650px,100%);
background:#0d120f;border:1px solid #354139;color:var(--text);border-radius:8px;padding:10px 12px}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:9px}table{border-collapse:collapse;width:100%;
min-width:920px}th{position:sticky;top:0;background:#1b251e;color:#b9c6bd;text-align:left;font-size:10px;
text-transform:uppercase;letter-spacing:.06em}th,td{padding:9px 11px;border-bottom:1px solid var(--line);
vertical-align:top}
tbody tr:hover{background:#18211b}td{font-size:12px}.badge{display:inline-block;padding:2px 7px;border-radius:999px;
border:1px solid #3b493f;font-size:10px;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap}
.badge.hit,.badge.passed,.badge.cuda{border-color:#4c7416;color:#b7e67a;background:#18250f}.badge.none{
border-color:#765a1b;color:#f4c964;background:#2a210d}.badge.no-data,.badge.failed{border-color:#7d3535;
color:#ff9b94;background:#2b1111}.case-name{overflow-wrap:anywhere}.footer{text-align:center;color:#66746a;
font-size:11px;margin-top:30px}details summary{cursor:pointer;font-weight:700}@media(max-width:700px){.page{
padding:24px 13px}h1{font-size:23px}.panel{padding:14px}}
"""

SCRIPT = """
document.querySelectorAll('[data-filter]').forEach(input => {
  input.addEventListener('input', () => {
    const needle = input.value.toLowerCase().trim();
    document.querySelectorAll(`#${input.dataset.filter} tbody tr`).forEach(row => {
      row.hidden = needle && !row.dataset.search.includes(needle);
    });
  });
});
"""

REPORT_MARKER = ".cbts-native-coverage-report"


@dataclass(frozen=True)
class CaseRecord:
    """One logical stage-prefixed case and its capture metadata."""

    test: str
    stage: str
    outcome: str
    expected_workers: int
    saved_procs: int


@dataclass(frozen=True)
class NativeHit:
    """A runtime Python hit joined to one static native-file mapping."""

    native_file: str
    python_file: str
    python_qualname: str
    min_hops: int
    evidence: str


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _page_name(value: str) -> str:
    """Return a stable, filesystem-safe HTML filename."""
    digest = hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{digest}.html"


def _badge(label: str, css_class: str | None = None) -> str:
    kind = css_class or label.lower().replace("_", "-")
    return f'<span class="badge {_escape(kind)}">{_escape(label)}</span>'


def _document(title: str, body: str, *, depth: int = 0) -> str:
    prefix = "../" * depth
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape(title)}</title><link rel="stylesheet" href="{prefix}assets/style.css"></head>
<body>{COPYRIGHT}<main class="page">{body}<div class="footer">CBTS native coverage explorer · inferred native hits</div>
</main><script src="{prefix}assets/report.js"></script></body></html>"""


def _source_link(path: str, *, depth: int) -> str:
    return f'<a class="mono" href="{"../" * (depth + 1)}{_escape(path)}">{_escape(path)}</a>'


def _metrics(items: list[tuple[int | str, str]]) -> str:
    return (
        '<div class="metrics">'
        + "".join(
            f'<div class="metric"><strong>{_escape(value)}</strong><span>{_escape(label)}</span></div>'
            for value, label in items
        )
        + "</div>"
    )


def _load_data(
    coverage_db: Path, native_codegraph: Path
) -> tuple[
    list[CaseRecord],
    dict[str, tuple[int, int]],
    dict[str, list[NativeHit]],
    dict[str, str],
    dict[str, str],
]:
    uri = f"file:{coverage_db.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute(
            "ATTACH DATABASE ? AS codegraph", (native_codegraph.resolve().as_posix(),)
        )
        cases = [
            CaseRecord(*row)
            for row in connection.execute(
                "SELECT test, stage, COALESCE(outcome, ''), expected_workers, saved_procs "
                "FROM test_case_meta WHERE test!='' ORDER BY test"
            )
        ]
        python_counts = {
            row[0]: (row[1], row[2])
            for row in connection.execute(
                "SELECT test, COUNT(DISTINCT file), "
                "COUNT(DISTINCT file || char(0) || qualname) "
                "FROM touch_rows WHERE test!='' GROUP BY test"
            )
        }
        hits_by_case: dict[str, list[NativeHit]] = defaultdict(list)
        for row in connection.execute(
            "SELECT DISTINCT t.test, p.native_file, t.file, t.qualname, "
            "p.min_hops, p.evidence FROM touch_rows AS t "
            "JOIN codegraph.python_native_file AS p "
            "ON p.python_file=t.file AND p.python_qualname=t.qualname "
            "WHERE t.test!='' ORDER BY t.test, p.native_file, t.file, t.qualname"
        ):
            hits_by_case[row[0]].append(NativeHit(*row[1:]))
        coverage_metadata = dict(connection.execute("SELECT key, value FROM meta"))
        codegraph_metadata = dict(connection.execute("SELECT key, value FROM codegraph.metadata"))
    return cases, python_counts, dict(hits_by_case), coverage_metadata, codegraph_metadata


def _case_status(python_symbols: int, hits: list[NativeHit]) -> tuple[str, str]:
    if hits:
        return "inferred hit", "hit"
    if python_symbols:
        return "no native mapping", "none"
    return "no Python data", "no-data"


def _render_index(
    output: Path,
    cases: list[CaseRecord],
    python_counts: dict[str, tuple[int, int]],
    hits_by_case: dict[str, list[NativeHit]],
    coverage_metadata: dict[str, str],
    codegraph_metadata: dict[str, str],
) -> None:
    mapped_cases = sum(bool(hits_by_case.get(case.test)) for case in cases)
    no_python_data = sum(python_counts.get(case.test, (0, 0))[1] == 0 for case in cases)
    native_files = {hit.native_file for hits in hits_by_case.values() for hit in hits}
    rows = []
    for case in cases:
        python_files, python_symbols = python_counts.get(case.test, (0, 0))
        hits = hits_by_case.get(case.test, [])
        case_native_files = {hit.native_file for hit in hits}
        cuda_files = {path for path in case_native_files if path.endswith((".cu", ".cuh"))}
        status, css_class = _case_status(python_symbols, hits)
        search = " ".join((case.test, case.stage, case.outcome, status)).lower()
        rows.append(
            f'<tr data-search="{_escape(search)}"><td>{_badge(status, css_class)}</td>'
            f'<td class="mono case-name"><a href="cases/{_page_name(case.test)}">{_escape(case.test)}</a></td>'
            f"<td>{_escape(case.outcome)}</td><td>{python_files}</td><td>{python_symbols}</td>"
            f"<td>{len(case_native_files)}</td><td>{len(cuda_files)}</td></tr>"
        )
    body = f"""
<nav class="nav"><a href="index.html">Cases</a><a href="native.html">Native files</a></nav>
<div class="eyebrow">CBTS · Python runtime coverage + static native codegraph</div>
<h1>Per-case inferred C++ / CUDA coverage</h1>
<p class="muted">Coverage DB build data joined with codegraph source commit
<code>{_escape(codegraph_metadata.get("source_commit", "?"))}</code>.</p>
<div class="notice"><strong>Important:</strong> “inferred hit” means the case runtime-hit a Python function that
the static graph can reach from a native file. It is not Bullseye/LLVM native runtime coverage and does not prove
that a particular C++ branch or line executed.</div>
{
        _metrics(
            [
                (len(cases), "captured case records"),
                (mapped_cases, "cases with inferred native hits"),
                (len(cases) - mapped_cases - no_python_data, "Python data, no native mapping"),
                (no_python_data, "cases with no Python rows"),
                (len(native_files), "inferred native files"),
                (
                    coverage_metadata.get("func_rate_pct", "?") + "%",
                    "Python function coverage rate",
                ),
            ]
        )
    }
<section class="panel"><h2>Cases</h2><div class="toolbar"><input class="filter" data-filter="cases"
placeholder="Search case, stage, outcome, or mapping status…"></div><div class="table-wrap"><table id="cases">
<thead><tr><th>Status</th><th>Case</th><th>Outcome</th><th>Python files</th><th>Python symbols</th>
<th>Native files</th><th>CUDA files</th></tr></thead><tbody>{
        "".join(rows)
    }</tbody></table></div></section>"""
    (output / "index.html").write_text(
        _document("CBTS per-case native coverage", body), encoding="utf-8"
    )


def _render_case_pages(
    output: Path,
    cases: list[CaseRecord],
    python_counts: dict[str, tuple[int, int]],
    hits_by_case: dict[str, list[NativeHit]],
) -> None:
    case_dir = output / "cases"
    for case in cases:
        python_files, python_symbols = python_counts.get(case.test, (0, 0))
        hits = hits_by_case.get(case.test, [])
        native_files = {hit.native_file for hit in hits}
        cuda_files = {path for path in native_files if path.endswith((".cu", ".cuh"))}
        rows = []
        for hit in hits:
            native_kind = "CUDA" if hit.native_file.endswith((".cu", ".cuh")) else "C++"
            search = " ".join(
                (hit.native_file, hit.python_file, hit.python_qualname, hit.evidence, native_kind)
            ).lower()
            kind_badge = _badge(native_kind, "cuda" if native_kind == "CUDA" else "hit")
            python_entry = (
                f'{_source_link(hit.python_file, depth=1)}::<span class="mono">'
                f"{_escape(hit.python_qualname)}</span>"
            )
            rows.append(
                f'<tr data-search="{_escape(search)}"><td>{kind_badge}</td>'
                f"<td>{_source_link(hit.native_file, depth=1)}</td>"
                f"<td>{python_entry}</td>"
                f"<td>{hit.min_hops}</td><td>{_escape(hit.evidence)}</td></tr>"
            )
        status, css_class = _case_status(python_symbols, hits)
        table = (
            '<div class="table-wrap"><table id="hits"><thead><tr><th>Kind</th><th>Native file</th>'
            "<th>Runtime-hit Python entry</th><th>Min hops</th><th>Evidence</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            if rows
            else '<p class="muted">No inferred native hit is available for this case.</p>'
        )
        python_hits_link = (
            f'<a href="../python/{_page_name(case.test)}">Browse all {python_symbols} '
            "runtime-hit Python symbols →</a>"
            if python_symbols
            else '<span class="muted">This case has no Python runtime rows.</span>'
        )
        body = f"""
<nav class="nav"><a href="../index.html">← Cases</a><a href="../native.html">Native files</a></nav>
<div class="eyebrow">Case detail · {_badge(status, css_class)}</div><h1 class="mono case-name">{
            _escape(case.test)
        }</h1>
<p class="muted">Stage: <code>{_escape(case.stage)}</code> · outcome: {
            _escape(case.outcome or "unknown")
        } ·
workers expected/saved: {case.expected_workers}/{case.saved_procs}</p>
{
            _metrics(
                [
                    (python_files, "runtime-hit Python files"),
                    (python_symbols, "runtime-hit Python symbols"),
                    (len(native_files), "inferred native files"),
                    (len(cuda_files), "inferred CUDA files"),
                    (len(hits), "Python-to-native mapping rows"),
                ]
            )
        }
<div class="notice">Each row joins a real PY_START hit for this case with static codegraph evidence.
It is a conservative dependency candidate, not native runtime line coverage.</div>
<section class="panel"><h2>Python runtime hits</h2><p>{python_hits_link}</p>
<p class="muted">PY_START records a hit set, not caller→callee edges or execution order, so this
cannot reconstruct an ordered runtime Python call chain.</p></section>
<section class="panel"><h2>Inferred native hits</h2><div class="toolbar"><input class="filter"
data-filter="hits" placeholder="Search native file, Python entry, or evidence…"></div>{
            table
        }</section>"""
        (case_dir / _page_name(case.test)).write_text(
            _document(case.test, body, depth=1), encoding="utf-8"
        )


def _render_python_page(output: Path, case: CaseRecord, rows: list[tuple[str, str]]) -> None:
    table_rows = []
    for python_file, python_qualname in rows:
        search = f"{python_file} {python_qualname}".lower()
        table_rows.append(
            f'<tr data-search="{_escape(search)}"><td>{_badge("runtime hit", "hit")}</td>'
            f"<td>{_source_link(python_file, depth=1)}</td>"
            f'<td class="mono">{_escape(python_qualname)}</td></tr>'
        )
    body = f"""
<nav class="nav"><a href="../index.html">Cases</a>
<a href="../cases/{
        _page_name(case.test)
    }">← Case detail</a><a href="../native.html">Native files</a></nav>
<div class="eyebrow">PY_START · runtime hit set</div><h1 class="mono case-name">{
        _escape(case.test)
    }</h1>
{
        _metrics(
            [
                (len({file for file, _qualname in rows}), "runtime-hit Python files"),
                (len(rows), "runtime-hit Python symbols"),
            ]
        )
    }
<div class="notice">These functions were observed by PY_START for the case. The DB does not retain
caller→callee edges, timestamps, or execution order; this is a hit set rather than an ordered call chain.</div>
<section class="panel"><h2>Python functions</h2><div class="toolbar"><input class="filter"
data-filter="python-hits" placeholder="Search Python file or qualname…"></div><div class="table-wrap">
<table id="python-hits"><thead><tr><th>Status</th><th>Python file</th><th>Qualname</th></tr></thead>
<tbody>{"".join(table_rows)}</tbody></table></div></section>"""
    (output / "python" / _page_name(case.test)).write_text(
        _document(f"Python hits · {case.test}", body, depth=1), encoding="utf-8"
    )


def _render_python_pages(coverage_db: Path, output: Path, cases: list[CaseRecord]) -> int:
    """Stream all per-case Python hit sets into pages without retaining the full DB in memory."""
    cases_by_test = {case.test: case for case in cases}
    uri = f"file:{coverage_db.resolve().as_posix()}?mode=ro"
    row_count = 0
    current_test = ""
    current_rows: list[tuple[str, str]] = []
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT test, file, qualname FROM touch_rows WHERE test!='' "
            "ORDER BY test, file, qualname"
        )
        for test, python_file, python_qualname in rows:
            if current_test and test != current_test:
                _render_python_page(output, cases_by_test[current_test], current_rows)
                current_rows = []
            current_test = test
            current_rows.append((python_file, python_qualname))
            row_count += 1
    if current_test:
        _render_python_page(output, cases_by_test[current_test], current_rows)
    return row_count


def _render_native_pages(output: Path, hits_by_case: dict[str, list[NativeHit]]) -> None:
    by_native: dict[str, list[tuple[str, NativeHit]]] = defaultdict(list)
    for test, hits in hits_by_case.items():
        for hit in hits:
            by_native[hit.native_file].append((test, hit))

    native_dir = output / "native"
    index_rows = []
    for native_file in sorted(by_native):
        rows = by_native[native_file]
        cases = {test for test, _hit in rows}
        entries = {(hit.python_file, hit.python_qualname) for _test, hit in rows}
        kind = "CUDA" if native_file.endswith((".cu", ".cuh")) else "C++"
        kind_badge = _badge(kind, "cuda" if kind == "CUDA" else "hit")
        index_rows.append(
            f'<tr data-search="{_escape((native_file + " " + kind).lower())}">'
            f"<td>{kind_badge}</td>"
            f'<td class="mono"><a href="native/{_page_name(native_file)}">{_escape(native_file)}</a></td>'
            f"<td>{len(cases)}</td><td>{len(entries)}</td></tr>"
        )
        detail_rows = []
        for test, hit in rows:
            search = " ".join((test, hit.python_file, hit.python_qualname, hit.evidence)).lower()
            case_link = f'<a href="../cases/{_page_name(test)}">{_escape(test)}</a>'
            python_entry = (
                f'{_source_link(hit.python_file, depth=1)}::<span class="mono">'
                f"{_escape(hit.python_qualname)}</span>"
            )
            detail_rows.append(
                f'<tr data-search="{_escape(search)}"><td class="mono case-name">'
                f"{case_link}</td><td>{python_entry}</td>"
                f"<td>{hit.min_hops}</td><td>{_escape(hit.evidence)}</td></tr>"
            )
        detail_body = f"""
<nav class="nav"><a href="../index.html">Cases</a><a href="../native.html">← Native files</a></nav>
<div class="eyebrow">Inferred native file · {_badge(kind, "cuda" if kind == "CUDA" else "hit")}</div>
<h1 class="mono case-name">{_escape(native_file)}</h1>
{_metrics([(len(cases), "cases"), (len(entries), "runtime-hit Python entries"), (len(rows), "mapping rows")])}
<div class="notice">Cases below runtime-hit a Python entry statically mapped to this file. This does not prove
the native file, function, branch, or line executed at runtime.</div>
<section class="panel"><h2>Cases and Python entries</h2><div class="toolbar"><input class="filter"
data-filter="native-hits" placeholder="Search case, Python entry, or evidence…"></div><div class="table-wrap">
<table id="native-hits"><thead><tr><th>Case</th><th>Runtime-hit Python entry</th><th>Min hops</th>
<th>Evidence</th></tr></thead><tbody>{"".join(detail_rows)}</tbody></table></div></section>"""
        (native_dir / _page_name(native_file)).write_text(
            _document(native_file, detail_body, depth=1), encoding="utf-8"
        )

    body = f"""
<nav class="nav"><a href="index.html">Cases</a><a href="native.html">Native files</a></nav>
<div class="eyebrow">CBTS · reverse native index</div><h1>Inferred C++ / CUDA files</h1>
<p class="muted">Browse from a native file back to cases whose runtime Python hits reach it in the static graph.</p>
{
        _metrics(
            [
                (len(by_native), "native files"),
                (sum(path.endswith((".cu", ".cuh")) for path in by_native), "CUDA files"),
                (len(hits_by_case), "cases with inferred hits"),
            ]
        )
    }
<section class="panel"><div class="toolbar"><input class="filter" data-filter="native-index"
placeholder="Search native path or kind…"></div><div class="table-wrap"><table id="native-index"><thead>
<tr><th>Kind</th><th>Native file</th><th>Cases</th><th>Python entries</th></tr></thead>
<tbody>{"".join(index_rows)}</tbody></table></div></section>"""
    (output / "native.html").write_text(_document("CBTS native file index", body), encoding="utf-8")


def render(coverage_db: Path, native_codegraph: Path, output: Path) -> dict[str, int]:
    """Build the portable HTML folder and return its summary counters."""
    cases, python_counts, hits_by_case, coverage_metadata, codegraph_metadata = _load_data(
        coverage_db, native_codegraph
    )
    if output.exists() and any(output.iterdir()) and not (output / REPORT_MARKER).is_file():
        raise ValueError(f"refusing to replace non-report directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for subdirectory in ("assets", "cases", "native", "python"):
        path = output / subdirectory
        if path.exists():
            for old_file in path.iterdir():
                old_file.unlink()
        else:
            path.mkdir()
    (output / REPORT_MARKER).write_text("generated by render_native_coverage_report.py\n")
    (output / "assets/style.css").write_text(STYLES, encoding="utf-8")
    (output / "assets/report.js").write_text(SCRIPT, encoding="utf-8")
    _render_index(output, cases, python_counts, hits_by_case, coverage_metadata, codegraph_metadata)
    _render_case_pages(output, cases, python_counts, hits_by_case)
    python_hit_rows = _render_python_pages(coverage_db, output, cases)
    _render_native_pages(output, hits_by_case)
    summary = {
        "cases": len(cases),
        "cases_with_python_data": sum(
            python_counts.get(case.test, (0, 0))[1] > 0 for case in cases
        ),
        "cases_with_inferred_native_hits": len(hits_by_case),
        "inferred_native_files": len(
            {hit.native_file for hits in hits_by_case.values() for hit in hits}
        ),
        "mapping_rows": sum(len(hits) for hits in hits_by_case.values()),
        "python_hit_rows": python_hit_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-db", type=Path, required=True)
    parser.add_argument("--native-codegraph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.coverage_db.is_file():
        parser.error(f"coverage DB does not exist: {args.coverage_db}")
    if not args.native_codegraph.is_file():
        parser.error(f"native codegraph does not exist: {args.native_codegraph}")
    summary = render(
        args.coverage_db.resolve(), args.native_codegraph.resolve(), args.output_dir.resolve()
    )
    print(json.dumps(summary, indent=2))
    print(f"Open {args.output_dir.resolve() / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
