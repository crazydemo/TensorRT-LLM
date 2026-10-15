# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explain native-codegraph mappings against a CBTS touch database."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from native_graph import NativeGraph, is_native_source
from touch_db import TouchDB


def build_native_impact_report(
    changed_files: Iterable[str], native_graph: NativeGraph, touch_db: TouchDB
) -> dict[str, Any]:
    """Return JSON-ready native mappings and their observed case union.

    This report is diagnostic. An independently unsafe Python residual may still
    make the complete coverage-tier decision decline.
    """
    file_reports: list[dict[str, Any]] = []
    all_cases: set[str] = set()
    mapped_files = 0
    production_files = 0
    native_test_files = 0
    total_mappings = 0

    for native_file in sorted({path for path in changed_files if is_native_source(path)}):
        is_native_test = native_file.startswith("cpp/tests/")
        if is_native_test:
            native_test_files += 1
        else:
            production_files += 1
        function_reports: list[dict[str, Any]] = []
        file_cases: set[str] = set()
        mappings = () if is_native_test else native_graph.mappings(native_file)
        if mappings and not is_native_test:
            mapped_files += 1
        total_mappings += len(mappings)
        for mapping in mappings:
            cases = touch_db.tests_touching_func(mapping.python_file, mapping.python_qualname)
            file_cases.update(cases)
            function_reports.append(
                {
                    "python_file": mapping.python_file,
                    "python_qualname": mapping.python_qualname,
                    "min_hops": mapping.min_hops,
                    "evidence": mapping.evidence,
                    "case_count": len(cases),
                }
            )
        all_cases.update(file_cases)
        file_reports.append(
            {
                "native_file": native_file,
                "mapping_status": "test_definition"
                if is_native_test
                else ("mapped" if mappings else "unmapped"),
                "python_functions": function_reports,
                "case_count": len(file_cases),
                "cases": sorted(file_cases),
            }
        )

    return {
        "codegraph_metadata": native_graph.metadata(),
        "summary": {
            "native_files": len(file_reports),
            "production_native_files": production_files,
            "native_test_files": native_test_files,
            "mapped_native_files": mapped_files,
            "unmapped_native_files": production_files - mapped_files,
            "python_function_mappings": total_mappings,
            "unique_cases": len(all_cases),
        },
        "native_files": file_reports,
        "cases": sorted(all_cases),
        "note": (
            "Diagnostic native impact only; the complete PR may still fall back when another "
            "residual file fails a coverage safety gate."
        ),
    }
