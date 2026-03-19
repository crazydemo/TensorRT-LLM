# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
CLI entry point for change-based test selection.

Usage:
    # Select tests based on git diff against main
    python -m tests.scripts.change_based_test_selection.cli --base-ref main

    # Select tests for specific changed files
    python -m tests.scripts.change_based_test_selection.cli --files tensorrt_llm/models/llama/model.py

    # Explain mode: show why each test was selected
    python -m tests.scripts.change_based_test_selection.cli --base-ref main --explain

    # Output filtered test list for a specific QA list
    python -m tests.scripts.change_based_test_selection.cli --base-ref main --test-list llm_function_core

    # Dump the parsed database for inspection
    python -m tests.scripts.change_based_test_selection.cli --dump-db test_database.json

    # Use cached database
    python -m tests.scripts.change_based_test_selection.cli --load-db test_database.json --files ...
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Change-based test selection for TensorRT-LLM QA CI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input source
    input_group = p.add_mutually_exclusive_group()
    input_group.add_argument(
        "--base-ref",
        default=None,
        help="Git ref to diff against (e.g. 'main', 'HEAD~3'). "
        "Uses 'git diff <base-ref>...HEAD'.",
    )
    input_group.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Explicit list of changed files (relative to repo root).",
    )

    # Repo paths
    p.add_argument(
        "--repo-root",
        default=".",
        help="Path to TensorRT-LLM repo root (default: current directory).",
    )
    p.add_argument(
        "--test-list-dir",
        default=None,
        help="Path to QA test list directory. "
        "Default: <repo-root>/tests/integration/test_lists/qa/",
    )
    p.add_argument(
        "--test-def-dir",
        default=None,
        help="Path to test definition directory. "
        "Default: <repo-root>/tests/integration/defs/",
    )

    # Output options
    p.add_argument(
        "--explain",
        action="store_true",
        help="Show detailed explanation of why each test was selected.",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Show selection statistics.",
    )
    p.add_argument(
        "--test-list",
        default=None,
        help="Filter output to only include tests from this test list "
        "(e.g. 'llm_function_core').",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write selected test IDs to this file (default: stdout).",
    )

    # Time budget
    p.add_argument(
        "--time-budget",
        default="8h",
        help="Maximum total test duration (default: 8h). Accepts formats "
        "like '8h', '480m', '28800s', or plain hours (e.g. '8' = 8h). Use '0' to disable. "
        "Tests are dropped by lowest feature value until budget is met.",
    )

    # Database caching
    p.add_argument(
        "--dump-db",
        default=None,
        help="Dump parsed test database to JSON file and exit.",
    )
    p.add_argument(
        "--load-db",
        default=None,
        help="Load test database from JSON file instead of parsing.",
    )

    # Reverse suspect analysis
    p.add_argument(
        "--suspect",
        action="store_true",
        help="Reverse suspect analysis: given a failing test and commit range, "
        "rank commits by likelihood of causing the regression.",
    )
    p.add_argument(
        "--good-ref",
        default=None,
        help="Last known good commit SHA (for --suspect mode).",
    )
    p.add_argument(
        "--bad-ref",
        default=None,
        help="First known bad commit SHA (for --suspect mode).",
    )
    p.add_argument(
        "--test-id",
        default=None,
        help="Failing test ID (for --suspect mode). "
        "Partial match supported (e.g. 'TestLlama3_1_8BInstruct/test_ngram').",
    )

    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    test_list_dir = (Path(args.test_list_dir) if args.test_list_dir else
                     repo_root / "tests" / "integration" / "test_lists" / "qa")
    test_def_dir = (Path(args.test_def_dir) if args.test_def_dir else
                    repo_root / "tests" / "integration" / "defs")

    # Build or load database
    from .parser import build_test_database, load_database, save_database

    if args.load_db:
        print(f"Loading database from {args.load_db}...", file=sys.stderr)
        database = load_database(Path(args.load_db))
        print(f"Loaded {len(database)} test entries.", file=sys.stderr)
    else:
        print(f"Parsing test lists from {test_list_dir}...", file=sys.stderr)
        print(f"Parsing test definitions from {test_def_dir}...",
              file=sys.stderr)
        database = build_test_database(test_list_dir, test_def_dir)
        print(f"Parsed {len(database)} test entries.", file=sys.stderr)

    # Dump database if requested
    if args.dump_db:
        save_database(database, Path(args.dump_db))
        print(f"Database saved to {args.dump_db}", file=sys.stderr)
        return

    # Reverse suspect analysis mode
    if args.suspect:
        if not args.good_ref or not args.bad_ref or not args.test_id:
            p.error("--suspect requires --good-ref, --bad-ref, and --test-id")

        from .selector import find_suspects, format_suspects

        print(
            f"Analyzing commits {args.good_ref[:10]}..{args.bad_ref[:10]} "
            f"for test: {args.test_id}",
            file=sys.stderr,
        )
        suspects = find_suspects(
            args.test_id, database,
            good_ref=args.good_ref, bad_ref=args.bad_ref,
            repo_root=str(repo_root),
        )
        output = format_suspects(suspects, args.test_id)
        if args.output:
            Path(args.output).write_text(output + '\n')
            print(f"Wrote suspect analysis to {args.output}", file=sys.stderr)
        else:
            print(output)
        return

    # Get changed files
    if args.files:
        changed_files = args.files
    elif args.base_ref:
        from .selector import get_changed_files
        changed_files = get_changed_files(args.base_ref, str(repo_root))
    else:
        # Default: diff against main
        from .selector import get_changed_files
        changed_files = get_changed_files("main", str(repo_root))

    if not changed_files:
        print("No changed files detected.", file=sys.stderr)
        return

    print(f"Changed files ({len(changed_files)}):", file=sys.stderr)
    for f in changed_files[:20]:
        print(f"  {f}", file=sys.stderr)
    if len(changed_files) > 20:
        print(f"  ... and {len(changed_files) - 20} more", file=sys.stderr)
    print(file=sys.stderr)

    # Parse time budget
    time_budget = 0.0
    if args.time_budget:
        budget_str = args.time_budget.strip().lower()
        if budget_str.endswith('h'):
            time_budget = float(budget_str[:-1]) * 3600
        elif budget_str.endswith('m'):
            time_budget = float(budget_str[:-1]) * 60
        elif budget_str.endswith('s'):
            time_budget = float(budget_str[:-1])
        else:
            # Plain number defaults to hours (consistent with default "8h")
            time_budget = float(budget_str) * 3600

    # Select tests
    from .selector import (
        apply_time_budget,
        compute_stats,
        format_explain,
        format_output,
        generate_maintenance_warnings,
        select_tests,
        _load_durations,
    )

    result = select_tests(database, changed_files,
                          base_ref=args.base_ref or "",
                          repo_root=str(repo_root))

    # Apply time budget if specified
    if time_budget > 0:
        durations = _load_durations(str(repo_root))
        apply_time_budget(result, time_budget, durations,
                          test_list_filter=args.test_list)

    # Output
    if args.explain:
        print(format_explain(result))
    elif args.stats:
        stats = compute_stats(result, database)
        print(json.dumps(stats, indent=2))
    else:
        output = format_output(result, test_list_filter=args.test_list)
        if args.output:
            Path(args.output).write_text(output + '\n')
            test_count = sum(
                1 for line in output.splitlines()
                if line and not line.startswith('#'))
            print(
                f"Wrote {test_count} test IDs to {args.output}",
                file=sys.stderr,
            )
        else:
            print(output)

    # Always print summary to stderr
    print(
        f"\nSelected {len(result.selected_tests)}/{len(database)} tests "
        f"({len(result.selected_tests)/len(database)*100:.1f}%)",
        file=sys.stderr,
    )
    budget_info = getattr(result, '_budget_info', None)
    if budget_info:
        remaining = budget_info['remaining_time']
        budget = budget_info['budget']
        over = remaining > budget
        status = (f"OVER by {(remaining - budget)/3600:.1f}h "
                  f"(protected tests)" if over else "within budget")
        print(
            f"Time budget: {budget/3600:.1f}h, "
            f"estimated: {remaining/3600:.1f}h ({status}), "
            f"dropped {budget_info['dropped']} tests "
            f"({budget_info['dropped_time']/3600:.1f}h saved)",
            file=sys.stderr,
        )

    # Maintenance warnings — highlight config issues that need attention
    maint_warnings = generate_maintenance_warnings(result, database)
    if maint_warnings:
        # Count top-level warnings (lines starting with [TAG])
        issue_count = sum(1 for w in maint_warnings if w.startswith('['))
        print(f"\n⚠ Maintenance warnings ({issue_count} issues):",
              file=sys.stderr)
        for w in maint_warnings:
            print(f"  {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
