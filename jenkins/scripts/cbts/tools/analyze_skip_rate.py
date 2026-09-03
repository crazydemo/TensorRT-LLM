#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Measure CBTS case skip rates for single commits and rolling commit windows.

This is a controlled replay against one caller-supplied coverage touch DB and
one repository snapshot. It evaluates the proposed gap policy: only
``tensorrt_llm/**/*.py`` and native source/header files under ``cpp/`` are
relevant. A native change falls back because the current touch DB maps Python
execution only; a window with no relevant change skips the entire test-case
universe; a Python-only window is evaluated by the coverage tier.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

CBTS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ROOT = CBTS_DIR.parents[2]
sys.path.insert(0, str(CBTS_DIR))

from blocks import (  # noqa: E402
    YAMLIndex,
    block_matches_stage,
    load_durations,
    parse_stages_from_groovy,
)
from coverage_tier import (  # noqa: E402
    DEFAULT_NO_DATA_POLICY,
    NO_DATA_POLICIES,
    apply_coverage_tier,
    compute_coverage_stage_counts,
    open_db,
)
from rules._helpers import strip_noop_diff_lines  # noqa: E402
from rules.base import PRInputs  # noqa: E402

DEFAULT_GAPS = (1, 2, 3, 5, 10, 15, 20)
NATIVE_SUFFIXES = frozenset(
    {".c", ".cc", ".cpp", ".cuh", ".cu", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
)
_MULTI_GPU_RE = re.compile(r"\d+_GPUs")
_STAGE_SHARD_SUFFIX_RE = re.compile(r"-\d+$")


@dataclass(frozen=True)
class Commit:
    sha: str
    timestamp: str
    subject: str


@dataclass(frozen=True)
class WindowResult:
    gap: int
    start_position: int
    end_position: int
    base: str
    head: str
    head_timestamp: str
    head_subject: str
    changed_files: int
    relevant_python_files: int
    native_files: int
    relevant_python_paths: tuple[str, ...]
    native_paths: tuple[str, ...]
    status: str
    reason_category: str
    reason: str
    selected_cases: int
    total_cases: int
    skip_rate: float
    impacted_tests: int = 0
    removed_cases: int = 0


class CaseUniverse:
    """Precomputed pre-merge case counts, de-duplicated across stage shards."""

    def __init__(self, yaml_index: YAMLIndex, stages: dict) -> None:
        by_stem: dict[str, list] = {}
        for block in yaml_index.blocks:
            by_stem.setdefault(block.yaml_stem, []).append(block)

        scheduled_groups: dict[str, dict[str, object]] = {}
        for name, stage in stages.items():
            if self._is_scheduled(name):
                group = _STAGE_SHARD_SUFFIX_RE.sub("", name)
                scheduled_groups.setdefault(group, {})[name] = stage

        full_counts: dict[str, int] = {}
        for group, members in scheduled_groups.items():
            stage = next(iter(members.values()))
            full_counts[group] = sum(
                len(block.tests)
                for block in by_stem.get(stage.yaml_stem, [])
                if block_matches_stage(block, stage)
            )

        self._scheduled_groups = scheduled_groups
        self._full_counts = full_counts
        self.total_cases = sum(full_counts.values())

    @staticmethod
    def _is_scheduled(name: str) -> bool:
        return (
            "-OnDemand-" not in name and "Post-Merge" not in name and not _MULTI_GPU_RE.search(name)
        )

    def selected_cases(
        self,
        affected_stages: set[str],
        kept_per_stage: dict[str, int],
    ) -> int:
        """Return kept cases for affected pre-merge stage groups."""
        selected = 0
        for group, members in self._scheduled_groups.items():
            affected_names = affected_stages & members.keys()
            if not affected_names:
                continue
            kept = [kept_per_stage[name] for name in affected_names if name in kept_per_stage]
            selected += max(kept) if kept else self._full_counts[group]
        return selected


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def _source_at(repo_root: Path, commit: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _cache_db_queries(db: Any) -> Any:
    """Memoize immutable read queries reused by thousands of replay windows."""
    for name in (
        "file_has_touch_rows",
        "known_by_family",
        "known_tests",
        "tests_touching_file",
        "tests_touching_func",
        "untrusted_tests",
    ):
        setattr(db, name, cache(getattr(db, name)))
    return db


def _parse_gaps(value: str) -> tuple[int, ...]:
    try:
        gaps = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("gaps must be comma-separated integers") from error
    if not gaps or any(gap <= 0 for gap in gaps):
        raise argparse.ArgumentTypeError("every gap must be a positive integer")
    return gaps


def _is_core_python(path: str) -> bool:
    return path.startswith("tensorrt_llm/") and path.endswith(".py")


def _is_native(path: str) -> bool:
    return path.startswith("cpp/") and Path(path).suffix.lower() in NATIVE_SUFFIXES


def _commits(
    repo_root: Path,
    ref: str,
    since: str,
    until: str | None,
) -> list[Commit]:
    args = ["log", "--first-parent", "--reverse", f"--since={since}"]
    if until:
        args.append(f"--until={until}")
    args.extend(["--format=%H%x09%cI%x09%s", ref])
    commits = []
    for line in _git(repo_root, *args).splitlines():
        sha, timestamp, subject = line.split("\t", 2)
        commits.append(Commit(sha=sha, timestamp=timestamp, subject=subject))
    return commits


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summarize(results: list[WindowResult]) -> dict[str, object]:
    rates = [result.skip_rate for result in results]
    coverage_rates = [result.skip_rate for result in results if result.status == "coverage"]
    relevant_rates = [result.skip_rate for result in results if result.status != "ignored"]
    total_cases = sum(result.total_cases for result in results)
    selected_cases = sum(result.selected_cases for result in results)
    statuses = Counter(result.status for result in results)
    fallback_categories = Counter(
        result.reason_category for result in results if result.status == "fallback"
    )
    return {
        "windows": len(results),
        "statuses": dict(sorted(statuses.items())),
        "fallback_categories": dict(sorted(fallback_categories.items())),
        "fallback_rate": statuses.get("fallback", 0) / len(results),
        "relevant_fallback_rate": (
            statuses.get("fallback", 0) / len(relevant_rates) if relevant_rates else 0.0
        ),
        "mean_skip_rate": statistics.mean(rates),
        "relevant_mean_skip_rate": statistics.mean(relevant_rates) if relevant_rates else 0.0,
        "weighted_skip_rate": 1 - selected_cases / total_cases,
        "median_skip_rate": statistics.median(rates),
        "p25_skip_rate": _percentile(rates, 0.25),
        "p75_skip_rate": _percentile(rates, 0.75),
        "coverage_hit_mean_skip_rate": (statistics.mean(coverage_rates) if coverage_rates else 0.0),
        "coverage_hit_median_skip_rate": (
            statistics.median(coverage_rates) if coverage_rates else 0.0
        ),
    }


def _evaluate_window(
    repo_root: Path,
    commits: list[Commit],
    start: int,
    gap: int,
    stages: dict,
    yaml_index: YAMLIndex,
    durations: dict[str, float],
    case_universe: CaseUniverse,
    db,
    no_data_policy: str,
) -> WindowResult:
    first = commits[start]
    last = commits[start + gap - 1]
    base = _git(repo_root, "rev-parse", f"{first.sha}^1").strip()
    paths = [
        path for path in _git(repo_root, "diff", "--name-only", base, last.sha).splitlines() if path
    ]
    python_files = [path for path in paths if _is_core_python(path)]
    native_files = [path for path in paths if _is_native(path)]
    common = {
        "gap": gap,
        "start_position": start + 1,
        "end_position": start + gap,
        "base": base,
        "head": last.sha,
        "head_timestamp": last.timestamp,
        "head_subject": last.subject,
        "changed_files": len(paths),
        "relevant_python_files": len(python_files),
        "native_files": len(native_files),
        "relevant_python_paths": tuple(python_files),
        "native_paths": tuple(native_files),
        "total_cases": case_universe.total_cases,
    }

    if native_files:
        return WindowResult(
            **common,
            status="fallback",
            reason_category="native",
            reason=f"native code is not mapped by the Python touch DB ({len(native_files)} file(s))",
            selected_cases=case_universe.total_cases,
            skip_rate=0.0,
        )
    if not python_files:
        return WindowResult(
            **common,
            status="ignored",
            reason_category="ignored",
            reason="no relevant core-Python or native change",
            selected_cases=0,
            skip_rate=1.0,
        )

    diffs = {
        path: strip_noop_diff_lines(
            _git(repo_root, "diff", base, last.sha, "--", path, check=False)
        )
        for path in python_files
    }
    pr = PRInputs(changed_files=python_files, diffs=diffs, post_merge=False)
    tier, reason = apply_coverage_tier(
        pr,
        pairs=[],
        handled=set(),
        stages=stages,
        yaml_index=yaml_index,
        repo_root=repo_root,
        db=db,
        no_data_policy=no_data_policy,
        read_source=lambda path: _source_at(repo_root, last.sha, path),
    )
    if tier is None:
        return WindowResult(
            **common,
            status="fallback",
            reason_category=_fallback_category(reason),
            reason=reason,
            selected_cases=case_universe.total_cases,
            skip_rate=0.0,
        )

    kept_counts, _ = compute_coverage_stage_counts(
        affected_stages=tier.affected_stages,
        stages=stages,
        yaml_index=yaml_index,
        removed=tier.removed,
        durations=durations,
    )
    selected_cases = case_universe.selected_cases(tier.affected_stages, kept_counts)
    return WindowResult(
        **common,
        status="coverage",
        reason_category="coverage",
        reason=reason,
        selected_cases=selected_cases,
        skip_rate=1 - selected_cases / case_universe.total_cases,
        impacted_tests=int(tier.detail.get("impacted") or 0),
        removed_cases=int(tier.detail.get("removed_cases") or 0),
    )


def _fallback_category(reason: str) -> str:
    for prefix, category in (
        ("coverage tier declined: zero-touch", "zero_touch"),
        ("coverage tier declined: import-executed", "import_executed"),
        ("coverage tier declined: no usable diff", "no_usable_diff"),
        ("coverage tier declined: unparsable source", "unparsable_source"),
        ("coverage tier declined: closure change", "closure"),
    ):
        if reason.startswith(prefix):
            return category
    return "other_coverage_decline"


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# CBTS rolling-window case skip-rate replay",
        "",
        f"- Ref: `{report['ref']}`",
        f"- Window: `{report['since']}` through `{report['until'] or 'now'}`",
        f"- Coverage DB: `{report['coverage_db']}`",
        f"- Baseline pre-merge case universe: {report['total_cases']} cases",
        "- Relevant gap files: `tensorrt_llm/**/*.py` and native source/headers under `cpp/`",
        "- Native or coverage-declined windows fall back and have 0% skip rate",
        "- Windows without relevant changes have 100% gap-only skip rate",
        "",
        "| Gap | Windows | Coverage | Ignored | Fallback | Fallback rate | Mean skip | "
        "Relevant fallback | Relevant mean skip | Hit-only mean | Hit-only median | "
        "Fallback breakdown |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    summaries = report["summaries"]
    assert isinstance(summaries, dict)
    for gap, raw in summaries.items():
        assert isinstance(raw, dict)
        statuses = raw["statuses"]
        assert isinstance(statuses, dict)
        categories = raw["fallback_categories"]
        assert isinstance(categories, dict)
        breakdown = ", ".join(f"{name}={count}" for name, count in categories.items()) or "none"
        lines.append(
            f"| {gap} | {raw['windows']} | {statuses.get('coverage', 0)} | "
            f"{statuses.get('ignored', 0)} | {statuses.get('fallback', 0)} | "
            f"{raw['fallback_rate']:.1%} | {raw['mean_skip_rate']:.1%} | "
            f"{raw['relevant_fallback_rate']:.1%} | "
            f"{raw['relevant_mean_skip_rate']:.1%} | "
            f"{raw['coverage_hit_mean_skip_rate']:.1%} | "
            f"{raw['coverage_hit_median_skip_rate']:.1%} | {breakdown} |"
        )
    lines += [
        "",
        "`Mean skip` includes fallback as 0% and ignored gap-only windows as 100%. "
        "`Relevant` columns exclude ignored windows. `Hit-only` columns include only windows "
        "accepted by coverage mapping.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--ref", default="upstream/main")
    parser.add_argument("--since", default="14 days ago")
    parser.add_argument("--until", default=None)
    parser.add_argument("--gaps", type=_parse_gaps, default=DEFAULT_GAPS)
    parser.add_argument("--coverage-db", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    parser.add_argument(
        "--no-data-policy",
        choices=sorted(NO_DATA_POLICIES),
        default=DEFAULT_NO_DATA_POLICY,
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    coverage_db = Path(args.coverage_db).resolve()
    test_db_dir = repo_root / "tests/integration/test_lists/test-db"
    groovy_path = repo_root / "jenkins/L0_Test.groovy"
    durations_path = repo_root / "tests/integration/defs/.test_durations"
    if not coverage_db.is_file():
        parser.error(f"coverage DB not found: {coverage_db}")
    if not test_db_dir.is_dir() or not groovy_path.is_file():
        parser.error(f"repo root does not contain CBTS inputs: {repo_root}")

    commits = _commits(repo_root, args.ref, args.since, args.until)
    if not commits:
        parser.error("no first-parent commits found in the requested interval")
    if max(args.gaps) > len(commits):
        parser.error(f"largest gap {max(args.gaps)} exceeds {len(commits)} commits")

    yaml_index = YAMLIndex.load(test_db_dir)
    stages = parse_stages_from_groovy(groovy_path, include_post_merge=True)
    durations = load_durations(durations_path)
    case_universe = CaseUniverse(yaml_index, stages)
    if case_universe.total_cases <= 0:
        parser.error("pre-merge case universe is empty")

    results_by_gap: dict[int, list[WindowResult]] = {}
    db = _cache_db_queries(open_db(str(coverage_db)))
    try:
        for gap in args.gaps:
            results = []
            for start in range(len(commits) - gap + 1):
                results.append(
                    _evaluate_window(
                        repo_root,
                        commits,
                        start,
                        gap,
                        stages,
                        yaml_index,
                        durations,
                        case_universe,
                        db,
                        args.no_data_policy,
                    )
                )
            results_by_gap[gap] = results
            summary = _summarize(results)
            print(
                f"gap={gap}: windows={summary['windows']} "
                f"fallback={summary['fallback_rate']:.1%} "
                f"mean_skip={summary['mean_skip_rate']:.1%}",
                file=sys.stderr,
            )
    finally:
        db.close()

    report: dict[str, object] = {
        "ref": args.ref,
        "since": args.since,
        "until": args.until,
        "coverage_db": str(coverage_db),
        "no_data_policy": args.no_data_policy,
        "total_commits": len(commits),
        "total_cases": case_universe.total_cases,
        "summaries": {str(gap): _summarize(results) for gap, results in results_by_gap.items()},
        "windows": {
            str(gap): [asdict(result) for result in results]
            for gap, results in results_by_gap.items()
        },
    }
    markdown = _markdown(report)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    if args.output_markdown:
        Path(args.output_markdown).write_text(markdown)
    if not args.output_json and not args.output_markdown:
        print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
