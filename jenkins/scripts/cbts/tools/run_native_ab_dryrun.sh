#!/bin/bash
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

set -euo pipefail

usage()
{
    cat <<'EOF'
Run an A/B CBTS historical dry-run with and without the native codegraph.

Usage:
  run_native_ab_dryrun.sh [options]

Options:
  --ref REF                 History tip (default: f04859d04de68b255343a6eb4a7d9d0bbaf82ca8)
  --window N                Number of commits to replay (default: 200)
  --coverage-db PATH        Compact PY_START coverage DB
  --native-codegraph PATH   Native codegraph SQLite DB
  --import-policy POLICY    decline, file, or importers (default: file)
  --out DIR                 Output root (default: ./cbts-native-ab-window<N>-import-<policy>)
  -h, --help                Show this help

The output root contains:
  python-only/              Baseline dry-run artifacts
  with-native/              Native-codegraph dry-run artifacts
  python-only.log           Complete baseline log
  with-native.log           Complete treatment log
  comparison.md             Human-readable A/B result
  comparison.json           Machine-readable A/B result
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel)"
ref="f04859d04de68b255343a6eb4a7d9d0bbaf82ca8"
window=200
coverage_db="/private/tmp/cbts-build-2942/cbts_touchmap.sqlite"
native_codegraph="${repo_root}/reports/cbts_codegraph_x86_compile_commands.sqlite"
import_policy="file"
output_root=""

while [[ $# -gt 0 ]]; do
    case "$1" in
    --ref)
        ref="${2:?--ref requires a value}"
        shift 2
        ;;
    --window)
        window="${2:?--window requires a value}"
        shift 2
        ;;
    --coverage-db)
        coverage_db="${2:?--coverage-db requires a value}"
        shift 2
        ;;
    --native-codegraph)
        native_codegraph="${2:?--native-codegraph requires a value}"
        shift 2
        ;;
    --import-policy)
        import_policy="${2:?--import-policy requires a value}"
        shift 2
        ;;
    --out)
        output_root="${2:?--out requires a value}"
        shift 2
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        echo "error: unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

if [[ ! "${window}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --window must be a positive integer: ${window}" >&2
    exit 2
fi
if [[ ! "${import_policy}" =~ ^(decline|file|importers)$ ]]; then
    echo "error: --import-policy must be decline, file, or importers: ${import_policy}" >&2
    exit 2
fi
if ! git -C "${repo_root}" rev-parse --verify "${ref}^{commit}" >/dev/null 2>&1; then
    echo "error: --ref does not resolve to a commit: ${ref}" >&2
    exit 2
fi
if [[ ! -f "${coverage_db}" ]]; then
    echo "error: coverage DB not found: ${coverage_db}" >&2
    exit 2
fi
if [[ ! -f "${native_codegraph}" ]]; then
    echo "error: native codegraph not found: ${native_codegraph}" >&2
    exit 2
fi

if [[ -z "${output_root}" ]]; then
    output_root="${repo_root}/cbts-native-ab-window${window}-import-${import_policy}"
elif [[ "${output_root}" != /* ]]; then
    output_root="${repo_root}/${output_root#./}"
fi

baseline_dir="${output_root}/python-only"
native_dir="${output_root}/with-native"
dryrun="${repo_root}/jenkins/scripts/cbts/tools/dryrun.py"
mkdir -p "${output_root}"

echo "CBTS native-codegraph A/B dry-run"
echo "  repo:             ${repo_root}"
echo "  ref:              ${ref}"
echo "  window:           ${window}"
echo "  coverage DB:      ${coverage_db}"
echo "  native codegraph: ${native_codegraph}"
echo "  import policy:    ${import_policy}"
echo "  output:           ${output_root}"
echo

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Python-only baseline"
PYTHONUNBUFFERED=1 python3 "${dryrun}" \
    --ref "${ref}" \
    --window "${window}" \
    --filter all \
    --coverage-db "${coverage_db}" \
    --import-executed-policy "${import_policy}" \
    --out "${baseline_dir}" \
    --repo-root "${repo_root}" 2>&1 | tee "${output_root}/python-only.log"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting native-codegraph treatment"
PYTHONUNBUFFERED=1 python3 "${dryrun}" \
    --ref "${ref}" \
    --window "${window}" \
    --filter all \
    --coverage-db "${coverage_db}" \
    --native-codegraph "${native_codegraph}" \
    --import-executed-policy "${import_policy}" \
    --out "${native_dir}" \
    --repo-root "${repo_root}" 2>&1 | tee "${output_root}/with-native.log"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building comparison"
python3 - "${baseline_dir}" "${native_dir}" "${output_root}" "${ref}" "${window}" \
    "${import_policy}" <<'PY'
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

baseline_root, native_root, output_root = map(Path, sys.argv[1:4])
ref = sys.argv[4]
window = int(sys.argv[5])
import_policy = sys.argv[6]


def load_decisions(root):
    return {
        path.parent.name: json.loads(path.read_text())
        for path in sorted(root.glob("*/decision.json"))
    }


def coverage_reason(decision):
    for reason in decision.get("reasons", []):
        if reason.get("source") == "coverage":
            return reason
    return {}


def coverage_attempted(decision):
    if decision.get("scope") == "coverage":
        return True
    return any("coverage_declined" in reason for reason in decision.get("reasons", []))


def declined_reasons(decisions):
    reasons = Counter()
    for decision in decisions.values():
        for reason in decision.get("reasons", []):
            declined = reason.get("coverage_declined")
            if declined:
                reasons[declined] += 1
    return reasons


def summarize(decisions):
    hits = [decision for decision in decisions.values() if decision.get("scope") == "coverage"]
    attempted = [decision for decision in decisions.values() if coverage_attempted(decision)]
    removed = [int(coverage_reason(decision).get("removed_cases", 0)) for decision in hits]
    return {
        "total_prs": len(decisions),
        "coverage_attempts": len(attempted),
        "coverage_hits": len(hits),
        "coverage_hit_rate_pct": round(100 * len(hits) / len(attempted), 2) if attempted else 0.0,
        "fallback_prs": sum(decision.get("scope") is None for decision in decisions.values()),
        "narrowed_prs": sum(
            coverage_reason(decision).get("outcome") == "narrowed" for decision in hits
        ),
        "removed_cases_total": sum(removed),
        "removed_cases_median": statistics.median(removed) if removed else 0,
        "dropped_stages_total": sum(
            int(coverage_reason(decision).get("dropped_stages", 0)) for decision in hits
        ),
    }


baseline = load_decisions(baseline_root)
native = load_decisions(native_root)
labels = sorted(set(baseline) & set(native))
if not labels:
    raise SystemExit("no matching dry-run decisions found")

native_impacts = {}
for label in labels:
    path = native_root / label / "native-impact.json"
    if path.is_file():
        native_impacts[label] = json.loads(path.read_text())


def native_summary(label):
    return native_impacts.get(label, {}).get("summary", {})


native_prs = [
    label for label in labels if int(native_summary(label).get("production_native_files", 0)) > 0
]
fully_mapped_native = [
    label
    for label in native_prs
    if int(native_summary(label).get("mapped_native_files", 0))
    == int(native_summary(label).get("production_native_files", 0))
]
native_rescued = [
    label
    for label in native_prs
    if baseline[label].get("scope") != "coverage" and native[label].get("scope") == "coverage"
]
coverage_regressions = [
    label
    for label in labels
    if baseline[label].get("scope") == "coverage" and native[label].get("scope") != "coverage"
]

baseline_summary = summarize(baseline)
native_summary_all = summarize(native)
comparison = {
    "provenance": {
        "ref": ref,
        "window": window,
        "import_executed_policy": import_policy,
        "matched_prs": len(labels),
        "baseline": baseline_root.as_posix(),
        "treatment": native_root.as_posix(),
    },
    "python_only": baseline_summary,
    "with_native": native_summary_all,
    "delta": {
        key: native_summary_all[key] - baseline_summary[key]
        for key in baseline_summary
        if isinstance(baseline_summary[key], (int, float))
    },
    "native_subset": {
        "native_prs": len(native_prs),
        "fully_mapped_native_prs": len(fully_mapped_native),
        "native_rescued_prs": len(native_rescued),
        "native_rescue_rate_pct": round(100 * len(native_rescued) / len(native_prs), 2)
        if native_prs
        else 0.0,
        "coverage_regressions": len(coverage_regressions),
        "native_rescued_labels": native_rescued,
        "coverage_regression_labels": coverage_regressions,
    },
    "coverage_declines": {
        "python_only": dict(declined_reasons(baseline).most_common()),
        "with_native": dict(declined_reasons(native).most_common()),
    },
}
(output_root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


def pct(value):
    return f"{value:.2f}%"


def delta(value):
    return f"{value:+g}"


metrics = [
    ("Coverage-tier attempts", "coverage_attempts"),
    ("Coverage-tier hits", "coverage_hits"),
    ("Coverage-tier hit rate", "coverage_hit_rate_pct"),
    ("Fallback PRs", "fallback_prs"),
    ("Narrowed PRs", "narrowed_prs"),
    ("Removed YAML entries", "removed_cases_total"),
    ("Median removed entries / hit", "removed_cases_median"),
    ("Dropped stages", "dropped_stages_total"),
]
lines = [
    "# CBTS native-codegraph A/B dry-run",
    "",
    f"- Ref: `{ref}`",
    f"- Window: `{window}` commits",
    f"- Import-executed policy: `{import_policy}`",
    f"- Matched decisions: `{len(labels)}`",
    "- Hit-rate denominator: PRs that actually attempted the coverage tier",
    "",
    "## Summary",
    "",
    "| Metric | Python-only | With native graph | Delta |",
    "|---|---:|---:|---:|",
]
for label, key in metrics:
    before = baseline_summary[key]
    after = native_summary_all[key]
    if key.endswith("_pct"):
        before_text = pct(before)
        after_text = pct(after)
        delta_text = f"{after - before:+.2f} pp"
    else:
        before_text = str(before)
        after_text = str(after)
        delta_text = delta(after - before)
    lines.append(f"| {label} | {before_text} | {after_text} | {delta_text} |")

rescue_rate = comparison["native_subset"]["native_rescue_rate_pct"]
lines += [
    "",
    "## Native PR subset",
    "",
    f"- PRs changing production native files: **{len(native_prs)}**",
    f"- Fully mapped native PRs: **{len(fully_mapped_native)}**",
    f"- Rescued into `scope=coverage`: **{len(native_rescued)}** ({rescue_rate:.2f}%)",
    f"- Coverage regressions: **{len(coverage_regressions)}**",
    "",
    "### Native rescues",
    "",
]
lines.extend(f"- [{label}](./with-native/{label}/summary.txt)" for label in native_rescued)
if not native_rescued:
    lines.append("- None")

lines += ["", "## Top coverage-tier declines", ""]
for mode, title in ((baseline, "Python-only"), (native, "With native graph")):
    lines.append(f"### {title}")
    lines.append("")
    reasons = declined_reasons(mode).most_common(15)
    lines.extend(f"- {count} × `{reason}`" for reason, count in reasons)
    if not reasons:
        lines.append("- None")
    lines.append("")

lines += [
    "## Interpretation",
    "",
    "A native rescue is a PR with production C++/CUDA changes whose baseline did not hit the coverage tier,",
    "but whose treatment run did. Forced-run untrusted/no-data entries remain part of the safety policy and",
    "should be reported separately from the statically impacted cases.",
]
(output_root / "comparison.md").write_text("\n".join(lines) + "\n")

print(json.dumps(comparison["native_subset"], indent=2))
print(f"Open {output_root / 'comparison.md'}")
PY

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Complete"
echo "Comparison: ${output_root}/comparison.md"
echo "JSON:       ${output_root}/comparison.json"
