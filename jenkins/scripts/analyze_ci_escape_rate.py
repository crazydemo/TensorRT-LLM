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
"""Analyze pre-merge CI outcomes for cases that fail in official post-merge CI."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, cast

DEFAULT_OPENSEARCH_URL = "http://gpuwa.nvidia.com/opensearch"
TEST_INDEX = "df-swdl-trtllm-infra-ci-prod-test_info-*"
PR_INDEX = "df-swdl-trtllm-infra-ci-prod-pr_info-*"
POST_MERGE_JOB = "LLM/main/L0_PostMerge"
PRE_MERGE_JOB = "LLM/main/L0_MergeRequest_PR"
POST_MERGE_STAGE_MARKER = "Post-Merge"
MAX_QUERY_HITS = 10_000
REQUEST_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class PostMergeBuild:
    build_id: str
    commit: str
    latest_timestamp_ms: int


@dataclass(frozen=True)
class PullRequest:
    number: str
    merge_commit: str
    head_commit: str
    ci_skipped: bool


@dataclass(frozen=True)
class TestRecord:
    build_id: str
    commit: str
    pr_number: str
    stage: str
    case: str
    status: str
    skipped_message: str
    timestamp_ms: int


@dataclass(frozen=True)
class FailedCase:
    stage: str
    stage_family: str
    case: str
    timestamp_ms: int


def normalize_stage(stage: str) -> str:
    """Return a stable stage family across CBTS naming and shard movement."""
    normalized = re.sub(r"-cbts$", "", stage)
    return re.sub(r"-\d+$", "", normalized)


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"expected {description} to be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, description: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"expected {description} to be an array")
    return cast(Sequence[object], value)


def _string(source: Mapping[str, object], field: str) -> str:
    value = source.get(field, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"expected {field} to be a string")
    return value


def _integer(source: Mapping[str, object], field: str) -> int:
    value = source.get(field, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected {field} to be numeric")
    return int(value)


def _boolean(source: Mapping[str, object], field: str) -> bool:
    value = source.get(field, False)
    if not isinstance(value, bool):
        raise ValueError(f"expected {field} to be boolean")
    return value


def _sources(response: Mapping[str, object]) -> list[Mapping[str, object]]:
    hits = _mapping(response.get("hits"), "hits")
    total = hits.get("total", 0)
    if isinstance(total, Mapping):
        total_value = _integer(cast(Mapping[str, object], total), "value")
    elif isinstance(total, int):
        total_value = total
    else:
        raise ValueError("expected hits.total to be numeric or an object")
    raw_hits = _sequence(hits.get("hits"), "hits.hits")
    if total_value > len(raw_hits):
        raise RuntimeError(
            f"query returned {len(raw_hits)} of {total_value} hits; narrow the build window"
        )
    return [_mapping(_mapping(hit, "hit").get("_source"), "hit._source") for hit in raw_hits]


class OpenSearchClient:
    """Small requests-free client for the read-only CI OpenSearch endpoint."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def search(self, index: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        request = urllib.request.Request(
            f"{self._base_url}/{index}/_search",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            parsed = json.loads(response.read())
        return _mapping(parsed, "OpenSearch response")


def discover_post_merge_builds(client: OpenSearchClient, build_count: int) -> list[PostMergeBuild]:
    payload: dict[str, object] = {
        "size": 0,
        "query": {"term": {"s_job_name": POST_MERGE_JOB}},
        "aggs": {
            "builds": {
                "terms": {
                    "field": "s_build_id",
                    "size": build_count,
                    "order": {"latest": "desc"},
                },
                "aggs": {
                    "latest": {"max": {"field": "ts_created"}},
                    "sample": {
                        "top_hits": {
                            "size": 1,
                            "_source": ["s_trigger_mr_commit"],
                        }
                    },
                },
            }
        },
    }
    response = client.search(TEST_INDEX, payload)
    aggregations = _mapping(response.get("aggregations"), "aggregations")
    builds_agg = _mapping(aggregations.get("builds"), "aggregations.builds")
    buckets = _sequence(builds_agg.get("buckets"), "aggregations.builds.buckets")
    builds = []
    for raw_bucket in buckets:
        bucket = _mapping(raw_bucket, "build bucket")
        build_id = _string(bucket, "key")
        latest = _mapping(bucket.get("latest"), "build.latest")
        sample = _mapping(bucket.get("sample"), "build.sample")
        sample_hits = _mapping(sample.get("hits"), "build.sample.hits")
        hit = _mapping(_sequence(sample_hits.get("hits"), "build.sample.hits.hits")[0], "hit")
        source = _mapping(hit.get("_source"), "hit._source")
        builds.append(
            PostMergeBuild(
                build_id=build_id,
                commit=_string(source, "s_trigger_mr_commit"),
                latest_timestamp_ms=_integer(latest, "value"),
            )
        )
    if len(builds) < 2:
        raise RuntimeError("fewer than two official post-merge builds were found")
    return sorted(builds, key=lambda build: build.latest_timestamp_ms)


def commits_between(repo: Path, previous_commit: str, current_commit: str) -> list[str]:
    """Return main commits after the previous post-merge revision through the current one."""
    command = [
        "git",
        "rev-list",
        "--reverse",
        f"{previous_commit}..{current_commit}",
    ]
    result = subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line]


def query_pull_requests(
    client: OpenSearchClient, merge_commits: Sequence[str]
) -> list[PullRequest]:
    if not merge_commits:
        return []
    payload: dict[str, object] = {
        "size": MAX_QUERY_HITS,
        "track_total_hits": True,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"s_repo_name": "NVIDIA/TensorRT-LLM"}},
                    {"terms": {"s_merge_commit_sha": list(merge_commits)}},
                ]
            }
        },
        "_source": [
            "s_pr_id",
            "s_merge_commit_sha",
            "s_head_sha",
            "b_ci_skipped",
            "ts_created",
        ],
    }
    latest_by_pr: dict[str, tuple[int, PullRequest]] = {}
    for source in _sources(client.search(PR_INDEX, payload)):
        pull_request = PullRequest(
            number=_string(source, "s_pr_id"),
            merge_commit=_string(source, "s_merge_commit_sha"),
            head_commit=_string(source, "s_head_sha"),
            ci_skipped=_boolean(source, "b_ci_skipped"),
        )
        timestamp_ms = _integer(source, "ts_created")
        previous = latest_by_pr.get(pull_request.number)
        if previous is None or timestamp_ms > previous[0]:
            latest_by_pr[pull_request.number] = (timestamp_ms, pull_request)
    return sorted(
        (entry[1] for entry in latest_by_pr.values()),
        key=lambda pull_request: int(pull_request.number),
    )


def query_failed_cases(client: OpenSearchClient, build: PostMergeBuild) -> list[FailedCase]:
    payload: dict[str, object] = {
        "size": MAX_QUERY_HITS,
        "track_total_hits": True,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"s_job_name": POST_MERGE_JOB}},
                    {"term": {"s_build_id": build.build_id}},
                    {"term": {"s_status": "FAILED"}},
                ],
                "must_not": [{"wildcard": {"s_stage_name": f"*{POST_MERGE_STAGE_MARKER}*"}}],
            }
        },
        "_source": ["s_stage_name", "s_turtle_name", "ts_created"],
    }
    unique: dict[tuple[str, str], FailedCase] = {}
    for source in _sources(client.search(TEST_INDEX, payload)):
        stage = _string(source, "s_stage_name")
        case = _string(source, "s_turtle_name")
        if not stage or not case:
            continue
        unique[(stage, case)] = FailedCase(
            stage=stage,
            stage_family=normalize_stage(stage),
            case=case,
            timestamp_ms=_integer(source, "ts_created"),
        )
    return sorted(unique.values(), key=lambda failure: (failure.stage, failure.case))


def query_pre_merge_records(
    client: OpenSearchClient,
    pull_requests: Sequence[PullRequest],
    failed_cases: Sequence[FailedCase],
    cutoff_timestamp_ms: int,
) -> list[TestRecord]:
    pr_numbers = sorted({pull_request.number for pull_request in pull_requests})
    cases = sorted({failure.case for failure in failed_cases})
    if not pr_numbers or not cases:
        return []
    payload: dict[str, object] = {
        "size": MAX_QUERY_HITS,
        "track_total_hits": True,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"s_job_name": PRE_MERGE_JOB}},
                    {"terms": {"s_trigger_mr_id": pr_numbers}},
                    {"terms": {"s_turtle_name": cases}},
                    {"range": {"ts_created": {"lte": cutoff_timestamp_ms}}},
                ]
            }
        },
        "_source": [
            "s_build_id",
            "s_trigger_mr_id",
            "s_trigger_mr_commit",
            "s_status",
            "s_stage_name",
            "s_turtle_name",
            "s_skipped_message",
            "ts_created",
        ],
    }
    unique_records: dict[tuple[object, ...], TestRecord] = {}
    for source in _sources(client.search(TEST_INDEX, payload)):
        record = TestRecord(
            build_id=_string(source, "s_build_id"),
            commit=_string(source, "s_trigger_mr_commit"),
            pr_number=_string(source, "s_trigger_mr_id"),
            stage=_string(source, "s_stage_name"),
            case=_string(source, "s_turtle_name"),
            status=_string(source, "s_status"),
            skipped_message=_string(source, "s_skipped_message"),
            timestamp_ms=_integer(source, "ts_created"),
        )
        unique_records[tuple(asdict(record).values())] = record
    return list(unique_records.values())


def analyze_failed_cases(
    failed_cases: Sequence[FailedCase],
    pull_requests: Sequence[PullRequest],
    records: Sequence[TestRecord],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Classify failed post-merge cases against all bot runs of each merged PR."""
    records_by_key: dict[tuple[str, str], list[TestRecord]] = defaultdict(list)
    for record in records:
        records_by_key[(normalize_stage(record.stage), record.case)].append(record)

    details = []
    outcome_counts: Counter[str] = Counter()
    matrix_counts: Counter[str] = Counter()
    all_candidate_prs_passed = 0
    for failure in failed_cases:
        matching = records_by_key[(failure.stage_family, failure.case)]
        records_by_pr: dict[str, list[TestRecord]] = defaultdict(list)
        for record in matching:
            records_by_pr[record.pr_number].append(record)

        pr_outcomes: dict[str, str] = {}
        for pull_request in pull_requests:
            pr_records = records_by_pr[pull_request.number]
            statuses = {record.status for record in pr_records}
            if "FAILED" in statuses and "PASSED" in statuses:
                outcome = "mixed_pass_fail"
            elif "FAILED" in statuses:
                outcome = "failed"
            elif "PASSED" in statuses:
                outcome = "passed"
            elif any(
                record.status == "SKIPPED"
                and record.skipped_message == "Reused from previous pipeline"
                for record in pr_records
            ):
                outcome = "reuse_unresolved"
            elif pr_records:
                outcome = "skipped"
            else:
                outcome = "not_run"
            pr_outcomes[pull_request.number] = outcome
            matrix_counts[outcome] += 1

        observed = set(pr_outcomes.values())
        if "failed" in observed or "mixed_pass_fail" in observed:
            case_outcome = "detected_then_merged"
        elif "passed" in observed:
            case_outcome = "passed_only"
        elif observed - {"not_run"}:
            case_outcome = "skipped_only"
        else:
            case_outcome = "not_run"
        outcome_counts[case_outcome] += 1

        strict_pass = bool(pr_outcomes) and all(
            outcome == "passed" for outcome in pr_outcomes.values()
        )
        all_candidate_prs_passed += int(strict_pass)
        details.append(
            {
                **asdict(failure),
                "outcome": case_outcome,
                "all_candidate_prs_passed": strict_pass,
                "pr_outcomes": pr_outcomes,
                "matching_records": [asdict(record) for record in matching],
            }
        )

    failed_case_count = len(failed_cases)
    undetected_count = failed_case_count - outcome_counts["detected_then_merged"]
    summary: dict[str, object] = {
        "failed_cases": failed_case_count,
        "failure_incidents": len(
            {(failure.stage, failure.timestamp_ms) for failure in failed_cases}
        ),
        "candidate_prs": len(pull_requests),
        "ci_skipped_prs": sum(pull_request.ci_skipped for pull_request in pull_requests),
        "case_outcomes": dict(sorted(outcome_counts.items())),
        "candidate_pr_case_outcomes": dict(sorted(matrix_counts.items())),
        "all_candidate_prs_passed": all_candidate_prs_passed,
        "undetected_count": undetected_count,
        "undetected_rate": undetected_count / failed_case_count if failed_case_count else None,
    }
    return summary, details


def analyze_interval(
    client: OpenSearchClient,
    repo: Path,
    previous_build: PostMergeBuild,
    current_build: PostMergeBuild,
) -> dict[str, object]:
    merge_commits = commits_between(repo, previous_build.commit, current_build.commit)
    pull_requests = query_pull_requests(client, merge_commits)
    failed_cases = query_failed_cases(client, current_build)
    records = query_pre_merge_records(
        client,
        pull_requests,
        failed_cases,
        current_build.latest_timestamp_ms,
    )
    summary, details = analyze_failed_cases(failed_cases, pull_requests, records)
    mapped_commits = {pull_request.merge_commit for pull_request in pull_requests}
    return {
        "previous_post_merge": asdict(previous_build),
        "current_post_merge": asdict(current_build),
        "merge_commits": merge_commits,
        "pull_requests": [asdict(pull_request) for pull_request in pull_requests],
        "unmapped_merge_commits": [
            commit for commit in merge_commits if commit not in mapped_commits
        ],
        "summary": summary,
        "failed_case_details": details,
    }


def _summarize_intervals(intervals: Sequence[Mapping[str, object]]) -> dict[str, object]:
    outcomes: Counter[str] = Counter()
    failed_cases = 0
    failure_incidents = 0
    undetected_count = 0
    for interval in intervals:
        summary = _mapping(interval.get("summary"), "interval.summary")
        failed_cases += _integer(summary, "failed_cases")
        failure_incidents += _integer(summary, "failure_incidents")
        undetected_count += _integer(summary, "undetected_count")
        raw_outcomes = _mapping(summary.get("case_outcomes"), "summary.case_outcomes")
        for outcome, count in raw_outcomes.items():
            if not isinstance(count, int):
                raise ValueError("expected case outcome count to be an integer")
            outcomes[outcome] += count
    return {
        "intervals": len(intervals),
        "failed_cases": failed_cases,
        "failure_incidents": failure_incidents,
        "case_outcomes": dict(sorted(outcomes.items())),
        "undetected_count": undetected_count,
        "undetected_rate": undetected_count / failed_cases if failed_cases else None,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare failures in official post-merge pre-merge stages with all bot runs of merged PRs."
        )
    )
    parser.add_argument(
        "--build-count",
        type=int,
        default=2,
        help="number of latest post-merge builds to analyze; produces N-1 intervals",
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--opensearch-url", default=DEFAULT_OPENSEARCH_URL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.build_count < 2:
        parser.error("--build-count must be at least 2")

    client = OpenSearchClient(args.opensearch_url)
    builds = discover_post_merge_builds(client, args.build_count)
    intervals = [
        analyze_interval(client, args.repo, previous, current)
        for previous, current in zip(builds, builds[1:])
    ]
    report = {
        "definition": {
            "post_merge_job": POST_MERGE_JOB,
            "pre_merge_job": PRE_MERGE_JOB,
            "excluded_stage_marker": POST_MERGE_STAGE_MARKER,
            "case_key": "normalized stage family + s_turtle_name",
            "pr_mapping": "pr_info.s_merge_commit_sha matched to each main commit",
            "pre_merge_evidence": "all bot-run builds and revisions for each mapped PR",
        },
        "summary": _summarize_intervals(intervals),
        "intervals": intervals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
