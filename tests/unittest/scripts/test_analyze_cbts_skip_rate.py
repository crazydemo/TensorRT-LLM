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
"""Tests for the CBTS rolling-window skip-rate analyzer."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "jenkins/scripts/cbts/tools/analyze_skip_rate.py"


@pytest.fixture()
def analyzer():
    spec = importlib.util.spec_from_file_location("analyze_skip_rate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_gaps_rejects_non_positive_values(analyzer):
    assert analyzer._parse_gaps("1,2,2,5") == (1, 2, 5)
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        analyzer._parse_gaps("1,0")


@pytest.mark.parametrize(
    ("path", "core_python", "native"),
    [
        ("tensorrt_llm/_torch/model.py", True, False),
        ("tensorrt_llm/usage/manifest.json", False, False),
        ("cpp/tensorrt_llm/runtime/runtime.cpp", False, True),
        ("cpp/tensorrt_llm/kernels/kernel.cu", False, True),
        ("examples/plugin.cpp", False, False),
        ("jenkins/L0_Test.groovy", False, False),
    ],
)
def test_relevant_path_classification(analyzer, path, core_python, native):
    assert analyzer._is_core_python(path) is core_python
    assert analyzer._is_native(path) is native


def test_summarize_counts_fallback_as_zero_skip(analyzer):
    common = {
        "gap": 1,
        "start_position": 1,
        "end_position": 1,
        "base": "base",
        "head": "head",
        "head_timestamp": "2026-09-03T00:00:00Z",
        "head_subject": "subject",
        "changed_files": 1,
        "relevant_python_files": 1,
        "native_files": 0,
        "relevant_python_paths": ("tensorrt_llm/test.py",),
        "native_paths": (),
        "reason_category": "test",
        "reason": "test",
        "total_cases": 100,
    }
    results = [
        analyzer.WindowResult(
            **common,
            status="coverage",
            selected_cases=20,
            skip_rate=0.8,
        ),
        analyzer.WindowResult(
            **common,
            status="fallback",
            selected_cases=100,
            skip_rate=0.0,
        ),
        analyzer.WindowResult(
            **common,
            status="ignored",
            selected_cases=0,
            skip_rate=1.0,
        ),
    ]

    summary = analyzer._summarize(results)

    assert summary["statuses"] == {"coverage": 1, "fallback": 1, "ignored": 1}
    assert summary["fallback_rate"] == pytest.approx(1 / 3)
    assert summary["relevant_fallback_rate"] == pytest.approx(1 / 2)
    assert summary["mean_skip_rate"] == pytest.approx(0.6)
    assert summary["relevant_mean_skip_rate"] == pytest.approx(0.4)
    assert summary["weighted_skip_rate"] == pytest.approx(0.6)
    assert summary["coverage_hit_mean_skip_rate"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("reason", "category"),
    [
        ("coverage tier declined: import-executed change", "import_executed"),
        ("coverage tier declined: zero-touch residual file", "zero_touch"),
        ("coverage tier declined: no usable diff", "no_usable_diff"),
        ("coverage tier declined: something new", "other_coverage_decline"),
    ],
)
def test_fallback_category(analyzer, reason, category):
    assert analyzer._fallback_category(reason) == category


def test_coverage_tier_forwards_replay_source_reader(analyzer, monkeypatch):
    coverage_tier = sys.modules["coverage_tier"]
    seen = {}

    class RejectingSelector:
        def __init__(self, _db, _repo_root, **kwargs):
            seen.update(kwargs)

        def decide(self, _residual, _diffs):
            return SimpleNamespace(ok=False, reason="test decline")

    monkeypatch.setattr(coverage_tier, "CoverageSelector", RejectingSelector)

    def read_source(_path):
        return "source"

    tier, _ = coverage_tier.apply_coverage_tier(
        analyzer.PRInputs(
            changed_files=["tensorrt_llm/test.py"],
            diffs={"tensorrt_llm/test.py": "diff"},
            post_merge=False,
        ),
        pairs=[],
        handled=set(),
        stages={},
        yaml_index=object(),
        repo_root=REPO_ROOT,
        db=object(),
        read_source=read_source,
    )

    assert tier is None
    assert seen["read_source"] is read_source
