#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""QA System Coverage Analysis.

Analyzes **non-accuracy** test cases from ``llm_function_core.txt`` and
generates a Markdown coverage report covering four categories:

- **E2E workflow** (``test_e2e.py``): end-to-end scenario tests
- **Disaggregated serving infra** (``disaggregated/test_*.py``): transport,
  routing, topology and worker-management tests
- **Serving API** (``examples/serve/test_serve*.py``): positive & negative
  API robustness tests
- **LLM API contract** (``llmapi/test_llm_api_qa.py``): configuration and
  backend type tests

Usage::

    python scripts/qa_system_coverage_analysis.py
    python scripts/qa_system_coverage_analysis.py --output-dir my_report
"""

from __future__ import annotations

import argparse
import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TEST_LIST = "tests/integration/test_lists/qa/llm_function_core.txt"
DEFAULT_OUTPUT_DIR = "reports/qa_system_coverage"

E2E_SOURCE = "tests/integration/defs/test_e2e.py"
DISAGG_SOURCES = [
    "tests/integration/defs/disaggregated/test_disaggregated.py",
    "tests/integration/defs/disaggregated/test_disaggregated_single_gpu.py",
    "tests/integration/defs/disaggregated/test_workers.py",
    "tests/integration/defs/disaggregated/test_auto_scaling.py",
]
SERVE_SOURCES = [
    "tests/integration/defs/examples/serve/test_serve.py",
    "tests/integration/defs/examples/serve/test_serve_negative.py",
]

# ---------------------------------------------------------------------------
# E2E: category classification patterns
# ---------------------------------------------------------------------------

# A: Legacy TRT engine backend tests (note: no \b on qwen — name continues with _large_new_tokens)
_CAT_A = re.compile(
    r"test_llama_e2e\b|test_mistral_e2e\b|test_qwen_e2e_cpprunner"
    r"|test_mistral_large_hidden_vocab_size\b"
)
# B: PyTorch backend tests (ptp quickstart, EAGLE-3 RCCA, MTP bench, LLM API logits,
#    multi-node model evaluation)
_CAT_B = re.compile(
    r"test_ptp_|test_relaxed_acceptance_|test_draft_token_tree_"
    r"|test_eagle3_output_repetition_|test_deepseek_r1_mtp_bench\b"
    r"|test_llmapi_generation_logits\b"
    r"|test_multi_nodes_eval\b"
)
# C: OpenAI API / trtllm-serve tests
_CAT_C = re.compile(
    r"test_openai_|test_trtllm_serve_"
    r"|test_trtllm_benchmark_serving\b|test_trtllm_multimodal_benchmark_serving\b"
    r"|test_build_time_benchmark_sanity\b"
)
# D: trtllm-bench CLI tests
_CAT_D = re.compile(r"test_trtllm_bench_")

# ---------------------------------------------------------------------------
# E2E: universe items (docs-based coverage targets)
# ---------------------------------------------------------------------------


@dataclass
class _UnivItem:
    """One documented coverage target (guide / example / feature)."""

    name: str
    doc_ref: str  # source document or script path
    func_pat: str = ""  # regex matched against func name (empty = skip)
    bracket_pat: str = ""  # regex matched against bracket param (empty = skip)


_DEPLOYMENT_GUIDES: list[_UnivItem] = [
    _UnivItem(
        "DeepSeek R1",
        "deployment-guide-for-deepseek-r1-on-trtllm.md",
        func_pat=r"deepseek_r1",
        bracket_pat=r"DeepSeek.R1",
    ),
    _UnivItem(
        "GPT-OSS 20B",
        "deployment-guide-for-gpt-oss-on-trtllm.md",
        bracket_pat=r"gpt.oss",
    ),
    _UnivItem(
        "Kimi K2 Thinking",
        "deployment-guide-for-kimi-k2-thinking-on-trtllm.md",
        func_pat=r"kimi",
        bracket_pat=r"[Kk]imi",
    ),
    _UnivItem(
        "Llama 3.3 70B",
        "deployment-guide-for-llama3.3-70b-on-trtllm.md",
        func_pat=r"pp_enabled",
        bracket_pat=r"[Ll]lama.?3[._-]3",  # codespell:ignore lama
    ),
    _UnivItem(
        "Llama 4 Scout",
        "deployment-guide-for-llama4-scout-on-trtllm.md",
        func_pat=r"chunked_prefill",
        bracket_pat=r"Llama-4-Scout",
    ),
    _UnivItem(
        "Qwen3",
        "deployment-guide-for-qwen3-on-trtllm.md",
        bracket_pat=r"[Qq]wen3(?!-?[Nn]ext)",
    ),
    _UnivItem(
        "Qwen3-Next",
        "deployment-guide-for-qwen3-next-on-trtllm.md",
        bracket_pat=r"[Qq]wen3.?[Nn]ext",
    ),
]

_LLM_API_EXAMPLES: list[_UnivItem] = [
    _UnivItem(
        "quickstart_example.py",
        "examples/llm-api/quickstart_example.py",
        func_pat=r"test_ptp_quickstart$",
    ),
    _UnivItem(
        "quickstart_advanced.py",
        "examples/llm-api/quickstart_advanced.py",
        func_pat=r"test_ptp_quickstart_advanced$",
    ),
    _UnivItem(
        "quickstart_multimodal.py",
        "examples/llm-api/quickstart_multimodal.py",
        func_pat=r"test_ptp_quickstart_multimodal",
    ),
    _UnivItem(
        "llm_inference.py",
        "examples/llm-api/llm_inference.py",
        func_pat=r"test_ptp_quickstart_advanced$",
    ),
    _UnivItem(
        "llm_inference_distributed.py",
        "examples/llm-api/llm_inference_distributed.py",
        func_pat=r"test_ptp_quickstart_advanced_multi_gpus|test_ptp_quickstart_advanced_pp_enabled",
    ),
    _UnivItem("llm_inference_async.py", "examples/llm-api/llm_inference_async.py"),
    _UnivItem(
        "llm_inference_async_streaming.py",
        "examples/llm-api/llm_inference_async_streaming.py",
    ),
    _UnivItem(
        "llm_speculative_decoding.py",
        "examples/llm-api/llm_speculative_decoding.py",
        func_pat=r"test_ptp_quickstart_advanced_ngram|test_eagle3_output_repetition|test_relaxed_acceptance_",
    ),
    _UnivItem("llm_multilora.py", "examples/llm-api/llm_multilora.py"),
    _UnivItem(
        "llm_logits_processor.py",
        "examples/llm-api/llm_logits_processor.py",
        func_pat=r"test_llmapi_generation_logits",
    ),
    _UnivItem("llm_sampling.py", "examples/llm-api/llm_sampling.py"),
    _UnivItem("llm_guided_decoding.py", "examples/llm-api/llm_guided_decoding.py"),
    _UnivItem("llm_kv_cache_offloading.py", "examples/llm-api/llm_kv_cache_offloading.py"),
    _UnivItem("llm_runtime.py", "examples/llm-api/llm_runtime.py"),
    _UnivItem("llm_sparse_attention.py", "examples/llm-api/llm_sparse_attention.py"),
    _UnivItem("star_attention.py", "examples/llm-api/star_attention.py"),
]

_SERVE_FEATURES: list[_UnivItem] = [
    _UnivItem(
        "Chat completions (multi-turn)",
        "quick-start-guide.md",
        func_pat=r"test_openai_multi_chat_example",
    ),
    _UnivItem(
        "Chat completions (consistent)",
        "quick-start-guide.md",
        func_pat=r"test_openai_consistent_chat",
    ),
    _UnivItem(
        "Chat completions (basic)",
        "quick-start-guide.md",
        func_pat=r"test_openai_chat_example",
    ),
    _UnivItem(
        "Completions endpoint",
        "quick-start-guide.md",
        func_pat=r"test_openai_completions_example",
    ),
    _UnivItem(
        "Guided decoding",
        "llm-api/index.md",
        func_pat=r"test_openai_chat_guided_decoding",
    ),
    _UnivItem(
        "Chat harmony",
        "trtllm-serve docs",
        func_pat=r"test_openai_chat_harmony",
    ),
    _UnivItem(
        "Benchmark serving",
        "perf-benchmarking.md",
        func_pat=r"test_trtllm_benchmark_serving$",
    ),
    _UnivItem(
        "Multimodal benchmark serving",
        "perf-benchmarking.md",
        func_pat=r"test_trtllm_multimodal_benchmark_serving",
    ),
    _UnivItem(
        "Multimodal serving (OpenAI)",
        "quick-start-guide.md",
        func_pat=r"test_openai_chat_multimodal_example|test_openai_mmencoder_example",
    ),
    _UnivItem("Streaming", "quick-start-guide.md"),
    _UnivItem(
        "Logprobs",
        "quick-start-guide.md",
        func_pat=r"test_trtllm_serve_top_logprobs",
    ),
    _UnivItem(
        "Tool calling",
        "quick-start-guide.md",
        func_pat=r"test_openai_tool_call",
    ),
    _UnivItem(
        "LoRA serving",
        "quick-start-guide.md",
        func_pat=r"test_openai_lora|test_trtllm_serve_lora",
    ),
    _UnivItem(
        "Prometheus metrics",
        "trtllm-serve docs",
        func_pat=r"test_openai_prometheus",
    ),
    _UnivItem(
        "Responses API",
        "trtllm-serve docs",
        func_pat=r"test_openai_responses",
    ),
    _UnivItem(
        "Multi-node serving",
        "trtllm-serve docs",
        func_pat=r"test_openai_multinodes|test_openai_disagg_multi_nodes",
    ),
]

_BENCH_FEATURES: list[_UnivItem] = [
    _UnivItem(
        "throughput — PyTorch backend",
        "perf-benchmarking.md",
        func_pat=r"test_trtllm_bench_pytorch_backend_sanity",
    ),
    _UnivItem(
        "throughput — TRT engine",
        "perf-benchmarking.md",
        func_pat=r"test_trtllm_bench_sanity",
    ),
    _UnivItem(
        "latency — PyTorch backend",
        "perf-benchmarking.md",
        func_pat=r"test_trtllm_bench_latency$",
    ),
    _UnivItem(
        "latency — TRT engine",
        "perf-benchmarking.md",
        func_pat=r"test_trtllm_bench_latency_sanity",
    ),
    _UnivItem(
        "LoRA / MoE benchmark",
        "perf-benchmarking.md",
        func_pat=r"test_trtllm_bench_mgmn",
    ),
]

# ---------------------------------------------------------------------------
# Disaggregated feature detection from function name
# (checked in order; multiple can match per function)
# ---------------------------------------------------------------------------
DISAGG_FEATURE_PATTERNS: list[tuple[str, str]] = [
    # Transport backends
    (r"_mpirun\b", "MPI"),
    (r"_ucx\b", "UCX"),
    (r"_nixl\b", "NIXL"),
    (r"_transceiver_runtime_python\b", "Python Transceiver"),
    (r"_trt_backend\b", "TensorRT Backend"),
    # Features
    (r"_cuda_graph\b", "CUDA Graph"),
    # Attention DP: most specific first
    (r"_attention_dp_gen_only", "Attention DP (gen-only)"),
    (r"_attention_dp_one_mtp", "Attention DP (one-model) + MTP"),
    (r"_attention_dp_one\b", "Attention DP (one-model)"),
    (r"_attention_dp\b", "Attention DP"),
    (r"_mtp\b", "MTP"),
    # Routing / worker management
    (r"_cache_aware_balance\b", "KV Cache-Aware Balance"),
    (r"_load_balance\b", "Load Balance"),
    (r"_conditional_disaggregation\b", "Conditional Disaggregation"),
    (r"_kv_cache_events\b", "KV Cache Events"),
    (r"_kv_cache_aware_router_eviction\b", "KV Cache-Aware Router (eviction)"),
    (r"_kv_cache_aware_router\b", "KV Cache-Aware Router"),
    (r"_kv_cache_time_output\b", "KV Cache Time Output"),
    (r"_kv_cache_overflow\b", "KV Cache Overflow"),
    (r"_trtllm_sampler\b", "TRTLLM Sampler"),
    # Topology
    (r"_ctxtp2pp2_gentp2pp2\b", "TP2+PP2 Topology"),
    (r"_ctxpp4_genpp4\b", "PP4 Topology"),
    (r"_multi_gpu_with_mpirun\b", "Multi-GPU (MPI)"),
    (r"_single_gpu_with_mpirun\b", "Single-GPU (MPI)"),
    # Output types
    (r"_logprobs\b", "Logprobs"),
    (r"_logits\b", "Logits"),
    (r"_cancel_gen_requests\b", "Cancel Requests"),
    # Auto-scaling
    (r"^test_service_discovery\b", "Service Discovery"),
    (r"^test_minimal_instances\b", "Minimal Instances"),
    (r"^test_worker_restart\b", "Worker Restart"),
    (r"^test_disagg_server_restart\b", "Disagg Server Restart"),
    # Long context (no \b — followed by _kv_cache_overflow in real test names)
    (r"_long_context", "Long Context"),
]

# Routing strategies and service-discovery backends detected from bracket params
_ROUTING_STRATEGIES = ["round_robin", "load_balancing", "kv_cache_aware"]
_SERVICE_DISCOVERY = ["etcd", "http"]

# Model keyword → display name (checked against bracket param string)
_MODEL_KW: list[tuple[str, str]] = [
    ("TinyLlama", "TinyLlama 1.1B"),
    ("DeepSeek-V3-Lite", "DeepSeek-V3-Lite"),
    ("Llama-4-Maverick", "Llama-4-Maverick"),
    ("Llama-4-Scout", "Llama-4-Scout"),
    ("Qwen3-8B", "Qwen3-8B"),
    ("Qwen3-30B", "Qwen3-30B"),
    ("Qwen3-235B", "Qwen3-235B"),
    ("Llama3.3-70B", "Llama 3.3 70B"),
    ("Llama3.1-405B", "Llama 3.1 405B"),
    ("Llama3.1-70B", "Llama 3.1 70B"),
    ("Llama3.1-8B", "Llama 3.1 8B"),
    ("Llama3.2-11B", "Llama 3.2 11B"),
    ("Llama-3.3", "Llama 3.3"),
    ("Llama-3.1", "Llama 3.1"),
    ("Mistral-Small", "Mistral-Small 3.1"),
    ("mistral-small", "Mistral-Small 3.1"),
    ("Mixtral-8x7B", "Mixtral 8x7B"),
    ("DeepSeek-R1", "DeepSeek R1"),
    ("phi4-multimodal", "Phi-4 Multimodal"),
    ("BertForSequenceClassification", "BERT"),
    ("gpt_oss", "GPT-OSS 20B"),
    ("meta-llama", "Llama 3.1"),
]

# Serving API: human-readable descriptions for known test functions
_SERVE_DESC: dict[str, str] = {
    "test_config_file_loading": "Config file loading via CLI flag",
    "test_env_overrides_pdl": "Environment variable overrides (PDL)",
    "test_invalid_max_tokens": "Invalid `max_tokens` parameter",
    "test_invalid_temperature": "Invalid `temperature` parameter",
    "test_invalid_top_p": "Out-of-range `top_p` value",
    "test_empty_messages_array": "Empty messages array",
    "test_missing_message_role": "Missing `role` in message",
    "test_invalid_token_ids": "Invalid token IDs",
    "test_extremely_large_token_id": "Extremely large token ID",
    "test_server_stability_under_invalid_requests": "Server stability under invalid requests",
    "test_concurrent_invalid_requests": "Concurrent invalid requests",
    "test_mixed_valid_invalid_requests": "Mixed valid/invalid requests",
    "test_health_check_during_errors": "Health check during error conditions",
    "test_request_exceeds_context_length": "Request exceeds context length",
    "test_malformed_json_request": "Malformed JSON request",
    "test_missing_content_type_header": "Missing `Content-Type` header",
    "test_extremely_large_batch": "Extremely large batch size",
}

# LLM API contract: descriptions
_LLMAPI_DESC: dict[str, str] = {
    "test_llm_args_logging": "LLM args logging behavior",
    "test_llm_args_type_tensorrt": "TensorRT backend arg type validation",
    "test_llm_args_type_default": "Default backend arg type validation",
}

_CHECK = "✅"
_CROSS = "❌"
_DASH = "—"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ParsedEntry:
    """One test entry from the QA test list."""

    raw: str
    file: str  # e.g. "test_e2e.py"
    func: str  # e.g. "test_ptp_quickstart_advanced"
    bracket: str  # e.g. "Llama3.1-8B-BF16-..."


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_ENTRY_RE = re.compile(r"^(.+?)::([^\[]+)(?:\[(.+)\])?$")


def _parse_entry(line: str) -> ParsedEntry | None:
    m = _ENTRY_RE.match(line.strip())
    if not m:
        return None
    return ParsedEntry(
        raw=line.strip(),
        file=m.group(1),
        func=m.group(2).strip(),
        bracket=m.group(3) or "",
    )


def load_test_list(path: Path) -> dict[str, list[ParsedEntry]]:
    """Load the QA test list and group entries by file path."""
    groups: dict[str, list[ParsedEntry]] = defaultdict(list)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entry = _parse_entry(line)
        if entry:
            groups[entry.file].append(entry)
    return dict(groups)


def toplevel_test_functions(path: Path) -> list[str]:
    """Return top-level ``test_*`` function names in a Python source file."""
    if not path.exists():
        return []
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    return [
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
    ]


# ---------------------------------------------------------------------------
# Feature / scenario extraction
# ---------------------------------------------------------------------------


def _e2e_category(func: str) -> str:
    """Return category letter (A/B/C/D/—) for an E2E test function."""
    if _CAT_A.search(func):
        return "A"
    if _CAT_D.search(func):
        return "D"
    if _CAT_C.search(func):
        return "C"
    if _CAT_B.search(func):
        return "B"
    return "—"


def _covering_entries(item: _UnivItem, entries: list[ParsedEntry]) -> list[ParsedEntry]:
    """Return entries that match a universe item (by func or bracket pattern)."""
    results = []
    for e in entries:
        func_match = bool(item.func_pat) and bool(re.search(item.func_pat, e.func))
        bk_match = bool(item.bracket_pat) and bool(re.search(item.bracket_pat, e.bracket))
        if func_match or bk_match:
            results.append(e)
    return results


_NVBUGS_RE = re.compile(r"nvbugs?(?:pro\.nvidia\.com/bug)?/(\d+)", re.IGNORECASE)


def extract_rcca_bugs(source: Path) -> list[tuple[str, str]]:
    """Return list of (bug_id, func_name) from nvbugs references in source.

    Handles both ``nvbugs/NNNNN`` and ``nvbugspro.nvidia.com/bug/NNNNN`` URLs.
    Uses line-number matching to associate each reference with its enclosing
    test function (works even for class-based test methods and decorated funcs).
    """
    if not source.exists():
        return []
    text = source.read_text()
    # Collect all nvbugs references with line numbers (1-indexed)
    bug_lines: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), 1):
        for bug_id in _NVBUGS_RE.findall(line):
            bug_lines.append((i, bug_id))
    if not bug_lines:
        return []
    # Build function ranges via AST (ast.walk finds nested methods too)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    func_ranges: list[tuple[int, int, str]] = []  # (start_line, end_line, name)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") and hasattr(node, "end_lineno"):
                func_ranges.append((node.lineno, node.end_lineno, node.name))
    # Match each bug to its innermost enclosing function
    results: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for bug_line, bug_id in bug_lines:
        # Find the innermost function containing this line
        best: tuple[str, int] | None = None  # (name, start_line)
        for start, end, name in func_ranges:
            if start <= bug_line <= end:
                if best is None or start > best[1]:
                    best = (name, start)
        if best is not None:
            key = (bug_id, best[0])
            if key not in seen:
                seen.add(key)
                results.append((bug_id, best[0]))
    return results


def models_from_bracket(bracket: str) -> list[str]:
    """Extract known model display names from a bracket parameter string."""
    found: list[str] = []
    for kw, display in _MODEL_KW:
        if kw.lower() in bracket.lower() and display not in found:
            found.append(display)
    return found


def disagg_features_from_func(func: str) -> list[str]:
    """Extract disaggregated feature tags from a function name."""
    return [feat for pat, feat in DISAGG_FEATURE_PATTERNS if re.search(pat, func)]


def disagg_routing_from_bracket(bracket: str) -> list[str]:
    """Detect routing strategies mentioned in bracket params."""
    return [s for s in _ROUTING_STRATEGIES if s in bracket]


def disagg_sd_from_bracket(bracket: str) -> list[str]:
    """Detect service-discovery backends mentioned in bracket params."""
    return [s for s in _SERVICE_DISCOVERY if s in bracket]


def disagg_model_from_bracket(bracket: str) -> str:
    """Best-effort model name extraction from a disaggregated test bracket.

    Returns an empty string for routing/service-discovery-only brackets
    (e.g. ``etcd-round_robin``) that contain no model information.
    """
    for kw, display in _MODEL_KW:
        if kw.lower() in bracket.lower():
            return display
    # Routing/service-discovery brackets contain no model info — skip them
    routing_terms = _ROUTING_STRATEGIES + _SERVICE_DISCOVERY
    if any(t in bracket for t in routing_terms):
        return ""
    return bracket[:60] if bracket else "TinyLlama 1.1B"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _render_summary(
    e2e: list[ParsedEntry],
    disagg: list[ParsedEntry],
    serve: list[ParsedEntry],
    llmapi: list[ParsedEntry],
    other_count: int,
) -> list[str]:
    total = len(e2e) + len(disagg) + len(serve) + len(llmapi) + other_count
    return [
        "## Summary",
        "",
        "| Category | Tests in QA | Source Files |",
        "|---|---:|---|",
        f"| E2E Workflow | {len(e2e)} | `test_e2e.py` |",
        f"| Disaggregated Serving Infra | {len(disagg)} | `disaggregated/test_*.py` |",
        f"| Serving API | {len(serve)} | `examples/serve/test_serve*.py` |",
        f"| LLM API Contract | {len(llmapi)} | `llmapi/test_llm_api_qa.py` |",
        f"| Other | {other_count} | misc |",
        f"| **Total** | **{total}** | |",
        "",
    ]


def _render_e2e(entries: list[ParsedEntry], source: Path) -> list[str]:
    lines: list[str] = [
        "## E2E Workflow Coverage (`test_e2e.py`)",
        "",
        "> Tests validate end-to-end scenarios: TRT/PyTorch backends, OpenAI API,"
        " benchmark tooling, and RCCA bug regression.",
        "",
    ]

    # Classify into A/B/C/D
    by_cat: dict[str, list[ParsedEntry]] = {"A": [], "B": [], "C": [], "D": [], "?": []}
    for e in entries:
        by_cat[_e2e_category(e.func)].append(e)

    cat_labels = {
        "A": "TRT Backend E2E (Legacy)",
        "B": "PyTorch Backend Quickstart",
        "C": "OpenAI API / trtllm-serve",
        "D": "trtllm-bench",
    }

    # Category summary
    lines += [
        "### Category Summary",
        "",
        "| Category | Description | Tests in QA |",
        "|---|---|---:|",
    ]
    for cat, label in cat_labels.items():
        n = len(by_cat[cat])
        lines.append(f"| {cat} | {label} | {n} |")
    total = sum(len(by_cat[c]) for c in cat_labels)
    lines += [f"| — | **Total** | **{total}** |", ""]

    # --- Category A ---
    cat_a = by_cat["A"]
    lines += [
        "---",
        "",
        f"### A. {cat_labels['A']} — {len(cat_a)} tests",
        "",
        "> Validates the legacy TRT engine backend for key models.",
        "",
        "| Test ID | Params |",
        "|---|---|",
    ]
    for e in cat_a:
        bracket_disp = f"[{e.bracket}]" if e.bracket else ""
        lines.append(f"| `{e.func}{bracket_disp}` | {e.bracket or _DASH} |")
    lines.append("")

    # --- Category B ---
    cat_b = by_cat["B"]
    lines += [
        "---",
        "",
        f"### B. {cat_labels['B']} — {len(cat_b)} tests",
        "",
        "> Universe: 7 documented deployment guides + 16 LLM API example scripts.",
        "",
        "#### B.1 Deployment Guide Coverage",
        "",
        "> Checked against all E2E entries (any category) since guides describe model-level deployment.",
        "",
        "| Model Guide | Doc | Status | Covered By |",
        "|---|---|:---:|---|",
    ]
    for item in _DEPLOYMENT_GUIDES:
        covering = _covering_entries(item, entries)  # all e2e entries
        if covering:
            first = covering[0]
            cat = _e2e_category(first.func)
            covered_by = f"`{first.func}` (Cat {cat})"
            if len(covering) > 1:
                covered_by += f" +{len(covering) - 1}"
            lines.append(f"| {item.name} | `{item.doc_ref}` | {_CHECK} | {covered_by} |")
        else:
            lines.append(f"| {item.name} | `{item.doc_ref}` | {_CROSS} | {_DASH} |")
    lines.append("")

    lines += [
        "#### B.2 LLM API Example Script Coverage",
        "",
        "| Script | Status | Covered By (func) |",
        "|---|:---:|---|",
    ]
    for item in _LLM_API_EXAMPLES:
        covering = _covering_entries(item, cat_b)
        if covering:
            funcs = sorted({e.func for e in covering})
            covered_by = "`" + "`, `".join(funcs[:2]) + "`"
            if len(funcs) > 2:
                covered_by += f" +{len(funcs) - 2}"
            lines.append(f"| `{item.name}` | {_CHECK} | {covered_by} |")
        else:
            lines.append(f"| `{item.name}` | {_CROSS} | {_DASH} |")
    lines.append("")

    lines += ["#### B.3 QA Test Cases", "", "| Test ID | Models / Params |", "|---|---|"]
    for e in cat_b:
        bracket_disp = f"[{e.bracket}]" if e.bracket else ""
        mods = models_from_bracket(e.bracket)
        mod_str = ", ".join(mods) if mods else (e.bracket[:60] if e.bracket else _DASH)
        lines.append(f"| `{e.func}{bracket_disp}` | {mod_str} |")
    lines.append("")

    # --- Category C ---
    cat_c = by_cat["C"]
    lines += [
        "---",
        "",
        f"### C. {cat_labels['C']} — {len(cat_c)} tests",
        "",
        "> Universe: documented OpenAI-compatible endpoints and trtllm-serve features.",
        "",
        "#### C.1 Feature Coverage",
        "",
        "| Feature | Doc | Status | Covered By |",
        "|---|---|:---:|---|",
    ]
    for item in _SERVE_FEATURES:
        covering = _covering_entries(item, cat_c)
        if covering:
            covered_by = "`" + "`, `".join(sorted({e.func for e in covering})[:2]) + "`"
            lines.append(f"| {item.name} | `{item.doc_ref}` | {_CHECK} | {covered_by} |")
        else:
            lines.append(f"| {item.name} | `{item.doc_ref}` | {_CROSS} | {_DASH} |")
    lines.append("")

    lines += ["#### C.2 QA Test Cases", "", "| Test ID | Params |", "|---|---|"]
    for e in cat_c:
        bracket_disp = f"[{e.bracket}]" if e.bracket else ""
        lines.append(f"| `{e.func}{bracket_disp}` | {e.bracket or _DASH} |")
    lines.append("")

    # --- Category D ---
    cat_d = by_cat["D"]
    lines += [
        "---",
        "",
        f"### D. {cat_labels['D']} — {len(cat_d)} {'test' if len(cat_d) == 1 else 'tests'}",
        "",
        "> Universe: documented trtllm-bench subcommands and modes.",
        "",
        "#### D.1 Subcommand Coverage",
        "",
        "| Subcommand / Mode | Doc | Status | Covered By |",
        "|---|---|:---:|---|",
    ]
    for item in _BENCH_FEATURES:
        covering = _covering_entries(item, cat_d)
        if covering:
            covered_by = "`" + "`, `".join(sorted({e.func for e in covering})[:2]) + "`"
            lines.append(f"| {item.name} | `{item.doc_ref}` | {_CHECK} | {covered_by} |")
        else:
            lines.append(f"| {item.name} | `{item.doc_ref}` | {_CROSS} | {_DASH} |")
    lines.append("")

    if cat_d:
        lines += ["#### D.2 QA Test Cases", "", "| Test ID | Params |", "|---|---|"]
        for e in cat_d:
            bracket_disp = f"[{e.bracket}]" if e.bracket else ""
            lines.append(f"| `{e.func}{bracket_disp}` | {e.bracket or _DASH} |")
        lines.append("")

    # --- Category E: RCCA ---
    lines += [
        "---",
        "",
        "### E. RCCA Bug Regression (cross-cutting)",
        "",
        "> Tests tagged with `nvbugs/` references in source file docstrings.",
        "",
        "| Bug ID | Function | Category |",
        "|---|---|:---:|",
    ]
    in_qa_funcs = {e.func for e in entries}
    rcca_bugs = [
        (bug_id, func) for bug_id, func in extract_rcca_bugs(source) if func in in_qa_funcs
    ]
    if rcca_bugs:
        for bug_id, func in rcca_bugs:
            cat = _e2e_category(func)
            lines.append(f"| `{bug_id}` | `{func}` | {cat} |")
    else:
        lines.append("| — | _No RCCA-tagged tests in QA list_ | — |")
    lines.append("")

    return lines


def _render_disagg(entries: list[ParsedEntry], sources: list[Path]) -> list[str]:
    lines: list[str] = [
        "## Disaggregated Serving Infrastructure Coverage",
        "",
        "> Tests validate transport backends, routing strategies, GPU topologies,"
        " worker management, and system reliability — **not** model output accuracy.",
        "",
        "| Test Function | Model | Features |",
        "|---|---|---|",
    ]

    for e in entries:
        feats: list[str] = []
        feats += disagg_features_from_func(e.func)
        feats += disagg_routing_from_bracket(e.bracket)
        feats += disagg_sd_from_bracket(e.bracket)
        feat_str = ", ".join(feats) if feats else _DASH
        model = disagg_model_from_bracket(e.bracket) or _DASH
        bracket_disp = f"[{e.bracket}]" if e.bracket else ""
        lines.append(f"| `{e.func}{bracket_disp}` | {model} | {feat_str} |")
    lines.append("")

    return lines


def _render_serve(entries: list[ParsedEntry], sources: list[Path]) -> list[str]:
    lines: list[str] = [
        "## Serving API Coverage (`examples/serve/`)",
        "",
        "> Tests validate `trtllm-serve` API correctness (positive) and"
        " robustness against invalid inputs (negative).",
        "",
    ]

    positive = [e for e in entries if "negative" not in e.file]
    negative = [e for e in entries if "negative" in e.file]

    lines += [
        f"- Positive tests: **{len(positive)}**",
        f"- Negative/robustness tests: **{len(negative)}**",
        "",
    ]

    if positive:
        lines += [
            "### Positive Tests",
            "",
            "| Test ID | Description |",
            "|---|---|",
        ]
        for e in positive:
            bracket_disp = f"[{e.bracket}]" if e.bracket else ""
            desc = _SERVE_DESC.get(e.func, _DASH)
            lines.append(f"| `{e.func}{bracket_disp}` | {desc} |")
        lines.append("")

    if negative:
        lines += [
            "### Negative Tests (API Robustness)",
            "",
            "| Test ID | Scenario |",
            "|---|---|",
        ]
        for e in negative:
            bracket_disp = f"[{e.bracket}]" if e.bracket else ""
            desc = _SERVE_DESC.get(e.func, _DASH)
            lines.append(f"| `{e.func}{bracket_disp}` | {desc} |")
        lines.append("")

    # Gap analysis
    all_defined: set[str] = set()
    for sp in sources:
        all_defined.update(toplevel_test_functions(sp))
    in_qa = {e.func for e in entries}
    gaps = sorted(all_defined - in_qa)
    if gaps:
        lines += [
            "### Gaps (defined in source, not in QA list)",
            "",
            "| Test Function |",
            "|---|",
        ]
        for f in gaps:
            lines.append(f"| `{f}` |")
        lines.append("")

    return lines


def _render_llmapi(entries: list[ParsedEntry]) -> list[str]:
    lines: list[str] = [
        "## LLM API Contract Coverage (`llmapi/test_llm_api_qa.py`)",
        "",
        "> Tests validate that `LLM(backend=...)` argument types and logging"
        " behaviour match the expected contract.",
        "",
        "| Test ID | Description |",
        "|---|---|",
    ]
    seen: set[str] = set()
    for e in entries:
        key = e.func
        if key in seen:
            continue
        seen.add(key)
        bracket_disp = f"[{e.bracket}]" if e.bracket else ""
        # func may be "TestClass::method" for class-based entries
        method = e.func.split("::")[-1]
        desc = _LLMAPI_DESC.get(method, _DASH)
        lines.append(f"| `{e.func}{bracket_disp}` | {desc} |")
    lines.append("")
    return lines


def _render_other(entries_by_file: dict[str, list[ParsedEntry]]) -> list[str]:
    lines: list[str] = []
    for file, entries in sorted(entries_by_file.items()):
        lines += [f"### `{file}` ({len(entries)} tests)", ""]
        for e in entries:
            bracket_disp = f"[{e.bracket}]" if e.bracket else ""
            lines.append(f"- `{e.func}{bracket_disp}`")
        lines.append("")
    if lines:
        lines = ["## Other Tests", ""] + lines
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-list", default=DEFAULT_TEST_LIST)
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    test_list_path = (
        Path(args.test_list) if Path(args.test_list).is_absolute() else repo_root / args.test_list
    )
    output_dir = (
        Path(args.output_dir)
        if Path(args.output_dir).is_absolute()
        else repo_root / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and group test list
    entries_by_file = load_test_list(test_list_path)

    def _collect(prefix: str) -> list[ParsedEntry]:
        result: list[ParsedEntry] = []
        for k, v in entries_by_file.items():
            if k.startswith(prefix):
                result.extend(v)
        return result

    e2e_entries = entries_by_file.get("test_e2e.py", [])
    disagg_entries = _collect("disaggregated/")
    serve_entries = _collect("examples/serve/")
    llmapi_entries = _collect("llmapi/test_llm_api_qa")

    other_entries_by_file = {
        k: v
        for k, v in entries_by_file.items()
        if not k.startswith("accuracy/")
        and k != "test_e2e.py"
        and not k.startswith("disaggregated/")
        and not k.startswith("examples/serve/")
        and not k.startswith("llmapi/test_llm_api_qa")
    }
    other_count = sum(len(v) for v in other_entries_by_file.values())

    # Source paths
    e2e_source = repo_root / E2E_SOURCE
    disagg_sources = [repo_root / p for p in DISAGG_SOURCES]
    serve_sources = [repo_root / p for p in SERVE_SOURCES]

    # Build report
    lines: list[str] = [
        "# QA System Coverage Report",
        "",
        "System-level test coverage for non-accuracy tests in `llm_function_core.txt`.",
        "",
    ]
    lines += _render_summary(
        e2e_entries, disagg_entries, serve_entries, llmapi_entries, other_count
    )
    lines += _render_e2e(e2e_entries, e2e_source)
    lines += _render_disagg(disagg_entries, disagg_sources)
    lines += _render_serve(serve_entries, serve_sources)
    lines += _render_llmapi(llmapi_entries)
    lines += _render_other(other_entries_by_file)

    index_path = output_dir / "index.md"
    index_path.write_text("\n".join(lines) + "\n")
    print(f"Report written to: {index_path}")


if __name__ == "__main__":
    main()
