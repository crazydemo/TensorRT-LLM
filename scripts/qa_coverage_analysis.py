#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""QA Coverage Analysis - Full Report Generator.

Generates:
  <output_dir>/index.md              - Summary table (all models, linked)
  <output_dir>/models/TestXxx.md     - Per-model detail (data type x feature matrix)

Usage:
  python scripts/qa_coverage_analysis.py \\
      --test-list tests/integration/test_lists/qa/llm_function_core.txt \\
      --output-dir coverage_report
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Feature dimensions aligned to docs/source/features/feature-combination-matrix.md
# Ordered to match the matrix columns.
# ---------------------------------------------------------------------------
FEATURE_DIMS = [
    "Overlap Scheduler",
    "CUDA Graph",
    "Tensor Parallelism",
    "Pipeline Parallelism",
    "Expert Parallelism",
    "Attention Data Parallelism",
    "Disaggregated Serving",
    "Chunked Prefill",
    "MTP",
    "EAGLE-3",
    "Ngram",
    "PARD / Other Spec",
    "Torch Compile",
    "Guided Decoding",
    "Beam Search",
    "KV Cache Reuse",
    "KV Cache Reuse OFF",
    "FP8 KV Cache",
    "EPLB (Static)",
    "EPLB (Online)",
    "Batch Wait",
    "CuteDSL Backend",
]

# ---------------------------------------------------------------------------
# Benchmark datasets used in accuracy tests.
# Maps class name (as it appears in test source) → display name.
# ---------------------------------------------------------------------------
BENCHMARK_CLASSES: dict[str, str] = {
    "CnnDailymail": "CNN/DailyMail",
    "MMLU": "MMLU",
    "GSM8K": "GSM8K",
    "GPQADiamond": "GPQA Diamond",
    "JsonModeEval": "JsonMode",
    "MMMU": "MMMU",
    "LongBenchV1": "LongBench V1",
    "LongBenchV2": "LongBench V2",
    "Humaneval": "HumanEval",
    "ZeroScrolls": "ZeroScrolls",
    "SlimPajama6B": "SlimPajama-6B",
    "PassKeyRetrieval64k": "PassKey-64k",
    "PassKeyRetrieval128k": "PassKey-128k",
}

_BENCHMARK_PAT = re.compile(r"\b(" + "|".join(re.escape(k) for k in BENCHMARK_CLASSES) + r")\b")


def benchmarks_from_body(body: str) -> set[str]:
    """Return display names of benchmark datasets instantiated in a method body."""
    return {BENCHMARK_CLASSES[m.group(1)] for m in _BENCHMARK_PAT.finditer(body)}


# ---------------------------------------------------------------------------
# Feature defaults: which features are ON by default in TorchLlmArgs.
# Grounded in tensorrt_llm/llmapi/llm_args.py:
#   - cuda_graph_config   defaults to CudaGraphConfig()   → CUDA Graph ON
#   - disable_overlap_scheduler defaults to False         → Overlap Scheduler ON
#   - kv_cache_config.enable_block_reuse defaults to True → KV Cache Reuse ON
# All other features default to OFF (require explicit configuration).
# ---------------------------------------------------------------------------
FEATURE_DEFAULTS: dict[str, bool] = {
    "CUDA Graph": True,
    "Overlap Scheduler": True,
    "KV Cache Reuse": True,
    # Everything else defaults to OFF
    "Chunked Prefill": False,
    "MTP": False,
    "EAGLE-3": False,
    "Ngram": False,
    "PARD / Other Spec": False,
    "Attention Data Parallelism": False,
    "Tensor Parallelism": False,
    "Pipeline Parallelism": False,
    "Expert Parallelism": False,
    "Disaggregated Serving": False,
    "FP8 KV Cache": False,
    "KV Cache Reuse OFF": False,
    "Beam Search": False,
    "Guided Decoding": False,
    "Torch Compile": False,
    "EPLB (Static)": False,
    "EPLB (Online)": False,
    "Batch Wait": False,
    "CuteDSL Backend": False,
}

# Patterns that indicate a default-ON feature has been explicitly disabled.
# If any of these match in a method body, that feature is NOT default-covered.
EXPLICIT_DISABLE_PATTERNS: dict[str, list[str]] = {
    "CUDA Graph": [
        r"cuda_graph_config\s*=\s*None",
    ],
    "Overlap Scheduler": [
        r"disable_overlap_scheduler\s*=\s*True",
    ],
    "KV Cache Reuse": [
        r"enable_block_reuse\s*=\s*False",
    ],
}

# Compiled disable patterns
_COMPILED_DISABLE_PATTERNS: dict[str, list[re.Pattern]] = {
    feat: [re.compile(p) for p in pats] for feat, pats in EXPLICIT_DISABLE_PATTERNS.items()
}

# ---------------------------------------------------------------------------
# Architecture-level coverage from docs/source/models/supported-models.md
# ---------------------------------------------------------------------------

# Maps HF MODEL_NAME prefix → architecture name in supported-models.md.
# Ordered most-specific first; first match wins.
MODEL_NAME_TO_ARCH: list[tuple[str, str]] = [
    # DeepSeek
    ("deepseek-ai/DeepSeek-V3.2", "DeepseekV32ForCausalLM"),
    ("deepseek-ai/DeepSeek-V3", "DeepseekV3ForCausalLM"),  # V3-Lite, V3-0324
    ("deepseek-ai/DeepSeek-R1", "DeepseekV3ForCausalLM"),  # R1 uses V3 arch
    ("DeepSeek-R1", "DeepseekV3ForCausalLM"),  # no-org prefix
    # Llama
    ("meta-llama/Llama-4", "Llama4ForConditionalGeneration"),
    ("meta-llama/Llama-3", "LlamaForCausalLM"),
    # Mistral / Mixtral — multimodal (Small 3.1, Large 3) before text-gen fallback
    ("mistralai/Mistral-Small-3.1-", "Mistral3ForConditionalGeneration"),
    ("mistral/Mistral-Large-3-", "Mistral3ForConditionalGeneration"),
    ("mistralai/Mixtral", "MixtralForCausalLM"),
    ("mistralai/", "MistralForCausalLM"),
    ("mistral/", "MistralForCausalLM"),
    # Gemma — multimodal (larger models with vision) before text-gen fallback
    ("google/gemma-3-27b", "Gemma3ForConditionalGeneration"),
    ("google/gemma-3-12b", "Gemma3ForConditionalGeneration"),
    ("google/gemma-3", "Gemma3ForCausalLM"),
    # GLM
    ("zai-org/GLM-4", "Glm4MoeForCausalLM"),
    # GPT-OSS
    ("openai/gpt-oss", "GptOssForCausalLM"),
    ("gpt_oss/gpt-oss", "GptOssForCausalLM"),  # MODEL_PATH fallback
    # MiniMax
    ("MiniMaxAI/MiniMax-M2", "MiniMaxM2ForCausalLM"),
    # Nemotron-H (separate arch from regular Nemotron)
    ("nvidia/Nemotron-H", "NemotronHForCausalLM"),
    # NemotronNAS (large Llama-based Nemotrons)
    ("nvidia/Llama-3_3-Nemotron-Super", "NemotronNASForCausalLM"),
    ("nvidia/Llama-3_1-Nemotron-Ultra", "NemotronNASForCausalLM"),
    ("nvidia/Llama-3.1-Nemotron-Nano", "NemotronNASForCausalLM"),
    ("nvidia/Nemotron-Super-V3", "NemotronNASForCausalLM"),
    # DeciLM (51B Nemotron)
    ("nvidia/Llama-3_1-Nemotron-51B", "DeciLMForCausalLM"),
    ("nemotron-nas/", "DeciLMForCausalLM"),
    # NemotronForCausalLM (Minitron, small Nemotrons)
    ("nvidia/Minitron", "NemotronForCausalLM"),
    ("nvidia/Nemotron-Mini", "NemotronForCausalLM"),
    ("nvidia/Nemotron-3-Nano", "NemotronForCausalLM"),
    ("nvidia/Nemotron-", "NemotronForCausalLM"),  # fallback
    # Phi — multimodal before text-gen
    ("microsoft/Phi-4-multimodal", "Phi4MMForCausalLM"),
    ("microsoft/Phi-4", "Phi3ForCausalLM"),
    ("microsoft/phi-4", "Phi3ForCausalLM"),
    # Qwen — multimodal VL before text-gen fallbacks
    ("Qwen/Qwen2-VL-", "Qwen2VLForConditionalGeneration"),
    ("Qwen/Qwen2.5-VL-", "Qwen2_5_VLForConditionalGeneration"),
    ("Qwen/Qwen3-VL-30B", "Qwen3VLMoeForConditionalGeneration"),
    ("Qwen/Qwen3-VL-", "Qwen3VLForConditionalGeneration"),
    ("Qwen/QwQ", "Qwen2ForCausalLM"),
    ("Qwen/Qwen2", "Qwen2ForCausalLM"),
    ("Qwen3/Qwen3-Next", "Qwen3NextForCausalLM"),
    ("Qwen3/Qwen3-30B-A3B", "Qwen3MoeForCausalLM"),
    ("Qwen3/Qwen3-235B", "Qwen3MoeForCausalLM"),
    ("Qwen3/Qwen3-", "Qwen3ForCausalLM"),
    # Llava / VILA
    ("llava-hf/", "LlavaNextForConditionalGeneration"),
    ("Efficient-Large-Model/NVILA", "LlavaLlamaModel (VILA)"),
    ("Efficient-Large-Model/VILA", "LlavaLlamaModel (VILA)"),
    # Nvidia VL models — before general Nemotron fallbacks
    ("nvidia/Nano-v2-VLM", "NemotronH_Nano_VL_V2"),
    ("nvidia/NVIDIA-Nemotron-Nano-", "NemotronH_Nano_VL_V2"),
    # EXAONE
    ("LGAI-EXAONE/EXAONE-4.0", "Exaone4ForCausalLM"),
    # Bielik is Llama-based
    ("speakleash/Bielik", "LlamaForCausalLM"),
]

# Columns in supported-models.md key model matrix → FEATURE_DIMS
# Both EAGLE-3 columns collapse to the same dim; "Yes" wins if either is Yes.
_MD_COL_TO_DIM: dict[str, str] = {
    # Text gen matrix columns
    "Overlap Scheduler": "Overlap Scheduler",
    "CUDA Graph": "CUDA Graph",
    "Attention Data Parallelism": "Attention Data Parallelism",
    "Disaggregated Serving": "Disaggregated Serving",
    "Chunked Prefill": "Chunked Prefill",
    "MTP": "MTP",
    "EAGLE-3(One Model Engine)": "EAGLE-3",
    "EAGLE-3(Two Model Engine)": "EAGLE-3",
    "Guided Decoding": "Guided Decoding",
    # Multimodal matrix columns (overlapping columns reuse same dim)
    "KV Cache Reuse": "KV Cache Reuse",
    "EPD Disaggregated Serving": "Disaggregated Serving",
}

# Feature dims for language-only key-model matrix (text gen matrix columns)
KEY_MODEL_FEAT_DIMS = [
    "Overlap Scheduler",
    "CUDA Graph",
    "Attention Data Parallelism",
    "Disaggregated Serving",
    "Chunked Prefill",
    "MTP",
    "EAGLE-3",
    "Guided Decoding",
]

# Feature dims for multimodal key-model matrix
MM_KEY_MODEL_FEAT_DIMS = [
    "Overlap Scheduler",
    "CUDA Graph",
    "Chunked Prefill",
    "KV Cache Reuse",
    "Disaggregated Serving",
]


@dataclass
class SupportedArch:
    arch: str  # e.g. "DeepseekV3ForCausalLM"
    model_display: str  # e.g. "DeepSeek-V3"
    hf_example: str  # e.g. "deepseek-ai/DeepSeek-V3"
    key_features: dict  # {feature_dim: "Yes"/"No"/"N/A"/"Untested"}
    is_multimodal: bool = False


def model_name_to_arch(model_name: str) -> str | None:
    """Map a HF MODEL_NAME to the architecture from supported-models.md."""
    for prefix, arch in MODEL_NAME_TO_ARCH:
        if model_name.startswith(prefix):
            return arch
    return None


def parse_supported_models(path: Path) -> list[SupportedArch]:
    """Parse supported-models.md.

    Returns all supported architectures with declared feature support for key models.
    """
    content = path.read_text()
    archs: dict[str, SupportedArch] = {}

    # Main table: | `ArchName` | Model display | `hf/example` |
    row_pat = re.compile(
        r"^\|\s*`([A-Za-z]\w*(?:ForCausalLM|ForConditionalGeneration|ForSequenceClassification))`"
        r"(?:\s*\[\^\d+\])?\s*\|"
        r"\s*([^|]+?)\s*\|"
        r"\s*`([^`]+)`\s*\|",
        re.MULTILINE,
    )
    for m in row_pat.finditer(content):
        arch, display, hf = m.group(1), m.group(2).strip(), m.group(3).strip()
        archs[arch] = SupportedArch(
            arch=arch, model_display=display, hf_example=hf, key_features={}
        )

    # Key model feature matrix section
    matrix_m = re.search(r"## Model-Feature Support Matrix.*?(?=\n#+ |\Z)", content, re.DOTALL)
    if matrix_m:
        table_lines = [
            ln.strip()
            for ln in matrix_m.group(0).splitlines()
            if ln.strip().startswith("|") and "---" not in ln
        ]
        if len(table_lines) >= 2:
            header_cells = [c.strip() for c in table_lines[0].split("|")[1:] if c.strip()]
            feat_cols = header_cells[1:]  # skip "Model Architecture/Feature"
            for line in table_lines[1:]:
                cells = [c.strip() for c in line.split("|")[1:] if c.strip()]
                if len(cells) < 2:
                    continue
                arch_name = re.sub(r"\s*\[\^\d+\]", "", cells[0]).strip().strip("`").strip()
                if arch_name not in archs:
                    continue
                for i, col in enumerate(feat_cols):
                    dim = _MD_COL_TO_DIM.get(col)
                    if not dim or i + 1 >= len(cells):
                        continue
                    val = re.sub(r"\[\^\d+\]", "", cells[i + 1]).strip()
                    # Keep "Yes" if already set (two EAGLE-3 cols → max)
                    if archs[arch_name].key_features.get(dim) != "Yes":
                        archs[arch_name].key_features[dim] = val

    # Multimodal Feature Support Matrix section (uses H1 heading in the doc)
    mm_matrix_m = re.search(
        r"# Multimodal Feature Support Matrix.*?(?=\n#+ |\Z)", content, re.DOTALL
    )
    if mm_matrix_m:
        table_lines = [
            ln.strip()
            for ln in mm_matrix_m.group(0).splitlines()
            if ln.strip().startswith("|") and "---" not in ln
        ]
        if len(table_lines) >= 2:
            header_cells = [c.strip() for c in table_lines[0].split("|")[1:] if c.strip()]
            feat_cols = header_cells[1:]  # skip "Model Architecture/Feature"
            for line in table_lines[1:]:
                cells = [c.strip() for c in line.split("|")[1:] if c.strip()]
                if len(cells) < 2:
                    continue
                # Arch name may include spaces/parens e.g. "LlavaLlamaModel (VILA)"
                arch_name = re.sub(r"\s*\[\^\d+\]", "", cells[0]).strip().strip("`").strip()
                if not arch_name:
                    continue
                # Create arch entry if not already discovered from main table
                if arch_name not in archs:
                    archs[arch_name] = SupportedArch(
                        arch=arch_name,
                        model_display=arch_name,
                        hf_example="",
                        key_features={},
                        is_multimodal=True,
                    )
                else:
                    archs[arch_name].is_multimodal = True
                for i, col in enumerate(feat_cols):
                    dim = _MD_COL_TO_DIM.get(col)
                    if not dim or i + 1 >= len(cells):
                        continue
                    val = re.sub(r"\[\^\d+\]", "", cells[i + 1]).strip()
                    if archs[arch_name].key_features.get(dim) != "Yes":
                        archs[arch_name].key_features[dim] = val

    return list(archs.values())


def parse_class_model_names(source_paths: list[Path]) -> dict[str, str]:
    """Extract MODEL_NAME (or MODEL_PATH as fallback) from each Test* class.

    Returns {cls: hf_model_name_or_path_key}.
    """
    result: dict[str, str] = {}
    model_name_pat = re.compile(r'MODEL_NAME\s*=\s*["\']([^"\']+)["\']')
    # MODEL_PATH fallback: extract the last two path components as a model id
    model_path_pat = re.compile(r'MODEL_PATH\s*=\s*f?"[^"]*?/([^/"]+/[^/"]+)"')
    class_pat = re.compile(r"^class (Test\w+)", re.MULTILINE)
    for path in source_paths:
        content = path.read_text()
        classes = list(class_pat.finditer(content))
        for i, m in enumerate(classes):
            cls_name = m.group(1)
            end = classes[i + 1].start() if i + 1 < len(classes) else len(content)
            chunk = content[m.start() : end][:2000]  # wider window
            mn = model_name_pat.search(chunk)
            if mn:
                result[cls_name] = mn.group(1)
            else:
                mp = model_path_pat.search(chunk)
                if mp:
                    result[cls_name] = mp.group(1)  # e.g. "gpt_oss/gpt-oss-120b"
    return result


# ---------------------------------------------------------------------------
# Mapping: quant type <- method name prefix (first match wins)
# ---------------------------------------------------------------------------
QUANT_FROM_METHOD: list[tuple[str, str]] = [
    ("cute_dsl_fp8", "FP8-BlockScales"),
    ("fp8_block_scales", "FP8-BlockScales"),
    ("fp8_blockscale", "FP8-BlockScales"),
    ("reasoning_fp8_prequantized", "FP8-PreQuant"),
    ("fp8_prequantized", "FP8-PreQuant"),
    ("fp8_vswa", "FP8"),
    ("fp8_guided", "FP8"),
    ("fp8_eagle3", "FP8"),
    ("fp8_beam", "FP8"),
    ("fp8_llm_sampler", "FP8"),
    ("fp8_chunked", "FP8"),
    ("fp8_tp", "FP8"),
    ("fp8_4gpus", "FP8"),
    ("fp8_8gpus", "FP8"),
    ("fp8", "FP8"),
    ("nvfp4", "NVFP4"),
    ("w4a8_mxfp4", "W4A8-MXFP4"),
    ("w4a16_mxfp4", "W4A16-MXFP4"),
    ("w4a16", "W4A16"),
    ("w4", "W4A16"),
    ("fp4", "FP4"),
    ("bfloat16", "BF16"),
    ("bf16", "BF16"),
    ("auto_dtype", "BF16"),
    ("auto_spec_decode", "BF16"),
    ("tp2", "BF16"),
    ("tp4", "BF16"),
    ("tp8", "BF16"),
]


def quant_from_method(method: str) -> str:
    low = method.lower()
    for prefix, q in QUANT_FROM_METHOD:
        if prefix in low:
            return q
    return "BF16"


# ---------------------------------------------------------------------------
# Feature extraction from bracket params & method name
# ---------------------------------------------------------------------------


def parse_bracket(bracket: str) -> dict:
    """Parse test-id bracket like: mtp_nextn=2-attention_dp=True-cuda_graph=True-...

    Returns {key: value} where value is bool/int/str.
    """
    params: dict = {}
    if not bracket:
        return params
    # Split on dash only when followed by a word character (key start)
    for token in re.split(r"-(?=[a-zA-Z_])", bracket):
        if "=" in token:
            k, v = token.split("=", 1)
            if v == "True":
                params[k] = True
            elif v == "False":
                params[k] = False
            elif re.fullmatch(r"-?\d+", v):
                params[k] = int(v)
            else:
                params[k] = v
        else:
            # Positional / flag-only tokens (e.g. "one_model", "overlap_scheduler")
            params[token] = True
    return params


def disabled_features_from_bracket(bracket: str) -> set[str]:
    """Return features that bracket params explicitly disable.

    For example, cuda_graph=False or overlap_scheduler=False in the bracket
    means that feature is explicitly turned off for this parametrize variant.
    """
    params = parse_bracket(bracket)
    disabled: set[str] = set()
    if params.get("cuda_graph") is False:
        disabled.add("CUDA Graph")
    if params.get("overlap_scheduler") is False:
        disabled.add("Overlap Scheduler")
    if params.get("disable_overlap_scheduler") is True:
        disabled.add("Overlap Scheduler")
    return disabled


def features_from_case(method: str, bracket: str) -> set[str]:
    """Derive the set of FEATURE_DIMS explicitly exercised by one test case."""
    params = parse_bracket(bracket)
    low_method = method.lower()
    low_bracket = bracket.lower()
    feats: set[str] = set()

    # --- params-driven ---
    if params.get("attention_dp") is True or "attention_dp=true" in low_bracket:
        feats.add("Attention Data Parallelism")
    if params.get("cuda_graph") is True or "cuda_graph=true" in low_bracket:
        feats.add("CUDA Graph")
    if params.get("overlap_scheduler") is True or "overlap_scheduler=true" in low_bracket:
        feats.add("Overlap Scheduler")
    if params.get("enable_chunked_prefill") is True or "enable_chunked_prefill=true" in low_bracket:
        feats.add("Chunked Prefill")
    if params.get("torch_compile") is True or "torch_compile=true" in low_bracket:
        feats.add("Torch Compile")
    if params.get("fp8kv") is True or "fp8kv=true" in low_bracket:
        feats.add("FP8 KV Cache")
    if params.get("disable_overlap_scheduler") is False:
        feats.add("Overlap Scheduler")

    mtp_nextn = params.get("mtp_nextn", 0)
    mtp_val = params.get("mtp", "disable")
    if (
        (isinstance(mtp_nextn, int) and mtp_nextn > 0)
        or (isinstance(mtp_val, str) and mtp_val != "disable")
        or "mtp" in low_bracket
    ):
        feats.add("MTP")

    if "eagle3_one_model" in params or "eagle3" in low_method:
        feats.add("EAGLE-3")

    moe = str(params.get("moe_backend", "")).upper()
    if moe == "CUTEDSL" or "cute_dsl" in low_method:
        feats.add("CuteDSL Backend")

    # batch wait
    bwt = params.get("batch_wait_timeout_iters", 0)
    bwr = params.get("batch_wait_max_tokens_ratio", 0)
    if (isinstance(bwt, int) and bwt > 0) or (isinstance(bwr, (int, float)) and bwr > 0):
        feats.add("Batch Wait")

    # --- method-name-driven ---
    if "guided_decoding" in low_method:
        feats.add("Guided Decoding")
    if "beam_search" in low_method:
        feats.add("Beam Search")
    if "ngram" in low_method:
        feats.add("Ngram")
    if "pard" in low_method:
        feats.add("PARD / Other Spec")
    if "chunked_prefill" in low_method:
        feats.add("Chunked Prefill")
    if "no_kv_cache_reuse" in low_method:
        feats.add("KV Cache Reuse OFF")
    if "static_eplb" in low_method:
        feats.add("EPLB (Static)")
    if "online_eplb" in low_method:
        feats.add("EPLB (Online)")
    if "nixl_backend" in low_method or "disagg" in low_method:
        feats.add("Disaggregated Serving")
    if "auto_dtype_with_helix" in low_method:
        feats.add("Disaggregated Serving")
    if "2_model_mtp" in low_method:
        feats.add("MTP")
    if "mtp_sa" in low_method:
        feats.add("MTP")
    if "cute_dsl" in low_method:
        feats.add("CuteDSL Backend")

    # TP / PP / EP from method name
    if re.search(r"\b(tp[2-9]|tp[1-9]\d|_2gpus|_4gpus|_8gpus|multi_gpu)", low_method):
        m_tp = re.search(r"tp(\d+)", low_method)
        if m_tp and int(m_tp.group(1)) > 1:
            feats.add("Tensor Parallelism")
    if re.search(r"pp[2-9]", low_method):
        feats.add("Pipeline Parallelism")
    if re.search(r"ep[2-9]", low_method):
        feats.add("Expert Parallelism")

    # TP/PP/EP from bracket
    for k in ("tp_size", "tp"):
        v = params.get(k)
        if isinstance(v, int) and v > 1:
            feats.add("Tensor Parallelism")
    for k in ("pp_size", "pp"):
        v = params.get(k)
        if isinstance(v, int) and v > 1:
            feats.add("Pipeline Parallelism")
    for k in ("ep_size", "ep"):
        v = params.get(k)
        if isinstance(v, int) and v > 1:
            feats.add("Expert Parallelism")

    return feats


def gpu_count_from_case(method: str, bracket: str) -> int:
    """Best-effort GPU count from method name and bracket."""
    low = method.lower()
    params = parse_bracket(bracket)
    tp = params.get("tp_size", 1) if isinstance(params.get("tp_size"), int) else 1
    pp = params.get("pp_size", 1) if isinstance(params.get("pp_size"), int) else 1
    ep = params.get("ep_size", 1) if isinstance(params.get("ep_size"), int) else 1
    from_params = max(tp * pp, ep)
    if from_params > 1:
        return from_params
    # from method name
    for pat, n in [
        ("_8gpus", 8),
        ("_4gpus", 4),
        ("_2gpus", 2),
        ("multi_gpu", 8),
        ("tp8", 8),
        ("tp4", 4),
        ("tp2", 2),
        ("pp4", 4),
        ("pp2", 2),
        ("ep4", 4),
        ("ep8", 8),
    ]:
        if pat in low:
            return n
    return 1


# ---------------------------------------------------------------------------
# Parse source file: {class_name: {method_name: MethodMeta}}
# ---------------------------------------------------------------------------


@dataclass
class MethodMeta:
    gpu_min: int = 1
    arch: str = "any"  # any / hopper / blackwell / gb200
    is_hopper_only: bool = False
    body_features: set = field(default_factory=set)  # explicitly configured features
    body_default_features: set = field(default_factory=set)  # default-ON features not disabled
    benchmarks: set = field(default_factory=set)  # benchmark datasets used


# ---------------------------------------------------------------------------
# Feature extraction from method body (static analysis of source code)
# Patterns detect feature-indicating constructs: class instantiations and
# hardcoded True flags. Variable-driven features are handled via bracket params.
# ---------------------------------------------------------------------------

# Each tuple: (regex_pattern, feature_name)
# Uses re.IGNORECASE where class names could vary.
BODY_FEATURE_PATTERNS: list[tuple[str, str]] = [
    # CUDA Graph: explicit CudaGraphConfig instantiation
    (r"CudaGraphConfig\s*\(", "CUDA Graph"),
    # Chunked Prefill: hardcoded
    (r"enable_chunked_prefill\s*=\s*True", "Chunked Prefill"),
    # EAGLE-3
    (r"Eagle3DecodingConfig\s*\(", "EAGLE-3"),
    (r"eagle3_one_model\s*=\s*(True|False)", "EAGLE-3"),  # parametrized but method always tests it
    # MTP
    (r"MTPDecodingConfig\s*\(", "MTP"),
    (r"MtpConfig\s*\(", "MTP"),  # alias fallback
    (r"\bmtp_nextn\s*=\s*[1-9]", "MTP"),  # hardcoded
    # Ngram speculative decoding
    (r"NGramDecodingConfig\s*\(", "Ngram"),
    (r"NgramDecodingConfig\s*\(", "Ngram"),  # alias fallback
    # PARD / other speculative decoding
    (r"PARDDecodingConfig\s*\(", "PARD / Other Spec"),
    (r"LookaheadDecodingConfig\s*\(", "PARD / Other Spec"),
    (r"MedusaDecodingConfig\s*\(", "PARD / Other Spec"),
    # Guided decoding
    (r"GuidedDecodingConfig\s*\(", "Guided Decoding"),
    (r"guided_decoding_backend\s*=", "Guided Decoding"),
    # Beam search
    (r"beam_width\s*=\s*[2-9]", "Beam Search"),
    # Torch compile: TorchCompileConfig instantiation or explicit arg
    (r"TorchCompileConfig\s*\(", "Torch Compile"),
    (r"torch_compile_config\s*=\s*(?!None)", "Torch Compile"),
    # KV cache reuse ON (block reuse enabled)
    (r"enable_block_reuse\s*=\s*True", "KV Cache Reuse"),
    # KV cache reuse OFF
    (r"enable_block_reuse\s*=\s*False", "KV Cache Reuse OFF"),
    # FP8 KV cache: dtype="fp8" in KvCacheConfig
    (r'KvCacheConfig\s*\([^)]*dtype\s*=\s*["\']fp8["\']', "FP8 KV Cache"),
    (r'kv_cache_dtype\s*=\s*["\']fp8["\']', "FP8 KV Cache"),
    # ADP
    (r"attention_dp_size\s*=\s*[2-9]", "Attention Data Parallelism"),
    (r"attention_dp\s*=\s*True", "Attention Data Parallelism"),
    # EPLB
    (r"StaticEplbConfig\s*\(", "EPLB (Static)"),
    (r"static_eplb\s*=\s*True", "EPLB (Static)"),
    (r"OnlineEplbConfig\s*\(", "EPLB (Online)"),
    (r"online_eplb\s*=\s*True", "EPLB (Online)"),
    # Batch wait
    (r"batch_wait_timeout_iters\s*=\s*[1-9]", "Batch Wait"),
    # CuteDSL
    (r'moe_backend\s*=\s*["\']CuteDSL["\']', "CuteDSL Backend"),
    (r"MoeBackend\.CuteDSL", "CuteDSL Backend"),
    # Tensor / Pipeline / Expert parallelism (hardcoded, not from params)
    (r"tensor_parallel_size\s*=\s*[2-9]", "Tensor Parallelism"),
    (r"pipeline_parallel_size\s*=\s*[2-9]", "Pipeline Parallelism"),
    (r"expert_parallel_size\s*=\s*[2-9]", "Expert Parallelism"),
    # Disaggregated serving
    (r"DisaggregatedConfig\s*\(", "Disaggregated Serving"),
    (r"nixl_backend", "Disaggregated Serving"),
    # Overlap scheduler (explicit True — confirming or overriding default)
    (r"overlap_scheduler\s*=\s*True", "Overlap Scheduler"),
]

# Compiled patterns (compiled once at import time)
_COMPILED_BODY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat), feat) for pat, feat in BODY_FEATURE_PATTERNS
]


def features_from_body(body: str) -> set[str]:
    """Scan a method body for explicitly configured features.

    Returns features where the body instantiates a config or sets a param.
    """
    feats: set[str] = set()
    for pattern, feat in _COMPILED_BODY_PATTERNS:
        if pattern.search(body):
            feats.add(feat)
    return feats


def default_features_for_body(body: str) -> set[str]:
    """Return features that are ON by default (per FEATURE_DEFAULTS) and NOT explicitly disabled in this method body."""
    disabled: set[str] = set()
    for feat, patterns in _COMPILED_DISABLE_PATTERNS.items():
        for pat in patterns:
            if pat.search(body):
                disabled.add(feat)
                break
    return {f for f, is_default in FEATURE_DEFAULTS.items() if is_default and f not in disabled}


def parse_source(path: Path) -> dict[str, dict[str, MethodMeta]]:
    """Extract all Test* classes and their test_* methods from a source file.

    Also parse skip decorators to determine GPU count and arch requirements,
    and scan method bodies for hardcoded feature patterns.
    """
    content = path.read_text()
    result: dict[str, dict[str, MethodMeta]] = {}

    class_pat = re.compile(r"^class (Test\w+)\s*\(", re.MULTILINE)
    classes = [(m.group(1), m.start()) for m in class_pat.finditer(content)]

    # Method pattern: up to 25 lines of decorators before def
    method_pat = re.compile(
        r"((?:[ \t]*@[^\n]+\n){0,25})"  # decorators
        r"[ \t]{4}def (test_\w+)\s*\(",
        re.MULTILINE,
    )

    for i, (cls_name, start) in enumerate(classes):
        end = classes[i + 1][1] if i + 1 < len(classes) else len(content)
        chunk = content[start:end]
        result[cls_name] = {}

        method_matches = list(method_pat.finditer(chunk))
        for j, mm in enumerate(method_matches):
            decorators = mm.group(1)
            method_name = mm.group(2)
            meta = MethodMeta()

            # GPU count
            m_dev = re.search(r"skip_less_device\((\d+)\)", decorators)
            if m_dev:
                meta.gpu_min = int(m_dev.group(1))

            # Architecture
            if "skip_pre_blackwell" in decorators:
                meta.arch = "blackwell"
            elif "skip_device_not_contain" in decorators and "GB200" in decorators:
                meta.arch = "gb200"
            elif "skip_no_hopper" in decorators:
                meta.arch = "hopper"
                meta.is_hopper_only = True
            elif "skip_pre_hopper" in decorators:
                meta.arch = "hopper"
            # post_blackwell means "pre-Blackwell only" (Hopper)
            if "skip_post_blackwell" in decorators:
                meta.is_hopper_only = True

            # Body features: scan text from end of "def test_xxx(" to next method
            body_end = method_matches[j + 1].start() if j + 1 < len(method_matches) else len(chunk)
            body = chunk[mm.end() : body_end]
            meta.body_features = features_from_body(body)
            meta.body_default_features = default_features_for_body(body)
            meta.benchmarks = benchmarks_from_body(body)

            result[cls_name][method_name] = meta

    return result


# ---------------------------------------------------------------------------
# Parse test list
# ---------------------------------------------------------------------------


@dataclass
class TestEntry:
    raw: str
    file: str  # e.g. "accuracy/test_llm_api_pytorch.py"
    cls: str  # e.g. "TestDeepSeekV3Lite"
    method: str  # e.g. "test_bfloat16"
    bracket: str  # raw content inside [...]
    quant: str  # derived
    features: set[str] = field(default_factory=set)  # explicitly configured
    default_features: set[str] = field(default_factory=set)  # default-ON, not disabled
    benchmarks: set[str] = field(default_factory=set)  # benchmark datasets used
    gpu_count: int = 1


def parse_test_list(path: Path) -> list[TestEntry]:
    entries: list[TestEntry] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("::")
            if len(parts) < 3:
                continue
            test_file = parts[0]
            cls = parts[1]
            test_full = parts[2]

            # Separate method name from bracket
            m = re.match(r"(\w+)(?:\[(.+)\])?$", test_full)
            if not m:
                continue
            method = m.group(1)
            bracket = m.group(2) or ""

            quant = quant_from_method(method)
            feats = features_from_case(method, bracket)
            gpu_c = gpu_count_from_case(method, bracket)
            # Bracket-level disable: features explicitly turned off via param
            bracket_disabled = disabled_features_from_bracket(bracket)
            # Default features for parametrized entries (body defaults applied later)
            init_default_feats = {
                f for f, on in FEATURE_DEFAULTS.items() if on and f not in bracket_disabled
            }

            entries.append(
                TestEntry(
                    raw=line,
                    file=test_file,
                    cls=cls,
                    method=method,
                    bracket=bracket,
                    quant=quant,
                    features=feats,
                    default_features=init_default_feats,
                    gpu_count=gpu_c,
                )
            )
    return entries


# ---------------------------------------------------------------------------
# Coverage model per class
# ---------------------------------------------------------------------------


@dataclass
class ModelCoverage:
    cls_name: str
    # From source: {method: MethodMeta}
    universe: dict[str, MethodMeta]
    # Covered entries (may come from multiple source files in the test list)
    covered: list[TestEntry]

    @property
    def covered_method_names(self) -> set[str]:
        """All covered method names (may include methods from other source files)."""
        return {e.method for e in self.covered}

    @property
    def covered_in_universe(self) -> set[str]:
        """Covered methods that also exist in the source universe (avoids >100%)."""
        return self.covered_method_names & set(self.universe.keys())

    @property
    def missing_methods(self) -> list[str]:
        return [m for m in self.universe if m not in self.covered_in_universe]

    @property
    def covered_quants(self) -> set[str]:
        return {e.quant for e in self.covered}

    @property
    def all_quants(self) -> set[str]:
        return {e.quant for e in self.covered}

    @property
    def covered_features_explicit(self) -> set[str]:
        """Features explicitly configured in at least one covered test."""
        result: set[str] = set()
        for e in self.covered:
            result |= e.features
        return result

    @property
    def covered_features_default(self) -> set[str]:
        """Features covered by default (not explicitly configured in any test)."""
        result: set[str] = set()
        for e in self.covered:
            result |= e.default_features - e.features
        return result

    @property
    def covered_features(self) -> set[str]:
        """All active features (explicit + default)."""
        return self.covered_features_explicit | self.covered_features_default

    @property
    def coverage_pct(self) -> float:
        if not self.universe:
            return 100.0
        return len(self.covered_in_universe) / len(self.universe) * 100

    @property
    def covered_benchmarks(self) -> set[str]:
        """All benchmark datasets used across covered test cases."""
        result: set[str] = set()
        for e in self.covered:
            result |= e.benchmarks
        return result

    def key_feature_count(self, arch_info: "SupportedArch") -> tuple[int, int]:
        """Returns (n_covered, n_yes): Yes-features covered vs. total Yes-features declared."""
        yes_feats = [f for f, s in arch_info.key_features.items() if s == "Yes"]
        covered = self.covered_features
        return sum(1 for f in yes_feats if f in covered), len(yes_feats)


# ---------------------------------------------------------------------------
# Render per-model report
# ---------------------------------------------------------------------------

ARCH_LABEL = {
    "any": "Any",
    "hopper": "H100+",
    "blackwell": "B200+",
    "gb200": "GB200",
}


def _feat_for_quant(model: ModelCoverage, feat: str, quant: str) -> str:
    """Returns: ✅ explicitly covered · 🔵 default only · '' not covered/not applicable.

    Universe is QA test list entries only.
    """
    quant_entries = [e for e in model.covered if e.quant == quant]
    if not quant_entries:
        return ""

    for e in quant_entries:
        if feat in e.features:
            return "✅"

    for e in quant_entries:
        if feat in e.default_features:
            return "🔵"

    return ""


def render_model_report(
    model: ModelCoverage,
    arch_info: "SupportedArch | None" = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Coverage: `{model.cls_name}`\n")
    lines.append("[← Back to Index](../index.md)\n")

    # Feature coverage summary
    if arch_info and arch_info.key_features:
        n_cov, n_yes = model.key_feature_count(arch_info)
        pct = n_cov / n_yes * 100 if n_yes else 0
        lines.append(
            f"> **Feature Coverage:** {n_cov}/{n_yes} ({pct:.0f}%) — "
            f"declared-Yes features covered by QA test list (`{arch_info.arch}`)\n"
        )

    # Key model feature gaps — pinned to top
    if arch_info and arch_info.key_features:
        lines.append(f"## Key Model Feature Coverage (`{arch_info.arch}`)\n")
        lines.append("| Feature | Declared | Status |")
        lines.append("|---------|:--------:|:------:|")
        for feat, status in arch_info.key_features.items():
            if status == "Yes":
                if feat in model.covered_features_explicit:
                    lines.append(f"| {feat} | Yes | ✅ Explicitly covered |")
                elif feat in model.covered_features_default:
                    lines.append(f"| {feat} | Yes | 🔵 Default only |")
                else:
                    lines.append(f"| **{feat}** | Yes | ❌ Not covered |")
            elif status not in ("",):
                lines.append(f"| {feat} | {status} | — |")
        lines.append("")

    # --- Data types ---
    lines.append("## Data Types\n")
    all_q = sorted(model.all_quants)
    cov_q = model.covered_quants
    lines.append("| Data Type | Status |")
    lines.append("|-----------|:------:|")
    for q in all_q:
        status = "✅" if q in cov_q else "❌"
        lines.append(f"| {q} | {status} |")
    lines.append("")

    # --- GPU Count / Arch ---
    lines.append("## GPU Requirements (from source decorators)\n")
    lines.append("| Method | Min GPUs | Min Arch |")
    lines.append("|--------|:--------:|:--------:|")
    for method in sorted(set(e.method for e in model.covered)):
        meta = model.universe.get(method)
        if meta:
            arch_label = ARCH_LABEL.get(meta.arch, meta.arch)
            if meta.is_hopper_only:
                arch_label += " only"
            lines.append(f"| `{method}` | {meta.gpu_min} | {arch_label} |")
    lines.append("")

    # --- Feature x Data Type matrix ---
    lines.append("## Feature × Data Type Coverage Matrix\n")
    lines.append(
        "_✅ explicitly tested · 🔵 covered by default (not explicitly disabled) · "
        "blank = not covered_\n"
    )

    all_quants_sorted = sorted(model.all_quants)
    header = "| Feature |" + "".join(f" {q} |" for q in all_quants_sorted)
    sep = "|---------|" + "".join(" :---: |" for _ in all_quants_sorted)
    lines.append(header)
    lines.append(sep)

    for feat in FEATURE_DIMS:
        row = f"| {feat} |"
        has_content = False
        for q in all_quants_sorted:
            cell = _feat_for_quant(model, feat, q)
            row += f" {cell} |"
            if cell:
                has_content = True
        if has_content:
            lines.append(row)
    lines.append("")

    # --- Benchmarks summary ---
    all_benchmarks: set[str] = set()
    for e in model.covered:
        all_benchmarks |= e.benchmarks
    if all_benchmarks:
        lines.append("## Benchmarks\n")
        lines.append("_Benchmark datasets used across covered test cases for this model._\n")
        for b in sorted(all_benchmarks):
            lines.append(f"- {b}")
        lines.append("")

    # --- Covered cases detail ---
    lines.append("## Covered Test Cases\n")
    lines.append(
        "| Test ID | Data Type | GPUs | Benchmarks | Explicit Features | Default Features |"
    )
    lines.append("|---------|-----------|:----:|-----------|------------------|-----------------|")
    for e in sorted(model.covered, key=lambda x: (x.method, x.bracket)):
        bench_str = ", ".join(sorted(e.benchmarks)) if e.benchmarks else "—"
        explicit_str = ", ".join(sorted(e.features)) if e.features else "—"
        default_only = e.default_features - e.features
        default_str = ", ".join(sorted(default_only)) if default_only else "—"
        lines.append(
            f"| `{e.raw}` | {e.quant} | {e.gpu_count} | {bench_str} | {explicit_str} | {default_str} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Architecture-level coverage sections
# ---------------------------------------------------------------------------


def _build_arch_to_models(
    supported: list[SupportedArch],
    cls_model_names: dict[str, str],
    models_by_cls: dict[str, "ModelCoverage"],
) -> dict[str, list["ModelCoverage"]]:
    """Return {arch_name: [ModelCoverage, ...]} for models that match this arch."""
    arch_to_models: dict[str, list] = defaultdict(list)
    for cls_name, model_name in cls_model_names.items():
        arch = model_name_to_arch(model_name)
        if arch and cls_name in models_by_cls:
            arch_to_models[arch].append(models_by_cls[cls_name])
    return arch_to_models


def _render_arch_table(
    archs: list[SupportedArch],
    arch_to_models: dict[str, list],
) -> list[str]:
    """Render a single Architecture Coverage table for the given arch list."""
    lines: list[str] = []
    lines.append("| Architecture | Model | Test Classes | Feature Coverage | Benchmarks |")
    lines.append("|---|---|---|:---:|---|")
    for a in sorted(archs, key=lambda x: x.arch):
        ms = arch_to_models.get(a.arch, [])
        if not ms:
            lines.append(f"| 🔴 `{a.arch}` | {a.model_display} | — | — | — |")
        else:
            cls_links = " ".join(f"[{m.cls_name}](models/{m.cls_name}.md)" for m in ms)
            # Aggregate benchmarks across all test classes for this arch
            all_benchmarks: set[str] = set()
            for m in ms:
                all_benchmarks |= m.covered_benchmarks
            bench_str = ", ".join(sorted(all_benchmarks)) if all_benchmarks else "—"

            if a.key_features:
                yes_feats = [f for f, s in a.key_features.items() if s == "Yes"]
                all_covered: set[str] = set()
                for m in ms:
                    all_covered |= m.covered_features
                n_cov = sum(1 for f in yes_feats if f in all_covered)
                n_yes = len(yes_feats)
                if n_yes == 0:
                    pct_str = "N/A"
                    indicator = "🟡"
                elif n_cov == n_yes:
                    pct_str = f"{n_cov}/{n_yes} (100%)"
                    indicator = "🟢"
                else:
                    pct_str = f"{n_cov}/{n_yes} ({n_cov / n_yes * 100:.0f}%)"
                    indicator = "🟡"
            else:
                pct_str = "—"
                indicator = "🟢" if ms else "🔴"
            lines.append(
                f"| {indicator} `{a.arch}` | {a.model_display} "
                f"| {cls_links} | {pct_str} | {bench_str} |"
            )
    return lines


def _render_key_feature_table(
    archs: list[SupportedArch],
    arch_to_models: dict[str, list],
    feat_dims: list[str],
) -> list[str]:
    """Render a Key Model Feature Coverage matrix for the given archs and feature dims."""
    key_archs = [a for a in archs if a.key_features]
    if not key_archs:
        return []
    lines: list[str] = []
    header = "| Architecture |" + "".join(f" {f.replace(' ', '<br>')} |" for f in feat_dims)
    sep = "|---|" + "".join(" :---: |" for _ in feat_dims)
    lines.append(header)
    lines.append(sep)
    for a in sorted(key_archs, key=lambda x: x.arch):
        ms = arch_to_models.get(a.arch, [])
        all_explicit_feats: set[str] = set()
        all_default_feats: set[str] = set()
        for m in ms:
            all_explicit_feats |= m.covered_features_explicit
            all_default_feats |= m.covered_features_default
        row = f"| `{a.arch}` |"
        for feat in feat_dims:
            declared = a.key_features.get(feat, "")
            if declared == "Yes":
                if feat in all_explicit_feats:
                    cell = "Yes ✅"
                elif feat in all_default_feats:
                    cell = "Yes 🔵"
                else:
                    cell = "Yes ❌"
            elif declared in ("No", "N/A", "Untested"):
                cell = declared
            else:
                cell = "—"
            row += f" {cell} |"
        lines.append(row)
    return lines


_CELL_LEGEND = (
    "| Cell format | Meaning |\n"
    "|---|---|\n"
    "| `Yes ✅` | Declared supported and explicitly tested |\n"
    "| `Yes 🔵` | Declared supported, covered by default (not explicitly disabled) |\n"
    "| `Yes ❌` | Declared supported but **not covered** — QA gap |\n"
    "| `No` | Declared not supported (no test needed) |\n"
    "| `N/A` | Not applicable to this architecture |\n"
    "| `Untested` | Not yet validated by NVIDIA |\n"
)


def render_arch_coverage_sections(
    supported: list[SupportedArch],
    arch_to_models: dict[str, list],
) -> str:
    """Render four sections (language / multimodal separated).

    1. Language Model Architecture Coverage
    2. Language Model Key Feature Coverage
    3. Multimodal Architecture Coverage
    4. Multimodal Key Feature Coverage
    """
    lang_archs = [a for a in supported if not a.is_multimodal]
    mm_archs = [a for a in supported if a.is_multimodal]

    lines: list[str] = []

    # ---- Language: Architecture Coverage ----
    lang_covered = sum(1 for a in lang_archs if arch_to_models.get(a.arch))
    lines.append("## Language Model Architecture Coverage\n")
    lines.append(
        "_Coverage relative to `docs/source/models/supported-models.md`. "
        "🔴 = no test class, 🟡 = partial, 🟢 = all Yes features covered._\n"
    )
    lines.append(
        f"**{lang_covered} / {len(lang_archs)} architectures** have at least one QA test class.\n"
    )
    lines.extend(_render_arch_table(lang_archs, arch_to_models))
    lines.append("")

    # ---- Language: Key Model Feature Coverage ----
    lang_key = [a for a in lang_archs if a.key_features]
    if lang_key:
        lines.append("## Language Model Key Feature Coverage\n")
        lines.append(
            "_Cross-reference of declared feature support (from `supported-models.md`) "
            "against actual test coverage._\n"
        )
        lines.append(_CELL_LEGEND)
        lines.extend(_render_key_feature_table(lang_archs, arch_to_models, KEY_MODEL_FEAT_DIMS))
        lines.append("")

    # ---- Multimodal: Architecture Coverage ----
    if mm_archs:
        mm_covered = sum(1 for a in mm_archs if arch_to_models.get(a.arch))
        lines.append("## Multimodal Architecture Coverage\n")
        lines.append(
            "_Coverage relative to Multimodal Feature Support Matrix in `supported-models.md`. "
            "🔴 = no test class, 🟡 = partial, 🟢 = all Yes features covered._\n"
        )
        lines.append(
            f"**{mm_covered} / {len(mm_archs)} architectures** have at least one QA test class.\n"
        )
        lines.extend(_render_arch_table(mm_archs, arch_to_models))
        lines.append("")

    # ---- Multimodal: Key Model Feature Coverage ----
    mm_key = [a for a in mm_archs if a.key_features]
    if mm_key:
        lines.append("## Multimodal Key Feature Coverage\n")
        lines.append(
            "_Cross-reference of declared feature support (from multimodal matrix) "
            "against actual test coverage._\n"
        )
        lines.append(_CELL_LEGEND)
        lines.extend(_render_key_feature_table(mm_archs, arch_to_models, MM_KEY_MODEL_FEAT_DIMS))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Render index
# ---------------------------------------------------------------------------


def render_index(
    models: list[ModelCoverage],
    test_list: str,
    arch_sections: str = "",
    cls_to_arch_info: "dict[str, SupportedArch] | None" = None,
) -> str:
    lines: list[str] = []
    lines.append("# QA Coverage Report — Index\n")
    lines.append(f"**Test list:** `{test_list}`  \n")

    # Summary
    total_cases = sum(len(m.covered) for m in models)
    lines.append("## Summary\n")
    lines.append(f"**Total test cases in test list:** {total_cases}\n")

    # Architecture Coverage + Key Model Feature Coverage come FIRST
    if arch_sections:
        lines.append(arch_sections)

    # --- Model Coverage Table ---
    lines.append("## Model Coverage Table\n")
    lines.append(
        "| Model | Test Cases | Data Types Covered | Feature Coverage | Benchmarks | Key Features Covered |"
    )
    lines.append(
        "|-------|:----------:|-------------------|:---------------:|-----------|---------------------|"
    )

    # Sort: models with cases first (by name), then zero-coverage last
    def sort_key(m: ModelCoverage):
        return (0 if m.covered else 1, m.cls_name)

    arch_map = cls_to_arch_info or {}
    for m in sorted(models, key=sort_key):
        indicator = "🔴" if not m.covered else "🟢"
        test_cases = len(m.covered)
        quants = ", ".join(sorted(m.covered_quants)) or "—"
        feats = sorted(m.covered_features)
        feat_str = "<br>".join(feats) if feats else "—"
        benchmarks_str = ", ".join(sorted(m.covered_benchmarks)) if m.covered_benchmarks else "—"
        model_link = f"[{m.cls_name}](models/{m.cls_name}.md)"
        arch_info = arch_map.get(m.cls_name)
        if arch_info and arch_info.key_features:
            n_cov, n_yes = m.key_feature_count(arch_info)
            feat_cov = f"{n_cov}/{n_yes}" if n_yes else "—"
        else:
            feat_cov = "—"
        lines.append(
            f"| {indicator} {model_link} "
            f"| {test_cases} "
            f"| {quants} "
            f"| {feat_cov} "
            f"| {benchmarks_str} "
            f"| {feat_str} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA coverage report for TensorRT-LLM test lists"
    )
    parser.add_argument(
        "--test-list",
        default="tests/integration/test_lists/qa/llm_function_core.txt",
    )
    parser.add_argument(
        "--test-source",
        help="Primary test source file (defines universe of tests per model). "
        "May be specified multiple times.",
        action="append",
        dest="test_sources",
    )
    parser.add_argument(
        "--output-dir",
        default="coverage_report",
        help="Directory to write index.md and models/*.md",
    )
    parser.add_argument(
        "--supported-models",
        default="docs/source/models/supported-models.md",
        help="Path to supported-models.md for architecture-level coverage",
    )
    args = parser.parse_args()

    test_list_path = Path(args.test_list)
    output_dir = Path(args.output_dir)
    supported_models_path = Path(args.supported_models)
    sources = args.test_sources or [
        "tests/integration/defs/accuracy/test_llm_api_pytorch.py",
        "tests/integration/defs/accuracy/test_disaggregated_serving.py",
        "tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py",
    ]

    for p in [test_list_path] + [Path(s) for s in sources]:
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)

    # Parse all source files → merge universe
    universe: dict[str, dict[str, MethodMeta]] = {}
    for src in sources:
        src_path = Path(src)
        print(f"Parsing source: {src_path}", file=sys.stderr)
        parsed = parse_source(src_path)
        for cls, methods in parsed.items():
            if cls not in universe:
                universe[cls] = {}
            universe[cls].update(methods)

    # Parse test list
    print(f"Parsing test list: {test_list_path}", file=sys.stderr)
    entries = parse_test_list(test_list_path)

    # Group entries by class (only classes that exist in universe)
    entries_by_cls: dict[str, list[TestEntry]] = defaultdict(list)
    for e in entries:
        entries_by_cls[e.cls].append(e)

    # Enrich entries with body features from source (hardcoded features not visible
    # in method names or bracket params). Also refine default features by intersecting
    # with what the source body actually keeps enabled (not disabled in body).
    for cls_name, methods in universe.items():
        for e in entries_by_cls.get(cls_name, []):
            meta = methods.get(e.method)
            if meta:
                e.features |= meta.body_features
                e.default_features &= meta.body_default_features
                e.benchmarks |= meta.benchmarks

    # Build ModelCoverage only for classes that appear in the test list
    models: list[ModelCoverage] = []
    for cls_name, methods in sorted(universe.items()):
        covered = entries_by_cls.get(cls_name, [])
        if not covered:
            continue  # skip classes not in QA test list
        models.append(
            ModelCoverage(
                cls_name=cls_name,
                universe=methods,
                covered=covered,
            )
        )

    total_m = sum(len(m.universe) for m in models)
    total_c = sum(len(m.covered_in_universe) for m in models)
    print(
        f"Universe: {len(models)} classes, {total_m} methods. "
        f"Covered: {total_c} ({total_c / total_m * 100:.1f}%)",
        file=sys.stderr,
    )

    # Architecture-level coverage from supported-models.md
    arch_sections = ""
    if supported_models_path.exists():
        print(f"Parsing supported models: {supported_models_path}", file=sys.stderr)
        supported = parse_supported_models(supported_models_path)
        cls_model_names = parse_class_model_names([Path(s) for s in sources])
        models_by_cls = {m.cls_name: m for m in models}
        arch_to_models = _build_arch_to_models(supported, cls_model_names, models_by_cls)
        arch_sections = render_arch_coverage_sections(supported, arch_to_models)
        cov_archs = sum(1 for a in supported if arch_to_models.get(a.arch))
        print(
            f"Architecture coverage: {cov_archs}/{len(supported)} architectures",
            file=sys.stderr,
        )
        # Build cls → arch_info for key models (those with declared feature support)
        cls_to_arch_info = {}
        for a in supported:
            if a.key_features:
                for mc in arch_to_models.get(a.arch, []):
                    cls_to_arch_info[mc.cls_name] = a
    else:
        print(
            f"Warning: {supported_models_path} not found, skipping arch coverage",
            file=sys.stderr,
        )
        cls_to_arch_info = {}

    # Generate output
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale model reports from previous runs
    for stale in models_dir.glob("*.md"):
        stale.unlink()

    # Per-model reports
    for m in models:
        out = models_dir / f"{m.cls_name}.md"
        out.write_text(render_model_report(m, cls_to_arch_info.get(m.cls_name)))

    # Index
    index_out = output_dir / "index.md"
    index_out.write_text(render_index(models, args.test_list, arch_sections, cls_to_arch_info))

    print(f"Report written to: {output_dir}/", file=sys.stderr)
    print(f"  {index_out}", file=sys.stderr)
    print(f"  {models_dir}/*.md  ({len(models)} files)", file=sys.stderr)


if __name__ == "__main__":
    main()
