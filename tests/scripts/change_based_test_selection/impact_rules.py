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
Impact rules for change-based test selection.

Defines three classification systems:
  1. Feature Tiers: how broadly a code change affects tests
  2. Path-to-Feature Mapping: which source paths map to which features
  3. Test Class to Architecture Mapping: which test classes test which model code
  4. Representative Coverage Set: minimum test set for broad-impact changes
"""

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from typing import Optional

# ============================================================================
# Feature Tier Classification
# ============================================================================
#
# Tier 0 (Core Infrastructure): Always-on components. A change here could
#   affect ANY test. Strategy: run the Representative Coverage Set.
#
# Tier 1 (Default-On Configurable): Enabled by default but can be configured
#   per-test. Strategy: run tests that explicitly configure this feature
#   + Representative Coverage Set.
#
# Tier 2 (Opt-In): Only used when explicitly enabled via a Config class.
#   Strategy: run only tests that use this feature (via L3 Config class match).
#
# Model-Specific: Changes to model implementation code. Strategy: run all
#   tests for that model architecture.


class Tier(Enum):
    CORE = 0  # Always-on infrastructure
    DEFAULT_ON = 1  # Default-enabled, configurable
    OPT_IN = 2  # Explicit opt-in only
    MODEL = 3  # Model-specific code
    TEST = 4  # Test file itself changed
    IGNORE = 5  # No test impact (docs, CI config, etc.)


@dataclass
class ImpactRule:
    """A rule mapping source file patterns to test selection behavior."""

    pattern: str  # glob pattern for file paths
    tier: Tier
    feature: Optional[str] = None  # feature name for Tier 1/2
    config_class: Optional[str] = None  # Config class name for Tier 2
    arch: Optional[str] = None  # architecture for MODEL tier
    description: str = ""


# ============================================================================
# Path → Feature/Tier Mapping
# ============================================================================

IMPACT_RULES: list[ImpactRule] = [
    # ---- Tier 0: Core Infrastructure ----
    # Changes here can affect everything → run Representative Coverage Set
    ImpactRule(
        pattern="tensorrt_llm/llmapi/llm.py",
        tier=Tier.CORE,
        description="Main LLM API entry point",
    ),
    ImpactRule(
        pattern="tensorrt_llm/llmapi/llm_args.py",
        tier=Tier.CORE,
        description="Configuration schema (all Config classes defined here)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/llmapi/llm_utils.py",
        tier=Tier.CORE,
        description="Model loading and default overrides",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/py_executor.py",
        tier=Tier.CORE,
        description="PyTorch executor core",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/py_executor_creator.py",
        tier=Tier.CORE,
        description="Executor creation logic",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/model_engine.py",
        tier=Tier.CORE,
        description="Model engine abstraction",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/model_loader.py",
        tier=Tier.CORE,
        description="Model loading pipeline",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/sampler.py",
        tier=Tier.CORE,
        description="Sampling implementation (used by all tests)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/request_utils.py",
        tier=Tier.CORE,
        description="Request handling utilities",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/resource_manager.py",
        tier=Tier.CORE,
        description="Resource management (used by all tests)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/seq_slot_manager.py",
        tier=Tier.CORE,
        description="Sequence slot management",
    ),
    ImpactRule(
        pattern="tensorrt_llm/executor/executor.py",
        tier=Tier.CORE,
        description="Executor abstraction layer",
    ),
    ImpactRule(
        pattern="tensorrt_llm/mapping.py",
        tier=Tier.CORE,
        description="Parallelism mapping (TP/PP/EP)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/models/modeling_utils.py",
        tier=Tier.CORE,
        description="Model base classes",
    ),
    ImpactRule(
        pattern="tensorrt_llm/models/automodel.py",
        tier=Tier.CORE,
        description="Auto model discovery",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_utils.py",
        tier=Tier.CORE,
        description="PyTorch model base classes",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_auto.py",
        tier=Tier.CORE,
        description="PyTorch auto model resolution",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/_util.py",
        tier=Tier.CORE,
        description="PyExecutor internal utilities",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/make_decoding_batch_input_output.py",
        tier=Tier.CORE,
        description="Decoding batch I/O construction",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_utils.py",
        tier=Tier.CORE,
        description="Top-level utility functions",
    ),
    # C++ core that binds to Python
    ImpactRule(
        pattern="cpp/tensorrt_llm/batch_manager/*",
        tier=Tier.CORE,
        description="C++ batch manager (scheduling, KV cache mgmt)",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/executor/*",
        tier=Tier.CORE,
        description="C++ executor core",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/common/attentionOp.*",
        tier=Tier.DEFAULT_ON,
        feature="attention",
        description="C++ attention op implementation",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/thop/attentionOp.*",
        tier=Tier.DEFAULT_ON,
        feature="attention",
        description="C++ attention op THop binding",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/thop/trtllmGenQKVProcessOp.*",
        tier=Tier.DEFAULT_ON,
        feature="attention",
        description="C++ QKV processing op",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/kernels/gptKernels.*",
        tier=Tier.CORE,
        description="C++ GPT kernels (shared across models)",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.*",
        tier=Tier.OPT_IN,
        feature="moe",
        description="C++ MoE AllToAll communication kernels",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/thop/moeAlltoAllOp.*",
        tier=Tier.OPT_IN,
        feature="moe",
        description="C++ MoE AllToAll THop binding",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/kernels/cutlass_kernels/fp4_gemm/*",
        tier=Tier.OPT_IN,
        feature="nvfp4",
        description="C++ FP4 GEMM CUTLASS kernels",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/thop/fp4Gemm.*",
        tier=Tier.OPT_IN,
        feature="nvfp4",
        description="C++ FP4 GEMM THop binding",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/*",
        tier=Tier.OPT_IN,
        feature="quantization",
        description="C++ FP-A INT-B GEMM CUTLASS kernels",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/thop/CMakeLists.txt",
        tier=Tier.CORE,
        description="C++ THop build config (affects all C++ ops)",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/layers/samplingLayer.*",
        tier=Tier.CORE,
        description="C++ sampling layer",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/layers/dynamicDecodeLayer.*",
        tier=Tier.CORE,
        description="C++ dynamic decoding",
    ),
    # Test infrastructure
    ImpactRule(
        pattern="tests/integration/defs/accuracy/accuracy_core.py",
        tier=Tier.CORE,
        description="Accuracy test harness and task definitions",
    ),
    ImpactRule(
        pattern="tests/integration/defs/conftest.py",
        tier=Tier.CORE,
        description="Test fixtures and skip decorators",
    ),

    # ---- Tier 1: Default-On Configurable ----
    # KV Cache (always used, but config varies)
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/kv_cache_connector.py",
        tier=Tier.DEFAULT_ON,
        feature="kv_cache",
        config_class="KvCacheConfig",
        description="KV cache connector",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.*",
        tier=Tier.DEFAULT_ON,
        feature="kv_cache",
        config_class="KvCacheConfig",
        description="KV cache manager V2",
    ),
    # Scheduler (overlap scheduler is default-on)
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/scheduler/*",
        tier=Tier.DEFAULT_ON,
        feature="scheduler",
        config_class="SchedulerConfig",
        description="Scheduler (overlap scheduler default-on)",
    ),
    # Attention backends (always used, but which backend varies)
    ImpactRule(
        pattern="tensorrt_llm/_torch/attention_backend/interface.py",
        tier=Tier.DEFAULT_ON,
        feature="attention",
        description="Attention backend interface (affects all backends)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/attention_backend/trtllm.py",
        tier=Tier.DEFAULT_ON,
        feature="attn_trtllm",
        description="TRT-LLM native attention backend",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/attention_backend/trtllm_gen.py",
        tier=Tier.DEFAULT_ON,
        feature="attn_trtllm",
        description="TRT-LLM gen attention backend",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/attention_backend/flashinfer.py",
        tier=Tier.DEFAULT_ON,
        feature="attn_flashinfer",
        description="FlashInfer attention backend",
    ),
    # Modules (shared building blocks)
    ImpactRule(
        pattern="tensorrt_llm/_torch/modules/attention.py",
        tier=Tier.DEFAULT_ON,
        feature="attention",
        description="Core attention module",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/modules/linear.py",
        tier=Tier.DEFAULT_ON,
        feature="linear",
        description="Quantization-aware linear layers",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/modules/rms_norm.py",
        tier=Tier.DEFAULT_ON,
        feature="normalization",
        description="RMSNorm implementation",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/modules/rotary_embedding.py",
        tier=Tier.DEFAULT_ON,
        feature="rope",
        description="RoPE implementation",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/modules/embedding.py",
        tier=Tier.DEFAULT_ON,
        feature="embedding",
        description="Embedding layers",
    ),
    # Distributed (TP/PP basics are always-on for multi-GPU)
    ImpactRule(
        pattern="tensorrt_llm/_torch/distributed/ops.py",
        tier=Tier.DEFAULT_ON,
        feature="distributed",
        description="Distributed operations (allreduce etc.)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/distributed/communicator.py",
        tier=Tier.DEFAULT_ON,
        feature="distributed",
        description="Communication backend",
    ),

    # ---- Tier 2: Opt-In Features ----
    # Speculative decoding
    ImpactRule(
        pattern="tensorrt_llm/_torch/speculative/eagle3.py",
        tier=Tier.OPT_IN,
        feature="eagle3",
        config_class="Eagle3DecodingConfig",
        description="Eagle3 speculative decoding",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/speculative/mtp.py",
        tier=Tier.OPT_IN,
        feature="mtp",
        config_class="MTPDecodingConfig",
        description="Multi-Token Prediction (DeepSeek)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/speculative/ngram.py",
        tier=Tier.OPT_IN,
        feature="ngram",
        config_class="NGramDecodingConfig",
        description="N-gram speculative decoding",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/speculative/suffix_automaton.py",
        tier=Tier.OPT_IN,
        feature="suffix_automaton",
        config_class="SADecodingConfig",
        description="Suffix automaton decoding",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/speculative/interface.py",
        tier=Tier.OPT_IN,
        feature="speculative",
        description="Speculative decoding interface (affects all spec decoding)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/speculative/drafter.py",
        tier=Tier.OPT_IN,
        feature="speculative",
        description="Draft model infrastructure",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/speculative/drafting_loops.py",
        tier=Tier.OPT_IN,
        feature="speculative",
        description="Drafting loop logic",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_speculative.py",
        tier=Tier.OPT_IN,
        feature="speculative",
        description="Speculative model architectures",
    ),
    # C++ speculative decoding
    ImpactRule(
        pattern="cpp/tensorrt_llm/layers/eagleDecodingLayer.*",
        tier=Tier.OPT_IN,
        feature="eagle3",
        config_class="Eagle3DecodingConfig",
        description="C++ Eagle3 decoding layer",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/kernels/speculativeDecoding/*",
        tier=Tier.OPT_IN,
        feature="speculative",
        description="C++ speculative decoding kernels",
    ),
    # Guided decoding
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/grammar_matcher.py",
        tier=Tier.OPT_IN,
        feature="guided_decoding",
        description="Grammar-based guided decoding",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/guided_decoder.py",
        tier=Tier.OPT_IN,
        feature="guided_decoding",
        description="Guided decoder interface",
    ),
    # CUDA graph
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py",
        tier=Tier.OPT_IN,
        feature="cuda_graph",
        config_class="CudaGraphConfig",
        description="CUDA graph capture/replay",
    ),
    # Torch compile
    ImpactRule(
        pattern="tensorrt_llm/_torch/compilation/*",
        tier=Tier.OPT_IN,
        feature="torch_compile",
        config_class="TorchCompileConfig",
        description="Torch compilation infrastructure",
    ),
    # Quantization
    ImpactRule(
        pattern="tensorrt_llm/quantization/*",
        tier=Tier.OPT_IN,
        feature="quantization",
        description="Quantization (FP8, NVFP4, W4, etc.)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/quantization/utils/fp8_utils.py",
        tier=Tier.OPT_IN,
        feature="fp8",
        description="FP8 quantization utilities",
    ),
    ImpactRule(
        pattern="tensorrt_llm/quantization/utils/fp4_utils.py",
        tier=Tier.OPT_IN,
        feature="nvfp4",
        description="NVFP4/FP4 quantization utilities",
    ),
    # MOE
    ImpactRule(
        pattern="tensorrt_llm/_torch/modules/fused_moe/*",
        tier=Tier.OPT_IN,
        feature="moe",
        config_class="MoeConfig",
        description="Fused MoE module",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/distributed/moe_alltoall.py",
        tier=Tier.OPT_IN,
        feature="moe",
        description="MoE AllToAll communication",
    ),
    # Disaggregated serving
    ImpactRule(
        pattern="tensorrt_llm/_torch/disaggregation/*",
        tier=Tier.OPT_IN,
        feature="disaggregated",
        description="Disaggregated serving",
    ),
    ImpactRule(
        pattern="tensorrt_llm/llmapi/disagg_utils.py",
        tier=Tier.OPT_IN,
        feature="disaggregated",
        description="Disaggregated serving API utils",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/executor/cache_transmission/*",
        tier=Tier.OPT_IN,
        feature="disaggregated",
        description="C++ cache transmission for disagg",
    ),
    # Beam search
    ImpactRule(
        pattern="cpp/tensorrt_llm/layers/beamSearchLayer.*",
        tier=Tier.OPT_IN,
        feature="beam_search",
        description="C++ beam search layer",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/kernels/beamSearchKernels.*",
        tier=Tier.OPT_IN,
        feature="beam_search",
        description="C++ beam search kernels",
    ),
    # Mamba (state-space models: Nemotron-H, Jamba, etc.)
    ImpactRule(
        pattern="tensorrt_llm/_torch/modules/mamba/*",
        tier=Tier.OPT_IN,
        feature="mamba",
        description="Mamba/SSM modules",
    ),

    # AutoDeploy shim (core infra for AD backend)
    ImpactRule(
        pattern="tensorrt_llm/_torch/auto_deploy/shim/*",
        tier=Tier.CORE,
        description="AutoDeploy executor shim (core AD infra)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/auto_deploy/llm_args.py",
        tier=Tier.CORE,
        description="AutoDeploy LLM args",
    ),
    # AutoDeploy transforms
    ImpactRule(
        pattern="tensorrt_llm/_torch/auto_deploy/transform/*",
        tier=Tier.OPT_IN,
        feature="auto_deploy",
        description="AutoDeploy graph transforms",
    ),
    # AutoDeploy custom ops
    ImpactRule(
        pattern="tensorrt_llm/_torch/auto_deploy/custom_ops/*",
        tier=Tier.OPT_IN,
        feature="auto_deploy",
        description="AutoDeploy custom ops (attention, MLA, FLA, mamba, quant)",
    ),
    # AutoDeploy models
    ImpactRule(
        pattern="tensorrt_llm/_torch/auto_deploy/models/*",
        tier=Tier.OPT_IN,
        feature="auto_deploy",
        description="AutoDeploy model patches and configs",
    ),
    # AutoDeploy distributed
    ImpactRule(
        pattern="tensorrt_llm/_torch/auto_deploy/distributed/*",
        tier=Tier.OPT_IN,
        feature="auto_deploy",
        description="AutoDeploy distributed utilities",
    ),
    # AutoDeploy utils
    ImpactRule(
        pattern="tensorrt_llm/_torch/auto_deploy/utils/*",
        tier=Tier.OPT_IN,
        feature="auto_deploy",
        description="AutoDeploy utility functions",
    ),
    # AutoDeploy config
    ImpactRule(
        pattern="tensorrt_llm/_torch/auto_deploy/config/*",
        tier=Tier.OPT_IN,
        feature="auto_deploy",
        description="AutoDeploy default config",
    ),
    # Mamba cache manager
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/mamba_cache_manager.py",
        tier=Tier.OPT_IN,
        feature="mamba",
        description="Mamba state-space model cache manager",
    ),
    # Speculative decoding model drafter and spec sampler base
    ImpactRule(
        pattern="tensorrt_llm/_torch/speculative/model_drafter.py",
        tier=Tier.OPT_IN,
        feature="speculative",
        description="Model-based speculative drafter",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/speculative/spec_sampler_base.py",
        tier=Tier.OPT_IN,
        feature="speculative",
        description="Speculative sampling base class",
    ),
    # Custom C++ ops (torch compile path)
    ImpactRule(
        pattern="tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py",
        tier=Tier.OPT_IN,
        feature="torch_compile",
        description="C++ custom ops for torch compile path",
    ),
    # Visual generation
    ImpactRule(
        pattern="tensorrt_llm/_torch/visual_gen/*",
        tier=Tier.OPT_IN,
        feature="visual_gen",
        description="Visual generation (Flux, WAN, LTX2) pipelines",
    ),
    ImpactRule(
        pattern="tensorrt_llm/llmapi/visual_gen.py",
        tier=Tier.OPT_IN,
        feature="visual_gen",
        description="Visual generation API entry point",
    ),
    # Serving layer
    ImpactRule(
        pattern="tensorrt_llm/serve/openai_server.py",
        tier=Tier.DEFAULT_ON,
        feature="serve",
        description="OpenAI-compatible server",
    ),
    ImpactRule(
        pattern="tensorrt_llm/serve/openai_client.py",
        tier=Tier.OPT_IN,
        feature="serve",
        description="OpenAI client utilities",
    ),
    ImpactRule(
        pattern="tensorrt_llm/serve/harmony_adapter.py",
        tier=Tier.OPT_IN,
        feature="serve",
        description="Harmony adapter for serving",
    ),
    ImpactRule(
        pattern="tensorrt_llm/serve/media_storage.py",
        tier=Tier.OPT_IN,
        feature="serve",
        description="Media storage for multimodal serving",
    ),
    ImpactRule(
        pattern="tensorrt_llm/serve/postprocess_handlers.py",
        tier=Tier.OPT_IN,
        feature="serve",
        description="Serve post-processing handlers",
    ),
    ImpactRule(
        pattern="tensorrt_llm/serve/responses_utils.py",
        tier=Tier.OPT_IN,
        feature="serve",
        description="Serve response utilities",
    ),
    ImpactRule(
        pattern="tensorrt_llm/serve/tool_parser/*",
        tier=Tier.OPT_IN,
        feature="serve",
        description="Serve tool call parsers",
    ),
    ImpactRule(
        pattern="tensorrt_llm/serve/scripts/*",
        tier=Tier.IGNORE,
        description="Serve benchmark scripts",
    ),
    # Reasoning parser
    ImpactRule(
        pattern="tensorrt_llm/llmapi/reasoning_parser.py",
        tier=Tier.OPT_IN,
        feature="reasoning",
        description="Reasoning output parser (chain-of-thought)",
    ),
    # Multimodal inputs
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_multimodal_inputs.py",
        tier=Tier.OPT_IN,
        feature="multimodal",
        description="Multimodal input processing",
    ),
    # MoE communication backends
    ImpactRule(
        pattern="tensorrt_llm/_torch/modules/fused_moe/communication/*",
        tier=Tier.OPT_IN,
        feature="moe",
        description="MoE communication backends (DeepEP, NVLink)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py",
        tier=Tier.OPT_IN,
        feature="moe",
        description="Configurable MoE module",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/modules/fused_moe/quantization.py",
        tier=Tier.OPT_IN,
        feature="moe",
        description="MoE quantization",
    ),
    # Disaggregated serving sub-modules
    ImpactRule(
        pattern="tensorrt_llm/_torch/disaggregation/base/*",
        tier=Tier.OPT_IN,
        feature="disaggregated",
        description="Disaggregated serving base modules",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/disaggregation/native/*",
        tier=Tier.OPT_IN,
        feature="disaggregated",
        description="Disaggregated serving native implementation",
    ),
    # Multimodal inputs utils
    ImpactRule(
        pattern="tensorrt_llm/inputs/*",
        tier=Tier.OPT_IN,
        feature="multimodal",
        description="Multimodal input utilities",
    ),
    # Benchmark utilities
    ImpactRule(
        pattern="tensorrt_llm/bench/*",
        tier=Tier.IGNORE,
        description="Benchmark utilities (trtllm-bench)",
    ),
    # Build scripts
    ImpactRule(
        pattern="scripts/build_wheel.py",
        tier=Tier.IGNORE,
        description="Wheel build script",
    ),
    ImpactRule(
        pattern="scripts/release_check.py",
        tier=Tier.IGNORE,
        description="Release check script",
    ),
    # Security scanning
    ImpactRule(
        pattern="security_scanning/*",
        tier=Tier.IGNORE,
        description="Security scanning configs",
    ),
    # Jenkins CI
    ImpactRule(
        pattern="jenkins/*",
        tier=Tier.IGNORE,
        description="Jenkins CI pipeline scripts",
    ),
    # Examples
    ImpactRule(
        pattern="examples/*",
        tier=Tier.IGNORE,
        description="Example configs and scripts",
    ),
    # Benchmarks
    ImpactRule(
        pattern="tests/microbenchmarks/*",
        tier=Tier.IGNORE,
        description="Microbenchmark scripts",
    ),
    ImpactRule(
        pattern="tests/scripts/*",
        tier=Tier.IGNORE,
        description="Test infrastructure scripts",
    ),
    # Dotfiles (.codex, .claude, .github)
    ImpactRule(
        pattern=".codex/*",
        tier=Tier.IGNORE,
        description="Codex agent config",
    ),
    ImpactRule(
        pattern=".claude/*",
        tier=Tier.IGNORE,
        description="Claude agent config",
    ),

    # ---- Model-Specific ----
    ImpactRule(
        pattern="tensorrt_llm/models/llama/*",
        tier=Tier.MODEL,
        arch="llama",
        description="LLaMA model family (Llama, Mistral, Mixtral, etc.)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_llama*.py",
        tier=Tier.MODEL,
        arch="llama",
        description="PyTorch LLaMA model",
    ),
    ImpactRule(
        pattern="tensorrt_llm/models/deepseek_v2/*",
        tier=Tier.MODEL,
        arch="deepseek_v2",
        description="DeepSeek V2/V3/R1 model family",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_deepseekv3.py",
        tier=Tier.MODEL,
        arch="deepseek_v2",
        description="PyTorch DeepSeek V3 model",
    ),
    ImpactRule(
        pattern="tensorrt_llm/models/qwen/*",
        tier=Tier.MODEL,
        arch="qwen",
        description="Qwen model family",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_qwen*.py",
        tier=Tier.MODEL,
        arch="qwen",
        description="PyTorch Qwen model",
    ),
    ImpactRule(
        pattern="tensorrt_llm/models/gemma/*",
        tier=Tier.MODEL,
        arch="gemma",
        description="Gemma model family",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_gemma*.py",
        tier=Tier.MODEL,
        arch="gemma",
        description="PyTorch Gemma model",
    ),
    ImpactRule(
        pattern="tensorrt_llm/models/gpt/*",
        tier=Tier.MODEL,
        arch="gpt",
        description="GPT model family (GPT-OSS)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_gptoss.py",
        tier=Tier.MODEL,
        arch="gpt",
        description="PyTorch GPT-OSS model",
    ),
    ImpactRule(
        pattern="tensorrt_llm/models/phi3/*",
        tier=Tier.MODEL,
        arch="phi",
        description="Phi model family",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_phi*.py",
        tier=Tier.MODEL,
        arch="phi",
        description="PyTorch Phi model",
    ),
    ImpactRule(
        pattern="tensorrt_llm/models/chatglm/*",
        tier=Tier.MODEL,
        arch="chatglm",
        description="ChatGLM/GLM model family",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_glm.py",
        tier=Tier.MODEL,
        arch="chatglm",
        description="PyTorch GLM model",
    ),
    ImpactRule(
        pattern="tensorrt_llm/models/nemotron_nas/*",
        tier=Tier.MODEL,
        arch="nemotron_nas",
        description="Nemotron NAS model family",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_nemotron*.py",
        tier=Tier.MODEL,
        arch="nemotron",
        description="PyTorch Nemotron models",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_minimaxm2.py",
        tier=Tier.MODEL,
        arch="deepseek_v2",
        description="PyTorch MiniMaxM2 model (DeepSeek V2 arch)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_parakeet.py",
        tier=Tier.MODEL,
        arch="parakeet",
        description="PyTorch Parakeet model",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_radio.py",
        tier=Tier.MODEL,
        arch="nemotron",
        description="PyTorch RADIO model (Nemotron family)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/modeling_nemotron_nano.py",
        tier=Tier.MODEL,
        arch="nemotron",
        description="PyTorch Nemotron Nano model",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/models/checkpoints/hf/qwen3_next_weight_mapper.py",
        tier=Tier.MODEL,
        arch="qwen",
        description="Qwen3Next HF weight mapper",
    ),
    ImpactRule(
        pattern="tensorrt_llm/models/eagle/*",
        tier=Tier.MODEL,
        arch="eagle",
        description="Eagle speculative decoding model",
    ),
    ImpactRule(
        pattern="tensorrt_llm/models/mamba/*",
        tier=Tier.MODEL,
        arch="mamba",
        description="Mamba state-space model",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/kernels/flashMLA/*",
        tier=Tier.MODEL,
        arch="deepseek_v2",
        description="Flash MLA kernels (DeepSeek-specific)",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/*",
        tier=Tier.MODEL,
        arch="deepseek_v2",
        description="DeepSeek V3 min-latency kernels",
    ),
    ImpactRule(
        pattern="cpp/tensorrt_llm/kernels/llama4MinLatencyKernels/*",
        tier=Tier.MODEL,
        arch="llama",
        description="Llama4 min-latency kernels",
    ),

    # ---- Test Files ----
    ImpactRule(
        pattern="tests/integration/defs/accuracy/test_llm_api_pytorch.py",
        tier=Tier.TEST,
        description="Main accuracy test definitions",
    ),
    ImpactRule(
        pattern="tests/integration/defs/accuracy/test_disaggregated_serving.py",
        tier=Tier.TEST,
        description="Disaggregated serving accuracy tests",
    ),
    ImpactRule(
        pattern="tests/integration/defs/test_e2e.py",
        tier=Tier.TEST,
        description="E2E test definitions",
    ),
    ImpactRule(
        pattern="tests/integration/defs/disaggregated/*",
        tier=Tier.TEST,
        feature="disaggregated",
        description="Disaggregated test definitions",
    ),
    ImpactRule(
        pattern="tests/integration/defs/llmapi/*",
        tier=Tier.TEST,
        description="LLM API test definitions",
    ),
    ImpactRule(
        pattern="tests/integration/defs/examples/serve/*",
        tier=Tier.TEST,
        feature="serve",
        description="Serve test definitions",
    ),
    ImpactRule(
        pattern="tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py",
        tier=Tier.TEST,
        description="Multimodal accuracy test definitions",
    ),
    ImpactRule(
        pattern="tests/integration/defs/accuracy/test_llm_api_autodeploy.py",
        tier=Tier.TEST,
        description="AutoDeploy accuracy test definitions",
    ),
    ImpactRule(
        pattern="tests/integration/defs/examples/test_ad_speculative_decoding.py",
        tier=Tier.TEST,
        description="AutoDeploy speculative decoding test definitions",
    ),
    ImpactRule(
        pattern="tests/integration/defs/perf/*",
        tier=Tier.TEST,
        description="Performance test definitions",
    ),
    ImpactRule(
        pattern="tests/integration/defs/stress_test/*",
        tier=Tier.TEST,
        description="Stress test definitions",
    ),

    # ---- Additional Tier 2 (Opt-In) ----
    # Sparse attention (DSA)
    ImpactRule(
        pattern="tensorrt_llm/_torch/attention_backend/sparse/*",
        tier=Tier.OPT_IN,
        feature="sparse_attention",
        description="Sparse/DSA attention backend",
    ),
    # Tokenizer GLM MoE DSA
    ImpactRule(
        pattern="tensorrt_llm/tokenizer/glm_moe_dsa/*",
        tier=Tier.MODEL,
        arch="chatglm",
        description="GLM MoE DSA tokenizer",
    ),
    # Speculative decoding __init__ (package-level imports)
    ImpactRule(
        pattern="tensorrt_llm/_torch/speculative/__init__.py",
        tier=Tier.OPT_IN,
        feature="speculative",
        description="Speculative decoding package init",
    ),

    # ---- Additional Core/Default-On ----
    ImpactRule(
        pattern="tensorrt_llm/_torch/model_config.py",
        tier=Tier.CORE,
        description="PyTorch model config (affects all models)",
    ),
    ImpactRule(
        pattern="tensorrt_llm/_torch/pyexecutor/config_utils.py",
        tier=Tier.CORE,
        description="Executor config utilities",
    ),
    ImpactRule(
        pattern="tensorrt_llm/functional.py",
        tier=Tier.CORE,
        description="Core functional operations (broad impact)",
    ),

    # ---- Ignore: No accuracy test impact ----
    ImpactRule(pattern="docs/*", tier=Tier.IGNORE, description="Documentation"),
    ImpactRule(pattern="*.md", tier=Tier.IGNORE, description="Markdown files"),
    ImpactRule(
        pattern=".github/*", tier=Tier.IGNORE, description="GitHub config"),
    ImpactRule(
        pattern="examples/configs/*",
        tier=Tier.IGNORE,
        description="Reference configs",
    ),
    ImpactRule(
        pattern="tests/integration/defs/accuracy/references/*",
        tier=Tier.IGNORE,
        description="Accuracy reference data (expected outputs, not code)",
    ),
    ImpactRule(
        pattern="tests/integration/test_lists/*",
        tier=Tier.IGNORE,
        description="Test list files (CI config, not test logic)",
    ),
    ImpactRule(
        pattern="tests/unittest/*",
        tier=Tier.IGNORE,
        description="Unit tests (don't affect integration test selection)",
    ),
    ImpactRule(
        pattern="scripts/*",
        tier=Tier.IGNORE,
        description="Build/CI scripts (not production code)",
    ),
]

# ============================================================================
# Test Class → Architecture Mapping
# ============================================================================
# Maps test class names to architecture keys for model-specific impact.
# Architecture keys must match the `arch` field in MODEL tier ImpactRules.

TESTCLASS_TO_ARCH: dict[str, str] = {
    # Llama family (all use LLaMAForCausalLM)
    "TestLlama3_1_8B": "llama",
    "TestLlama3_1_8BInstruct": "llama",
    "TestLlama3_2_1B": "llama",
    "TestLlama3_2_3B": "llama",
    "TestLlama3_1_70B": "llama",
    "TestLlama3_1_405BInstructFp4": "llama",
    "TestLlama3_3_70BInstruct": "llama",
    "TestLlama3_3NemotronSuper49Bv1": "llama",
    "TestLlama4MaverickInstruct": "llama",
    "TestLlama4ScoutInstruct": "llama",
    # Mistral variants (use LLaMAForCausalLM)
    "TestMistral7B": "llama",
    "TestMistralSmall24B": "llama",
    "TestMinistral8BInstruct": "llama",
    "TestMistralNemo12B": "llama",
    "TestMistralLarge3_675B": "llama",
    "TestCodestral_22B_V01": "llama",
    "TestBielik11BInstruct": "llama",
    # Mixtral (MoE, but uses LLaMAForCausalLM)
    "TestMixtral8x7B": "llama",
    # DeepSeek family (DeepseekV2ForCausalLM)
    "TestDeepSeekV3Lite": "deepseek_v2",
    "TestDeepSeekV3": "deepseek_v2",
    "TestDeepSeekV32": "deepseek_v2",
    "TestDeepSeekR1": "deepseek_v2",
    "TestDeepSeekR1DistillLlama70B": "llama",
    "TestDeepSeekR1LongBenchV2": "deepseek_v2",
    "TestDeepSeekV32Exp": "deepseek_v2",
    # Kimi (uses DeepseekV2 architecture)
    "TestKimiK2": "deepseek_v2",
    "TestKimiK25": "deepseek_v2",
    # Qwen family
    "TestQwen2_7BInstruct": "qwen",
    "TestQwen3_4B": "qwen",
    "TestQwen3_8B": "qwen",
    "TestQwen3_30B_A3B": "qwen",
    "TestQwen3_235B_A22B": "qwen",
    "TestQwQ_32B": "qwen",
    "TestQwen3NextThinking": "qwen",
    "TestQwen3NextInstruct": "qwen",
    # Gemma
    "TestGemma3_27BInstruct": "gemma",
    "TestGemma3_1BInstruct": "gemma",
    # GPT-OSS
    "TestGPTOSS": "gpt",
    # Phi
    "TestPhi4": "phi",
    "TestPhi4MM": "phi",
    "TestPhi4MiniInstruct": "phi",
    # GLM/ChatGLM
    "TestGLM4_6": "chatglm",
    "TestGLM4_5Air": "chatglm",
    "TestGLM5FP8": "chatglm",
    # Nemotron
    "TestMinitron4BBaseInstruct": "nemotron",
    "TestNemotronNas": "nemotron_nas",
    "TestNemotronUltra": "nemotron_nas",
    "TestNemotronV3Nano": "nemotron",
    "TestNemotronV3Super": "nemotron",
    "TestNemotron3Super120B": "nemotron",
    # Others
    "TestEXAONE4": "llama",
    "TestSeedOss_36B": "llama",
    "TestKanana_Instruct": "llama",
    "TestStarcoder2_3B": "llama",
    "TestStarcoder2_7B": "llama",
    "TestStarcoder2_15B": "llama",
    "TestLlama3_1_8B_Instruct_RocketKV": "llama",
    "TestMiniMaxM2": "deepseek_v2",
    # AutoDeploy
    "TestLlama3_1_8B_Instruct_Eagle3": "llama",
    # Multimodal
    "TestLlava_V1_6_Mistral_7B": "llama",
    "TestNVILA_8B": "llama",
    "TestNemotron_Nano_12B_V2_VL": "nemotron",
    "TestPhi4MMFusedVisionLora": "phi",
    "TestQwen2_5_VL_7B": "qwen",
    "TestQwen2_VL_7B": "qwen",
    "TestQwen3VL_MOE": "qwen",
    "TestVILA1_5_3B": "llama",
    # LLM API QA
    "TestLlmDefaultBackend": "llama",
}

# Model name/path keywords → architecture mapping.
# Used to match module-level test functions (e.g. test_e2e.py) that carry
# model names in their parametrize values rather than in class attributes.
# Patterns are matched case-insensitively against each entry in model_names.
# Order matters: more specific patterns are checked first.
MODEL_NAME_TO_ARCH: list[tuple[str, str]] = [
    # DeepSeek family (must come before "llama" patterns)
    ("deepseek-v3", "deepseek_v2"),
    ("deepseek-v2", "deepseek_v2"),
    ("deepseek-r1", "deepseek_v2"),
    ("deepseekv3", "deepseek_v2"),
    # Kimi / MiniMax (DeepSeek V2 arch)
    ("kimi-k2", "deepseek_v2"),
    ("minimax", "deepseek_v2"),
    # Llama family (includes Mistral, Mixtral, etc.)
    ("llama", "llama"),
    ("mistral", "llama"),
    ("mixtral", "llama"),
    ("codestral", "llama"),
    ("exaone", "llama"),
    ("starcoder", "llama"),
    ("bielik", "llama"),
    ("kanana", "llama"),
    ("seed-oss", "llama"),
    # Nemotron
    ("nemotron-ultra", "nemotron_nas"),
    ("nemotron-super", "nemotron_nas"),
    ("nemotron-nas", "nemotron_nas"),
    ("nemotron", "nemotron"),
    ("minitron", "nemotron"),
    # Qwen
    ("qwen", "qwen"),
    ("qwq", "qwen"),
    # GLM / ChatGLM
    ("glm", "chatglm"),
    # Gemma
    ("gemma", "gemma"),
    # GPT-OSS
    ("gpt-oss", "gpt"),
    ("gpt_oss", "gpt"),
    # Phi
    ("phi", "phi"),
    # Parakeet
    ("parakeet", "parakeet"),
    # RADIO
    ("radio", "nemotron"),
    # Mamba
    ("mamba", "mamba"),
    # Eagle (spec decode draft model)
    ("eagle", "eagle"),
]


def model_name_to_arch(model_name: str) -> str:
    """Map a model name/path string to an architecture key.

    Returns empty string if no match.
    """
    name_lower = model_name.lower()
    for pattern, arch in MODEL_NAME_TO_ARCH:
        if pattern in name_lower:
            return arch
    return ""


# ============================================================================
# Representative Coverage Set
# ============================================================================
# Minimum set of tests that covers all major dimensions. Used when Tier 0 or
# Tier 1 changes occur. Each entry represents one key combination of:
#   architecture × feature × GPU scale
#
# Selection criteria:
#   - Every model architecture has at least 1 test
#   - Every major opt-in feature has at least 1 test
#   - GPU scales 1/2/4/8 are each covered
#   - Prefer fast tests (small models, 1 GPU) where possible

REPRESENTATIVE_COVERAGE_SET: list[str] = [
    # --- Architecture coverage (1 GPU each) ---
    # llama — also covers: chunked_prefill, attn_trtllm
    "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_chunked_prefill[use_temperature=False-attn_backend=TRTLLM]",
    # deepseek_v2 (MTP + cuda_graph + overlap + attn_dp + chunked_prefill)
    "accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_bfloat16[mtp_nextn=2-attention_dp=True-cuda_graph=True-overlap_scheduler=True-torch_compile=False-enable_chunked_prefill=True]",
    # qwen
    "accuracy/test_llm_api_pytorch.py::TestQwen2_7BInstruct::test_auto_dtype",
    # gemma
    "accuracy/test_llm_api_pytorch.py::TestGemma3_1BInstruct::test_auto_dtype",
    # phi
    "accuracy/test_llm_api_pytorch.py::TestPhi4MiniInstruct::test_auto_dtype",
    # nemotron
    "accuracy/test_llm_api_pytorch.py::TestNemotronV3Nano::test_auto_dtype",
    # gpt-oss — also covers: w4, moe
    "accuracy/test_llm_api_pytorch.py::TestGPTOSS::test_w4_1gpu[v1_kv_cache-True-True-cutlass-auto]",

    # chatglm — also covers: nvfp4, mtp
    "accuracy/test_llm_api_pytorch.py::TestGLM4_6::test_nvfp4_2_model_mtp[2model]",
    # nemotron_nas
    "accuracy/test_llm_api_pytorch.py::TestNemotronNas::test_auto_dtype_tp8",

    # --- Opt-in feature coverage (1 GPU each) ---
    # Eagle3 (speculative decoding)
    "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_eagle3[sampler_async_worker=False-eagle3_one_model=True-overlap_scheduler=True]",
    # NGram (speculative decoding)
    "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_ngram",
    # PARD (speculative decoding)
    "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_pard[overlap_scheduler=True]",
    # Guided decoding
    "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_guided_decoding[xgrammar]",
    # Beam search
    "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_auto_dtype_beam_search[enable_cuda_graph=False-enable_padding=False-disable_overlap_scheduler=False-sampler_async_worker=False]",
    # FP8
    "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_fp8_llm_sampler",
    # NVFP4
    "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8B::test_nvfp4",
    # FlashInfer attention backend
    "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_chunked_prefill[use_temperature=False-attn_backend=FLASHINFER]",
    # Sparse attention (DSA) — also covers: fp8 + moe + mtp + cuda_graph
    "accuracy/test_llm_api_pytorch.py::TestDeepSeekV32::test_fp8_blockscale[baseline]",
    # FP8 KV cache — also covers: nvfp4 + moe_backend + moe
    "accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_nvfp4[moe_backend=CUTLASS-mtp_nextn=0-fp8kv=True-attention_dp=False-cuda_graph=False-overlap_scheduler=False-torch_compile=False]",

    # --- Disaggregated serving ---
    "accuracy/test_disaggregated_serving.py::TestLlama3_1_8BInstruct::test_auto_dtype[False-False-False-True]",

    # --- E2E smoke ---
    "test_e2e.py::test_openai_multi_chat_example",
    "test_e2e.py::test_ptp_quickstart",
]

# ============================================================================
# "Speculative" feature: matches any speculative decoding feature
# ============================================================================
SPECULATIVE_FEATURES = {
    'eagle3', 'mtp', 'ngram', 'pard', 'suffix_automaton', 'auto_spec_decode',
    'speculative'
}

# Feature → Config class mapping (for Tier 1/2 matching)
FEATURE_TO_CONFIG: dict[str, set[str]] = {
    'kv_cache': {'KvCacheConfig'},
    'cuda_graph': {'CudaGraphConfig'},
    'torch_compile': {'TorchCompileConfig'},
    'eagle3': {'Eagle3DecodingConfig'},
    'mtp': {'MTPDecodingConfig'},
    'pard': {'PARDDecodingConfig'},
    'ngram': {'NGramDecodingConfig'},
    'suffix_automaton': {'SADecodingConfig'},
    'moe': {'MoeConfig'},
    'eplb': {'MoeLoadBalancerConfig'},
    'scheduler': {'SchedulerConfig'},
}


def match_rule(changed_file: str, rule: ImpactRule) -> bool:
    """Check if a changed file matches an impact rule's pattern."""
    return fnmatch(changed_file, rule.pattern)
