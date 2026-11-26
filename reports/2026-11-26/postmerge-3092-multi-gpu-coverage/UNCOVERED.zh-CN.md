<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# L0_PostMerge #3092：最终全量 coverage 仍未覆盖的文件与函数

- 分析日期：2026-11-26（Asia/Shanghai）。
- 构建：[L0_PostMerge #3092](https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/main/job/L0_PostMerge/3092/)，结果 `FAILURE`，最终 coverage 生成和上传成功。
- 源码 commit：`38b3273d289ea24ee2e4d2d47f28817ca1cc61a1`。
- 覆盖数据使用最终全量包，已包含成功收集到的单 GPU 和单节点 multi-GPU 数据；不是单 GPU checkpoint。
- 与前一份 [multi-GPU 净增量报告](https://github.com/crazydemo/TensorRT-LLM/blob/cbts-monitor/reports/2026-11-26/postmerge-3092-multi-gpu-coverage/REPORT.zh-CN.md) 使用同一 build、同一源码库存及同一官方覆盖率口径。

## 结论

官方统计范围内共 **3,753 个文件、62,388 个函数**。在 #3092 最终全量数据中，仍有 **433 个文件完全未触达、19,543 个函数未被观测覆盖**。文件覆盖率为 **88.4626%**；函数覆盖率为 **68.6751%**。

未覆盖函数不能只从“完全未触达文件”中查找：**4,672 个**来自完全未触达文件，另有 **14,871 个（76.0937%）**来自已有文件触达记录的文件。后者是更大的函数覆盖缺口。

未覆盖函数最多的三个目录是 `tensorrt_llm/_torch/modeling_v2/`（9,810）、`tensorrt_llm/_torch/attention/`（2,287）和 `tensorrt_llm/_torch/moe/`（1,704）。目录统计不重复计算函数。

## 阅读索引

- “统计口径与边界”：说明本报告的“未覆盖”能证明什么。
- “总量及文件状态”“模块分布”：定位缺口规模。
- “未覆盖函数最多的文件”：快速查看高数量文件。
- 附录 A：433 个完全未触达文件的完整路径表。
- 附录 B：完全未触达文件中的 4,672 个未覆盖函数完整清单。
- 附录 C：已触达但没有匹配任何库存函数的 259 个文件、2,159 个函数完整清单。
- 附录 D：部分库存函数已覆盖的 1,543 个文件、12,712 个未覆盖函数完整清单。

## 统计口径与边界

1. 使用 #3092 源码 commit 对应的 `coveragerc.template` 和 `pystart_report.py` 的 AST 枚举实现，复现官方分母；结果为 3,753 个文件、62,388 个函数，且最终覆盖分子与官方报告一致。
2. 源码枚举根目录是仓库根（构建命令 `--source-root llm`）。include 为 `tensorrt_llm/*.py`、`examples/*.py`、`scripts/*.py`、`tests/test_common/*.py`、`conftest.py`、`tests/conftest.py`、`tests/*/conftest.py`。模式中的 `*` 可以匹配 `/`。因此这不是仅 `tensorrt_llm/` 的统计。
3. 函数按 `(file, qualname)` 去重，包含方法和嵌套函数；同一函数跨测试、架构、stage、GPU rank 的重复执行只算一个已覆盖函数。只使用 `test != ''` 的 touch 记录。
4. `未覆盖函数 = AST 函数库存 − 最终观测函数集合`。`完全未触达文件 = AST 文件库存 − 最终观测文件集合`。文件库存只包含至少有一个可枚举函数的 Python 文件；未纳入分母的文件不在本报告中。
5. 文件触达可以仅来自 `<module>` 或无法匹配 AST 库存的运行时符号；文件触达不代表文件内已有库存函数被覆盖。报告因此单列“已触达但未匹配任何库存函数”这一类，而不把它们误称为部分函数已覆盖。
6. “未被观测覆盖”表示最终采集数据中没有该函数在非空测试上下文下的匹配记录，不等同于证明该代码从未执行，也不直接证明缺少测试。条件定义、抽象接口、特定硬件/模型变体，以及采集缺失等因素都需要逐项核实；本报告不据此自动给所有条目判定缺陷。
7. 这是 Python PY_START 函数入口覆盖，不是分支、行或 C++/CUDA kernel 覆盖；Python 中定义的 DSL/JIT 前端函数与设备端 kernel 的实际执行覆盖不能混为一谈。
8. 构建结果为 FAILURE；最终 `test_case_meta` 中 3,212 条为 passed 且 untainted，350 条为 non-passed 或 tainted。统计复现官方 observed coverage，不额外剔除这些记录；“最终全量”不保证所有预期 stage 都成功回传。

## 总量及文件状态

| 官方统计项 | 总量 | 已覆盖 | 未覆盖 | 覆盖率 |
|---|---:|---:|---:|---:|
| 文件 | 3,753 | 3,320 | 433 | 88.4626% |
| 函数 | 62,388 | 42,845 | 19,543 | 68.6751% |

| 文件状态 | 文件数 | 未覆盖函数数 | 占全部未覆盖函数 |
|---|---:|---:|---:|
| 文件完全未触达 | 433 | 4,672 | 23.9063% |
| 文件已触达，但未匹配任何库存函数 | 259 | 2,159 | 11.0474% |
| 文件已触达，部分库存函数未覆盖 | 1,543 | 12,712 | 65.0463% |
| 库存函数已全部覆盖 | 1,518 | 0 | 0.0000% |

其中，函数覆盖仍有缺口的文件共 **2,235 个**；另有 **1,518 个文件**的库存函数全部被覆盖。这里的“全部覆盖”也只指函数入口，不代表所有分支或行为均已验证。

仅 `tensorrt_llm/` 的补充统计：

| 范围 | 文件总量 | 未触达文件 | 函数总量 | 未覆盖函数 |
|---|---:|---:|---:|---:|
| `tensorrt_llm/` | 3,507 | 299 | 60,155 | 18,407 |
| examples / scripts / 配置允许的测试辅助代码 | 246 | 134 | 2,233 | 1,136 |

## 模块分布

按目录归类并按未覆盖函数数降序排列；每个库存文件只归入一行。`_torch` 下按下一层目录划分，其他产品路径按 `tensorrt_llm/` 下一级划分。保留零缺口行，便于复核汇总。

| 模块/目录 | 文件数 | 完全未触达文件 | 函数总量 | 未覆盖函数 | 函数覆盖率 |
|---|---:|---:|---:|---:|---:|
| `tensorrt_llm/_torch/modeling_v2/` | 1,698 | 249 | 21,336 | 9,810 | 54.0214% |
| `tensorrt_llm/_torch/attention/` | 366 | 12 | 7,664 | 2,287 | 70.1592% |
| `tensorrt_llm/_torch/moe/` | 204 | 7 | 4,426 | 1,704 | 61.5002% |
| `tensorrt_llm/_torch/models/` | 143 | 0 | 4,042 | 715 | 82.3107% |
| `tensorrt_llm/_torch/kernels/` | 78 | 0 | 1,917 | 571 | 70.2139% |
| `examples/` | 131 | 91 | 786 | 554 | 29.5165% |
| `tensorrt_llm/_torch/visual_gen/` | 204 | 0 | 3,384 | 442 | 86.9385% |
| `scripts/` | 58 | 34 | 826 | 438 | 46.9734% |
| `tensorrt_llm/_torch/modules/` | 121 | 0 | 2,194 | 390 | 82.2242% |
| `tensorrt_llm/scaffolding/` | 34 | 20 | 499 | 358 | 28.2565% |
| `tensorrt_llm/_torch/pyexecutor/` | 136 | 0 | 4,173 | 338 | 91.9003% |
| `tensorrt_llm/serve/` | 85 | 0 | 1,759 | 196 | 88.8573% |
| `tensorrt_llm/_torch/disaggregation/` | 46 | 0 | 1,393 | 170 | 87.7961% |
| `tensorrt_llm/executor/` | 23 | 0 | 547 | 154 | 71.8464% |
| `tests/` | 57 | 9 | 621 | 144 | 76.8116% |
| `tensorrt_llm/evaluate/` | 21 | 1 | 275 | 134 | 51.2727% |
| `tensorrt_llm/_torch/distributed/` | 14 | 1 | 387 | 128 | 66.9251% |
| `tensorrt_llm/_torch/speculative/` | 41 | 0 | 798 | 128 | 83.9599% |
| `tensorrt_llm/_torch/cute_dsl_kernels/` | 8 | 3 | 204 | 124 | 39.2157% |
| `tensorrt_llm/_torch/` | 29 | 0 | 576 | 97 | 83.1597% |
| `tensorrt_llm/_torch/custom_ops/` | 21 | 0 | 849 | 96 | 88.6926% |
| `tensorrt_llm/llmapi/` | 20 | 0 | 747 | 93 | 87.5502% |
| `tensorrt_llm/bench/` | 23 | 1 | 179 | 53 | 70.3911% |
| `tensorrt_llm/models/` | 4 | 1 | 64 | 53 | 17.1875% |
| `tensorrt_llm/tools/` | 8 | 2 | 119 | 50 | 57.9832% |
| `tensorrt_llm/inputs/` | 13 | 0 | 331 | 41 | 87.6133% |
| `tensorrt_llm/_utils.py` | 1 | 0 | 139 | 39 | 71.9424% |
| `tensorrt_llm/tokenizer/` | 5 | 0 | 117 | 30 | 74.3590% |
| `tensorrt_llm/grpc/` | 21 | 0 | 274 | 27 | 90.1460% |
| `tensorrt_llm/_torch/compilation/` | 10 | 0 | 115 | 24 | 79.1304% |
| `tensorrt_llm/_torch/peft/` | 11 | 0 | 127 | 22 | 82.6772% |
| `tensorrt_llm/quantization/` | 8 | 1 | 105 | 21 | 80.0000% |
| `tensorrt_llm/_torch/kv_cache_compression/` | 7 | 0 | 71 | 17 | 76.0563% |
| `tensorrt_llm/_torch/memory/` | 22 | 0 | 265 | 15 | 94.3396% |
| `tensorrt_llm/mapping.py` | 1 | 0 | 63 | 12 | 80.9524% |
| `tensorrt_llm/functional.py` | 1 | 0 | 28 | 11 | 60.7143% |
| `tensorrt_llm/_torch/configs/` | 14 | 0 | 100 | 9 | 91.0000% |
| `tensorrt_llm/watermark/` | 7 | 0 | 33 | 8 | 75.7576% |
| `tensorrt_llm/visual_gen/` | 6 | 0 | 73 | 7 | 90.4110% |
| `tensorrt_llm/commands/` | 8 | 0 | 113 | 6 | 94.6903% |
| `tensorrt_llm/_torch/weight_sharing/` | 4 | 0 | 88 | 5 | 94.3182% |
| `tensorrt_llm/_jit_share.py` | 1 | 0 | 46 | 4 | 91.3043% |
| `tensorrt_llm/profiler.py` | 1 | 0 | 20 | 4 | 80.0000% |
| `tensorrt_llm/_torch/debug/` | 1 | 0 | 31 | 3 | 90.3226% |
| `tensorrt_llm/_torch/locality_domain/` | 4 | 0 | 33 | 3 | 90.9091% |
| `tensorrt_llm/logger.py` | 1 | 0 | 28 | 2 | 92.8571% |
| `tensorrt_llm/sampling_params.py` | 1 | 0 | 26 | 2 | 92.3077% |
| `tensorrt_llm/_bootstrap.py` | 1 | 0 | 15 | 1 | 93.3333% |
| `tensorrt_llm/media/` | 3 | 0 | 37 | 1 | 97.2973% |
| `tensorrt_llm/metrics/` | 4 | 0 | 48 | 1 | 97.9167% |
| `tensorrt_llm/usage/` | 7 | 1 | 144 | 1 | 99.3056% |
| `tensorrt_llm/__init__.py` | 1 | 0 | 2 | 0 | 100.0000% |
| `tensorrt_llm/_device_copy.py` | 1 | 0 | 7 | 0 | 100.0000% |
| `tensorrt_llm/_dlpack_utils.py` | 1 | 0 | 4 | 0 | 100.0000% |
| `tensorrt_llm/_errors.py` | 1 | 0 | 1 | 0 | 100.0000% |
| `tensorrt_llm/_ipc_utils.py` | 1 | 0 | 14 | 0 | 100.0000% |
| `tensorrt_llm/_startup.py` | 1 | 0 | 5 | 0 | 100.0000% |
| `tensorrt_llm/_startup_errors.py` | 1 | 0 | 25 | 0 | 100.0000% |
| `tensorrt_llm/_torch/shared_tensor/` | 1 | 0 | 16 | 0 | 100.0000% |
| `tensorrt_llm/conversation_params.py` | 1 | 0 | 1 | 0 | 100.0000% |
| `tensorrt_llm/disaggregated_params.py` | 1 | 0 | 3 | 0 | 100.0000% |
| `tensorrt_llm/kv_hints/` | 2 | 0 | 5 | 0 | 100.0000% |
| `tensorrt_llm/math_utils.py` | 1 | 0 | 2 | 0 | 100.0000% |
| `tensorrt_llm/runtime/` | 4 | 0 | 68 | 0 | 100.0000% |

## 未覆盖函数最多的文件

以下只是按数量排序的定位入口，不是已确认的补测优先级；完整函数名见附录。

### 完全未触达文件：前 20

| 文件 | 库存函数 | 已覆盖 | 未覆盖 |
|---|---:|---:|---:|
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/moe/token_comm_gen_specialized.py` | 116 | 0 | 116 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/router/fused_o_proj/kernel_tp4.py` | 77 | 0 | 77 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/local_moe/m31_local_moe_kernel.py` | 71 | 0 | 71 |
| `tensorrt_llm/_torch/cute_dsl_kernels/minimax_m3/moe/m3_moe16_kernel.py` | 64 | 0 | 64 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/gemm/splitk.py` | 62 | 0 | 62 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/routed_moe/ptx.py` | 61 | 0 | 61 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_mla.py` | 59 | 0 | 59 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/attention/gvr_balanced.py` | 59 | 0 | 59 |
| `tensorrt_llm/_torch/moe/kernels/megamoe_scheduler/sami/fabric.py` | 58 | 0 | 58 |
| `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/internode/router_internode.py` | 57 | 0 | 57 |
| `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/internode/token_comm_internode.py` | 56 | 0 | 56 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/router_hoist_host.py` | 53 | 0 | 53 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/qkv_fusion/kernel.py` | 51 | 0 | 51 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/dispatch.py` | 50 | 0 | 50 |
| `examples/scaffolding/benchmarks/multiround_chat_benchmark.py` | 42 | 0 | 42 |
| `tests/integration/defs/triton_server/conftest.py` | 41 | 0 | 41 |
| `scripts/generate_fmha_params.py` | 38 | 0 | 38 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/comm/allreduce_norm_quant/router_exchange.py` | 37 | 0 | 37 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/ptx.py` | 36 | 0 | 36 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/absorbed_query/shared_indexer_cute.py` | 35 | 0 | 35 |

### 文件已触达，但没有库存函数覆盖：前 20

| 文件 | 库存函数 | 已覆盖 | 未覆盖 |
|---|---:|---:|---:|
| `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain/token_comm_dedup.py` | 115 | 0 | 115 |
| `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/local_mega/local_routing.py` | 77 | 0 | 77 |
| `tensorrt_llm/_torch/models/modeling_vila.py` | 54 | 0 | 54 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_comm/k3_latent_ar_norm.py` | 46 | 0 | 46 |
| `tensorrt_llm/_torch/models/modeling_hyperclovax.py` | 41 | 0 | 41 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__helix8_moeep8/weights.py` | 35 | 0 | 35 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__helix8_moetp2ep4/weights.py` | 35 | 0 | 35 |
| `tensorrt_llm/scaffolding/trace_replay/replay.py` | 35 | 0 | 35 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_mla.py` | 34 | 0 | 34 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__tp1/weights.py` | 34 | 0 | 34 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_model.py` | 31 | 0 | 31 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/dspark_markov_worker.py` | 31 | 0 | 31 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/tail.py` | 28 | 0 | 28 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_indexer.py` | 27 | 0 | 27 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/dspark_markov_chain.py` | 27 | 0 | 27 |
| `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/smem_transformed_kv.py` | 25 | 0 | 25 |
| `tensorrt_llm/_torch/attention/kernels/rubin/dsv4_qb_fusion/kernel.py` | 24 | 0 | 24 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/comm/allgather_cast.py` | 24 | 0 | 24 |
| `tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_kernels.py` | 22 | 0 | 22 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/_k3_fp4_mla_kernels.py` | 21 | 0 | 21 |

### 已有部分库存函数覆盖：前 30

| 文件 | 库存函数 | 已覆盖 | 未覆盖 |
|---|---:|---:|---:|
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/modeling.py` | 279 | 8 | 271 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__helix8_moetp2ep4/modeling.py` | 258 | 9 | 249 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/modeling.py` | 221 | 21 | 200 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__helix8_moeep8/modeling.py` | 258 | 69 | 189 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/modeling.py` | 224 | 38 | 186 |
| `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_context/fmha_resources.py` | 281 | 108 | 173 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/modeling.py` | 166 | 3 | 163 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2ctx/modeling.py` | 145 | 7 | 138 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/modeling.py` | 195 | 57 | 138 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/modeling.py` | 203 | 68 | 135 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_fp4_mla/fp4_mla_mufu16.py` | 136 | 18 | 118 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/modeling.py` | 176 | 58 | 118 |
| `tensorrt_llm/_torch/attention/backends/sparse/csa2/kernel.py` | 211 | 96 | 115 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__dep16/modeling.py` | 192 | 83 | 109 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__tp1/modeling.py` | 159 | 59 | 100 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/modeling.py` | 163 | 64 | 99 |
| `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_2cta/resources.py` | 106 | 9 | 97 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/dspark/worker.py` | 115 | 26 | 89 |
| `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/smem_resources.py` | 102 | 21 | 81 |
| `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain/token_comm.py` | 136 | 58 | 78 |
| `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/tmem_corr.py` | 80 | 7 | 73 |
| `tensorrt_llm/_torch/distributed/communicator.py` | 144 | 73 | 71 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark.py` | 81 | 13 | 68 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/dspark.py` | 68 | 3 | 65 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/msa_backend.py` | 65 | 3 | 62 |
| `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/token_comm_gen_specialized.py` | 122 | 61 | 61 |
| `tensorrt_llm/evaluate/nemo_skills_eval.py` | 63 | 2 | 61 |
| `tensorrt_llm/scaffolding/task_collection.py` | 85 | 25 | 60 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_indexer.py` | 61 | 2 | 59 |
| `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_epilogue.py` | 114 | 55 | 59 |

## 附录 A：完全未触达文件（完整清单）

以下 **433 个文件**在最终 `test != ''` 观测集合中完全没有文件触达记录，其 **4,672 个库存函数**均未覆盖。

| 文件 | 未覆盖库存函数数 |
|---|---:|
| `examples/disaggregated/slurm/benchmark/get_env.py` | 1 |
| `examples/disaggregated/slurm/benchmark/submit_dwdp.py` | 5 |
| `examples/dwdp/reproduce.py` | 18 |
| `examples/infinitebench/args.py` | 1 |
| `examples/infinitebench/compute_scores.py` | 17 |
| `examples/infinitebench/construct_synthetic_dataset.py` | 2 |
| `examples/infinitebench/eval_utils.py` | 8 |
| `examples/kimi_k3/measure_dspark_acceptance.py` | 5 |
| `examples/kimi_k3/quick_start_kimi_k3.py` | 2 |
| `examples/layer_wise_benchmarks/parse_e2e.py` | 4 |
| `examples/llm-api/llm_kv_cache_compression.py` | 5 |
| `examples/llm-api/llm_kv_cache_offloading.py` | 1 |
| `examples/llm-api/llm_lmcache_connector.py` | 1 |
| `examples/llm-api/llm_runtime.py` | 3 |
| `examples/llm-api/llm_sampling.py` | 10 |
| `examples/llm-api/llm_sparse_attention.py` | 7 |
| `examples/llm-api/out_of_tree_example/main.py` | 1 |
| `examples/llm-api/quickstart_multimodal.py` | 4 |
| `examples/longbench/eval_longbench_v1.py` | 13 |
| `examples/models/core/deepseek_v4/dspark_benchmark/analyze.py` | 10 |
| `examples/models/core/deepseek_v4/dspark_benchmark/build_datasets.py` | 8 |
| `examples/models/core/deepseek_v4/dspark_benchmark/make_configs.py` | 2 |
| `examples/models/core/gpt_oss/openai_chat_client_function_calling.py` | 3 |
| `examples/models/core/kimi_k2/kimi_k2_tool_calling_example.py` | 5 |
| `examples/quantization/quantize_mixed_precision_moe.py` | 7 |
| `examples/ray_orchestrator/llm_inference_async_ray.py` | 3 |
| `examples/ray_orchestrator/llm_inference_distributed_ray.py` | 2 |
| `examples/scaffolding/benchmarks/__main__.py` | 2 |
| `examples/scaffolding/benchmarks/agent_benchmark.py` | 5 |
| `examples/scaffolding/benchmarks/benchmark_utils.py` | 5 |
| `examples/scaffolding/benchmarks/chat_benchmark.py` | 1 |
| `examples/scaffolding/benchmarks/coder_benchmark.py` | 6 |
| `examples/scaffolding/benchmarks/multiround_chat_benchmark.py` | 42 |
| `examples/scaffolding/contrib/AsyncGeneration/stream_generation_controller.py` | 4 |
| `examples/scaffolding/contrib/Coder/run_coder.py` | 4 |
| `examples/scaffolding/contrib/Coder/run_swebench.py` | 16 |
| `examples/scaffolding/contrib/DeepConf/run_generation.py` | 5 |
| `examples/scaffolding/contrib/DeepConf/utils.py` | 3 |
| `examples/scaffolding/contrib/Dynasor/scaffolding_dynasor_run.py` | 3 |
| `examples/scaffolding/contrib/TreeInference/run_mcts_example.py` | 2 |
| `examples/scaffolding/contrib/TreeInference/run_tot_example.py` | 2 |
| `examples/scaffolding/contrib/iter_research/run_iter_research.py` | 5 |
| `examples/scaffolding/contrib/open_deep_research/run_open_deep_research.py` | 5 |
| `examples/scaffolding/contrib/tree_of_thought_research/run_tot_research.py` | 5 |
| `examples/scaffolding/mcp/e2b/e2bserver.py` | 1 |
| `examples/scaffolding/mcp/e2b/main.py` | 1 |
| `examples/scaffolding/mcp/fetch_webpage/fetch_webpage.py` | 8 |
| `examples/scaffolding/mcp/fetch_webpage/visit_controller.py` | 10 |
| `examples/scaffolding/mcp/google_scholar/google_scholar.py` | 6 |
| `examples/scaffolding/mcp/google_search/google_search.py` | 5 |
| `examples/scaffolding/mcp/mcptest.py` | 2 |
| `examples/scaffolding/mcp/tavily_search/tavily_controller.py` | 5 |
| `examples/scaffolding/mcp/tavily_search/tavily_search.py` | 6 |
| `examples/scaffolding/mcp/weather/weather.py` | 4 |
| `examples/scaffolding/mcp/wordllama/main.py` | 1 |
| `examples/scaffolding/mcp/wordllama/wordllama_serve.py` | 2 |
| `examples/scaffolding/run_basic_generation.py` | 5 |
| `examples/scaffolding/run_best_of_n_with_reward.py` | 2 |
| `examples/scaffolding/run_majority_vote_aime24.py` | 3 |
| `examples/scaffolding/token_budget_majority_vote.py` | 6 |
| `examples/scaffolding/trace_replay/aggregate_pareto.py` | 1 |
| `examples/scaffolding/trace_replay/analysis/aggregation.py` | 3 |
| `examples/scaffolding/trace_replay/analysis/annotate.py` | 1 |
| `examples/scaffolding/trace_replay/analysis/blocks.py` | 7 |
| `examples/scaffolding/trace_replay/analysis/branch_summary.py` | 9 |
| `examples/scaffolding/trace_replay/analysis/cache_hit.py` | 10 |
| `examples/scaffolding/trace_replay/analysis/compute_cache_hit_trace.py` | 2 |
| `examples/scaffolding/trace_replay/analysis/io.py` | 8 |
| `examples/scaffolding/trace_replay/analysis/streams.py` | 14 |
| `examples/scaffolding/trace_replay/metrics.py` | 10 |
| `examples/scaffolding/trace_replay/run_trace_replay.py` | 7 |
| `examples/scaffolding/trace_replay/run_trace_replay_pareto.py` | 8 |
| `examples/serve/compatibility/chat_completions/example_06_tool_calling.py` | 1 |
| `examples/serve/compatibility/responses/example_02_streaming_chat.py` | 1 |
| `examples/serve/compatibility/responses/example_05_tool_calling.py` | 2 |
| `examples/serve/prometheus_metrics.py` | 3 |
| `examples/visual_gen/models/cosmos3/generate_bouncing_ball_control.py` | 3 |
| `examples/visual_gen/models/fasth3.py` | 1 |
| `examples/visual_gen/models/glm_image.py` | 3 |
| `examples/visual_gen/models/ltx23.py` | 1 |
| `examples/visual_gen/models/ltx23_retake.py` | 2 |
| `examples/visual_gen/models/minimax_h3.py` | 1 |
| `examples/visual_gen/models/qwen_image_edit.py` | 2 |
| `examples/visual_gen/serve/async_video_gen.py` | 1 |
| `examples/visual_gen/serve/delete_video.py` | 1 |
| `examples/visual_gen/serve/sync_image_gen.py` | 1 |
| `examples/visual_gen/serve/sync_video_gen.py` | 1 |
| `examples/wide_ep/ep_load_balancer/generate_eplb_config.py` | 2 |
| `examples/wide_ep/ep_load_balancer/report_load_statistics.py` | 1 |
| `examples/wide_ep/ep_load_balancer/utils.py` | 2 |
| `examples/wide_ep/slurm_scripts/process_gen_iterlog.py` | 1 |
| `scripts/attribute.py` | 6 |
| `scripts/attribution/sbom/src/trtllm_sbom/cli.py` | 6 |
| `scripts/attribution/sbom/src/trtllm_sbom/core.py` | 19 |
| `scripts/attribution/sbom/src/trtllm_sbom/storage.py` | 18 |
| `scripts/attribution/sbom/tests/conftest.py` | 2 |
| `scripts/attribution/sbom/tests/test_core.py` | 5 |
| `scripts/attribution/sbom/tests/test_sbom.py` | 2 |
| `scripts/attribution/sbom/tests/test_storage.py` | 10 |
| `scripts/attribution/sbom/tests/test_workflows.py` | 3 |
| `scripts/attribution/scan/_package_resolvers.py` | 17 |
| `scripts/attribution/scan/_pattern_matcher.py` | 15 |
| `scripts/attribution/scan/identify_build_inputs.py` | 23 |
| `scripts/attribution/scan/map_dependencies.py` | 11 |
| `scripts/attribution/scan/scan_build_artifacts.py` | 3 |
| `scripts/bump_version.py` | 8 |
| `scripts/check_binding_stubs.py` | 6 |
| `scripts/check_pinned_memory_usage.py` | 4 |
| `scripts/check_qa_selection_rules.py` | 28 |
| `scripts/check_skill_naming_convention.py` | 4 |
| `scripts/clean_tests.py` | 16 |
| `scripts/cuda_driver_stub.py` | 3 |
| `scripts/dco_check.py` | 2 |
| `scripts/format_test_list.py` | 2 |
| `scripts/generate_fmha_params.py` | 38 |
| `scripts/generate_lock_file.py` | 6 |
| `scripts/generate_moe_impl_catalog.py` | 3 |
| `scripts/git_replace.py` | 2 |
| `scripts/package_trt_llm.py` | 5 |
| `scripts/pyrefly_exclusion.py` | 17 |
| `scripts/pyrefly_workflow.py` | 15 |
| `scripts/release_check.py` | 5 |
| `scripts/rename_docker_images.py` | 11 |
| `scripts/vendor_triton_kernels.py` | 7 |
| `scripts/visualize_kv_cache_stats.py` | 8 |
| `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/crossing_glmflash_mid.py` | 6 |
| `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/crossing_glmfull_short.py` | 6 |
| `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp4_glm_fulltile_filter.py` | 12 |
| `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp4_glmflash_queue_recovery.py` | 10 |
| `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp4_shared_kv_safe_filter.py` | 5 |
| `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp8_fulltile_filter.py` | 11 |
| `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp8_glm_mid_loop_filter.py` | 11 |
| `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp8_glmflash_queue_recovery.py` | 8 |
| `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp8_qwen_packed_filter.py` | 10 |
| `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp8_shared_kv_safe_filter.py` | 7 |
| `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/native_bf16_math.py` | 7 |
| `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/sampler_upper_glmflash_fp4.py` | 10 |
| `tensorrt_llm/_torch/cute_dsl_kernels/minimax_m3/moe/m3_moe16_kernel.py` | 64 |
| `tensorrt_llm/_torch/cute_dsl_kernels/minimax_m3/moe/op16.py` | 6 |
| `tensorrt_llm/_torch/cute_dsl_kernels/minimax_m3/mx_gemv/m3_mx_gemv_kernel.py` | 26 |
| `tensorrt_llm/_torch/distributed/pg_utils.py` | 1 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/attention.py` | 15 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/attention_forward.py` | 10 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/dispatch.py` | 50 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/absorbed_query/kernels.py` | 31 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/absorbed_query/query_proj.py` | 24 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/absorbed_query/shared_indexer.py` | 30 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/absorbed_query/shared_indexer_cute.py` | 35 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/cute_fused.py` | 19 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/cute_oproj.py` | 10 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/cute_oproj_ring.py` | 14 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/into_hop.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/kernels.py` | 12 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/prefetch.py` | 10 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/tail.py` | 19 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_layer_expert_prefetch/kernels.py` | 15 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_layer_expert_prefetch/spec_experts.py` | 22 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_rank_topk_cluster_merge/cp_indexer_decode.py` | 5 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_rank_topk_cluster_merge/cp_workspace.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_rank_topk_global_emit/bridge.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_rank_topk_global_emit/cp_indexer_decode.py` | 5 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_rank_topk_global_emit/cp_workspace.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/decode_prologue/prologue.py` | 10 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_input_prep/compat.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_input_prep/guard.py` | 15 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_input_prep/packet.py` | 23 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_input_prep/paged_mqa_schedule.py` | 1 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_input_prep/sampler_store.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_input_prep/stager.py` | 34 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/hop.py` | 18 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/hop_kernel.py` | 15 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/packet.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/proj_kernel.py` | 34 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/projection.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/ptx.py` | 36 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/query_entry.py` | 8 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/scope.py` | 5 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm_evict_hints/entry.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm_evict_hints/hop.py` | 5 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm_evict_hints/hop_kernel.py` | 12 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm_evict_hints/proj_kernel.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm_evict_hints/projection.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm_evict_hints/ptx.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/dense_mlp/dense_mlp.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_front/draft_front.py` | 6 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_front/kernels.py` | 16 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_front/native_norm.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_live_rows/hop.py` | 10 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_live_rows/step0.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_loop/draft_loop.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_loop/draft_step.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_loop/embedding.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_loop/verify.py` | 15 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_routed_moe/routed_moe.py` | 11 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_tail/draft_tail.py` | 22 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/fused_kv_indexer_projection/entry.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/fused_kv_indexer_projection/proj_kernel.py` | 17 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/fused_kv_indexer_projection/projection.py` | 11 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/fused_kv_indexer_projection/ptx.py` | 35 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/hop_routing/hop.py` | 10 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/hop_routing/routed_moe.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/host_page_retention/host_policy.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_layer_front/cp_indexer_decode.py` | 5 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_layer_front/cp_logits_runner.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_layer_front/fp8_paged_mqa_logits_cp.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_layer_front/kernels.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_layer_front/query_proj.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_layer_front/rope.py` | 1 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/cross_rank_indexer_decode.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/cross_rank_logits_runner.py` | 8 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/cross_rank_policy.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/cross_rank_workspace.py` | 8 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/fp8_paged_mqa_logits_cross_rank.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/fp8_paged_mqa_logits_rows.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/fp8_paged_mqa_logits_rowsplit.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/indexer_decode.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/logits_runner.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/policy.py` | 1 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/region.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/topk_select.py` | 12 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/oproj_weight_staging/cute_oproj_stage.py` | 23 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/oproj_weight_staging/kernels.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/oproj_weight_staging/tail.py` | 6 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_attention_hop/__init__.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_attention_hop/from_partials.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_attention_hop/hop.py` | 12 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_moe_hop/cute_kernel.py` | 34 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_moe_hop/health.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_moe_hop/hop.py` | 23 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_moe_hop/kernels.py` | 6 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_moe_hop/workspace.py` | 11 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/rope_kv_append/ops.py` | 15 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/routed_moe/kernel.py` | 25 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/routed_moe/kernel_xc.py` | 18 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/routed_moe/ptx.py` | 61 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/routed_moe/routed_moe.py` | 31 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/router_gemm/hop.py` | 6 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/router_gemm/router_kernel.py` | 24 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/shared_expert_prefetch/kernel_xc.py` | 5 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/shared_expert_prefetch/routed_moe.py` | 8 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/sparse_mla_decode/sparse_mla.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/layers.py` | 27 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/modeling.py` | 12 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/runtime.py` | 5 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/spec_worker.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/weights.py` | 1 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_dsa.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_dsa_custom_ops.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_mla.py` | 59 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_model.py` | 33 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_mtp.py` | 14 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_mtp_config.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_mtp_worker.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_speculative.py` | 8 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_weight_loader.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/attention/fp8_paged_mqa_logits.py` | 12 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/attention/gvr_balanced.py` | 59 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/gemm/rmsnorm_early_trigger.py` | 20 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/gemm/splitk.py` | 62 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/moe/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py` | 19 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/moe/ptx.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/moe/token_comm_gen_specialized.py` | 116 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/moe/topk_reduce_add_rmsnorm.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/spec/draft_glue_rmsnorm.py` | 8 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/modeling.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/full_query_logits.py` | 6 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/kv_norm_pack.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/projection.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/projection_op.py` | 5 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/q_a_norm.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/gemm/low_m.py` | 6 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/moe/megamoe_runner.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/spec/_draft_exchange.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/spec/draft_sample_glue.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/spec/mtp_tail.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/weights.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_gemm/low_m_bf16_splitk.py` | 25 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_spec/dspark_draft_ar.py` | 18 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_spec/dspark_draft_gemm_push.py` | 19 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/_k3_fp4_mla_cute_dsl_utils.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/_k3_fp4_mla_v_repack.py` | 11 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/fp4_mla_kv_append.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/spec/k3_draft_ar_norm.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/spec/k3_draft_embed.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/spec/k3_draft_gemm_push.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/csrc/msa_aot/build_bf16x_extension.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/csrc/msa_aot/build_k2q_extension.py` | 1 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/csrc/msa_aot/generate_fmha_variants.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/csrc/msa_aot/pack_cute_aot.py` | 1 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/csrc/msa_aot/torch_extension_aot.py` | 1 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/draft_probs_one_request_kernel.py` | 23 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/flagfree_gather_kernel.py` | 14 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3_draft_head_gemm.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3_lowm_splitk.py` | 35 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3ll/locality.py` | 1 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/qkv_fusion/clc_tile_scheduler.py` | 15 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/qkv_fusion/kernel.py` | 51 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/qkv_fusion/plain_gemm.py` | 14 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/corun_band.py` | 1 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/engagement.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/idx_pdl.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/idxk_producer.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/modeling.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/proj_pin.py` | 5 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/softmax_mc.py` | 10 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark_draft_fmha.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark_draft_head_fp4.py` | 23 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark_draft_mlp.py` | 11 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m31_markov16.py` | 8 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/spec/draft_bf16.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/dspark_draft_head.py` | 21 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/seams.py` | 25 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4_hoist_block.py` | 18 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/api/contracts.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/api/ctx.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/api/params.py` | 6 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/api/pdl.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/api/stage.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/api/variant.py` | 5 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/front/_bind.py` | 5 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/moe/_common.py` | 10 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/moe/sht4_n48.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/moe/sht4_n48_q.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/moe/sht_x4.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/router/lr4_prr4.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/router/lr4_prr4_q.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/comm/allreduce_norm_quant/allreduce_norm_quant_build.py` | 10 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/comm/allreduce_norm_quant/allreduce_norm_quant_host.py` | 6 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/comm/allreduce_norm_quant/lamport_allreduce_host.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/comm/allreduce_norm_quant/route_warp_tp4_fc1_per_token_host.py` | 19 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/comm/allreduce_norm_quant/route_warp_tp4_host.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/comm/allreduce_norm_quant/router_exchange.py` | 37 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/common/allreduce_push_workspace.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/common/host_hooks.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/common/model_geometry.py` | 13 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/common/post_mxfp8_buffers.py` | 1 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/common/stock_stages.py` | 32 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/expert_shards.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/fused/host.py` | 24 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/fused_die_local_host.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/fused_host.py` | 8 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/fused_shared_trigger_host.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/fused_tp4_host.py` | 6 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/post_nvfp4_handoff.py` | 8 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/router_hoist_host.py` | 53 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/router_hoist_weights.py` | 30 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/knobs.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/router/fused_o_proj/host.py` | 24 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/router/fused_o_proj/kernel_tp4.py` | 77 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/router/fused_o_proj_tp4_host.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/runtime/arena.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/runtime/comm.py` | 8 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/runtime/ctx.py` | 8 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/runtime/debug.py` | 6 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/runtime/knobs.py` | 14 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/toolchain/child_env.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/toolchain/paths.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/toolchain/procenv.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/dspark_draft_head.py` | 21 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_target_head.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_tokmap_guard.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/local_moe/bootstrap.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/local_moe/local_plan.py` | 10 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/local_moe/m31_local_moe_kernel.py` | 71 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/local_moe/m31_local_moe_op.py` | 14 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/local_moe/m31_local_moe_ws_receipt.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/local_moe/serving_hook.py` | 2 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/gemm/m3_mxfp8_pinned_quantize.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_router_gemv.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_target_head.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_router_gemv.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_target_head.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/cache_manager.py` | 9 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/kernels/msa_v1_decode.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/prewarm/core.py` | 15 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/prewarm/hook.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/prewarm/sdwarm.py` | 25 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/prewarm/selftest.py` | 5 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/comm/allgather_cast_kernel.py` | 14 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/draft_probs_one_request.py` | 31 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/allgather_bf16_kernel.py` | 14 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/dspark_glue_install.py` | 7 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/dspark_markov_chain_kernel.py` | 20 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/dspark_markov_chain_kernel_b1.py` | 20 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/dspark_markov_chain_kernel_b1_ll.py` | 23 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/gemm/router_half_pin.py` | 4 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/green_partition.py` | 6 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/idxk_split.py` | 16 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/moe/router_split.py` | 3 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/step_hooks.py` | 14 |
| `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/internode/lsa_peer_mapper.py` | 19 |
| `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/internode/node_local_barrier.py` | 4 |
| `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/internode/relay_message.py` | 2 |
| `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/internode/router_internode.py` | 57 |
| `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/internode/token_comm_internode.py` | 56 |
| `tensorrt_llm/_torch/moe/kernels/megamoe_scheduler/cuda_scheduler/multirank_dist.py` | 5 |
| `tensorrt_llm/_torch/moe/kernels/megamoe_scheduler/sami/fabric.py` | 58 |
| `tensorrt_llm/bench/dataclasses/enums.py` | 1 |
| `tensorrt_llm/evaluate/lm_eval_tasks/gpqa/cot_zeroshot_aa/_generate_configs.py` | 1 |
| `tensorrt_llm/models/automodel.py` | 3 |
| `tensorrt_llm/quantization/utils/fp8_matrix_weight_dequant.py` | 2 |
| `tensorrt_llm/scaffolding/contrib/AsyncGeneration/stream_generation.py` | 3 |
| `tensorrt_llm/scaffolding/contrib/Coder/coder.py` | 13 |
| `tensorrt_llm/scaffolding/contrib/Coder/swebench_utils.py` | 10 |
| `tensorrt_llm/scaffolding/contrib/DeepConf/deep_conf_controller.py` | 14 |
| `tensorrt_llm/scaffolding/contrib/DeepConf/deep_conf_utils.py` | 7 |
| `tensorrt_llm/scaffolding/contrib/Dynasor/dynasor_controller.py` | 4 |
| `tensorrt_llm/scaffolding/contrib/Dynasor/evaluator.py` | 21 |
| `tensorrt_llm/scaffolding/contrib/TreeInference/tree_controllers.py` | 27 |
| `tensorrt_llm/scaffolding/contrib/iter_research/agent.py` | 8 |
| `tensorrt_llm/scaffolding/contrib/iter_research/utils.py` | 5 |
| `tensorrt_llm/scaffolding/contrib/mcp/chat_handler.py` | 5 |
| `tensorrt_llm/scaffolding/contrib/mcp/chat_task.py` | 2 |
| `tensorrt_llm/scaffolding/contrib/mcp/mcp_controller.py` | 2 |
| `tensorrt_llm/scaffolding/contrib/mcp/mcp_task.py` | 2 |
| `tensorrt_llm/scaffolding/contrib/mcp/mcp_utils.py` | 6 |
| `tensorrt_llm/scaffolding/contrib/mcp/mcp_worker.py` | 5 |
| `tensorrt_llm/scaffolding/contrib/open_deep_research/researcher.py` | 12 |
| `tensorrt_llm/scaffolding/contrib/open_deep_research/supervisor.py` | 12 |
| `tensorrt_llm/scaffolding/contrib/open_deep_research/utils.py` | 1 |
| `tensorrt_llm/scaffolding/contrib/tree_of_thought_research/tot_research_controllers.py` | 21 |
| `tensorrt_llm/tools/ppl.py` | 1 |
| `tensorrt_llm/tools/profiler/nsys_profile_tools/gputrc2graph.py` | 14 |
| `tensorrt_llm/usage/schemas/__main__.py` | 1 |
| `tests/conftest.py` | 6 |
| `tests/integration/defs/cpp/conftest.py` | 10 |
| `tests/integration/defs/kv_cache/conftest.py` | 1 |
| `tests/integration/defs/triton_server/conftest.py` | 41 |
| `tests/microbenchmarks/attention_perf/conftest.py` | 1 |
| `tests/microbenchmarks/bench_moe/conftest.py` | 1 |
| `tests/qa_selection/tests/cases/conftest.py` | 1 |
| `tests/qa_selection/tests/conftest.py` | 1 |
| `tests/test_common/cluster_env.py` | 7 |

## 附录 B：完全未触达文件中的未覆盖函数（完整清单）

共 **433 个文件、4,672 个未覆盖函数**。各代码块只列未覆盖的 `qualname`；与对应文件路径组合后得到唯一函数键。

### `examples/disaggregated/slurm/benchmark/get_env.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
main
```

### `examples/disaggregated/slurm/benchmark/submit_dwdp.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
generate_mpi_worker_config
generate_mpi_worker_config.<locals>._build_urls
main
parse_args
submit_dwdp_job
```

### `examples/dwdp/reproduce.py`

库存函数 18；已覆盖 0；未覆盖 18。

```text
_as_bool
_build_sub_file
_cuda_graph_batch_sizes
_slugify
build_accuracy_config
build_full_config
build_job_name
build_output_path
build_worker_config
calc_seq_lens
load_experiments
load_yaml_file
main
merge_nested_dicts
parse_args
resolve_dataset_file
submit_config
write_config_file
```

### `examples/infinitebench/args.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
parse_args
```

### `examples/infinitebench/compute_scores.py`

库存函数 17；已覆盖 0；未覆盖 17。

```text
compute_scores
f1_score
first_int_match
get_labels
get_preds
get_score
get_score_one
get_score_one_kv_retrieval
get_score_one_passkey
iter_jsonl
load_json
normalize_answer
normalize_answer.<locals>.lower
normalize_answer.<locals>.remove_articles
normalize_answer.<locals>.remove_punc
normalize_answer.<locals>.white_space_fix
split_retrieval_answer
```

### `examples/infinitebench/construct_synthetic_dataset.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
build_kv_retrieval
build_passkey
```

### `examples/infinitebench/eval_utils.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
create_prompt
dump_json
dump_jsonl
get_answer
iter_jsonl
load_data
load_json
truncate_input
```

### `examples/kimi_k3/measure_dspark_acceptance.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_remove_warmup
build_llm
load_prompts
main
parse_arguments
```

### `examples/kimi_k3/quick_start_kimi_k3.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
main
parse_arguments
```

### `examples/layer_wise_benchmarks/parse_e2e.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
comma_separated_ints
is_gemm
iter_in_window
query_kernels
```

### `examples/llm-api/llm_kv_cache_compression.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_generate
main
parse_arguments
run_nvfp4_cold_page
run_triattention
```

### `examples/llm-api/llm_kv_cache_offloading.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
main
```

### `examples/llm-api/llm_lmcache_connector.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
main
```

### `examples/llm-api/llm_runtime.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
example_cuda_graph_config
example_kv_cache_config
main
```

### `examples/llm-api/llm_sampling.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
demonstrate_beam_search
demonstrate_combined_sampling
demonstrate_greedy_decoding
demonstrate_multiple_sequences
demonstrate_temperature_sampling
demonstrate_top_k_sampling
demonstrate_top_p_sampling
demonstrate_with_logprobs
main
run_all_demonstrations
```

### `examples/llm-api/llm_sparse_attention.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_build_default_prompt
main
parse_arguments
read_input
run_DSA
run_RocketKV
run_llm
```

### `examples/llm-api/out_of_tree_example/main.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
main
```

### `examples/llm-api/quickstart_multimodal.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
add_lora_args
add_multimodal_args
main
parse_arguments
```

### `examples/longbench/eval_longbench_v1.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
build_chat
calculate_metrics
calculate_task_summary
determine_chat_template
evaluate_single_dataset
format_prompt_style
initialize_llm
load_longbench_config
main
parse_arguments
post_process
save_results
setup_longbench_imports
```

### `examples/models/core/deepseek_v4/dspark_benchmark/analyze.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
average
describe
frontier
iso_gain
load_runs
main
plot
round_half_up
setting
value_at
```

### `examples/models/core/deepseek_v4/dspark_benchmark/build_datasets.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
arena_prompts
fetch
gsm8k_prompts
main
resolve_placeholders
sha256
sharegpt_prompts
speed_prompts
```

### `examples/models/core/deepseek_v4/dspark_benchmark/make_configs.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
base_config
main
```

### `examples/models/core/gpt_oss/openai_chat_client_function_calling.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
get_current_weather
get_multiple_weathers
main
```

### `examples/models/core/kimi_k2/kimi_k2_tool_calling_example.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
extract_tool_call_info
get_tool_call_requests
get_tools
get_weather
parse_specified_format_tool_calls
```

### `examples/quantization/quantize_mixed_precision_moe.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
get_scales_from_amax
load_and_preprocess_state_dict
main
main.<locals>.get_file_name
main.<locals>.get_tensor
parse_args
quantize_fp8_block_scale_to_int4
```

### `examples/ray_orchestrator/llm_inference_async_ray.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
main
main.<locals>.main
main.<locals>.task
```

### `examples/ray_orchestrator/llm_inference_distributed_ray.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
main
parse_arguments
```

### `examples/scaffolding/benchmarks/__main__.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
main
parse_arguments
```

### `examples/scaffolding/benchmarks/agent_benchmark.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
async_agent_benchmark
async_burst_agent_benchmark
cleanup_agent_resources
create_agent_resources
run_agent_benchmark_core
```

### `examples/scaffolding/benchmarks/benchmark_utils.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
load_prompts_from_json
print_benchmark_results
run_async_in_thread
run_benchmark_in_thread
shutdown_llm
```

### `examples/scaffolding/benchmarks/chat_benchmark.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
async_chat_benchmark
```

### `examples/scaffolding/benchmarks/coder_benchmark.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
async_coder_benchmark
cleanup_coder_resources
create_coder_resources
load_coder_prompts
register_coder_image
run_coder_benchmark_core
```

### `examples/scaffolding/benchmarks/multiround_chat_benchmark.py`

库存函数 42；已覆盖 0；未覆盖 42。

```text
ConstantDistribution.__init__
ConstantDistribution.__repr__
ConstantDistribution.sample
Distribution.sample
ExponentialDistribution.__init__
ExponentialDistribution.__repr__
ExponentialDistribution.sample
LognormalDelayDistribution.__init__
LognormalDelayDistribution.__repr__
LognormalDelayDistribution.sample
LognormalDistribution.__init__
LognormalDistribution.__repr__
LognormalDistribution._generate_lognormal_by_median
LognormalDistribution.sample
MultiroundChatController.__init__
MultiroundChatController.clone
MultiroundChatController.generate
MultiroundChatController.process
MultiroundChatWorker.__init__
MultiroundChatWorker._get_conversation_for_worker
MultiroundChatWorker._sample_user_delay
MultiroundChatWorker.multiround_handler
PoissonDistribution.__init__
PoissonDistribution.__repr__
PoissonDistribution.sample
SyntheticDataConfig.get_input_tokens_distribution
SyntheticDataConfig.get_num_turns_distribution
SyntheticDataConfig.get_output_tokens_distribution
SyntheticDataConfig.get_prefix_tokens_distribution
SyntheticDataConfig.to_gen_conv_args
UniformDistribution.__init__
UniformDistribution.__repr__
UniformDistribution.sample
UserDelayConfig.get_distribution
ZipfDistribution.__init__
ZipfDistribution.__repr__
ZipfDistribution.sample
async_multiround_chat_benchmark
generate_conversations
load_or_generate_conversations
load_sharegpt_conversations
print_conversation_stats
```

### `examples/scaffolding/contrib/AsyncGeneration/stream_generation_controller.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
NativeStreamGenerationController.__init__
NativeStreamGenerationController.process
NativeStreamGenerationController.set_output_threshold
NativeStreamGenerationController.set_stream_step
```

### `examples/scaffolding/contrib/Coder/run_coder.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_dump_task_metrics_summary
main
parse_arguments
register_image
```

### `examples/scaffolding/contrib/Coder/run_swebench.py`

库存函数 16；已覆盖 0；未覆盖 16。

```text
_duration_ms
_load_instance_ids_file
_log_image_progress
_split_instance_ids
_tool_call_time_summary
build_prompt
default_swebench_output_dir
main
parse_arguments
register_swebench_images
resolve_instance_ids
safe_output_component
save_trajectory
update_preds
write_selected_instances
write_tool_call_time_summary
```

### `examples/scaffolding/contrib/DeepConf/run_generation.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
main
parse_arguments
run_scaffolding_llm
test_majority_vote_controller
test_single_vote_controller
```

### `examples/scaffolding/contrib/DeepConf/utils.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
equal_func
prepare_prompt
quick_parse
```

### `examples/scaffolding/contrib/Dynasor/scaffolding_dynasor_run.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
main
main.<locals>.task
parse_arguments
```

### `examples/scaffolding/contrib/TreeInference/run_mcts_example.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
main
parse_arguments
```

### `examples/scaffolding/contrib/TreeInference/run_tot_example.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
main
parse_arguments
```

### `examples/scaffolding/contrib/iter_research/run_iter_research.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_dump_task_metrics_summary
_load_config
_mcp_streamable_http_urls
main
parse_arguments
```

### `examples/scaffolding/contrib/open_deep_research/run_open_deep_research.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_dump_task_metrics_summary
_load_config
_mcp_streamable_http_urls
main
parse_arguments
```

### `examples/scaffolding/contrib/tree_of_thought_research/run_tot_research.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_dump_task_metrics_summary
_load_config
_mcp_streamable_http_urls
main
parse_arguments
```

### `examples/scaffolding/mcp/e2b/e2bserver.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
run_code
```

### `examples/scaffolding/mcp/e2b/main.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
main
```

### `examples/scaffolding/mcp/fetch_webpage/fetch_webpage.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_fetch_single_url
_fetch_via_jina
_fetch_via_scraper
_load_config
_load_config.<locals>._get
_mcp_bind_from_cfg
_parse_pdf
fetch_webpage
```

### `examples/scaffolding/mcp/fetch_webpage/visit_controller.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
VisitController.__init__
VisitController.clone
VisitController.process
_extract_pdf_text_lines
_get_last_assistant_message_content
_looks_like_raw_pdf
_parse_json_output
_prepare_webpage_content
_sanitize_text
_truncate_with_notice
```

### `examples/scaffolding/mcp/google_scholar/google_scholar.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
_load_config
_load_config.<locals>._get
_mcp_bind_from_cfg
_scholar_single
_search_single
google_scholar
```

### `examples/scaffolding/mcp/google_search/google_search.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_load_config
_load_config.<locals>._get
_mcp_bind_from_cfg
_search_single
google_search
```

### `examples/scaffolding/mcp/mcptest.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
main
parse_arguments
```

### `examples/scaffolding/mcp/tavily_search/tavily_controller.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
TavilyController.__init__
TavilyController.clone
TavilyController.process
_last_assistant_content
_truncate_with_notice
```

### `examples/scaffolding/mcp/tavily_search/tavily_search.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
_load_config
_load_config.<locals>._get
_mcp_bind_from_cfg
_resolve_config_path
_search_single
tavily_search
```

### `examples/scaffolding/mcp/weather/weather.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
format_alert
get_alerts
get_forecast
make_nws_request
```

### `examples/scaffolding/mcp/wordllama/main.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
main
```

### `examples/scaffolding/mcp/wordllama/wordllama_serve.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
_sync_search
web_search
```

### `examples/scaffolding/run_basic_generation.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
main
parse_arguments
test_async
test_async.<locals>.test_async_func
test_sync
```

### `examples/scaffolding/run_best_of_n_with_reward.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
main
parse_arguments
```

### `examples/scaffolding/run_majority_vote_aime24.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
load_test_file
main
parse_arguments
```

### `examples/scaffolding/token_budget_majority_vote.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
TokenBudgetMajorityVoteController.__init__
TokenBudgetMajorityVoteController.clone
TokenBudgetMajorityVoteController.majority_vote
TokenBudgetMajorityVoteController.process
main
parse_arguments
```

### `examples/scaffolding/trace_replay/aggregate_pareto.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
main
```

### `examples/scaffolding/trace_replay/analysis/aggregation.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_rate_stats
_safe_div
aggregate_dataset_record
```

### `examples/scaffolding/trace_replay/analysis/annotate.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
build_annotated_trace
```

### `examples/scaffolding/trace_replay/analysis/blocks.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
TokenPrefixCache.__init__
TokenPrefixCache.insert
TokenPrefixCache.match
_TokenNode.__init__
engine_aligned_block_count
engine_aligned_hit_blocks
validate_tokens_per_block
```

### `examples/scaffolding/trace_replay/analysis/branch_summary.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
_branch_depth_key
_branch_root_key
_group_summary
_numeric_suffix
_rollup
_safe_div
_system_prompt_key
compute_branch_rollups
merge_rollup_arrays
```

### `examples/scaffolding/trace_replay/analysis/cache_hit.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
_Totals.__init__
_Totals.add
_as_nonnegative_int
_branch_path
_build_record
_collect_system_prefix_lengths
_reject_unsupported_events
_score_assistant_event
compute_cache_hit_upper_bound
compute_cache_hit_upper_bound.<locals>._commit_stream
```

### `examples/scaffolding/trace_replay/analysis/compute_cache_hit_trace.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
_parse_args
main
```

### `examples/scaffolding/trace_replay/analysis/io.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_trace_files_in_directory
default_annotated_trace_path
default_dataset_output_path
default_output_path
load_trace
resolve_input_trace_files
resolve_trace_file
write_json
```

### `examples/scaffolding/trace_replay/analysis/streams.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
ConversationSegments.__init__
ConversationSegments._segments_for
ConversationSegments._store
ConversationSegments.allocate_tokens
ConversationSegments.assemble_prompt
ConversationSegments.record_assistant_content
ConversationSegments.record_system
ConversationSegments.record_user_or_tool
ConversationSegments.seed_uuid_for
SystemPromptRegistry.__init__
SystemPromptRegistry.all_streams
SystemPromptRegistry.tokens
TokenIdAllocator.__init__
TokenIdAllocator.take
```

### `examples/scaffolding/trace_replay/metrics.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
_tpot_ms
collect_trace_file_stats
compute_replay_run_metrics
compute_steady_state_pareto
compute_steady_state_window
count_assistant_completion_tokens
count_parallel_regions
percentile
steady_state_excl_count
summarize_trace_events
```

### `examples/scaffolding/trace_replay/run_trace_replay.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_default_output_json
_normalize_base_url
_run_one_replay
_run_replay_with_client
_safe_close_client
main
parse_args
```

### `examples/scaffolding/trace_replay/run_trace_replay_pareto.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_default_output_json
_normalize_base_url
_one_session
_run
_system_prefixes
_warmup
main
parse_args
```

### `examples/serve/compatibility/chat_completions/example_06_tool_calling.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
get_weather
```

### `examples/serve/compatibility/responses/example_02_streaming_chat.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
print_streaming_responses_item
```

### `examples/serve/compatibility/responses/example_05_tool_calling.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
get_weather
process_tool_call
```

### `examples/serve/prometheus_metrics.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
fetch_metrics
main
parse_and_display_metrics
```

### `examples/visual_gen/models/cosmos3/generate_bouncing_ball_control.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
draw_frame
main
parse_args
```

### `examples/visual_gen/models/fasth3.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
main
```

### `examples/visual_gen/models/glm_image.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_output_paths
main
parse_args
```

### `examples/visual_gen/models/ltx23.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
main
```

### `examples/visual_gen/models/ltx23_retake.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
main
parse_args
```

### `examples/visual_gen/models/minimax_h3.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
main
```

### `examples/visual_gen/models/qwen_image_edit.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
main
parse_args
```

### `examples/visual_gen/serve/async_video_gen.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
test_async_video_generation
```

### `examples/visual_gen/serve/delete_video.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
test_delete_video
```

### `examples/visual_gen/serve/sync_image_gen.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
test_image_generation
```

### `examples/visual_gen/serve/sync_video_gen.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
test_sync_video_generation
```

### `examples/wide_ep/ep_load_balancer/generate_eplb_config.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
save_eplb_config
save_eplb_config.<locals>.represent_list_inline
```

### `examples/wide_ep/ep_load_balancer/report_load_statistics.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
calculate_load_statistics
```

### `examples/wide_ep/ep_load_balancer/utils.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
load_expert_statistic
load_expert_statistic.<locals>.parse_key
```

### `examples/wide_ep/slurm_scripts/process_gen_iterlog.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
process_files
```

### `scripts/attribute.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
_ensure_sbom_cli
_get_existing_dependencies
_get_sbom_core
_write_json
_write_readme
attempt_generate_or_prepare_payloads
```

### `scripts/attribution/sbom/src/trtllm_sbom/cli.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
generate
import_deps
list_dependencies
main
map_files
validate
```

### `scripts/attribution/sbom/src/trtllm_sbom/core.py`

库存函数 19；已覆盖 0；未覆盖 19。

```text
_add_dependency
_dep_key
_dep_name_version
_load_active_versions
_load_files_from_json
_warn_if_copyright_missing
_write_attribution_file
_write_missing_files
_write_sbom_file
add_dependencies_from_json
add_dependency_from_args
fetch_dependencies_list
format_dependencies_list
generate_outputs
register_files_from_args
register_files_from_json
run_generate
run_validate
validate_database
```

### `scripts/attribution/sbom/src/trtllm_sbom/storage.py`

库存函数 18；已覆盖 0；未覆盖 18。

```text
DependencyDatabase.__init__
DependencyDatabase._validate_file_mappings
DependencyDatabase._validate_metadata_entry
DependencyDatabase.add_dependency_metadata
DependencyDatabase.cas_get
DependencyDatabase.cas_put
DependencyDatabase.map_files_to_dependencies
DependencyDatabase.map_files_to_dependencies.<locals>.process_results
DependencyDatabase.register_dependency_files
DependencyDatabase.validate_all
_cas_get
_cas_put
_dump_yaml
_ensure_dirs
_extract_file_copyrights
_load_yaml
_read_and_hash_file
compute_cas_address
```

### `scripts/attribution/sbom/tests/conftest.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
stub_requests_head
stub_requests_head.<locals>._head
```

### `scripts/attribution/sbom/tests/test_core.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
test_add_register_generate_flow
test_json_add_and_register
test_transitive_attributions_are_appended
test_warning_on_missing_copyright_in_license
write
```

### `scripts/attribution/sbom/tests/test_sbom.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
cyclonedx_schema
test_cyclonedx_json_schema_validation
```

### `scripts/attribution/sbom/tests/test_storage.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
test_cas_put_get_roundtrip
test_compute_cas_address_stable_and_length
test_extract_spdx_lines
test_mixed_nv_notice_is_attributed
test_nv_only_notice_is_mapped
test_register_and_map_files
test_register_enforces_file_copyright_when_license_missing_notice
test_register_requires_metadata
test_validate_all_detects_bad_cas
write
```

### `scripts/attribution/sbom/tests/test_workflows.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_write_files
test_workflow_add_update
test_workflow_generate_outputs
```

### `scripts/attribution/scan/_package_resolvers.py`

库存函数 17；已覆盖 0；未覆盖 17。

```text
DpkgResolver.__init__
DpkgResolver._cache_all_package_files
DpkgResolver.get_license_path
DpkgResolver.get_package
PythonPackageResolver._get_all_license_files
PythonPackageResolver.get_attribution_path
PythonPackageResolver.get_license_path
PythonPackageResolver.get_package
Resolver.__init__
Resolver.get_license_path
Resolver.get_package
RpmResolver.get_license_path
RpmResolver.get_package
_find_matching_files
_normalize_package
_normalize_package_name
_normalize_version
```

### `scripts/attribution/scan/_pattern_matcher.py`

库存函数 15；已覆盖 0；未覆盖 15。

```text
DependencyMapping.to_dict
PatternMatcher.__init__
PatternMatcher._add_dependency
PatternMatcher._add_project_root_pattern
PatternMatcher._get_git_version
PatternMatcher._get_special_version
PatternMatcher._get_version
PatternMatcher._load_fetch_content
PatternMatcher._load_yaml
PatternMatcher.get_copyright_files
PatternMatcher.get_license_path
PatternMatcher.get_source_root
PatternMatcher.get_source_url
PatternMatcher.has_unknown_vendor
PatternMatcher.match
```

### `scripts/attribution/scan/identify_build_inputs.py`

库存函数 23；已覆盖 0；未覆盖 23。

```text
Artifact.to_dict
ArtifactCollector.__init__
ArtifactCollector._create_header_artifact
ArtifactCollector._deduplicate_artifacts
ArtifactCollector._extract_package_data_from_setup
ArtifactCollector._find_link_depfiles
ArtifactCollector._find_setup_py
ArtifactCollector._find_so_targets_from_patterns
ArtifactCollector._get_all_ninja_inputs
ArtifactCollector._get_headers_from_depfiles
ArtifactCollector._get_libraries_from_link_depfiles
ArtifactCollector._get_needed_libraries
ArtifactCollector._get_ninja_inputs
ArtifactCollector._is_header_file
ArtifactCollector._is_library_file
ArtifactCollector._parse_depfile_tokens
ArtifactCollector._parse_header_depfile
ArtifactCollector._parse_link_depfile
ArtifactCollector.collect_all
_parse_depfile_headers
collect_build_inputs
get_input_file_paths
main
```

### `scripts/attribution/scan/map_dependencies.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
DependencyMapper.__init__
DependencyMapper.get_attribution_path
DependencyMapper.get_license_path
DependencyMapper.map_file
DependencyMapper.map_files
create_attribution_payloads
create_attribution_payloads.<locals>.is_vendor
detect_active_versions
filter_ignored_deps
main
map_files_to_dependencies
```

### `scripts/attribution/scan/scan_build_artifacts.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
generate_reports
main
validate_yaml_files
```

### `scripts/bump_version.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_full_version
_replace_versions
_replace_versions.<locals>.replace
_sync_schemas
check
main
read_version
set_version
```

### `scripts/check_binding_stubs.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
_compiled_extension_roots
_const_strings
_root_extension_module
_setup_package_data_patterns
_stub_roots
main
```

### `scripts/check_pinned_memory_usage.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
PinnedMemoryUsageChecker.__init__
PinnedMemoryUsageChecker.visit_Call
_check_file
main
```

### `scripts/check_qa_selection_rules.py`

库存函数 28；已覆盖 0；未覆盖 28。

```text
ConftestLookups.__init__
ConftestLookups.read
IniMarkers.__init__
IniMarkers.read
InlineSkipif.location
MarkerDriftCheck.__init__
MarkerDriftCheck.faults
MarkerDriftCheck.run
MarkerFaults.__bool__
MarkerFaults.__init__
MarkerFaults.report_lines
RepoPaths.enable_package_import
RuleFaults.__bool__
RuleFaults.drifted_lines
RuleFaults.inline_only_lines
RuleFaults.missing_lines
RuleFaults.report_lines
RuleFaults.unverifiable_lines
RuleTableDriftCheck.__init__
RuleTableDriftCheck.faults
RuleTableDriftCheck.run
SkipDecorator.location
SkipifScanner.__init__
SkipifScanner.inline_with_reason
SkipifScanner.is_skipif_call
SkipifScanner.read
SkipifScanner.reason_of
main
```

### `scripts/check_skill_naming_convention.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
check_prefix
collect_items
load_frontmatter_name
main
```

### `scripts/clean_tests.py`

库存函数 16；已覆盖 0；未覆盖 16。

```text
_collect_top_level_defs
_dir_ignored
_is_symbol_check_target
_parse_test_name
_read_file_text
collect_py_files
find_cross_file_unused
get_unused_functions
get_used_tests
main
make_unused_func_predicate
make_unused_func_predicate.<locals>.should_remove
make_unused_test_predicate
make_unused_test_predicate.<locals>.should_remove
process_file
run_cmd
```

### `scripts/cuda_driver_stub.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_find_stub
_real_driver_present
ensure_cuda_driver_stub
```

### `scripts/dco_check.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
commit_message_has_signoff
main
```

### `scripts/format_test_list.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
main
normalize_whitespace
```

### `scripts/generate_fmha_params.py`

库存函数 38；已覆盖 0；未覆盖 38。

```text
Schema.nested
Usage.under
_accessor_name
_base_type
_check_file
_collect_members
_dependency_order
_dependency_order.<locals>.visit
_element_cpp
_filename
_forwardable
_header_lines
_is_property
_is_supported
_is_tensor
_metadata_call
_optional_inner
_read_expression
_read_mirror
_read_struct
_render_accessor
_render_update
_self_target
_set_element
_strip_quotes
_struct_name
_write_if_changed
generate
load_schemas
main
parse_args
render_accessors
render_all
render_field_type
render_fields
render_forwarding
render_update
scan_usage
```

### `scripts/generate_lock_file.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
_find_unresolvable_spec
_is_resolution_failure
_normalize_name
generate_metadata_json
get_project_info
poetry_add
```

### `scripts/generate_moe_impl_catalog.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_build_parser
_generated
main
```

### `scripts/git_replace.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
main
replace_content
```

### `scripts/package_trt_llm.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_check_banned_symbols
_check_banned_symbols.<locals>.form_command
_clean_files
compress
main
```

### `scripts/pyrefly_exclusion.py`

库存函数 17；已覆盖 0；未覆盖 17。

```text
ImportScope.__init__
ImportScope.__init__.<locals>.visit
ImportScope.resolve
filter_input_paths
filter_report
has_local_import
is_excluded_path
is_excluded_symbol
is_uninstalled_dependency
load_excluded_paths
load_exclusions
load_import_scope
main
matches_symbol
matches_symbol_reference
relative_source_path
uninstalled_dependencies
```

### `scripts/pyrefly_workflow.py`

库存函数 15；已覆盖 0；未覆盖 15。

```text
clear_path_module_exclusions
extra_test_import_paths
filter_diff_lines
filter_diff_lines.<locals>.changed_lines
filter_diff_lines.<locals>.relative_path
load_native_config
main
merge_reports
print_summary
report_status
select_project_files
test_import_path_sources
toml_value
wrap_unfiltered_report
write_test_config
```

### `scripts/release_check.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
handle_check_failure
main
run_cmd
run_precommit_with_timing
run_precommit_with_timing.<locals>.Result.__init__
```

### `scripts/rename_docker_images.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
extract_line_after_prefix
find_and_replace_in_files
find_script_directory
get_current_commit
get_current_timestamp
image_prefix
main
parse_arguments
rename_images
replace_text_between_dashes
run_shell_command
```

### `scripts/vendor_triton_kernels.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
add_attribution_header
clone_triton
copy_license
copy_triton_kernels
create_readme
create_version_file
main
```

### `scripts/visualize_kv_cache_stats.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_aggregate_windows
_field_series
extract_kv_entries
load_stats
main
plot_all
plot_all.<locals>._add_phase_markers
plot_per_window
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/crossing_glmflash_mid.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
Crossing.__call__
Crossing.__init__
Crossing.kernel
bucket_for
compile_selector
numeric_bucket
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/crossing_glmfull_short.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
Crossing.__call__
Crossing.__init__
Crossing.kernel
bucket_for
compile_selector
numeric_bucket
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp4_glm_fulltile_filter.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
FP4PrefillMQALogitsKernel.__call__
FP4PrefillMQALogitsKernel.__init__
FP4PrefillMQALogitsKernel._gemm_body
FP4PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP4PrefillMQALogitsKernel._setup_mma
FP4PrefillMQALogitsKernel.kernel
_subview
_target_is_rubin
predicated_store32
predicated_store64
publish_bank
utccp_required_smem_warp_transpose
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp4_glmflash_queue_recovery.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
FP4PrefillMQALogitsKernel.__call__
FP4PrefillMQALogitsKernel.__init__
FP4PrefillMQALogitsKernel._gemm_body
FP4PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP4PrefillMQALogitsKernel._setup_mma
FP4PrefillMQALogitsKernel.kernel
_subview
_target_is_rubin
publish_bank
utccp_required_smem_warp_transpose
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp4_shared_kv_safe_filter.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
FP4PrefillMQALogitsKernel.__call__
FP4PrefillMQALogitsKernel.__init__
FP4PrefillMQALogitsKernel._gemm_body
FP4PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP4PrefillMQALogitsKernel._setup_mma
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp8_fulltile_filter.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
FP8PrefillMQALogitsKernel.__call__
FP8PrefillMQALogitsKernel.__init__
FP8PrefillMQALogitsKernel._gemm_body
FP8PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP8PrefillMQALogitsKernel._setup_mma
FP8PrefillMQALogitsKernel.kernel
_subview
interleaved_partial
predicated_store32
predicated_store64
publish_bank
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp8_glm_mid_loop_filter.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
FP8PrefillMQALogitsKernel.__call__
FP8PrefillMQALogitsKernel.__init__
FP8PrefillMQALogitsKernel._gemm_body
FP8PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP8PrefillMQALogitsKernel._setup_mma
FP8PrefillMQALogitsKernel.kernel
_subview
interleaved_partial
predicated_store32
predicated_store64
publish_bank
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp8_glmflash_queue_recovery.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
FP8PrefillMQALogitsKernel.__call__
FP8PrefillMQALogitsKernel.__init__
FP8PrefillMQALogitsKernel._gemm_body
FP8PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP8PrefillMQALogitsKernel._setup_mma
FP8PrefillMQALogitsKernel.kernel
_subview
publish_bank
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp8_qwen_packed_filter.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
FP8PrefillMQALogitsKernel.__call__
FP8PrefillMQALogitsKernel.__init__
FP8PrefillMQALogitsKernel._gemm_body
FP8PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP8PrefillMQALogitsKernel._setup_mma
FP8PrefillMQALogitsKernel.kernel
_subview
predicated_store32
predicated_store64
publish_bank
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp8_shared_kv_safe_filter.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
FP8PrefillMQALogitsKernel.__call__
FP8PrefillMQALogitsKernel.__init__
FP8PrefillMQALogitsKernel._gemm_body
FP8PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP8PrefillMQALogitsKernel._setup_mma
FP8PrefillMQALogitsKernel.kernel
compile_shared_filter
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/native_bf16_math.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
native_bfadd
native_bffma
native_bfhi
native_bflo
native_bfpack
native_bfpack_relu
native_bfsum_fp32
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/sampler_upper_glmflash_fp4.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
FP4PrefillMQALogitsKernel.__call__
FP4PrefillMQALogitsKernel.__init__
FP4PrefillMQALogitsKernel._gemm_body
FP4PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP4PrefillMQALogitsKernel._setup_mma
FP4PrefillMQALogitsKernel.kernel
_subview
_target_is_rubin
short_margin_rank
utccp_required_smem_warp_transpose
```

### `tensorrt_llm/_torch/cute_dsl_kernels/minimax_m3/moe/m3_moe16_kernel.py`

库存函数 64；已覆盖 0；未覆盖 64。

```text
_add_rn
_adv
_ar_push
_asm
_atom_add_u64
_b1_addr
_bar_arrive
_bf16_bits
_bf16_round
_bf16x2_word
_cfg
_dbg_delay
_div_full
_div_rn
_e2m1x2
_e4m3_bits
_e4m3_to_f32
_e4m3x2
_ex2
_ex2_full
_fence_acq_rel_gpu
_fence_proxy_async_global
_fma_rn
_gtime
_issue_fc1
_issue_fc2
_l2_prefetch
_ld_acquire_u32
_ld_relaxed_u32
_ld_relaxed_u64
_ld_volatile_s32
_match_any
_min_xorsign_abs
_mul_rn
_mx_osc
_mx_words
_nanosleep
_nsegs
_plan_range
_quant_block
_quant_words
_rcp
_red_release_add
_redux_max_u32
_seg
_sfb1_to_tmem
_sh_issue
_shared_cta
_smid
_st_global_b32_if
_st_relaxed_u32
_st_relaxed_u64
_stamp
_sub_rn
_swiglu_quant
_tptr
_tstore
_twiddle
_ue8m0_rp
_wait
_waitd
_waitt
m3_moe
m3_moe_kernel
```

### `tensorrt_llm/_torch/cute_dsl_kernels/minimax_m3/moe/op16.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
M3Moe16State.__init__
M3Moe16State.layer
M3Moe16State.run
_group_slots
make_phase_plan16
make_phase_plan16.<locals>.counts_for
```

### `tensorrt_llm/_torch/cute_dsl_kernels/minimax_m3/mx_gemv/m3_mx_gemv_kernel.py`

库存函数 26；已覆盖 0；未覆盖 26。

```text
_absmax
_atom_map
_bf16_bits
_bf16_round
_bf16x2_word
_bytes_map
_din_load4
_e4m3x4
_k_tile
_ld_shared_v4_f32
_ld_volatile_s32
_mapa_u32
_rcp_approx_ftz
_rope_bits
_st_async_v4
_st_global_b32
_stamp
_swiglu_oai_bf16x2
_test_wait_cluster
a_col0
acc_cols
m3_mx_gemv
m3_mx_gemv_kernel
num_k_tiles
read_globaltimer
tmem_cols
```

### `tensorrt_llm/_torch/distributed/pg_utils.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
split
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/attention.py`

库存函数 15；已覆盖 0；未覆盖 15。

```text
GlmDsaBackend.adopt
GlmDsaBackend.sparse_attn_predict
GlmDsaHooks.adopt
GlmDsaHooks.forward
GlmDsaHooks.forward_custom_op
GlmIndexer._copy_reused_topk
GlmIndexer._fill_decode_topk
GlmIndexer._store_mtp_shared_rows
GlmIndexer.adopt
GlmMLA._adopt_attention_tail
GlmMLA.adopt
GlmMLA.forward
GlmMLA.forward_absorption_generation
_deferred_norm_view
reclass
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/attention_forward.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
_absorbed_query_packet
_deferred_norm_materialize
_forward_dsa_attn
_glm_moe_dsa_mla_dsa_attn_inplace_fake
forward_absorption_generation
forward_absorption_generation.<locals>._mla_gen_rope
forward_generation_sparse_attn
forward_sparse_attn
forward_sparse_attn_custom_op
glm_moe_dsa_mla_dsa_attn_inplace
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/dispatch.py`

库存函数 50；已覆盖 0；未覆盖 50。

```text
Plan.describe
Plan.enabled
Plan.require
Plan.served
TargetPaths.__post_init__
_Step.__enter__
_Step.__exit__
_Step.__init__
_any_of
_bases
_bases.<locals>.visit
_decline_site
_dependents
_log_info
_log_once
_rank_table
_record_decline
_refuse_removed_knobs
_site
_world_allgather
check_rank_uniform
check_rank_uniform_adopted
classify_decline
disable_closure
enabled_features
feature_record
feature_record.<locals>._json
feature_variants
get_impl
install
installed
is_enabled
knob_digest
log_capability
missed
missed_counts
parse_disable
phase
rank_uniform_inputs
read_knobs
register
require_mode
reset_fallback_state
resolve
set_require
shared_env
side_selectors
step
uncovered
validate_mode
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/absorbed_query/kernels.py`

库存函数 31；已覆盖 0；未覆盖 31。

```text
_absorb_job
_absorb_program
_fi_butterfly32
_fi_rmsnorm_rows_exact
_fma_rn
_fp8_ue8m0_scale
_gemv_cols
_iq_norm_gemm
_iq_program
_kpost_program
_kv_program
_ld_w
_ld_wuk
_ld_x
_mul_rn
_norm_gemm
_nx
_prefetch_tile_gemm
_prenorm_gemm
_qb_program
_qpost_program
_rms_rcp
_rms_rcp_partials
_rope_pairs
_sq
_stress_delay
_v8_leaves
hidden_gemv_kernel
qproj_indexer_kernel
qproj_kernel
qproj_prefetch_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/absorbed_query/query_proj.py`

库存函数 24；已覆盖 0；未覆盖 24。

```text
AbsorbedQueryProjection.__init__
AbsorbedQueryProjection.forward_preabsorbed
AbsorbedQueryProjection.fuse_indexer_kw
AbsorbedQueryProjection.hidden_gemv
AbsorbedQueryProjection.indexer_fusable
AbsorbedQueryProjection.launch
AbsorbedQueryProjection.launch_indexer
AbsorbedQueryProjection.launch_prefetch
AbsorbedQueryProjection.native_bridges
AbsorbedQueryProjection.native_bridges.<locals>.aux
AbsorbedQueryProjection.native_bridges.<locals>.main
AbsorbedQueryProjection.native_q_norm
AbsorbedQueryProjection.project
absorb_for
absorb_layout
call_eligibility
check_package_selectors
dsa_proj
indexer_tail_from_projected
module_eligibility
mtp_dsa_proj
selected_form
triton_cdiv
triton_next_pow2
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/absorbed_query/shared_indexer.py`

库存函数 30；已覆盖 0；未覆盖 30。

```text
SharedIndexerProjection.__init__
SharedIndexerProjection._ts_buffer
SharedIndexerProjection.counters_quiescent
SharedIndexerProjection.launch
SharedIndexerProjection.project
_fi_butterfly32
_fi_lane_sumsq
_fi_rmsnorm_kernel
_fi_rmsnorm_rows
_fi_rmsnorm_rows_vec
_fi_rmsnorm_vec_kernel
_l2_prefetch_1k
_ld_sq
_ld_wuk_t
_native_rmsnorm_is_flashinfer_cute
_poll_once
_seq8_at
_shared_query_absorb_job
_shared_query_gemm
_shared_query_gemm_t
_shared_query_ld_wuk
_tree8
_tree8_at
_ts
_v8_leaves
_wait_ge
fi_rmsnorm_reproduction
shared_dsa_proj
shared_module_eligibility
shared_query_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/absorbed_query/shared_indexer_cute.py`

库存函数 35；已覆盖 0；未覆盖 35。

```text
SharedQueryClusterKernel.__call__
SharedQueryClusterKernel.__init__
SharedQueryClusterKernel.kernel
_asm
_bf16_bits
_bf16x2_hi
_bf16x2_lo
_cluster_sync
_copy_weights
_cp_async16
_cp_async_commit
_cp_async_commit_wait_all
_cp_async_wait_group
_fake_tensors
_gaddr
_ld_global_v4
_ld_shared_b32
_ld_shared_v4
_ldsm_x4
_mbar_wait_cluster
_mma_bf16
_mma_chain
_norm_row_to_smem
_pack_bf16x2
_saddr
_st_async_cluster_v4
_st_cluster_v4
_st_global_b16
_st_global_b64
_st_global_v4
_st_shared_b16
_st_shared_v4
_stamp
compiled_kernel
launch
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/cute_fused.py`

库存函数 19；已覆盖 0；未覆盖 19。

```text
_build_fused
_build_fused.<locals>.FusedTail.__call__
_build_fused.<locals>.FusedTail.kernel
_build_fused.<locals>._asm
_build_fused.<locals>.bulk_g2s
_build_fused.<locals>.compute_sync
_build_fused.<locals>.fake
_build_fused.<locals>.fence_gpu
_build_fused.<locals>.global_timer
_build_fused.<locals>.grid_dependency_wait
_build_fused.<locals>.ld_acquire_u64
_build_fused.<locals>.ld_cg_b32
_build_fused.<locals>.ld_relaxed_u64
_build_fused.<locals>.red_add_u64
_build_fused.<locals>.sleep_ns
_build_fused.<locals>.trap
check_fused_shapes
fused_tail
new_state
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/cute_oproj.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
_build
_build.<locals>.StreamOproj.__call__
_build.<locals>.StreamOproj.kernel
_build.<locals>.bulk_g2s
_build.<locals>.grid_dependency_wait
_build.<locals>.grid_launch_dependents
_sm_count
check_shapes
plan
stream_oproj
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/cute_oproj_ring.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
_build
_build.<locals>.RingOproj.__call__
_build.<locals>.RingOproj.kernel
_build.<locals>._asm
_build.<locals>.bulk_g2s
_build.<locals>.fake
_build.<locals>.grid_dependency_wait
_build.<locals>.grid_launch_dependents
_build.<locals>.ld_volatile_u32
_build.<locals>.st_global_b32
_dummy
check_shapes
plan
ring_oproj
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/into_hop.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
_build
_build.<locals>.StreamOprojIntoHop.__call__
_build.<locals>.StreamOprojIntoHop.kernel
_build.<locals>.bulk_g2s
_build.<locals>.grid_dependency_wait
_build.<locals>.grid_launch_dependents
_build.<locals>.ld_volatile_u32
_build.<locals>.st_global_b32
_build.<locals>.st_global_v4_b32
_sm_count
check_shapes
plan
stream_oproj_into_hop
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/kernels.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
_gdc_wait_with_memory_clobber
_oproj_kernel
_prefetch_l2
_wuv_kernel
attn_tail
check_tail_shapes
check_weights
default_state
launch_oproj
launch_wuv
pdl_enabled
pdl_entry_enabled
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/prefetch.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
WeightPrefetch.__init__
WeightPrefetch.fork
WeightPrefetch.forked
WeightPrefetch.join
_aux_stream
_bulk_prefetch_l2
_l2_prefetch_kernel
_nanosleep
check_prefetch_tensors
launch_prefetch
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/attention_output_tail/tail.py`

库存函数 19；已覆盖 0；未覆盖 19。

```text
AttnTail.__init__
AttnTail._check_static
AttnTail._join_prefetch
AttnTail._prefetch_tensors
AttnTail._prefetch_wanted
AttnTail._ring_oproj
AttnTail.arm
AttnTail.before_attention
AttnTail.defer_w_uv
AttnTail.fork_ahead
AttnTail.project
AttnTail.request_into_hop
AttnTail.set_successor_weights
AttnTail.take_into_hop_published
_selected
_static_reason
make_attn_tail
tail_prefetch_config
tail_prefetch_selected
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_layer_expert_prefetch/kernels.py`

库存函数 15；已覆盖 0；未覆盖 15。

```text
_bulk_prefetch_l2
_delay_kernel
_gemm_rank_kernel
_globaltimer
_nanosleep
_prefetch_experts_kernel
_rank_and_stats_kernel
_row_topk_union
_scores_topk_union
_store_ranked
delay
gemm_rank
prefetch_experts
prefetch_experts.<locals>.per_expert_chunks
rank_and_stats
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_layer_expert_prefetch/spec_experts.py`

库存函数 22；已覆盖 0；未覆盖 22。

```text
ExpertPrefetchLink.__init__
ExpertPrefetchLink._eligible
ExpertPrefetchLink._fork
ExpertPrefetchLink._fused_ready
ExpertPrefetchLink._launch_prefetch
ExpertPrefetchLink._predict
ExpertPrefetchLink._prefetch_ready
ExpertPrefetchLink.after_moe
ExpertPrefetchLink.after_post_moe
ExpertPrefetchLink.on_hop
_DeviceState.__init__
_bind_xc
_expert_tensors
_is_moe_with_gate
_prefetch_reason
_state
_xc_hook
_xc_hook.<locals>.hook
_xc_issue_reason
config
join
link_layers
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_rank_topk_cluster_merge/cp_indexer_decode.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_record_decision
cp_decode_topk
cp_topk_from_local_logits
try_cp_indexer_decode
world_size_reason
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_rank_topk_cluster_merge/cp_workspace.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
CpWorkspace.mc_ptr
CpWorkspace.uc_ptr
_create
_key
_local_capability_reason
cluster_reason
failure_reason
get_workspace
op_info
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_rank_topk_global_emit/bridge.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
_check_plan
_same_copy
_same_remap
_same_tensor
copy_reason
new_bridge_packet
predict_copy
predict_remap
remap_topk_to_global
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_rank_topk_global_emit/cp_indexer_decode.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_record_decision
bridge_reason
cp_topk_from_local_logits
try_cp_indexer_decode
world_size_reason
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/cross_rank_topk_global_emit/cp_workspace.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
CpWorkspace.mc_ptr
CpWorkspace.uc_ptr
_create
_key
_local_capability_reason
cluster_reason
failure_reason
get_workspace
op_info
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/decode_prologue/prologue.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
DecodePrologueHook.__call__
DecodePrologueHook.take_embedding
_correction_ineligibility
_embedding_target
_metadata_ineligibility
_replay_host_on_update_kv_lens
_target_prologue_kernel
prologue_ineligibility
take_embedding
target_prologue
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_input_prep/compat.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
declare_overlap_loop
loop_domain
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_input_prep/guard.py`

库存函数 15；已覆盖 0；未覆盖 15。

```text
DebugWindow.__init__
DebugWindow._hook_triton
DebugWindow._hook_triton.<locals>.run
DebugWindow._unhook_triton
DebugWindow.check
DebugWindow.close
DebugWindow.internal
DebugWindow.open
DebugWindow.register
DebugWindow.verify
_GuardMode.__init__
_GuardMode.__torch_dispatch__
_call_site
_reads_only_metadata
tensor_span
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_input_prep/packet.py`

库存函数 23；已覆盖 0；未覆盖 23。

```text
PrepPacket.__init__
PrepPacket._alloc_payload
PrepPacket._check_dst
PrepPacket._check_dsts
PrepPacket._dst_ptr
PrepPacket._emit_split
PrepPacket._num_chunks
PrepPacket._reserve
PrepPacket.abort
PrepPacket.add_block_offsets
PrepPacket.add_clamp_min
PrepPacket.add_copy
PrepPacket.add_gather
PrepPacket.add_values
PrepPacket.add_zero
PrepPacket.begin
PrepPacket.commit
PrepPacket.num_descriptors
PrepPacket.set_completion_event
_NeverDone.query
_PinnedSlot.__init__
_PinnedSlot.free
_prep_commit_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_input_prep/paged_mqa_schedule.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
paged_mqa_logits_schedule
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_input_prep/sampler_store.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
SamplerStoreWriter.__init__
SamplerStoreWriter._slot_buffer
SamplerStoreWriter.set_completion_event
SamplerStoreWriter.store_and_copy_out
SamplerStoreWriter.supported
SamplerStoreWriter.tensors_supported
_sampler_store_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_input_prep/stager.py`

库存函数 34；已覆盖 0；未覆盖 34。

```text
PrepStager.__init__
PrepStager._add
PrepStager._apply
PrepStager._detach
PrepStager._flush
PrepStager._ineligible_reason
PrepStager._internal
PrepStager._native
PrepStager._scheduler_metadata
PrepStager._stage
PrepStager._usable_static_inputs
PrepStager._zero_except
PrepStager.begin
PrepStager.begin_if_eligible
PrepStager.block_offsets
PrepStager.clamp_min
PrepStager.commit
PrepStager.copy
PrepStager.defer
PrepStager.discard
PrepStager.end
PrepStager.flush
PrepStager.gather
PrepStager.overlap_inputs
PrepStager.position_ids
PrepStager.scheduler_metadata
PrepStager.set_completion_event
PrepStager.static_inputs_tokens
PrepStager.values
PrepStager.zero
_host_memory_pinned
_log_prep_decision
_stager_internal
_stager_internal.<locals>.wrapper
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/hop.py`

库存函数 18；已覆盖 0；未覆盖 18。

```text
DeferredNormHop.__call__
DeferredNormHop.__init__
DeferredNormHop._contract_violations
DeferredNormHop._launch
DeferredNormHop._post_moe_routed_mode
DeferredNormHop._precompile
DeferredNormHop._routed_view
DeferredNormHop._vector_layout
DeferredNormHop.can_run
DeferredNormHop.can_run_routed
DeferredNormHop.enable_routed_input
DeferredNormHop.hop_routed
DeferredNormHop.launch
DeferredNormHop.launch_routed
DeferredNormHop.materialize
_graph_mode_ok
_vector_ready
materialize
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/hop_kernel.py`

库存函数 15；已覆盖 0；未覆盖 15。

```text
DeferredNormHopKernel.__call__
DeferredNormHopKernel.__init__
DeferredNormHopKernel.kernel
DeferredNormHopKernel.partial_count
DeferredNormMaterializeKernel.__call__
DeferredNormMaterializeKernel.__init__
DeferredNormMaterializeKernel.kernel
_globaltimer_st
_hop_stamp
_ld_f32
_st_b32
_st_f32_if
compiled_hop
compiled_materialize
hop_kernel_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/packet.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
DeferredNormHidden.device
DeferredNormHidden.dtype
DeferredNormHidden.materialize
DeferredNormHidden.num_rows
DeferredNormHidden.shape
frag_elems
frag_from_rows
is_deferred
rows_from_frag
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/proj_kernel.py`

库存函数 34；已覆盖 0；未覆盖 34。

```text
DeferredNormProjKernel.__call__
DeferredNormProjKernel.__init__
DeferredNormProjKernel._issue_copies
DeferredNormProjKernel._prefetch_slice
DeferredNormProjKernel._wait_staged
DeferredNormProjKernel.kernel
DeferredNormProjKernel.kernel.<locals>.load_a
_asm
_bulk_g2s
_fence_proxy_async
_globaltimer
_ld_b32
_ld_nc_b32
_ld_shared_f32
_ld_v2
_ld_v2_if
_ldsm_x2
_mbar_expect_tx
_mbar_fence_init
_mbar_init
_mbar_wait
_min_i32
_prefetch_l2
_row_src
_sel_i32
_sel_i64
_shfl_idx_f32
_st_global_b64
_st_shared_f32
_stamp
_store_out
compiled_projection
is_compiled
projection_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/projection.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
DeferredNormProjection.__init__
DeferredNormProjection.kernel_config
DeferredNormProjection.precompile
DeferredNormProjection.project
DeferredNormProjection.run
DeferredNormProjection.served_config
DeferredNormProjection.set_prefetch
DeferredNormProjection.supported
_kernel_config
_sm_count
precompile_served
projection_eligibility
served_kernel_configs
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/ptx.py`

库存函数 36；已覆盖 0；未覆盖 36。

```text
_asm
_struct
_unpack
_v
add
arrive
bf16_bits
bits_f32
cp_async16
cp_async_commit
cp_async_wait_group
error_word_address
fault_delay
flag_error
flag_timeout
fma
inv_rms
ld_v4
ldsm_x4
load_state
mma_bf16
mul
pack_bf16x2
perm_check
poison
poll8
round_bf16
saddr
sanitize
select3
select_f32
shfl_xor
st_global_b16
st_shared_v4
st_v4
unpack_bf16x2
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/query_entry.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_absorbed_query_runner
_deferred_norm_runner
_shared_query_runner
deferred_norm_dsa_proj
indexer_query_from_projected
indexer_query_from_projected.<locals>.aux
indexer_query_from_projected.<locals>.main
runner_max_rows
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm/scope.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
constructing_layer
decline_reason
is_full_indexer_layer
next_layer_kind
selected_scope
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm_evict_hints/entry.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
_evict_hints_runner
evict_hints_dsa_proj
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm_evict_hints/hop.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
EvictHintsDeferredHop.__init__
EvictHintsDeferredHop._compiled
EvictHintsDeferredHop._launch
EvictHintsDeferredHop._precompile
EvictHintsDeferredHop.enable_routed_input
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm_evict_hints/hop_kernel.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
EvictHintsHopKernel.__call__
EvictHintsHopKernel.__init__
EvictHintsHopKernel._launch_kernel
EvictHintsHopKernel._prefetch_own
EvictHintsHopKernel._prefetch_own3
EvictHintsHopKernel._prefetch_poll
EvictHintsHopKernel._prefetch_three
EvictHintsHopKernel.call_raw
EvictHintsHopKernel.kernel
_smid_stamp
compiled_evict_hints_hop
set_library_carveout
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm_evict_hints/proj_kernel.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
EvictHintsProjKernel.__call__
EvictHintsProjKernel.__init__
EvictHintsProjKernel._compute
EvictHintsProjKernel._compute.<locals>.load_a
EvictHintsProjKernel._copy_chunk
EvictHintsProjKernel._copy_one
EvictHintsProjKernel._defer_point
EvictHintsProjKernel._issue_copies
EvictHintsProjKernel._produce
EvictHintsProjKernel._pw_stamp
EvictHintsProjKernel.kernel_pw
compiled_evict_hints_projection
is_compiled
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm_evict_hints/projection.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
EvictHintsProjection.__init__
EvictHintsProjection.kernel_config
EvictHintsProjection.precompile
EvictHintsProjection.run
_evict_hints_kernel_config
config_for
parse_cfg
precompile_served
served_kernel_configs
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/deferred_norm_evict_hints/ptx.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
_asm
_v
bar_sync_named
bulk_g2s_hint
globaltimer
policy_evict_first
policy_evict_last
prefetch_l2
prefetch_line_l2
smid
spin_until
st_b64
st_v4_hint
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/dense_mlp/dense_mlp.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_linear_is_plain_bf16
_output_allreduce_requested
check_eligibility
dense_mlp_forward
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_front/draft_front.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
_aligned
_enable_pdl
_row_operand_reason
pack_draft_front
plan_segments
unsupported_reason
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_front/kernels.py`

库存函数 16；已覆盖 0；未覆盖 16。

```text
DraftFrontPackKernel.__call__
DraftFrontPackKernel.__init__
DraftFrontPackKernel._normalize_segment
DraftFrontPackKernel._tv_layout
DraftFrontPackKernel.kernel
DraftFrontPackKernel.smem_bytes
_column_predicate
_compile_kernel
_compile_kernel.<locals>.row_major_fake
_compile_kernel.<locals>.weight_fake
_row_reduce_sum
_torch_to_cutlass_dtype
_warp_reduce_sum
get_compiled_kernel
threads_per_block
threads_per_row
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_front/native_norm.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_flashinfer_path_reason
flashinfer_path_reason
native_norm_reason
norm_module_reason
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_live_rows/hop.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
LiveRowHop.__call__
LiveRowHop.__init__
LiveRowHop._agreed
LiveRowHop.call_ineligibility
LiveRowHop.can_run
LiveRowHop.for_post_attn_hop
_count
engagement_counts
required_sms
reset_engagement_counts
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_live_rows/step0.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
Step0LiveRows.__init__
Step0LiveRows.armed
select_step0
step0_ineligibility
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_loop/draft_loop.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
DraftLoopOps.__init__
DraftLoopOps.accept
DraftLoopOps.finalize
DraftLoopOps.first_drafter_inputs
DraftLoopOps.ineligibility
DraftLoopOps.step0_rows
DraftLoopOps.step_update
DraftLoopOps.step_update.<locals>.run_draft_loop
_accept_kernel
_finalize_kernel
_gather_rows_kernel
gather_rows
make_draft_loop_ops
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_loop/draft_step.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_draft_step_kernel
_schedule_probe_kernel
_write_schedule
draft_step_update
paged_mqa_schedule
schedule_self_check
step_ineligibility
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_loop/embedding.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
DraftEmbedding.__call__
DraftEmbedding.__getattr__
DraftEmbedding.__init__
DraftEmbedding.native
_draft_embedding_kernel
draft_embedding
ids_ineligibility
module_ineligibility
wrap_draft_embedding
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_loop/verify.py`

库存函数 15；已覆盖 0；未覆盖 15。

```text
_Counters.__init__
_Counters.add_mismatch
_Counters.add_one
_Counters.ensure_pool
_Counters.get
_Counters.report
_bits
_device_fields
_native_step_tail
_report_at_exit
verified_step_update
verify_accept
verify_enabled
verify_finalize
verify_rows
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_routed_moe/routed_moe.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
_Scratch.__init__
_owner_reason
_pdl_enabled
_routing_reason
_scratch
_wrapper_reason
eligibility
launch_config
resolve_owner
routed_output
run
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/draft_tail/draft_tail.py`

库存函数 22；已覆盖 0；未覆盖 22。

```text
DraftTailOptions.served
DraftTailOptions.shape
DraftTailOptions.validate
DraftTailRunner.__init__
DraftTailRunner._agree
DraftTailRunner._allocate_exchange
DraftTailRunner._common_args
DraftTailRunner.conform
DraftTailRunner.exchange_mode
DraftTailRunner.prepare
DraftTailRunner.ready
DraftTailRunner.run_from_logits
DraftTailRunner.run_fused
DraftTailRunner.select
DraftTailRunner.step
_head_is_plain_vocab_shard
_kernel_layout_ok
_max_draft_rows
_shared_step_reason
get_runner
select_draft_tail
select_draft_tail.<locals>.tail
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/fused_kv_indexer_projection/entry.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_fused_kv_runner
_host_runner
_max_rows
fused_kv_dsa_proj
kind_selected
layer_kind
served_layer_kinds
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/fused_kv_indexer_projection/proj_kernel.py`

库存函数 17；已覆盖 0；未覆盖 17。

```text
FusedKvProjKernel.__call__
FusedKvProjKernel.__init__
FusedKvProjKernel._issue_copies
FusedKvProjKernel._mma
FusedKvProjKernel._mma.<locals>.load_a
FusedKvProjKernel._mma.<locals>.load_b
FusedKvProjKernel._mma.<locals>.load_d
FusedKvProjKernel._mma.<locals>.stash
FusedKvProjKernel.kernel
_ordered_sum
_row_src
_stamp
_store_out
compiled_projection
is_compiled
projection_supported
split_ctas
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/fused_kv_indexer_projection/projection.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
FusedKvProjConfig.chunk_steps
FusedKvProjConfig.parse
FusedKvProjection.__init__
FusedKvProjection.kernel_config
FusedKvProjection.precompile
FusedKvProjection.run
FusedKvProjection.supported
_sm_count
_tensor_map_words
kernel_config
projection_eligibility
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/fused_kv_indexer_projection/ptx.py`

库存函数 35；已覆盖 0；未覆盖 35。

```text
_asm
_struct
_unpack
add_f32
bf16_bits
bulk_g2s
cp_async16
cp_async_mbar_arrive
fence_tensormap_acquire
globaltimer
ld_ca_b32
ld_ca_b32_if
ld_cg_b32
ld_cg_b32_if
ld_nc_b32
ld_shared_f32
ldsm_x2
ldsm_x4
mbar_arrive_expect_tx
mbar_fence_init
mbar_init
mbar_wait
min_i32
mma_bf16
prefetch_l2
saddr
sel_i32
sel_i64
st_global_b16
st_global_b64
st_shared_f32
st_shared_v4
sync_warp
tensor_g2s_3d
xor_i32
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/hop_routing/hop.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
HopRoutingTransport.__call__
HopRoutingTransport.__init__
HopRoutingTransport.can_run
HopRoutingTransport.create
_GatherState.__init__
_count
_gather_state
engagement_counts
reset_engagement_counts
split_routing
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/hop_routing/routed_moe.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
HopRoutedMoe._compiled
HopRoutedMoe.ready_for_capture
_engaged_runner
_kernel_class
_kernel_class.<locals>.HopRoutedExpertClusterKernel._route_tokens
_kernel_class.<locals>.ld_nc_u32
routed_output
routed_output_into_hop
routing_reason
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/host_page_retention/host_policy.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
MimallocOptions.__init__
MimallocOptions._check_table
MimallocOptions._descriptor
MimallocOptions.get
MimallocOptions.index
MimallocOptions.set
_mapped_base
_read_symbols
_read_symbols.<locals>.read_section
_unique
apply_purge_policy
install_host_policy
torch_libc10_path
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_layer_front/cp_indexer_decode.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_modules
_record_decision
_try
cp_decode_topk
try_cp_indexer_decode
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_layer_front/cp_logits_runner.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_compile
_key
_num_sms
_unused_schedule
ensure_compiled
paged_mqa_logits_cp
use_pdl
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_layer_front/fp8_paged_mqa_logits_cp.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
FP8MQALogitsCPZ14Kernel.__call__
FP8MQALogitsCPZ14Kernel.__init__
FP8MQALogitsCPZ14Kernel._setup_mma
FP8MQALogitsCPZ14Kernel.kernel
add_f16x2
fma_f16x2
max_f16x2
pack_f16x2
unpack_f16x2
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_layer_front/kernels.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
_front_absorb_job
_front_kpost_program
_front_kv_program
_front_norm_gemm
_front_qpost_program
_front_rmsnorm_rows
_front_rstd
_prefetch_w_tile
_trigger
_wait_norm
front_qproj_indexer_kernel
grid_size
num_norm_programs
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_layer_front/query_proj.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
IndexerFrontQueryProjection.__init__
IndexerFrontQueryProjection.bridges
IndexerFrontQueryProjection.bridges.<locals>.aux
IndexerFrontQueryProjection.bridges.<locals>.main
IndexerFrontQueryProjection.launch_front
IndexerFrontQueryProjection.project_front
IndexerFrontQueryProjection.supported
_native_q_norm_reproduced
_runner
config_for
dsa_proj
front_module_eligibility
reuses_shared_topk
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_layer_front/rope.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
mla_rope_generation
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/cross_rank_indexer_decode.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_record_decision
cp_decode_topk
cp_topk_from_local_logits
try_cp_indexer_decode
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/cross_rank_logits_runner.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_compile
_num_sms
_unused_schedule
ensure_compiled
local_capacity
local_length
local_to_global
paged_mqa_logits_cp
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/cross_rank_policy.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
decide_cp
decide_cp_uniform
rank_local_reason
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/cross_rank_workspace.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
CpWorkspace.mc_ptr
CpWorkspace.uc_ptr
_create
_key
_local_capability_reason
failure_reason
get_workspace
op_info
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/fp8_paged_mqa_logits_cross_rank.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
FP8MQALogitsCPKernel.__call__
FP8MQALogitsCPKernel.__init__
FP8MQALogitsCPKernel._setup_mma
FP8MQALogitsCPKernel.kernel
add_f16x2
fma_f16x2
max_f16x2
pack_f16x2
unpack_f16x2
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/fp8_paged_mqa_logits_rows.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
FP8MQALogitsKernel.__call__
FP8MQALogitsKernel.__init__
FP8MQALogitsKernel._setup_mma
FP8MQALogitsKernel.kernel
add_f16x2
fma_f16x2
max_f16x2
pack_f16x2
unpack_f16x2
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/fp8_paged_mqa_logits_rowsplit.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
FP8MQALogitsRowSplitKernel.__call__
FP8MQALogitsRowSplitKernel.__init__
FP8MQALogitsRowSplitKernel._setup_mma
FP8MQALogitsRowSplitKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/indexer_decode.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_record_decision
decode_topk
paged_mqa_logits
try_indexer_decode
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/logits_runner.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_compile
_num_sms
paged_mqa_logits_rows
rowsplit_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/policy.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
decide
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/region.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
cross_rank_topk_decode
indexer_topk_decode
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/indexer_topk/topk_select.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
RadixTopKWorkspace.__init__
_digit_pass
_fence_all
_final_select
_hist_kernel
_keys
_load_source
_row_len
_select_kernel
default_chunks
get_workspace
radix_topk_decode
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/oproj_weight_staging/cute_oproj_stage.py`

库存函数 23；已覆盖 0；未覆盖 23。

```text
_build
_build.<locals>.StreamOprojStage.__call__
_build.<locals>.StreamOprojStage.kernel
_build.<locals>.bulk_g2s
_build.<locals>.bulk_g2s_evict_first
_build.<locals>.copy_row
_build.<locals>.globaltimer
_build.<locals>.grid_dependency_wait
_build.<locals>.grid_launch_dependents
_build.<locals>.ld_volatile_u32
_build.<locals>.prefetch_l2
_build.<locals>.prefetch_l2_evict_last
_build.<locals>.st_global_b32
_build.<locals>.st_global_b64
_build.<locals>.st_global_v4_b32
_build.<locals>.stage_row
_build.<locals>.stamp_addr
_dummies
_sm_count
check_shapes
check_stage
plan
stream_oproj_stage
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/oproj_weight_staging/kernels.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
_gdc_wait_with_memory_clobber
_prefetch_l2
_prefetch_l2_evict_last
_wuv_stage_kernel
check_stage_config
check_tail_config
launch_wuv_stage
stage_tail
wuv_stage_geometry
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/oproj_weight_staging/tail.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
StagedAttnTail.__init__
StagedAttnTail._check_static
StagedAttnTail.project
config_reason
geometry_reason
wrap_attn_tail
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_attention_hop/__init__.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
_into_hop_engagement_counts
_make_into_hop_transport
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_attention_hop/from_partials.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
IntoHopTransport.__call__
IntoHopTransport.__init__
IntoHopTransport.can_run
IntoHopTransport.create
_count
into_hop_engagement_counts
reset_into_hop_engagement_counts
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_attention_hop/hop.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
PostAttnHop.__call__
PostAttnHop.__init__
PostAttnHop.call_ineligibility
PostAttnHop.can_run
PostAttnHop.get_or_create
PostAttnHop.reset_protocol
_count
engagement_counts
native_allreduce_ineligibility
required_sms
reset_engagement_counts
static_ineligibility
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_moe_hop/cute_kernel.py`

库存函数 34；已覆盖 0；未覆盖 34。

```text
PostMoeHopClusterKernel.__call__
PostMoeHopClusterKernel.__init__
PostMoeHopClusterKernel.kernel
_add
_arrive
_asm
_bits_f32
_error_word
_fault_delay
_flag_error
_flag_timeout
_fma
_inv_rms
_ld_shared_f32
_ld_v4
_mul
_pack_bf16x2
_perm_check
_poison
_poll8
_round_bf16
_sanitize
_select
_shfl_xor
_st_cluster_f32
_st_v4
_struct
_unpack
_unpack_bf16x2
_v
cluster_kernel_supported
compiled_cluster_hop
launch_cluster_hop
launch_cluster_hop_routed
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_moe_hop/health.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
PostMoeHopHealth.errors
check_health
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_moe_hop/hop.py`

库存函数 23；已覆盖 0；未覆盖 23。

```text
PostMoeHop.__call__
PostMoeHop.__init__
PostMoeHop._contract_violations
PostMoeHop._needs_routed_buffer
PostMoeHop._precompile_cluster_kernel
PostMoeHop._routed_buffer_problem
PostMoeHop._routed_device_buffer
PostMoeHop._routed_view
PostMoeHop._vector_layout
PostMoeHop.can_run
PostMoeHop.enable_routed_input
PostMoeHop.error_bits
PostMoeHop.hop_routed
PostMoeHop.raise_if_error
PostMoeHop.reset
PostMoeHop.routed_acc_buffer
PostMoeHop.routed_input_buffer
PostMoeHop.routed_raw_buffers
PostMoeHop.selected_routed_mode
_local_capability
_vector_ready
native_finalize_reference
producer_capability
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_moe_hop/kernels.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
_fault_delay
_globaltimer
_post_moe_hop_kernel
_program_clock
_trace
launch_post_moe_hop
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/post_moe_hop/workspace.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
PostMoeHopWorkspace.__init__
PostMoeHopWorkspace._reset_local
PostMoeHopWorkspace.calls
PostMoeHopWorkspace.comm_for
PostMoeHopWorkspace.counters
PostMoeHopWorkspace.error_bits
PostMoeHopWorkspace.get_or_create
PostMoeHopWorkspace.instances
PostMoeHopWorkspace.reset
_all_ranks_agree
_workspace_comm
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/rope_kv_append/ops.py`

库存函数 15；已覆盖 0；未覆盖 15。

```text
_consume
_sm_ok
_static_rope_domain
bridge_eligible
check_indexer_k_scatter_consumed
defer_indexer_k_scatter
defer_mtp_row_stash
defer_reuse_copy
defer_shared_copy
mla_rope_generation
new_bridge_packet
remap_topk_to_global
rope_eligible
run_indexer_k_scatter_native
topk_bridge
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/routed_moe/kernel.py`

库存函数 25；已覆盖 0；未覆盖 25。

```text
RoutedMoeKernel.__init__
RoutedMoeKernel._cta_of
RoutedMoeKernel._delay_spin
RoutedMoeKernel._epilogue
RoutedMoeKernel._fc1_tile_epilogue
RoutedMoeKernel._fc2_mma_ring
RoutedMoeKernel._fetch_fc2_operands
RoutedMoeKernel._fetch_fc2_operands_ll
RoutedMoeKernel._issue_prefetches
RoutedMoeKernel._ll_finish
RoutedMoeKernel._ll_issue
RoutedMoeKernel._ll_publish
RoutedMoeKernel._mma_issuer
RoutedMoeKernel._prefetch_fc1
RoutedMoeKernel._prefetch_fc2
RoutedMoeKernel._quant_rows
RoutedMoeKernel._quantize_x
RoutedMoeKernel._route_one
RoutedMoeKernel._route_tokens
RoutedMoeKernel._s2t_partition
RoutedMoeKernel._setup
RoutedMoeKernel._tma_producer
RoutedMoeKernel.kernel
RoutedMoeKernel.run
_stamp
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/routed_moe/kernel_xc.py`

库存函数 18；已覆盖 0；未覆盖 18。

```text
RoutedMoeExpertClusterKernel.__init__
RoutedMoeExpertClusterKernel._quant_rows_xs
RoutedMoeExpertClusterKernel._setup
RoutedMoeExpertClusterKernel._swiglu_quant_to_stage
RoutedMoeExpertClusterKernel._xc_delay
RoutedMoeExpertClusterKernel._xc_epilogue
RoutedMoeExpertClusterKernel._xc_finalize_last_arriver
RoutedMoeExpertClusterKernel._xc_l2_prefetch
RoutedMoeExpertClusterKernel._xc_mma_issuer
RoutedMoeExpertClusterKernel._xc_quantize_x
RoutedMoeExpertClusterKernel._xc_spec_prefetch
RoutedMoeExpertClusterKernel._xc_spec_store
RoutedMoeExpertClusterKernel._xc_tma_producer
RoutedMoeExpertClusterKernel.kernel
RoutedMoeExpertClusterKernel.run
RoutedMoeFinalizeKernel.__init__
RoutedMoeFinalizeKernel.kernel
RoutedMoeFinalizeKernel.run
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/routed_moe/ptx.py`

库存函数 61；已覆盖 0；未覆盖 61。

```text
_asm
atom_add_relaxed_gpu_u32
atom_or_shared_u32
bar_arrive
bar_sync
bf16_bits
bf16_round
bf16x2_to_f32
bits_f32
bulk_g2s
bulk_s2cluster
clock64
cluster_ctarank
cvt_e2m1x2
cvt_e2m1x8
cvt_e4m3_rn_sat
e4m3_to_f32
ex2_approx
f32_bits
fence_acq_rel_gpu
fence_proxy_async_global
fence_proxy_async_shared
fence_proxy_async_shared_cluster
fma_rn
globaltimer_u64
ld_acquire_gpu_u32
ld_ll2
ld_nc_f32
ld_nc_u16
ld_nc_v4_u32
ld_relaxed_gpu_u32
ld_v4_u32
lds_u32
lds_u64
lds_v2_u32
lds_v4_u32
mapa_u32
mbar_arrive_expect_tx
mbar_arrive_remote
pack_bf16x2
popc_u32
prefetch_l2_bulk
rcp_approx_ftz
red_add_f32
redux_max_u32
smid_u32
st_cluster_u32
st_cluster_v4_u32
st_f32
st_ll1
st_ll2
st_relaxed_gpu_u32
st_u16
st_u32
st_u64
st_v4_u32
sts_u32
sts_u64
sts_u8
sts_v2_u32
tmem_ld_32x32b_x8
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/routed_moe/routed_moe.py`

库存函数 31；已覆盖 0；未覆盖 31。

```text
RoutedMoe.__call__
RoutedMoe.__init__
RoutedMoe._compiled
RoutedMoe._key
RoutedMoe._pointers
RoutedMoe._spec
RoutedMoe._xc_ctas
RoutedMoe.from_native
RoutedMoe.inkernel_finalize
RoutedMoe.ready_for_capture
RoutedMoe.schedule
_Workspace.__init__
_Workspace.nbytes
_Workspace.order_on
_Workspace.tensors
_bulk_copy_ready
_check_vector
_engaged_runner
_logits_from_hop
_max_active_clusters
_owner_reason
_param
_routing_reason
_sm_count
_workspace
_wrapper_reason
eligibility
resolve_expert_owner
routed_output
routed_output_into_hop
xc_kernel_args
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/router_gemm/hop.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
_layer_domain
engagement_counts
make_moe_router
make_moe_router.<locals>._route
pre_moe_hop
router_logits
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/router_gemm/router_kernel.py`

库存函数 24；已覆盖 0；未覆盖 24。

```text
_build_kernel_class
_build_kernel_class.<locals>.RouterGemmKernel.__call__
_build_kernel_class.<locals>.RouterGemmKernel.__init__
_build_kernel_class.<locals>.RouterGemmKernel.kernel
_build_kernel_class.<locals>._asm
_build_kernel_class.<locals>.accumulate
_build_kernel_class.<locals>.add_rn
_build_kernel_class.<locals>.bf16x2_to_f32x2
_build_kernel_class.<locals>.cp_async_16
_build_kernel_class.<locals>.cp_async_commit
_build_kernel_class.<locals>.cp_async_wait
_build_kernel_class.<locals>.cp_async_wait_all
_build_kernel_class.<locals>.fma_rn
_build_kernel_class.<locals>.fma_rn_x2
_build_kernel_class.<locals>.griddep_launch_dependents
_build_kernel_class.<locals>.griddep_wait
_build_kernel_class.<locals>.lds_v4
_build_kernel_class.<locals>.widen
_get_compiled
_smem_bytes
fit_experts_per_cta
pdl_enabled
router_gemm
router_gemm_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/shared_expert_prefetch/kernel_xc.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
SharedPrefetchExpertClusterKernel.__init__
SharedPrefetchExpertClusterKernel._se_prefetch
SharedPrefetchExpertClusterKernel.kernel
SharedPrefetchExpertClusterKernel.run
_chunk
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/shared_expert_prefetch/routed_moe.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
SharedPrefetchRoutedMoe.__call__
SharedPrefetchRoutedMoe._compiled
SharedPrefetchRoutedMoe._key
SharedPrefetchRoutedMoe._xc_ctas
SharedPrefetchRoutedMoe.ready_for_capture
_engaged_runner
routed_output_into_hop
shared_weights
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/kernels/sparse_mla_decode/sparse_mla.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
SparseMlaDecodeFmha._is_available
SparseMlaDecodeFmha._is_supported
SparseMlaDecodeFmha.forward
device_supported
engagement_snapshot
pool_base_pointer
request_ineligibility
sparse_mla_decode
static_ineligibility
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/layers.py`

库存函数 27；已覆盖 0；未覆盖 27。

```text
GlmDecoderLayer.adopt
GlmDecoderLayer.forward
GlmDecoderLayer.forward_MoE
GlmDecoderLayer.forward_MoE.<locals>._run_MoE
GlmDecoderLayer.forward_decode
GlmDecoderLayer.forward_mlp
GlmMoE.adopt
GlmMoE.compute_routed_output
GlmMoE.forward
GlmMoE.forward.<locals>._compute_routed_output
GlmMoE.forward.<locals>._compute_shared_output
GlmMtpLayer._adopt_draft
GlmMtpLayer.adopt
GlmMtpLayer.arm_live_rows
GlmMtpLayer.disarm_live_rows
GlmMtpLayer.forward
GlmMtpLayer.forward.<locals>.norm_embeds
GlmMtpLayer.forward.<locals>.norm_hidden
_AdoptedLayerMixin._adopt_fast_paths
_AdoptedLayerMixin._adopt_post_moe
_AdoptedLayerMixin._adopt_pre_moe
_AdoptedLayerMixin._adopt_tail_prefetch
_PostAttentionHopMixin._can_use_post_attn_hop
_PostAttentionHopMixin._post_attn_hop
_PostAttentionHopMixin._post_attn_router_weight
adopt_model
adopted_state
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/modeling.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
DecodeTarget.__init__
DecodeTarget.forward
GlmMoeDsaModelingV2._select_target
GlmMoeDsaModelingV2.adopt
GlmMoeDsaModelingV2.forward
ModelingV2Glm52Nvfp4Sm103Tp8.__init__
ModelingV2Glm52Nvfp4Sm103Tp8.forward
ModelingV2Glm52Nvfp4Sm103Tp8.load_weights
ModelingV2Glm52Nvfp4Sm103Tp8.runtime_hooks
ModelingV2Glm52Nvfp4Sm103Tp8.setup_aliases
PrefillTarget.__init__
PrefillTarget.forward
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/runtime.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
GlmExecutorEvents.__init__
GlmExecutorEvents.after_warmup
GlmExecutorEvents.at_shutdown
GlmExecutorEvents.before_overlap_loop
make_runtime_hooks
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/spec_worker.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
GlmMtpWorker._forward_impl
GlmMtpWorker._forward_linear_draft_loop
GlmMtpWorker._run_draft_forward
GlmMtpWorker.adopt
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/weights.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
load
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_dsa.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_forward_dsa_attn
_get_short_seq_mha_threshold
forward_context_sparse_attn
forward_generation_sparse_attn
forward_sparse_mla_kvcache_bf16
initialize_sparse_attn
should_use_short_mha
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_dsa_custom_ops.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_extract_extra_attrs
_mla_dsa_attn_inplace_fake
mla_dsa_attn_inplace
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_mla.py`

库存函数 59；已覆盖 0；未覆盖 59。

```text
Glm52MLA.__init__
Glm52MLA.__init__.<locals>.yarn_get_mscale
Glm52MLA._apply_output_gate
Glm52MLA._attention_scaling
Glm52MLA._attention_scaling.<locals>._get_attn_scale
Glm52MLA._attn_forward_gen
Glm52MLA._bind_helix_v_up_weight
Glm52MLA._bind_v_b_proj_weight
Glm52MLA._bmm_bf16_out
Glm52MLA._build_helix_v_up_weight
Glm52MLA._copy_absorption_weight_shards
Glm52MLA._create_outputs
Glm52MLA._forward_custom_op
Glm52MLA._forward_impl
Glm52MLA._get_kvbproj_head_weights
Glm52MLA._has_full_absorption_weights
Glm52MLA._helix_project_v_partials
Glm52MLA._init_fused_q_fp8_context
Glm52MLA._project_context_kv
Glm52MLA._project_output
Glm52MLA._project_output_impl
Glm52MLA._project_q_by_phase
Glm52MLA._q_a_layernorm_maybe_fused
Glm52MLA._rebuild_locality_domain_bf16_bmm_weight_shards
Glm52MLA._resolve_qa_fused_scale
Glm52MLA._use_fused_q_fp8_context
Glm52MLA._use_kvbproj_strided_out
Glm52MLA.apply_rope
Glm52MLA.cache_derived_state
Glm52MLA.cached_warmup_forward_context_with_cached_kv
Glm52MLA.cached_warmup_forward_context_with_cached_kv.<locals>.warmup
Glm52MLA.create_output
Glm52MLA.create_weights
Glm52MLA.forward
Glm52MLA.forward_absorption_context
Glm52MLA.forward_absorption_generation
Glm52MLA.forward_absorption_generation.<locals>._mla_gen_rope
Glm52MLA.forward_context
Glm52MLA.forward_context_default
Glm52MLA.forward_context_with_cached_kv
Glm52MLA.forward_context_with_chunked_prefill
Glm52MLA.forward_context_with_chunked_prefill_fp4
Glm52MLA.forward_context_with_chunked_prefill_fp4.<locals>.expand_kv
Glm52MLA.forward_context_with_chunked_prefill_fp4.<locals>.partition_forward_args
Glm52MLA.forward_impl
Glm52MLA.post_load_weights
Glm52MLA.resmooth_parameters
Glm52MLA.transform_weights
_create_attention
_create_mla_outputs_fake
_cute_dsl_fp8_bmm_out
_extract_mla_extra_attrs
_get_sparse_mla_hooks
_is_cute_dsl_fp8_bmm_available
_slice_hidden_states_to_num_tokens
create_mla_outputs
create_mla_outputs_impl
fp8_block_scaling_bmm_out
mla_custom_op_inplace
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_model.py`

库存函数 33；已覆盖 0；未覆盖 33。

```text
Glm52Attention.__init__
Glm52Attention.cache_derived_state
Glm52DecoderLayer.__init__
Glm52DecoderLayer._apply_next_layer_layernorm
Glm52DecoderLayer._compute_mlp_tp_size
Glm52DecoderLayer._get_decoder_layer_quant_config
Glm52DecoderLayer._moe_tail_fusion_applies
Glm52DecoderLayer.forward
Glm52DecoderLayer.forward_MoE
Glm52DecoderLayer.forward_MoE.<locals>._run_MoE
Glm52DecoderLayer.forward_mlp
Glm52Gate.__init__
Glm52Gate.apply
Glm52Gate.forward
Glm52Gate.get_experts_per_token
Glm52Gate.load_weights
Glm52Gate.routing_method
Glm52Linear.__init__
Glm52Linear.apply_linear
Glm52MoE.__init__
Glm52MoE._compute_shared_expert_tp_size
Glm52MoE._expert_weight_loading_mode
Glm52MoE._get_experts_quant_config
Glm52MoE._get_shared_experts_quant_config
Glm52MoE.compute_routed_output
Glm52MoE.forward
Glm52MoE.forward.<locals>._compute_routed_output
Glm52MoE.forward.<locals>._compute_shared_output
Glm52Model.__init__
Glm52Model.forward
_static_nvfp4_input_scale
moe_reduce_add_shared_output
setup_aliases
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_mtp.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
Glm52MTP.__init__
Glm52MTP._fused_draft_glue
Glm52MTP._resolve_draft_glue
Glm52MTP.draft_sample_glue_window
Glm52MTP.forward
Glm52MTP.forward.<locals>.norm_embeds
Glm52MTP.forward.<locals>.norm_hidden
Glm52MTPForCausalLM.__init__
Glm52MTPForCausalLM._quantization_named_modules
Glm52MTPForCausalLM.apply_quant_config
Glm52MTPHead.__init__
Glm52MTPHead.forward
Glm52MTPHead.get_last_token_states
build_mtp_draft
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_mtp_config.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_copy_model_config_with_moe_backend
_enable_trtllm_moe_preload_for_draft
_get_mtp_moe_quant_config
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_mtp_worker.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
RubinMTPEagleWorker._adp_plan
RubinMTPEagleWorker._forward_linear_draft_loop
RubinMTPEagleWorker._run_draft_forward
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_speculative.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
Glm52SpecDecOneEngineForCausalLM.__init__
Glm52SpecDecOneEngineForCausalLM.cache_derived_state
Glm52SpecDecOneEngineForCausalLM.forward
Glm52SpecDecOneEngineForCausalLM.load_draft_weights
Glm52SpecDecOneEngineForCausalLM.mtp_head_module_names
Glm52SpecDecOneEngineForCausalLM.set_guided_decoder
Glm52SpecDecOneEngineForCausalLM.supports_spec_decode_fast_paths
_slice_spec_position_ids
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_weight_loader.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
Glm52WeightLoader.__init__
Glm52WeightLoader.load_weights
Glm52WeightLoader.load_weights.<locals>.detect_shared_mtp_weights
Glm52WeightLoader.load_weights.<locals>.load_kv_b_proj_and_k_b_proj_trans
Glm52WeightLoader.load_weights.<locals>.load_kv_b_proj_and_k_b_proj_trans_dequant
Glm52WeightLoader.load_weights.<locals>.rename_moe_weight
Glm52WeightLoader.load_weights.<locals>.requantize_weight_with_new_scale
Glm52WeightLoader.load_weights.<locals>.split
Glm52WeightLoader.load_weights.<locals>.split_kv_b_proj
Glm52WeightLoader.load_weights.<locals>.split_matrix_tp
weight_dequant
weight_dequant.<locals>.grid
weight_dequant_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/attention/fp8_paged_mqa_logits.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
FP8MQALogitsKernel.__call__
FP8MQALogitsKernel.__init__
FP8MQALogitsKernel._copy_q
FP8MQALogitsKernel._q_b_stage
FP8MQALogitsKernel._row_sum
FP8MQALogitsKernel._setup_mma
FP8MQALogitsKernel.kernel
add_f16x2
fma_f16x2
max_f16x2
pack_f16x2
unpack_f16x2
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/attention/gvr_balanced.py`

库存函数 59；已覆盖 0；未覆盖 59。

```text
GvrMainKernel.__call__
GvrMainKernel.__init__
GvrMainKernel._balanced_map
GvrMainKernel._ballot_pair_emit
GvrMainKernel._emitc
GvrMainKernel._rung_collect
GvrMainKernel._rung_put
GvrMainKernel._sample_rebin
GvrMainKernel.kern
_fmaf
_ld_g_nc_v4_f32
_ldcg_v2_i32
_ldg_f32_rs
_lds_v2_u64
_pin_i32
_pin_i64
_prefetch_l2
_red_shared_add1
_shfl_up_add
_smem_addr_reg
_smem_v4_tensor
_st_g_u32
_st_g_u64
_st_s_v2_u32
_udiv
atomic_add_cta
atomic_add_u64_gpu
atomic_max_cta
atomic_min_cta
ballot
clz_i32
f2s_rz
f2u_rz
f32_of_i32
f32_of_u32
ffs_m1
fkey
fkey_bits
fmax_f32
fmin_f32
g2r_atom_f32
gather_hint
get_compiled
invkey
invkey_bits
ld_g_f32x4
ld_g_i32
ldg_f32
lds128_i32
popc
scan_cross0
smem_atom_i32_128
sts128_i32
threadfence_gpu
u32_of_f32
warp_add_i32
warp_incl_scan_add
warp_max_u32
warp_min_u32
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/gemm/rmsnorm_early_trigger.py`

库存函数 20；已覆盖 0；未覆盖 20。

```text
RMSNormEarlyTriggerKernel.__call__
RMSNormEarlyTriggerKernel.__init__
RMSNormEarlyTriggerKernel._compute_cluster_n
RMSNormEarlyTriggerKernel._compute_num_threads
RMSNormEarlyTriggerKernel._compute_threads_per_row
RMSNormEarlyTriggerKernel._estimate_smem_bytes
RMSNormEarlyTriggerKernel._make_tv_layout
RMSNormEarlyTriggerKernel._smem_size_in_bytes
RMSNormEarlyTriggerKernel.kernel
_get_compiled_rmsnorm_early_trigger_kernel
block_reduce_multirow
cluster_reduce_multirow
elem_pointer
get_sm_version
predicate_k
rmsnorm_early_trigger
row_reduce_sum_multirow
set_block_rank
store_shared_remote
warp_reduce
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/gemm/splitk.py`

库存函数 62；已覆盖 0；未覆盖 62。

```text
SplitKDenseGemmKernel.__call__
SplitKDenseGemmKernel.__init__
SplitKDenseGemmKernel._drain_producer
SplitKDenseGemmKernel._publish_tile
SplitKDenseGemmKernel._row_rms_b_tiles
SplitKDenseGemmKernel.dma_warp
SplitKDenseGemmKernel.epilog_warp
SplitKDenseGemmKernel.hnorm_b_warp
SplitKDenseGemmKernel.kernel
SplitKDenseGemmKernel.mma_warp
WideLaunch.run_kwargs
_add_bf16x2_squares
_add_rn
_align_up
_arrive_shared_remote
_bmm_bias
_bmm_no_bias
_bmm_publish
_bmm_row_rms
_cvt_bf16x2_lamport
_detect_leading_dim
_f32_hex
_fence_proxy_async_shared
_fma_squares_bf16x2
_from_dlpack_dynamic
_get_compiled_splitk_kernel
_lamport_current_buffer
_ld_global_cg_v4
_ld_shared_v4_b32
_ld_shared_v4_f32
_make_compile_repr_tensors
_make_layout_tensor
_map_shared_rank
_max_ab_stages_for
_mul_bf16x2
_publish_smem_bytes
_rmsnorm_bf16x2
_rmsnorm_rstd
_smem_bytes
_st_shared_v4
_store_global_v4
_store_shared_remote_v4
_to_cute_swap
_validate_runtime_tensors
_wait_parity_cluster
_wide_candidate
default_tactic
hnorm_b_supported
mailbox_fits_stages
narrow_tactic
one_wave_tactic
one_wave_tactic.<locals>.cta_count
row_rms_b_supported
row_rms_launch
run_splitk_dense
run_splitk_row_rms
splitk_default_tactic
tactic_launch_smem_bytes
validate_tactic
wide_mailbox_in_stages
wide_tactic
wide_tuned_launch
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/moe/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py`

库存函数 19；已覆盖 0；未覆盖 19。

```text
BlockScaledSwapAbGenphaseMoeKernel.__call__
BlockScaledSwapAbGenphaseMoeKernel.__call__.<locals>.rewrite_tensor_shape
BlockScaledSwapAbGenphaseMoeKernel.__init__
BlockScaledSwapAbGenphaseMoeKernel._build_device_workspace
BlockScaledSwapAbGenphaseMoeKernel._build_mainloop_and_smem
BlockScaledSwapAbGenphaseMoeKernel._kernel
BlockScaledSwapAbGenphaseMoeKernel._resolve_schedule_hint
BlockScaledSwapAbGenphaseMoeKernel._validate_communication_geometry
BlockScaledSwapAbGenphaseMoeKernel._validate_geometry
BlockScaledSwapAbGenphaseMoeKernel.aot_compile
BlockScaledSwapAbGenphaseMoeKernel.aot_compile.<locals>.fake_tensor
BlockScaledSwapAbGenphaseMoeKernel.get_workspace_sizes
BlockScaledSwapAbGenphaseMoeKernel.impl_desc_require
BlockScaledSwapAbGenphaseMoeKernel.load_compiled
BlockScaledSwapAbGenphaseMoeKernel.name
BlockScaledSwapAbGenphaseMoeKernel.name.<locals>.dtype_name
BlockScaledSwapAbGenphaseMoeKernel.pre_reduced_activation_plane
BlockScaledSwapAbGenphaseMoeKernel.problem_desc_require
BlockScaledSwapAbGenphaseMoeKernel.require_zero_workspace_leading_bytes
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/moe/ptx.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
ld_relaxed_sys_b64
st_relaxed_sys_b64
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/moe/token_comm_gen_specialized.py`

库存函数 116；已覆盖 0；未覆盖 116。

```text
GenphaseTokenComm.__extract_mlir_values__
GenphaseTokenComm.__init__
GenphaseTokenComm.__new_from_mlir_values__
GenphaseTokenComm._advance_exchange_epoch
GenphaseTokenComm._bin_of_position
GenphaseTokenComm._build_smem_workspace
GenphaseTokenComm._canonicalize_wire_indices
GenphaseTokenComm._claim_next_block
GenphaseTokenComm._classify_tokens
GenphaseTokenComm._count_exchange_tag_reader
GenphaseTokenComm._exchange_tag
GenphaseTokenComm._fold_count_matrix
GenphaseTokenComm._is_unowned_expert
GenphaseTokenComm._kernel
GenphaseTokenComm._load_count_matrix
GenphaseTokenComm._pack_word
GenphaseTokenComm._padded_row_elements
GenphaseTokenComm._payload_plane_by_dense_row
GenphaseTokenComm._place_expert_at_slot
GenphaseTokenComm._poll_source_words
GenphaseTokenComm._poll_word
GenphaseTokenComm._publish_padded_prefix
GenphaseTokenComm._publish_payload_flags
GenphaseTokenComm._publish_refine_block_prefix
GenphaseTokenComm._publish_source_segment_bases
GenphaseTokenComm._push_one_plane
GenphaseTokenComm._rank_by_counting_in_warp
GenphaseTokenComm._rank_by_load_histogram
GenphaseTokenComm._rank_experts_by_load
GenphaseTokenComm._refine_one_block
GenphaseTokenComm._refine_row_source
GenphaseTokenComm._register_local_regions
GenphaseTokenComm._register_symmetric_regions
GenphaseTokenComm._resolve_block_expert
GenphaseTokenComm._run_exchange
GenphaseTokenComm._run_exchange_to_all_peers
GenphaseTokenComm._run_fenceless_exchange
GenphaseTokenComm._run_helper
GenphaseTokenComm._run_pusher
GenphaseTokenComm._run_refine
GenphaseTokenComm._run_sort
GenphaseTokenComm._run_sorted_exchange
GenphaseTokenComm._run_sorted_sort
GenphaseTokenComm._run_tagged_exchange_and_sort
GenphaseTokenComm._scatter_block_atoms
GenphaseTokenComm._score_bits_view
GenphaseTokenComm._stage_block_rows
GenphaseTokenComm._stage_refine_expert_table
GenphaseTokenComm._store_block_direct
GenphaseTokenComm._validate_configuration
GenphaseTokenComm._wire_index
GenphaseTokenComm._wire_index_view
GenphaseTokenComm._word_high
GenphaseTokenComm._word_payload
GenphaseTokenComm.activation_dtype
GenphaseTokenComm.activation_sf_dtype
GenphaseTokenComm.bin_search_steps
GenphaseTokenComm.canonical_activation_sf_tensor
GenphaseTokenComm.comm_done_target
GenphaseTokenComm.dense_activation_sf_tensor
GenphaseTokenComm.dense_activation_tensor
GenphaseTokenComm.destination_mask_words
GenphaseTokenComm.expert_sizes_tensor
GenphaseTokenComm.experts_per_rank
GenphaseTokenComm.fc1_topk_scores_tensor
GenphaseTokenComm.gather_index_tensor
GenphaseTokenComm.grid_cta_count
GenphaseTokenComm.helper_cta_index
GenphaseTokenComm.hidden_sf
GenphaseTokenComm.histogram_bucket_count
GenphaseTokenComm.impl_desc_require
GenphaseTokenComm.invalid_expert_marker
GenphaseTokenComm.launch
GenphaseTokenComm.launch_cluster_count
GenphaseTokenComm.local_expert_bucket_count
GenphaseTokenComm.metadata_ready_target
GenphaseTokenComm.payload_ready_target
GenphaseTokenComm.plain_activation_row_elements
GenphaseTokenComm.plain_activation_sf_row_elements
GenphaseTokenComm.plain_activation_sf_tensor
GenphaseTokenComm.plain_activation_tensor
GenphaseTokenComm.problem_desc_require
GenphaseTokenComm.pusher_cta_begin
GenphaseTokenComm.pusher_global_warp_count
GenphaseTokenComm.pusher_tokens_per_warp
GenphaseTokenComm.pusher_warp_count
GenphaseTokenComm.rank_count_ready_target
GenphaseTokenComm.ranks_experts_in_warp
GenphaseTokenComm.refine_atoms_per_stage
GenphaseTokenComm.refine_output_stage_elements
GenphaseTokenComm.refine_row_columns
GenphaseTokenComm.refine_run_columns
GenphaseTokenComm.refine_run_count
GenphaseTokenComm.refine_staging_group_stride
GenphaseTokenComm.refine_staging_row_stride
GenphaseTokenComm.refine_total_atoms
GenphaseTokenComm.register_device_workspace
GenphaseTokenComm.route_rounds
GenphaseTokenComm.routes_per_rank
GenphaseTokenComm.scheduler_expert_sizes_tensor
GenphaseTokenComm.slot_to_expert_tensor
GenphaseTokenComm.smem_bytes
GenphaseTokenComm.smem_workspace
GenphaseTokenComm.token_src_metadata_tensor
GenphaseTokenComm.topk_index_inbox_tensor
GenphaseTokenComm.topk_lane_rounds
GenphaseTokenComm.topk_ready_target
GenphaseTokenComm.topk_score_inbox_tensor
GenphaseTokenComm.topk_slab_elements
GenphaseTokenComm.trash_bucket_index
GenphaseTokenComm.worst_case_data_rows
GenphaseTokenComm.worst_case_padded_rows
GenphaseTokenComm.worst_case_refine_blocks
GenphaseTokenComm.worst_case_sf_rows
_first_matching_lane
_warp_inclusive_sum
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/moe/topk_reduce_add_rmsnorm.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
TopkReduceAddRMSNormKernel.__call__
TopkReduceAddRMSNormKernel.__init__
TopkReduceAddRMSNormKernel._smem_size_in_bytes
TopkReduceAddRMSNormKernel.kernel
_aligned
_compile_topk_reduce_add_rmsnorm
_num_threads
topk_reduce_add_rmsnorm
topk_reduce_add_rmsnorm_matches_flashinfer
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/spec/draft_glue_rmsnorm.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
DraftGlueRMSNormKernel.__call__
DraftGlueRMSNormKernel.__init__
DraftGlueRMSNormKernel._smem_size_in_bytes
DraftGlueRMSNormKernel.kernel
_compile_draft_glue_kernel
_predicate_cols
draft_glue_matches_flashinfer
draft_glue_rmsnorm_cat
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/modeling.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
DecodeTarget.__init__
DecodeTarget.forward
Glm52Nvfp4Sm107HighThroughputCore.__init__
Glm52Nvfp4Sm107HighThroughputCore._select_target
Glm52Nvfp4Sm107HighThroughputCore.forward
ModelingV2Glm52Nvfp4Sm107HighThroughput.__init__
ModelingV2Glm52Nvfp4Sm107HighThroughput.cache_derived_state
ModelingV2Glm52Nvfp4Sm107HighThroughput.get_preferred_kv_cache_manager_version
ModelingV2Glm52Nvfp4Sm107HighThroughput.get_preferred_transceiver_runtime
ModelingV2Glm52Nvfp4Sm107HighThroughput.load_weights
ModelingV2Glm52Nvfp4Sm107HighThroughput.setup_aliases
PrefillTarget.__init__
PrefillTarget.forward
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/full_query_logits.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
CuteDSLPagedMQALogitsRunner._compile
CuteDSLPagedMQALogitsRunner.forward
_fake_logits
_get_num_sms
cute_dsl_fp8_paged_mqa_logits
fp8_logits_early_kv_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/kv_norm_pack.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_kv_norm_pack_fake
_kv_norm_pack_kernel
is_kv_norm_pack_supported
kv_norm_pack
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/projection.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
RubinDSAHooks.__init__
RubinDSAHooks.forward
RubinDSAHooks.forward_custom_op
RubinDSAHooks.initialize
_fold_q_b
_forward_dsa_proj_fallback
_fused_a_supported
_q_b_norm_fold_ready
_q_b_with_norm
build_projection_weights
forward_dsa_proj
forward_dsa_proj.<locals>.kv_path
forward_dsa_proj.<locals>.q_path
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/projection_op.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_extract_extra_attrs
_mla_dsa_proj_fake
_same_cache_view
indexer_cache_for_projection
mla_dsa_proj
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/q_a_norm.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_fake_q_a_norm_early
_uses_cute_rmsnorm
is_q_a_norm_early_supported
q_a_norm_early
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/gemm/low_m.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
RubinUnquantizedLinearMethod.__init__
RubinUnquantizedLinearMethod.apply
_device_limits
_fake_low_m_gemm
install_low_m_methods
low_m_gemm
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/moe/megamoe_runner.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
RubinMegaMoERunner._build_kernel_once
RubinMegaMoERunner._build_unique_id
_construct_rubin_kernel
_construct_rubin_kernel.<locals>.construct
_fake_run_rubin_megamoe
get_rubin_runner
query_rubin_kernel_properties
run_rubin_megamoe
runner_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/spec/_draft_exchange.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
DraftExchangeWorkspace.__init__
DraftExchangeWorkspace.for_mapping
_all_succeeded
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/spec/draft_sample_glue.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
DraftSampleGlueAdp.__init__
DraftSampleGlueAdp.gather
DraftSampleGlueAdp.get
DraftSampleGlueAdp.run
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/spec/mtp_tail.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_fake_scatter_mtp_tail
_glm_rubin_mtp_tail_scatter_kernel
scatter_mtp_tail
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/weights.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
load_weights
normalize_model_config
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_gemm/low_m_bf16_splitk.py`

库存函数 25；已覆盖 0；未覆盖 25。

```text
SplitKDenseGemmKernel.__call__
SplitKDenseGemmKernel.__init__
SplitKDenseGemmKernel._drain_producer
SplitKDenseGemmKernel.dma_warp
SplitKDenseGemmKernel.epilog_warp
SplitKDenseGemmKernel.kernel
SplitKDenseGemmKernel.mma_warp
_align_up
_bmm_bias
_bmm_no_bias
_detect_leading_dim
_from_dlpack_dynamic
_get_compiled_splitk_kernel
_make_compile_repr_tensors
_make_layout_tensor
_map_shared_rank
_max_ab_stages_for
_smem_bytes
_store_shared_remote_v4
_to_cute_swap
_validate_runtime_tensors
autotune_tactics
default_tactic
run_splitk_dense
validate_tactic
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_spec/dspark_draft_ar.py`

库存函数 18；已覆盖 0；未覆盖 18。

```text
DraftArNormKernel.__call__
DraftArNormKernel.__init__
DraftArNormKernel.kernel
DraftEmbedKernel.__call__
DraftEmbedKernel.__init__
DraftEmbedKernel.kernel
_add_bf16x2
_compiled_ar_norm
_compiled_embed
_ld_id
_rms_scale
_shfl_add
_sm_count
_stream
ar_norm_config
grid_config
run_draft_ar_norm
run_draft_embed
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_spec/dspark_draft_gemm_push.py`

库存函数 19；已覆盖 0；未覆盖 19。

```text
SplitKArPushGemmKernel.__call__
SplitKArPushGemmKernel.__init__
SplitKArPushGemmKernel.dma_warp
SplitKArPushGemmKernel.kernel
SplitKArPushGemmKernel.push_epilog_warp
_NoopKernel.__call__
_NoopKernel.kernel
_bmm_push
_compiled
_lamport_cur
_lds_v4
_make_layout_tensor
_push_vector
_swapped
ensure_cute_runtime
run_gemm
run_gemm_push
smem_bytes
validate_push_tactic
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/_k3_fp4_mla_cute_dsl_utils.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_PyIRStdoutFilter.__getattr__
_PyIRStdoutFilter.__init__
_PyIRStdoutFilter.finish
_PyIRStdoutFilter.flush
_PyIRStdoutFilter.write
_compile_cutedsl
_current_cu_stream
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/_k3_fp4_mla_v_repack.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
_compile_fp4_mla_v_repack
_kv_cache_3d_layout
_kv_cache_3d_view
_make_v_repack_ptrs
_validate_tensor_pointer_alignment
_validate_v_repack_args
fp4_mla_repack_v_cache
fp4_mla_repack_v_cache_reference
fp4_mla_v_repack_host
fp4_mla_v_repack_kernel
fp4_mla_v_repack_tma_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/fp4_mla_kv_append.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
Fp4MlaKvAppend.__call__
Fp4MlaKvAppend.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/spec/k3_draft_ar_norm.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
K3DraftArNorm.__call__
K3DraftArNorm.is_valid
K3DraftArNorm.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/spec/k3_draft_embed.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
K3DraftEmbed.__call__
K3DraftEmbed.is_valid
K3DraftEmbed.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/spec/k3_draft_gemm_push.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
K3DraftGemmPush.__call__
K3DraftGemmPush.is_valid
K3DraftGemmPush.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/csrc/msa_aot/build_bf16x_extension.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
flags
main
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/csrc/msa_aot/build_k2q_extension.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
main
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/csrc/msa_aot/generate_fmha_variants.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_load_jit
_params
main
render
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/csrc/msa_aot/pack_cute_aot.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
main
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/csrc/msa_aot/torch_extension_aot.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
build_extension
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/draft_probs_one_request_kernel.py`

库存函数 23；已覆盖 0；未覆盖 23。

```text
_add_rn
_atomic_add_acq_rel
_bar_arrive
_bar_sync
_better
_block_dot
_fetch_w1_row
_fma_rn
_hi_f32
_keep_better
_lo_f32
_pack_bf16x2
_post_ll
_reduce_slots
_warp_best
buffer_words
k3_markov
k3_markov_kernel
ll_region_words
ll_workspace_words
rows_per_cta
smem_bytes
supports
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/flagfree_gather_kernel.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
_atomic_add_acq_rel
_canon
_globaltimer
_has_empty
_post
_store_fp32x8
_store_out
_trap
flagfree_gather
flagfree_gather_kernel
row_words
slot_words
vectors
workspace_words
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3_draft_head_gemm.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
StridedMxfp8Gemm.__call__
StridedMxfp8Gemm.__init__
StridedMxfp8Gemm._alpha_for
StridedMxfp8Gemm.compile
_aot_sources
_compiled
_kernel_base
_kernel_class
_kernel_class.<locals>.StridedCKernel.wrapper_ldc
_max_active_clusters
flashinfer_kernel_digest
tactics
why_invalid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3_lowm_splitk.py`

库存函数 35；已覆盖 0；未覆盖 35。

```text
SplitKDenseGemmKernel.__call__
SplitKDenseGemmKernel.__init__
SplitKDenseGemmKernel._drain_producer
SplitKDenseGemmKernel._push_tile
SplitKDenseGemmKernel._store_tile
SplitKDenseGemmKernel.dma_warp
SplitKDenseGemmKernel.epilog_warp
SplitKDenseGemmKernel.kernel
SplitKDenseGemmKernel.mma_warp
_align_up
_aot_sources
_bf16x2_word
_bmm_bias
_bmm_no_bias
_bmm_push
_detect_leading_dim
_from_dlpack_dynamic
_get_compiled_push
_get_compiled_splitk_kernel
_ld_volatile_s32
_make_compile_repr_tensors
_make_layout_tensor
_map_shared_rank
_max_ab_stages_for
_smem_bytes
_st_global_b32
_store_shared_remote_v4
_to_cute_swap
_validate_runtime_tensors
autotune_tactics
compile_push
default_tactic
run_splitk_dense
run_splitk_dense_push
validate_tactic
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3ll/locality.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
LocalityMap.get
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/qkv_fusion/clc_tile_scheduler.py`

库存函数 15；已覆盖 0；未覆盖 15。

```text
ClcDynamicPersistentTileScheduler.__extract_mlir_values__
ClcDynamicPersistentTileScheduler.__init__
ClcDynamicPersistentTileScheduler.__new_from_mlir_values__
ClcDynamicPersistentTileScheduler._swizzle_and_rasterize
ClcDynamicPersistentTileScheduler.advance_to_next_work
ClcDynamicPersistentTileScheduler.create
ClcDynamicPersistentTileScheduler.get_current_work
ClcDynamicPersistentTileScheduler.get_grid_shape
ClcDynamicPersistentTileScheduler.initial_work_tile_info
ClcDynamicPersistentTileScheduler.num_tiles_executed
ClcDynamicPersistentTileScheduler.work_tile_info_from_clc_response
ClcDynamicPersistentTileSchedulerParams.__extract_mlir_values__
ClcDynamicPersistentTileSchedulerParams.__init__
ClcDynamicPersistentTileSchedulerParams.__new_from_mlir_values__
ClcDynamicPersistentTileSchedulerParams.get_grid_shape
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/qkv_fusion/kernel.py`

库存函数 51；已覆盖 0；未覆盖 51。

```text
M3QkvFusedRunner.__call__
M3QkvFusedRunner.__init__
M3QkvFusedRunner.make_args
M3QkvFusedRunner.make_args.<locals>.dl
M3QkvFusedRunner.make_args.<locals>.flat_u8
M3QkvFusedRunner.make_args.<locals>.span_u8
M3QkvFusedRunner.run_args
M3QkvGemmNormRopeKvInsert.__call__
M3QkvGemmNormRopeKvInsert.__init__
M3QkvGemmNormRopeKvInsert._cluster_layouts
M3QkvGemmNormRopeKvInsert._mainloop_s2t_copies
M3QkvGemmNormRopeKvInsert._mainloop_s2t_copy_and_partition
M3QkvGemmNormRopeKvInsert._mainloop_s2t_copy_and_partition.<locals>.append_mn_broadcast_mode
M3QkvGemmNormRopeKvInsert._make_tiled_mmas
M3QkvGemmNormRopeKvInsert._make_tiled_mmas.<locals>.make
M3QkvGemmNormRopeKvInsert._make_tma_atoms
M3QkvGemmNormRopeKvInsert._norm_rope
M3QkvGemmNormRopeKvInsert._nvfp4_pack
M3QkvGemmNormRopeKvInsert._setup_attributes
M3QkvGemmNormRopeKvInsert._store_kv
M3QkvGemmNormRopeKvInsert._unpack
M3QkvGemmNormRopeKvInsert.kernel
M3QkvGemmNormRopeKvInsert.kernel.<locals>.load_head
M3QkvGemmNormRopeKvInsert.kernel.<locals>.release_acc
_absmax
_add_rn
_aot_sources
_asm
_bf16x2_rn
_cache_strides
_e2m1x8
_e4m3_byte
_e4m3_byte_to_f32
_e4m3x4
_fma_rn
_hi_f32
_ldg_s32
_ldg_u16
_ldg_v4_f32
_lds_v4_f32
_lo_f32
_mul_rn
_rcp_approx_ftz
_rsqrt_approx
_stg_u8
_stg_v2
_stg_v4
_sts_f32
compile
compile.<locals>.compile_fn
compile.<locals>.compile_fn.<locals>.sym
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/qkv_fusion/plain_gemm.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
M3ClcMxfp8Gemm.__call__
M3ClcMxfp8Gemm.__init__
M3ClcMxfp8Gemm._weight_dl
M3ClcMxfp8Gemm.check
M3ClcMxfp8Gemm.make_args
M3ClcMxfp8Gemm.run_args
_dl
_dummy_producer_args
_pad_up
get_gemm
m3_clc_mxfp8_gemm
num_fixed_slots
num_heads_q_for
swizzled_sf_numel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/corun_band.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
install
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/engagement.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_activation
arm_identity
deployment_mismatch
identity_lines
is_dep
megamoe_local_requested
moe_engagement
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/idx_pdl.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
install
status
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/idxk_producer.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
install
status
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/modeling.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
ModelingV2MiniMaxM31FlashNvfp4Sm107Dep.__init__
ModelingV2MiniMaxM31FlashNvfp4Sm107Dep.cache_derived_state
ModelingV2MiniMaxM31FlashNvfp4Sm107Dep.forward
_install_sampler_overrides
_setters
armed_hooks
dep_runtime_hooks
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/proj_pin.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_installed
_log_once
_pinned_forward
install
proj_decline_reason
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/softmax_mc.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
_temperature_args
engagement
log_engagement
mc_op
partials_numel
pathb_op
prebuild
route
softmax_mc
write_ctas
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark_draft_fmha.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
SplitDecode.__call__
SplitDecode.__call__.<locals>.one
SplitDecode.__init__
SplitDecode._allocate
SplitDecode._log
SplitDecode._ready
SplitDecode.classify
SplitDecode.why_not
_nbytes
_page_size
install
installed
window_cap
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark_draft_head_fp4.py`

库存函数 23；已覆盖 0；未覆盖 23。

```text
DraftHeadFp4.__call__
DraftHeadFp4.__init__
DraftHeadFp4._gemm
DraftHeadFp4._run
DraftHeadFp4._self_check
DraftHeadFp4.backend
DraftHeadFp4.serves_head
DraftHeadFp4.why_not
_act_quantizer
_default_flashinfer_tactic
_flashinfer_unfit
_raise_if_cuda_fault
act_block_scales
act_global_scale
format_tactic
head_input_rows
norm_gains
padded_n
plan_tactics
prebuild
quantize_weight
subnormal_or_zero
unfit
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark_draft_mlp.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
_Route.__init__
_Route.bytes
_layers
_raise_if_cuda_fault
install
install.<locals>._apply_activation
install.<locals>.apply_linear
prebuild
prepare
run
unfit
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m31_markov16.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_dots_of_words
_row_words
m31_markov16
m31_markov16_kernel
reg_blocks
smem_bytes
streamed_blocks
supports
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/spec/draft_bf16.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
_
_kernel
_load
arm_draft_bf16
arm_draft_bf16.<locals>.precompile
arm_draft_bf16.<locals>.precompile.<locals>.agree_with_b1
gather
prepare
try_head_and_refine
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/dspark_draft_head.py`

库存函数 21；已覆盖 0；未覆盖 21。

```text
DraftHeadMx4.__call__
DraftHeadMx4.__init__
DraftHeadMx4._real_gemm
DraftHeadMx4._scratch_stride
DraftHeadMx4._warm
DraftHeadMx4.into
DraftHeadMx4.run_into
DraftHeadMx4.self_check
DraftHeadMx4.serves_head
DraftHeadMx4.why_not
DraftHeadMx4.with_tactic
_cute_dsl_quantizer
dequantize
padded_n
parse_tactic
pinned_tactic
quantize_weight
raise_if_cuda_fault
tactic_invalid
tactic_name
unfit
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/seams.py`

库存函数 25；已覆盖 0；未覆盖 25。

```text
DistFacade.__init__
DistFacade.all_gather_object
DistFacade.barrier
DistFacade.get_rank
DistFacade.get_world_size
DistFacade.is_initialized
DistFacade.new_group
LamportExt.__init__
LamportExt.push_params_bytes
LamportExt.ws_create
LamportExt.ws_destroy
LamportExt.ws_open
Mnnvl.__init__
Mnnvl.alloc
PostExt.__init__
PostExt._rh_find
PostExt._router_exchange_reset
PostExt.allreduce_norm_quant_route
PostExt.router_exchange_create
PostExt.router_exchange_open
_dist_modules
install
load
no_jit
no_jit.<locals>.load
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4_hoist_block.py`

库存函数 18；已覆盖 0；未覆盖 18。

```text
Geo.__init__
HoistBlock.__call__
HoistBlock.arm
HoistBlock.ffn_out
HoistBlock.residual_mid
HoistBlock.residual_out
_build_prepared
_prepare
_prepare.<locals>.e
build
die_shards
domain_tensor
engine_matches
engine_module_config
layer_views
layer_views.<locals>.nk
log_engaged
unfold_gemma
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/api/contracts.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_c
allowed
violations
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/api/ctx.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
BuildCtx.comm
BuildCtx.counters
BuildCtx.csrc_dir
BuildCtx.data_path
BuildCtx.debug
BuildCtx.load
BuildCtx.local
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/api/params.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
Params.fields
Params.from_dict
Params.to_dict
_coerce
_plain
params
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/api/pdl.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
Footprint.__add__
Footprint.covers
PdlSpec.__post_init__
Spin.__post_init__
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/api/stage.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
Stage.__call__
StageFn.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/api/variant.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
Variant.__post_init__
Variant.footprint_for
Variant.id
Variant.kernels_for
Variant.pdl_for
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/front/_bind.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
bind
build_root
g3_knobs
knobs_table
stage
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/moe/_common.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
base_knobs
bind
chain
chain.<locals>.fn
engine_dir
g3_knobs
jdump
knobs
stage
waiver
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/moe/sht4_n48.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
make
runtime_knobs
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/moe/sht4_n48_q.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
make
runtime_knobs
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/moe/sht_x4.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
engine_pdl
make
runtime_knobs
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/router/lr4_prr4.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
make
runtime_knobs
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/families/router/lr4_prr4_q.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
make
runtime_knobs
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/comm/allreduce_norm_quant/allreduce_norm_quant_build.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
BoundaryWorkspace.__enter__
BoundaryWorkspace.__exit__
BoundaryWorkspace.__init__
BoundaryWorkspace.close
BoundaryWorkspace.new_site
ext
ext.<locals>._load
new_site
workspace
ws
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/comm/allreduce_norm_quant/allreduce_norm_quant_host.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
_call
_call.<locals>.fn
_post_prepushed_mx
_post_prepushed_mx.<locals>.fn
_prx_ws
make_post_prepushed_mx
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/comm/allreduce_norm_quant/lamport_allreduce_host.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
LamportWorkspaces.__enter__
LamportWorkspaces.__exit__
LamportWorkspaces.__init__
LamportWorkspaces.bucket
LamportWorkspaces.close
LamportWorkspaces.handle
LamportWorkspaces.push_params
_cpu
_ext
_gather_open
_pr
build_pr
workspaces
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/comm/allreduce_norm_quant/route_warp_tp4_fc1_per_token_host.py`

库存函数 19；已覆盖 0；未覆盖 19。

```text
_consts
_moe_pt
_moe_pt.<locals>.build_x4
_moe_pt.<locals>.build_x4.<locals>.observe
_moe_pt.<locals>.build_x4.<locals>.observe.<locals>.run_obs
_pt_bufs
_sht_pt_build
_sht_pt_build.<locals>.b
install
install.<locals>._kw
install.<locals>._load
make_moe_pt
make_oproj
make_post_pt
make_post_pt.<locals>.fn_chk
postchk
prep_mw
qat_row_quant
src_hash
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/comm/allreduce_norm_quant/route_warp_tp4_host.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
install
install.<locals>._kw
install.<locals>._load
make_oproj_ra4_p
make_post_rh_x4_p
ptrig_on
src_hash
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/comm/allreduce_norm_quant/router_exchange.py`

库存函数 37；已覆盖 0；未覆盖 37。

```text
RxSlot.__init__
RxSlot.index
RxWorkspace.__init__
RxWorkspace.reset
X4Bufs.__init__
_arm
_check_rb
_chrono
_cl
_dbg
_deadline
_dump
_gloo
_kw
_loud
_roll
_rows
_warm
_x4_on
check_x4_pair
collect
collect.<locals>.arm_of
ext
ext.<locals>._load
post_rh
post_rh.<locals>.call
post_rh.<locals>.fn
post_tab
post_tab.<locals>.call
post_tab.<locals>.fn
prx_spec
register_counter
rx_reset_all
rx_slot
src_hash
write
x4_bufs
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/common/allreduce_push_workspace.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_bucket
_g3
push_spec
push_workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/common/host_hooks.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
get
provide
provided
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/common/model_geometry.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
Geo.hq
Geo.inter
Geo.nh
Geo.o_k
Geo.qkv_n
L2Flush.__call__
L2Flush.__init__
capture
log0
max_over_ranks
rank
rel_l2
time_graph
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/common/post_mxfp8_buffers.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
_mx_bufs
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/common/stock_stages.py`

库存函数 32；已覆盖 0；未覆盖 32。

```text
Stock_AR.__init__
Stock_AR.ar_norm
join_aux
make_boundary_in
make_boundary_in.<locals>.fn
make_boundary_post
make_boundary_post.<locals>.fn
make_combine
make_combine.<locals>.fn
make_decode
make_decode.<locals>.fn
make_moe_experts
make_moe_experts.<locals>.fn
make_moe_quant
make_moe_quant.<locals>.fn
make_oproj
make_oproj.<locals>.fn
make_producer
make_producer.<locals>.fn
make_qkv
make_qkv.<locals>.fn
make_router
make_router.<locals>.fn
make_scorer
make_scorer.<locals>.fn
make_shared
make_shared.<locals>.fn
make_topk
make_topk.<locals>.fn
mx_mm
mx_quant
resolve
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/expert_shards.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
_shards
_shards.<locals>.cut
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/fused/host.py`

库存函数 24；已覆盖 0；未覆盖 24。

```text
M3MoeLayer.__call__
M3MoeLayer.__init__
M3MoeState.__init__
M3MoeState._ext_args
M3MoeState._x4_args
M3MoeState.layer
M3MoeState.run
_check_dl_roles
_fine_ranges
_kernel_module
_maxnreg
_maxnreg.<locals>.gen
_pretile_shared
_pretile_shared.<locals>.sf_bytes
_pretiled
_shared_views
_view
ar_push_desc
ar_push_desc_g3
is_supported
make_phase_plan
make_phase_plan.<locals>.counts_for
make_phase_plan_dl
push_params_v3
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/fused_die_local_host.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_build
_c2
_dbg_topk
_dbg_topk.<locals>.fn_dbg
_dev_b2lag
_op_dl
build_shipped
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/fused_host.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_bucket
_dump_tdbg
_dump_tdbg.<locals>.dump
_g3
_push_reg
make_k3
make_k3.<locals>.fn
make_k3.<locals>.pf_init
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/fused_shared_trigger_host.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_cfg
_check
_dump
_q
_sht_build
_sht_build.<locals>.b
make_x4_sht
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/fused_tp4_host.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
_nsh
_over
_patched
_patched.<locals>.b
_sht_n
make_x4_sht4_n48
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/post_nvfp4_handoff.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_env
_ext_x4
_post_x4
_x4_build
_x4_build.<locals>.build_x4
_x4_build.<locals>.build_x4.<locals>.observe
_x4_build.<locals>.build_x4.<locals>.observe.<locals>.run_obs
make_post_rh_x4
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/router_hoist_host.py`

库存函数 53；已覆盖 0；未覆盖 53。

```text
RouteBuf.__init__
RouteBuf.host_publish
RouteBuf.lines
RouteBuf.retract
RouteBuf.tag
_LayerProxy.__call__
_LayerProxy.__getattr__
_LayerProxy.__init__
_align_for_compile
_arm_label
_boundary_claim
_bucket
_c1_build
_c1_port
_c2_build
_c2_classes
_check_dl_grid
_check_pairing
_dump
_ext
_ext.<locals>.fn_dbg
_hold
_itp
_l_cfg
_learner
_learner.<locals>.fn
_no_rh_oproj
_npart
_oproj_rh
_oproj_rh.<locals>.fn
_plain
_post_rows
_rank_world
_rh_cfg
_rh_weights
_route_kwargs
_route_kwargs.<locals>.layer
_route_state
_seed_routes
_seed_table
_state_c1
_state_c2
_untrack_k3
check_call_time
check_route_alignment
check_route_consumed
hang_flags
learn_table
make_oproj_rh
make_oproj_rh_solo
make_post_rh
route_giveup_rows
stamp_rows
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/engine/router_hoist_weights.py`

库存函数 30；已覆盖 0；未覆盖 30。

```text
_rand_case
assign_jobs
bf16_bits
blob_weights
build_from_blk
build_jobs
build_rank
canon_word
check_swizzle_against
e8m0_scale
emulate_partials
f32_sum_rank_order
flip_report
fold_oproj
fold_router
hoisted_logits
job_list
main
mxfp8_dequant
mxfp8_quant_rows
pack_route
r_cols
route_minimax2
selftest
sf128x4_index
stock_boundary
stock_logits
swizzle_128x4
unpack_route
unswizzle_128x4
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/knobs.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
Knob.env
knob
require
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/router/fused_o_proj/host.py`

库存函数 24；已覆盖 0；未覆盖 24。

```text
_arg
_compile
_declines
_dummy_timers
_extra_args
_kernel
_mma_n
_placeholder_scales
_placeholder_words
_resolve
_run
_smem_bytes
_views
ensure_compiled
freeze
frozen
get_config
get_env_enable_pdl
is_compiled
m3_mx_gemv
m3_mx_gemv_bf16
precompile
serves
use_world
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/router/fused_o_proj/kernel_tp4.py`

库存函数 77；已覆盖 0；未覆盖 77。

```text
_absmax
_al
_asm
_atom_map
_bf16_bits
_bf16_round
_bf16x2_word
_bytes_map
_din_load4
_e4m3x4
_f4
_k_tile
_ld_global_s64
_ld_shared_v4_f32
_ld_volatile_s32
_mapa_u32
_p5_atom_add
_p5_e8m0
_p5_f
_p5_fix
_p5_fix64
_p5_ld64
_p5_ld_bf16
_p5_rowsq_pub
_p5_rowsq_spin
_p5_rsqrt
_p5_spin
_p5_st_bf16
_p5_st_s32
_p5_st_sys
_p5_st_u8
_px_bcast0
_px_poll
_ra3_bf4
_ra3_e4m3x4
_ra3_entry
_ra3_ld2_cg
_ra3_try
_ra3_wred
_ra_bar
_ra_bfly
_ra_e4m3x4
_ra_entry
_ra_exit
_ra_ext_bytes
_ra_f
_ra_fma
_ra_launch
_ra_ld2bf_cg
_ra_ld4b_cg
_ra_ld4f_nc
_ra_ld_relaxed
_ra_lds
_ra_lds4
_ra_load_a1
_ra_scale_cg
_ra_st_tag1
_ra_st_tag2
_ra_stamp
_ra_sts
_ra_trap
_rc_a_base
_rc_entry
_rc_load_a
_rcp_approx_ftz
_rope_bits
_st_async_v4
_st_global_b32
_stamp
_test_wait_cluster
a_col0
acc_cols
m3_mx_gemv
m3_mx_gemv_kernel
num_k_tiles
read_globaltimer
tmem_cols
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/kernels/moe/router/fused_o_proj_tp4_host.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
_blob_dev
_ck
_dump
_rank
blobs
emulate
make_oproj_ra4
make_oproj_ra4.<locals>.fn
selftest
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/runtime/arena.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
Arena._view
Arena.allocate
Arena.carve
Arena.offset
Arena.report
Arena.zero
_align
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/runtime/comm.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
CommArena.__post_init__
CommArena.allocate
CommArena.carve
LoopbackComm.__post_init__
LoopbackComm.carve
_wrap_ptr
check
cudart
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/runtime/ctx.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
BuildContext._scope
BuildContext.comm
BuildContext.counters
BuildContext.csrc_dir
BuildContext.data_path
BuildContext.debug
BuildContext.load
BuildContext.local
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/runtime/debug.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
Debug.__init__
Debug.get
Debug.snapshot
_knobs
host_env_set
stray_debug_env
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/runtime/knobs.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
Knobs.__delitem__
Knobs.__getitem__
Knobs.__init__
Knobs.__iter__
Knobs.__len__
Knobs.__repr__
Knobs.__setitem__
Knobs.bind
Knobs.clear_for_tests
Knobs.owners
Knobs.scope
Knobs.snapshot
stage_scoped
stage_scoped.<locals>.call
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/toolchain/child_env.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
_SubprocessProxy.Popen
_SubprocessProxy.__getattr__
_SubprocessProxy.__init__
_SubprocessProxy._wrap
_SubprocessProxy.check_call
_SubprocessProxy.check_output
_SubprocessProxy.run
child_env
merge_env
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/toolchain/paths.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
csrc_abs
csrc_dir
csrc_root
data_path
pkg_abs
pkg_root
repo_root
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/hoist/tp4c1/toolchain/procenv.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
cccl_escape
select_cuda
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/dspark_draft_head.py`

库存函数 21；已覆盖 0；未覆盖 21。

```text
DraftHeadMx4.__call__
DraftHeadMx4.__init__
DraftHeadMx4._real_gemm
DraftHeadMx4._scratch_stride
DraftHeadMx4._warm
DraftHeadMx4.into
DraftHeadMx4.run_into
DraftHeadMx4.self_check
DraftHeadMx4.serves_head
DraftHeadMx4.why_not
DraftHeadMx4.with_tactic
_cute_dsl_quantizer
dequantize
padded_n
parse_tactic
pinned_tactic
quantize_weight
raise_if_cuda_fault
tactic_invalid
tactic_name
unfit
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_target_head.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_num_sms
head_supported
logits
prepare_head
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_tokmap_guard.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_guarded
_indexed
_map_offset
_no_missed
_no_taken
_one_request_map
install
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/local_moe/bootstrap.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
install_tp8
install_tp8.<locals>.make_state
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/local_moe/local_plan.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
_balance
_balance.<locals>.counts_for
_fine_ranges
geometry
logical_map
make_domain_phase_plan
remap_unit
split_structure
stock_phase_plan
validate_plan
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/local_moe/m31_local_moe_kernel.py`

库存函数 71；已覆盖 0；未覆盖 71。

```text
_a1_crd
_a2_crd
_add_rn
_adv
_ar_parity
_ar_push
_ar_push_pre
_asm
_b1_addr
_bar_arrive
_bf16_bits
_bf16_round
_bf16x2_word
_build_b1
_build_b1_fused
_build_b1_part
_bulk_g2s
_cfg
_div_full
_div_rn
_e2m1x2
_e4m3_bits
_e4m3_to_f32
_e4m3x2
_ex2
_ex2_full
_fc1_geo
_fence_proxy_async_global
_fma_rn
_gtime
_l2_prefetch
_ld_acquire_u32
_ld_relaxed_u32
_ld_relaxed_u64
_ld_volatile_s32
_match_any
_min_xorsign_abs
_mul_rn
_mx_osc
_mx_words
_nanosleep
_nkb2
_nsegs
_plan_range
_rcp
_red_release_add
_redux_max_u32
_seg
_seg_local_geometry
_sf1_cp
_sf2_atom
_sf2_cp
_sfa_to_tmem
_sfb1_to_tmem
_sh_issue
_sh_unit_stage
_shared_cta
_smid
_st_global_b32_if
_st_relaxed_u32
_st_relaxed_u64
_stamp
_sub_rn
_swiglu_quant
_tptr
_twiddle
_ue8m0_rp
_wait
_waitd
m3_moe
m3_moe_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/local_moe/m31_local_moe_op.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
LocalMoeState.__init__
LocalMoeState._place
LocalMoeState.layer
LocalMoeState.run
_apply_tunes
_fc1_image_half
_log
_max_smem_per_sm
_place_halves
_report_tunes
aot_sources
local_moe_enabled
make_state
tune_request
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/local_moe/m31_local_moe_ws_receipt.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_dom
enabled
receipt
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/local_moe/serving_hook.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
install
install.<locals>.build_state
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/gemm/m3_mxfp8_pinned_quantize.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3Mxfp8PinnedQuantize.__call__
M3Mxfp8PinnedQuantize.reference
_runner
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_router_gemv.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
m3_router_gemv
m3_router_gemv_fp4
m3_router_tc
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_target_head.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_num_sms
head_supported
logits
prepare_head
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_router_gemv.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
m3_router_gemv
m3_router_gemv_fp4
m3_router_tc
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_target_head.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_num_sms
head_supported
logits
prepare_head
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/cache_manager.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
MiniMaxM31KVCacheManagerV2.__init__
MiniMaxM31KVCacheManagerV2._build_nvfp4_index_k_sink
MiniMaxM31KVCacheManagerV2._extra_bytes_per_token
MiniMaxM31KVCacheManagerV2._index_k_bytes_per_token
MiniMaxM31KVCacheManagerV2._torch_dtype_for_index_cache
MiniMaxM31KVCacheManagerV2.get_nvfp4_index_k_sink
_IndexKViewShim.get_index_k_buffer
_is_nvfp4_index_k
get_sparse_disable_index_value_layer_ids
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/kernels/msa_v1_decode.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_device_supported
_get_or_plan
_log_once
_unit_scale
_wrapper_class
is_enabled
try_decode
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/prewarm/core.py`

库存函数 15；已覆盖 0；未覆盖 15。

```text
_build_plan
_build_variant
_install_observer
_install_observer.<locals>._compile_only
_install_observer.<locals>.get_fmha_variant
_modules
_so
compile_variants
enumerate_variants
families_for
geometry
log
prewarm_for_runner
set_phase
stack_parts
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/prewarm/hook.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_sdwarm
before_msa_indexer_prewarm
warmup_scope
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/prewarm/sdwarm.py`

库存函数 25；已覆盖 0；未覆盖 25。

```text
_call
_install_compile_observer
_install_compile_observer.<locals>.msa_jit_post_compile_hook
_jit_fn
_layout
_layout_key
_make_recorder
_make_recorder.<locals>.recorder
_max_batch
_record
_reset
_restore
_rows
_skip
_warm
begin
cache_entries
decode_query_lens
enabled
finish
log
plan
plan_variants
plan_variants.<locals>.al
row_outermost
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/prewarm/selftest.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
main
main.<locals>._Noop.run
main.<locals>.no_compile
main.<locals>.recorder
main.<locals>.rows_tensors
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/comm/allgather_cast_kernel.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
_atomic_add_acq_rel
_globaltimer
_ld_acquire_sys
_multimem_max_release_sys
_multimem_st_v4
_store_fp32x8
_store_out
_trap
allgather_cast
allgather_cast_kernel
flag_base
slot_words
vectors
workspace_words
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/draft_probs_one_request.py`

库存函数 31；已覆盖 0；未覆盖 31。

```text
ProbLL.__init__
ProbLL._base
ProbLL._chain_args
ProbLL._consts
ProbLL._decline
ProbLL._scratch
ProbLL._stream
ProbLL.build
ProbLL.describe
ProbLL.launch_chain
ProbLL.launch_gather
ProbLL.launch_noise
ProbLL.launch_proposal
ProbLL.launch_reference_chain
ProbLL.run
ProbLL.self_check
ProbLL.why_not
_agree
_all_equal
_arg
_checksum
_chi2
_chi2_limit
_dot_gamma
_mix64
_ordered
_probll_noise
_probll_stats
_probll_write
_read_switch
attach
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/allgather_bf16_kernel.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
_atomic_add_acq_rel
_globaltimer
_ld_acquire_sys
_multimem_max_release_sys
_multimem_st_v4
_store_fp32x8
_store_out
_trap
allgather_cast
allgather_cast_kernel
flag_base
slot_words
vectors
workspace_words
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/dspark_glue_install.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
KvRewind.__call__
KvRewind.__init__
_env_disabled
engagement
install_dspark_glue
install_dspark_glue.<locals>.pick
replicated_embedding_inputs
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/dspark_markov_chain_kernel.py`

库存函数 20；已覆盖 0；未覆盖 20。

```text
_add_rn
_atomic_add_acq_rel
_better
_block_dot
_fetch_w1_row
_fma_rn
_hi_f32
_keep_better
_lo_f32
_pack_bf16x2
_reduce_slots
_warp_best
buffer_words
fits
k3_markov
k3_markov_kernel
rows_per_cta
smem_bytes
smem_rows
supports
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/dspark_markov_chain_kernel_b1.py`

库存函数 20；已覆盖 0；未覆盖 20。

```text
_add_rn
_atomic_add_acq_rel
_better
_block_dot
_fetch_w1_row
_fma_rn
_hi_f32
_keep_better
_lo_f32
_pack_bf16x2
_reduce_slots
_warp_best
buffer_words
fits
k3_markov
k3_markov_kernel
rows_per_cta
smem_bytes
smem_rows
supports
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/dspark_markov_chain_kernel_b1_ll.py`

库存函数 23；已覆盖 0；未覆盖 23。

```text
_add_rn
_atomic_add_acq_rel
_bar_arrive
_bar_sync
_better
_block_dot
_fetch_w1_row
_fma_rn
_hi_f32
_keep_better
_lo_f32
_pack_bf16x2
_post_ll
_reduce_slots
_warp_best
buffer_words
k3_markov
k3_markov_kernel
ll_region_words
ll_workspace_words
rows_per_cta
smem_bytes
supports
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/gemm/router_half_pin.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_load
arm_router_half_pin
arm_router_half_pin.<locals>.route_owned
try_router_half_pin
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/green_partition.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
GreenPartition.describe
GreenPartition.disjoint
_build
_ck
_probe_cubin
get
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/idxk_split.py`

库存函数 16；已覆盖 0；未覆盖 16。

```text
IdxkPlan.rows
IdxkWorkspace.layer
compact_width
create_workspace_mnnvl
create_workspaces_emulated
exchange_q
forward
init_local
layer_words
local_topk
make_plan
merge
ops_available
plan_step
reference_table
score
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/moe/router_split.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_router_split_kernel
router_split_applies
router_split_logits
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/step_hooks.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
M3DSparkStepFold.__init__
M3DSparkStepFold.begin_step
M3DSparkStepFold.bind_engine_buffers
M3DSparkStepFold.bind_sampler_store
M3DSparkStepFold.gather
M3DSparkStepFold.sampler_rows_stored
M3DSparkStepFold.scatter
_engaged
_op
msa_kv_update
ops_available
spec_offsets
step_gather
step_prologue
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/internode/lsa_peer_mapper.py`

库存函数 19；已覆盖 0；未覆盖 19。

```text
LsaPeerMapperDevice.__extract_mlir_attributes__
LsaPeerMapperDevice.__extract_mlir_values__
LsaPeerMapperDevice.__get_mlir_types__
LsaPeerMapperDevice.__new_from_mlir_values__
LsaPeerMapperDevice._offset_for_local_peer
LsaPeerMapperDevice.map
LsaPeerMapperDevice.map_pointer
LsaPeerMapperDevice.ranks_per_node
LsaPeerMapperHost._as_int32
LsaPeerMapperHost._as_int64
LsaPeerMapperHost.make_device_object
LsaPeerMapperHost.map_peer_offset
LsaPeerMapperHost.validate
_LsaPeerMapperHostAdapter.__c_pointers__
_LsaPeerMapperHostAdapter.__extract_mlir_values__
_LsaPeerMapperHostAdapter.__get_mlir_types__
_LsaPeerMapperHostAdapter.__init__
_LsaPeerMapperHostAdapter.__new_from_mlir_values__
_device_trap
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/internode/node_local_barrier.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
NodeLocalNvlinkBarrier.__init__
NodeLocalNvlinkBarrier.arrive_and_wait
NodeLocalNvlinkBarrier.assign_local_rank
NodeLocalNvlinkBarrier.remove_device_members
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/internode/relay_message.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
_round_up
build_relay_message_layout
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/internode/router_internode.py`

库存函数 57；已覆盖 0；未覆盖 57。

```text
InternodeMetadataRouter.__init__
InternodeMetadataRouter._bind_router_payload_inputs
InternodeMetadataRouter._copy_vector16
InternodeMetadataRouter._device_gin
InternodeMetadataRouter._exchange_remote_splits
InternodeMetadataRouter._internode_histogram
InternodeMetadataRouter._is_first_hit_of_node
InternodeMetadataRouter._pack_one_message
InternodeMetadataRouter._pack_relay_messages
InternodeMetadataRouter._prepare_router_gin_resources
InternodeMetadataRouter._register_extra_data_smem
InternodeMetadataRouter._register_router_workspace
InternodeMetadataRouter._release_router_gin_resources
InternodeMetadataRouter._remote_route
InternodeMetadataRouter._remote_slot_geometry
InternodeMetadataRouter._remote_slot_geometry_for
InternodeMetadataRouter._resolve_pack_cta_count
InternodeMetadataRouter._route_word_and_weight
InternodeMetadataRouter._router_extra_cta_body
InternodeMetadataRouter._store_route_word
InternodeMetadataRouter._wait_remote_splits_kernel
InternodeMetadataRouter._window_tensor
InternodeMetadataRouter.barrier_signal_base
InternodeMetadataRouter.combine_arrival_signal_base
InternodeMetadataRouter.combine_range_bytes
InternodeMetadataRouter.combine_row_bytes
InternodeMetadataRouter.exchanges_remote_splits
InternodeMetadataRouter.final_pool_base_tensor
InternodeMetadataRouter.final_records_tensor
InternodeMetadataRouter.final_scores_tensor
InternodeMetadataRouter.final_sizes_tensor
InternodeMetadataRouter.header_plane_host
InternodeMetadataRouter.impl_desc_require
InternodeMetadataRouter.intra_count_tensor
InternodeMetadataRouter.launch_wait_remote_splits
InternodeMetadataRouter.node_experts
InternodeMetadataRouter.problem_desc_require
InternodeMetadataRouter.relay_bucket_count
InternodeMetadataRouter.relay_header_bytes
InternodeMetadataRouter.relay_header_plane_bytes
InternodeMetadataRouter.relay_header_weights_offset
InternodeMetadataRouter.relay_node_signal_base
InternodeMetadataRouter.relay_range_bytes
InternodeMetadataRouter.remote_node_count
InternodeMetadataRouter.remote_node_slots
InternodeMetadataRouter.required_signal_count
InternodeMetadataRouter.send_count_stride
InternodeMetadataRouter.send_count_tensor
InternodeMetadataRouter.send_token_count_tensor
InternodeMetadataRouter.splits_row_bytes
InternodeMetadataRouter.splits_row_pitch_bytes
InternodeMetadataRouter.splits_row_pitch_int32
InternodeMetadataRouter.splits_rows_host
InternodeMetadataRouter.stamp
InternodeMetadataRouter.tokens_to_node_tensor
InternodeMetadataRouter.wait_for_layout_ready
InternodeMetadataRouter.wait_for_relay_records
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/internode/token_comm_internode.py`

库存函数 56；已覆盖 0；未覆盖 56。

```text
TokenCommInternodeGin.__init__
TokenCommInternodeGin._build_fc2_cache
TokenCommInternodeGin._combine_join_barrier
TokenCommInternodeGin._finish_gather_worker
TokenCommInternodeGin._gather_items
TokenCommInternodeGin._gather_reduce_and_return
TokenCommInternodeGin._issue_rail_arrival
TokenCommInternodeGin._issue_return_put
TokenCommInternodeGin._rail_barrier
TokenCommInternodeGin._relay_done_barrier
TokenCommInternodeGin._relay_header_address
TokenCommInternodeGin._return_partials_and_wait
TokenCommInternodeGin._scan_relay_routes
TokenCommInternodeGin._sm0_metadata_phase
TokenCommInternodeGin._stamp_local_fc2_done
TokenCommInternodeGin._stamp_records_and_fc2_completion
TokenCommInternodeGin._validate_internode_configuration
TokenCommInternodeGin._wait_layout_published
TokenCommInternodeGin._window_tensor
TokenCommInternodeGin.assign_device_members
TokenCommInternodeGin.combine_landing_tensor
TokenCommInternodeGin.combine_worker_total
TokenCommInternodeGin.combine_worker_warps_per_cta
TokenCommInternodeGin.epilogue_tail_body
TokenCommInternodeGin.fc1_topk_scores_tensor
TokenCommInternodeGin.fc2_done_buffer_space
TokenCommInternodeGin.fc2_done_fanout
TokenCommInternodeGin.fc2_done_release_scope
TokenCommInternodeGin.impl_desc_require
TokenCommInternodeGin.local_expert_sizes
TokenCommInternodeGin.message_index_tensor
TokenCommInternodeGin.name_token
TokenCommInternodeGin.pool_expert_base_tensor
TokenCommInternodeGin.prepare_device_resources
TokenCommInternodeGin.publishes_fc2_done
TokenCommInternodeGin.pull_source_addresses
TokenCommInternodeGin.register_smem_regions
TokenCommInternodeGin.relay_landing_tensor
TokenCommInternodeGin.relay_record_topk_marker
TokenCommInternodeGin.relay_slot_count
TokenCommInternodeGin.relay_warp_body
TokenCommInternodeGin.relay_warp_total
TokenCommInternodeGin.relays
TokenCommInternodeGin.remove_device_members
TokenCommInternodeGin.requires_gin_resources
TokenCommInternodeGin.reset_tail
TokenCommInternodeGin.token_back_pushes
TokenCommInternodeGin.token_src_metadata_pointer
TokenCommInternodeGin.token_src_metadata_tensor
TokenCommInternodeGin.wait_for_pool_record
TokenCommInternodeGin.wait_for_sizes_ready
_FanoutSysReleaseFlagBatchTracker._make
_FanoutSysReleaseFlagBatchTracker.fire
_Fc2DoneFanout.make_tracker
_fanout_desc_class
_fanout_tracker_class
```

### `tensorrt_llm/_torch/moe/kernels/megamoe_scheduler/cuda_scheduler/multirank_dist.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
FabricSymmetricBuffer.__init__
FabricSymmetricBuffer._map
FabricSymmetricBuffer.close
FabricSymmetricBuffer.import_peer
_check
```

### `tensorrt_llm/_torch/moe/kernels/megamoe_scheduler/sami/fabric.py`

库存函数 58；已覆盖 0；未覆盖 58。

```text
FabricAllocation._map
FabricAllocation.create
FabricAllocation.import_peer
MulticastGroup._create_single_for_members
MulticastGroup.aliases
MulticastGroup.close_collectively
MulticastGroup.closed
MulticastGroup.create
MulticastGroup.create_for_members
MulticastGroup.create_for_members.<locals>.all_gather
MulticastGroup.create_for_members.<locals>.validate_aligned_group
MulticastGroup.granularity_for
MulticastGroup.metadata
MulticastTeam.__init__
MulticastTeam._call
MulticastTeam._initialize
MulticastTeam._initialize.<locals>.access_uc
MulticastTeam._initialize.<locals>.bind
MulticastTeam._initialize.<locals>.create
MulticastTeam._initialize.<locals>.export
MulticastTeam._initialize.<locals>.granularity
MulticastTeam._initialize.<locals>.import_member
MulticastTeam._initialize.<locals>.map_mc
MulticastTeam._initialize.<locals>.map_uc
MulticastTeam._initialize.<locals>.own_backing
MulticastTeam._initialize.<locals>.reserve_mc
MulticastTeam._initialize.<locals>.reserve_uc
MulticastTeam._initialize.<locals>.uniform
MulticastTeam._phase
MulticastTeam._retain
MulticastTeam._validate
MulticastTeam.close_collectively
MulticastTeam.close_collectively.<locals>.release_backing
MulticastTeam.close_collectively.<locals>.release_creator
MulticastTeam.close_collectively.<locals>.release_imports
MulticastTeam.close_collectively.<locals>.unbind
MulticastTeam.close_collectively.<locals>.unmap_mc
MulticastTeam.close_collectively.<locals>.unmap_uc
MulticastTeam.create
MulticastTeam.create_dual
MulticastTeam.granularities_for
MulticastTeam.metadata
MulticastTeam.owns_backing
MulticastTeamError.__init__
MulticastTeamPair.__getattr__
MulticastTeamPair.__init__
MulticastTeamPair._phase
MulticastTeamPair._retain
MulticastTeamPair.aliases
MulticastTeamPair.close_collectively
MulticastTeamPair.create
MulticastTeamPair.create.<locals>.validate_arguments
MulticastTeamPair.metadata
MulticastTeamPair.selected_alias
_align_up
_allocation_properties
_read_write_access
pick_multicast_creator
```

### `tensorrt_llm/bench/dataclasses/enums.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
KVCacheDtypeEnum.get_build_options
```

### `tensorrt_llm/evaluate/lm_eval_tasks/gpqa/cot_zeroshot_aa/_generate_configs.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
main
```

### `tensorrt_llm/models/automodel.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
AutoConfig.from_hugging_face
AutoModelForCausalLM.from_hugging_face
AutoModelForCausalLM.get_trtllm_model_class
```

### `tensorrt_llm/quantization/utils/fp8_matrix_weight_dequant.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
dequant_fp8_nk_weight_auto_scale_layout
dequant_fp8_weight_two_dim_block_grid
```

### `tensorrt_llm/scaffolding/contrib/AsyncGeneration/stream_generation.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
StreamGenerationTask.create_from_generation_task
stream_generation_handler
stream_generation_handler.<locals>.get_step_or_more_tokens
```

### `tensorrt_llm/scaffolding/contrib/Coder/coder.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
Coder._finalize
Coder._user_prompt
CoderTask.create_from_prompt
SWEBenchCoder._finalize
_BaseCoderController.__init__
_BaseCoderController._finalize
_BaseCoderController._user_prompt
_BaseCoderController.clone
_BaseCoderController.process
_build_coder_scaffolding_llm
_wrap_with_detailed_profiler
create_coder_scaffolding_llm
create_swebench_coder_scaffolding_llm
```

### `tensorrt_llm/scaffolding/contrib/Coder/swebench_utils.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
_last_complete_task_tool_call
_parse_tool_arguments
_prepend_diff_git_headers
_repo_path_from_minus_header
_repo_path_from_plus_header
_strip_git_index_lines_after_diff_git
_strip_markdown_fences
extract_swebench_complete_task_for_preds
extract_swebench_model_patch_for_preds
normalize_swebench_pred_patch
```

### `tensorrt_llm/scaffolding/contrib/DeepConf/deep_conf_controller.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
DeepConfOfflineController.__init__
DeepConfOfflineController.clone
DeepConfOfflineController.process
DeepConfOfflineMajorityVoteController.__init__
DeepConfOfflineMajorityVoteController.clone
DeepConfOfflineMajorityVoteController.process
DeepConfOnlineController.__init__
DeepConfOnlineController.clone
DeepConfOnlineController.process
DeepConfOnlineMajorityVoteController.__init__
DeepConfOnlineMajorityVoteController.clone
DeepConfOnlineMajorityVoteController.get_conf_bar
DeepConfOnlineMajorityVoteController.parallel_generate
DeepConfOnlineMajorityVoteController.process
```

### `tensorrt_llm/scaffolding/contrib/DeepConf/deep_conf_utils.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
ConfidenceInfo.get_min_conf_grouped
ConfidenceInfo.get_statistics
ConfidenceInfo.should_stop
ConfidenceInfo.update_confidence_info
basic_majority_vote
majority_vote
weighted_majority_vote
```

### `tensorrt_llm/scaffolding/contrib/Dynasor/dynasor_controller.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
DynasorGenerationController.__init__
DynasorGenerationController.count_not_empty
DynasorGenerationController.obtain_answer
DynasorGenerationController.process
```

### `tensorrt_llm/scaffolding/contrib/Dynasor/evaluator.py`

库存函数 21；已覆盖 0；未覆盖 21。

```text
_fix_a_slash_b
_fix_fracs
_fix_sqrt
call_with_timeout
choice_answer_clean
convert_word_number
count_not_empty
equal_group
extract_answer
extract_boxed_answer
extract_first_boxed_answer
is_digit
math_equal
math_equal_process
numeric_equal
parse_digits
str_to_pmatrix
strip_string
symbolic_equal
symbolic_equal.<locals>._parse
symbolic_equal_process
```

### `tensorrt_llm/scaffolding/contrib/TreeInference/tree_controllers.py`

库存函数 27；已覆盖 0；未覆盖 27。

```text
MCTSController.__init__
MCTSController._backpropagate
MCTSController._create_expansion_prompt
MCTSController._parse_thoughts
MCTSController._select_best_leaf
MCTSController._select_best_leaf.<locals>.traverse
MCTSController.process
MCTSNode.select_best_child
MCTSNode.ucb1_score
TOTController.__init__
TOTController._combine_state_and_thought
TOTController._evaluation_prompt
TOTController._generate_prompt
TOTController._parse_approaches
TOTController._parse_approaches.<locals>.flush_current
TOTController._parse_approaches.<locals>.is_new_item
TOTController._parse_evaluation
TOTController._select_best_solution
TOTController._select_best_solution.<locals>.traverse
TOTController._select_thoughts
TOTController._select_thoughts.<locals>.confidence_weight
TOTController.process
TOTNode.__post_init__
TreeNode.add_child
TreeNode.get_path_to_root
TreeNode.is_leaf
TreeNode.is_root
```

### `tensorrt_llm/scaffolding/contrib/iter_research/agent.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
IterResearchController.__init__
IterResearchController.clone
IterResearchController.process
_parse_tool_arguments
_render_tool_call
create_iter_research_controller
create_iter_research_controller.<locals>.wrap_with_profiler
create_iter_research_scaffolding_llm
```

### `tensorrt_llm/scaffolding/contrib/iter_research/utils.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
check_report_action
extract_tags
get_tool_definitions
random_date
truncate_text
```

### `tensorrt_llm/scaffolding/contrib/mcp/chat_handler.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
add_param_if_not_none
chat_handler
combine_params_with_chat_task
fill_chat_task_with_response
is_deterministic_mode
```

### `tensorrt_llm/scaffolding/contrib/mcp/chat_task.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
ChatTask.create_from_prompt
ChatTask.from_messages
```

### `tensorrt_llm/scaffolding/contrib/mcp/mcp_controller.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
MCPController.__init__
MCPController.process
```

### `tensorrt_llm/scaffolding/contrib/mcp/mcp_task.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
MCPCallTask.create_mcptask
MCPListTask.create_mcptask
```

### `tensorrt_llm/scaffolding/contrib/mcp/mcp_utils.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
MCPClient.__init__
MCPClient.call_tool
MCPClient.cleanup
MCPClient.connect_to_streamable_http_server
MCPClient.list_tools
main
```

### `tensorrt_llm/scaffolding/contrib/mcp/mcp_worker.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
MCPWorker.__init__
MCPWorker.call_handler
MCPWorker.init_with_urls
MCPWorker.list_handler
MCPWorker.shutdown
```

### `tensorrt_llm/scaffolding/contrib/open_deep_research/researcher.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
Compressor.__init__
Compressor.clone
Compressor.process
ResearchChatWithMCPController.__init__
ResearchChatWithMCPController.clone
ResearchChatWithMCPController.process
ResearchTask.from_topic
Researcher.__init__
Researcher.clone
Researcher.process
_extraction_goal_from_chat
_parse_tool_arguments
```

### `tensorrt_llm/scaffolding/contrib/open_deep_research/supervisor.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
BriefController.__init__
BriefController.process
FinalReportController.__init__
FinalReportController.process
ResearchPlanningController.__init__
ResearchPlanningController.process
Supervisor.__init__
Supervisor.clone
Supervisor.process
create_open_deep_research_controller
create_open_deep_research_controller.<locals>.wrap_with_detailed_profiler
create_open_deep_research_scaffolding_llm
```

### `tensorrt_llm/scaffolding/contrib/open_deep_research/utils.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
get_today_str
```

### `tensorrt_llm/scaffolding/contrib/tree_of_thought_research/tot_research_controllers.py`

库存函数 21；已覆盖 0；未覆盖 21。

```text
TOTResearchController.__init__
TOTResearchController._branch_from_chat_task
TOTResearchController._branch_tool_call_id
TOTResearchController._evaluation_messages
TOTResearchController._expansion_messages
TOTResearchController._final_messages
TOTResearchController._make_expansion_tasks
TOTResearchController._parse_score
TOTResearchController._render_branch_state
TOTResearchController._render_path
TOTResearchController._run_tool
TOTResearchController._thought_from_tool
TOTResearchController.clone
TOTResearchController.process
_TOTBranch.path
_json_dumps
_parse_tool_arguments
_truncate_text
create_tot_research_controller
create_tot_research_controller.<locals>.wrap_with_profiler
create_tot_research_scaffolding_llm
```

### `tensorrt_llm/tools/ppl.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
ppl
```

### `tensorrt_llm/tools/profiler/nsys_profile_tools/gputrc2graph.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
GPUTrace2Graph.__init__
GPUTrace2Graph.anno_gpu_kernname
GPUTrace2Graph.anno_gpu_kernname.<locals>.anno_gpu_kernname_helper
GPUTrace2Graph.gen_graph
GPUTrace2Graph.gen_nonoverlapped_sum_from_gputrace
GPUTrace2Graph.gen_sum_file
GPUTrace2Graph.is_valid_file
GPUTrace2Graph.make_html
GPUTrace2Graph.make_nongpu_row
GPUTrace2Graph.should_gen_file
GPUTrace2Graph.sum_non_overlapping_intervals
load_engine_model
main
parse_tuple
```

### `tensorrt_llm/usage/schemas/__main__.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
validate
```

### `tests/conftest.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
pytest_configure
pytest_runtest_logfinish
pytest_runtest_logreport
pytest_runtest_setup
pytest_runtest_teardown
pytest_sessionfinish
```

### `tests/integration/defs/cpp/conftest.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
build_dir
build_fmha_head104_tests
build_google_tests
build_kv_cache_compression_tests
build_type
cpp_resources_dir
keep_log_files
lora_setup
python_exe
root_dir
```

### `tests/integration/defs/kv_cache/conftest.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
pytest_addoption
```

### `tests/integration/defs/triton_server/conftest.py`

库存函数 41；已覆盖 0；未覆盖 41。

```text
cleanup_engine_outputs
custom_user_workspace
find_repo_root
get_device_count
get_device_memory
inflight_batcher_llm_client_root
install_root_requirements
llm_backend_all_models_root
llm_backend_dataset_root
llm_backend_gpt_example_root
llm_backend_inflight_batcher_llm_root
llm_backend_llmapi_example_root
llm_backend_multimodal_example_root
llm_backend_root
llm_backend_venv
llm_backend_whisper_example_root
llm_models_root
models_root
output_dir
pytest_runtest_teardown
setup_cache_data
skip_by_device_count
skip_by_device_memory
tensorrt_llm_eagle_example_root
tensorrt_llm_enc_dec_example_root
tensorrt_llm_example_root
tensorrt_llm_gpt_example_root
tensorrt_llm_gptj_example_root
tensorrt_llm_llama_example_root
tensorrt_llm_medusa_example_root
tensorrt_llm_mllama_example_root
tensorrt_llm_opt_example_root
tensorrt_llm_whisper_example_root
test_case_name
tiny_llama_lora_model_root
total_gpu_memory_mib
trt_performance_cache_name
venv_check_call
venv_check_call.<locals>._war_check_call
venv_check_output
venv_check_output.<locals>._war_check_output
```

### `tests/microbenchmarks/attention_perf/conftest.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
pytest_configure
```

### `tests/microbenchmarks/bench_moe/conftest.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
pytest_configure
```

### `tests/qa_selection/tests/cases/conftest.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
get_sm_version
```

### `tests/qa_selection/tests/conftest.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
selection
```

### `tests/test_common/cluster_env.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_local_net_device
apply_ucx_env
detect_gpu_type
get_ucx_env
get_ucx_tls_cmd
gpu_type_from_stage_name
gpu_type_from_supported_gpus
```

## 附录 C：已触达但无库存函数覆盖的文件及函数（完整清单）

共 **259 个文件、2,159 个未覆盖函数**。各代码块只列未覆盖的 `qualname`；与对应文件路径组合后得到唯一函数键。

### `tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_foldq.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
_fold_pages_kernel
_fold_q_kernel
_fold_reduce_kernel
_sf_offset
run_fp4_mla_foldq
run_fp4_mla_foldq.<locals>.workspace
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_kernels.py`

库存函数 22；已覆盖 0；未覆盖 22。

```text
_fp4_e2m1_quantize
_fp4_e2m1_to_f32
_fp4_mla_chunked_cache_gather_kernel
_fp4_mla_context_cache_update_kernel
_fp4_mla_floor_half_up
_fp4_mla_generation_fused_qk_rope_cache_update_kernel
_fp4_mla_generation_q1_kv_tail8_update
_fp4_mla_generation_q1_kv_tile_update
_fp4_mla_pack_e2m1x2_rne
_fp4_mla_pack_e2m1x8_rne
_fp4_mla_q_sf_offset
_fp4_mla_q_sf_offset_h128
_fp4_mla_quantize_q_groups
_fp4_mla_quantize_q_groups_words
_fp4_mla_quantize_q_residual_groups
_fp4_mla_rcp_approx_ftz
_fp4_mla_rebuild_v_scale_from_k_scale_kernel
_fp4_mla_rope_fp32
_fp4_mla_swizzled_sf_offset
_fp4_mla_unpack_e2m1x2
_fp4_mla_warp_max
_fp8_mla_context_block_table_kernel
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_triton.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
_fp4_e2m1_quantize
_fp4_e2m1_quantize_packed
_fp4_mla_attention_group_reduce_stats_kernel
_fp4_mla_attention_page_stats_kernel
_fp4_mla_attention_prob_scale_kernel
_fp4_mla_attention_pv_kernel
_fp4_mla_attention_pv_reduce_kernel
_fp4_mla_attention_reduce_stats_kernel
_fp4_mla_qk_scores_tile
_fp4_mla_swizzled_sf_offset_row_block
_fp4_pack_high_nibbles
_fp4_pack_low_nibbles
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_context/sage_scales.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
_load_float4_at
_section_address
load_sage_k_tile
load_sage_v_channels
prefetch_sage_k_tile
sage_k_block_word
sage_k_section_address
sage_k_tile_bytes
sage_k_tile_words
sage_k_transport
sage_score_scaling
sage_v_section_address
validate_context_sage_layout
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/smem_transformed_kv.py`

库存函数 25；已覆盖 0；未覆盖 25。

```text
SmemTransformedKvResource._build_kv_desc
SmemTransformedKvResource._create_initial_task_locals
SmemTransformedKvResource._create_work_tile_task_locals
SmemTransformedKvResource._dst_q_smem_byte_offset
SmemTransformedKvResource._init_placeholder_state
SmemTransformedKvResource._load_nvfp4_scale_byte
SmemTransformedKvResource._transform_load
SmemTransformedKvResource._transform_load_fp8_chunk
SmemTransformedKvResource._transform_load_nvfp4_chunk
SmemTransformedKvResource.get_smem_requirements
SmemTransformedKvResource.get_tmem_requirements
SmemTransformedKvResource.init_descriptor_state
SmemTransformedKvResource.init_load_state
SmemTransformedKvResource.k_desc_0
SmemTransformedKvResource.k_desc_1
SmemTransformedKvResource.transform_k0
SmemTransformedKvResource.transform_k1
SmemTransformedKvResource.transform_v0
SmemTransformedKvResource.transform_v1
SmemTransformedKvResource.v_desc_0
SmemTransformedKvResource.v_desc_1
_cvt_e2m1_word_to_e4m3_words
_cvt_e2m1x2_byte_to_bf16x2
_cvt_e4m3x2_byte_to_bf16x2
_cvt_e4m3x2_word_to_bf16x2
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_2cta/nvfp4_kv_load.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
arm_raw_load
convert_k_and_exchange_v
issue_k_gathers
load_nvfp4_kv_stage
load_nvfp4_source_stage
load_shared_row_quad
load_shared_words
nvfp4_kv_scratch_bytes
publish_nvfp4_v
shared_byte_view
store_v_remote
tile_is_fp8
wait_raw_load
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/reduction.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
_run_active_gmem_reduction_kernel
gmem_reduction_launch_shape
run_gmem_reduction_kernel
runtime_seq_len_kv_for_reduction
slice_split_num_reduction_ctas
slice_split_num_slices
slice_split_rows_per_slice
uses_q_row_gmem_reduction
uses_slice_split_gmem_reduction
uses_two_head_gmem_reduction
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/sage/plan.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
build_sage_prepared_scales
get_compiled_sage_prepare
get_compiled_sage_prepare.<locals>.fake_compact
get_compiled_sage_prepare.<locals>.prepare_adapter
```

### `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/triton_kernels.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_block_max_score_kernel
_sparse_softmax_kernel
triton_block_max_score
triton_sparse_softmax
```

### `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/triton_metadata.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
MiniMaxM3AttentionMetadata._maybe_get_m3_static_buffers
MiniMaxM3AttentionMetadata.on_update_kv_lens
MiniMaxM3AttentionMetadata.prepare
MiniMaxM3TritonSparseAttentionMetadata.prepare
_build_runtime_metadata_fresh
allocate_minimax_m3_static_buffers
build_runtime_metadata_from_kv_manager
derive_decode_cache_slots
derive_q_positions_and_cache_slots
ensure_metadata_on_device
ensure_metadata_on_device.<locals>._move
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/adapter/_capture_status.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
_capture_runtime
_check_capture_status
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/adapter/kernels/continuous_topk_metadata.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
SelectiveContinuousCapacityRefreshKernel.__call__
SelectiveContinuousCapacityRefreshKernel._total_capacity_code
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/boundary_extremes.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
select_extreme
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/native_lowp_math.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
native_add
native_fma
native_fp8_pack
native_fp8_pack_relu
native_hi
native_lo
native_pack
native_pack_relu
native_sum_fp32
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/proposal_warp_128.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
propose_warp_128
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/short_domain.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
ShortDomainFP4.__call__
ShortDomainFP8.__call__
ShortDomainSetup.__init__
ShortDomainSetup.kernel
ShortDomainSetup.launch
candidate_bounds
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/subset_named_barrier.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
SubsetNamedBarrier.arrive
SubsetNamedBarrier.arrive_and_wait
SubsetNamedBarrier.wait_unaligned
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/upper_proposal.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
propose
propose_cached
```

### `tensorrt_llm/_torch/attention/kernels/rubin/dsv4_qb_fusion/clc_tile_scheduler.py`

库存函数 15；已覆盖 0；未覆盖 15。

```text
ClcDynamicPersistentTileScheduler.__extract_mlir_values__
ClcDynamicPersistentTileScheduler.__init__
ClcDynamicPersistentTileScheduler.__new_from_mlir_values__
ClcDynamicPersistentTileScheduler._swizzle_and_rasterize
ClcDynamicPersistentTileScheduler.advance_to_next_work
ClcDynamicPersistentTileScheduler.create
ClcDynamicPersistentTileScheduler.get_current_work
ClcDynamicPersistentTileScheduler.get_grid_shape
ClcDynamicPersistentTileScheduler.initial_work_tile_info
ClcDynamicPersistentTileScheduler.num_tiles_executed
ClcDynamicPersistentTileScheduler.work_tile_info_from_clc_response
ClcDynamicPersistentTileSchedulerParams.__extract_mlir_values__
ClcDynamicPersistentTileSchedulerParams.__init__
ClcDynamicPersistentTileSchedulerParams.__new_from_mlir_values__
ClcDynamicPersistentTileSchedulerParams.get_grid_shape
```

### `tensorrt_llm/_torch/attention/kernels/rubin/dsv4_qb_fusion/kernel.py`

库存函数 24；已覆盖 0；未覆盖 24。

```text
CuteBlockScaledGemmFusedRMSNormRopeQuant.__call__
CuteBlockScaledGemmFusedRMSNormRopeQuant.__init__
CuteBlockScaledGemmFusedRMSNormRopeQuant._check_cluster
CuteBlockScaledGemmFusedRMSNormRopeQuant._cluster_layouts
CuteBlockScaledGemmFusedRMSNormRopeQuant._compute_grid
CuteBlockScaledGemmFusedRMSNormRopeQuant._mainloop_s2t_copies
CuteBlockScaledGemmFusedRMSNormRopeQuant._mainloop_s2t_copy_and_partition
CuteBlockScaledGemmFusedRMSNormRopeQuant._mainloop_s2t_copy_and_partition.<locals>.append_mn_broadcast_mode
CuteBlockScaledGemmFusedRMSNormRopeQuant._make_tiled_mmas
CuteBlockScaledGemmFusedRMSNormRopeQuant._make_tiled_mmas.<locals>.make
CuteBlockScaledGemmFusedRMSNormRopeQuant._make_tma_atoms
CuteBlockScaledGemmFusedRMSNormRopeQuant._position_for_row
CuteBlockScaledGemmFusedRMSNormRopeQuant._setup_attributes
CuteBlockScaledGemmFusedRMSNormRopeQuant.kernel
CuteBlockScaledGemmFusedRMSNormRopeQuant.kernel_body
CuteBlockScaledGemmFusedRMSNormRopeQuant.kernel_body.<locals>.cs_transpose_half
CuteBlockScaledGemmFusedRMSNormRopeQuant.kernel_body.<locals>.rope_chunk_body
CuteBlockScaledGemmFusedRMSNormRopeQuant.kernel_body.<locals>.store_chunk
_CompiledHandle.__call__
_CompiledHandle.__init__
_fma_packed_f32x2_bf16x2_f32x2_f32x2
_prmt_trunc_bf16x2
_prmt_trunc_bf16x2_neg_lo
compile
```

### `tensorrt_llm/_torch/cuda_tile_utils.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
ceil_div
next_power_of_2
```

### `tensorrt_llm/_torch/custom_ops/cuda_tile_custom_ops.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_
cuda_tile_rms_norm
cuda_tile_rms_norm_fuse_residual_
```

### `tensorrt_llm/_torch/cute_dsl_kernels/minimax_m3/embedding_firstnorm/m3_embedding_firstnorm_kernel.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
m3_embed_norm
m3_embed_norm_kernel
```

### `tensorrt_llm/_torch/disaggregation/orchestration/interfaces.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
ActiveRequestRegistry.active_requests
ActiveRequestRegistry.canceled_request_ids
ActiveRequestRegistry.contains
ActiveRequestRegistry.remove
ExecutorEffects.fail_fatal
ExecutorEffects.fail_requests
ExecutorEffects.prepare_gen_resources
ExecutorEffects.revert_ctx_alloc
ExecutorEffects.stage_transfer_response
ExecutorEffects.terminate_request
```

### `tensorrt_llm/_torch/distributed/symm_mem_allgather.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
SymmetricMemoryAllGather.__init__
SymmetricMemoryAllGather._allgather_dim0
SymmetricMemoryAllGather._normalize_dim
SymmetricMemoryAllGather.can_use_symm_mem
SymmetricMemoryAllGather.forward
```

### `tensorrt_llm/_torch/distributed/symm_mem_allreduce.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
SymmetricMemoryAllReduce.__init__
SymmetricMemoryAllReduce.can_use_symm_mem
SymmetricMemoryAllReduce.forward
SymmetricMemoryAllReduce.process_group
```

### `tensorrt_llm/_torch/kernels/rms_norm.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
rms_norm_kernel
rms_norm_kernel_gather
rms_norm_kernel_static_persistent
```

### `tensorrt_llm/_torch/kernels/rms_norm_fuse_residual.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
rms_norm_fuse_residual_kernel
rms_norm_fuse_residual_kernel_gather
rms_norm_fuse_residual_kernel_static_persistent
```

### `tensorrt_llm/_torch/kernels/rubin/dense_gemm_persistent.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
SM107PersistentDenseGemmKernel.__call__
SM107PersistentDenseGemmKernel.__init__
SM107PersistentDenseGemmKernel._create_tiled_mma
SM107PersistentDenseGemmKernel._create_tiled_mma_bkeep
SM107PersistentDenseGemmKernel._create_tiled_mma_breuse
SM107PersistentDenseGemmKernel._get_mma_permutation_mnk
SM107PersistentDenseGemmKernel._setup_attributes
SM107PersistentDenseGemmKernel.can_implement
SM107PersistentDenseGemmKernel.check_mma_tiler_and_cluster_shape
SM107PersistentDenseGemmKernel.check_supported_dtypes
SM107PersistentDenseGemmKernel.kernel
SM107PersistentDenseGemmKernel.wrapper
```

### `tensorrt_llm/_torch/kernels/utils/gemm/sm100.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
_check_tile_scheduler
epilogue
epilogue_smem_copy_and_partition
epilogue_tma_store
epilogue_tmem_copy_and_partition
transform_partitioned_tensor_layout
```

### `tensorrt_llm/_torch/late_compile.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
_CuteCompile.__call__
_CuteCompile.__getattr__
_CuteCompile.__getitem__
_CuteCompile.__init__
_hook_cute
_hook_triton
_hook_triton.<locals>.hook
mark_server_ready
note
```

### `tensorrt_llm/_torch/llm.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
LLM.__init__
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/fake_ops.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/moe_frontend_router_tc.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
PersistentDenseGemmKernel.__init__
_compile_gemm
_device_index
_require_pdl_contract
_splitk2_reduce_kernel
_validate_router_tc_partials
_work_clusters
compile_router_tc_partials
router_tc_uses_pdl
run_router_tc_partials
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/fake_ops.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_fake_route_quant
head_gather
sample_glue_adp
worklist_member
worklist_pipelined
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/attention/gvr_topk.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_supported_device
run
supports
workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/kernels/attention/indexer_prologue.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_apply_rope
_dsa_indexer_prologue_fused_kernel
_fp8_row_scale
_heads_per_prog
_rcp_approx_ftz
_rope_terms
_unravel_offset
dsa_indexer_prologue_fused
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/full_query_policy.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
fp8_full_query_weight_cache_limit
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/indexer.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
RubinIndexer.__init__
RubinIndexer._fused_prologue_eligible
RubinIndexer._is_prefix_view
RubinIndexer._prologue_fused
RubinIndexer._update_k_cache
RubinIndexer.can_skip_projection
RubinIndexer.pre_indexer_proj
RubinIndexer.reused_projection_outputs
RubinIndexer.sparse_attn_indexer
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/owner_worklist.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
Nvfp4OwnerWorklist.__init__
Nvfp4OwnerWorklist._group
Nvfp4OwnerWorklist._pool
Nvfp4OwnerWorklist.begin_call
Nvfp4OwnerWorklist.ensure_capacity
Nvfp4OwnerWorklist.owns
Nvfp4OwnerWorklist.record_remap
Nvfp4OwnerWorklist.reset
Nvfp4OwnerWorklist.try_gather
_max_group_size
_tensor
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/topk.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
RubinTopK._forward_decode_gvr
_fake_gvr_topk_decode
gvr_topk_decode
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_dsa.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_forward_dsa_attn
_get_short_seq_mha_threshold
forward_context_sparse_attn
forward_dsa_proj
forward_generation_sparse_attn
forward_sparse_attn
forward_sparse_attn_custom_op
should_use_short_mha
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_dsa_backend.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
Glm52DSAAttention.__init__
Glm52DSAAttention._grouped_remap_topk_to_global
Glm52DSAAttention._remap_topk_to_global
Glm52DSAAttention.mla_rope_append_paged_kv_assign_q
Glm52DSAAttention.sparse_attn_predict
Glm52DSAAttention.sparse_kv_predict
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_dsa_custom_ops.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_create_mla_outputs_fake
_extract_extra_attrs
_mla_dsa_attn_inplace_fake
_mla_dsa_proj_fake
create_mla_outputs
mla_dsa_attn_inplace
mla_dsa_proj
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_indexer.py`

库存函数 27；已覆盖 0；未覆盖 27。

```text
Glm52Indexer.__init__
Glm52Indexer._call_mqa_logits
Glm52Indexer._call_paged_mqa_logits
Glm52Indexer._copy_reused_topk
Glm52Indexer._fill_decode_topk
Glm52Indexer._mtp_last_accepted_rows
Glm52Indexer._prep_q_or_k
Glm52Indexer._qk_projection_and_rope
Glm52Indexer._store_mtp_shared_rows
Glm52Indexer._update_k_cache
Glm52Indexer._weight_scale
Glm52Indexer.cache_derived_state
Glm52Indexer.forward
Glm52Indexer.forward_from_projected
Glm52Indexer.forward_prepared
Glm52Indexer.maybe_join_prev_topk_copy
Glm52Indexer.post_load_weights
Glm52Indexer.pre_indexer_proj
Glm52Indexer.sparse_attn_indexer
_prefill_q_split_threshold
_scale
_split_prefill_queries
_tf32_matmul_enabled
_to_float
transform_local_topk_and_prepare_pool_view
transform_local_topk_and_prepare_pool_view_grouped
use_cute_dsl_paged_mqa_logits
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_megakernel.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
GlmMegakernel.__init__
GlmMegakernel.begin
GlmMegakernel.create
_Layer.__init__
_Layer.bind
_Layer.bind.<locals>.put
_Layer.forward
_Shared.__init__
_Shared.__init__.<locals>.empty
_u8
build_roles
build_roles.<locals>.add
pack_bf16
swizzle_64b
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_mla.py`

库存函数 34；已覆盖 0；未覆盖 34。

```text
Glm52MLA.__init__
Glm52MLA.__init__.<locals>.yarn_get_mscale
Glm52MLA._attn_forward_gen
Glm52MLA._bind_v_b_proj_weight
Glm52MLA._bmm_bf16_out
Glm52MLA._copy_absorption_weight_shards
Glm52MLA._create_outputs
Glm52MLA._get_kvbproj_head_weights
Glm52MLA._has_full_absorption_weights
Glm52MLA._init_fused_q_fp8_context
Glm52MLA._project_context_kv
Glm52MLA._q_a_layernorm_maybe_fused
Glm52MLA._rebuild_locality_domain_bf16_bmm_weight_shards
Glm52MLA._resolve_qa_fused_scale
Glm52MLA._use_fused_q_fp8_context
Glm52MLA._use_kvbproj_strided_out
Glm52MLA.cache_derived_state
Glm52MLA.cached_warmup_forward_context_with_cached_kv
Glm52MLA.cached_warmup_forward_context_with_cached_kv.<locals>.warmup
Glm52MLA.create_output
Glm52MLA.create_weights
Glm52MLA.forward
Glm52MLA.forward_absorption_context
Glm52MLA.forward_absorption_generation
Glm52MLA.forward_absorption_generation.<locals>._mla_gen_rope
Glm52MLA.forward_context
Glm52MLA.forward_context_default
Glm52MLA.forward_context_with_cached_kv
Glm52MLA.forward_context_with_chunked_prefill
Glm52MLA.forward_impl
Glm52MLA.post_load_weights
Glm52MLA.transform_weights
_create_attention
fp8_block_scaling_bmm_out
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_model.py`

库存函数 31；已覆盖 0；未覆盖 31。

```text
Glm52Attention.__init__
Glm52DecoderLayer.__init__
Glm52DecoderLayer._apply_next_layer_layernorm
Glm52DecoderLayer._compute_mlp_tp_size
Glm52DecoderLayer._get_decoder_layer_quant_config
Glm52DecoderLayer.forward
Glm52DecoderLayer.forward_MoE
Glm52DecoderLayer.forward_MoE.<locals>._run_MoE
Glm52DecoderLayer.forward_mlp
Glm52Gate.__init__
Glm52Gate.apply
Glm52Gate.forward
Glm52Gate.get_experts_per_token
Glm52Gate.load_weights
Glm52Gate.routing_method
Glm52Linear.__init__
Glm52Linear.apply_linear
Glm52MoE.__init__
Glm52MoE._compute_shared_expert_tp_size
Glm52MoE._expert_weight_loading_mode
Glm52MoE._get_experts_quant_config
Glm52MoE._get_shared_experts_quant_config
Glm52MoE.compute_routed_output
Glm52MoE.forward
Glm52MoE.forward.<locals>._compute_routed_output
Glm52MoE.forward.<locals>._compute_shared_output
Glm52Model.__init__
Glm52Model.forward
_static_nvfp4_input_scale
moe_reduce_add_shared_output
setup_aliases
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_speculative.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
Glm52SpecDecOneEngineForCausalLM.__init__
Glm52SpecDecOneEngineForCausalLM.__post_init__
Glm52SpecDecOneEngineForCausalLM._quantization_named_modules
Glm52SpecDecOneEngineForCausalLM.cache_derived_state
Glm52SpecDecOneEngineForCausalLM.forward
Glm52SpecDecOneEngineForCausalLM.load_draft_weights
Glm52SpecDecOneEngineForCausalLM.mtp_head_module_names
Glm52SpecDecOneEngineForCausalLM.set_guided_decoder
_slice_spec_position_ids
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_weight_loader.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
Glm52WeightLoader.__init__
Glm52WeightLoader.load_weights
Glm52WeightLoader.load_weights.<locals>.detect_shared_mtp_weights
Glm52WeightLoader.load_weights.<locals>.load_kv_b_proj_and_k_b_proj_trans
Glm52WeightLoader.load_weights.<locals>.load_kv_b_proj_and_k_b_proj_trans_dequant
Glm52WeightLoader.load_weights.<locals>.rename_moe_weight
Glm52WeightLoader.load_weights.<locals>.requantize_weight_with_new_scale
Glm52WeightLoader.load_weights.<locals>.split
Glm52WeightLoader.load_weights.<locals>.split_kv_b_proj
Glm52WeightLoader.load_weights.<locals>.split_matrix_tp
weight_dequant
weight_dequant.<locals>.grid
weight_dequant_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/comm/_ipc_workspace.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
IpcWorkspace.args
_bind_local
bind_ipc_workspace
residual_norm_reference
validate_ipc_operands
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/comm/_mnnvl_workspace.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
LamportExchangeWorkspace.__init__
_all_reduce_workspace
hop_buffer_bytes
hop_supported
hop_workspace
publish_target
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/comm/intra_node_moe_finalize.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
GlmIntraNodeMoeFinalize.__call__
GlmIntraNodeMoeFinalize.is_valid
GlmIntraNodeMoeFinalize.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/comm/intra_node_reduce.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
GlmIntraNodeReduce.__call__
GlmIntraNodeReduce.is_valid
GlmIntraNodeReduce.reference
IpcCollectives._check
IpcCollectives.bind
IpcCollectives.finalize
IpcCollectives.plain_reduce
IpcCollectives.premoe
IpcCollectives.supported
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/comm/mnnvl_allgather.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
GlmMnnvlAllgather.__call__
GlmMnnvlAllgather.is_valid
GlmMnnvlAllgather.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/comm/mnnvl_hop.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
GlmMnnvlHop.__call__
GlmMnnvlHop.is_valid
GlmMnnvlHop.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/comm/mnnvl_moe_finalize_hop.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
GlmMnnvlMoeFinalizeHop.__call__
GlmMnnvlMoeFinalizeHop.is_valid
GlmMnnvlMoeFinalizeHop.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/comm/pre_moe_fp4_linear.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
GlmPreMoeFp4Linear.__call__
GlmPreMoeFp4Linear.is_valid
GlmPreMoeFp4Linear.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/weights.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_draft_weights
_draft_weights.<locals>.shard
load_weights
normalize_model_config
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_rms_norm_gated/k3_rms_norm_gated_kernel.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_gated_rows_load
_gated_rows_store
_layer_norm_fwd_1pass_kernel
_rms_norm_gated_fwd_multirow_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_situ_and_mul/k3_situ_and_mul_kernel.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
k3_situ_and_mul_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_comm/k3_latent_ar_norm.py`

库存函数 46；已覆盖 0；未覆盖 46。

```text
K3LatentArNormKernel.__call__
K3LatentArNormKernel.__init__
K3LatentArNormKernel.kernel
K3LatentArRowKernel.__call__
K3LatentArRowKernel.__init__
K3LatentArRowKernel.kernel
_amax_bf16x8
_amax_pair
_asm
_bar_sync
_bf16x2_f32
_clean4
_cluster_sync_release_acquire
_compiled
_compiled_row
_fadd
_fmul
_fv
_is_supported_device
_iv
_ld_acquire_gpu_u32
_ld_v4
_ld_vol_v4
_ld_vol_v4_pred
_lds_f32
_lv
_pack_bf16x2
_red_add_release_gpu
_rms_scale
_sanitize
_shfl_bfly
_shfl_bfly_b32
_sq_bf16x2
_st_cluster_f32
_st_sentinel_v4
_st_u32
_st_u8
_st_v2
_st_v4
_sts_f32
_unpack
_warp_sync
can_run
can_run_row
run_k3_latent_ar_norm
run_k3_latent_ar_norm_row
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_fp4_mla/fp4_mla_kv_append.py`

库存函数 19；已覆盖 0；未覆盖 19。

```text
_ceil_div
_fp4_e2m1_quantize
_fp4_e2m1_to_f32
_fp4_mla_floor_half_up
_fp4_mla_generation_kv_update_kernel
_fp4_mla_pack_e2m1x2_rne
_fp4_mla_pack_e2m1x8_rne
_fp4_mla_q1_kv_blocks_per_program
_fp4_mla_q1_prefix_blocks_per_program
_fp4_mla_quantize_q_groups
_fp4_mla_quantize_q_groups_words
_fp4_mla_quantize_q_residual_groups
_fp4_mla_rcp_approx_ftz
_fp4_mla_rope_fp32
_fp4_mla_unpack_e2m1x2
_scatter_generation_kv_only
_scatter_generation_kv_only_2d
_scatter_generation_kv_only_2d.<locals>.launch_generation_update
fp4_mla_kv_append_generation
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_kda/_kda_decode.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_as_token_rows
_dummy_tensor
_require_cuda_bf16
_require_cuda_fp32
run_kda_decode_fusion_cuda
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_situ/situ_and_mul_kernel.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
situ_and_mul_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_spec/mla_split_plan.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
_mla_split_kv_plan_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__helix8_moeep8/weights.py`

库存函数 35；已覆盖 0；未覆盖 35。

```text
KimiK3DefaultTextWeights._attention_fp8_linears
KimiK3DefaultTextWeights._finalize_weight_load
KimiK3DefaultTextWeights._load_attention_fp8
KimiK3DefaultTextWeights._load_expert_slices
KimiK3DefaultTextWeights._load_expert_slices.<locals>.checkpoint_key
KimiK3DefaultTextWeights._load_expert_slices.<locals>.drop_file_pages
KimiK3DefaultTextWeights._load_expert_slices.<locals>.ensure_prepared
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_expert
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_expert_file
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_experts_from_weights
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_split_file_expert
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_split_file_expert.<locals>.get_optional_tensor
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_split_file_expert.<locals>.get_tensor
KimiK3DefaultTextWeights._load_expert_slices.<locals>.maybe_finalize_layer
KimiK3DefaultTextWeights._load_mla_fp8_kv_b
KimiK3DefaultTextWeights._load_trunk_params
KimiK3DefaultTextWeights._load_trunk_params.<locals>.load_param
KimiK3DefaultTextWeights._trunk_parameters
KimiK3DefaultTextWeights._validate_checkpoint_keys
KimiK3DefaultTextWeights.checkpoint_name_plan
KimiK3DefaultTextWeights.load_weights
KimiK3DefaultWeights.load_draft_weights
KimiK3DefaultWeights.load_weights
_checkpoint_fp8_pair
_dequantize_fp8_block_scaled
_fp8_block_scale_key
_full_projection_shape
_gate_up_ckpt_keys
_helix_cp_v_b_shard
_load_fp8_projection
_materialize
_projection_shard
_shard_dim
_shard_head_major_param
_shards_itself
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__helix8_moetp2ep4/weights.py`

库存函数 35；已覆盖 0；未覆盖 35。

```text
KimiK3DefaultTextWeights._attention_fp8_linears
KimiK3DefaultTextWeights._finalize_weight_load
KimiK3DefaultTextWeights._load_attention_fp8
KimiK3DefaultTextWeights._load_expert_slices
KimiK3DefaultTextWeights._load_expert_slices.<locals>.checkpoint_key
KimiK3DefaultTextWeights._load_expert_slices.<locals>.drop_file_pages
KimiK3DefaultTextWeights._load_expert_slices.<locals>.ensure_prepared
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_expert
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_expert_file
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_experts_from_weights
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_split_file_expert
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_split_file_expert.<locals>.get_optional_tensor
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_split_file_expert.<locals>.get_tensor
KimiK3DefaultTextWeights._load_expert_slices.<locals>.maybe_finalize_layer
KimiK3DefaultTextWeights._load_mla_fp8_kv_b
KimiK3DefaultTextWeights._load_trunk_params
KimiK3DefaultTextWeights._load_trunk_params.<locals>.load_param
KimiK3DefaultTextWeights._trunk_parameters
KimiK3DefaultTextWeights._validate_checkpoint_keys
KimiK3DefaultTextWeights.checkpoint_name_plan
KimiK3DefaultTextWeights.load_weights
KimiK3DefaultWeights.load_draft_weights
KimiK3DefaultWeights.load_weights
_checkpoint_fp8_pair
_dequantize_fp8_block_scaled
_fp8_block_scale_key
_full_projection_shape
_gate_up_ckpt_keys
_helix_cp_v_b_shard
_load_fp8_projection
_materialize
_projection_shard
_shard_dim
_shard_head_major_param
_shards_itself
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__tp1/weights.py`

库存函数 34；已覆盖 0；未覆盖 34。

```text
KimiK3DefaultTextWeights._attention_fp8_linears
KimiK3DefaultTextWeights._finalize_weight_load
KimiK3DefaultTextWeights._load_attention_fp8
KimiK3DefaultTextWeights._load_expert_slices
KimiK3DefaultTextWeights._load_expert_slices.<locals>.checkpoint_key
KimiK3DefaultTextWeights._load_expert_slices.<locals>.drop_file_pages
KimiK3DefaultTextWeights._load_expert_slices.<locals>.ensure_prepared
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_expert
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_expert_file
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_experts_from_weights
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_split_file_expert
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_split_file_expert.<locals>.get_optional_tensor
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_split_file_expert.<locals>.get_tensor
KimiK3DefaultTextWeights._load_expert_slices.<locals>.maybe_finalize_layer
KimiK3DefaultTextWeights._load_mla_fp8_kv_b
KimiK3DefaultTextWeights._load_trunk_params
KimiK3DefaultTextWeights._load_trunk_params.<locals>.load_param
KimiK3DefaultTextWeights._trunk_parameters
KimiK3DefaultTextWeights._validate_checkpoint_keys
KimiK3DefaultTextWeights.checkpoint_name_plan
KimiK3DefaultTextWeights.load_weights
KimiK3DefaultWeights.load_draft_weights
KimiK3DefaultWeights.load_weights
_checkpoint_fp8_pair
_dequantize_fp8_block_scaled
_fp8_block_scale_key
_full_projection_shape
_gate_up_ckpt_keys
_load_fp8_projection
_materialize
_projection_shard
_shard_dim
_shard_head_major_param
_shards_itself
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/_k3_fp4_mla_kernels.py`

库存函数 21；已覆盖 0；未覆盖 21。

```text
_fp4_e2m1_quantize
_fp4_e2m1_to_f32
_fp4_mla_chunked_cache_gather_kernel
_fp4_mla_context_cache_update_kernel
_fp4_mla_floor_half_up
_fp4_mla_generation_fused_qk_rope_cache_update_kernel
_fp4_mla_generation_q1_kv_tail8_update
_fp4_mla_generation_q1_kv_tile_update
_fp4_mla_pack_e2m1x2_rne
_fp4_mla_pack_e2m1x8_rne
_fp4_mla_q_sf_offset
_fp4_mla_q_sf_offset_h128
_fp4_mla_quantize_q_groups
_fp4_mla_quantize_q_groups_words
_fp4_mla_quantize_q_residual_groups
_fp4_mla_rcp_approx_ftz
_fp4_mla_rope_fp32
_fp4_mla_swizzled_sf_offset
_fp4_mla_unpack_e2m1x2
_fp4_mla_warp_max
_fp8_mla_context_block_table_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/k3_fp4_mla_decode.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
K3Fp4MlaDecode.__call__
K3Fp4MlaDecode.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/k3_attn_res_lamport.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
K3AttnResLamport.__call__
K3AttnResLamport.is_valid
K3AttnResLamport.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/k3_latent_ar_rmsnorm.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
K3LatentArRmsnorm.__call__
K3LatentArRmsnorm.is_valid
K3LatentArRmsnorm.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/common/sm107_launch.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
_value
log_constants
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/config.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_is_vl_config
_moe_layer_ids
_sparse_layer_ids
_text_model_config
_validate_runtime
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/attention/triton_sparse_decode.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_dequant_nvfp4_rows
_e2m1x2_to_f16x2
_gqa_sparse_decode_kernel
_gqa_sparse_decode_pair_kernel
_merge_topk_attn_out_kernel
_rcp_approx_ftz
_store_mxfp8_head
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3_force_vec.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
force_vec_kernel
force_vec_req_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3ll/die_local.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
DieLocal.__init__
assign_dies
block_want
build_roles
check_table
cta_die_bytes
granule_runs
place
prepare
probe_sm_ids
probe_sm_ids.<locals>.ck
role_table
unit_want
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/msa_kv_update.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
msa_kv_update
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/spec/m3_fold_tail.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
_fold_tail_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/spec/m3_force_vec.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
force_vec_kernel
force_vec_req_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/m31_capture_fork.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
_rank
capture_or_inline
capture_stream
fork_capture
join
pending
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/m31_decode_fsel.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
prepare
select
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/m31_drafter_layout.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
route_his_layout
route_his_layout.<locals>._compile_now
route_his_layout.<locals>._gemv_rows
route_his_layout.<locals>._his_taken
route_his_layout.<locals>._route_his_linear
route_his_layout.<locals>._route_his_linear.<locals>.apply_linear
route_his_layout.<locals>._route_his_mlp
route_his_layout.<locals>._route_his_mlp.<locals>.forward
route_his_layout.<locals>._row_parallel_reducing
route_his_layout.<locals>._wrap_row_parallel_forward
route_his_layout.<locals>._wrap_row_parallel_forward.<locals>.forward
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/m31_rubin_mxfp8.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_runner
declined
run
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/m31_tail_bookkeep.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
TailBookkeepMixin._forward_impl
TailBookkeepMixin._rng_state_per_request
TailBookkeepMixin._tailbk_rng_window
TailBookkeepMixin._tailbk_skips_padding_rows
TailBookkeepMixin._zero_padding_rows
tail_bookkeep_draft_len
tail_bookkeep_log_once
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/gumbel_draft.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
engagement
log_engagement
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/tail.py`

库存函数 28；已覆盖 0；未覆盖 28。

```text
_add
_forward_moe_norm
_is_deferred_topk_reduce
_log_once
_matches_flashinfer
_moe_forward
_on_pinned_mxfp8
_on_pinned_mxfp8.<locals>.run
_replaceable
_replaced
_routed_and_shared
_routed_and_shared.<locals>._compute_experts
_routed_and_shared.<locals>._compute_routed_output
_routed_and_shared.<locals>._compute_shared_on_side
_routed_and_shared.<locals>._compute_shared_output
_routed_beside_shared_expert
_routed_beside_shared_expert.<locals>.route_and_stage
_set_shared_plan
_tail_rows
boundary_norm_decline_reason
config_decline_reason
flashinfer_norm_decline_reason
install
install_shared_plan
moe_decline_reason
norm_decline_reason
serves
shared_plan_decline_reason
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/idx_scorer.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
_decline
_log_once
_module
_off_reason
_release_plan
_step_lengths
_to_device
_wrapper
accepts
prewarm
register
score
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/drafter_mxfp8.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_his_quantize
_projections
apply_linear_reached
executed_rows
mxfp8_from_bf16
route_mx_linears
route_mx_linears.<locals>.apply_linear
route_projection_layout
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark_draft_head.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
DraftHeadMx8.__call__
DraftHeadMx8.__init__
DraftHeadMx8._run
DraftHeadMx8.serves_head
DraftHeadMx8.why_not
_quantizer
_quantizer.<locals>.quantize
padded_n
quantize_weight
unfit
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark_t6.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
_oneshot_rows
_takes
drafter_gemm_routes
named_projections
route_lowm_linears
route_lowm_linears.<locals>.apply
route_of
routes_line
run_lowm
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark_tailfold.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
StepInputs.__init__
capturing
drafter_owns_noise_hook
memcpy86k_setup_line
own_noise
recording_inputs
recording_inputs.<locals>.hook
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark_tailfork.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
CtxStorePrep.__init__
copy_captures
ctx_store_launch
ctx_store_prep
view_key
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/fake_ops.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_ar_prepushed_fake
_s1b_fold_fake
_s2_moe_tail_fake
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/lc2_mixed.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
lc2_mixed_decode
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/long_select.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
long_select_blocks
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m31_route_ids16.py`

库存函数 18；已覆盖 0；未覆盖 18。

```text
_RouteIds16Kernel.__call__
_RouteIds16Kernel.__init__
_RouteIds16Kernel.kernel
_add_rn
_aot_sources
_asm
_bf16_bits
_compile
_div_rn
_fma_rn
_mul_rn
_redux_max_u32
_sub_rn
_twiddle
is_compiled
operands_ok
prebuild
route
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_draft_lowm.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
arm_push
draft_linear
installed
prebuild
take_pushed
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_draft_step.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_fixed_order
_grid
_num_sms
draft_step
head_supported
prepare_head
workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_drafter_attn.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
M3DrafterFullCtxFmha.__init__
M3DrafterFullCtxFmha._fused_rope_applies
M3DrafterFullCtxFmha._is_supported
M3DrafterFullCtxFmha._multi_token
M3DrafterFullCtxFmha._run_fused_rope
M3DrafterFullCtxFmha._shape
M3DrafterFullCtxFmha.run_generation
M3DrafterSlidingFiFmha.is_environment_available
M3DrafterSlidingFiFmha.run_context
M3DrafterSlidingFiFmha.run_generation
fwd_has_no_mrope
install
install_sliding_fi
sliding_fi_bypass_reason
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_fullctx_decode.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
FullCtxPlan.run
emits_mxfp8
has_own_reducer
plan_for
prebuild
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_mc_allreduce.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
McAllReduceWorkspace.create
McAllReduceWorkspace.flags_for
McBuckets.__init__
McBuckets.for_tokens
_ar_opts
_bucket
_capture_args
_tp_comm
decline_reason
get_workspace
m3_mc_allreduce_norm
m3_mc_finalize_allreduce_norm
tp_agree
tp_agree_built
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_moe/op16.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
M3Moe16State.__init__
M3Moe16State.layer
M3Moe16State.run
_aot_sources
_group_slots
make_phase_plan16
make_phase_plan16.<locals>.counts_for
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_nvfp4_index_decode.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
Nvfp4IndexDecodeScorer.__init__
Nvfp4IndexDecodeScorer._wrapper
Nvfp4IndexDecodeScorer.available
Nvfp4IndexDecodeScorer.history_workspace
Nvfp4IndexDecodeScorer.prebuild
Nvfp4IndexDecodeScorer.score
_interface
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_sampled_accept.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
SampledAcceptWorkspace.__init__
SampledAcceptWorkspace.create
_ws_key
sampled_accept
workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_shard_accept.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_num_sms
_workspace
create_workspace
shard_accept
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_target_head.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_num_sms
head_supported
logits
prepare_head
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_tokmap_guard.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_guarded
_indexed
_map_offset
_no_missed
_no_taken
_one_request_map
install
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/attention/m3_fullctx_decode.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3FullCtxDecode.__call__
M3FullCtxDecode.is_valid
M3FullCtxDecode.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/gemm/m31_agg_prefill_cublaslt.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_op
self_check
try_gemm
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/gemm/m31_fc_shard.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
FcShard.__call__
FcShard.__init__
FcShard._check
FcShard._rows
FcShard.applies
FcShard.build
FcShard.gather
FcShard.partial
_probes
fc_shard_op
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/gemm/m3_mxfp8_cublaslt.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
M3Mxfp8Cublaslt.__call__
M3Mxfp8Cublaslt.is_valid
M3Mxfp8Cublaslt.reference
_e4m3
_ue8m0
mxfp8_quantize
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/moe/m31_s1b_fold.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_bitwise
_slice_sum
_w_operand
fold_op
prebuild
route
w_strided
x_slices
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/moe/m3_moe_tactic_pin.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
run_pinned
tactic_for_rows
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/sampling/m31_chain_spec_mc.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
M31ChainSpecMc.__call__
M31ChainSpecMc.is_valid
M31ChainSpecMc.reference
chain_spec_mc
chain_spec_mc_op
chain_spec_mc_slots
chain_spec_mc_slots_op
ctas_per_row
max_batch
max_slot_batch
prebuild
sm_count
ws_bytes_per_row
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/sampling/m31_topp_det.py`

库存函数 13；已覆盖 0；未覆盖 13。

```text
M31ToppDet.__call__
M31ToppDet.is_valid
M31ToppDet.reference
_load_op
_log_once
ctas_per_row
is_v3
max_rows
prebuild
sm_count
top_p_renorm_det
topp_det_op
ws_bytes_per_row
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/spec/m31_markov16.py`

库存函数 18；已覆盖 0；未覆盖 18。

```text
M31Markov16Chain.__call__
M31Markov16Chain.is_valid
M31Markov16Chain.reference
_
_arg
_kernel
_launch
_m31_tp2_markov16
_m31_tp2_markov16_rewind
_sm_count
base_stride
compile_call
grid_decline
is_compiled
markov16_chain_rewind
pick_grid
supports
workspace_fits
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/spec/m3_spec_offsets.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3SpecOffsets.__call__
M3SpecOffsets.is_valid
M3SpecOffsets.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/spec_step_fold.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
M3SpecStepFold.__init__
M3SpecStepFold._note_tail_ran
M3SpecStepFold._wait_store_copies
M3SpecStepFold.begin_step
M3SpecStepFold.bind_engine_buffers
M3SpecStepFold.bind_sampler_store
M3SpecStepFold.gather
M3SpecStepFold.sampler_rows_stored
M3SpecStepFold.scatter
M3SpecStepFold.store_in_forward
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/drafter_mxfp8.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
_compile_now
_his_taken
_projections
_route_his_linear
_route_his_linear.<locals>.apply_linear
_route_his_mlp
_route_his_mlp.<locals>.forward
_row_parallel_reducing
_wrap_row_parallel_forward
_wrap_row_parallel_forward.<locals>.forward
mxfp8_from_bf16
route_his_layout
route_mx_linears
route_mx_linears.<locals>.apply_linear
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/dspark_markov_grid.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
_b1_module
_sm_count
catalog_grid
fits_workspace
grid_ok
is_compiled
markov_chain_b1
pick_grid
plan
precompile
supports
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/dspark_t6.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
drafter_gemm_routes
named_projections
route_of
routes_line
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/m3_draft_lowm.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
arm_push
draft_linear
installed
prebuild
take_pushed
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/m3_draft_step.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_fixed_order
_grid
_num_sms
draft_step
head_supported
prepare_head
workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/m3_drafter_attn.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
M3DrafterFullCtxFmha.__init__
M3DrafterFullCtxFmha._fused_rope_applies
M3DrafterFullCtxFmha._is_supported
M3DrafterFullCtxFmha._multi_token
M3DrafterFullCtxFmha._run_fused_rope
M3DrafterFullCtxFmha._shape
M3DrafterFullCtxFmha.run_generation
fwd_has_no_mrope
install
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/m3_fullctx_decode.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
FullCtxPlan.run
emits_mxfp8
has_own_reducer
plan_for
prebuild
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/m3_sampled_accept.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
SampledAcceptWorkspace.__init__
SampledAcceptWorkspace.create
_ws_key
sampled_accept
workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/m3_shard_accept.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_num_sms
_workspace
create_workspace
shard_accept
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/m3_target_head.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_num_sms
head_supported
logits
prepare_head
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/m3_tokmap_guard.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_guarded
_indexed
_map_offset
_no_missed
_no_taken
_one_request_map
install
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/ops/attention/m3_fullctx_decode.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3FullCtxDecode.__call__
M3FullCtxDecode.is_valid
M3FullCtxDecode.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/ops/attention/m3_msa_attention.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3MsaAttention.__call__
M3MsaAttention.reference
_selected_blocks
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/ops/gemm/m3_mxfp8_pinned_gemm.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3Mxfp8PinnedGemm.__call__
M3Mxfp8PinnedGemm.reference
_runner
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/ops/gemm/m3_mxfp8_pinned_quantize.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3Mxfp8PinnedQuantize.__call__
M3Mxfp8PinnedQuantize.reference
_runner
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/ops/sampling/m31_chain_spec_mc.py`

库存函数 19；已覆盖 0；未覆盖 19。

```text
M31ChainSpecMc.__call__
M31ChainSpecMc.is_valid
M31ChainSpecMc.reference
_chain_spec_mc
_f3_taken
_rej_v2_taken
_window_for
chain_spec_mc
chain_spec_mc_op
chain_spec_mc_slots
chain_spec_mc_slots_op
ctas_per_row
defer_window
flush_window
kernel_version
prebuild
sm_count
window_ops_built
ws_bytes_per_row
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/ops/sampling/m31_softmax_mc.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
M31SoftmaxMc.__call__
M31SoftmaxMc.is_valid
M31SoftmaxMc.reference
max_rows
partials_numel
prebuild
softmax_mc
softmax_mc_op
write_ctas
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/ops/sampling/m31_topp_det.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
M31ToppDet.__call__
M31ToppDet.is_valid
M31ToppDet.reference
_load_op
ctas_per_row
is_v3
max_rows
prebuild
sm_count
top_p_renorm_det
topp_det_op
ws_bytes_per_row
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/ops/spec/m3_spec_offsets.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3SpecOffsets.__call__
M3SpecOffsets.is_valid
M3SpecOffsets.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/drafter_down_splitk.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
_load_provider
_module_admitted
_prepare_aot
_route
_route.<locals>.log_taken
_tactic_admitted
_wrap_forward
_wrap_forward.<locals>.forward
route_down_splitk
route_oproj_splitk
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/drafter_mxfp8.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
_draft_qkv_ring2_cfg
_projections
route_gu_swiglu
route_gu_swiglu.<locals>._apply_activation
route_gu_swiglu.<locals>.apply_linear
route_his_layout
route_mx_linears
route_mx_linears.<locals>.apply_linear
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/dspark_ctxlen.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_log_once
prepare
prepare_1st_drafter_inputs_ctxlen_fused
resume_ctx_len
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/dspark_markov_worker.py`

库存函数 31；已覆盖 0；未覆盖 31。

```text
S4WorkerMixin._apply_kv_rewind_after_draft
S4WorkerMixin._dhead8_all_ok
S4WorkerMixin._forward_impl
S4WorkerMixin._host_step_hook
S4WorkerMixin._p1_engages
S4WorkerMixin._p1_step
S4WorkerMixin._prebuild_dhead8
S4WorkerMixin._prebuild_prob1
S4WorkerMixin._refine_block_logits
S4WorkerMixin._s4_chain
S4WorkerMixin._s4_common
S4WorkerMixin._s4_log_count
S4WorkerMixin._s4_log_ll
S4WorkerMixin._s4_mode
S4WorkerMixin._s4_ranks_agree
S4WorkerMixin._s4_serves
S4WorkerMixin._s4l_log
S4WorkerMixin._store_block_draft_probs
S4WorkerMixin._tailbk_c_log
S4WorkerMixin._tailbk_rewind_args
S4WorkerMixin.advanced_sample_draft_block
S4WorkerMixin.prebuild_s4
S4WorkerMixin.sample_draft_tokens
_S4State.__init__
_S4State.buffer
_ShardedDraftLogits.__init__
_ShardedDraftLogits.forward
_biased_shard_logits
_head_into
_head_is_sharded
disabled
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/long_select.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
long_select_blocks
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_ctxlen_resume.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
_ck
_compile
_key
ctxlen_resume_
ctxlen_resume_eager_
outside_contract
prepare
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_draft_step.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
_draft_step_split
_fixed_order
_grid
_num_sms
_place_per_domain
draft_step
head_supported
prepare_head
workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_drafter_attn.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
M3DrafterFullCtxFmha.__init__
M3DrafterFullCtxFmha._fused_rope_applies
M3DrafterFullCtxFmha._is_supported
M3DrafterFullCtxFmha._multi_token
M3DrafterFullCtxFmha._run_fused_rope
M3DrafterFullCtxFmha._shape
M3DrafterFullCtxFmha.run_generation
M3DrafterSlidingFiFmha.is_environment_available
fwd_has_no_mrope
install
install_sliding_fi
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_fc_wait.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
fc_wait
fc_wait_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_force_vec.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
force_vec_kernel
force_vec_req_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_mc_allreduce.py`

库存函数 17；已覆盖 0；未覆盖 17。

```text
McAllReduceWorkspace.create
McAllReduceWorkspace.flags_for
McBuckets.__init__
McBuckets.for_tokens
_ar_opts
_bucket
_capture_args
_own_op_loaded
_tp_comm
capture_supported
decline_reason
get_workspace
m3_mc_allreduce_norm
m3_mc_finalize_allreduce_norm
own_op
tp_agree
tp_agree_built
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_sampled_accept.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
SampledAcceptWorkspace.__init__
SampledAcceptWorkspace.create
_ilp_walk
_op_args
one_round_k
sampled_accept
workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_shard_accept.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_num_sms
_workspace
create_workspace
shard_accept
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/msa_kv_update.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
_takes
msa_kv_update
msa_kv_update_tail
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/attention/msa_kv_update.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
MsaKvUpdate.__call__
MsaKvUpdate.is_valid
MsaKvUpdate.reference
msa_kv_update_tail
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/comm/m3_mc_allreduce_norm.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
M3McAllreduceNorm.__call__
M3McAllreduceNorm.is_valid
M3McAllreduceNorm.reference
boundary
mxfp8_rows
sf_written
swizzle_sf
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/comm/m3_mc_finalize_allreduce_norm.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
M3McFinalizeAllreduceNorm.__call__
M3McFinalizeAllreduceNorm.is_valid
M3McFinalizeAllreduceNorm.reference
finalize_partial
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/gemm/m3_dense_swiglu.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3DenseSwiglu.__call__
M3DenseSwiglu.is_valid
M3DenseSwiglu.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/gemm/m3_mxfp8_cublaslt.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
M3Mxfp8Cublaslt.__call__
M3Mxfp8Cublaslt.is_valid
M3Mxfp8Cublaslt.reference
_e4m3
_ue8m0
mxfp8_quantize
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/spec/m3_draft_silu.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3DraftSilu.__call__
M3DraftSilu.is_valid
M3DraftSilu.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/spec/m3_force_vec.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
M3ForceVec.__call__
M3ForceVec.reference
M3ForceVecReq.__call__
M3ForceVecReq.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/spec/m3_spec_offsets.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3SpecOffsets.__call__
M3SpecOffsets.is_valid
M3SpecOffsets.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/spec/m3_step_prologue.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3StepPrologue.__call__
M3StepPrologue.is_valid
M3StepPrologue.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_force_vec.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
force_vec_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/comm/m3_allreduce.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3AllReduce.__call__
M3AllReduce.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/comm/m3_mc_allreduce_norm.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
M3McAllreduceNorm.__call__
M3McAllreduceNorm.is_valid
M3McAllreduceNorm.reference
boundary
mxfp8_rows
sf_written
swizzle_sf
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/comm/m3_mc_finalize_allreduce_norm.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
M3McFinalizeAllreduceNorm.__call__
M3McFinalizeAllreduceNorm.is_valid
M3McFinalizeAllreduceNorm.reference
finalize_partial
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/long_select.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
long_select_blocks
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/m3_force_vec.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
force_vec_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/m3_router_gemv.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
m3_router_gemv
m3_router_gemv_fp4
m3_router_tc
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/comm/m3_mc_allreduce_norm.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3McAllreduceNorm.__call__
M3McAllreduceNorm.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/comm/m3_mc_finalize_allreduce_norm.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3McFinalizeAllreduceNorm.__call__
M3McFinalizeAllreduceNorm.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/long_select.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
long_select_blocks
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/comm/m3_mc_allreduce_norm.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3McAllreduceNorm.__call__
M3McAllreduceNorm.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/comm/m3_mc_finalize_allreduce_norm.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3McFinalizeAllreduceNorm.__call__
M3McFinalizeAllreduceNorm.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/long_select.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
long_select_blocks
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/m3_sampled_accept.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
SampledAcceptWorkspace.__init__
SampledAcceptWorkspace.create
sampled_accept
workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/m3_shard_accept.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_num_sms
_workspace
create_workspace
shard_accept
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/comm/m3_mc_allreduce_norm.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3McAllreduceNorm.__call__
M3McAllreduceNorm.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/comm/m3_mc_finalize_allreduce_norm.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3McFinalizeAllreduceNorm.__call__
M3McFinalizeAllreduceNorm.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/long_select.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
long_select_blocks
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_fc_wait.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
fc_wait
fc_wait_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_shard_accept.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
_num_sms
_workspace
create_workspace
shard_accept
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/kernels/index_k_producer.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
index_k_producer_cache
pack_nvfp4_index_k
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/msa_decode.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
M31MsaDecodeFmha.__init__
M31MsaDecodeFmha._is_available
M31MsaDecodeFmha._is_supported
M31MsaDecodeFmha._reserve_dense_workspace
M31MsaDecodeFmha._run_dense
M31MsaDecodeFmha._run_sparse
M31MsaDecodeFmha.prepare_workspace
M31MsaDecodeFmha.run_generation
dense_decode_max_seq_len
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/attention/m3_dense_qk_norm_rope_fp8.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3DenseQkNormRopeFp8.__call__
M3DenseQkNormRopeFp8.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/attention/m3_draft_attention.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3DraftAttention.__call__
M3DraftAttention.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/bcg_ptr_guard.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
StaticPointerGuard.__init__
StaticPointerGuard.armed
StaticPointerGuard.check
StaticPointerGuard.note
StaticPointerGuard.reset
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/comm/allgather_cast.py`

库存函数 24；已覆盖 0；未覆盖 24。

```text
AllgatherCast.__call__
AllgatherCast.is_valid
AllgatherCast.reference
AllgatherF32.__call__
AllgatherF32.is_valid
AllgatherF32.reference
_
_all_ranks_ok
_allgather_cast
_allgather_f32
_arg
_consts
_default_grid
_kernel_for
_kernel_module
configure_capacity
gather_last_dim_f32
is_compiled
pick_grid
precompile
precompile_f32
supports
supports_f32
workspace_args
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/comm/m31_allreduce.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3AllReduce.__call__
M3AllReduce.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/comm/m3_allreduce.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3AllReduce.__call__
M3AllReduce.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/comm/m3_mc_finalize_allreduce_norm.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
M3McFinalizeAllreduceNorm.__call__
M3McFinalizeAllreduceNorm.is_valid
M3McFinalizeAllreduceNorm.reference
finalize_partial
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/comm/mc_workspace.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
McAllReduceWorkspace.create
McAllReduceWorkspace.flags_for
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/comm/reference.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
McAllreduceNormReference.reference
McFinalizeAllreduceNormReference.reference
boundary
finalize_partial
mxfp8_rows
sf_written
swizzle_sf
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/drafter_sliding_fi.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3DrafterSlidingFiFmha.is_environment_available
install_sliding_fi
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/dspark_markov_chain.py`

库存函数 27；已覆盖 0；未覆盖 27。

```text
DsparkMarkovChain.__call__
DsparkMarkovChain.is_valid
DsparkMarkovChain.reference
_
_accepted_inputs
_arg
_dspark_markov_chain
_dspark_markov_chain_rewind
_fits_all
_kernel_module
_launch
_sm_count
cached_inputs_taken
dspark_markov_chain_rewind
enable_batch1_variant
enable_cached_inputs
enable_fp32_sum
enable_ll_exchange
ll_exchange_engaged
pad_logits_buffer
pad_w2_shard
padded_rows
pick_grid
precompile
supports
supports_rewind
workspace_args
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/gemm/m3_draft_lowm.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
M3DraftLowM.__call__
M3DraftLowM.is_valid
M3DraftLowM.reference
tactic_for
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/gemm/mxfp8_state.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
_Mxfp8GemmState.__init__
_mx_mode
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/gqamix_sparse_decode.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
_check_nvfp4_inputs
_dequant_nvfp4_rows
_e2m1x2_to_f16x2
_gqa_sparse_decode_kernel
_gqa_sparse_decode_pair_kernel
_merge_topk_attn_out_kernel
_pdl_enabled
_sm100f_nvfp4_launch_options
_sm100f_nvfp4_merge_launch_options
_sm100f_nvfp4_num_topk_chunks
_sm100f_nvfp4_query_group_size
_sm100f_nvfp4_use_linear_softmax
minimax_m3_sparse_attn_decode
resolve_num_topk_chunks
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/moe/m3_router_gemv_fp4.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3RouterGemvFp4.__call__
M3RouterGemvFp4.is_valid
M3RouterGemvFp4.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/moe/megamoe_local/impl_delta.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
SharedExpertPlanMixin._init_shared_expert_plan
SharedExpertPlanMixin.default_shared_corun_band
SharedExpertPlanMixin.megamoe_launch_bucket
SharedExpertPlanMixin.shared_corun_band
SharedExpertPlanMixin.shared_corun_pin
SharedExpertPlanMixin.shared_corun_pin_kernel
SharedExpertPlanMixin.shared_expert_serial
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/moe/megamoe_local/local_plan.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
_non_negative_int
lever_enabled
lever_environ
megamoe_local_requested
shared_corun_band
shared_corun_pin
shared_corun_pin_above_rows
shared_corun_pin_kernel
shared_expert_serial
shared_serial_max_rows
shared_serial_min_ep_size
vd_static_rejection
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/spec/accept_workspace.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
create_accept_workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/spec/m3_draft_silu.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3DraftSilu.__call__
M3DraftSilu.is_valid
M3DraftSilu.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/spec/m3_draft_step.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
M3DraftStep.__call__
M3DraftStep.is_valid
M3DraftStep.reference
bf16_fragments
from_fragments
head_weight
max_tile_pieces
shard_logits
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/spec/m3_force_vec_req.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3ForceVecReq.__call__
M3ForceVecReq.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/spec/m3_sampled_accept.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
M3SampledAccept.__call__
M3SampledAccept.is_valid
M3SampledAccept.reference
M3SampledAccept.reference.<locals>.accept
nucleus_probs
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/spec/m3_sampled_accept_ws_words.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
M3SampledAcceptWsWords.__call__
M3SampledAcceptWsWords.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/spec/m3_step_prologue.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
M3StepPrologue.__call__
M3StepPrologue.is_valid
M3StepPrologue.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/swiglu_rows_plan.py`

库存函数 17；已覆盖 0；未覆盖 17。

```text
bytes_moved
cta_batches
cta_pad_rows
cta_rows
ctas_per_sm
env_word
grid
k_rows_from_env
loads_in_flight_bytes
ms_per_1k
one_row_cta_rows
one_row_grid
padded_rows
switch_set
switch_warns
threads
uses_rows_form
```

### `tensorrt_llm/_torch/models/checkpoints/hf/cosmos3_weight_mapper.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
Cosmos3HfWeightMapper.__init__
Cosmos3HfWeightMapper.preprocess_weights
Cosmos3HfWeightMapper.should_drop_checkpoint_key
```

### `tensorrt_llm/_torch/models/checkpoints/hf/exaone4_5_weight_mapper.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
Exaone4_5HfWeightMapper.preprocess_weights
```

### `tensorrt_llm/_torch/models/checkpoints/hf/llava_next_weight_mapper.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
LlavaNextHfWeightMapper.preprocess_weights
```

### `tensorrt_llm/_torch/models/checkpoints/hf/mixtral_weight_mapper.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
MixtralHfWeightMapper.apply_callbacks
MixtralHfWeightMapper.map_weights
```

### `tensorrt_llm/_torch/models/checkpoints/hf/qwen2vl_weight_mapper.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
Qwen2VLHfWeightMapper.preprocess_weights
```

### `tensorrt_llm/_torch/models/checkpoints/hf/qwen3vl_weight_mapper.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
Qwen3VLHfWeightMapper.preprocess_weights
```

### `tensorrt_llm/_torch/models/modeling_hunyuan_dense.py`

库存函数 17；已覆盖 0；未覆盖 17。

```text
HunYuanAttention.__init__
HunYuanAttention.apply_qk_norm
HunYuanAttention.apply_qk_norm.<locals>.k_l2norm
HunYuanAttention.apply_qk_norm.<locals>.q_l2norm
HunYuanAttention.apply_rope
HunYuanAttention.forward
HunYuanDecoderLayer.__init__
HunYuanDecoderLayer.forward
HunYuanDenseV1ForCausalLM.__init__
HunYuanDenseV1ForCausalLM.forward
HunYuanDenseV1ForCausalLM.load_weights
HunYuanDenseV1ForCausalLM.load_weights.<locals>.filter_weights
HunYuanModel.__init__
HunYuanModel.forward
HunYuanPretrainedConfig.__init__
HunYuanPretrainedConfig._rope_scaling_validation
HunYuanTRTRotaryEmbedding.__init__
```

### `tensorrt_llm/_torch/models/modeling_hyperclovax.py`

库存函数 41；已覆盖 0；未覆盖 41。

```text
HCXVisionCAbstractor.__init__
HCXVisionCAbstractor._forward
HCXVisionCAbstractor._forward_adaptive_num_query
HCXVisionCAbstractor.build_mlp
HCXVisionCAbstractor.build_net
HCXVisionCAbstractor.forward
HCXVisionForCausalLM.__init__
HCXVisionForCausalLM.forward
HCXVisionForCausalLM.infer_max_seq_len
HCXVisionForCausalLM.load_weights
HCXVisionForCausalLM.post_config
HCXVisionForCausalLM.vocab_size_padded
HCXVisionInputProcessor.__init__
HCXVisionInputProcessor._create_vision_query_generator
HCXVisionInputProcessor._post_process
HCXVisionInputProcessor._preprocess
HCXVisionInputProcessor._reset_vision_query_generator
HCXVisionInputProcessor.call_with_text_prompt
HCXVisionInputProcessor.config
HCXVisionInputProcessor.dtype
HCXVisionInputProcessor.get_mm_token_ids
HCXVisionInputProcessor.get_num_tokens_per_image
HCXVisionInputProcessor.get_vocab_size
HCXVisionInputProcessor.model_path
HCXVisionInputProcessor.processor
HCXVisionInputProcessor.tokenizer
HCXVisionModel.__init__
HCXVisionModel._init_possible_resolutions
HCXVisionModel._parse_and_batch_multimodal_data
HCXVisionModel.forward
HCXVisionModel.load_weights
HCXVisionModel.post_config
_filter_weights
adaptive_anyres_postprocessing
anyres_postprocessing
compute_adaptive_params
determine_non_vision_query_lengths
get_anyres_image_grid_shape
reshape_and_unpad_image_features
select_best_resolution
unpad_image
```

### `tensorrt_llm/_torch/models/modeling_mllama.py`

库存函数 11；已覆盖 0；未覆盖 11。

```text
MllamaDecoderLayer.__init__
MllamaDecoderLayer.forward
MllamaForCausalLM.__init__
MllamaForCausalLM.forward
MllamaForCausalLM.infer_max_seq_len
MllamaForConditionalGeneration.__init__
MllamaForConditionalGeneration.forward
MllamaForConditionalGeneration.infer_max_seq_len
MllamaForConditionalGeneration.load_weights
MllamaTextModel.__init__
MllamaTextModel.forward
```

### `tensorrt_llm/_torch/models/modeling_nemotron.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
NemotronAttention.__init__
NemotronDecoderLayer.__init__
NemotronDecoderLayer.forward
NemotronForCausalLM.__init__
NemotronLayerNormPlus1.__init__
NemotronLayerNormPlus1.forward
NemotronLayerNormPlus1.reset_parameters
NemotronModel.__init__
NemotronModel.forward
```

### `tensorrt_llm/_torch/models/modeling_starcoder2.py`

库存函数 8；已覆盖 0；未覆盖 8。

```text
Starcoder2Attention.__init__
Starcoder2Attention.forward
Starcoder2DecoderLayer.__init__
Starcoder2DecoderLayer.forward
Starcoder2ForCausalLM.__init__
Starcoder2ForCausalLM.load_weights
Starcoder2Model.__init__
Starcoder2Model.forward
```

### `tensorrt_llm/_torch/models/modeling_vila.py`

库存函数 54；已覆盖 0；未覆盖 54。

```text
Conversation.append_message
Conversation.copy
Conversation.get_prompt
DownSample2x2BlockFix.forward
DownSample3x3BlockFix.forward
DownSampleBlock.flat_square
DownSampleBlock.forward
IdentityMap.__init__
IdentityMap.config
IdentityMap.forward
SimpleResBlock.__init__
SimpleResBlock.forward
VilaInputProcessor.__init__
VilaInputProcessor._postprocess
VilaInputProcessor._preprocess
VilaInputProcessor._process
VilaInputProcessor.call_with_text_prompt
VilaInputProcessor.config
VilaInputProcessor.dtype
VilaInputProcessor.get_mm_token_ids
VilaInputProcessor.model_path
VilaInputProcessor.processor
VilaInputProcessor.tokenizer
VilaModel.__init__
VilaModel.forward
VilaModel.get_llm
VilaModel.infer_max_seq_len
VilaModel.llm_checkpoint_dir
VilaModel.load_weights
VilaModel.mm_token_ids
VilaModel.post_config
VilaModel.vocab_size_padded
VilaMultimodalProjector.__init__
VilaMultimodalProjector.forward
VilaMultimodalProjectorConfig.__init__
_apply_chat_template
_convert_dtype
_get_conversation_mode
_get_model_paths
_resize_embeds
_resize_lm_head
_resize_token_embeddings
_validate_vila_mm_alignment
flat_square_2x2
flat_square_3x3
infer_stop_tokens
infer_stop_tokens.<locals>._maybe_add_sentinel_token
init_llm
init_mm_projector
init_tokenizer
init_vision_tower
process_image
process_image.<locals>.expand2square
process_images
```

### `tensorrt_llm/_torch/modules/fla/op.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
gather
gdc_wait_with_memory_clobber
make_tensor_descriptor
safe_exp
```

### `tensorrt_llm/_torch/modules/kimi_k3_attn_res/_attn_res_kernels.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_default_get_sm_version
get_attn_res_sm_version
intree_attn_res_fwd
is_attn_res_optimized_supported
is_intree_attn_res_available
```

### `tensorrt_llm/_torch/modules/mamba/softplus.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
softplus
```

### `tensorrt_llm/_torch/moe/fused_moe/fused_moe_vanilla.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
VanillaMoE.__init__
VanillaMoE.can_implement
VanillaMoE.create_experts
VanillaMoE.create_weights
VanillaMoE.forward
VanillaMoE.load_weights
VanillaMoE.pack_params
VanillaMoE.reducescatter_or_allreduce
VanillaMoE.run_experts
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain/token_comm_cta_mainloop.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
TokenCommCtaMainloop.__init__
TokenCommCtaMainloop.register_smem_regions
TokenCommCtaMainloop.register_token_back_smem
TokenCommCtaMainloop.register_token_in_smem
TokenCommCtaMainloop.run
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain/token_comm_dedup.py`

库存函数 115；已覆盖 0；未覆盖 115。

```text
TokenCommDedup.__extract_mlir_values__
TokenCommDedup.__init__
TokenCommDedup.__new_from_mlir_values__
TokenCommDedup._issue_fill
TokenCommDedup._reduce_and_push
TokenCommDedup._register_main_workspace
TokenCommDedup._token_back_available_bytes_per_warp
TokenCommDedup._validate_configuration
TokenCommDedup.activation_dtype
TokenCommDedup.activation_sf_dtype
TokenCommDedup.activation_sf_hidden_padded
TokenCommDedup.activation_sf_vector_size
TokenCommDedup.assign_device_members
TokenCommDedup.bytes_per_token
TokenCommDedup.combine_sf_hidden_padded
TokenCommDedup.expert_count_padded
TokenCommDedup.expert_count_with_trash
TokenCommDedup.experts_per_rank
TokenCommDedup.fc1_activation_sf_tensor
TokenCommDedup.fc1_activation_tensor
TokenCommDedup.fc1_ready_counter_pointer
TokenCommDedup.fc1_topk_scores_tensor
TokenCommDedup.fc2_activation_sf_tensor
TokenCommDedup.fc2_activation_tensor
TokenCommDedup.fc2_done_counter_tensor
TokenCommDedup.hybrid_token_back
TokenCommDedup.impl_desc_require
TokenCommDedup.launch_router
TokenCommDedup.local_expert_sizes
TokenCommDedup.max_fc1_ready_slot_count
TokenCommDedup.metadata_ready_target
TokenCommDedup.next_token
TokenCommDedup.order_block_bytes
TokenCommDedup.order_chunk_bytes
TokenCommDedup.order_chunks_per_row
TokenCommDedup.output_token_bytes
TokenCommDedup.pre_reduced_activation_sf_tensor
TokenCommDedup.pre_reduced_activation_tensor
TokenCommDedup.problem_desc_require
TokenCommDedup.register_device_workspace
TokenCommDedup.register_smem_regions
TokenCommDedup.remove_device_members
TokenCommDedup.reset_tail
TokenCommDedup.router_data_cta_count
TokenCommDedup.router_elements_per_lane
TokenCommDedup.router_grid_cta_count
TokenCommDedup.router_push_cta_count
TokenCommDedup.router_smem_workspace
TokenCommDedup.router_tokens_per_cta
TokenCommDedup.router_warps_per_cta
TokenCommDedup.token_back
TokenCommDedup.token_back_chunk_bytes
TokenCommDedup.token_back_enabled
TokenCommDedup.token_back_push_data
TokenCommDedup.token_back_push_sf
TokenCommDedup.token_back_stage_slots
TokenCommDedup.token_back_worker_groups
TokenCommDedup.token_in
TokenCommDedup.token_src_metadata_tensor
TokenCommDedup.valid_rank_mask_tensor
TokenCommDedup.wait_for_sizes_ready
TokenCommDedup.worst_case_sf_token_count
TokenCommDedup.worst_case_token_count
_CopyInfo.pack
_CopyInfo.unpack
_DedupMetadataPushRouter.__extract_mlir_values__
_DedupMetadataPushRouter.__init__
_DedupMetadataPushRouter.__new_from_mlir_values__
_DedupMetadataPushRouter._broadcast_sizes_to_peers
_DedupMetadataPushRouter._broadcast_sizes_to_peers.<locals>.finalize
_DedupMetadataPushRouter._build_histogram
_DedupMetadataPushRouter._build_router_smem_workspace
_DedupMetadataPushRouter._compute_push_tables
_DedupMetadataPushRouter._dump_contiguous_router_output
_DedupMetadataPushRouter._dump_router_output_by_expert
_DedupMetadataPushRouter._load_router_inputs
_DedupMetadataPushRouter._register_router_workspace
_DedupMetadataPushRouter._route_metadata
_DedupMetadataPushRouter._router_helper_multiple_ctas
_DedupMetadataPushRouter._router_helper_single_cta
_DedupMetadataPushRouter._router_kernel
_DedupMetadataPushRouter._router_launch_configuration
_DedupMetadataPushRouter._router_launch_configuration.<locals>.next_power_of_two
_DedupMetadataPushRouter._router_multiple_ctas
_DedupMetadataPushRouter._router_push_metadata
_DedupMetadataPushRouter._router_single_cta
_DedupMetadataPushRouter._router_slot_swizzle
_DedupMetadataPushRouter._router_value_coordinate
_DedupMetadataPushRouter._sort_token_routes_and_scatter
_DedupMetadataPushRouter._validate_router_configuration
_DedupMetadataPushRouter.experts_per_rank
_DedupMetadataPushRouter.fc1_topk_scores_tensor
_DedupMetadataPushRouter.impl_desc_require
_DedupMetadataPushRouter.launch_router
_DedupMetadataPushRouter.local_expert_sizes
_DedupMetadataPushRouter.metadata_ready_target
_DedupMetadataPushRouter.pool_expert_base_tensor
_DedupMetadataPushRouter.problem_desc_require
_DedupMetadataPushRouter.register_device_workspace
_DedupMetadataPushRouter.router_max_routes_per_cta
_DedupMetadataPushRouter.router_smem_workspace
_DedupMetadataPushRouter.sizes_tensor
_DedupMetadataPushRouter.token_src_metadata_pointer
_DedupMetadataPushRouter.token_src_metadata_tensor
_DedupMetadataPushRouter.valid_rank_mask_tensor
_DedupMetadataPushRouter.wait_for_metadata_ready
_DedupMetadataPushRouter.wait_for_sizes_ready
_DedupMetadataPushRouter.worst_case_padded_tokens
_SortedElement.from_packed
_SortedElement.pack
_SortedElement.storage_type
_copy_atom
_odd_even_merge_sort_pairs
_odd_even_merge_sort_pairs.<locals>.merge
_odd_even_merge_sort_pairs.<locals>.sort
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/fc2_ready_geometry.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
derive_fc2_ready_geometry
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/helper_weight_ready_gate.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
HelperWeightReadyGenerationGate.__extract_mlir_values__
HelperWeightReadyGenerationGate.__init__
HelperWeightReadyGenerationGate.__new_from_mlir_values__
HelperWeightReadyGenerationGate.__new_from_mlir_values__.<locals>.rebuild
HelperWeightReadyGenerationGate._sweep_terminal_flags
HelperWeightReadyGenerationGate._typed_terminal_flags
HelperWeightReadyGenerationGate._wait_all
HelperWeightReadyGenerationGate.wait_before_first_helper_publish
_helper_weight_ready_fail_closed_trap
_validate_helper_weight_ready_binding
derive_helper_weight_ready_topology
make_helper_weight_ready_generation_gate
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/iket_compat.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
_IketShim.mark
_IketShim.range_end
_IketShim.range_pop
_IketShim.range_push
_IketShim.range_start
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/megamoe_aot.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
make_megamoe_aot_callable
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/local_mega/block_scaled_swap_ab_fc12_epilogue.py`

库存函数 12；已覆盖 0；未覆盖 12。

```text
Fc2OutputRouter.data_output
Fc2OutputRouter.get_data_dst
Fc2OutputRouter.prefetch
SwapABFc2Epilogue.__extract_mlir_values__
SwapABFc2Epilogue.__init__
SwapABFc2Epilogue.__new_from_mlir_values__
SwapABFc2Epilogue._make_output_router
SwapABFc2Epilogue.advance_fc1_flag_phase
SwapABGatedActEpilogue.__init__
SwapABGatedActEpilogue.impl_desc_require
SwapABGatedActEpilogue.prepare_tma_store_params
SwapABGatedActEpilogue.run
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/local_mega/block_scaled_swap_ab_fc12_mainloop.py`

库存函数 20；已覆盖 0；未覆盖 20。

```text
BlockScaledSwapAbFc12Mainloop.__init__
BlockScaledSwapAbFc12Mainloop._s2t_copy_and_partition
BlockScaledSwapAbFc12Mainloop._select_ab_stage_plan
BlockScaledSwapAbFc12Mainloop._select_ab_stage_plan.<locals>.packed_bytes
BlockScaledSwapAbFc12Mainloop._validate_configuration
BlockScaledSwapAbFc12Mainloop.advance_pipeline_state
BlockScaledSwapAbFc12Mainloop.assign_device_members
BlockScaledSwapAbFc12Mainloop.create_ab_pipeline
BlockScaledSwapAbFc12Mainloop.create_acc_pipeline
BlockScaledSwapAbFc12Mainloop.create_tmem_allocator
BlockScaledSwapAbFc12Mainloop.impl_desc_require
BlockScaledSwapAbFc12Mainloop.make_tiled_mma
BlockScaledSwapAbFc12Mainloop.materialize_codegen_members
BlockScaledSwapAbFc12Mainloop.prepare_tma_load_params
BlockScaledSwapAbFc12Mainloop.problem_desc_require
BlockScaledSwapAbFc12Mainloop.register_smem_regions
BlockScaledSwapAbFc12Mainloop.run_mma
BlockScaledSwapAbFc12Mainloop.run_tma_a
BlockScaledSwapAbFc12Mainloop.run_tma_b
BlockScaledSwapAbFc12Mainloop.run_tma_gather_b1
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/local_mega/block_scaled_swap_ab_local_mega_moe_kernel.py`

库存函数 19；已覆盖 0；未覆盖 19。

```text
BlockScaledSwapAbLocalMegaMoeKernel.__call__
BlockScaledSwapAbLocalMegaMoeKernel.__call__.<locals>.rewrite_tensor_shape
BlockScaledSwapAbLocalMegaMoeKernel.__init__
BlockScaledSwapAbLocalMegaMoeKernel._aot_symbol_name
BlockScaledSwapAbLocalMegaMoeKernel._build_device_workspace
BlockScaledSwapAbLocalMegaMoeKernel._build_mainloop_and_smem
BlockScaledSwapAbLocalMegaMoeKernel._kernel
BlockScaledSwapAbLocalMegaMoeKernel._resolve_schedule_hint
BlockScaledSwapAbLocalMegaMoeKernel._validate_geometry
BlockScaledSwapAbLocalMegaMoeKernel._validate_helper_configuration
BlockScaledSwapAbLocalMegaMoeKernel.aot_compile
BlockScaledSwapAbLocalMegaMoeKernel.aot_compile.<locals>.fake_tensor
BlockScaledSwapAbLocalMegaMoeKernel.get_workspace_sizes
BlockScaledSwapAbLocalMegaMoeKernel.impl_desc_require
BlockScaledSwapAbLocalMegaMoeKernel.load_compiled
BlockScaledSwapAbLocalMegaMoeKernel.name
BlockScaledSwapAbLocalMegaMoeKernel.name.<locals>.dtype_name
BlockScaledSwapAbLocalMegaMoeKernel.problem_desc_require
BlockScaledSwapAbLocalMegaMoeKernel.require_zero_workspace_leading_bytes
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/local_mega/local_routing.py`

库存函数 77；已覆盖 0；未覆盖 77。

```text
LocalRouting.__extract_mlir_values__
LocalRouting.__init__
LocalRouting.__new_from_mlir_values__
LocalRouting._register_main_workspace
LocalRouting._validate_configuration
LocalRouting.activation_dtype
LocalRouting.activation_sf_dtype
LocalRouting.activation_sf_hidden_padded
LocalRouting.activation_sf_vector_size
LocalRouting.assign_device_members
LocalRouting.expert_sizes
LocalRouting.fc1_activation_sf_tensor
LocalRouting.fc1_ready_counter_pointer
LocalRouting.fc1_topk_scores_tensor
LocalRouting.impl_desc_require
LocalRouting.launch_router
LocalRouting.max_fc1_ready_slot_count
LocalRouting.pre_reduced_activation_tensor
LocalRouting.problem_desc_require
LocalRouting.register_device_workspace
LocalRouting.register_smem_regions
LocalRouting.remove_device_members
LocalRouting.reset_tail
LocalRouting.token_in
LocalRouting.token_src_metadata_region
LocalRouting.token_src_metadata_tensor
LocalRouting.wait_for_sizes_ready
LocalRouting.worst_case_sf_token_count
LocalRouting.worst_case_token_count
LocalRoutingArgs.__extract_mlir_values__
LocalRoutingArgs.__new_from_mlir_values__
RoutingMetadata._pointer
RoutingMetadata.load
RoutingMetadata.pack
_MetadataPushRouter.__extract_mlir_values__
_MetadataPushRouter.__init__
_MetadataPushRouter.__new_from_mlir_values__
_MetadataPushRouter._build_histogram
_MetadataPushRouter._build_router_smem_workspace
_MetadataPushRouter._compute_local_tables
_MetadataPushRouter._dump_contiguous_router_output
_MetadataPushRouter._dump_router_output_by_expert
_MetadataPushRouter._load_router_inputs
_MetadataPushRouter._publish_sizes
_MetadataPushRouter._publish_sizes.<locals>.finalize
_MetadataPushRouter._register_router_workspace
_MetadataPushRouter._router_helper_multiple_ctas
_MetadataPushRouter._router_helper_single_cta
_MetadataPushRouter._router_kernel
_MetadataPushRouter._router_launch_configuration
_MetadataPushRouter._router_launch_configuration.<locals>.next_power_of_two
_MetadataPushRouter._router_multiple_ctas
_MetadataPushRouter._router_push_metadata
_MetadataPushRouter._router_single_cta
_MetadataPushRouter._router_value_coordinate
_MetadataPushRouter._sort_router_elements
_MetadataPushRouter._validate_router_configuration
_MetadataPushRouter.expert_sizes
_MetadataPushRouter.fc1_topk_scores_tensor
_MetadataPushRouter.impl_desc_require
_MetadataPushRouter.launch_router
_MetadataPushRouter.metadata_ready_target
_MetadataPushRouter.pool_expert_base_tensor
_MetadataPushRouter.problem_desc_require
_MetadataPushRouter.register_device_workspace
_MetadataPushRouter.router_smem_workspace
_MetadataPushRouter.sizes_tensor
_MetadataPushRouter.token_src_metadata_pointer
_MetadataPushRouter.token_src_metadata_tensor
_MetadataPushRouter.wait_for_metadata_ready
_MetadataPushRouter.wait_for_sizes_ready
_MetadataPushRouter.worst_case_padded_tokens
_MetadataPushRouter.worst_case_token_count
_SortedElement.from_packed
_SortedElement.pack
_copy_atom
_smem_exclusive_prefix
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_epilogue_dedup.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
Fc2DedupBulkRouter.get_data_dst
Fc2DedupBulkRouter.prefetch
SwapABFc2EpilogueDedup.__call__
SwapABFc2EpilogueDedup.__init__
SwapABFc2EpilogueDedup._make_bulk_router
SwapABGatedActEpilogueDedup.__init__
SwapABGatedActEpilogueDedup._make_fc2_epilogue
SwapABGatedActEpilogueDedup.impl_desc_require
SwapABGatedActEpilogueDedup.problem_desc_require
fc2_dedup_bulk_store_function
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/topk_reduce_dedup.py`

库存函数 3；已覆盖 0；未覆盖 3。

```text
TopkReduceDedup.__call__
TopkReduceDedup.__init__
TopkReduceDedup._reduce_bf16
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/clc_mixed_cga.py`

库存函数 10；已覆盖 0；未覆盖 10。

```text
ClcMixedCgaSchedulerWorker.__extract_mlir_values__
ClcMixedCgaSchedulerWorker.__init__
ClcMixedCgaSchedulerWorker.__new_from_mlir_values__
ClcMixedCgaSchedulerWorker.__new_from_mlir_values__.<locals>.rebuild
ClcMixedCgaSchedulerWorker.assign_device_members
ClcMixedCgaSchedulerWorker.claim_next_work
ClcMixedCgaSchedulerWorker.initialize_fallback_group
ClcMixedCgaSchedulerWorker.produce_tail
ClcMixedCgaSchedulerWorker.register_device_workspace
ClcMixedCgaSchedulerWorker.register_smem_regions
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/genphase_decode_scheduler.py`

库存函数 17；已覆盖 0；未覆盖 17。

```text
GenphaseDecodeFc12Scheduler.__extract_mlir_values__
GenphaseDecodeFc12Scheduler.__init__
GenphaseDecodeFc12Scheduler.__new_from_mlir_values__
GenphaseDecodeFc12Scheduler._validate
GenphaseDecodeFc12Scheduler.assign_device_members
GenphaseDecodeFc12Scheduler.compute_stream_token_blocks
GenphaseDecodeFc12Scheduler.finish_claim_channel
GenphaseDecodeFc12Scheduler.gen_next_work
GenphaseDecodeFc12Scheduler.get_grid_shape
GenphaseDecodeFc12Scheduler.impl_desc_require
GenphaseDecodeFc12Scheduler.problem_desc_require
GenphaseDecodeFc12Scheduler.register_device_workspace
GenphaseDecodeFc12Scheduler.register_smem_regions
_count_stream_token_blocks
_decode_lead
_lead_slot_owner
_lead_slot_token_blocks
```

### `tensorrt_llm/_torch/moe/kernels/megamoe_scheduler/geometry.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
hierarchy_group_sizes
```

### `tensorrt_llm/_torch/moe/kernels/megamoe_scheduler/sami/_util.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
Comm.__init__
Comm.allgather
Comm.barrier
Comm.bcast
check_cuda
```

### `tensorrt_llm/_torch/moe/kernels/megamoe_scheduler/sami/arena.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
BoundLiveWeightArena.local_slot_count
LiveWeightArena._reject_overlaps
LiveWeightArena._reject_terminal_overlap
LiveWeightArena.bind
LiveWeightArenaProvider.build_live_arena
_TensorIdentity.capture
_TensorIdentity.metadata
_device_index
_dtype
_element_size
_exact_positive_int
_pointer
_shape
_stride
```

### `tensorrt_llm/_torch/peft/prompt_adapter.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
PromptAdapterManager.__init__
PromptAdapterManager._generate_uid
PromptAdapterManager.load_from_ckpt
PromptAdapterManager.uid_to_weights
```

### `tensorrt_llm/_torch/pyexecutor/connectors/mooncake_store/metadata.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
MooncakeStoreMetadata.__bool__
```

### `tensorrt_llm/_torch/pyexecutor/engine/runners/interface.py`

库存函数 4；已覆盖 0；未覆盖 4。

```text
ModelRunner.release_graphs
ModelRunner.wait_for_input_copy
ScheduledModelRunner.forward
ScheduledModelRunner.warmup
```

### `tensorrt_llm/_torch/speculative/auto_heuristic.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
suggest_spec_config
```

### `tensorrt_llm/_torch/speculative/group_sharded_sampling.py`

库存函数 14；已覆盖 0；未覆盖 14。

```text
GroupShardedVerify.__init__
GroupShardedVerify.buffer_layout
GroupShardedVerify.create
GroupShardedVerify.logits
GroupShardedVerify.sample
GroupShardedVerify.supports
_DeviceArray.__init__
_collect_kernel
_gather_kernel
_row_uniform
_select_kernel
_slice_stats_kernel
_wait_kernel
_wrap
```

### `tensorrt_llm/_torch/speculative/sa_enhancer.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
SADraftEnhancer.__init__
SADraftEnhancer._ensure_buffers
SADraftEnhancer._ensure_stream
SADraftEnhancer.extend_and_prepare
SADraftEnhancer.maybe_override_all_draft_tokens
```

### `tensorrt_llm/_torch/visual_gen/cache/base.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
CacheAccelerator.get_stats
CacheAccelerator.is_enabled
CacheAccelerator.refresh
CacheAccelerator.unwrap
CacheAccelerator.wrap
```

### `tensorrt_llm/_torch/visual_gen/kernels/blackwell/conv/dense_implicit_gemm_fprop.py`

库存函数 19；已覆盖 0；未覆盖 19。

```text
PersistentConvKernel.__call__
PersistentConvKernel.__init__
PersistentConvKernel._setup_attributes
PersistentConvKernel._setup_conv_input_attrs
PersistentConvKernel._setup_conv_tma
PersistentConvKernel._setup_conv_tma.<locals>.add_dummy_batch_dimension
PersistentConvKernel.can_implement
PersistentConvKernel.cluster_specific_kernel
PersistentConvKernel.kernel
_check_swizzle_size
_check_tensor_alignment
_check_tensor_alignment.<locals>.check_contiguous_16B_alignment
_parse_comma_separated_ints
compile_conv
compute_zpq
create_cute_tensor
prepare_tensors
run
run.<locals>.generate_tensors
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/ltx2_core/audio_vae/attention.py`

库存函数 2；已覆盖 0；未覆盖 2。

```text
AttnBlock.__init__
AttnBlock.forward
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/ltx2_core/perturbations.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
BatchedPerturbationConfig.all_in_batch
BatchedPerturbationConfig.any_in_batch
BatchedPerturbationConfig.mask
BatchedPerturbationConfig.mask_like
Perturbation.is_perturbed
PerturbationConfig.is_perturbed
build_stg_perturbation_config
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/ltx2_core/protocols.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
DiffusionStepProtocol.step
Patchifier.get_patch_grid_bounds
Patchifier.patch_size
Patchifier.patchify
Patchifier.unpatchify
SchedulerProtocol.execute
```

### `tensorrt_llm/_torch/visual_gen/models/ltx23/checkpoint.py`

库存函数 1；已覆盖 0；未覆盖 1。

```text
resolve_ltx23_checkpoint
```

### `tensorrt_llm/bench/dataset/prepare_real_data.py`

库存函数 9；已覆盖 0；未覆盖 9。

```text
DatasetConfig.check_prompt
DatasetConfig.get_images
DatasetConfig.get_input
DatasetConfig.get_output
DatasetConfig.get_prompt
DatasetConfig.query
load_dataset_from_hf
real_dataset
validate_output_len_dist
```

### `tensorrt_llm/scaffolding/trace_replay/execution_trace.py`

库存函数 7；已覆盖 0；未覆盖 7。

```text
ExecutionTrace.load
ExecutionTrace.save
_compact_trace_event_dict
_full_trace_event_dict
_parse_event
_strip_keys
_strip_none
```

### `tensorrt_llm/scaffolding/trace_replay/replay.py`

库存函数 35；已覆盖 0；未覆盖 35。

```text
DropPathStats.__init__
DropPathStats.next_drop_index
DropPathStats.record
DropPathStats.records
QueueExecutor.__init__
QueueExecutor._emit_retention_probes
QueueExecutor._handle_drop_kv_cache
QueueExecutor._handle_message
QueueExecutor._handle_tool_call
QueueExecutor._run
QueueExecutor._store_segment
QueueManager.__init__
QueueManager.allocate_queue
QueueManager.close_queue
QueueManager.drain_branch
QueueManager.get_queue
QueueManager.unregister_queue
QueueManager.wait_all_done
ReplayEngine.__init__
ReplayEngine.launch_trace
ReplayGenerationStats.__init__
ReplayGenerationStats.entries
ReplayGenerationStats.record_assistant
ReplayGenerationStats.record_parallel_region
ReplayGenerationStats.sum_replay_output_tokens
ReplayGenerationStats.sum_trace_completion_tokens
RetentionProbeStats.__init__
RetentionProbeStats.pending_probes
RetentionProbeStats.record
RetentionProbeStats.record_pending
RetentionProbeStats.records
_default_rng_factory
_generate_random_token_ids
make_seeded_rng_factory
make_seeded_rng_factory.<locals>.factory
```

### `tensorrt_llm/serve/openai_service.py`

库存函数 5；已覆盖 0；未覆盖 5。

```text
OpenAIService.is_ready
OpenAIService.openai_chat_completion
OpenAIService.openai_completion
OpenAIService.setup
OpenAIService.teardown
```

### `tensorrt_llm/serve/scripts/benchmark_utils.py`

库存函数 6；已覆盖 0；未覆盖 6。

```text
InfEncoder.clear_inf
InfEncoder.iterencode
convert_to_pytorch_benchmark_format
download_and_cache_file
is_file_valid_json
write_to_json
```

## 附录 D：部分函数已覆盖文件中的未覆盖函数（完整清单）

共 **1,543 个文件、12,712 个未覆盖函数**。各代码块只列未覆盖的 `qualname`；与对应文件路径组合后得到唯一函数键。

### `examples/apps/chat.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
main
```

### `examples/apps/fastapi_server.py`

库存函数 9；已覆盖 5；未覆盖 4。

```text
LlmServer.__call__
LlmServer.__init__.<locals>.lifespan
LlmServer.stats
entrypoint
```

### `examples/configs/database/database.py`

库存函数 14；已覆盖 10；未覆盖 4。

```text
CuratedRecipe._validate_config_path
CuratedRecipeList.__iter__
CuratedRecipeList.__len__
CuratedRecipeList.from_yaml
```

### `examples/disaggregated/slurm/benchmark/submit.py`

库存函数 25；已覆盖 1；未覆盖 24。

```text
_parse_positive_concurrency
allocate_gpus
allocate_gpus.<locals>.assign_server_compact
allocate_gpus.<locals>.assign_server_round_robin
allocate_gpus.<locals>.assign_servers
build_server_environment
build_worker_environment
calculate_nodes
convert_envs_to_str
format_export_string
get_server_health_timeout
load_config
main
mooncake_pool_config
parse_args
replace_env_in_file
resolve_mooncake_pool
resolve_remote_cache
save_env_file
save_env_file.<locals>.get_env_var_str
save_mooncake_worker_config
save_worker_config
submit_job
upsert_env_config
```

### `examples/disaggregated/slurm/cache_transceiver_test/report.py`

库存函数 23；已覆盖 22；未覆盖 1。

```text
_parse_proto_info_by_case_ts.<locals>._ci_for
```

### `examples/disaggregated/slurm/cache_transceiver_test/run_cache_transceiver_test.py`

库存函数 36；已覆盖 29；未覆盖 7。

```text
_alarm_handler
_coordinate_abort_after_leader_flush
_preserve_cpp_csvs
main.<locals>._on_hang
main.<locals>.flush_status
main.<locals>.hard_abort_sweep
main.<locals>.record_remaining
```

### `examples/disaggregated/slurm/cache_transceiver_test/submit.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
main
parse_args
```

### `examples/kimi_k3/make_synthetic_dflash_drafter.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
target_dims_from_ckpt
```

### `examples/layer_wise_benchmarks/parser_utils.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
_core_scs
```

### `examples/layer_wise_benchmarks/run.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
comma_separated_floats
```

### `examples/llm-api/llm_kv_cache_connector_vswa.py`

库存函数 17；已覆盖 16；未覆盖 1。

```text
main
```

### `examples/llm-api/quickstart_advanced.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
_parse_kv_cache_manager_v2
```

### `examples/models/core/multimodal/qwen_image_bench_eval.py`

库存函数 13；已覆盖 8；未覆盖 5。

```text
create_sampling_params
evaluate
main
make_trtllm_input
parse_args
```

### `examples/scaffolding/mcp/coder/coder_mcp.py`

库存函数 31；已覆盖 2；未覆盖 29。

```text
_build_grep_command
_build_list_dir_command
_build_read_file_command
_cid
_current_plan_state
_entry_label
_entry_sort_key
_extract_code
_format_grep_result
_format_http_error
_format_list_result
_format_read_result
_format_shell_result
_format_tool_error
_install_uvloop
_q
complete_task
create_starlette_app
create_starlette_app.<locals>.lifespan
exec_tool
grep_files
handle_health
list_dir
main
python_interpreter
read_file
shell
think
update_plan
```

### `examples/serve/anthropic_compatibility/gateway.py`

库存函数 52；已覆盖 23；未覆盖 29。

```text
BufferedUpstream.__init__
BufferedUpstream.read
BufferedUpstream.read_exact
BufferedUpstream.read_line
Fleet.reload_users
Gateway.proxy
SseTracker.__init__
SseTracker.feed
SseTracker.terminal
apply_probe
chunk_frame
discovery_loop
emit_sse_payload
health_loop
json_response
main
main_async
parse_args
probe
relay
relay_chunked_sse
relay_close_delimited_sse
relay_sized_sse
rewrite_sse_head
run_serve_sh
run_slurm_command
slurm_job_status
submit_successor
supervisor_loop
```

### `examples/serve/grafana/dashboard.py`

库存函数 18；已覆盖 17；未覆盖 1。

```text
write_outputs
```

### `examples/visual_gen/models/cosmos3/cosmos3.py`

库存函数 14；已覆盖 12；未覆盖 2。

```text
_default_action_output_path
_json_object
```

### `scripts/bolt/apply_bolt.py`

库存函数 17；已覆盖 8；未覆盖 9。

```text
_record_hash
_rewrite_record
bolt_standalone_wheel
is_elf
main
process_wheel
run
sha256_file
verify_manifest
```

### `scripts/bolt/internal/check_a53_veneers.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
main
```

### `scripts/bolt/manifest.py`

库存函数 12；已覆盖 2；未覆盖 10。

```text
cmd_build
cmd_verify
detect_glibc
detect_llvm_bolt_version
detect_ref
detect_triple
hash_originals
hash_profiles
main
sha256
```

### `scripts/build_wheel.py`

库存函数 40；已覆盖 16；未覆盖 24。

```text
_fmha_generation_stamp
add_arguments.<locals>._plat_name_type
check_missing_libs
clear_folder
create_cuda_stub_links
create_venv
generate_fmha_cu
generate_fmha_cu.<locals>.move_if_updated
generate_python_stubs_linux
generate_python_stubs_windows
get_build_dir
get_fmha_gen_dirs
get_project_dir
get_source_dir
install_editable_package
main
main.<locals>.copy_resolving_symlink
main.<locals>.get_binding_lib
main.<locals>.symlink_remove_dst
main.<locals>.symlink_remove_dst_tree
setup_venv
stage_nccl_extensions_package
sysconfig_scheme
working_directory
```

### `scripts/check_cpp_licenses.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
_raise_walk_error
```

### `scripts/check_test_list.py`

库存函数 40；已覆盖 31；未覆盖 9。

```text
_ensure_dir_indexed
_format_entry_tuple
_get_parametrize_with_ids_ids
_has_method
_matches_parameterized
check_waive_duplicates
install_python_dependencies
main
verify_l0_test_lists
```

### `scripts/generate_config_table.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
build_curated_rows
```

### `scripts/generate_llm_args_golden_manifest.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
golden_manifest
```

### `scripts/generate_openengine_protos.py`

库存函数 18；已覆盖 10；未覆盖 8。

```text
_generate
_load_manifest
_read_generator_requirements
_read_runtime_ceiling
_read_runtime_floor
_rewrite_generated_file
_validate_gencode_versions
_version_tuple
```

### `scripts/generate_trtllm_serve_schemas.py`

库存函数 32；已覆盖 31；未覆盖 1。

```text
main
```

### `scripts/get_wheel_from_package.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
bolt_optimize_wheels
```

### `scripts/legacy_utils.py`

库存函数 25；已覆盖 12；未覆盖 13。

```text
do_check
do_prune
eprint
generate_precommit_block
generate_pyproject_block
generate_ruff_legacy_toml
hash_file
hash_files
precommit_mode
render_template
show_diff
update_baseline_mode
validate_paths
```

### `scripts/split_qa_by_gpu.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
main
```

### `scripts/trtllm_gen_fmha_stage_kv/gen_stage_kv_cubins.py`

库存函数 23；已覆盖 5；未覆盖 18。

```text
apply_e4
build_extension
enumerate_kernels
fail
generate
jitify_headers
log
main
make_exec_dir
meta_entry
nvrtc_compile
parse_heads
redirect_fds
smem_bytes
smem_bytes.<locals>.check
split_options
trtllm_package_dir
write_archive
```

### `scripts/vendor/manage.py`

库存函数 79；已覆盖 78；未覆盖 1。

```text
_command_list
```

### `scripts/visualgen_eval/visual_gen_lpips_score_eval.py`

库存函数 19；已覆盖 10；未覆盖 9。

```text
_candidate_model_roots
_extract_image
_extract_video
_load_yaml
_resolve_model
_resolve_output_path
_save_image
_save_video
_visual_gen_params_from_dict
```

### `tensorrt_llm/_bootstrap.py`

库存函数 15；已覆盖 14；未覆盖 1。

```text
_init.<locals>._print_stacks
```

### `tensorrt_llm/_jit_share.py`

库存函数 46；已覆盖 42；未覆盖 4。

```text
_copy
_copy_file
_publish
main
```

### `tensorrt_llm/_torch/alltoall_watchdog.py`

库存函数 40；已覆盖 25；未覆盖 15。

```text
AlltoAllWatchdog.__enter__
AlltoAllWatchdog.__exit__
AlltoAllWatchdog._handle_timeout
AlltoAllWatchdog._missing_ranks
AlltoAllWatchdog._stop_after_error
AlltoAllWatchdog.last_error
AlltoAllWatchdog.wait_until_idle
AlltoAllWatchdogCoordinator._matching_config
AlltoAllWatchdogCoordinator.active_rank_mask_tensor
CompletionFlagReader.read_completion_flags
EPGroupHealthLike.get_mask
EPGroupHealthLike.get_mask_words
EPGroupHealthLike.snapshot
EPGroupHealthSnapshotLike.generation
EPGroupHealthSnapshotLike.mask
```

### `tensorrt_llm/_torch/async_llm.py`

库存函数 15；已覆盖 4；未覆盖 11。

```text
AsyncLLM.__aenter__
AsyncLLM.__aexit__
AsyncLLM.__await__
AsyncLLM.__enter__
AsyncLLM.__init__
AsyncLLM.collective_rpc
AsyncLLM.generate_async
AsyncLLM.pause_generation
AsyncLLM.resume_generation
AsyncLLM.setup_async
AsyncLLM.update_weights
```

### `tensorrt_llm/_torch/attention/attention.py`

库存函数 39；已覆盖 38；未覆盖 1。

```text
Attention.apply_qk_norm
```

### `tensorrt_llm/_torch/attention/backends/_cute_dsl_compat.py`

库存函数 13；已覆盖 11；未覆盖 2。

```text
JitSpecCuteDsl.compile_and_persist
_reset_compile_lock
```

### `tensorrt_llm/_torch/attention/backends/flashinfer.py`

库存函数 81；已覆盖 79；未覆盖 2。

```text
FlashInferAttentionMetadata.update_for_spec_dec
_slice_paged_kv_cache_heads
```

### `tensorrt_llm/_torch/attention/backends/fmha/beam_shared_prefix.py`

库存函数 42；已覆盖 40；未覆盖 2。

```text
BeamSharedPrefixFmha._unfused_tail_and_merge
BeamSharedPrefixFmha.run_context
```

### `tensorrt_llm/_torch/attention/backends/fmha/beam_shared_prefix_kernels.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_tail_merge_kernel
```

### `tensorrt_llm/_torch/attention/backends/fmha/cute_dsl_mla.py`

库存函数 20；已覆盖 19；未覆盖 1。

```text
_mla_split_kv_plan_kernel
```

### `tensorrt_llm/_torch/attention/backends/fmha/flash_attn4_kernels.py`

库存函数 16；已覆盖 1；未覆盖 15。

```text
_merge_splits_kernel
_rope_write_kv_paged_kernel
_split_tables_kernel
_write_kv_paged_kernel
allocate_split_workspace
fa4_paged_generation
fa4_paged_generation_split
fused_rope_supported
generation_splits
num_sms
page_layout
rope_write_kv_paged
rotary_dim
split_tables
write_kv_paged
```

### `tensorrt_llm/_torch/attention/backends/fmha/flashinfer_trtllm_gen.py`

库存函数 34；已覆盖 30；未覆盖 4。

```text
FlashInferTrtllmGenFmha._get_kv_scale_params
FlashInferTrtllmGenFmha.layout
FlashInferTrtllmGenFmha.run_mla_generation
_install_flashinfer_mla_decode_tuning_config_cache.<locals>._cached_build
```

### `tensorrt_llm/_torch/attention/backends/fmha/gqa_decode.py`

库存函数 58；已覆盖 53；未覆盖 5。

```text
GqaKvReplicaFusedFmha._rope_gather
GqaKvReplicaFusedFmha._run_kv_replica
GqaKvReplicaFusedFmha.prepare_workspace
_GqaKvReplicaFmha._kv_replica_workspace
_GqaKvReplicaFmha._takes
```

### `tensorrt_llm/_torch/attention/backends/fmha/gqa_kv_replica.py`

库存函数 9；已覆盖 6；未覆盖 3。

```text
_cp_prep_draft
_cp_prep_table
_kv_replica_prep_kernel
```

### `tensorrt_llm/_torch/attention/backends/fmha/gqa_segmented_decode.py`

库存函数 9；已覆盖 5；未覆盖 4。

```text
_seg_merge_kernel
_seg_prep_kernel
draft_block
seg_geometry
```

### `tensorrt_llm/_torch/attention/backends/fmha/interface.py`

库存函数 9；已覆盖 7；未覆盖 2。

```text
Fmha.forward
Fmha.prewarm
```

### `tensorrt_llm/_torch/attention/backends/fmha/phased.py`

库存函数 14；已覆盖 10；未覆盖 4。

```text
PhasedFmha.run_context
PhasedFmha.run_generation
PhasedFmha.run_mla_context
PhasedFmha.run_mla_generation
```

### `tensorrt_llm/_torch/attention/backends/fmha/shared_prefix_fa4.py`

库存函数 14；已覆盖 6；未覆盖 8。

```text
Fa4Tail.__init__
Fa4Tail.build_tables
Fa4Tail.run
_fa4_tail_table_kernel
_gather_q_kernel
_merge_fa4_tail_kernel
fa4_prefix_splits
fa4_shared_prefix_attention
```

### `tensorrt_llm/_torch/attention/backends/fmha/triton_custom_mask.py`

库存函数 7；已覆盖 4；未覆盖 3。

```text
TritonCustomMaskFmha._get_multi_processor_count
TritonCustomMaskFmha.prepare_workspace
TritonCustomMaskFmha.run_context
```

### `tensorrt_llm/_torch/attention/backends/fmha/utils.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
get_multi_ctas_kv_counter
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/cache_layout.py`

库存函数 46；已覆盖 45；未覆盖 1。

```text
Fp4MlaCacheLayout.blocks_in_primary_pool
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/cache_manager.py`

库存函数 22；已覆盖 21；未覆盖 1。

```text
Fp4MlaV2CacheLayoutPolicy.blocks_in_primary_pool
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/call_cache.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
LruCache.__len__
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/config.py`

库存函数 26；已覆盖 25；未覆盖 1。

```text
_fp4_mla_fold_q_enabled
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/cute_dsl_utils.py`

库存函数 7；已覆盖 5；未覆盖 2。

```text
_PyIRStdoutFilter.__getattr__
_PyIRStdoutFilter.flush
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/cutedsl_common.py`

库存函数 42；已覆盖 37；未覆盖 5。

```text
_compact_sfb_copy_cols_per_scale_chunk
_pack_e2m1x8
_pack_p4_16_scores_with_sum
_qk_sfb_cols_per_kblock
_softmax_exp2_pair
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/decode.py`

库存函数 23；已覆盖 20；未覆盖 3。

```text
_cutedsl_pad_q_and_sf_kernel
_cutedsl_swizzled_sf_offset
_launch_folded_query_decode
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/decode_4x.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
_fp4_mla_4x_record.<locals>._dump
_fp4_mla_cutedsl_4x_reason
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_cutedsl_4x.py`

库存函数 45；已覆盖 44；未覆盖 1。

```text
_pack_p4_16_scores_from_group
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_cutedsl_4x_host.py`

库存函数 26；已覆盖 20；未覆盖 6。

```text
_storage_bytes_from_offset
check_table_contents
compile_variant
compiled_variants
describe
zero_queries
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_cutedsl_mufu16.py`

库存函数 75；已覆盖 74；未覆盖 1。

```text
_load_global_page_native_tile_pair
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_cutedsl_v_repack.py`

库存函数 11；已覆盖 10；未覆盖 1。

```text
fp4_mla_v_repack_kernel
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_dispatch.py`

库存函数 14；已覆盖 13；未覆盖 1。

```text
describe
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_splitkv.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
_partition_pages_kernel
_repeat_q_kernel
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/fp4_mla_splitkv_vector.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_reduce_splitkv_vector_kernel
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/graph_resolvers.py`

库存函数 18；已覆盖 17；未覆盖 1。

```text
fp4_mla_4x_unavailable
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/metadata.py`

库存函数 20；已覆盖 13；未覆盖 7。

```text
_fp4_mla_append_metadata_kernel
_fp4_mla_generation_lengths_kernel
_fp4_mla_materialize_page_table_kernel
_fp4_mla_store_sequence_append_metadata
_helix_spec_buffers_kernel
_host_int_list
_host_int_list_during_forward
```

### `tensorrt_llm/_torch/attention/backends/fp4_mla/state.py`

库存函数 12；已覆盖 11；未覆盖 1。

```text
Fp4MlaState.decode_graph_variant
```

### `tensorrt_llm/_torch/attention/backends/interface.py`

库存函数 69；已覆盖 57；未覆盖 12。

```text
AttentionBackend.forward
AttentionBackend.runs_without_host_plan
AttentionBackend.runtime_workspace_is_chunked_prefill_bounded
AttentionBackend.runtime_workspace_min_tokens
AttentionBackend.runtime_workspace_profiled_tokens
AttentionBackend.support_mla
AttentionBackend.supports_variable_window
AttentionMetadata.decode_graph_variant
AttentionMetadata.set_decode_graph_variant
AttentionMetadata.update_for_spec_dec
AttentionMetadata.update_helix_param
PositionalEmbedder.__call__
```

### `tensorrt_llm/_torch/attention/backends/mixed_precision_rows.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
_store_mixed_precision_low_rows_kernel
copy_rows
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/_block_sparse/compiler.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
_compile_block_sparse.<locals>.contiguous_adapter
_compile_block_sparse.<locals>.paged_tensor_adapter
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/_block_sparse/inspection.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
_bsr_error_reason
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/_block_sparse/runtime.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
prepare_block_sparse_run_unchecked
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/_q_token_kv_block_sparse_policy.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
candidate_union_tokens
suggest_sparse_group
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/block_sparse.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
_BlockSparseWrapperBase._policy
_BlockSparseWrapperBase._published_state
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/context.py`

库存函数 85；已覆盖 70；未覆盖 15。

```text
_build_context_sage_state
_context_sage_layout
_get_compiled_context.<locals>.tensor_adapter
_get_compiled_paged_context.<locals>.tensor_adapter
_mxfp8_scale_atoms
_needs_packed_dense_k_mask
_refresh_variable_window_cta_starts
_resolve_geometry
_validate_context_sage_config
_validate_runtime_inputs
_validate_value_layout
_validate_variable_window_bounds
_validate_variable_window_cta_starts
_variable_window_tile_size_q
batch_prefill
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/decode.py`

库存函数 89；已覆盖 77；未覆盖 12。

```text
BatchDecodePagedTSWrapper._policy
PrimsTSBatchDecodePlan.run
_get_compiled_decode.<locals>.main_tensor_adapter
_get_compiled_decode.<locals>.reduction_tensor_adapter
_read_packed_q_plan_metadata
_resolve_decode_launch_spec.<locals>.fits_one_service_wave
_validate_packed_q_plan_values
_validate_prims_ts_q_token_kv_block_sparse_group_layout
_validate_scale
prepare_prims_ts_batch_decode_with_kv_cache
suggest_q_token_kv_block_sparse_group_size
validate_q_token_kv_block_sparse_group_size
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_context/fmha_kernel.py`

库存函数 46；已覆盖 37；未覆盖 9。

```text
FmhaTs.__call__
FmhaTs.__call__.<locals>.launch_kernel
FmhaTs.kernel
VariableWindowDomainTask.__init__
VariableWindowDomainTask.get_domain
_check_mxfp8_qk
_check_sage_attention
_configure_head_paired_tilers
_configure_head_paired_tma_copy_metadata
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_context/fmha_resources.py`

库存函数 281；已覆盖 108；未覆盖 173。

```text
FmhaConfig.sage_k_tile_bytes
FmhaConfig.sage_k_tile_words
FmhaConfig.sage_score_blocks
FmhaConfig.sage_score_seed_tile_bytes
FmhaConfig.sage_uses_ldtm_stat
GmemOResource.init_store_state
GmemOResource.tma_store
GmemQKVResource.compute_coords
GmemQKVResource.compute_page_coords
SmemKVResource._init_smem_state
SmemKVResource._tma_load
SmemKVResource._zero_paged_v_tail
SmemKVResource.holds_k
SmemKVResource.holds_v
SmemKVResource.init_descriptor_state
SmemKVResource.init_load_state
SmemKVResource.k_desc
SmemKVResource.k_load
SmemKVResource.k_load_stage
SmemKVResource.k_stage_desc
SmemKVResource.loop_offset_sensitive
SmemKVResource.v_desc
SmemKVResource.v_desc_paged
SmemKVResource.v_load
SmemKVResource.v_load_stage
SmemKVResource.v_load_stage_cached
SmemOResource._init_smem_state
SmemOResource._init_work_tile_state
SmemOResource._store_o
SmemOResource.compute_output_coords
SmemOResource.init_output_state
SmemOResource.init_output_work_tile_state
SmemOResource.init_store_state
SmemOResource.init_store_work_tile_state
SmemOResource.store_o
SmemPageOffsetsKvResource._init_smem_state
SmemPageOffsetsKvResource._producer_load_page_offsets
SmemPageOffsetsKvResource.cache_tile_page_ids
SmemPageOffsetsKvResource.dma_consumer_release_labels_for
SmemPageOffsetsKvResource.init_cached_read_state
SmemPageOffsetsKvResource.init_load_state
SmemPageOffsetsKvResource.init_read_state
SmemPageOffsetsKvResource.load_k
SmemPageOffsetsKvResource.load_v
SmemPageOffsetsKvResource.page_ids
SmemPageOffsetsKvResource.read_offsets
SmemQResource._init_smem_state
SmemQResource.init_descriptor_state
SmemQResource.init_load_state
SmemQResource.q0_desc
SmemQResource.q1_desc
SmemQResource.tma_load
SmemSageKScalesResource.__init__
SmemSageKScalesResource.copy_tile
SmemSageKScalesResource.get_smem_requirements
SmemSageKScalesResource.open_stage
TmemOResource._correct_impl
TmemOResource._init_function_state
TmemOResource._init_work_tile_state
TmemOResource._pv_mma_impl
TmemOResource._vc_mean_mma
TmemOResource._vc_rowsum_desc
TmemOResource.correct
TmemOResource.init_correction_state
TmemOResource.init_correction_work_tile_state
TmemOResource.init_mma_state
TmemOResource.init_mma_work_tile_state
TmemOResource.loop_offset_sensitive
TmemOResource.pv_mma
TmemOResource.set_p_base
TmemOResource.vc_mean_mma
TmemOResource.vc_mean_mma_last
TmemPResource.create_function_variables
TmemPResource.get_tmem_requirements
TmemPResource.p_base
TmemSPResource._apply_causal_mask
TmemSPResource._apply_causal_mask_for_kv_tile
TmemSPResource._default_p_chunk
TmemSPResource._dequantize_s_chunks
TmemSPResource._exp2_p_store
TmemSPResource._exp2_p_store_d128_fp8_cadence
TmemSPResource._exp2_p_store_d256_fp8_cadence
TmemSPResource._exp2_p_store_expcast
TmemSPResource._exp2_p_store_packed_words
TmemSPResource._init_function_state
TmemSPResource._init_work_tile_state
TmemSPResource._load_s_chunks
TmemSPResource._load_s_chunks_and_reduce_row_max
TmemSPResource._mask_fixed_dense_k_tail
TmemSPResource._mask_window_chunks
TmemSPResource._mx_copy_scales
TmemSPResource._qk_mma_impl
TmemSPResource._reduce_row_max
TmemSPResource._row_sum_reduction
TmemSPResource._sage_block_scales
TmemSPResource._sage_dequantized_tile_max
TmemSPResource._sage_load_s_chunks_and_reduce_row_max
TmemSPResource._sage_scale_block_maxima
TmemSPResource._score_seed_words
TmemSPResource._stage_col_offset
TmemSPResource._store_p_row_smem
TmemSPResource._store_vc_rowsum_group
TmemSPResource._varlen_batch_coord
TmemSPResource._vc_coords
TmemSPResource._vc_row_base
TmemSPResource._vc_rowsum_view
TmemSPResource._zero_vc_rowsum_row
TmemSPResource.cache_q_offset
TmemSPResource.cache_sage_scales
TmemSPResource.cache_seqlen_k
TmemSPResource.cache_variable_window_bounds
TmemSPResource.compute_row_max
TmemSPResource.exp2_p
TmemSPResource.exp2_p_hi
TmemSPResource.exp2_p_lo
TmemSPResource.exp2_p_smem
TmemSPResource.fixed_dense_k_tail_masked_row_max
TmemSPResource.freeze_row_max
TmemSPResource.init_mma_state
TmemSPResource.init_mma_work_tile_state
TmemSPResource.init_softmax_state
TmemSPResource.init_softmax_state_early
TmemSPResource.init_softmax_work_tile_state
TmemSPResource.invalid_exp2_p
TmemSPResource.invalid_row_max
TmemSPResource.left_masked_row_max
TmemSPResource.load_sage_k_words
TmemSPResource.load_scale_softmax_log2
TmemSPResource.loop_masked_row_max
TmemSPResource.loop_offset_sensitive
TmemSPResource.masked_exp2_p
TmemSPResource.masked_exp2_p_smem
TmemSPResource.masked_row_max
TmemSPResource.needs_window_tail_left_mask
TmemSPResource.p_read
TmemSPResource.packed_dense_k_masked_row_max
TmemSPResource.qk_mma
TmemSPResource.qk_mma_stage
TmemSPResource.read_staged_sage_k_words
TmemSPResource.right_masked_row_max
TmemSPResource.sage_dequantized_exp2_p_smem
TmemSPResource.sage_dequantized_row_max
TmemSPResource.sage_exp2_p_smem
TmemSPResource.sage_row_max
TmemSPResource.seed_scores
TmemSPResource.softmax_aux_identity
TmemSPResource.softmax_aux_reduce
TmemSPResource.store_p
TmemSPResource.variable_window_row_max
TmemSPResource.vc_compute_row_max
TmemSPResource.vc_fixed_dense_k_tail_masked_row_max
TmemSPResource.vc_init_row_scale
TmemSPResource.vc_store_p
TmemSPResource.vc_store_p_repair
TmemSPResource.vc_store_rowsum_final
TmemStatsResource._init_function_state
TmemStatsResource._init_work_tile_state
TmemStatsResource._read_vec
TmemStatsResource._stage_col_offset
TmemStatsResource._stats_smem_ptr
TmemStatsResource._store_final_lse
TmemStatsResource.init_read_state
TmemStatsResource.init_read_work_tile_state
TmemStatsResource.init_store_state
TmemStatsResource.init_store_work_tile_state
TmemStatsResource.load_output_scale
TmemStatsResource.load_scale_softmax_log2
TmemStatsResource.read_vec
TmemStatsResource.store_vec
_StructuredWaitPipelineAsync._retry_wait
_bmsk_clamp
_mask_score_quad
_mx_scale_slot
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_context/fmha_tasks.py`

库存函数 73；已覆盖 53；未覆盖 20。

```text
PackedContextWorkQueue.skip_work_tile_if
_prefetch_reused_page_windows_for_work_tile
create_epilogue_task.<locals>._create_single_instance_task
create_epilogue_task.<locals>._create_single_instance_task.<locals>.epilogue_schedule
create_load_task.<locals>.load_page_offsets_split_schedule
create_load_task.<locals>.load_reused_page_windows_schedule
create_load_task.<locals>.load_reused_page_windows_schedule_body
create_load_task.<locals>.load_reused_page_windows_schedule_body.<locals>.cache_v_tile
create_load_task.<locals>.load_reused_page_windows_schedule_body.<locals>.load_k_tile
create_load_task.<locals>.load_reused_page_windows_schedule_body.<locals>.load_v_tile
create_load_task.<locals>.load_reused_page_windows_split_schedule
create_load_task.<locals>.load_sage_schedule
create_mma_task.<locals>.mma_schedule_pr
create_mma_task.<locals>.mma_split_schedule
create_mma_task.<locals>.mma_split_schedule_pr
create_page_offsets_task.<locals>.reused_page_windows_schedule
create_softmax_task.<locals>.softmax_schedule_body.<locals>.sage_row_max
create_softmax_task.<locals>.softmax_schedule_body.<locals>.start_sage_k_words
create_softmax_task.<locals>.softmax_schedule_pb_skr
create_softmax_task.<locals>.softmax_schedule_pr
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_context/helpers.py`

库存函数 13；已覆盖 5；未覆盖 8。

```text
add_f16x2_to_f32x2
bottom_right_window_left_bound
bottom_right_window_max_tiles
bottom_right_window_tile_start
exp2_f16x2_word
f16x2_to_f32x2
pack_f16x2x2_to_fp8_e4m3
variable_window_cta_min_start
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_context/vc_attention/vc_resources.py`

库存函数 12；已覆盖 3；未覆盖 9。

```text
SmemMuResource._build_desc
SmemMuResource._init_smem_state
SmemMuResource._tma_load_tile
SmemMuResource.init_descriptor_state
SmemMuResource.init_load_state
SmemMuResource.loop_offset_sensitive
SmemMuResource.mu_desc
SmemMuResource.mu_desc_last
SmemMuResource.mu_load
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_context/vc_attention/vc_scales.py`

库存函数 7；已覆盖 4；未覆盖 3。

```text
VCKScaleTable.fill
VCKScaleTable.next_scale
VCKScaleTable.view
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/block_sparse_inspect.py`

库存函数 17；已覆盖 11；未覆盖 6。

```text
_InspectBlockSparseBsr.__call__
_InspectBlockSparseBsr._launch
_InspectBlockSparseBsr.kernel
_InspectPagedBlockSparseMetadata.__call__
_InspectPagedKvMetadata.__call__
_InspectPagedKvMetadata.kernel
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/block_sparse_prepare.py`

库存函数 37；已覆盖 32；未覆盖 5。

```text
_PrepareBitmaskRoutes.__call__
_PrepareBitmaskRoutes.kernel
_PrepareBsrRoutes.__call__
_PrepareBsrRoutes.kernel
_load_bsr_proxy_word
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/direct_sparse_metadata.py`

库存函数 16；已覆盖 5；未覆盖 11。

```text
DirectSparseMetadataView.__getitem__
DirectSparseMetadataView.load_page
DirectSparseMetadataView.map_page
DirectSparseMetadataView.page_capacity
DirectSparseMetadataView.resolve_route
DirectSparseMetadataView.selected_block_bound
DirectSparseMetadataView.with_head
HeadIndexedMetadataView.__extract_mlir_values__
HeadIndexedMetadataView.__getitem__
HeadIndexedMetadataView.__init__
HeadIndexedMetadataView.__new_from_mlir_values__
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_config.py`

库存函数 262；已覆盖 236；未覆盖 26。

```text
FmhaDecodeConfig._in_grouped_q_selection_scope
FmhaDecodeConfig.acc_dtype_bytes
FmhaDecodeConfig.fp8_copy_can_use_full_tile_fast_path
FmhaDecodeConfig.has_valid_split_kv_fanout
FmhaDecodeConfig.kv_dtype
FmhaDecodeConfig.max_runtime_row_split_segments
FmhaDecodeConfig.num_fp8_output_regs
FmhaDecodeConfig.ordered_softmax_early_release
FmhaDecodeConfig.resolved_softmax_order_barrier_threads
FmhaDecodeConfig.sf_tma_reshape_factor
FmhaDecodeConfig.smem_transformed_kv_tile_bytes
FmhaDecodeConfig.softmax_num_warps
FmhaDecodeConfig.tmem_alloc_cols
FmhaDecodeConfig.transform_kv_task_num_registers
FmhaDecodeConfig.use_bf16_q
FmhaDecodeConfig.use_fp8_q
QTileGeometry.tma_rows_per_cta
_grouped_q_cost_kv_width
_q_tma_rows_per_cta
compute_runtime_active_splits_kv
enumerate_auto_splits_kv
groups_tokens_heads_q_from_config_source
make_grouped_q_launch_candidate
mask_type_name
select_grouped_q_direct_wave_candidate
select_grouped_q_launch_candidate
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_kernel.py`

库存函数 30；已覆盖 28；未覆盖 2。

```text
_has_unmodeled_tmem_p_alias_protocol
_resolve_block_sparse_per_inst_load_topology
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/helpers_common.py`

库存函数 64；已覆盖 54；未覆盖 10。

```text
UniformQueryOffsets.__getitem__
_is_last_loop_iteration
_keeps_route_atom
_kv_dtype_elements
_named_barrier_arrive
_named_barrier_sync
_placeholder_smem_array
_q_dtype_elements
_shape_tuple
_swaps_routed_coordinate
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/helpers_output.py`

库存函数 12；已覆盖 9；未覆盖 3。

```text
_copy_transposed_smem8b_to_gmem
_fp8_smem_load_xor_bytes_for_shape
_store_transposed_smem8b_x1
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/helpers_softmax.py`

库存函数 14；已覆盖 11；未覆盖 3。

```text
_attention_sink_for_local_head
_attention_sink_for_scale_idx
_compute_p_values_and_local_sums
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/sage_scales.py`

库存函数 30；已覆盖 19；未覆盖 11。

```text
SageKScalesResource._ring
SageKScalesResource._sequence_words
SageKScalesResource._stage_words
SageKScalesResource.copy_route
SageKScalesResource.copy_tile
SageKScalesResource.get_tmem_requirements
SageKScalesResource.open_stage
SageVScalesResource.channel_scales
SageVScalesResource.get_tmem_requirements
SageVScalesResource.stage_tile
SageVScalesResource.staged
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/smem_block_sparse_metadata.py`

库存函数 34；已覆盖 10；未覆盖 24。

```text
SmemBlockSparseKvMetadataResource._create_initial_task_locals
SmemBlockSparseKvMetadataResource._prepared_route_logical_origin
SmemBlockSparseKvMetadataResource._prepared_route_physical_page_id_if_valid
SmemBlockSparseKvMetadataResource._route_record_word_offset
SmemBlockSparseKvMetadataResource.get_tmem_requirements
SmemBlockSparseKvMetadataResource.init_load_state
SmemBlockSparseKvMetadataResource.prefetch_route
SmemBlockSparseKvMetadataResource.resolve_route
SmemBlockSparseKvMetadataResource.route_atom_valid_mask
SmemBlockSparseKvMetadataResource.route_is_proxy
SmemBlockSparseKvMetadataResource.route_tma_coordinate
SmemBlockSparseKvMetadataResource.store_route
SmemBlockSparseSoftmaxMetadataResource._consumer_stage_base
SmemBlockSparseSoftmaxMetadataResource._create_initial_task_locals
SmemBlockSparseSoftmaxMetadataResource._load_route_swaps_values
SmemBlockSparseSoftmaxMetadataResource._producer_stage_base
SmemBlockSparseSoftmaxMetadataResource._store_route_keeps
SmemBlockSparseSoftmaxMetadataResource._store_route_swaps
SmemBlockSparseSoftmaxMetadataResource.get_tmem_requirements
SmemBlockSparseSoftmaxMetadataResource.init_load_state
SmemBlockSparseSoftmaxMetadataResource.init_read_state
SmemBlockSparseSoftmaxMetadataResource.load_route
SmemBlockSparseSoftmaxMetadataResource.store_route
_swaps_forwards_packed_route_full
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/smem_p.py`

库存函数 29；已覆盖 4；未覆盖 25。

```text
SmemPResource._bind_fragment_ready
SmemPResource._compute_keeps_p
SmemPResource._compute_p_fragments_impl
SmemPResource._compute_p_impl
SmemPResource._create_initial_task_locals
SmemPResource._create_work_tile_task_locals
SmemPResource._exponent_addend
SmemPResource._exponentiate_fragment_pairs
SmemPResource._fragment_exponent_terms
SmemPResource._load_score_fragment
SmemPResource._scale_fragment_pairs
SmemPResource._take_fragment_words
SmemPResource.compute_p
SmemPResource.compute_p_fragments
SmemPResource.compute_proxy_route_p
SmemPResource.compute_proxy_route_p_fragments
SmemPResource.compute_sage_p
SmemPResource.compute_sage_p_fragments
SmemPResource.compute_sage_proxy_route_p_fragments
SmemPResource.create_function_variables
SmemPResource.init_compute_state
SmemPResource.init_descriptor_state
SmemPResource.initialize_runtime_state_internal
SmemPResource.p_operands
SmemPResource.wait_p_fragment
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/smem_resources.py`

库存函数 102；已覆盖 21；未覆盖 81。

```text
SmemKvResource._build_kv_desc
SmemKvResource._create_initial_task_locals
SmemKvResource._create_work_tile_task_locals
SmemKvResource._local_tile_idx
SmemKvResource._logical_coords
SmemKvResource._maybe_runtime_tile_idx
SmemKvResource._producer_load_kv
SmemKvResource._producer_load_kv_tile_256
SmemKvResource._sf_stage_base
SmemKvResource._stage_base
SmemKvResource.get_tmem_requirements
SmemKvResource.init_descriptor_state
SmemKvResource.init_load_state
SmemKvResource.k_desc_0
SmemKvResource.k_desc_1
SmemKvResource.load_k0
SmemKvResource.load_k0_cached
SmemKvResource.load_k1
SmemKvResource.load_k1_cached
SmemKvResource.load_v0
SmemKvResource.load_v0_cached
SmemKvResource.load_v1
SmemKvResource.load_v1_cached
SmemKvResource.v_desc_0
SmemKvResource.v_desc_1
SmemKvTileResource._build_desc
SmemKvTileResource._create_initial_task_locals
SmemKvTileResource._create_work_tile_task_locals
SmemKvTileResource._local_tile_idx
SmemKvTileResource._maybe_runtime_tile_idx
SmemKvTileResource._producer_load
SmemKvTileResource._sf_stage_base
SmemKvTileResource._stage_base
SmemKvTileResource.get_tmem_requirements
SmemKvTileResource.init_descriptor_state
SmemKvTileResource.init_load_state
SmemKvTileResource.kv_desc
SmemKvTileResource.load_k0
SmemKvTileResource.load_k1
SmemKvTileResource.load_v0
SmemKvTileResource.load_v1
SmemKvTileResource.v_desc
SmemPageOffsetsKvResource._create_initial_task_locals
SmemPageOffsetsKvResource._prefetch_page_offsets_async
SmemPageOffsetsKvResource._producer_load_page_offsets
SmemPageOffsetsKvResource._q_token_kv_block_sparse_membership_word
SmemPageOffsetsKvResource.cache_page_ids
SmemPageOffsetsKvResource.encoded_locator_window_tiles
SmemPageOffsetsKvResource.get_tmem_requirements
SmemPageOffsetsKvResource.init_cached_read_state
SmemPageOffsetsKvResource.init_load_state
SmemPageOffsetsKvResource.init_read_state
SmemPageOffsetsKvResource.load_k0
SmemPageOffsetsKvResource.load_k1
SmemPageOffsetsKvResource.load_v0
SmemPageOffsetsKvResource.load_v1
SmemPageOffsetsKvResource.page_id
SmemPageOffsetsKvResource.page_id_pointer
SmemPageOffsetsKvResource.page_ids
SmemPageOffsetsKvResource.prefetch_query_memberships_async
SmemPageOffsetsKvResource.q_token_kv_block_sparse_page_keep_word16
SmemPageOffsetsKvResource.q_token_kv_block_sparse_page_membership
SmemPageOffsetsKvResource.q_token_kv_block_sparse_page_membership_word4
SmemPageOffsetsKvResource.query_membership_global_word
SmemPageOffsetsKvResource.query_membership_keep_word
SmemPageOffsetsKvResource.read_offsets
SmemPageOffsetsKvResource.read_offsets_k0
SmemPageOffsetsKvResource.read_offsets_k1
SmemPageOffsetsKvResource.read_offsets_v0
SmemPageOffsetsKvResource.read_offsets_v1
SmemQResource._complete_grouped_q_padding
SmemQResource._create_initial_task_locals
SmemQResource._create_work_tile_task_locals
SmemQResource._tma_coords
SmemQResource.current_consumer_q_desc
SmemQResource.get_tmem_requirements
SmemQResource.init_descriptor_state
SmemQResource.init_load_state
SmemQResource.q_desc
SmemQResource.tma_load
_q_token_membership_keep_word16
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/tmem_corr.py`

库存函数 80；已覆盖 7；未覆盖 73。

```text
TmemCorrResource._apply_sage_channel_scales
TmemCorrResource._cluster_complete_inactive_row_bytes
TmemCorrResource._cluster_init_transaction_barrier
TmemCorrResource._cluster_partial_o_i32_offset
TmemCorrResource._cluster_partial_row_idx
TmemCorrResource._cluster_partial_stats_offset
TmemCorrResource._cluster_reduction_cta_for_row
TmemCorrResource._cluster_reduction_local_row
TmemCorrResource._cluster_reduction_row_is_owned
TmemCorrResource._cluster_reduction_rows_per_cta
TmemCorrResource._cluster_store_async_vec2_f32
TmemCorrResource._cluster_store_async_vec4_i32
TmemCorrResource._cluster_wait_transaction_barrier
TmemCorrResource._copy_staged_fp16_o_segments
TmemCorrResource._copy_staged_fp16_o_to_output
TmemCorrResource._copy_staged_fp16_o_to_partial
TmemCorrResource._correction_tail_epilogue_impl
TmemCorrResource._final_row_weight
TmemCorrResource._fold_split_o_vec8
TmemCorrResource._gmem_partial_row_offset
TmemCorrResource._keeps_tail_epilogue
TmemCorrResource._kv_tile_256_merge_owned_columns
TmemCorrResource._kv_tile_256_merge_spatial_output
TmemCorrResource._kv_tile_256_output_row_state
TmemCorrResource._kv_tile_256_tail_epilogue
TmemCorrResource._kv_tile_256_temporal_fragment
TmemCorrResource._merge_kv_tile_256_peer_fragment
TmemCorrResource._multi_cta_counter_q_groups
TmemCorrResource._online_softmax_correction_scale
TmemCorrResource._pack_final_o_regs
TmemCorrResource._pack_fp8_output_quads
TmemCorrResource._pack_separate_partial_o_pair
TmemCorrResource._publish_and_reduce_cluster_swaps_partials
TmemCorrResource._publish_and_reduce_cluster_swaps_partials_impl
TmemCorrResource._reduce_and_store_gmem_split_o_vec8
TmemCorrResource._reduce_cluster_partial_segment
TmemCorrResource._reduce_cluster_partials_impl
TmemCorrResource._reduce_fused_gmem_partial_segment
TmemCorrResource._reduce_fused_gmem_partials
TmemCorrResource._reduce_fused_gmem_partials_impl
TmemCorrResource._runtime_splits_kv
TmemCorrResource._safe_norm_rcp
TmemCorrResource._separate_partial_lse
TmemCorrResource._separate_partial_norm_scale
TmemCorrResource._softmax_output_row_state
TmemCorrResource._split_partial_scale
TmemCorrResource._split_reduction_row_is_owned
TmemCorrResource._split_reduction_rows_per_cta
TmemCorrResource._stage_and_copy_swaps_partial_o
TmemCorrResource._stage_fp16_o_regs_to_smem
TmemCorrResource._store_16bit_output_chunks
TmemCorrResource._store_final_o_columns
TmemCorrResource._store_kv_tile_256_direct_fragment
TmemCorrResource._store_kv_tile_256_partial_fragment
TmemCorrResource._store_kv_tile_256_partial_stats
TmemCorrResource._store_softmax_normalized_o_vec8
TmemCorrResource._store_split_stats_from_scale_arrays
TmemCorrResource._swaps_load_o_stage_chunks
TmemCorrResource._swaps_load_two_o_stage_chunks
TmemCorrResource._swaps_o_stage_base_addr
TmemCorrResource._swaps_q8_tail_epilogue
TmemCorrResource._swaps_rescale_o_stage_in_tmem
TmemCorrResource._swaps_store_scaled_o_chunk
TmemCorrResource._swaps_wide_tail_epilogue
TmemCorrResource._sync_gmem_split_reducers
TmemCorrResource._warp_can_skip_o_correction
TmemCorrResource._warp_reduce_col_group_sum_pair
TmemCorrResource._zero_o_vec8
TmemCorrResource.correction_loop_epilogue
TmemCorrResource.correction_tail_epilogue
TmemCorrResource.create_function_variables
TmemCorrResource.get_tmem_requirements
TmemCorrResource.init_epilogue_state
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/tmem_o.py`

库存函数 13；已覆盖 3；未覆盖 10。

```text
TmemOResource._return_o_stage_state
TmemOResource._vp_mma
TmemOResource._vp_mma_fragment
TmemOResource.init_stage_state
TmemOResource.update_o_stage_loop
TmemOResource.update_o_stage_tail
TmemOResource.vp_mma_loop
TmemOResource.vp_mma_loop_fragment
TmemOResource.vp_mma_tail
TmemOResource.vp_mma_tail_fragment
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/tmem_s.py`

库存函数 61；已覆盖 6；未覆盖 55。

```text
TmemSResource._advance_qk_descs_after_mma_k
TmemSResource._apply_q_token_kv_block_sparse_swaps_page_membership_mask
TmemSResource._compute_softmax_loop_keeps
TmemSResource._compute_softmax_loop_keeps_fragments
TmemSResource._compute_softmax_loop_swaps
TmemSResource._create_initial_task_locals
TmemSResource._create_work_tile_task_locals
TmemSResource._dequantize_fragment_scores
TmemSResource._dequantize_sage_keeps_row
TmemSResource._fold_sage_fragment_max
TmemSResource._fold_sage_group_maxima
TmemSResource._load_keeps_fragment
TmemSResource._load_keeps_fragment_impl
TmemSResource._mask_score_fragments
TmemSResource._proxy_score_shifts
TmemSResource._publish_keeps_softmax_state
TmemSResource._q_desc_for_head_dim_stage
TmemSResource._q_token_kv_block_sparse_keeps_kv128_membership_word
TmemSResource._qk_head_stage_slot_offset
TmemSResource._qk_loop_stage_slot_offset
TmemSResource._qk_mma
TmemSResource._qk_tmem_col
TmemSResource._reduce_keeps_row_max
TmemSResource._reduce_keeps_tile_max
TmemSResource._resolve_keeps_tile_context
TmemSResource._seed_mma_tile_words
TmemSResource._seed_scores
TmemSResource._softmax_anchor
TmemSResource._softmax_loop_score_tmem_addr
TmemSResource._softmax_loop_stage_slot_offset
TmemSResource._stage_slot_offset_from_slot
TmemSResource.compute_block_sparse_softmax_loop
TmemSResource.compute_sage_block_sparse_softmax_loop
TmemSResource.compute_sage_softmax_loop
TmemSResource.compute_softmax_loop
TmemSResource.create_function_variables
TmemSResource.fill_score_seed_tiles
TmemSResource.init_softmax_state
TmemSResource.load_global_sum
TmemSResource.load_p_local_sum
TmemSResource.load_sage_q_scale
TmemSResource.qk_mma_head
TmemSResource.qk_mma_head_from_q_ref
TmemSResource.qk_mma_loop
TmemSResource.qk_mma_loop_from_q_ref
TmemSResource.reduce_sums
TmemSResource.seed_scores_head
TmemSResource.seed_scores_loop
TmemSResource.store_global_sum
TmemSResource.store_p_local_sum
_dense_fragment_keep_word
_sparse_effective_keep_word
_swaps_token_word_covers_kv_tail
_swaps_uses_origin0_k32_full_guard
_swaps_uses_token_only_score_validity
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/tmem_softmax_stats.py`

库存函数 31；已覆盖 7；未覆盖 24。

```text
TmemSoftmaxGlobalResource.get_tmem_requirements
TmemSoftmaxGlobalResource.global_correction
TmemSoftmaxLocalResource._create_initial_task_locals
TmemSoftmaxLocalResource._create_work_tile_task_locals
TmemSoftmaxLocalResource._load_stats_vector
TmemSoftmaxLocalResource._return_stats_state
TmemSoftmaxLocalResource._stats_ptr
TmemSoftmaxLocalResource._stats_smem_ptr
TmemSoftmaxLocalResource._stats_vector
TmemSoftmaxLocalResource._store_stats_vector
TmemSoftmaxLocalResource.get_consumer_requirements
TmemSoftmaxLocalResource.get_producer_requirements
TmemSoftmaxLocalResource.init_stats_state
TmemSoftmaxLocalResource.load_head_stats
TmemSoftmaxLocalResource.load_loop_stats
TmemSoftmaxLocalResource.load_tail_stats
TmemSoftmaxLocalResource.store_loop_old_new_stats
TmemSoftmaxLocalResource.store_loop_sum_new_stats
TmemSoftmaxLocalResource.store_tail_sum_new_stats
TmemSoftmaxOrderResource.prime_softmax1
TmemSoftmaxOrderResource.release_softmax0
TmemSoftmaxOrderResource.release_softmax1
TmemSoftmaxOrderResource.wait_softmax0
TmemSoftmaxOrderResource.wait_softmax1
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_resources/tmem_transformed_kv.py`

库存函数 21；已覆盖 1；未覆盖 20。

```text
TmemTransformedKvResource._create_initial_task_locals
TmemTransformedKvResource._create_work_tile_task_locals
TmemTransformedKvResource._destination_stage_addr
TmemTransformedKvResource._source_stage
TmemTransformedKvResource._store_k
TmemTransformedKvResource._store_v
TmemTransformedKvResource.get_smem_requirements
TmemTransformedKvResource.get_tmem_requirements
TmemTransformedKvResource.init_descriptor_state
TmemTransformedKvResource.init_load_state
TmemTransformedKvResource.k_desc_0
TmemTransformedKvResource.k_desc_1
TmemTransformedKvResource.transform_k0
TmemTransformedKvResource.transform_k1
TmemTransformedKvResource.transform_v0
TmemTransformedKvResource.transform_v1
TmemTransformedKvResource.v_desc_0
TmemTransformedKvResource.v_desc_1
_mul_e2m1x4_e4m3x4
_mul_e2m1x4_e4m3x4_bx
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_tasks.py`

库存函数 110；已覆盖 85；未覆盖 25。

```text
DecodeGenTask._begin_work_tile
DecodeGenTask._end_work_tile
DecodeGenTask._initial_body_work_tile
DecodeGenTask._q_group_has_tokens
DecodeGenTask._refresh_packed_q_bounds
DecodeGenTask._run_packed_skip_iteration
DecodeGenTask._run_task_body_impl
DecodeGenTask._run_task_body_persistent
DecodeGenTask.make_task_cache
PackedDecodeWorkQueue.skip_work_tile_if
ScheduleTokenThrottleResource.consume_schedule_token
ScheduleTokenThrottleResource.publish_schedule_token
_load_prepared_sparse_row_warp
create_block_sparse_load_tasks_per_inst
create_load_task_one_inst_qkv.<locals>.load_schedule
create_mma_task_split_kv.<locals>.mma_keeps_schedule
create_page_offsets_task_split_kv
create_page_offsets_task_split_kv.<locals>.page_offsets_schedule
create_page_offsets_task_split_kv.<locals>.page_offsets_schedule_body
create_transform_kv_task
create_transform_kv_task.<locals>.transform_kv_schedule_body
create_transform_kv_task.<locals>.transform_kv_schedule_body.<locals>.transform_one
create_transform_kv_task.<locals>.transform_shared_kv_schedule
create_transform_kv_task.<locals>.transform_split_kv_schedule
create_transform_kv_task.<locals>.transform_split_kv_schedule.<locals>.source_for_label
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/reduction.py`

库存函数 9；已覆盖 7；未覆盖 2。

```text
_attention_sink_log2_lse
decode_gen_separate_reduction_kernel
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/context_parallel.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
initialize_empty_query_rows
initialize_empty_rows
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/gather.py`

库存函数 18；已覆盖 13；未覆盖 5。

```text
CachedSparseMask.__new_from_mlir_values__
CachedSparseMask.is_invalid
gather4_cached
gather4_quad
make_sparse_rope_map
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/layout.py`

库存函数 26；已覆盖 20；未覆盖 6。

```text
kv_stage_elements
kv_stage_smem_element_offset
num_fp8_output_regs
q_stage_elements
tma_page_token_elems
v_stage_elements
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/math.py`

库存函数 27；已覆盖 26；未覆盖 1。

```text
max_softmax_rescale_threshold_log2
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/nvfp4_kv.py`

库存函数 32；已覆盖 19；未覆盖 13。

```text
Nvfp4KVConfig.max_cache_record_size
Nvfp4KVConfig.raw_record_stride
convert_nvfp4_record
copy_fp8_record
issue_whole_record
load_nvfp4_panels
make_whole_record_map
nvfp4_row_bytes
prefetch_nvfp4_kv_row
release_raw_record
snapshot_nvfp4_record
store_nvfp4_record
store_shared_u32x2
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/ops.py`

库存函数 28；已覆盖 26；未覆盖 2。

```text
_fp8_rescale_threshold_p_scale
store_transposed_smem8b_x1
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/output_quant.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
MlaOutputQuant.rotate
MlaOutputQuant.store
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/query.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
copy_q_fp8_to_tmem
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/schedule.py`

库存函数 17；已覆盖 16；未覆盖 1。

```text
staged_pv_mma_v_tile
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/sparse_indices.py`

库存函数 25；已覆盖 9；未覆盖 16。

```text
SparseKVIndices.__new_from_mlir_values__
SparseKVIndices.bind_query_row
SparseKVIndices.select_source_tile
SparseKVIndices.tensor_scale
SparseKVIndices.tensor_scale_is_extra
SparseKVIndices.tensor_scale_ratio
SparseKVLengths.__getitem__
SparseKVLengths.__new_from_mlir_values__
SparseSourceTile.__new_from_mlir_values__
SparseSourceTile._bounded_index
SparseSourceTile.is_invalid
SparseSourceTile.load_full_quad
SparseSourceTile.load_quad
SparseSourceTile.load_row
SparseSourceTile.prefix_mask32
SparseSourceTile.tile_is_full
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/tile.py`

库存函数 28；已覆盖 25；未覆盖 3。

```text
active_split_kv_count
batch_idx_for_stage
runtime_num_ctas_kv
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/tile_scheduler.py`

库存函数 19；已覆盖 8；未覆盖 11。

```text
MLAStaticTileScheduler.__extract_mlir_values__
MLAStaticTileScheduler.__new_from_mlir_values__
MLAStaticTileScheduler.advance_to_next_work
MLAStaticTileScheduler.get_current_work
MLAStaticTileScheduler.initial_work_tile_info
WorkTileInfo.__extract_mlir_values__
WorkTileInfo.__init__
WorkTileInfo.__new_from_mlir_values__
WorkTileInfo.is_valid_tile
WorkTileInfo.tile_idx
create_mla_static_tile_scheduler
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/parallel_reduction_topology.py`

库存函数 12；已覆盖 10；未覆盖 2。

```text
ParallelReductionTopology.padding_slots
ParallelReductionTopology.split_for_slot
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/sparse_policy.py`

库存函数 12；已覆盖 8；未覆盖 4。

```text
d576_v_split
prefer_sparse_d576_reload
resolve_sparse_mla_profile.<locals>.wave_splits
select_sparse_mla_d576_split
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_2cta/kernel.py`

库存函数 14；已覆盖 8；未覆盖 6。

```text
MlaDecodeTs.__call__
MlaDecodeTs.__call__.<locals>._flatten_query_rows_for_tma
MlaDecodeTs.initialize_workspace
MlaDecodeTs.parallel_reduction_kernel
MlaDecodeTs.reduction_kernel
MlaDecodeTs.split_kv_kernel
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_2cta/resources.py`

库存函数 106；已覆盖 9；未覆盖 97。

```text
GmemOResource._load_rotary_slice
GmemOResource._normalization_scale
GmemOResource._prefetch_output_rope
GmemOResource._query_row_state
GmemOResource._store_quantized_slice
GmemOResource._store_staged_bf16_slice
GmemOResource.epilogue_store
GmemOResource.epilogue_store_slice
GmemOResource.tma_store_output
MlaTsWorkTileInfo.__init__
MlaTsWorkTileInfo.__new_from_mlir_values__
MlaTsWorkTileInfo.is_valid_tile
MlaTsWorkTileInfo.k_index_base
MlaTsWorkTileInfo.k_len
MlaTsWorkTileInfo.k_tile_count
MlaTsWorkTileInfo.tile_idx
MlaTsWorkTileInfo.update_from
MlaWorkQueue._get_and_advance_work_tile_impl
MlaWorkQueue._initial_static_work_tile
MlaWorkQueue._linear_idx_from_tile
MlaWorkQueue._make_work_tile_info
MlaWorkQueue._next_static_work_tile
MlaWorkQueue._work_tile_from_block_idx
MlaWorkQueue._work_tile_from_clc_tile
MlaWorkQueue._work_tile_from_linear_idx
MlaWorkQueue.advance_tile
MlaWorkQueue.init_work_tile
MlaWorkQueue.initial_work_tile_info
MlaWorkQueue.k_tile_count_for_tile
MlaWorkQueue.skip_work_tile_if
MlaWorkQueue.wrap_work_tile
PageOffsetWindowResource.advance_dense_page_window
PageOffsetWindowResource.bind_sparse_request
PageOffsetWindowResource.forward_page_ids
PageOffsetWindowResource.init_dense_k_window
PageOffsetWindowResource.init_dense_v_window
PageOffsetWindowResource.init_read_state
PageOffsetWindowResource.init_sparse_state
PageOffsetWindowResource.load_sparse_offsets
PageOffsetWindowResource.read_dense_k_window
PageOffsetWindowResource.read_dense_v_window
PageOffsetWindowResource.read_page_offset_window
PageOffsetWindowResource.read_sparse_mask
PageOffsetWindowResource.sparse_stage
SmemKResource.bind_sparse_request
SmemKResource.init_load_state
SmemKResource.k_desc
SmemKResource.tma_load_paged
SmemKResource.tma_load_shared
SmemKVResource.init_load_state
SmemKVResource.k_desc
SmemKVResource.tma_load
SmemKVResource.v_desc
SmemPResource._store_p_impl
SmemPResource.p_desc
SmemPResource.store_p
SmemPResource.store_p_odd
SmemQResource._query_tma_coords
SmemQResource._tma_load
SmemQResource.init_load_state
SmemQResource.q_desc
SmemQResource.tma_load
SmemVResource.combined_stage
SmemVResource.init_load_state
SmemVResource.tma_load_paged
SmemVResource.tma_load_shared
SmemVResource.v_desc
SmemVResource.v_desc_n_major
TmemCorrResource._store_corr_impl
TmemCorrResource.init_load_state
TmemCorrResource.load_corr
TmemCorrResource.prepare_epilogue_slice_store
TmemCorrResource.store_corr
TmemCorrResource.store_corr_odd
TmemOResource.pv_mma
TmemOResource.pv_mma_n_major
TmemOResource.rescale_o
TmemOResource.rescale_o_slice
TmemSResource._finish_softmax_impl
TmemSResource._finish_softmax_impl.<locals>.load_peer_state
TmemSResource._load_s_impl
TmemSResource.bind_sparse_request
TmemSResource.finish_row_sum
TmemSResource.finish_row_sum_odd
TmemSResource.finish_softmax
TmemSResource.finish_softmax_odd
TmemSResource.init_softmax_state
TmemSResource.init_softmax_state_odd
TmemSResource.load_s
TmemSResource.load_s_cached
TmemSResource.load_s_odd
TmemSResource.qk_mma
with_cluster_acquire_waits
with_cluster_acquire_waits.<locals>.ClusterAcquirePipeline.consumer_try_wait
with_cluster_acquire_waits.<locals>.ClusterAcquirePipeline.consumer_wait
with_cluster_acquire_waits.<locals>.ClusterAcquirePipeline.try_cluster_acquire
with_cluster_acquire_waits.<locals>.ClusterAcquirePipeline.wait_cluster_acquire
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_2cta/tasks.py`

库存函数 61；已覆盖 42；未覆盖 19。

```text
MlaClcTask.get_domain
MlaInterleavedTask._create_stage_info
MlaInterleavedTask._end_work_tile
MlaInterleavedTask._run_one_mla_work_tile
MlaInterleavedTask.get_domain
MlaOutputTmaTask._drain
MlaOutputTmaTask._drain_mla_work_tile_tails
MlaStagedOutputTask._drain
MlaStagedOutputTask._drain_mla_work_tile_tails
MlaTask._drain_mla_work_tile_tails
MlaTask._run_one_mla_work_tile
MlaTask._run_task_body_persistent
_fixed_lane_work_tile_bounds
create_load_combined_task
create_load_combined_task.<locals>.combined_schedule
create_mma_pv_task
create_mma_pv_task.<locals>.mma_pv_schedule
create_mma_qk_task
create_mma_qk_task.<locals>.mma_qk_schedule
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_2cta/work_partition.py`

库存函数 8；已覆盖 3；未覆盖 5。

```text
_validate_partition_inputs
active_split_count
row_prefix_active_split_count
runtime_active_split_count
split_tile_range
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/config.py`

库存函数 100；已覆盖 82；未覆盖 18。

```text
MlaConfig.kv_tiles_per_multi_cta_group
MlaConfig.num_steps_per_cta_kv
MlaConfig.nvfp4_row_bytes
MlaConfig.padding_num_warps
MlaConfig.padding_warp_idx
MlaConfig.raw_record_ring_fits
MlaConfig.raw_record_stage_bytes
MlaConfig.raw_record_stages
MlaConfig.smem_p_tile_bytes
MlaConfig.sparse_d512
MlaConfig.tmem_total_cols
MlaConfig.v_smem_tile_bytes
MlaConfig.whole_kv_tile_bytes
explicit_split_profile
forced_persistent_profiles
keeps_mma_ab_explicit_split_profile
keeps_mma_ab_forced_persistent_profile
tile_size_q_for_heads
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/kernel.py`

库存函数 31；已覆盖 18；未覆盖 13。

```text
ThroughputLatencyMlaClcWorkQueue.skip_work_tile_if
ThroughputLatencyMlaDecodeTs.__call__
ThroughputLatencyMlaDecodeTs.dense_kernel
ThroughputLatencyMlaDecodeTs.gmem_reduction_kernel
ThroughputLatencyMlaDecodeTs.parallel_gmem_reduction_kernel
ThroughputLatencyMlaDecodeTs.validate_split_kv_launch
ThroughputLatencyMlaStaticWorkQueue.__init__
ThroughputLatencyMlaStaticWorkQueue._get_and_advance_work_tile_impl
ThroughputLatencyMlaStaticWorkQueue._make_initial_work_tile
ThroughputLatencyMlaStaticWorkQueue._work_tile_from_linear
ThroughputLatencyMlaStaticWorkQueue.initial_work_tile_info
ThroughputLatencyMlaStaticWorkQueue.skip_work_tile_if
_make_static_work_queue
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/parallel_reduction.py`

库存函数 13；已覆盖 12；未覆盖 1。

```text
parallel_gmem_reduction_elements_per_thread
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/common.py`

库存函数 7；已覆盖 3；未覆盖 4。

```text
MlaResource._init_tmem_state
MlaResource.get_tmem_requirements
ScheduleTokenThrottleResource.consume_schedule_token
ScheduleTokenThrottleResource.publish_schedule_token
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/smem_p.py`

库存函数 11；已覆盖 1；未覆盖 10。

```text
SmemPResource._init_order_p01_barriers
SmemPResource._init_smem_state
SmemPResource._init_smem_state_from_context
SmemPResource._ordered_sequence_arrive
SmemPResource._ordered_sequence_wait
SmemPResource.init_descriptor_state
SmemPResource.init_materialize_state
SmemPResource.initialize_runtime_state_internal
SmemPResource.materialize_p
SmemPResource.p_desc
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/smem_resources.py`

库存函数 50；已覆盖 7；未覆盖 43。

```text
SmemKvResource._init_smem_state
SmemKvResource._producer_copy_rope_stage
SmemKvResource._producer_load_kv_stage
SmemKvResource._raw_record_view
SmemKvResource._set_desc
SmemKvResource._stage_base
SmemKvResource.init_descriptor_state
SmemKvResource.init_load_state
SmemKvResource.k_desc_0
SmemKvResource.k_desc_1
SmemKvResource.kv_stage_offset
SmemKvResource.load_k0
SmemKvResource.load_k1
SmemKvResource.load_nvfp4_stage_payload
SmemKvResource.load_v0
SmemKvResource.load_v1
SmemKvResource.snapshot_raw_record
SmemKvResource.v_desc_0
SmemKvResource.v_desc_1
SmemKvResource.v_desc_reused
SmemKvResource.v_desc_reused_0
SmemKvResource.v_desc_reused_1
SmemPageOffsetsResource._init_smem_state
SmemPageOffsetsResource._producer_load_page_offsets
SmemPageOffsetsResource.bind_sparse_request
SmemPageOffsetsResource.init_load_state
SmemPageOffsetsResource.init_read_state
SmemPageOffsetsResource.load_k0
SmemPageOffsetsResource.load_k1
SmemPageOffsetsResource.load_v0
SmemPageOffsetsResource.load_v1
SmemPageOffsetsResource.page_id
SmemPageOffsetsResource.read_offsets
SmemQResource._init_smem_state
SmemQResource._load_q_stage
SmemQResource._query_tma_coords
SmemQResource.copy_nope_to_tmem
SmemQResource.init_descriptor_state
SmemQResource.init_load_state
SmemQResource.load_q
SmemQResource.q_desc
SmemQResource.stage_q_via_tma
_StructuredWaitPipelineAsync._retry_wait
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_corr.py`

库存函数 38；已覆盖 2；未覆盖 36。

```text
TmemCorrResource._apply_loop_correction_in_tmem
TmemCorrResource._cluster_local_smem_ptr
TmemCorrResource._cluster_lse_byte_offset
TmemCorrResource._cluster_o_byte_offset
TmemCorrResource._cluster_o_bytes
TmemCorrResource._cluster_remote_smem_ptr
TmemCorrResource._cluster_rows_per_cta
TmemCorrResource._cluster_wait_transaction_barrier
TmemCorrResource._compute_tail_softmax_scales
TmemCorrResource._init_cluster_reduction_barrier
TmemCorrResource._init_store_state_from_context
TmemCorrResource._normalize_and_store_tail_o
TmemCorrResource._o_base_col
TmemCorrResource._reduce_cluster_partials_and_store
TmemCorrResource._reduce_cluster_partials_and_store_for_tile
TmemCorrResource._rescale_loop_o
TmemCorrResource._stage_generic_tail_o_to_smem
TmemCorrResource._stage_swaps_tail_o_to_smem
TmemCorrResource._stage_tile32_tail_o_to_smem
TmemCorrResource._store_bf16_tail_regs_to_smem
TmemCorrResource._store_fp8_tail_vals_to_smem
TmemCorrResource._store_keeps_tail_o_stage
TmemCorrResource._store_lse_to_gmem
TmemCorrResource._store_o_slice_to_gmem
TmemCorrResource._store_partial_lse_to_cluster_smem
TmemCorrResource._store_partial_o_to_cluster_smem
TmemCorrResource._store_swap_quantized_stage
TmemCorrResource.correct_loop_and_store
TmemCorrResource.correct_tail_and_store
TmemCorrResource.create_cluster_function_variables
TmemCorrResource.init_store_state
TmemCorrResource.initialize_runtime_state_internal
TmemCorrResource.publish_neutral_cluster_partial_and_reduce
TmemCorrResource.tensor_scale_tile
TmemCorrResource.ws_row_scale
TmemCorrResource.ws_sum_reduce
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_o.py`

库存函数 16；已覆盖 1；未覆盖 15。

```text
TmemOResource._issue_smem_p_pv_mma
TmemOResource._issue_tmem_p_pv_mma
TmemOResource._p_desc_for_inst
TmemOResource._pv_o_base_addr
TmemOResource._tail_has_prior_o
TmemOResource.init_mma_state
TmemOResource.init_stage_state
TmemOResource.init_stage_work_tile_state
TmemOResource.o_stage
TmemOResource.pv_mma_loop_0
TmemOResource.pv_mma_loop_1
TmemOResource.pv_mma_loop_tmem_p
TmemOResource.pv_mma_tail_0
TmemOResource.pv_mma_tail_1
TmemOResource.pv_mma_tail_tmem_p
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_p.py`

库存函数 6；已覆盖 1；未覆盖 5。

```text
TmemPResource._p_base_addr
TmemPResource.init_stage_state
TmemPResource.init_stage_work_tile_state
TmemPResource.materialize_p
TmemPResource.p_stage
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_s.py`

库存函数 12；已覆盖 2；未覆盖 10。

```text
TmemSResource._make_initial_softmax_vars
TmemSResource._reset_softmax_state
TmemSResource.init_descriptor_state
TmemSResource.init_softmax_state
TmemSResource.init_softmax_work_tile_state
TmemSResource.qk_mma
TmemSResource.reset_softmax_work_tile_state
TmemSResource.set_q_desc
TmemSResource.update_softmax
TmemSResource.update_softmax_sum
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_softmax_stats.py`

库存函数 19；已覆盖 3；未覆盖 16。

```text
SmemSoftmaxLocalResource._load_stats_payload
SmemSoftmaxLocalResource._stats_pointer
SmemSoftmaxLocalResource.store_loop_stats
TmemSoftmaxGlobalResource.track_global
TmemSoftmaxLocalResource._load_stats_payload
TmemSoftmaxLocalResource._make_initial_stats_vars
TmemSoftmaxLocalResource._stats_base_addr
TmemSoftmaxLocalResource._stats_tuple
TmemSoftmaxLocalResource._store_loop_stats_payload
TmemSoftmaxLocalResource._store_tail_stats_payload
TmemSoftmaxLocalResource.init_stats_state
TmemSoftmaxLocalResource.init_stats_work_tile_state
TmemSoftmaxLocalResource.load_initial_stats
TmemSoftmaxLocalResource.load_loop_stats
TmemSoftmaxLocalResource.load_tail_stats
TmemSoftmaxLocalResource.store_loop_stats
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/tasks.py`

库存函数 53；已覆盖 45；未覆盖 8。

```text
MlaDecodeTask._load_allocated_tmem_base
MlaDecodeTask._resolve_domain
MlaDecodeTask.init_variables
MlaDecodeTask.make_task_cache
create_throughput_latency_load_task.<locals>.load_page_offsets_wq_schedule
create_throughput_latency_load_task.<locals>.load_schedule
create_throughput_latency_load_task.<locals>.load_wq_schedule
create_throughput_latency_load_task.<locals>.load_wq_throttle_schedule
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/task_compat.py`

库存函数 15；已覆盖 3；未覆盖 12。

```text
TaskCacheStageInfoMixin.__init__
TaskCacheStageInfoMixin._begin_work_tile
TaskCacheStageInfoMixin._classify_steps_around_domain_loop
TaskCacheStageInfoMixin._classify_steps_around_domain_loop.<locals>.walk
TaskCacheStageInfoMixin._create_stage_info
TaskCacheStageInfoMixin._end_work_tile
TaskCacheStageInfoMixin._first_domain_loop
TaskCacheStageInfoMixin._first_domain_loop.<locals>.walk
TaskCacheStageInfoMixin._is_work_tile_loop_body
TaskCacheStageInfoMixin._resolve_bound
TaskCacheStageInfoMixin.run_nodes
runtime_domain_end.<locals>.runtime_domain
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/tcgen05_compat.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
_ld_32x32b_max_ptx
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/mla_decode.py`

库存函数 60；已覆盖 54；未覆盖 6。

```text
_derive_max_seq_len_q
_sparse_d576_workspace_bytes
_validate_mla_nonnegative_int32_extent
_validate_qo_indptr
batch_sparse_mla_decode_with_paged_kv_cache
resolve_sparse_d576_reload_plan
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/q_token_kv_block_sparse_metadata.py`

库存函数 34；已覆盖 30；未覆盖 4。

```text
_build_prims_ts_q_token_kv_block_sparse_metadata
_build_q_token_kv_block_sparse_metadata
_get_q_token_kv_block_sparse_metadata_output_shapes
q_token_kv_block_sparse_attention_with_paged_kv_cache
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/sage/layout.py`

库存函数 22；已覆盖 21；未覆盖 1。

```text
_SageQSection.tile_index
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/sage/prepare.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
_PrepareSageScales.__call__
_PrepareSageScales.kernel
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/vc_attention/__init__.py`

库存函数 10；已覆盖 7；未覆盖 3。

```text
_per_token_scales
validate_vc_params
vc_reference
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/vc_attention/prepare.py`

库存函数 37；已覆盖 26；未覆盖 11。

```text
VcKvPass1.__call__
VcKvPass1.kernel
VcKvPass2.__call__
VcKvPass2.kernel
VcQKPass.__call__
VcQKPass._block
VcQKPass.kernel
VcRepairRowsPass.__call__
VcRepairRowsPass.kernel
VcSelectPass.__call__
VcSelectPass.kernel
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/vc_attention/quantization.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
_hadamard_matrix_cached
hadamard_matrix
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/backend.py`

库存函数 48；已覆盖 47；未覆盖 1。

```text
CSA2TrtllmAttention.runtime_workspace_is_chunked_prefill_bounded
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/block_select.py`

库存函数 29；已覆盖 13；未覆盖 16。

```text
_kernel_type.<locals>.BlockSelect.__call__
_kernel_type.<locals>.BlockSelect._block_scan
_kernel_type.<locals>.BlockSelect._emit
_kernel_type.<locals>.BlockSelect._first_blocks32
_kernel_type.<locals>.BlockSelect._first_tier
_kernel_type.<locals>.BlockSelect._key32
_kernel_type.<locals>.BlockSelect._key8
_kernel_type.<locals>.BlockSelect._load_blocks8
_kernel_type.<locals>.BlockSelect._load_children32
_kernel_type.<locals>.BlockSelect._load_first
_kernel_type.<locals>.BlockSelect._load_listed8
_kernel_type.<locals>.BlockSelect._load_positions
_kernel_type.<locals>.BlockSelect._packed_select
_kernel_type.<locals>.BlockSelect._publish
_kernel_type.<locals>.BlockSelect._select
_kernel_type.<locals>.BlockSelect.kernel
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/cache_manager.py`

库存函数 84；已覆盖 80；未覆盖 4。

```text
CSA2CacheManager.get_cache_bytes_per_token
CSA2CacheManager.get_layer_page_index_scale
CSA2CacheManager.get_max_resource_count
CSA2CacheManager.get_needed_resource_to_completion
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/compressor.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
_compress_project_kernel
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/decode_select.py`

库存函数 16；已覆盖 11；未覆盖 5。

```text
_consumer_kernel_class.<locals>.MappedRadixTopKKernel.__call__
_consumer_kernel_class.<locals>.MappedRadixTopKKernel.mapped_kernel
_kernel_type.<locals>.FusedDecodeSelect.__call__
_kernel_type.<locals>.FusedDecodeSelect._block_scan
_kernel_type.<locals>.FusedDecodeSelect.kernel
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/direct_decode.py`

库存函数 62；已覆盖 54；未覆盖 8。

```text
_kernel_type.<locals>.DirectSparseDecode.__call__
_kernel_type.<locals>.DirectSparseDecode._channel_owner
_kernel_type.<locals>.DirectSparseDecode._decode_rows
_kernel_type.<locals>.DirectSparseDecode._merge4
_kernel_type.<locals>.DirectSparseDecode._part
_kernel_type.<locals>.DirectSparseDecode._resolve
_kernel_type.<locals>.DirectSparseDecode._rope_query
_kernel_type.<locals>.DirectSparseDecode.kernel
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/indexer.py`

库存函数 50；已覆盖 49；未覆盖 1。

```text
CSA2Indexer._decode_radix_selector
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/kernel.py`

库存函数 211；已覆盖 96；未覆盖 115。

```text
_cache_kernels.<locals>.GatherDequantKernel.__call__
_cache_kernels.<locals>.GatherDequantKernel.kernel
_cache_kernels.<locals>.QuantizeScatterKernel.__call__
_cache_kernels.<locals>.QuantizeScatterKernel._job_row
_cache_kernels.<locals>.QuantizeScatterKernel._quantize_row
_cache_kernels.<locals>.QuantizeScatterKernel.kernel
_cache_kernels.<locals>.QuantizeScatterVecKernel._job_row
_cache_kernels.<locals>.QuantizeScatterVecKernel._quantize_row_vec
_cache_kernels.<locals>.ScatterPackedKernel.__call__
_cache_kernels.<locals>.ScatterPackedKernel.kernel
_cache_kernels.<locals>.SharedCompactKernel.__call__
_cache_kernels.<locals>.SharedCompactKernel.kernel
_cache_kernels.<locals>.SharedDecodeKernel.__call__
_cache_kernels.<locals>.SharedDecodeKernel._slot
_cache_kernels.<locals>.SharedDecodeKernel._word
_cache_kernels.<locals>.SharedDecodeKernel.kernel
_cache_kernels.<locals>.SharedRowScaleKernel.__call__
_cache_kernels.<locals>.SharedRowScaleKernel.kernel
_cache_kernels.<locals>.SharedScaleKernel.__call__
_cache_kernels.<locals>.SharedScaleKernel.kernel
_cache_kernels.<locals>.StageCompactKernel.__call__
_cache_kernels.<locals>.StageCompactKernel.kernel
_cache_kernels.<locals>.StageDedupKernel.__call__
_cache_kernels.<locals>.StageDedupKernel.kernel
_cache_kernels.<locals>.StageDequantKernel.__call__
_cache_kernels.<locals>.StageDequantKernel._resolve_row
_cache_kernels.<locals>.StageDequantKernel._stage_block
_cache_kernels.<locals>.StageDequantKernel._stage_resolved
_cache_kernels.<locals>.StageDequantKernel._stage_rows
_cache_kernels.<locals>.StageDequantKernel.kernel
_cache_kernels.<locals>.StageQueryKernel.__call__
_cache_kernels.<locals>.StageQueryKernel.kernel
_cache_kernels.<locals>.StageScaleFinalizeKernel.__call__
_cache_kernels.<locals>.StageScaleFinalizeKernel.kernel
_cache_kernels.<locals>.StageScalesKernel.__call__
_cache_kernels.<locals>.StageScalesKernel._largest_scale
_cache_kernels.<locals>.StageScalesKernel.kernel
_cache_kernels.<locals>.ceil_log2_bits
_cache_kernels.<locals>.decode_channels
_cache_kernels.<locals>.decode_channels_x16
_cache_kernels.<locals>.e2m1_to_f32
_cache_kernels.<locals>.e2m1x2_to_f32
_cache_kernels.<locals>.e4m3_f16x2
_cache_kernels.<locals>.e4m3_to_f32
_cache_kernels.<locals>.e4m3x2_to_f32
_cache_kernels.<locals>.f16x2_splat
_cache_kernels.<locals>.f32_to_e4m3_byte
_cache_kernels.<locals>.highest_scale_byte
_cache_kernels.<locals>.largest_scale_byte
_cache_kernels.<locals>.load_rows_x16
_cache_kernels.<locals>.map_logical_slot
_cache_kernels.<locals>.pdl_prologue
_cache_kernels.<locals>.q_amax_bits
_cache_kernels.<locals>.quantize_e4m3x16
_cache_kernels.<locals>.rescale_e4m3_chunks
_cache_kernels.<locals>.scaled_e2m1x4
_cache_kernels.<locals>.scaled_e4m3x4
_cache_kernels.<locals>.stage_row_x16
_cache_kernels.<locals>.staged_exponent
_cache_kernels.<locals>.staging_quant
_cache_kernels.<locals>.store_bf16x16
_cache_kernels.<locals>.store_bf16x4
_cache_kernels.<locals>.store_bmm_scales
_cache_kernels.<locals>.store_u32x4
_cache_kernels.<locals>.ue8m0_to_f32
_cache_kernels.<locals>.warp_max
_fake_indexer_projection
_glue_kernels.<locals>.CandidateBlockMaxKernel.__call__
_glue_kernels.<locals>.CandidateBlockMaxKernel.kernel
_glue_kernels.<locals>.CandidateBlockPrefetchKernel.__call__
_glue_kernels.<locals>.CandidateBlockPrefetchKernel.kernel
_glue_kernels.<locals>.CandidateExpandKernel.__call__
_glue_kernels.<locals>.CandidateExpandKernel.kernel
_glue_kernels.<locals>.CandidateGatherKernel.__call__
_glue_kernels.<locals>.CandidateGatherKernel.kernel
_glue_kernels.<locals>.FinalizeSelectionKernel.__call__
_glue_kernels.<locals>.FinalizeSelectionKernel.kernel
_glue_kernels.<locals>.FinalizeShuffleSelectionKernel.__call__
_glue_kernels.<locals>.FinalizeShuffleSelectionKernel.__init__
_glue_kernels.<locals>.FinalizeShuffleSelectionKernel.kernel
_glue_kernels.<locals>.IndexerDescriptorsKernel.__call__
_glue_kernels.<locals>.IndexerDescriptorsKernel.kernel
_glue_kernels.<locals>.MapSelectionKernel.__call__
_glue_kernels.<locals>.MapSelectionKernel.kernel
_glue_kernels.<locals>.MaskLogitsKernel.__call__
_glue_kernels.<locals>.MaskLogitsKernel.kernel
_glue_kernels.<locals>.MaskPackedLogitsKernel.__call__
_glue_kernels.<locals>.MaskPackedLogitsKernel.kernel
_glue_kernels.<locals>.OwnerSlotsKernel.__call__
_glue_kernels.<locals>.OwnerSlotsKernel.kernel
_glue_kernels.<locals>.PackedSortFinalizeKernel.__call__
_glue_kernels.<locals>.PackedSortFinalizeKernel.kernel
_glue_kernels.<locals>.PageTableConvertKernel.__call__
_glue_kernels.<locals>.PageTableConvertKernel.kernel
_glue_kernels.<locals>.PrefixGatherKernel.__call__
_glue_kernels.<locals>.PrefixGatherKernel._piece
_glue_kernels.<locals>.PrefixGatherKernel.kernel
_glue_kernels.<locals>.SwaRefreshKernel.__call__
_glue_kernels.<locals>.SwaRefreshKernel.kernel
_glue_kernels.<locals>.TableCopyKernel.__call__
_glue_kernels.<locals>.TableCopyKernel.kernel
_glue_kernels.<locals>._int32_ptr
_glue_kernels.<locals>.floor_div
_glue_kernels.<locals>.floor_mod
_glue_kernels.<locals>.tensor1d
_glue_kernels.<locals>.tensor2d
_indexer_projection_runner_type
_indexer_projection_runner_type.<locals>._CSA2IndexerQKernel._indexer_q_transform_rows
_indexer_projection_runner_type.<locals>._CSA2IndexerQRunner.forward
_indexer_projection_runner_type.<locals>._CSA2IndexerQRunner.get_valid_tactics
_indexer_projection_runner_type.<locals>._CSA2IndexerQRunner.unique_id
_packed_attention_kernel_type.<locals>.PackedAttention.__call__
_packed_attention_kernel_type.<locals>.PackedAttention.kernel
_packed_attention_kernel_type.<locals>.PackedAttention.reduce
csa2_indexer_q_gemm_rope_fp4
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/metadata.py`

库存函数 99；已覆盖 98；未覆盖 1。

```text
CSA2TrtllmMetadata.prepare_context_mla_with_cached_kv
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/projection.py`

库存函数 38；已覆盖 24；未覆盖 14。

```text
_block128_kernel_type.<locals>.GroupedBlock128OutputGemm.wrapper_mxfp8
_gptj_rope_small_m_kernel
_inverse_rope_block128_small_m_kernel
_inverse_rope_mxfp8_kernel
_kernel_type.<locals>.GroupedOutputGemm.wrapper_grouped
_l2_touch_kernel
_mxfp8_blocks
_n256_kernel_type.<locals>.GroupedN256OutputGemm.wrapper_mxfp8
_prefetch_share
_rms_norm_mxfp8_kernel
_split_k_epilogue
_store_small_m_output
_touch_range
_wait_prefetching
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/query_projection.py`

库存函数 13；已覆盖 9；未覆盖 4。

```text
_fixup_fp8_query
_kernel_type.<locals>.QueryGemm.wrapper_query
_kernel_type.<locals>.QueryGemm.wrapper_query_fp8
_rescale_fp8_query
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/reindex_select.py`

库存函数 24；已覆盖 18；未覆盖 6。

```text
_kernel_class.<locals>.ReindexSelectKernel.__call__
_kernel_class.<locals>.ReindexSelectKernel._radix_round_push
_kernel_class.<locals>.ReindexSelectKernel._row_threshold
_kernel_class.<locals>.ReindexSelectKernel._s2t
_kernel_class.<locals>.ReindexSelectKernel.fused_kernel
_kernel_class.<locals>.ReindexSelectKernel.logits_to_smem
```

### `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py`

库存函数 67；已覆盖 62；未覆盖 5。

```text
DeepseekV4CacheManager._guard_reservation_detail
DeepseekV4CacheManager._iter_guard_candidate_buffers
DeepseekV4CacheManager._resolve_guard_candidate
DeepseekV4CacheManager.blocks_in_primary_pool
DeepseekV4CacheManager.get_max_resource_count
```

### `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/footer_scale_kv.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
_quant_scatter_kernel
```

### `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/indexer.py`

库存函数 14；已覆盖 13；未覆盖 1。

```text
DeepseekV4Indexer._qk_projection_and_rope
```

### `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/kernels.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_deepseek_v4_local_to_global_kernel
```

### `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/module.py`

库存函数 40；已覆盖 37；未覆盖 3。

```text
_deepseek_v4_q_b_gemm_norm_rope_fused_fp8
_rubin_dsv4_o_b_proj_out
_rubin_dsv4_o_lora_quantized_input
```

### `tensorrt_llm/_torch/attention/backends/sparse/dsa/cache_manager.py`

库存函数 48；已覆盖 46；未覆盖 2。

```text
DSACacheManager.get_batch_indexer_k_cache_indices
_get_indexer_k_cache_size_per_token
```

### `tensorrt_llm/_torch/attention/backends/sparse/dsa/custom_ops.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
_mla_dsa_attn_inplace_fake
_mla_dsa_proj_fake
```

### `tensorrt_llm/_torch/attention/backends/sparse/dsa/indexer.py`

库存函数 66；已覆盖 59；未覆盖 7。

```text
Indexer._gather_k_cache_for_chunk
Indexer.forward
IndexerParams.all_kv_tokens
IndexerParams.block_stride
IndexerParams.scale_size
IndexerParams.total_tokens
_unravel_indices
```

### `tensorrt_llm/_torch/attention/backends/sparse/dsa/kernels.py`

库存函数 8；已覆盖 4；未覆盖 4。

```text
_convert_req_index_to_global_index_kernel_with_stride_factor
_deterministic_indexer_projection_kernel
_fused_dsa_decode_metadata_kernel
_triton_gather_k_cache_kernel
```

### `tensorrt_llm/_torch/attention/backends/sparse/dsa/metadata.py`

库存函数 68；已覆盖 67；未覆盖 1。

```text
DSAtrtllmAttentionMetadata.gen_token_repeats
```

### `tensorrt_llm/_torch/attention/backends/sparse/dsa/module.py`

库存函数 14；已覆盖 12；未覆盖 2。

```text
DSAHooks.forward
forward_sparse_attn
```

### `tensorrt_llm/_torch/attention/backends/sparse/dsa/throwaway_topk.py`

库存函数 14；已覆盖 13；未覆盖 1。

```text
ThrowawayPrefillTopK.compress_ratio
```

### `tensorrt_llm/_torch/attention/backends/sparse/dsa/vanilla_backend.py`

库存函数 54；已覆盖 46；未覆盖 8。

```text
DSAVanillaAttention.mla_rope_append_paged_kv_assign_q
DSAVanillaAttention.sparse_attn_predict
DSAVanillaAttention.sparse_kv_predict
DSAVanillaAttention.support_fused_rope
DSAVanillaIndexer._mtp_last_accepted_rows
DSAVanillaIndexer.forward
DSAVanillaIndexer.maybe_join_prev_topk_copy
DSAVanillaIndexer.post_load_weights
```

### `tensorrt_llm/_torch/attention/backends/sparse/dsa_flashinfer.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
_latent_append
```

### `tensorrt_llm/_torch/attention/backends/sparse/glm_kpool/backend.py`

库存函数 22；已覆盖 18；未覆盖 4。

```text
GlmKpoolSparseAttention._dispatch_sparse_core_sm120
GlmKpoolSparseAttention.create_output
GlmKpoolSparseAttention.support_fused_qkv
GlmKpoolSparseAttention.support_fused_rope
```

### `tensorrt_llm/_torch/attention/backends/sparse/glm_kpool/cache_manager.py`

库存函数 39；已覆盖 38；未覆盖 1。

```text
Glm5NextCacheManager.cuda_replay_state_indices
```

### `tensorrt_llm/_torch/attention/backends/sparse/glm_kpool/kernels.py`

库存函数 17；已覆盖 9；未覆盖 8。

```text
_append_paged_state_kernel
_compact_sparse_rows_kernel
_gather_fp8_kv_rows_kernel
_kpool_expand_kernel
_kpool_score_kernel
_kpool_update_kernel
_merge_split_attention_kernel
_pack_glm53_kv_rows_kernel
```

### `tensorrt_llm/_torch/attention/backends/sparse/glm_kpool/params.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
GlmKpoolSparseParams.indices_block_size
```

### `tensorrt_llm/_torch/attention/backends/sparse/hooks.py`

库存函数 19；已覆盖 17；未覆盖 2。

```text
MLASparseHooks.forward
MLASparseHooks.initialize
```

### `tensorrt_llm/_torch/attention/backends/sparse/inline_scale_kv.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
_quant_scatter_kernel
```

### `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py`

库存函数 62；已覆盖 52；未覆盖 10。

```text
MiniMaxM3KVCacheManagerV2.get_index_v_buffer
MiniMaxM3KVCacheManagerV2.has_index_value
MiniMaxM3KVCacheManagerV2.uses_hybrid_nvfp4_kv_cache
MiniMaxM3SparseIndexCache.__init__
MiniMaxM3SparseIndexCache.get_index_k_buffer
MiniMaxM3SparseIndexCache.get_index_v_buffer
MiniMaxM3SparseIndexCache.has_index_value
MiniMaxM3SparseIndexCache.set_index_k
MiniMaxM3SparseIndexCache.set_index_v
get_minimax_m3_kv_cache_manager_cls
```

### `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/msa_scatter.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
_fused_nvfp4_paged_scatter_kernel
_fused_paged_scatter_kernel
```

### `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/kernels/triton_sparse_decode.py`

库存函数 14；已覆盖 9；未覆盖 5。

```text
_dequant_nvfp4_rows
_e2m1x2_to_f16x2
_gqa_sparse_decode_kernel
_gqa_sparse_decode_pair_kernel
_merge_topk_attn_out_kernel
```

### `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py`

库存函数 56；已覆盖 55；未覆盖 1。

```text
MiniMaxM3MsaSparseAttention.support_fused_rope
```

### `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/triton_backend.py`

库存函数 17；已覆盖 1；未覆盖 16。

```text
MiniMaxM3SparseRuntimeBackend.forward
MiniMaxM3SparseRuntimeBackend.forward_sparse
MiniMaxM3SparseRuntimeBackend.support_fused_rope
MiniMaxM3TritonSparseAttention.forward
MiniMaxM3TritonSparseAttention.write_caches
_compute_index_attn_chunk_q
_compute_sparse_gqa_chunk_q
_gather_paged_batched
_index_attention_and_select
_scatter_topk_to_block_mask
_sparse_chunk_budget_bytes
_sparse_gqa_masked
_write_main_kv_slots
_write_main_kv_slots_to_pool
minimax_m3_sparse_decode
minimax_m3_sparse_prefill
```

### `tensorrt_llm/_torch/attention/backends/sparse/mixed_precision_high_rows.py`

库存函数 10；已覆盖 1；未覆盖 9。

```text
_copy_recorded_rows_kernel
_mixed_precision_high_compressed_row_mask_kernel
_mixed_precision_high_pool
_overlay_claim
_overlay_mixed_precision_high_rows_kernel
_store_mixed_precision_high_rows_kernel
overlay_mixed_precision_high_rows
store_mixed_precision_high_compressed_rows
store_mixed_precision_high_rows
```

### `tensorrt_llm/_torch/attention/backends/sparse/offload.py`

库存函数 36；已覆盖 35；未覆盖 1。

```text
SparseOffloadCacheMixin._sparse_offload_nvfp4_head_dim
```

### `tensorrt_llm/_torch/attention/backends/sparse/offload_kernels.py`

库存函数 9；已覆盖 5；未覆盖 4。

```text
_gather_sparse_batch_rows_kernel
_merge_sparse_arena_indices_kernel
_select_sparse_history_units_kernel
sparse_arena_rows
```

### `tensorrt_llm/_torch/attention/backends/sparse/params.py`

库存函数 9；已覆盖 6；未覆盖 3。

```text
MTPIndexShareMetadata.set_in_mtp_draft_loop
MTPIndexShareMetadata.set_mtp_num_accepted
MTPIndexShareMetadata.set_skip_topk
```

### `tensorrt_llm/_torch/attention/backends/sparse/qsa/indexer.py`

库存函数 13；已覆盖 12；未覆盖 1。

```text
QSAIndexer.project
```

### `tensorrt_llm/_torch/attention/backends/sparse/qsa/kernels.py`

库存函数 32；已覆盖 17；未覆盖 15。

```text
_expand_qsa_block_indices_kernel
_qsa_decode_pre_indexer_kernel
_qsa_decode_token_mapping_kernel
_qsa_gemma_norm_rope
_qsa_generation_token_mapping_kernel
_qsa_merge_splitk_kernel
_qsa_mtp_reuse_topk_kernel
_qsa_paged_index_scores_kernel
_qsa_paged_index_scores_prefetch_tile
_qsa_paged_index_scores_qtile_kernel
_qsa_paged_kv_store_kernel
_qsa_paged_sparse_gqa_kernel
_qsa_paged_sparse_gqa_splitk_kernel
_qsa_prefill_compress_kernel
_qsa_unscale_block_table_kernel
```

### `tensorrt_llm/_torch/attention/backends/sparse/qsa/params.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
QSASparseParams.indices_block_size
```

### `tensorrt_llm/_torch/attention/backends/sparse/qsa/prims_ts_indexer/kernel.py`

库存函数 51；已覆盖 34；未覆盖 17。

```text
GmemScoreResource.init_store_state
GmemScoreResource.stage
SmemKResource._init_state
SmemKResource.build_descriptor
SmemKResource.init_load_state
SmemKResource.init_mma_state
SmemKResource.tma_load
SmemQResource._init_state
SmemQResource.build_descriptor
SmemQResource.init_load_state
SmemQResource.init_mma_state
SmemQResource.tma_load
TmemSResource._init_tmem_state
TmemSResource.init_accumulator_state
TmemSResource.init_store_state
TmemSResource.load_subtile
TmemSResource.mma
```

### `tensorrt_llm/_torch/attention/backends/sparse/rocket/cache_manager.py`

库存函数 30；已覆盖 26；未覆盖 4。

```text
RocketKVCacheManager.free_resources
RocketKVCacheManager.get_cache_size_per_token
RocketKVCacheManager.prepare_resources
RocketKVCacheManager.update_resources
```

### `tensorrt_llm/_torch/attention/backends/sparse/rocket/kernels.py`

库存函数 32；已覆盖 14；未覆盖 18。

```text
_bitonic_merge
_compare_and_swap
_index_gather_kernel
argsort
bmm_kernel
convert_to_sortable_uint16
extract_bin_idx
flatten_to_batch_kernel
is_partial_match
process_histogram_step
rocket_batch_to_flatten_kernel
rocket_paged_kt_cache_bmm_kernel
rocket_qk_split_kernel
rocket_reduce_scores_kernel
rocket_update_kt_cache_ctx_kernel
rocket_update_kt_cache_gen_kernel
softmax_kernel
topk_kernel
```

### `tensorrt_llm/_torch/attention/backends/sparse/skip_softmax/params.py`

库存函数 21；已覆盖 19；未覆盖 2。

```text
SkipSoftmaxFormula.parse_from_ckpt_sparse_attention_config
_looks_like_skip_softmax_group
```

### `tensorrt_llm/_torch/attention/backends/triton_prefill.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
_fwd_kernel
_tanh
```

### `tensorrt_llm/_torch/attention/backends/trtllm.py`

库存函数 148；已覆盖 124；未覆盖 24。

```text
TrtllmAttention._build_context_plan
TrtllmAttention._context_plan_fmha_for
TrtllmAttention._context_plan_matches
TrtllmAttention._context_plan_sig
TrtllmAttention._is_plain_context_call
TrtllmAttention._run_context_plan
TrtllmAttention.is_sm_version_trtllm_gen_kernel
TrtllmAttention.rope_base
TrtllmAttention.rope_long_m_scale
TrtllmAttention.rope_max_positions
TrtllmAttention.rope_original_max_positions
TrtllmAttention.rope_scale
TrtllmAttention.rope_scale_type
TrtllmAttention.rope_short_m_scale
TrtllmAttentionMetadata._invalidate_mla_scheduler_buffers.<locals>.locality_domain_from_splits
TrtllmAttentionMetadata.build_locality_domain_block_offsets
TrtllmAttentionMetadata.decode_graph_variant
TrtllmAttentionMetadata.set_decode_graph_variant
_build_locality_domain_slices
_get_generation_only_token_splits
_locality_domain_cat_slices
_resize_locality_domain_workspaces_from_domain0
_scatter_locality_domain_rows
_slice_locality_domain_cu_seqlens
```

### `tensorrt_llm/_torch/attention/backends/utils.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
append_mla_latent_cache_generation_cuda_graph_safe
```

### `tensorrt_llm/_torch/attention/backends/vanilla.py`

库存函数 23；已覆盖 21；未覆盖 2。

```text
VanillaAttention._single_request_sparse_attn_predict
VanillaAttention._single_request_sparse_kv_predict
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/dspark/attention.py`

库存函数 20；已覆盖 19；未覆盖 1。

```text
DSparkAttention.wrapper
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/dspark/attention_kernel.py`

库存函数 34；已覆盖 10；未覆盖 24。

```text
DSparkAttentionKernel.__call__
DSparkAttentionKernel._compute_grid
DSparkAttentionKernel._create_tile_scheduler
DSparkAttentionKernel._tmem_load_partition
DSparkAttentionKernel.compute
DSparkAttentionKernel.correction
DSparkAttentionKernel.epilogue
DSparkAttentionKernel.exchange_p_cor_metadata
DSparkAttentionKernel.get_cache_seq_len
DSparkAttentionKernel.get_window_end_pos
DSparkAttentionKernel.get_window_valid_len
DSparkAttentionKernel.is_score_valid
DSparkAttentionKernel.kernel
DSparkAttentionKernel.load_history
DSparkAttentionKernel.load_page_table
DSparkAttentionKernel.load_tma
DSparkAttentionKernel.load_tma_qk_one_k_tile
DSparkAttentionKernel.load_tma_v_one_k_tile
DSparkAttentionKernel.make_paged_tiled_tma_atom
DSparkAttentionKernel.mma
DSparkAttentionKernel.mma_pv
DSparkAttentionKernel.mma_qk
DSparkAttentionKernel.rescale
DSparkAttentionKernel.softmax
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/dspark/draft_routing.py`

库存函数 11；已覆盖 6；未覆盖 5。

```text
DSparkDraftRoutingKernel.__call__
DSparkDraftRoutingKernel.kernel
DSparkDraftRoutingKernel.quantize
DSparkDraftRoutingKernel.route
DSparkDraftRoutingKernel.score
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/dspark/generation_prepare.py`

库存函数 43；已覆盖 28；未覆盖 15。

```text
DSparkAcceptPrepareKernel.__call__
DSparkAcceptPrepareKernel.kernel
DSparkArgmaxPartialKernel.__call__
DSparkArgmaxPartialKernel.kernel
DSparkCaptureMeanKernel.__call__
DSparkCaptureMeanKernel.kernel
DSparkDraftBlockInputsKernel.__call__
DSparkDraftBlockInputsKernel.kernel
DSparkGenerationPrepareKernel.__call__
DSparkGenerationPrepareKernel.kernel
DSparkNextNewTokensKernel.__call__
DSparkNextNewTokensKernel.kernel
DSparkStrictAcceptKernel.__call__
DSparkStrictAcceptKernel.accept
DSparkStrictAcceptKernel.kernel
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/dspark/markov_chain.py`

库存函数 16；已覆盖 13；未覆盖 3。

```text
DSparkMarkovChainKernel.__call__
DSparkMarkovChainKernel._exchange
DSparkMarkovChainKernel.kernel
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/dspark/rmsnorm_rope.py`

库存函数 19；已覆盖 6；未覆盖 13。

```text
DSparkBackfillKernel.__call__
DSparkBackfillKernel.kernel
DSparkConcatFP8QuantKernel.__call__
DSparkConcatFP8QuantKernel.kernel
DSparkRMSNormFP8QuantKernel.__call__
DSparkRMSNormFP8QuantKernel.kernel
DSparkRMSNormRoPECacheWriteKernel.__call__
DSparkRMSNormRoPECacheWriteKernel.kernel
DSparkRMSNormRoPEDraftBlockKernel.__call__
DSparkRMSNormRoPEDraftBlockKernel.kernel
DSparkRMSNormRoPEKernel.__call__
DSparkRMSNormRoPEKernel.kernel
DSparkRMSNormRoPEKernel.store_normalized_row
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/dspark/tp_argmax.py`

库存函数 19；已覆盖 13；未覆盖 6。

```text
DSparkGlobalArgmaxKernel.__call__
DSparkGlobalArgmaxKernel.kernel
DSparkLocalCandidatesKernel.__call__
DSparkLocalCandidatesKernel.kernel
DSparkMaskedEmbeddingKernel.__call__
DSparkMaskedEmbeddingKernel.kernel
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/gqa_kv_replica_partial_attn.py`

库存函数 53；已覆盖 39；未覆盖 14。

```text
GqaKvReplicaPartialAttnKernel.__call__
GqaKvReplicaPartialAttnKernel._combine
GqaKvReplicaPartialAttnKernel._gemm2
GqaKvReplicaPartialAttnKernel._ll_store
GqaKvReplicaPartialAttnKernel._merge
GqaKvReplicaPartialAttnKernel._plan
GqaKvReplicaPartialAttnKernel._plan_store
GqaKvReplicaPartialAttnKernel._plan_two
GqaKvReplicaPartialAttnKernel._plan_two_store
GqaKvReplicaPartialAttnKernel._plan_wide
GqaKvReplicaPartialAttnKernel._produce
GqaKvReplicaPartialAttnKernel._softmax_tile
GqaKvReplicaPartialAttnKernel.kernel
GqaKvReplicaPartialAttnKernel.kernel_two
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/gqa_split_kv_decode_attn.py`

库存函数 50；已覆盖 35；未覆盖 15。

```text
GqaSplitKvDecodeAttnKernel.__call__
GqaSplitKvDecodeAttnKernel._fold_append
GqaSplitKvDecodeAttnKernel._fold_cs_entry
GqaSplitKvDecodeAttnKernel._fold_cs_prefill
GqaSplitKvDecodeAttnKernel._fold_load_row
GqaSplitKvDecodeAttnKernel._fold_load_static
GqaSplitKvDecodeAttnKernel._fold_q
GqaSplitKvDecodeAttnKernel._fold_row_fp8
GqaSplitKvDecodeAttnKernel._fold_tables
GqaSplitKvDecodeAttnKernel._fold_weights
GqaSplitKvDecodeAttnKernel._gemm2
GqaSplitKvDecodeAttnKernel._produce
GqaSplitKvDecodeAttnKernel._softmax_tile
GqaSplitKvDecodeAttnKernel.kernel
_ld_global_s64
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/kimi_k3_kda/fused_prefill_conv.py`

库存函数 39；已覆盖 23；未覆盖 16。

```text
KdaConv.__call__
KdaConv.issue
KdaConv.kernel
KdaConv.load8
KdaConv.loadw
KdaConv.meta
KdaConv.redo
KdaConv.runs
KdaConv.state_addr
KdaConv.step
KdaConv.store
KdaConv.storew
KdaConv.tile
_pick_tokens
fma_rn
smem_u32
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/kimi_k3_kda/kda_chunked_bt16.py`

库存函数 195；已覆盖 181；未覆盖 14。

```text
CompiledKDA.__call__
_occupancy_pick_route
_parse_args
_state_kv_to_vk
_state_vk_to_kv
_uniform_cu_seqlens
checkpoint_read_done_arrive
checkpoint_read_done_wait
fadd2
mul_f16x2
raw_f32_s128_smem_index
run
softplus_log2_f32
verify
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/kimi_k3_kda/kda_chunked_bt16_small_bh.py`

库存函数 29；已覆盖 18；未覆盖 11。

```text
ChunkKdaFwdK1.__call__
ChunkKdaFwdK1.add_identity
ChunkKdaFwdK1.compute_gate_tile
ChunkKdaFwdK1.get_row
ChunkKdaFwdK1.kernel
ChunkKdaFwdK1.qk_l2_norm_row
ChunkKdaFwdK1.sigmoid
ChunkKdaFwdK1.transpose
ChunkKdaFwdK2.__call__
ChunkKdaFwdK2.kernel
clear_compilation_cache
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/kimi_k3_kda/kda_mtp_decode.py`

库存函数 12；已覆盖 10；未覆盖 2。

```text
read_globaltimer
swz128
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/minimax_m3_index_decode_score.py`

库存函数 7；已覆盖 5；未覆盖 2。

```text
IndexDecodeScoreKernel.__call__
IndexDecodeScoreKernel.kernel
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/mixed_precision_kv.py`

库存函数 10；已覆盖 6；未覆盖 4。

```text
_NVFP4Codec.decode
_NVFP4Codec.encode
_PagedMixedPrecisionKV.__call__
_PagedMixedPrecisionKV.convert_paged_kv
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/mla/mla_decode_fp16.py`

库存函数 54；已覆盖 11；未覆盖 43。

```text
BlackwellMultiHeadLatentAttentionForwardFP16.__call__
BlackwellMultiHeadLatentAttentionForwardFP16._run
BlackwellMultiHeadLatentAttentionForwardFP16._run.<locals>._flatten_compact_q_rows
BlackwellMultiHeadLatentAttentionForwardFP16._run.<locals>._flatten_fixed_q_rows
BlackwellMultiHeadLatentAttentionForwardFP16._run.<locals>._make_compact_unpacked_o
BlackwellMultiHeadLatentAttentionForwardFP16._tmem_load_partition
BlackwellMultiHeadLatentAttentionForwardFP16.compute
BlackwellMultiHeadLatentAttentionForwardFP16.correction
BlackwellMultiHeadLatentAttentionForwardFP16.epilogue
BlackwellMultiHeadLatentAttentionForwardFP16.exchange_p_cor_metadata
BlackwellMultiHeadLatentAttentionForwardFP16.get_k_tile_count
BlackwellMultiHeadLatentAttentionForwardFP16.get_query_kv_bound
BlackwellMultiHeadLatentAttentionForwardFP16.get_split_kv
BlackwellMultiHeadLatentAttentionForwardFP16.get_split_kv_simplified
BlackwellMultiHeadLatentAttentionForwardFP16.get_valid_q_rows
BlackwellMultiHeadLatentAttentionForwardFP16.get_workspace_size
BlackwellMultiHeadLatentAttentionForwardFP16.initialize_workspace
BlackwellMultiHeadLatentAttentionForwardFP16.issue_q_tma_load
BlackwellMultiHeadLatentAttentionForwardFP16.load_page_table
BlackwellMultiHeadLatentAttentionForwardFP16.load_tma
BlackwellMultiHeadLatentAttentionForwardFP16.load_tma_qk_one_k_tile
BlackwellMultiHeadLatentAttentionForwardFP16.load_tma_v_one_k_tile
BlackwellMultiHeadLatentAttentionForwardFP16.make_paged_tiled_tma_atom
BlackwellMultiHeadLatentAttentionForwardFP16.mma
BlackwellMultiHeadLatentAttentionForwardFP16.mma_pv
BlackwellMultiHeadLatentAttentionForwardFP16.mma_qk
BlackwellMultiHeadLatentAttentionForwardFP16.reduction_kernel
BlackwellMultiHeadLatentAttentionForwardFP16.rescale
BlackwellMultiHeadLatentAttentionForwardFP16.run_with_softmax_stats
BlackwellMultiHeadLatentAttentionForwardFP16.softmax
BlackwellMultiHeadLatentAttentionForwardFP16.split_kv_kernel
parse_comma_separated_ints
parse_mma_tiler
run
run.<locals>.compact_tensor
run.<locals>.create_block_split_kvs
run.<locals>.create_cache_seqs
run.<locals>.create_data_tensor
run.<locals>.create_page_table
run.<locals>.create_stats
run.<locals>.create_workspace
run.<locals>.generate_tensors
run.<locals>.torch_reference_mla
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/mla/mla_decode_fp8.py`

库存函数 58；已覆盖 10；未覆盖 48。

```text
BlackwellMultiHeadLatentAttentionForwardFP8.__call__
BlackwellMultiHeadLatentAttentionForwardFP8._run
BlackwellMultiHeadLatentAttentionForwardFP8._run.<locals>._flatten_compact_q_rows
BlackwellMultiHeadLatentAttentionForwardFP8._run.<locals>._flatten_fixed_q_rows
BlackwellMultiHeadLatentAttentionForwardFP8._run.<locals>._make_compact_unpacked_o
BlackwellMultiHeadLatentAttentionForwardFP8._tmem_load_partition
BlackwellMultiHeadLatentAttentionForwardFP8.compute
BlackwellMultiHeadLatentAttentionForwardFP8.correction
BlackwellMultiHeadLatentAttentionForwardFP8.epilogue
BlackwellMultiHeadLatentAttentionForwardFP8.exchange_p_cor_metadata
BlackwellMultiHeadLatentAttentionForwardFP8.get_k_tile_count
BlackwellMultiHeadLatentAttentionForwardFP8.get_query_kv_bound
BlackwellMultiHeadLatentAttentionForwardFP8.get_split_kv
BlackwellMultiHeadLatentAttentionForwardFP8.get_split_kv_simplified
BlackwellMultiHeadLatentAttentionForwardFP8.get_valid_q_rows
BlackwellMultiHeadLatentAttentionForwardFP8.get_workspace_size
BlackwellMultiHeadLatentAttentionForwardFP8.init_p_cor_metadata
BlackwellMultiHeadLatentAttentionForwardFP8.initialize_workspace
BlackwellMultiHeadLatentAttentionForwardFP8.issue_q_tma_load
BlackwellMultiHeadLatentAttentionForwardFP8.load_other_group_metadata
BlackwellMultiHeadLatentAttentionForwardFP8.load_tma_qk
BlackwellMultiHeadLatentAttentionForwardFP8.load_tma_qk_one_k_tile
BlackwellMultiHeadLatentAttentionForwardFP8.load_tma_v
BlackwellMultiHeadLatentAttentionForwardFP8.load_tma_v_one_k_tile
BlackwellMultiHeadLatentAttentionForwardFP8.make_paged_tiled_tma_atom
BlackwellMultiHeadLatentAttentionForwardFP8.mma
BlackwellMultiHeadLatentAttentionForwardFP8.mma_pv
BlackwellMultiHeadLatentAttentionForwardFP8.mma_pv_warp_body
BlackwellMultiHeadLatentAttentionForwardFP8.mma_qk
BlackwellMultiHeadLatentAttentionForwardFP8.reduction_kernel
BlackwellMultiHeadLatentAttentionForwardFP8.rescale
BlackwellMultiHeadLatentAttentionForwardFP8.run_with_softmax_stats
BlackwellMultiHeadLatentAttentionForwardFP8.softmax
BlackwellMultiHeadLatentAttentionForwardFP8.softmax_advance_to_next_group
BlackwellMultiHeadLatentAttentionForwardFP8.split_kv_kernel
BlackwellMultiHeadLatentAttentionForwardFP8.store_p_cor_row_sum
parse_comma_separated_ints
parse_mma_tiler
run
run.<locals>.compact_tensor
run.<locals>.create_block_split_kvs
run.<locals>.create_cache_seqs
run.<locals>.create_data_tensor
run.<locals>.create_page_table
run.<locals>.create_stats
run.<locals>.create_workspace
run.<locals>.generate_tensors
run.<locals>.torch_reference_mla
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/mla/mla_helpers.py`

库存函数 20；已覆盖 19；未覆盖 1。

```text
get_variable_query_tile_info
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/paged_mqa_logits/fp4_paged_mqa_logits.py`

库存函数 41；已覆盖 18；未覆盖 23。

```text
FP4MQALogitsKernel.__call__
FP4MQALogitsKernel._emit_block_max
FP4MQALogitsKernel._flush_cand_window
FP4MQALogitsKernel._flush_cand_window_bucketed
FP4MQALogitsKernel._flush_hit_agg
FP4MQALogitsKernel._flush_seed_counts
FP4MQALogitsKernel.kernel
_ad
_fm
_mx
_pk
_subview
_up
add_bf16x2
add_f16x2
fma_bf16x2
fma_f16x2
max_bf16x2
max_f16x2
pack_bf16x2
pack_f16x2
unpack_bf16x2
unpack_f16x2
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/paged_mqa_logits/fp8_paged_mqa_logits.py`

库存函数 21；已覆盖 15；未覆盖 6。

```text
FP8MQALogitsFusedWeightScaleKernel.__call__
FP8MQALogitsFusedWeightScaleKernel.__init__
FP8MQALogitsKernel.__call__
FP8MQALogitsKernel._launch_body
FP8MQALogitsKernel.kernel
_subview
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/paged_mqa_logits/mqa_logits_util.py`

库存函数 23；已覆盖 10；未覆盖 13。

```text
KernelLaunch.run
PagedMQALogitsMetadataKernel.__call__
PagedMQALogitsMetadataKernel.__init__
PagedMQALogitsMetadataKernel.kernel
align
arbitrate_elem_tolerance
calc_diff
choose_atom_split
compile_paged_mqa_logits_metadata_kernel
compute_schedule_metadata
frontend_cache_token
get_paged_mqa_logits_metadata_cute_dsl
verify_schedule_meta
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/paged_mqa_logits/sparse_logits_topk_dsl.py`

库存函数 18；已覆盖 12；未覆盖 6。

```text
SparseLogitsTopKKernel.__call__
SparseLogitsTopKKernel.bucket
SparseLogitsTopKKernel.flags
SparseLogitsTopKKernel.kernel
SparseLogitsTopKKernel.radix
SparseLogitsTopKKernel.scan
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/sparse_mla/kernel.py`

库存函数 21；已覆盖 20；未覆盖 1。

```text
pack_e4m3x4
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/sparse_mla/kernel_gb300.py`

库存函数 41；已覆盖 12；未覆盖 29。

```text
SparseMlaAttentionGB300.__call__
SparseMlaAttentionGB300.correction
SparseMlaAttentionGB300.gather
SparseMlaAttentionGB300.kernel
SparseMlaAttentionGB300.load_q
SparseMlaAttentionGB300.mma_pv
SparseMlaAttentionGB300.mma_qk
SparseMlaAttentionGB300.num_tiles
SparseMlaAttentionGB300.pv_tile
SparseMlaAttentionGB300.qk
SparseMlaAttentionGB300.rope_mxfp8_block
SparseMlaAttentionGB300.softmax
SparseMlaAttentionGB300.tile_of
SparseMlaAttentionGB300.wait
SparseMlaAttentionGB300.warp_arrive
absmax_tree
bf16_round
f32_op
ld_global_f32
ld_shared_bf16x8
mx_exponent
mx_reciprocal
mx_scale_offset
rope_inv_pair
select_f32
st_global_b16
st_global_b32
tma_store_tile3
warp_sync
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/adapter/_topk_capacity.py`

库存函数 16；已覆盖 15；未覆盖 1。

```text
_CapacityStepInputs.stage_payload
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/adapter/kernels/selective_prefill_metadata.py`

库存函数 4；已覆盖 1；未覆盖 3。

```text
SelectivePrefillMetadata.__call__
SelectivePrefillMetadata.publish
SelectivePrefillMetadata.validate
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/adapter/kernels/topk_metadata.py`

库存函数 6；已覆盖 1；未覆盖 5。

```text
TopKCapacityRefreshKernel._block_inclusive_sum
TopKCapacityRefreshKernel._block_or
TopKCapacityRefreshKernel._prefix_code
TopKCapacityRefreshKernel._total_capacity_code
TopKCapacityRefreshKernel.publish_kernel
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/adapter/selective_prefill.py`

库存函数 24；已覆盖 16；未覆盖 8。

```text
_CapacitySelectiveTopKPlan._bind_engine
_CapacitySelectiveTopKPlan._bind_engine.<locals>.rebind
_CapacitySelectiveTopKPlan._layout_needs_engine
_CapacitySelectiveTopKPlan.active_shape
_CapacitySelectiveTopKPlan.clear_pools
_CapacitySelectiveTopKPlan.error
_CapacitySelectiveTopKPlan.input_buffers
_CapacitySelectiveTopKPlan.repair_error
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/interface.py`

库存函数 12；已覆盖 4；未覆盖 8。

```text
FP8MQATopKWrapper.active_shape
FP8MQATopKWrapper.clear_pools
FP8MQATopKWrapper.compression_ratio
FP8MQATopKWrapper.error
FP8MQATopKWrapper.input_buffers
FP8MQATopKWrapper.math_mode
FP8MQATopKWrapper.repair_error
FP8MQATopKWrapper.strategy
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/mqa_logits_util.py`

库存函数 6；已覆盖 3；未覆盖 3。

```text
align
calc_diff
ceil_div
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/api.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
ThrowawayTopK.run
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/crossing_precounted.py`

库存函数 6；已覆盖 3；未覆盖 3。

```text
Crossing.__call__
Crossing.kernel
bucket_for
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/crossing_precounted_mxfp4.py`

库存函数 6；已覆盖 3；未覆盖 3。

```text
Crossing.__call__
Crossing.kernel
bucket_for
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp4_queue_recovery.py`

库存函数 10；已覆盖 5；未覆盖 5。

```text
FP4PrefillMQALogitsKernel.__call__
FP4PrefillMQALogitsKernel._gemm_body
FP4PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP4PrefillMQALogitsKernel.kernel
publish_bank
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp4_safe_filter.py`

库存函数 15；已覆盖 6；未覆盖 9。

```text
FP4PrefillMQALogitsKernel.__call__
FP4PrefillMQALogitsKernel._gemm_body
FP4PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP4PrefillMQALogitsKernel.kernel
_uniform_weight
interleaved_weighted_sum
predicated_store32
predicated_store64
utccp_interleaved_qsf
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp8_glmfull_fulltile_filter.py`

库存函数 10；已覆盖 2；未覆盖 8。

```text
FP8PrefillMQALogitsKernel.__call__
FP8PrefillMQALogitsKernel._gemm_body
FP8PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP8PrefillMQALogitsKernel.kernel
_subview
predicated_store32
predicated_store64
publish_bank
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp8_queue_recovery.py`

库存函数 8；已覆盖 3；未覆盖 5。

```text
FP8PrefillMQALogitsKernel.__call__
FP8PrefillMQALogitsKernel._gemm_body
FP8PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP8PrefillMQALogitsKernel.kernel
publish_bank
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/fp8_safe_filter.py`

库存函数 11；已覆盖 6；未覆盖 5。

```text
FP8PrefillMQALogitsKernel.__call__
FP8PrefillMQALogitsKernel._gemm_body
FP8PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP8PrefillMQALogitsKernel.kernel
interleaved_partial
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/integer_relu_fast.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
mixed_integer_relu4
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/native_heads.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
native_head_fma4
native_weight_pair
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/policy_glm_short.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
block_samples
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/policy_q4_128.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
block_samples
qwen_cached_sample_key
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/policy_qwen.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
block_samples
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/qsa_publish.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
publish
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/recovery_selection.py`

库存函数 7；已覆盖 4；未覆盖 3。

```text
select_candidates
select_narrow
select_or_prepare_retry
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/sampler_upper_fp4.py`

库存函数 9；已覆盖 5；未覆盖 4。

```text
FP4PrefillMQALogitsKernel.__call__
FP4PrefillMQALogitsKernel._gemm_body
FP4PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP4PrefillMQALogitsKernel.kernel
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/sampler_upper_fp8.py`

库存函数 8；已覆盖 3；未覆盖 5。

```text
FP8PrefillMQALogitsKernel.__call__
FP8PrefillMQALogitsKernel._gemm_body
FP8PrefillMQALogitsKernel._gemm_body.<locals>._kv_lo_for_block
FP8PrefillMQALogitsKernel.kernel
publish_bank
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/selection.py`

库存函数 29；已覆盖 11；未覆盖 18。

```text
append
append_cached
block_samples
finish_recovery
histogram_window
load_banked
load_spilled
overflow_view
propose
propose_cached
publish_bank
publish_private
runtime_samples
sample_tile
select
select_candidates
select_narrow
select_or_prepare_retry
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/throwaway_topk/prefill/weighted_heads.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
weighted_heads_cached
```

### `tensorrt_llm/_torch/attention/kernels/fused_qk_norm_rope_gate.py`

库存函数 20；已覆盖 16；未覆盖 4。

```text
_fp8_like_cxx
_fused_qkv_gemma_rmsnorm_rope_gate_kernel
_fused_sigmoid_mul_kernel
_qkv_rope_gate_fp8_append_kernel
```

### `tensorrt_llm/_torch/attention/mla.py`

库存函数 65；已覆盖 64；未覆盖 1。

```text
MLA.resmooth_parameters
```

### `tensorrt_llm/_torch/attention/qk_norm_attention.py`

库存函数 15；已覆盖 12；未覆盖 3。

```text
compute_yarn_parameters.<locals>.find_correction_dim
compute_yarn_parameters.<locals>.find_correction_range
compute_yarn_parameters.<locals>.get_mscale
```

### `tensorrt_llm/_torch/autotuner.py`

库存函数 101；已覆盖 96；未覆盖 5。

```text
AutoTuner._cudaGetErrorEnum
AutoTunerProfilingCache.__getitem__
OptimizationProfile.get_hash_key
TunableRunner.forward
TunableRunner.get_valid_tactics
```

### `tensorrt_llm/_torch/bolt_profiling.py`

库存函数 11；已覆盖 2；未覆盖 9。

```text
_addr_in_lib
_call_at_offset
_call_via_address
_call_via_dlsym
_clear_offsets_from_file
_iter_loaded_target_libs
_nm_tools
_self_check
_symbol_offset
```

### `tensorrt_llm/_torch/compilation/backend.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
Backend.bypass_optimization
Backend.enable_optimization
```

### `tensorrt_llm/_torch/compilation/multi_stream/auto_multi_stream.py`

库存函数 17；已覆盖 16；未覆盖 1。

```text
dump_dag_as_dot
```

### `tensorrt_llm/_torch/compilation/patterns/ar_residual_norm.py`

库存函数 56；已覆盖 38；未覆盖 18。

```text
register_ar_residual_norm.<locals>.empty_pattern
register_ar_residual_norm_fp4_quant.<locals>.empty_pattern
register_ar_residual_norm_fp4_quant.<locals>.extra_check
register_ar_residual_norm_fp4_quant.<locals>.target_pattern
register_ar_residual_norm_fp8_quant.<locals>.empty_pattern
register_ar_residual_norm_out_fp4_quant.<locals>.empty_pattern
register_ar_residual_norm_out_fp8_quant.<locals>.empty_pattern
register_ar_residual_norm_out_fp8_quant.<locals>.target_pattern
register_ar_residual_norm_out_fp8_quant.<locals>.target_pattern_without_scale
register_ub_patterns.<locals>.insert_copy_for_graph_output.<locals>.empty_copy_for_graph_output_pattern
register_ub_patterns.<locals>.register_convert_supported_ar_to_ub.<locals>.empty_convert_supported_ar_to_ub
register_ub_patterns.<locals>.register_ub_finalize_patterns.<locals>.empty_finalize_pattern
register_ub_patterns.<locals>.register_ub_prologue_patterns.<locals>.register_add_prologue.<locals>.empty_add_prologue_pattern
register_ub_patterns.<locals>.register_ub_prologue_patterns.<locals>.register_mm_prologue.<locals>.empty_mm_prologue_pattern
register_ub_patterns.<locals>.register_ub_prologue_patterns.<locals>.register_nvfp4_gemm_prologue.<locals>.empty_nvfp4_gemm_bias_prologue_pattern
register_ub_patterns.<locals>.register_ub_prologue_patterns.<locals>.register_nvfp4_gemm_prologue.<locals>.empty_nvfp4_gemm_prologue_pattern
register_ub_patterns.<locals>.register_ub_prologue_patterns.<locals>.register_nvfp4_gemm_prologue.<locals>.target_nvfp4_gemm_bias_prologue_pattern
register_ub_patterns.<locals>.register_ub_prologue_patterns.<locals>.register_scaled_mm_prologue.<locals>.empty_scaled_mm_prologue_pattern
```

### `tensorrt_llm/_torch/compilation/patterns/residual_add_norm.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
register_add_norm.<locals>.empty_pattern
register_add_norm_quant.<locals>.empty_pattern
```

### `tensorrt_llm/_torch/compilation/utils.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
get_symint_val
```

### `tensorrt_llm/_torch/configs/cosmos3.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
Cosmos3Config.from_pretrained
```

### `tensorrt_llm/_torch/configs/deepseek_v41.py`

库存函数 37；已覆盖 34；未覆盖 3。

```text
DeepseekV41Config.write_layer_plan_dump
DeepseekV41TextConfig.layer_descriptor
DeepseekV41TextConfig.write_layer_plan_dump
```

### `tensorrt_llm/_torch/configs/kimi_linear.py`

库存函数 6；已覆盖 1；未覆盖 5。

```text
KimiLinearConfig.is_full_attn_layer
KimiLinearConfig.is_kda_layer
KimiLinearConfig.is_linear_attn
KimiLinearConfig.is_mla
KimiLinearConfig.is_moe
```

### `tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py`

库存函数 21；已覆盖 11；未覆盖 10。

```text
_register_fake.<locals>._.<locals>.create_output
_register_fake.<locals>.allgather_list
_register_fake.<locals>.allgather_list.<locals>.create_output_tensor
_register_fake.<locals>.block_causal_layout_
_register_fake.<locals>.copy_kv_slots_
_register_fake.<locals>.group_rms_norm_base
_register_fake.<locals>.group_rms_norm_heuristic
_register_fake.<locals>.group_rms_norm_large_batch
_register_fake.<locals>.minimax_m3_fp8_indexer_qk_norm_rope_fake
_register_fake.<locals>.reducescatter
```

### `tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py`

库存函数 385；已覆盖 336；未覆盖 49。

```text
CuteDSLBlockScaledRubinLinear.__eq__
CuteDSLBlockScaledRubinLinear.__hash__
CuteDSLDsv4QbGemmFusedRmsnormRopeQuantRunner._fallback_tactic
CuteDSLDsv4QbGemmFusedRmsnormRopeQuantRunner.forward
CuteDSLDsv4QbGemmFusedRmsnormRopeQuantRunner.forward.<locals>._from_dlpack
CuteDSLDsv4QbGemmFusedRmsnormRopeQuantRunner.get_valid_tactics
CuteDSLDsv4QbGemmFusedRmsnormRopeQuantRunner.unique_id
CuteDSLFp8RubinPerTensorGemmRunner.forward
CuteDSLNVFP4BlackwellRunner.__eq__
CuteDSLNVFP4BlackwellRunner.__hash__
CuteDSLNVFP4BlackwellRunner.should_profile_tactic_in_subprocess
CuteDSLNVFP4DenseGemmFC2Runner.__eq__
CuteDSLNVFP4DenseGemmFC2Runner.__hash__
CuteDSLNVFP4DenseGemmSwigluRunner.__eq__
CuteDSLNVFP4DenseGemmSwigluRunner.__hash__
CuteDSLNVFP4SwigluFP4OutBlackwellRunner.get_valid_tactics
CuteDSLTopKDecodeRadixFilterSPMultiCTARunner.auto_cluster_size_truncate
CuteDSLTopKDecodeSinglePassMultiCTARunner._get_possible_chunk_sizes
Fc12FusedInputsHelper.inputs_pre_hook
GatherGroupedGemmInputsHelper._resize_locality_domain_outputs
GroupedGemmInputsHelper.infer_shape_input0_rows
GroupedGemmInputsHelper.infer_shape_input5_rows
Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner.get_valid_tactics
Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner.forward.<locals>._fc12_wrapper
Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner.get_valid_tactics
Sm107ContiguousGatherGroupedGemmSwigluFusionRunner._bf16_inputs_pre_hook
Sm107ContiguousGatherGroupedGemmSwigluFusionRunner.get_valid_tactics
Sm107ContiguousGroupedGemmFinalizeFusionRunner._bf16_inputs_pre_hook_finalize
Sm107ContiguousGroupedGemmFinalizeFusionRunner.get_valid_tactics
Sm107Mxfp8FusedFc12MoeRunner.forward.<locals>._fused_fc12_wrapper
_.<locals>.i32
_fake_single_b
_fake_single_b_zero
_gen_dsv4_qb_fusion_tuning_buckets
_log_nvmmh_fallback
_map_dsv4_qb_fusion_tuning_bucket
_prepare_dsv4_qb_fusion_tuning_inputs
_validate_16_byte_aligned_dense_tensor
_validate_cute_dsl_bf16_bmm_locality_domain_inputs
_validate_cute_dsl_bf16_gemm_locality_domain_inputs
cute_dsl_bf16_bmm_locality_domain_inplace_rubin
cute_dsl_bf16_bmm_locality_domain_inplace_rubin.<locals>.launch_partition
cute_dsl_bf16_gemm_locality_domain_inplace_rubin
cute_dsl_bf16_gemm_locality_domain_inplace_rubin.<locals>.launch_partition
cute_dsl_bf16_grouped_gemm_finalize_rubin
cute_dsl_mxfp8_gemm_inplace_rubin
cute_dsl_nvfp4_gemm_blackwell
cute_dsl_nvfp4_gemm_locality_domain_inplace_rubin.<locals>.launch_partition
warmup_cute_dsl_radix_topk_decode
```

### `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_custom_ops.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
_kda_prefill_conv_fake
_kda_prefill_fake
```

### `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_kda_mtp_ops.py`

库存函数 25；已覆盖 24；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/custom_ops/cute_dsl_mxfp8_shared_expert_custom_op.py`

库存函数 15；已覆盖 14；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/custom_ops/dspark_attention_custom_op.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
_
cute_dsl_dspark_context_wkv_gemm_rubin
```

### `tensorrt_llm/_torch/custom_ops/dspark_rmsnorm_rope_custom_op.py`

库存函数 24；已覆盖 23；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/custom_ops/fast_custom_op.py`

库存函数 7；已覆盖 5；未覆盖 2。

```text
FastCustomOp.__call__
FastCustomOp.python_impl
```

### `tensorrt_llm/_torch/custom_ops/mxfp8_gemm.py`

库存函数 24；已覆盖 21；未覆盖 3。

```text
FlashInferMXFP8GemmRunner.unique_id
Sm107MXFP8GemmRunner.unique_id
TrtllmMXFP8GemmRunner.unique_id
```

### `tensorrt_llm/_torch/custom_ops/nvfp4_sfc_finalize.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
_e4m3_encode
_k2_sfc_kernel
```

### `tensorrt_llm/_torch/custom_ops/router_logit_partials.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
RouterLogitPartials.split
```

### `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py`

库存函数 168；已覆盖 153；未覆盖 15。

```text
FP8RowwiseGemmRunner.get_valid_tactics
FlashInferMXFP8GemmRunner.get_valid_tactics
FlashInferMXFP8GemmRunner.unique_id
Fp4GemmAllreduceRunner.__init__
Fp4GemmAllreduceRunner.forward
Fp4GemmAllreduceRunner.get_valid_tactics
Fp4GemmAllreduceRunner.unique_id
MarlinNVFP4Runner.__init__
MarlinNVFP4Runner._prepare_marlin_weights
MarlinNVFP4Runner.forward
MarlinNVFP4Runner.get_valid_tactics
QuantizeE4M3PerTensorRunner._quantize_te
_fp8_block_scaling_gemm_sm100_constraint
flashinfer_mxfp8_gemm_cute_dsl
nvfp4_gemm_allreduce
```

### `tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py`

库存函数 101；已覆盖 97；未覆盖 4。

```text
FP4BlockScaleMoERunner._table_tactic
_
_TrtllmGenMoERunner._kernel_key
calculate_tile_tokens_dim
```

### `tensorrt_llm/_torch/cute_dsl_file_cache.py`

库存函数 22；已覆盖 20；未覆盖 2。

```text
KernelCache.__contains__
_file_digest
```

### `tensorrt_llm/_torch/cute_dsl_kernels/minimax_m3/embedding_firstnorm/op.py`

库存函数 6；已覆盖 1；未覆盖 5。

```text
DecodeEmbedding.__call__
DecodeEmbedding.__init__
_compile
arg
m3_embed_norm
```

### `tensorrt_llm/_torch/cute_dsl_kernels/minimax_m3/moe/m3_moe_kernel.py`

库存函数 62；已覆盖 59；未覆盖 3。

```text
_build_b1
_gtime
_smid
```

### `tensorrt_llm/_torch/cute_dsl_kernels/minimax_m3/moe/op.py`

库存函数 16；已覆盖 15；未覆盖 1。

```text
ar_push_desc
```

### `tensorrt_llm/_torch/cute_dsl_kernels/minimax_m3/mx_gemv/op.py`

库存函数 22；已覆盖 5；未覆盖 17。

```text
_arg
_compile
_dummy_timers
_extra_args
_mma_n
_placeholder_scales
_placeholder_words
_resolve
_smem_bytes
_views
freeze
frozen
is_compiled
m3_mx_gemv_bf16
m3_mx_gemv_swiglu
precompile
serves
```

### `tensorrt_llm/_torch/cute_dsl_utils.py`

库存函数 16；已覆盖 15；未覆盖 1。

```text
_GraphSafeDlpack.__dlpack_device__
```

### `tensorrt_llm/_torch/debug/debug_hook.py`

库存函数 31；已覆盖 28；未覆盖 3。

```text
DebuggerContext.get_current_model_loop_index
Filter.__init__
Filter.filter
```

### `tensorrt_llm/_torch/device_mesh.py`

库存函数 26；已覆盖 14；未覆盖 12。

```text
DeviceMeshTopologyImpl.__getstate__
DeviceMeshTopologyImpl.cp_group
DeviceMeshTopologyImpl.cp_group_name
DeviceMeshTopologyImpl.cp_group_pg
DeviceMeshTopologyImpl.cp_rank
DeviceMeshTopologyImpl.moe_ep_group
DeviceMeshTopologyImpl.moe_ep_group_pg
DeviceMeshTopologyImpl.moe_tp_group
DeviceMeshTopologyImpl.moe_tp_group_pg
DeviceMeshTopologyImpl.pp_group
DeviceMeshTopologyImpl.pp_group_pg
SingleProcessGroup.size
```

### `tensorrt_llm/_torch/disaggregation/base/agent.py`

库存函数 17；已覆盖 4；未覆盖 13。

```text
BaseTransferAgent.check_remote_descs
BaseTransferAgent.deregister_memory
BaseTransferAgent.get_local_agent_desc
BaseTransferAgent.invalidate_remote_agent
BaseTransferAgent.load_remote_agent
BaseTransferAgent.notify_sync_message
BaseTransferAgent.register_memory
BaseTransferAgent.shutdown
BaseTransferAgent.submit_transfer_requests
MemoryDescs.from_arrays
MemoryDescs.from_arrays_uniform_device
TransferStatus.is_completed
TransferStatus.wait
```

### `tensorrt_llm/_torch/disaggregation/base/backend.py`

库存函数 27；已覆盖 6；未覆盖 21。

```text
Attempt.cancel
Attempt.poll
Attempt.quiesced
Backend.capabilities
Backend.fetch
Backend.publish
Backend.shutdown
LayerGroupRows.__len__
LayerGroupRows.layer_group_index
LayerGroupRows.object_keys
LayerGroupRows.page_ordinals
LayerGroupRows.region_indices
LayerGroupRows.slots
SlotRegion.address
SlotRegion.layout_label
SlotRegion.memory_kind
SlotRegion.nbytes
SlotRegion.slot_bytes
SlotRegion.slot_count
View.groups
View.row_masks
```

### `tensorrt_llm/_torch/disaggregation/base/region.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
RegionExtractorBase.extract
RegionMapperBase.map
```

### `tensorrt_llm/_torch/disaggregation/base/role_layout.py`

库存函数 54；已覆盖 47；未覆盖 7。

```text
RoleLayoutV2.dtype
RoleLayoutV2.even
RoleLayoutV2.index_of
RoleLayoutV2.pieces
RoleLayoutV2.positions
RoleLayoutV2.space
RoleLayoutV2.stride
```

### `tensorrt_llm/_torch/disaggregation/base/transfer.py`

库存函数 13；已覆盖 4；未覆盖 9。

```text
RxSessionBase.receive
RxSessionBase.wait_complete
TxSessionBase.send
TxSessionBase.wait_complete
_SessionBase.close
_SessionBase.disagg_request_id
_SessionBase.exception
_SessionBase.is_completed
_SessionBase.wait_complete
```

### `tensorrt_llm/_torch/disaggregation/blob/backend.py`

库存函数 57；已覆盖 56；未覆盖 1。

```text
RegionHold.release
```

### `tensorrt_llm/_torch/disaggregation/blob/store.py`

库存函数 41；已覆盖 27；未覆盖 14。

```text
RegistersRegionsTogether.register_regions
RegistersRegionsTogether.unregister_regions
StoreDriver.access_ended
StoreDriver.access_unprovable
StoreDriver.cancel
StoreDriver.capabilities
StoreDriver.close
StoreDriver.drain
StoreDriver.get
StoreDriver.poll
StoreDriver.probe
StoreDriver.put
StoreDriver.register
StoreDriver.unregister
```

### `tensorrt_llm/_torch/disaggregation/diagnostics.py`

库存函数 25；已覆盖 24；未覆盖 1。

```text
_reset_sink_after_fork
```

### `tensorrt_llm/_torch/disaggregation/kv_cache_transceiver.py`

库存函数 47；已覆盖 29；未覆盖 18。

```text
BindKvCacheTransceiver.check_gen_transfer_complete
BindKvCacheTransceiver.get_disaggregated_params
BindKvCacheTransceiver.get_status_dump
CacheTransBufferManager.__init__
CacheTransBufferManager.pre_alloc_buffer_size
KvCacheTransceiver.cancel_request
KvCacheTransceiver.check_context_transfer_status
KvCacheTransceiver.check_gen_transfer_complete
KvCacheTransceiver.check_gen_transfer_status
KvCacheTransceiver.get_data_transceiver_state
KvCacheTransceiver.get_disaggregated_params
KvCacheTransceiver.get_status_dump
KvCacheTransceiver.prepare_context_requests
KvCacheTransceiver.reads_unique_tokens
KvCacheTransceiver.request_and_receive_async
KvCacheTransceiver.request_and_receive_sync
KvCacheTransceiver.respond_and_send_async
KvCacheTransceiver.set_gen_deferred_receives
```

### `tensorrt_llm/_torch/disaggregation/native/auxiliary.py`

库存函数 19；已覆盖 13；未覆盖 6。

```text
AuxBufferBase.alloc_slot
AuxBufferBase.fill_slot
AuxBufferBase.free_slot
AuxBufferBase.get_slot_data
AuxBufferBase.get_slot_tokens
AuxBufferBase.meta
```

### `tensorrt_llm/_torch/disaggregation/native/extent.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
Attempt.poll
```

### `tensorrt_llm/_torch/disaggregation/native/messenger.py`

库存函数 20；已覆盖 12；未覆盖 8。

```text
MessengerInterface.endpoint
MessengerInterface.receive
MessengerInterface.send
MessengerInterface.start
MessengerInterface.start_listener
MessengerInterface.stop
ZMQMessenger.start
ZMQMessenger.start_listener.<locals>.handle_listener_exceptions
```

### `tensorrt_llm/_torch/disaggregation/native/perf_logger.py`

库存函数 27；已覆盖 26；未覆盖 1。

```text
PerfLogManager.use_file
```

### `tensorrt_llm/_torch/disaggregation/native/transfer.py`

库存函数 250；已覆盖 235；未覆盖 15。

```text
RankInfoServer.__enter__
RankInfoServer.__exit__
Receiver.__enter__
Receiver.__exit__
RxSession.__enter__
RxSession.__exit__
Sender.__enter__
Sender.__exit__
Sender._send_aux_result_to_receiver
TransferWorker.__enter__
TransferWorker.__exit__
TransferWorker._on_quiescence_fatal
TxSession.__enter__
TxSession.__exit__
TxSession._claim_unsubmitted_aux_failure
```

### `tensorrt_llm/_torch/disaggregation/nixl/_agent_cpp.py`

库存函数 28；已覆盖 21；未覆盖 7。

```text
BindingsNixlTransferAgent.__enter__
BindingsNixlTransferAgent.__exit__
BindingsNixlTransferAgent.bounce_reject_counts
BindingsNixlTransferAgent.get_local_connection_info
BindingsNixlTransferAgent.get_notified_sync_messages
BindingsNixlTransferAgent.load_remote_agent_by_connection
BindingsNixlTransferAgent.notify_sync_message
```

### `tensorrt_llm/_torch/disaggregation/nixl/_agent_py.py`

库存函数 16；已覆盖 12；未覆盖 4。

```text
NixlTransferAgent.__enter__
NixlTransferAgent.__exit__
NixlTransferAgent.notify_sync_message
NixlTransferAgent.shutdown
```

### `tensorrt_llm/_torch/disaggregation/orchestration/coordinator.py`

库存函数 94；已覆盖 92；未覆盖 2。

```text
DisaggTransferCoordinator.fail_timed_out_synced
DisaggTransferCoordinator.take_pending_timed_out
```

### `tensorrt_llm/_torch/disaggregation/resource/cache_reuse.py`

库存函数 17；已覆盖 10；未覆盖 7。

```text
CacheReuseAdapter._global_cached_token_count
CacheReuseAdapter.commit_blocks_for_reuse
CacheReuseAdapter.enable_block_reuse
CacheReuseAdapter.get_block_ids
CacheReuseAdapter.get_block_ordinals
CacheReuseAdapter.tokens_per_block
_CacheReuseAdapterV2.get_block_ids
```

### `tensorrt_llm/_torch/disaggregation/resource/page.py`

库存函数 19；已覆盖 18；未覆盖 1。

```text
LayerGroup.to_dict
```

### `tensorrt_llm/_torch/disaggregation/resource/utils.py`

库存函数 24；已覆盖 16；未覆盖 8。

```text
get_layer_group_num_layers
get_num_buffer_entries
get_pool_view_num_layers
get_pool_views
get_total_buffer_entries
get_total_pool_bytes
get_total_pools
get_total_slots
```

### `tensorrt_llm/_torch/disaggregation/testing/driver_conformance.py`

库存函数 116；已覆盖 112；未覆盖 4。

```text
DriverConformance.make_blocked_put
DriverConformance.make_driver
DriverConformance.make_put_still_reading_the_source
Endpoint.key
```

### `tensorrt_llm/_torch/disaggregation/testing/host_memory.py`

库存函数 11；已覆盖 6；未覆盖 5。

```text
Memory.address
Memory.canaries_intact
Memory.nbytes
Memory.read
Memory.write
```

### `tensorrt_llm/_torch/disaggregation/testing/memory_store.py`

库存函数 93；已覆盖 92；未覆盖 1。

```text
MemoryDriver.open_ops
```

### `tensorrt_llm/_torch/disaggregation/transceiver.py`

库存函数 94；已覆盖 90；未覆盖 4。

```text
KvCacheTransceiverV2._poll_gen_sessions_for_poll_interval
KvCacheTransceiverV2.get_context_state
KvCacheTransceiverV2.get_status_dump
KvCacheTransceiverV2.get_status_dump.<locals>.summarize
```

### `tensorrt_llm/_torch/distributed/communicator.py`

库存函数 144；已覆盖 73；未覆盖 71。

```text
Distributed.allgather
Distributed.allreduce
Distributed.barrier
Distributed.broadcast
Distributed.broadcast_or_none
Distributed.cp_allgather
Distributed.cp_allgather_int64
Distributed.cp_broadcast
Distributed.has_cp_ulysses
Distributed.has_tp
Distributed.local_world_size
Distributed.tp_allgather
Distributed.tp_allreduce
Distributed.tp_barrier
Distributed.tp_broadcast
MPIDist.pp_gather
MPIDist.send
MultiHandleWrapper.__init__
MultiHandleWrapper.wait
PPCommTorch.__init__
PPCommTorch._global_to_local_rank
PPCommTorch.recv
PPCommTorch.send
TorchDist.__init__
TorchDist._cuda_allgather_cpu_tensor
TorchDist._cuda_allgather_object
TorchDist._cuda_allreduce_scalar
TorchDist._cuda_bcast_size_buffers
TorchDist._cuda_broadcast_object
TorchDist._cuda_gather_object
TorchDist._cuda_obj_guard
TorchDist._cuda_obj_pg
TorchDist._cuda_obj_pinned
TorchDist._cuda_obj_staging_for
TorchDist._cuda_obj_stream_ctx
TorchDist._get_cluster_info
TorchDist._init_cuda_object_groups
TorchDist._init_cuda_object_groups.<locals>._small_comm_options
TorchDist.allgather
TorchDist.allreduce
TorchDist.barrier
TorchDist.broadcast
TorchDist.broadcast_or_none
TorchDist.broadcast_or_none_with_header
TorchDist.cp_allgather
TorchDist.cp_broadcast
TorchDist.isend
TorchDist.isend_object
TorchDist.isend_tensor
TorchDist.local_world_size
TorchDist.pp_allgather
TorchDist.pp_broadcast
TorchDist.pp_gather
TorchDist.rank
TorchDist.recv
TorchDist.recv_object
TorchDist.recv_tensor
TorchDist.send
TorchDist.send_object
TorchDist.setup_local_comm
TorchDist.supports_single_round_broadcast
TorchDist.tp_allgather
TorchDist.tp_allgather_int64
TorchDist.tp_allreduce
TorchDist.tp_barrier
TorchDist.tp_broadcast
TorchDist.tp_gather
_chunked_transfer_loop
reduce_op_to_torch
safe_allgather.<locals>._allgatherv
safe_gather.<locals>._gatherv
```

### `tensorrt_llm/_torch/distributed/ll_allgather.py`

库存函数 13；已覆盖 12；未覆盖 1。

```text
_ll_allgather_kernel
```

### `tensorrt_llm/_torch/distributed/ll_exchange.py`

库存函数 57；已覆盖 35；未覆盖 22。

```text
_argmax_exchange_kernel
_bf16_hi
_bf16_lo
_bf16_pack
_ll_empty_words
_ll_exchange_words
_ll_group_kernel
_ll_group_poll
_ll_group_push
_ll_half
_ll_load
_ll_merge_kernel
_ll_peer
_ll_poll
_ll_reset
_ll_word
_orderable_key_hi
_pack_argmax_keys_kernel
_tp_key_exchange_kernel
argmax_exchange_finish
fast_gumbel_key
ll_pos_gather
```

### `tensorrt_llm/_torch/distributed/mnnvl_memory.py`

库存函数 48；已覆盖 34；未覆盖 14。

```text
HelixCpMnnvlMemory.comm_process_group
MnnvlCheckpointCommunicator.Get_rank
MnnvlCheckpointCommunicator.Get_size
MnnvlCheckpointCommunicator.allgather
MnnvlCheckpointCommunicator.barrier
MnnvlCheckpointCommunicator.irecv
MnnvlCheckpointCommunicator.isend
MnnvlMemory.comm_process_group
ProcessGroupComm.Get_rank
ProcessGroupComm.Get_size
ProcessGroupComm.__init__
ProcessGroupComm.allgather
ProcessGroupComm.barrier
TpMnnvlMemory.comm_process_group
```

### `tensorrt_llm/_torch/distributed/ops.py`

库存函数 65；已覆盖 55；未覆盖 10。

```text
AllReduce.gemm_allreduce_norm
AllReduce.gemm_twoshot_allreduce_norm
MNNVLAllReduce.prescale_workspace
MiniMaxAllReduceRMS.forward_qk
_MpiCommProtocol.Dup
_MpiCommProtocol.Free
_MpiCommProtocol.Get_size
_MpiCommProtocol.allreduce
_MpiCommProtocol.py2f
_warmup_named_allreduce_topology
```

### `tensorrt_llm/_torch/fused_relu2_triton.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
_relu2_kernel
```

### `tensorrt_llm/_torch/host_staging.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
_copy_kernel
```

### `tensorrt_llm/_torch/hostfunc.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
free_hostfunc_user_data
```

### `tensorrt_llm/_torch/kernels/argmax.py`

库存函数 33；已覆盖 25；未覆盖 8。

```text
ArgmaxKernel.__call__
ArgmaxKernel.kernel
CUDAGraphCompatibleWrapper.__dlpack_device__
ReductionBase._calculate_threads_per_row
ReductionBase._get_reduction_buffer_layout
ReductionBase._initialize_cluster
ReductionBase._set_cluster_n
ReductionBase._smem_size_in_bytes
```

### `tensorrt_llm/_torch/kernels/blackwell/blockwise_gemm/blockwise_gemm.py`

库存函数 20；已覆盖 16；未覆盖 4。

```text
Sm100BlockwiseGemmKernel.__call__
Sm100BlockwiseGemmKernel._mxfp8_out_transform
Sm100BlockwiseGemmKernel.kernel
Sm100BlockwiseGemmKernel.wrapper
```

### `tensorrt_llm/_torch/kernels/blackwell/blockwise_gemm/blockwise_gemm_n256.py`

库存函数 16；已覆盖 12；未覆盖 4。

```text
Sm100BlockwiseGemmMxfp8N256Kernel.__call__
Sm100BlockwiseGemmMxfp8N256Kernel._epilog_mxfp8_from_registers
Sm100BlockwiseGemmMxfp8N256Kernel._out_tile_sync
Sm100BlockwiseGemmMxfp8N256Kernel.kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/custom_pipeline.py`

库存函数 18；已覆盖 16；未覆盖 2。

```text
PipelineCpAsyncUmma._compute_is_leader_cta
PipelineTmaUmma.producer_commit
```

### `tensorrt_llm/_torch/kernels/blackwell/dense_blockscaled_gemm_act_fusion.py`

库存函数 24；已覆盖 15；未覆盖 9。

```text
Sm100BlockScaledPersistentDenseGemmActFusionKernel.__call__
Sm100BlockScaledPersistentDenseGemmActFusionKernel._apply_gelu_epilogue
Sm100BlockScaledPersistentDenseGemmActFusionKernel._apply_swiglu_epilogue
Sm100BlockScaledPersistentDenseGemmActFusionKernel.kernel
Sm100BlockScaledPersistentDenseGemmActFusionKernel.wrapper
Sm100BlockScaledPersistentDenseGemmActFusionKernel.wrapper_deferred_fp4out
Sm100BlockScaledPersistentDenseGemmActFusionKernel.wrapper_fp4out
Sm100BlockScaledPersistentDenseGemmActFusionKernel.wrapper_indexer_q
cvt_sf_MKL_to_M32x4xrm_K4xrk_L
```

### `tensorrt_llm/_torch/kernels/blackwell/dense_blockscaled_gemm_persistent.py`

库存函数 31；已覆盖 19；未覆盖 12。

```text
Sm100BlockScaledPersistentDenseGemmKernel.__call__
Sm100BlockScaledPersistentDenseGemmKernel._indexer_q_transform_rows
Sm100BlockScaledPersistentDenseGemmKernel._mxfp8_out_transform
Sm100BlockScaledPersistentDenseGemmKernel._q_amax_update
Sm100BlockScaledPersistentDenseGemmKernel._q_rope_load
Sm100BlockScaledPersistentDenseGemmKernel._q_rope_rotate
Sm100BlockScaledPersistentDenseGemmKernel._q_rope_transform
Sm100BlockScaledPersistentDenseGemmKernel.kernel
Sm100BlockScaledPersistentDenseGemmKernel.needs_unpack_tma
Sm100BlockScaledPersistentDenseGemmKernel.wrapper
Sm100BlockScaledPersistentDenseGemmKernel.wrapper_indexer_q_swap_ab
cvt_sf_MKL_to_M32x4xrm_K4xrk_L
```

### `tensorrt_llm/_torch/kernels/blackwell/dense_gemm_persistent.py`

库存函数 28；已覆盖 14；未覆盖 14。

```text
PersistentDenseGemmKernel.__call__
PersistentDenseGemmKernel._split_k_reduction_kernel
PersistentDenseGemmKernel.check_epilog_store_option
PersistentDenseGemmKernel.check_mma_tiler_and_cluster_shape
PersistentDenseGemmKernel.check_tensor_alignment
PersistentDenseGemmKernel.check_tensor_alignment.<locals>.is_aligned
PersistentDenseGemmKernel.kernel
PersistentDenseGemmKernel.split_k_reduction
PersistentDenseGemmKernel.wrapper
PersistentDenseGemmKernel.wrapper_strided
PersistentDenseGemmKernel.wrapper_strided_c
PersistentDenseGemmKernel.wrapper_strided_fp8_quantized
_atomic_max_shared_u32
_f32_as_u32
```

### `tensorrt_llm/_torch/kernels/blackwell/engram/decode_hash.py`

库存函数 9；已覆盖 4；未覆盖 5。

```text
EngramDecodeHashKernel.__call__
EngramDecodeHashKernel.kernel
EngramDecodeHashKernel.store_hashes
EngramDecodeRefreshKernel.__call__
EngramDecodeRefreshKernel.kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/engram/lookup.py`

库存函数 14；已覆盖 8；未覆盖 6。

```text
EngramLookupKernel.__call__
EngramLookupKernel.kernel
EngramLookupKernel.load_chunk
EngramLookupKernel.store_chunk
EngramLookupKernel.store_packed
EngramLookupKernel.table_row
```

### `tensorrt_llm/_torch/kernels/blackwell/engram/packed_projection.py`

库存函数 9；已覆盖 6；未覆盖 3。

```text
EngramPackedProjectionKernel.__call__
EngramPackedProjectionKernel.kernel
EngramPackedProjectionKernel.wrapper
```

### `tensorrt_llm/_torch/kernels/blackwell/fp8_block_scale_low_m_gemm.py`

库存函数 120；已覆盖 101；未覆盖 19。

```text
Fp8BlockScaleLowMGemmKernel.__call__
Fp8BlockScaleLowMGemmKernel._launch
Fp8BlockScaleLowMGemmKernel.call_gated
Fp8BlockScaleLowMGemmKernel.call_nvfp4
Fp8BlockScaleLowMGemmKernel.epilogue
Fp8BlockScaleLowMGemmKernel.kernel
Fp8BlockScaleLowMGemmKernel.mma_warp
Fp8BlockScaleLowMGemmKernel.quantize_activations
Fp8BlockScaleMidMGemmKernel.__call__
Fp8BlockScaleMidMGemmKernel._launch
Fp8BlockScaleMidMGemmKernel.call_nvfp4
Fp8BlockScaleMidMGemmKernel.epilogue
Fp8BlockScaleMidMGemmKernel.kernel
Fp8BlockScaleMidMGemmKernel.mma_warp
Fp8BlockScaleMidMGemmKernel.scale_warp
MxFp8QuantKernel.__call__
MxFp8QuantKernel.kernel
_Nvfp4OutputMixin._store_nvfp4
precompile_prequantized
```

### `tensorrt_llm/_torch/kernels/blackwell/gate_add_rmsnorm_fp8.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
GateAddRMSNormFp8Kernel.__call__
GateAddRMSNormFp8Kernel.kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/gdn_verify_tcgen05.py`

库存函数 42；已覆盖 41；未覆盖 1。

```text
_packed_slot_rows
```

### `tensorrt_llm/_torch/kernels/blackwell/gemm_allreduce_norm.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
GemmAllReduceNormKernel.__call__
GemmAllReduceNormKernel.kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/kimi_k3_attn_res.py`

库存函数 55；已覆盖 50；未覆盖 5。

```text
KimiK3AttnResKernel.__call__
KimiK3AttnResKernel.kernel
KimiK3AttnResKernel.kernel.<locals>.load
KimiK3AttnResKernel.kernel.<locals>.unpack
KimiK3AttnResKernel.kernel.<locals>.wait_and_release
```

### `tensorrt_llm/_torch/kernels/blackwell/low_m_mma_gemm.py`

库存函数 27；已覆盖 10；未覆盖 17。

```text
LowMMmaGemmKernel.__call__
LowMMmaGemmKernel._clamp_row
LowMMmaGemmKernel._cluster_sum
LowMMmaGemmKernel._conv_epilogue
LowMMmaGemmKernel._cta
LowMMmaGemmKernel._epilogue_k1
LowMMmaGemmKernel._finish
LowMMmaGemmKernel._mainloop_bf16
LowMMmaGemmKernel._mainloop_bf16_swap
LowMMmaGemmKernel._mainloop_fp8
LowMMmaGemmKernel._mainloop_fp8_swap
LowMMmaGemmKernel._put_c
LowMMmaGemmKernel._store
LowMMmaGemmKernel._swiglu
LowMMmaGemmKernel._weights_bf16
LowMMmaGemmKernel.kernel
_e4m3x4_to_bf16x4
```

### `tensorrt_llm/_torch/kernels/blackwell/lowm_fp8_mma_gemm.py`

库存函数 24；已覆盖 14；未覆盖 10。

```text
LowMFp8MmaGemmKernel.__call__
LowMFp8MmaGemmKernel._convert_weights
LowMFp8MmaGemmKernel._inverse_rope
LowMFp8MmaGemmKernel._main_loop
LowMFp8MmaGemmKernel._output_value
LowMFp8MmaGemmKernel._pack_epilogue
LowMFp8MmaGemmKernel._rms_scale
LowMFp8MmaGemmKernel._store
LowMFp8MmaGemmKernel._swiglu_epilogue
LowMFp8MmaGemmKernel.kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/masked_embedding.py`

库存函数 7；已覆盖 5；未覆盖 2。

```text
MaskedEmbeddingKernel.__call__
MaskedEmbeddingKernel.kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/mhc/mhc_boundary.py`

库存函数 106；已覆盖 82；未覆盖 24。

```text
MhcBoundaryKernel.__call__
MhcBoundaryKernel.coefficients
MhcBoundaryKernel.kernel
MhcBoundaryKernel.layer_input_words
MhcBoundaryKernel.sinkhorn
MhcCoefficientKernel.__call__
MhcCoefficientKernel.coefficients8
MhcCoefficientKernel.kernel
MhcCoefficientKernel.load_chunk
MhcFinalizeKernel.__call__
MhcFinalizeKernel.kernel
MhcSinkhornKernel.__call__
MhcSinkhornKernel.kernel
MhcSmallCoefficientKernel.__call__
MhcSmallCoefficientKernel.finish_group
MhcSmallCoefficientKernel.kernel
MhcSmallCoefficientKernel.load_chunk
MhcStreamKernel.__call__
MhcStreamKernel.kernel
MhcStreamKernel.load_chunk
Nvfp4Fp8QuantizeKernel.__call__
Nvfp4Fp8QuantizeKernel.kernel
Nvfp4QuantizeKernel.__call__
Nvfp4QuantizeKernel.kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/mhc/mhc_coefficient_fma.py`

库存函数 35；已覆盖 31；未覆盖 4。

```text
MhcCoefficientFmaKernel.__call__
MhcCoefficientFmaKernel.kernel
MhcCoefficientFmaKernel.normalize_columns
MhcCoefficientFmaKernel.token_tail
```

### `tensorrt_llm/_torch/kernels/blackwell/mhc/mhc_coefficient_tc.py`

库存函数 14；已覆盖 6；未覆盖 8。

```text
MhcCoefficientTcKernel.__call__
MhcCoefficientTcKernel.coefficients32
MhcCoefficientTcKernel.finalize
MhcCoefficientTcKernel.kernel
MhcCoefficientTcKernel.loaded_sums
MhcCoefficientTcKernel.square_sums
MhcCoefficientTcKernel.staged_sums
compiled
```

### `tensorrt_llm/_torch/kernels/blackwell/mhc/mhc_split_boundary.py`

库存函数 31；已覆盖 17；未覆盖 14。

```text
MhcAllReduceBoundaryKernel.__call__
MhcAllReduceBoundaryKernel.ar_kernel
MhcPostCollapseNormKernel.__call__
MhcPostCollapseNormKernel._boundary
MhcPostCollapseNormKernel._collapse_norm
MhcPostCollapseNormKernel._kernel_single
MhcPostCollapseNormKernel._kernel_vectors
MhcPostCollapseNormKernel._kernel_vectors.<locals>.load_residual
MhcPostCollapseNormKernel._load_coefficients
MhcPostCollapseNormKernel._moe_x_words
MhcPostCollapseNormKernel._store_mxfp8
MhcPostCollapseNormKernel._store_nvfp4
MhcPostCollapseNormKernel.kernel
_add
```

### `tensorrt_llm/_torch/kernels/blackwell/mnnvl_allreduce_norm.py`

库存函数 48；已覆盖 39；未覆盖 9。

```text
MnnvlAllReduceNormKernel.__call__
MnnvlAllReduceNormKernel.kernel
MnnvlAllReduceNormKernel.kernel_mt
_cluster_arrive_relaxed
_cluster_wait
_min_s64
_prefetch_l2_bulk_if
_st_dsmem_f32
l2_prefetch_slice
```

### `tensorrt_llm/_torch/kernels/blackwell/mnnvl_slot_allreduce_norm.py`

库存函数 13；已覆盖 11；未覆盖 2。

```text
MnnvlSlotAllReduceNormKernel.__call__
MnnvlSlotAllReduceNormKernel.kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/moe_bf16_engine.py`

库存函数 23；已覆盖 13；未覆盖 10。

```text
MoeBf16Engine.__call__
MoeBf16Engine._b_stage
MoeBf16Engine.epilogue
MoeBf16Engine.kernel
MoeBf16Engine.load_a
MoeBf16Engine.load_b
MoeBf16Engine.mma
MoeBf16Engine.route
MoeBf16Engine.rows
MoeBf16Engine.segments
```

### `tensorrt_llm/_torch/kernels/blackwell/moe_decode_front.py`

库存函数 52；已覆盖 42；未覆盖 10。

```text
MoeDecodeFrontKernel.__call__
MoeDecodeFrontKernel._finish_token
MoeDecodeFrontKernel._gemm_cta
MoeDecodeFrontKernel._quant_cta
MoeDecodeFrontKernel._route
MoeDecodeFrontKernel._router_cta
MoeDecodeFrontKernel._tables_warp
MoeDecodeFrontKernel._tokens
MoeDecodeFrontKernel.kernel
_v
```

### `tensorrt_llm/_torch/kernels/blackwell/moe_lowm_fc2.py`

库存函数 18；已覆盖 11；未覆盖 7。

```text
MoeLowMFc2Nvfp4.__call__
MoeLowMFc2Nvfp4._unit
MoeLowMFc2Nvfp4.epilogue
MoeLowMFc2Nvfp4.kernel
MoeLowMFc2Nvfp4.load_a
MoeLowMFc2Nvfp4.load_b
MoeLowMFc2Nvfp4.mma
```

### `tensorrt_llm/_torch/kernels/blackwell/moe_nvfp4_engine.py`

库存函数 31；已覆盖 22；未覆盖 9。

```text
MoeNvfp4Engine.__call__
MoeNvfp4Engine._b_stage
MoeNvfp4Engine._route_deferred
MoeNvfp4Engine._route_prologue
MoeNvfp4Engine.epilogue
MoeNvfp4Engine.kernel
MoeNvfp4Engine.load_a
MoeNvfp4Engine.load_b
MoeNvfp4Engine.mma
```

### `tensorrt_llm/_torch/kernels/blackwell/moe_persistent_engine.py`

库存函数 18；已覆盖 4；未覆盖 14。

```text
PersistentMoeEngine._push
PersistentMoeEngine.begin_call
PersistentMoeEngine.decode
PersistentMoeEngine.end_call
PersistentMoeEngine.epilogue_tmem
PersistentMoeEngine.first_unit
PersistentMoeEngine.free_unit
PersistentMoeEngine.init_barriers
PersistentMoeEngine.load_acc
PersistentMoeEngine.scheduler
PersistentMoeEngine.take
PersistentMoeEngine.unit
PersistentMoeEngine.wait_dependency
PersistentMoeEngine.wait_fc1_done
```

### `tensorrt_llm/_torch/kernels/blackwell/moe_route_front.py`

库存函数 24；已覆盖 16；未覆盖 8。

```text
MoeRouteFrontKernel.__call__
MoeRouteFrontKernel._quant_cta
MoeRouteFrontKernel._route_cta
MoeRouteFrontKernel._route_cta_lean
MoeRouteFrontKernel._route_warp
MoeRouteFrontKernel._route_warp_fast
MoeRouteFrontKernel._topk_full
MoeRouteFrontKernel.kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/moe_router_gemm.py`

库存函数 12；已覆盖 8；未覆盖 4。

```text
MoeRouterGemmKernel.__call__
MoeRouterGemmKernel._quant_phase
MoeRouterGemmKernel._tile_candidates
MoeRouterGemmKernel.kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/mxfp8_argmax_gemv.py`

库存函数 13；已覆盖 8；未覆盖 5。

```text
MXFP8ArgmaxGemvKernel.__call__
MXFP8ArgmaxGemvKernel._issue_tile
MXFP8ArgmaxGemvKernel._reduce_and_publish
MXFP8ArgmaxGemvKernel._stage_activation
MXFP8ArgmaxGemvKernel.kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/block_scan.py`

库存函数 6；已覆盖 3；未覆盖 3。

```text
block_prefix_sum
host_block_prefix_sum
test_block_prefix_sum
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/filtered_top_k_decode_varlen.py`

库存函数 17；已覆盖 10；未覆盖 7。

```text
ComputeDynamicCTAOffsets.__call__
ComputeDynamicCTAOffsets.__init__
ComputeDynamicCTAOffsets.compute_offsets_kernel
FilteredTopKKernelVarlenDecode.__call__
FilteredTopKKernelVarlenDecode.filtered_topk_kernel
cute_dsl_radix_filter_topk_single_pass_multi_cta_wrapper
run_filtered_topk_decode.<locals>.generate_inputs
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/filtered_top_k_prefill_varlen.py`

库存函数 10；已覆盖 2；未覆盖 8。

```text
FilteredTopKKernelVarlenPrefill.__call__
FilteredTopKKernelVarlenPrefill.filtered_topk_kernel
_bucket_num_cols
_next_positive_power_of_2
cute_dsl_topk_prefill_wrapper
run_filtered_topk_prefill
run_filtered_topk_prefill.<locals>.generate_inputs
run_topk_prefill
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/filtered_top_k_varlen_util.py`

库存函数 34；已覆盖 11；未覆盖 23。

```text
FilteredTopKKernelVarlen._cluster_collect
FilteredTopKKernelVarlen._cluster_reduce_histogram
FilteredTopKKernelVarlen._collect_below_threshold_coarse
FilteredTopKKernelVarlen._collect_below_threshold_refine
FilteredTopKKernelVarlen._compute_smem_input_size
FilteredTopKKernelVarlen._filter_and_histogram_all_vectors_coarse
FilteredTopKKernelVarlen._filter_and_histogram_coarse
FilteredTopKKernelVarlen._filter_and_histogram_per_elem_coarse
FilteredTopKKernelVarlen._filter_and_histogram_per_elem_refine
FilteredTopKKernelVarlen._filter_and_histogram_refine
FilteredTopKKernelVarlen._filter_and_histogram_vector_coarse
FilteredTopKKernelVarlen._phase3_writeback
FilteredTopKKernelVarlen._reread_always_gmem_combined_scan
FilteredTopKKernelVarlen._reread_always_gmem_output_scan
FilteredTopKKernelVarlen._reread_always_per_elem_combined
FilteredTopKKernelVarlen._reread_always_per_elem_output
FilteredTopKKernelVarlen._reread_gmem_rescan
FilteredTopKKernelVarlen.filtered_topk_kernel_per_row
FilteredTopKKernelVarlen.prefix_sum_and_find_threshold_coarse
FilteredTopKKernelVarlen.prefix_sum_and_find_threshold_fine_grained
FilteredTopKKernelVarlen.to_coarse_key
FilteredTopKKernelVarlen.to_ordered
FilteredTopKKernelVarlen.to_ordered_and_coarse
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/gvr_topk_decode.py`

库存函数 57；已覆盖 23；未覆盖 34。

```text
GvrTopKKernel.__call__
GvrTopKKernel._block_bound
GvrTopKKernel._full_build_active
GvrTopKKernel._hist_build
GvrTopKKernel._kth_bin_search
GvrTopKKernel._kth_bin_search_rw
GvrTopKKernel._list_ld
GvrTopKKernel._list_st
GvrTopKKernel._load_fp32
GvrTopKKernel._p4_coarse_rw
GvrTopKKernel._p4_fine_rw
GvrTopKKernel._run_phases
GvrTopKKernel._smem_ld
GvrTopKKernel._smem_ref
GvrTopKKernel._smem_st
GvrTopKKernel.block_count_ge
GvrTopKKernel.block_count_ge_multi
GvrTopKKernel.block_fused_snap_iter
GvrTopKKernel.gvr_topk_kernel
GvrTopKKernel.load_slice_to_smem
GvrTopKKernel.phase0_scan_bucket
GvrTopKKernel.phase1_preidx_stats
GvrTopKKernel.phase1b_hspace_rungs
GvrTopKKernel.phase1b_hspace_rungs_cached
GvrTopKKernel.phase1r_data_reseed
GvrTopKKernel.phase2_secant_search
GvrTopKKernel.phase3_collect_candidates
GvrTopKKernel.phase4_histogram_snap
GvrTopKKernel.phase4_rank_scatter
GvrTopKKernel.run_one_row
GvrTopKKernel.warp_reduce_max_f32
GvrTopKKernel.warp_reduce_min_f32
GvrTopKKernel.warp_reduce_sum_f32
GvrTopKKernel.warp_reduce_sum_i32
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/gvr_topk_decode_direct.py`

库存函数 12；已覆盖 5；未覆盖 7。

```text
DirectTopKKernel.__call__
DirectTopKKernel._bin_select
DirectTopKKernel._emit_final
DirectTopKKernel._emit_prefix_ge
DirectTopKKernel._row_n_eff
DirectTopKKernel._warp_suffix_scan
DirectTopKKernel.direct_topk_kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/gvr_topk_decode_load_balance.py`

库存函数 6；已覆盖 2；未覆盖 4。

```text
GvrTopKLBKernel.__call__
GvrTopKLBKernel.main_kernel
GvrTopKLBPrepareKernel.__call__
GvrTopKLBPrepareKernel.prepare_kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/gvr_topk_decode_reg.py`

库存函数 12；已覆盖 7；未覆盖 5。

```text
GvrRegKernel.__call__
GvrRegKernel.count_reg
GvrRegKernel.exchange_counts
GvrRegKernel.gvr_reg_kernel
GvrRegKernel.max_below_reg
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/gvr_topk_decode_self_sampling.py`

库存函数 294；已覆盖 239；未覆盖 55。

```text
ComplementKernel.__call__
ComplementKernel.kernel
ComplementKernel.kernel_bf
GvrClusKernel.__call__
GvrClusKernel._ballot_pair_emit
GvrClusKernel._body
GvrClusKernel._emitk
GvrClusKernel._fin_tail
GvrClusKernel._merge_scan0_local
GvrClusKernel._p5_emit
GvrClusKernel.call_fin
GvrClusKernel.kern
GvrClusKernel.kern_fin
GvrMainKernel.__call__
GvrMainKernel._ballot_pair_emit
GvrMainKernel._classify_staged_f16
GvrMainKernel._dense_bin
GvrMainKernel._emitc
GvrMainKernel._m2_idx
GvrMainKernel._stage_index_only
GvrMainKernel.kern
GvrRegClusKernel.__call__
GvrRegClusKernel._radix_rank_long16
GvrRegClusKernel.kern
GvrTopkRegKernel.__call__
GvrTopkRegKernel.kern
GvrTopkRegKernel.kern_cr
GvrTopkRegKernel.kern_cr2
GvrTopkRegKernel.kern_crc
GvrTopkRegKernel.kern_crl
_atom_shared_cluster_add_i32
_bf16_pick16
_bf16_survivor_val__main
_bf16_window_emit_ballot
_ld_shared_cluster_f32
_ldcg_v4_i32
_legacy_dummy_kv
_red_shared_addv
_st_shared_cluster_f32
atomic_or_cta
bf16x2_ge_mask
f16x2_ge_mask
fkey16x2_of_pair
fused_cluster_store
g2r_atom_i32
ge16x2_mask
ld_g_f16x8
merge_scan0
reg_topk
regclus_topk
run
run__clus
run_manual
warp_add_u32
workspace_bytes
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/gvr_topk_decode_self_sampling_bf16.py`

库存函数 29；已覆盖 15；未覆盖 14。

```text
GvrPackedClampWindowKernel.__call__
GvrPackedClampWindowKernel._emit_window
GvrPackedClampWindowKernel._fill_window_histogram
GvrPackedClampWindowKernel._row_body
GvrPackedClampWindowKernel._window_controls
GvrPackedClampWindowKernel.kern
_copy_output_quad
_packed_sample_accumulate
_packed_sample_collapse
_peer_count
_scan_cross_warp
_scan_cross_warp_parts
_sign_quad
_store_output_quad
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/gvr_topk_decode_self_sampling_fp16.py`

库存函数 27；已覆盖 25；未覆盖 2。

```text
GvrCodeRadixCounterKernel.__call__
GvrCodeRadixCounterKernel.kern_cr
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/gvr_topk_decode_self_sampling_host.py`

库存函数 80；已覆盖 79；未覆盖 1。

```text
_reset_for_tests
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/gvr_topk_decode_tall_ws.py`

库存函数 34；已覆盖 12；未覆盖 22。

```text
GvrTallWsKernel.__call__
GvrTallWsKernel._attempts
GvrTallWsKernel._chunk_v0
GvrTallWsKernel._clear_hist
GvrTallWsKernel._consume_gmem
GvrTallWsKernel._consume_stage
GvrTallWsKernel._gather
GvrTallWsKernel._gather_issue
GvrTallWsKernel._gather_wait
GvrTallWsKernel._issue_chunk
GvrTallWsKernel._issue_first
GvrTallWsKernel._ladder
GvrTallWsKernel._plan_row
GvrTallWsKernel._rcp
GvrTallWsKernel._reserve
GvrTallWsKernel._row_len
GvrTallWsKernel._select
GvrTallWsKernel._stream_gmem
GvrTallWsKernel._stream_ring
GvrTallWsKernel._vec_mask
GvrTallWsKernel._window
GvrTallWsKernel.kern_ws
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/gvr_topk_decode_tp.py`

库存函数 35；已覆盖 22；未覆盖 13。

```text
GvrTpKernel.__call__
GvrTpKernel._ld_float4
GvrTpKernel._push_cand
GvrTpKernel._row_n_eff
GvrTpKernel.collect_at
GvrTpKernel.count_pass
GvrTpKernel.exchange_counts
GvrTpKernel.fused_count_collect
GvrTpKernel.gvr_tp_kernel
GvrTpKernel.max_below_pass
GvrTpKernel.phase1
GvrTpKernel.sample_count
_st_shared_cluster_i32
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/single_pass_multi_cta_radix_topk.py`

库存函数 33；已覆盖 9；未覆盖 24。

```text
SinglePassMultiCTARadixTopKKernel.__call__
SinglePassMultiCTARadixTopKKernel._admit_listed_ties
SinglePassMultiCTARadixTopKKernel._chunk_slot
SinglePassMultiCTARadixTopKKernel._collect_gt_and_list_ties
SinglePassMultiCTARadixTopKKernel._collect_pass_eq
SinglePassMultiCTARadixTopKKernel._collect_pass_gt
SinglePassMultiCTARadixTopKKernel._collect_ties_by_scan
SinglePassMultiCTARadixTopKKernel._compute_prefix_mask
SinglePassMultiCTARadixTopKKernel._emit_selected
SinglePassMultiCTARadixTopKKernel._radix_round
SinglePassMultiCTARadixTopKKernel._radix_round_single_cta
SinglePassMultiCTARadixTopKKernel._single_cta_radix_select_and_collect
SinglePassMultiCTARadixTopKKernel._write_listed_ties
SinglePassMultiCTARadixTopKKernel.build_and_merge_histogram
SinglePassMultiCTARadixTopKKernel.build_local_histogram
SinglePassMultiCTARadixTopKKernel.collect_output
SinglePassMultiCTARadixTopKKernel.collect_output_single_cta
SinglePassMultiCTARadixTopKKernel.compute_local_gt_count
SinglePassMultiCTARadixTopKKernel.from_ordered
SinglePassMultiCTARadixTopKKernel.load_chunk_to_smem
SinglePassMultiCTARadixTopKKernel.prefix_sum_and_find_threshold
SinglePassMultiCTARadixTopKKernel.single_pass_multi_cta_topk_kernel
SinglePassMultiCTARadixTopKKernel.to_ordered
fence_acq_rel_gpu
```

### `tensorrt_llm/_torch/kernels/blackwell/top_k/single_pass_multi_cta_radix_topk_cluster.py`

库存函数 11；已覆盖 6；未覆盖 5。

```text
SinglePassMultiCTARadixTopKClusterKernel.__call__
SinglePassMultiCTARadixTopKClusterKernel._radix_round_cluster
SinglePassMultiCTARadixTopKClusterKernel.collect_output_cluster
SinglePassMultiCTARadixTopKClusterKernel.merge_histogram_dsmem
SinglePassMultiCTARadixTopKClusterKernel.single_pass_multi_cta_topk_kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/tp2_allreduce_norm.py`

库存函数 15；已覆盖 12；未覆盖 3。

```text
Tp2WarpAllReduceNormKernel.__call__
Tp2WarpAllReduceNormKernel._finalize
Tp2WarpAllReduceNormKernel.kernel
```

### `tensorrt_llm/_torch/kernels/blackwell/utils.py`

库存函数 31；已覆盖 17；未覆盖 14。

```text
_Pointer.__add__
_Pointer.__repr__
_Pointer.align
_Pointer.dtype
_Pointer.size_in_bytes
_Pointer.verify
atomic_add_func
blk_reduce_fp16
blk_reduce_fp32
gelu_tanh_f32
sigmoid_f32
silu_f32
vectorized_atomic_add_fp16x8
vectorized_atomic_add_fp32x2
```

### `tensorrt_llm/_torch/kernels/fused_ops/add_rmsnorm_fp8_quant.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
_add_rmsnorm_fp8_quant_kernel
```

### `tensorrt_llm/_torch/kernels/fused_ops/fp8_static_quant.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_fp8_static_quant_kernel
```

### `tensorrt_llm/_torch/kernels/fused_ops/gelu_tanh_mul.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_gelu_tanh_mul_kernel
```

### `tensorrt_llm/_torch/kernels/fused_ops/gelu_tanh_mul_fp4_quant.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
_gelu_tanh_mul_fp4_kernel
_rcp_approx
```

### `tensorrt_llm/_torch/kernels/fused_ops/gelu_tanh_mul_fp8_quant.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_gelu_tanh_mul_fp8_kernel
```

### `tensorrt_llm/_torch/kernels/fused_ops/rmsnorm_residual_add.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
_quant_fp8
_rmsnorm_residual_add_kernel
```

### `tensorrt_llm/_torch/kernels/fused_ops/soft_cap_logits.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
_soft_cap_logits_kernel
```

### `tensorrt_llm/_torch/kernels/nvfp4_quant.py`

库存函数 7；已覆盖 4；未覆盖 3。

```text
_e2m1x2
_nvfp4_quant_kernel
_rcp_approx_ftz
```

### `tensorrt_llm/_torch/kernels/rubin/blockwise_gemm/blockwise_gemm.py`

库存函数 23；已覆盖 17；未覆盖 6。

```text
SM107BlockwiseGemmKernel.__call__
SM107BlockwiseGemmKernel._epilog_fp8_register_scratch
SM107BlockwiseGemmKernel._split_acc_update
SM107BlockwiseGemmKernel.kernel
SM107BlockwiseGemmKernel.wrapper
SM107BlockwiseGemmKernel.wrapper_fp8_quantized
```

### `tensorrt_llm/_torch/kernels/rubin/dense_bf16_gemm_persistent.py`

库存函数 17；已覆盖 10；未覆盖 7。

```text
PersistentDenseGemmKernelPreferredCluster.__call__
PersistentDenseGemmKernelPreferredCluster.check_mma_tiler_and_cluster_shape
PersistentDenseGemmKernelPreferredCluster.cluster_specific_kernel
PersistentDenseGemmKernelPreferredCluster.kernel
PersistentDenseGemmKernelPreferredCluster.wrapper
PersistentDenseGemmKernelPreferredCluster.wrapper_strided
bmm_preferred_cluster
```

### `tensorrt_llm/_torch/kernels/rubin/dense_blockscaled_gemm_persistent.py`

库存函数 40；已覆盖 17；未覆盖 23。

```text
Sm107BlockScaledPersistentDenseGemmKernel.__call__
Sm107BlockScaledPersistentDenseGemmKernel._mainloop_s2t_interleaved_copies
Sm107BlockScaledPersistentDenseGemmKernel.kernel
Sm107BlockScaledPersistentDenseGemmKernel.kernel_impl
Sm107BlockScaledPersistentDenseGemmKernel.wrapper
Sm107BlockScaledPersistentDenseGemmMixedClustersKernel.__call__
Sm107BlockScaledPersistentDenseGemmMixedClustersKernel.__init__
Sm107BlockScaledPersistentDenseGemmMixedClustersKernel._compute_mixed_cluster_grid
Sm107BlockScaledPersistentDenseGemmMixedClustersKernel._setup_attributes
Sm107BlockScaledPersistentDenseGemmMixedClustersKernel.mixed_cluster_kernel
Sm107BlockScaledPersistentDenseGemmMixedClustersKernel.wrapper
ceil_div
construct_abc_cute_pointers_emulated
create_and_init_tensors_emulated
create_scale_factor_tensor
cvt_sf_MKL_to_M32x4xrm_K4xrk_L
prepare_parser
prepare_parser.<locals>.parse_comma_separated_ints
run
run_scaled_mm_with_emulated_dtype
run_scaled_mm_with_emulated_dtype.<locals>.benchmark_gemm
run_scaled_mm_with_emulated_dtype.<locals>.generate_inputs
scaled_mm
```

### `tensorrt_llm/_torch/kernels/rubin/nvfp4_plus_quantize.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
_Nvfp4PlusQuantize.__call__
_Nvfp4PlusQuantize.kernel
```

### `tensorrt_llm/_torch/kv_cache_compression/compaction.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
_pack_move_sources_kernel
```

### `tensorrt_llm/_torch/kv_cache_compression/quantization_for_cold_page/quantization_for_cold_page.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
ColdPageQuantizationCompression.build_codec_state
ColdPageQuantizationCompression.build_lifecycle_metadata
```

### `tensorrt_llm/_torch/kv_cache_compression/triattention/triattention.py`

库存函数 24；已覆盖 23；未覆盖 1。

```text
TriAttentionCompressionManager._create_kv_layout
```

### `tensorrt_llm/_torch/kv_cache_compression/triattention/triattention_cute_score_fused.py`

库存函数 12；已覆盖 9；未覆盖 3。

```text
_TriAttentionScoreKernel.__call__
_TriAttentionScoreKernel._stage_raw_band_copies
_TriAttentionScoreKernel.kernel
```

### `tensorrt_llm/_torch/kv_cache_compression/triattention/triattention_cute_selection.py`

库存函数 8；已覆盖 5；未覆盖 3。

```text
_TriAttentionNormalizeUnionKernel.__call__
_TriAttentionNormalizeUnionKernel._reduce_and_store_union_rows
_TriAttentionNormalizeUnionKernel.kernel
```

### `tensorrt_llm/_torch/kv_cache_compression/triattention/triattention_kernels.py`

库存函数 11；已覆盖 4；未覆盖 7。

```text
_compute_mean_phase_kernel
_fold_union_ranks_kernel
_gather_mean_phase_kernel
_score_per_head_reduce_kernel
_score_row_stats_kernel
_settle_ties_kernel
gather_mean_phase
```

### `tensorrt_llm/_torch/locality_domain/layout.py`

库存函数 11；已覆盖 10；未覆盖 1。

```text
PartitionedTensorLayout.partition_slice
```

### `tensorrt_llm/_torch/locality_domain/runtime.py`

库存函数 9；已覆盖 7；未覆盖 2。

```text
LocalityDomainRuntime.partition_mempool
LocalityDomainRuntime.prepare_for_capture
```

### `tensorrt_llm/_torch/locality_domain_utils.py`

库存函数 25；已覆盖 21；未覆盖 4。

```text
LocalityDomainResourceManager.__del__
LocalityDomainResourceManager.cleanup
cleanup_locality_domain_resources
reset_locality_domain_resource_manager
```

### `tensorrt_llm/_torch/memory/gpu_memory_backend.py`

库存函数 21；已覆盖 11；未覆盖 10。

```text
GMSBackend.get_source_identity
GPUMemoryBackend.cleanup
GPUMemoryBackend.connect
GPUMemoryBackend.finalize_write
GPUMemoryBackend.get_source_identity
GPUMemoryBackend.has_committed_weights
GPUMemoryBackend.is_rw
GPUMemoryBackend.materialize_module
GPUMemoryBackend.mem_pool_scope
GPUMemoryBackend.move_untracked_params
```

### `tensorrt_llm/_torch/memory/monitor/capture.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
_check_failed
```

### `tensorrt_llm/_torch/memory/monitor/providers/__init__.py`

库存函数 16；已覆盖 13；未覆盖 3。

```text
ExecutorToken.__repr__
MonitorProvider.available
MonitorProvider.collect
```

### `tensorrt_llm/_torch/memory/monitor/triggers.py`

库存函数 33；已覆盖 32；未覆盖 1。

```text
_device_nearly_full
```

### `tensorrt_llm/_torch/mmap_utils.py`

库存函数 10；已覆盖 9；未覆盖 1。

```text
_errno_name
```

### `tensorrt_llm/_torch/mnnvl_alltoall_workspace.py`

库存函数 18；已覆盖 16；未覆盖 2。

```text
_WorkspaceClient._mnnvl_checkpoint_is_idle
_WorkspaceClient._mnnvl_checkpoint_reset
```

### `tensorrt_llm/_torch/model_config.py`

库存函数 62；已覆盖 60；未覆盖 2。

```text
ModelConfig.fuse_pos_embd
_is_lock_infra_error
```

### `tensorrt_llm/_torch/modeling_v2/catalog/_op.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
OpWrapper.__call__
OpWrapper.reference
```

### `tensorrt_llm/_torch/modeling_v2/catalog/attention/mla_rope_append_paged_kv_assign_q.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
MlaRopeAppendPagedKvAssignQ.reference
```

### `tensorrt_llm/_torch/modeling_v2/catalog/attention/thop_attention.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
ThopAttention.softmax_stats
```

### `tensorrt_llm/_torch/modeling_v2/catalog/comm/allreduce.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
Allreduce.is_valid
Allreduce.reference
```

### `tensorrt_llm/_torch/modeling_v2/catalog/comm/mnnvl_allreduce.py`

库存函数 6；已覆盖 3；未覆盖 3。

```text
MnnvlAllreduce.__call__
MnnvlAllreduce.is_valid
MnnvlAllreduce.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/beam/_tp4/component_bf16.py`

库存函数 32；已覆盖 24；未覆盖 8。

```text
BeamCheckpointLayoutBf16Sm103Tp4._bank
BeamCheckpointLayoutBf16Sm103Tp4._layer
BeamCheckpointLayoutBf16Sm103Tp4.bank
BeamCheckpointLayoutBf16Sm103Tp4.block
BeamCheckpointLayoutBf16Sm103Tp4.dense_ffn
BeamCheckpointLayoutBf16Sm103Tp4.final_norm
BeamCheckpointLayoutBf16Sm103Tp4.route
_check_ids
```

### `tensorrt_llm/_torch/modeling_v2/models/beam/full_model_nvfp4__sm_103__tp4/component.py`

库存函数 14；已覆盖 8；未覆盖 6。

```text
BeamCheckpointLayoutNvfp4Sm103Tp4._load_expert
BeamCheckpointLayoutNvfp4Sm103Tp4._load_scalars
BeamCheckpointLayoutNvfp4Sm103Tp4.bank
BeamCheckpointLayoutNvfp4Sm103Tp4.expert_bank
BeamCheckpointLayoutNvfp4Sm103Tp4.moe.<locals>.provider
_f32_bits
```

### `tensorrt_llm/_torch/modeling_v2/models/beam/full_model_nvfp4__sm_103__tp4/layout.py`

库存函数 21；已覆盖 19；未覆盖 2。

```text
QuantSpec.family
checkpoint_dir_of
```

### `tensorrt_llm/_torch/modeling_v2/models/beam/full_model_nvfp4__sm_103__tp4/model.py`

库存函数 5；已覆盖 2；未覆盖 3。

```text
BeamFullModelNvfp4Sm103Tp4.__init__
BeamFullModelNvfp4Sm103Tp4._select_rows
BeamFullModelNvfp4Sm103Tp4.check_call
```

### `tensorrt_llm/_torch/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_10x__default/modeling.py`

库存函数 44；已覆盖 29；未覆盖 15。

```text
DecodeTarget.mtp_forward
DeepseekV3DefaultModelingV2._mtp_kv_layer
DeepseekV3DefaultModelingV2._mtp_pad_rows
DraftModel.__init__
DraftModel.embed_tokens
MTPLayer.__call__
MTPLayer.__init__
MTPLayer.forward
MTPLayer.shared_head
PrefillTarget.mtp_forward
_mtp_experts
_mtp_shared
_mtp_tail
_mtp_views
_residual_norm.<locals>.across_trays
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/dims.py`

库存函数 10；已覆盖 9；未覆盖 1。

```text
Dims.intermediate_local
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/modeling.py`

库存函数 47；已覆盖 44；未覆盖 3。

```text
ModelingV2Glm53FlashFp8Sm10xTp4.multimodal_token_ids
ModelingV2Glm53FlashFp8Sm10xTp4.set_guided_decoder
_VocabParallelHead.weight
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/attention/compact_sparse_rows.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_compact_sparse_rows_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/attention/head_major.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
_quantize_kernel
_saturate_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/attention/kpool_expand.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
_kpool_expand_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/attention/kpool_score.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_kpool_score_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/attention/kpool_select.py`

库存函数 9；已覆盖 3；未覆盖 6。

```text
_in_prefix
_kpool_select_kernel
_order_key
_radix_step
_select_chunked
_select_in_registers
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/attention/kpool_update.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_kpool_update_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/attention/latent_values.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_latent_values_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/attention/mla_context_prologue.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
_context_prologue_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/attention/mla_prologue.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
_prologue_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/comm/mhc_ar_boundary.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
MhcArBoundary.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/moe/grouped_blockwise_gemm.py`

库存函数 15；已覆盖 10；未覆盖 5。

```text
Sm100GroupedBlockwiseGemmKernel.__call__
Sm100GroupedBlockwiseGemmKernel._tile_expert
Sm100GroupedBlockwiseGemmKernel.kernel
Sm100GroupedBlockwiseGemmKernel.wrapper_gathered
Sm100GroupedBlockwiseGemmKernel.wrapper_grouped
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/moe/prefill_moe.py`

库存函数 12；已覆盖 8；未覆盖 4。

```text
_act_kernel
_combine_kernel
_place_kernel
_topk_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/norm/rms_norm_gated_token_major.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
_rms_norm_gated_fwd_multirow_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/ssm/reset_recurrent_state_rows.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_reset_recurrent_state_rows_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/schedule.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
LayerSchedule.num_layers
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_100__default/modeling.py`

库存函数 69；已覆盖 63；未覆盖 6。

```text
Glm5NextDecoderLayer.forward
Glm5NextForCausalLM.get_model_defaults
Glm5NextForCausalLM.get_preferred_kv_cache_manager_version
Glm5NextForCausalLM.get_preferred_transceiver_runtime
Glm5NextForCausalLM.num_hidden_layers
Glm5NextIndexer.cache_state_dim
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_100__default_mtp_fused/modeling.py`

库存函数 81；已覆盖 74；未覆盖 7。

```text
Glm5NextDecoderLayer.forward
Glm5NextForCausalLM.get_model_defaults
Glm5NextForCausalLM.get_preferred_kv_cache_manager_version
Glm5NextForCausalLM.get_preferred_transceiver_runtime
Glm5NextForCausalLM.num_hidden_layers
Glm5NextForCausalLM.set_guided_decoder
Glm5NextIndexer.cache_state_dim
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/activation/mlp_dense_swiglu_n3072_bf16.py`

库存函数 10；已覆盖 8；未覆盖 2。

```text
Schedule.ctas_per_row
Schedule.threads
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/activation/mlp_prefill_swiglu_mxf8.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/attention/filtered_top_k_varlen_util.py`

库存函数 11；已覆盖 4；未覆盖 7。

```text
FilteredTopKKernelVarlen._fill_oob
FilteredTopKKernelVarlen.filtered_topk_kernel_per_row
FilteredTopKKernelVarlen.predicate_tile
FilteredTopKKernelVarlen.prefix_sum_and_find_threshold_coarse
FilteredTopKKernelVarlen.prefix_sum_and_find_threshold_fine_grained
FilteredTopKKernelVarlen.to_coarse_key
FilteredTopKKernelVarlen.to_ordered
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/attention/fmha_helpers.py`

库存函数 36；已覆盖 9；未覆盖 27。

```text
FmhaStaticTileScheduler.__extract_mlir_values__
FmhaStaticTileScheduler.__init__
FmhaStaticTileScheduler.__new_from_mlir_values__
FmhaStaticTileScheduler.advance_to_next_work
FmhaStaticTileScheduler.check_valid_work_for_seqlen_q
FmhaStaticTileScheduler.get_current_work
FmhaStaticTileScheduler.initial_work_tile_info
FmhaStaticTileScheduler.set_current_work_linear_idx
FusedMask.apply_mask
FusedMask.get_leading_mask_id
FusedMask.get_masked_info
FusedMask.get_masked_leading_count
FusedMask.get_masked_trailing_count
FusedMask.get_trailing_mask_id
FusedMask.get_trip_start
FusedMask.get_unmasked_trip_count
_band_mask_32_regs_setp_selp
_causal_mask_32_regs_setp_selp
_map_compact_ragged_fmha_work
_warp_inclusive_prefix_sum_i32
create_fmha_static_tile_scheduler
ex2_emulation_packed_f32x2
ex2_emulation_packed_f32x2.<locals>.combine_int_frac_ex2
ex2_emulation_packed_f32x2.<locals>.polynomial_deg3_packed_f32x2
fmha_work_tile_info_from_clc_response
fmha_work_tile_info_from_dynamic_response
with_collector_op
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/attention/fmha_prefill_d256.py`

库存函数 15；已覆盖 8；未覆盖 7。

```text
FusedMultiHeadAttentionPrefillD256.__call__
FusedMultiHeadAttentionPrefillD256.can_implement
FusedMultiHeadAttentionPrefillD256.correction_epilog
FusedMultiHeadAttentionPrefillD256.correction_rescale
FusedMultiHeadAttentionPrefillD256.kernel
FusedMultiHeadAttentionPrefillD256.mma_pv
FusedMultiHeadAttentionPrefillD256.softmax_step
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/attention/self_attn_indexer_q_side_fused.py`

库存函数 17；已覆盖 16；未覆盖 1。

```text
QSideFusedConfig.threads
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/attention/self_attn_indexer_topk_decode.py`

库存函数 24；已覆盖 12；未覆盖 12。

```text
_make_decode_cluster_topk_class.<locals>.IndexerTopKDecodeCluster.__call__
_make_decode_cluster_topk_class.<locals>.IndexerTopKDecodeCluster.chunk_pass
_make_decode_cluster_topk_class.<locals>.IndexerTopKDecodeCluster.chunk_visit
_make_decode_cluster_topk_class.<locals>.IndexerTopKDecodeCluster.collect_at_or_above
_make_decode_cluster_topk_class.<locals>.IndexerTopKDecodeCluster.finish_candidates
_make_decode_cluster_topk_class.<locals>.IndexerTopKDecodeCluster.histogram_topk_decode_cluster_kernel
_make_decode_cluster_topk_class.<locals>.IndexerTopKDecodeCluster.resolve_list
_make_decode_cluster_topk_class.<locals>.IndexerTopKDecodeCluster.select_topk_row_cluster
_make_decode_cluster_topk_class.<locals>.IndexerTopKDecodeCluster.select_topk_row_guess
_make_decode_cluster_topk_class.<locals>.IndexerTopKDecodeCluster.select_topk_row_head
_make_decode_histogram_topk_class.<locals>.IndexerTopKDecodeHistogram.__call__
_make_decode_histogram_topk_class.<locals>.IndexerTopKDecodeHistogram.histogram_topk_decode_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/attention/self_attn_indexer_topk_histogram.py`

库存函数 23；已覆盖 2；未覆盖 21。

```text
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.__call__
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram._fill_oob
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.allocate_row_smem
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.extract_bin
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.find_threshold_bin
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.first_varying_digit
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.histogram_add
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.histogram_topk_prefill_kernel
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.is_partial_match
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.key_range_of_list
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.predicate_tile
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.process_step
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.refine_candidates
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.refine_digit
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.refine_digit_params
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.select_tied_in_column_order
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.select_topk_row
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.smem_bytes
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.sort_final_candidates
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.to_ordered
_make_prefill_histogram_topk_class.<locals>.IndexerTopKPrefillHistogram.warp_append
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/attention/self_attn_indexer_topk_prefill.py`

库存函数 18；已覆盖 15；未覆盖 3。

```text
_make_prefill_varlen_topk_class.<locals>.IndexerTopKPrefillVarlen.__call__
_make_prefill_varlen_topk_class.<locals>.IndexerTopKPrefillVarlen.filtered_topk_prefill_kernel
_make_prefill_varlen_topk_class.<locals>.IndexerTopKPrefillVarlen.to_coarse_key
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/attention/self_attn_prefill_mha_kv_gather_mxf8.py`

库存函数 14；已覆盖 13；未覆盖 1。

```text
dequantize
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/attention/self_attn_prefill_mha_varlen_h16_d256_bf16.py`

库存函数 29；已覆盖 17；未覆盖 12。

```text
GlmPrefillMhaVarlenD256.__call__
GlmPrefillMhaVarlenD256._launch
GlmPrefillMhaVarlenD256.correction_epilog_q8
GlmPrefillMhaVarlenD256.correction_epilog_stage
GlmPrefillMhaVarlenD256.correction_epilog_tail
GlmPrefillMhaVarlenD256.correction_rescale
GlmPrefillMhaVarlenD256.epi_rows
GlmPrefillMhaVarlenD256.epi_signal_end
GlmPrefillMhaVarlenD256.kernel
GlmPrefillMhaVarlenD256.softmax_step
GlmPrefillMhaVarlenD256Fp8.__call__
GlmPrefillMhaVarlenD256Fp8Q8.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/attention/self_attn_prefill_mla_rope_append_fp8q_h16_kv512_rope64_bf16_fp8.py`

库存函数 33；已覆盖 31；未覆盖 2。

```text
_load_e4m3x8_f32
compile
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/comm/tp4_allreduce_fused_rms.py`

库存函数 40；已覆盖 34；未覆盖 6。

```text
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_kernel
_glm_tp_allreduce_residual_rms_norm_twoshot_nvls_kernel
_multimem_ld_reduce_bf16x8_words
_twoshot_row_epilogue
_twoshot_stage_group
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/common/hidden_rmsnorm_add_h6144_bf16.py`

库存函数 25；已覆盖 24；未覆盖 1。

```text
rmsnorm_rows
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/gemm/_projection_module_dispatch.py`

库存函数 15；已覆盖 14；未覆盖 1。

```text
PersistentCompiler.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/gemm/_projection_tcgen05_mxf8_template.py`

库存函数 28；已覆盖 26；未覆盖 2。

```text
Mxf8ProjectionConfig.local_stage_steps
Mxf8ProjectionConfig.weight_bytes
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/gemm/_projection_tcgen05_template.py`

库存函数 32；已覆盖 30；未覆盖 2。

```text
ProjectionConfig.n_tail
ProjectionConfig.out_dtype_name
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/gemm/dense_blockscaled_gemm_persistent.py`

库存函数 19；已覆盖 10；未覆盖 9。

```text
Sm100BlockScaledPersistentDenseGemmKernel.__call__
Sm100BlockScaledPersistentDenseGemmKernel.can_implement
Sm100BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size
Sm100BlockScaledPersistentDenseGemmKernel.is_valid_layouts
Sm100BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape
Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment
Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment.<locals>.check_contigous_128_alignment
Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment.<locals>.check_contigous_16B_alignment
Sm100BlockScaledPersistentDenseGemmKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/gemm/dense_gemm_persistent.py`

库存函数 25；已覆盖 21；未覆盖 4。

```text
PersistentDenseGemmKernel.__call__
PersistentDenseGemmKernel.check_mma_tiler_and_cluster_shape
PersistentDenseGemmKernel.check_mma_tiler_and_cluster_shape.<locals>.is_power_of_2
PersistentDenseGemmKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/gemm/mlp_prefill_dense_up_swiglu_mxf8.py`

库存函数 10；已覆盖 8；未覆盖 2。

```text
GlmMxf8DenseSwigluGemm.__call__
GlmMxf8DenseSwigluGemm.kernel_swiglu
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/gemm/moe_prefill_shared_up_swiglu_mxf8.py`

库存函数 12；已覆盖 9；未覆盖 3。

```text
GlmMxf8SwigluGemm.__call__
GlmMxf8SwigluGemm.kernel_swiglu
silu_rn
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/gemm/self_attn_decode_projections_mxf8.py`

库存函数 14；已覆盖 13；未覆盖 1。

```text
_emulate
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/gemm/self_attn_prefill_kv_b_bmm_bf16.py`

库存函数 14；已覆盖 12；未覆盖 2。

```text
_build_legacy
_build_legacy.<locals>.launcher
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/gemm/self_attn_prefill_projections_bf16.py`

库存函数 16；已覆盖 15；未覆盖 1。

```text
PrefillProjection.forward
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/gemm/self_attn_prefill_projections_mxf8.py`

库存函数 18；已覆盖 16；未覆盖 2。

```text
GlmMxf8Gemm.__call__
can_implement
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/gemm/tgv_gemm_dsl.py`

库存函数 11；已覆盖 4；未覆盖 7。

```text
TgvGemmKernel.__call__
TgvGemmKernel.can_implement
TgvGemmKernel.dma_a_warp
TgvGemmKernel.dma_b_warp
TgvGemmKernel.epilog_warp
TgvGemmKernel.kernel
TgvGemmKernel.mma_warp
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/device_bound_scheduler.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
DeviceBoundPersistentTileScheduler.create_n
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/moe_decode_l2_prefetch.py`

库存函数 10；已覆盖 8；未覆盖 2。

```text
_nanosleep
run_hold
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/moe_decode_router_partials.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
run_router_partials
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/moe_decode_topk_routing_rep.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
check_rank
run_fused_topk_routing_rep
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/moe_frontend_router.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
run_router_vec2
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/moe_frontend_zero.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
run_zero_rows
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/moe_prefill_nvfp4_gg2.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
sched_load_work_tile
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/moe_prefill_topk_routing_rep.py`

库存函数 11；已覆盖 10；未覆盖 1。

```text
buffers
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/moe_topk_routing.py`

库存函数 19；已覆盖 18；未覆盖 1。

```text
run_fused_topk_routing
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/moe_topk_routing_push.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
run_fused_topk_routing
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/moe_topk_routing_push_wide.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
run_fused_topk_routing_wide
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/moe_topk_routing_push_wide_rep.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
run_fused_topk_routing_wide_rep
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/mtp_moe_routing.py`

库存函数 18；已覆盖 10；未覆盖 8。

```text
_cooperative_cluster_leader_barrier
_launch_hierarchical_workspace
_launch_multipass_workspace
routing_cooperative_init_kernel
routing_hierarchical_dsmem_cooperative_kernel
routing_multipass_histogram_kernel
routing_multipass_offsets_kernel
routing_multipass_workspace_init_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/static_persistent_tile_scheduler.py`

库存函数 22；已覆盖 16；未覆盖 6。

```text
StaticPersistentTileScheduler.advance_to_next_work
StaticPersistentTileScheduler.get_current_work
StaticPersistentTileScheduler.initial_work_tile_info
StaticPersistentTileScheduler.num_tiles_executed
WorkTileInfo.is_valid_tile
WorkTileInfo.tile_idx
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/norm/self_attn_qkv_a_rmsnorm.py`

库存函数 41；已覆盖 39；未覆盖 2。

```text
_ctx_locate_request
_require_bf16_cuda_contiguous
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/quantization/moe_frontend_quantize_nvfp4.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
run_quantize_linear
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/spec/draft_loop.py`

库存函数 10；已覆盖 5；未覆盖 5。

```text
_accept_kernel
_draft_step_kernel
_finalize_kernel
_schedule_probe_kernel
_write_schedule
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/modeling.py`

库存函数 154；已覆盖 144；未覆盖 10。

```text
GlmMoeDsaTep4ModelingV2._kv_b_augmented
GlmMoeDsaTep4ModelingV2._mtp_experts
GlmMoeDsaTep4ModelingV2._mtp_routed
GlmMoeDsaTep4ModelingV2._route
PrefillTarget._dense_causal_absorbed
_DcGroupsUploader.__call__
_DcGroupsUploader.__init__
_DcGroupsUploader._grow
_dc_bounds_nosync
_dc_groups_host
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/attention/mla_rope_append_paged_kv_assign_q.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
MlaRopeAppendPagedKvAssignQ.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/comm/_lamport.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
check_rows
check_rows_by_numel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/comm/cute_dsl_allgather_widen.py`

库存函数 4；已覆盖 1；未覆盖 3。

```text
CuteDslAllgatherWiden.is_valid
CuteDslAllgatherWiden.reference
takes
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/comm/cute_dsl_allreduce.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
CuteDslAllreduce.is_valid
CuteDslAllreduce.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/comm/cute_dsl_allreduce_add_rmsnorm.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
CuteDslAllreduceAddRmsnorm.is_valid
CuteDslAllreduceAddRmsnorm.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/comm/cute_dsl_allreduce_add_rmsnorm_quant_nvfp4.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
CuteDslAllreduceAddRmsnormQuantNvfp4.is_valid
CuteDslAllreduceAddRmsnormQuantNvfp4.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/moe/cute_dsl_moe_l2_prefetch.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
CuteDslMoeL2Prefetch.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/moe/cute_dsl_moe_prefill_topk_routing_rep.py`

库存函数 10；已覆盖 9；未覆盖 1。

```text
CuteDslMoePrefillTopkRoutingRep.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/moe/cute_dsl_moe_topk_routing_rep.py`

库存函数 10；已覆盖 9；未覆盖 1。

```text
CuteDslMoeTopkRoutingRep.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/moe/cute_dsl_moe_topk_routing_wide_rep.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
CuteDslMoeTopkRoutingWideRep.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/spec_worker.py`

库存函数 34；已覆盖 32；未覆盖 2。

```text
DraftLoopVerify.step
Glm52MTPEagleWorker._prepare_next_new_tokens_builtin
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/fake_ops.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
_
_host_query.<locals>.fake
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/_indexer.py`

库存函数 61；已覆盖 2；未覆盖 59。

```text
Glm52Indexer.__init__
Glm52Indexer._call_mqa_logits
Glm52Indexer._call_paged_mqa_logits
Glm52Indexer._copy_reused_topk
Glm52Indexer._decode_mxfp4
Glm52Indexer._fill_decode_topk
Glm52Indexer._gather_k_cache_for_chunk
Glm52Indexer._mtp_last_accepted_rows
Glm52Indexer._prep_q_or_k
Glm52Indexer._qk_projection_and_rope
Glm52Indexer._store_mtp_shared_rows
Glm52Indexer._update_k_cache
Glm52Indexer._weight_scale
Glm52Indexer.build_indexer_params
Glm52Indexer.cache_derived_state
Glm52Indexer.forward
Glm52Indexer.forward_from_projected
Glm52Indexer.forward_prepared
Glm52Indexer.maybe_join_prev_topk_copy
Glm52Indexer.post_load_weights
Glm52Indexer.pre_indexer_proj
Glm52Indexer.prepare
Glm52Indexer.prepare_for_chunked_prefill
Glm52Indexer.prepare_for_update_k_cache
Glm52Indexer.prepare_one_prefill_chunk
Glm52Indexer.prepare_scheduler_metadata
Glm52Indexer.recompute_context_kv_gather_mappings
Glm52Indexer.recompute_slot_mappings
Glm52Indexer.sparse_attn_indexer
IndexerParams.__post_init__
IndexerParams.all_kv_tokens
IndexerParams.batch_size
IndexerParams.block_stride
IndexerParams.cached_kv_tokens
IndexerParams.kv_lens
IndexerParams.new_kv_tokens
IndexerParams.scale_size
IndexerParams.total_tokens
_atom_ctx_offsets
_compute_slot_mappings
_effective_compress_ratio_divisor
_packed_logits_elem_budget
_pick_dsl_expand
_pick_dsl_split
_prefill_q_split_threshold
_scale
_select_indexer_compress_ratio
_split_prefill_queries
_tf32_matmul_enabled
_to_float
_unravel_indices
compute_cu_seqlen_kv_bounds_with_cache
group_prefill_requests_by_logits_budget
plan_mla_chunked_prefill_chunks
plan_mla_chunked_prefill_chunks.<locals>.flush_full_run
rotate_activation
select_indexer_chunk_size
split_prefill_chunks
use_cute_dsl_paged_mqa_logits
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/attention/backend.py`

库存函数 7；已覆盖 3；未覆盖 4。

```text
RubinDSATrtllmAttention.__init__
RubinDSATrtllmAttention.mla_rope_append_paged_kv_assign_q
RubinDSATrtllmAttention.sparse_kv_predict
_get_nvfp4_mla_kv_cache_amax
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/ops/moe/megamoe.py`

库存函数 11；已覆盖 1；未覆盖 10。

```text
RubinMegaMoE.__init__
RubinMegaMoE._alloc_symm_provider
RubinMegaMoE._direct_stage_topk_slots
RubinMegaMoE._launch_megamoe_kernel
RubinMegaMoE._run_moe
RubinMegaMoE.defer_next_topk_reduce
RubinMegaMoE.prepare_direct_stage
RubinMegaMoE.quantize_input_into
RubinMegaMoE.route_and_quantize_into
create_rubin_moe
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_full_context.py`

库存函数 5；已覆盖 2；未覆盖 3。

```text
FullContextIndexer.reuse_draft_selection
FullContextIndexer.select
FullContextIndexer.select.<locals>.topk_select
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_mtp.py`

库存函数 11；已覆盖 1；未覆盖 10。

```text
Glm52MTP.__init__
Glm52MTP.forward
Glm52MTP.forward.<locals>.norm_embeds
Glm52MTP.forward.<locals>.norm_hidden
Glm52MTPForCausalLM.__init__
Glm52MTPForCausalLM._quantization_named_modules
Glm52MTPForCausalLM.apply_quant_config
Glm52MTPHead.__init__
Glm52MTPHead.forward
Glm52MTPHead.get_last_token_states
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_mtp_config.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
_replacement_mtp_model_config
_set_draft_kv_cache_quant_algo
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_mtp_worker.py`

库存函数 10；已覆盖 3；未覆盖 7。

```text
MTPDraftWorker.__init__
MTPDraftWorker._draft_on_target
MTPDraftWorker._drafting_target
MTPDraftWorker._prepare_next_new_tokens
MTPDraftWorker._sample_and_accept_draft_tokens_base
MTPDraftWorker._sample_tokens_for_batch
MTPDraftWorker.prepare_1st_drafter_inputs
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/_small_batch_draft.py`

库存函数 13；已覆盖 7；未覆盖 6。

```text
SmallBatchDraftKernels._moe_postnorm
SmallBatchDraftKernels._moe_postnorm.<locals>.routed
SmallBatchDraftKernels.supports
_bf16_matrix
_ordinary_norm
_static_layout_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/fake_ops.py`

库存函数 25；已覆盖 1；未覆盖 24。

```text
_glm_dense_mlp_small_m
_glm_draft_ehproj_table
_glm_draft_sample_glue
_glm_dsa_hist_shard_meta
_glm_dsa_hist_topk
_glm_fused_a_gemm
_glm_fused_embedding
_glm_intra_node_moe_finalize
_glm_intra_node_reduce
_glm_lowm_bf16_moe
_glm_megakernel
_glm_megakernel_indexer_tmap
_glm_megakernel_weight_table
_glm_mla_kv_half_append
_glm_mnnvl_allgather
_glm_mnnvl_hop
_glm_mnnvl_moe_finalize_hop
_glm_pre_moe_fp4_linear
_glm_router_gemm
_glm_sparse_mla_flash_decode
_glm_sparse_mla_flash_decode_latent
_glm_verify_head_accept
_glm_verify_head_sample_tokens
_host_query.<locals>.fake
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/modeling.py`

库存函数 49；已覆盖 1；未覆盖 48。

```text
DecodeTarget.__init__
DecodeTarget._attention
DecodeTarget._attention.<locals>.kv_half
DecodeTarget._attention.<locals>.kv_half_then_q_b
DecodeTarget._attention.<locals>.q_b
DecodeTarget._attention.<locals>.select
DecodeTarget._attention_builtin
DecodeTarget._builtin_layer
DecodeTarget._check_attention
DecodeTarget._check_shared_expert
DecodeTarget._draft_moe
DecodeTarget._draft_moe.<locals>.routed
DecodeTarget._fast
DecodeTarget._fold_attention_weights
DecodeTarget._hop
DecodeTarget._lm_head_shard
DecodeTarget._moe
DecodeTarget._moe.<locals>.routed
DecodeTarget._post_attention_hop
DecodeTarget._prepare_tp8_rows
DecodeTarget._select_rows
DecodeTarget._shared_expert
DecodeTarget._stash_selection
DecodeTarget.build_draft
DecodeTarget.draft_ehproj_table
DecodeTarget.draft_first_step
DecodeTarget.draft_head
DecodeTarget.draft_next_step
DecodeTarget.draft_sample_glue
DecodeTarget.drafts
DecodeTarget.forward
DecodeTarget.logits
DecodeTarget.supports_small_batch_draft
Glm52Nvfp4Sm107LowLatencyCore.__init__
Glm52Nvfp4Sm107LowLatencyCore._select_target
Glm52Nvfp4Sm107LowLatencyCore.build_draft
Glm52Nvfp4Sm107LowLatencyCore.decode_target
Glm52Nvfp4Sm107LowLatencyCore.forward
Glm52Nvfp4Sm107LowLatencyCore.post_load_weights
ModelingV2Glm52Nvfp4Sm107LowLatency.__init__
ModelingV2Glm52Nvfp4Sm107LowLatency.forward
ModelingV2Glm52Nvfp4Sm107LowLatency.get_preferred_kv_cache_manager_version
ModelingV2Glm52Nvfp4Sm107LowLatency.get_preferred_transceiver_runtime
ModelingV2Glm52Nvfp4Sm107LowLatency.load_weights
ModelingV2Glm52Nvfp4Sm107LowLatency.post_load_weights
ModelingV2Glm52Nvfp4Sm107LowLatency.setup_aliases
PrefillTarget.__init__
PrefillTarget.forward
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/dsa/_fp8_paged_mqa_logits_kernel.py`

库存函数 12；已覆盖 5；未覆盖 7。

```text
FP8MQALogitsKernel.__call__
FP8MQALogitsKernel.kernel
add_f16x2
fma_f16x2
max_f16x2
pack_f16x2
unpack_f16x2
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/dsa/_hist_topk.py`

库存函数 5；已覆盖 2；未覆盖 3。

```text
create_hist_topk_workspace
get_hist_topk_workspace
hist_topk_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/dsa/_indexer_prologue.py`

库存函数 10；已覆盖 4；未覆盖 6。

```text
_apply_rope
_dsa_indexer_prologue_fused_kernel
_fp8_row_scale
_rcp_approx_ftz
_rope_terms
_unravel_offset
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/dsa/_mtp_topk_stash.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_stash_last_accepted_rows_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/dsa/hist_topk.py`

库存函数 6；已覆盖 1；未覆盖 5。

```text
GlmHistTopk.__call__
GlmHistTopk.is_valid
GlmHistTopk.reference
canonical_rows
ordered_keys
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/gemm/_low_m_splitk.py`

库存函数 59；已覆盖 49；未覆盖 10。

```text
SplitKDenseGemmKernel.__call__
SplitKDenseGemmKernel._drain_producer
SplitKDenseGemmKernel._publish_tile
SplitKDenseGemmKernel.dma_warp
SplitKDenseGemmKernel.epilog_warp
SplitKDenseGemmKernel.hnorm_b_warp
SplitKDenseGemmKernel.kernel
SplitKDenseGemmKernel.mma_warp
_bmm_bias
bmm_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/gemm/_low_m_splitk_fused.py`

库存函数 48；已覆盖 35；未覆盖 13。

```text
SplitKFusedEpilogueGemmKernel.__call__
SplitKFusedEpilogueGemmKernel._rms_out
SplitKFusedEpilogueGemmKernel._row_rstd
SplitKFusedEpilogueGemmKernel._store_tile
SplitKFusedEpilogueGemmKernel.epilog_fused
SplitKFusedEpilogueGemmKernel.kernel
SplitKIndexerQTailGemmKernel.__call__
SplitKIndexerQTailGemmKernel._qtail_prefetch
SplitKIndexerQTailGemmKernel._store_qtail
SplitKIndexerQTailGemmKernel.epilog_qtail
SplitKIndexerQTailGemmKernel.kernel
_st_global_v2
_st_global_v4
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/spec/_draft_exchange.py`

库存函数 8；已覆盖 2；未覆盖 6。

```text
DraftExchangeWorkspace.for_ehproj_table
DraftExchangeWorkspace.for_mapping
DraftExchangeWorkspace.for_sample_glue
ehproj_table_slot_bytes
ehproj_table_supports_tp
sample_glue_slot_bytes
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/spec/_draft_glue_rmsnorm.py`

库存函数 7；已覆盖 5；未覆盖 2。

```text
DraftGlueRMSNormKernel.__call__
DraftGlueRMSNormKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/spec/_draft_step_state.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
assert_state_args_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/spec/_gather_last_accepted_rows.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_gather_last_accepted_rows_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/spec/draft_ehproj_table.py`

库存函数 5；已覆盖 2；未覆盖 3。

```text
GlmDraftEhprojTable.__call__
GlmDraftEhprojTable.is_valid
GlmDraftEhprojTable.is_valid.<locals>.overlaps
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/ops/spec/draft_sample_glue.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
GlmDraftSampleGlue.__call__
GlmDraftSampleGlue.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/gpt_oss/gpt_oss_120b__sm_10x__default/modeling.py`

库存函数 29；已覆盖 28；未覆盖 1。

```text
_residual_norm.<locals>.across_trays
```

### `tensorrt_llm/_torch/modeling_v2/models/inkling2/config.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
InklingDims.is_block_boundary
InklingDims.num_block_states
```

### `tensorrt_llm/_torch/modeling_v2/models/inkling2/inkling_2_small_nvfp4__sm_103__tep4/cache.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
Inkling2KVCacheManager.free_resources
```

### `tensorrt_llm/_torch/modeling_v2/models/inkling2/inkling_2_small_nvfp4__sm_103__tep4/modeling.py`

库存函数 17；已覆盖 6；未覆盖 11。

```text
Inkling2SmallCore._all_reduce
Inkling2SmallCore._attention
Inkling2SmallCore._conv
Inkling2SmallCore._moe
Inkling2SmallCore._plan
Inkling2SmallCore._plan.<locals>.i32
Inkling2SmallCore.forward
Inkling2SmallCore.post_load_weights
ModelingV2Inkling2SmallNvfp4Sm103Tep4.forward
ModelingV2Inkling2SmallNvfp4Sm103Tep4.load_weights
ModelingV2Inkling2SmallNvfp4Sm103Tep4.post_load_weights
```

### `tensorrt_llm/_torch/modeling_v2/models/inkling2/inkling_2_small_nvfp4__sm_103__tep4/weights.py`

库存函数 32；已覆盖 31；未覆盖 1。

```text
layer_view
```

### `tensorrt_llm/_torch/modeling_v2/models/inkling2/kernels/mxfp8_moe.py`

库存函数 6；已覆盖 2；未覆盖 4。

```text
_dequant_tile
_gemm1_kernel
_gemm2_kernel
_round_e4m3
```

### `tensorrt_llm/_torch/modeling_v2/models/inkling2/kernels/paged_attention.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
_append_kv_kernel
_attention_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/inkling2/kernels/short_conv.py`

库存函数 4；已覆盖 1；未覆盖 3。

```text
_conv_out_kernel
_update_state_kernel
_window_entry
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/attn_res.py`

库存函数 13；已覆盖 12；未覆盖 1。

```text
KimiK3RMSNorm.forward
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/common/multi_stream.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
_record_stream
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/dspark/drafter.py`

库存函数 75；已覆盖 50；未覆盖 25。

```text
K3DSparkDrafter._draft_fused_a_proj
K3DSparkDrafter._fused_ar_residual_norm
K3DSparkDrafter._mla_freqs_real
K3DSparkDrafter._mla_tail_v2_supported
K3DSparkDrafter._reduce_partial
_DSparkHeadMixin.apply_markov_chain_logits
_DSparkHeadMixin.has_markov_head
_PendingDown.partial
_all_reduce_add_norm
_draft_mlp
_dspark_markov_chain_logits
_dspark_markov_chain_logits.<locals>._step_bias
_helix_nccl_post_alltoall
_helix_nccl_pre_alltoall
_helix_note_nccl_specialization
_helix_post_process
_helix_sanitize_empty_kv
_rmsnorm
_rope_last_dims_batched
apply_dspark_mla_rope
apply_dspark_rotary_batched
dspark_markov_chain
dspark_markov_step_bias
get_masked_input_and_mask
markov_prev_embeddings
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/dspark/kernels/dspark_rmsnorm_rope.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
DSparkRMSNormRoPEKernel.__call__
DSparkRMSNormRoPEKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/dspark/primitives.py`

库存函数 8；已覆盖 1；未覆盖 7。

```text
DraftArSites.__init__
DraftArSites.linear_ar_norm
HelixDraftShard.local_lengths
HelixDraftShard.local_slots
combine_helix_partials
draft_embed_site
push_tactic
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/dspark/shell.py`

库存函数 9；已覆盖 6；未覆盖 3。

```text
KimiK3SpecForCausalLM.forward
KimiK3SpecForCausalLM.load_draft_weights
KimiK3SpecForCausalLM.set_guided_decoder
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/dspark/worker.py`

库存函数 115；已覆盖 26；未覆盖 89。

```text
K3DFlashWorker._advance_ctx_len
K3DFlashWorker._apply_dflash2_selector
K3DFlashWorker._apply_kv_rewind_after_draft
K3DFlashWorker._apply_lilicorr
K3DFlashWorker._blk_owned_mask
K3DFlashWorker._block_gather_ids
K3DFlashWorker._check_block_within_allocation
K3DFlashWorker._check_no_real_generation
K3DFlashWorker._ctx_kv_pool_ptrs
K3DFlashWorker._ctx_latent_dtype
K3DFlashWorker._ctx_paged_index_args
K3DFlashWorker._dflash2_global_top_k
K3DFlashWorker._draft_first_slot
K3DFlashWorker._draft_slot_ids
K3DFlashWorker._draft_tokens_per_req
K3DFlashWorker._draft_vocab_parallel_head
K3DFlashWorker._ensure_spec_dec_state_restored
K3DFlashWorker._force_accept_kernel_args
K3DFlashWorker._forward_impl
K3DFlashWorker._fused_gen_drafter_inputs_applies
K3DFlashWorker._fused_gen_prepare_applies
K3DFlashWorker._fused_kv_rewind_operands
K3DFlashWorker._fused_prepare_constants
K3DFlashWorker._fused_select_accept
K3DFlashWorker._get_ctx_paged_append
K3DFlashWorker._init_ctx_block_tables
K3DFlashWorker._k3_vocab_parallel_draft_tokens
K3DFlashWorker._launch_ctx_side
K3DFlashWorker._prepare_attn_metadata_for_dflash
K3DFlashWorker._prepare_gen_drafter_inputs_fused
K3DFlashWorker._prepare_kv_for_draft_forward
K3DFlashWorker._prepare_next_new_tokens
K3DFlashWorker._prepare_paged_latent_gen_inputs
K3DFlashWorker._rebuild_batch_to_slot
K3DFlashWorker._refine_block_logits
K3DFlashWorker._restore_ctx_len_host
K3DFlashWorker._sample_and_accept_draft_tokens_base
K3DFlashWorker._sample_tokens_for_batch
K3DFlashWorker._store_context_kv_paged
K3DFlashWorker._vocab_parallel_head
K3DFlashWorker._vocab_parallel_markov_weights
K3DFlashWorker._vocab_parallel_target_selection_applies
K3DFlashWorker.bind_draft_lm_head
K3DFlashWorker.bind_target_lm_head
K3DFlashWorker.post_load_weights
K3DFlashWorker.prepare_1st_drafter_inputs
K3DFlashWorker.set_guided_decoder
K3DFlashWorker.setup_aliases
K3DFlashWorker.target_logits
K3DFlashWorker.target_vocab_parallel_head
K3DSparkWorker._apply_dspark_markov_bias
K3DSparkWorker._draft_block_width
K3DSparkWorker._draft_first_slot
K3DSparkWorker._draft_slot_ids
K3DSparkWorker._draft_vocab_parallel_head
K3DSparkWorker._refine_block_logits
K3DSparkWorker._vocab_parallel_markov_weights
K3VocabParallelHead.__init__
K3VocabParallelHead._exchange_and_reduce
K3VocabParallelHead._launch_local
K3VocabParallelHead.argmax
K3VocabParallelHead.gather_full_logits
K3VocabParallelHead.local_logits
K3VocabParallelHead.markov_tokens
K3VocabParallelHead.markov_tokens_chain
K3VocabParallelHead.markov_tokens_v3
K3VocabParallelHead.markov_v3_supported
K3VocabParallelHead.markov_w2_shard
K3VocabParallelHead.sample
K3VocabParallelHead.selection_pairs
K3VocabParallelHead.shard_logits
K3VocabParallelHead.supported
K3VocabShardedLogits.__init__
K3VocabShardedLogits.device
K3VocabShardedLogits.gather
K3VocabShardedLogits.select_pairs
K3VocabShardedLogits.select_tokens
PrivateMNNVLAllReduce.__init__
PrivateMNNVLAllReduce.fits
PrivateMNNVLAllReduce.rms_norm
_PairExchange.__init__
_PairExchange.chain_buffers
_PairExchange.exchange
_PairExchange.markov_v3_buffers
dflash_allocated_ctx_limit
dflash_draft_slot_ids
dflash_noise_block_embedding
get_dflash_paged_append
last_accepted_hidden
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_ctm_gemv/op.py`

库存函数 23；已覆盖 22；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_decode_gemv/op.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_embed/op.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/front_op.py`

库存函数 10；已覆盖 9；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/k3_moe_kernel.py`

库存函数 35；已覆盖 29；未覆盖 6。

```text
_ar_reduce_tile
_bf16_hi
_bf16_lo
_claim_ar_task
_device_sm_count
_read_globaltimer
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/k3_route_quant_ag.py`

库存函数 12；已覆盖 9；未覆盖 3。

```text
_atomic_add_acq_rel
k3_route_quant_ag
k3_route_quant_ag_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/latent_op.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/op.py`

库存函数 20；已覆盖 19；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_head_gemv/k3_head_gemv_kernel.py`

库存函数 19；已覆盖 13；未覆盖 6。

```text
_atomic_add_acq_rel
_atomic_add_relaxed
k3_head_gemv
k3_head_gemv_kernel
num_units
supports
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_head_gemv/op.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
_
pick_chunk
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_kda_attn/k3_kda_attn_kernel.py`

库存函数 24；已覆盖 23；未覆盖 1。

```text
_st_f32
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_kda_attn/op.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_kda_verify/op.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_mla/k3_mla_attn_kernel.py`

库存函数 12；已覆盖 11；未覆盖 1。

```text
softmax_scale_log2
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_mla/k3_mla_q_kernel.py`

库存函数 11；已覆盖 10；未覆盖 1。

```text
_st_async_v4
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_mla/op.py`

库存函数 20；已覆盖 19；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_mla/split_kv_plan.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
_mla_split_kv_plan_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_rms_norm_gated/op.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_route_quant/op.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_sandwich/k3_sandwich_kernel.py`

库存函数 32；已覆盖 29；未覆盖 3。

```text
_clock64
_ld_acquire_sys
_slab_word
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_sandwich/op.py`

库存函数 19；已覆盖 17；未覆盖 2。

```text
_
slab_tensor
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_situ_and_mul/op.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_attn_res/k3_attn_res.py`

库存函数 64；已覆盖 54；未覆盖 10。

```text
K3AttnResFastKernel.__call__
K3AttnResFastKernel.kernel
K3AttnResFastKernel.kernel.<locals>.load
K3AttnResFastKernel.kernel.<locals>.unpack
_run_lamport_add
_selp_b32
_sts_f32
k3_attn_res_lamport_add_rmsnorm
k3_attn_res_lamport_add_rmsnorm_into
k3_attn_res_lamport_add_rmsnorm_mx
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_comm/k3_lamport_ar.py`

库存函数 57；已覆盖 40；未覆盖 17。

```text
K3LamportFinishKernel.__call__
K3LamportFinishKernel.kernel
K3LamportFinishKernel.reduce_slots
K3LamportPushKernel.__call__
K3LamportPushKernel.kernel
arrivals_addr
arrive_count
arrive_finish
arrive_rotate
atom_add_acq_rel_u32
atom_add_relaxed_u32
clear_dirty_stages
finish_two_shot_v4
one_shot_for
poll_reduce_v4
reduce_chunk
two_shot_token
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_fp4_mla/fp4_mla_mufu16.py`

库存函数 136；已覆盖 18；未覆盖 118。

```text
_PreparedFusedCall.run
_add_f16x2
_arrive_smem_p4_mapped_leader_mbar
_as_shared_cta
_bitcast_i32_to_f32
_cache_fused_executor
_cache_prepared_fused_call
_ceil_div
_class_defines_callables
_compact_sfb_copy_cols_per_scale_chunk
_compile_fused
_compute_smem_p4_lazy_rebase_scale
_cross_arrive_smem_p4_cta2_mbar
_dealloc_tmem_cluster
_fence_async_shared_cta
_finalize_p4_n256_score_half_psf_source
_finalize_pipelined_tile
_finish_pipelined_score_loads
_float_to_ordered_u32_for_atomic_max
_get_fused_executor
_get_prepared_fused_call
_griddepcontrol_launch_dependents
_griddepcontrol_wait
_horizontal_pack_two_f16x2_sums
_issue_mxf4nvf4_mma_tile_from_base
_issue_p4_n256_psf_bank_to_tmem_from_desc
_issue_pipelined_score_loads
_issue_runtime_t336_qk_prefix_rank1
_issue_runtime_t336_qk_stage
_issue_smem_p4_pv_score_slot_from_base
_issue_smem_p4_pv_vsf_ntile_from_base
_issue_smem_p4_qk_ksf_range_from_base
_issue_smem_p4_qk_ksf_stage_from_desc
_issue_smem_p4_qk_score_slot_from_base
_issue_smem_p4_qk_stage_burst_from_base
_kv_cache_3d_layout
_load_global_page_native_tile_pair
_load_p4_n256_atomic_running_rowmax
_load_p4_n256_atomic_running_rowmax_with_validity
_load_p4_n256_score_half_from_tmem
_load_qk_konly_kblock_stage
_load_qk_qonly_kblock_stage
_load_runtime_t336_qk_tile
_load_runtime_t336_qk_tile_rank1
_load_smem_i32
_load_staged_page_native_tile_pair
_load_v_tile_stage
_lookup_physical_page
_make_fused_ptrs
_make_p4_n256_psf_bank_to_tmem_desc
_make_smem_p4_qk_ksf_stage_desc
_mapa_shared_cluster
_mask_invalid_score
_mbarrier_arrive_release_cta_shared_cluster
_mma_sf_ids_per_scale_atom
_ordered_u32_to_float_after_atomic_max
_p4_n256_prequant_rowsum_from_f16x2_psf
_p4_n256_rowsum_from_packed_psf_word
_pack_e2m1x8
_pack_e2m1x8_from_f16x2
_pack_e4m3x4_natural
_pack_e4m3x4_natural_from_f16x2
_pack_f32_pair_to_f16x2
_pack_p4_16_scores_from_group
_pack_p4_16_scores_from_group_packed_f16x2
_packed_request_tiles
_permute_raw_trtllm_q_tail_in_smem
_prepack_p4_n256_score_half_before_rowmax
_prepack_pipelined_tile
_prepare_p4_n256_score_half_tmem_addresses
_publish_p4_n256_atomic_rowmax
_qk_compact_ring_stage_byte_offset
_qk_sfb_cols_per_kblock
_reload_p4_n256_owned_psf_word_after_ready
_rescale_smem_p4_resident_o_in_tmem
_run_mla_decode_body
_run_pipelined_leading_tiles
_runtime_finalize_final_p4_n256_score_half_psf_source
_runtime_issue_pv_tile_dispatch
_runtime_issue_pv_tile_slot
_runtime_load_final_p4_n256_score_half_from_tmem
_runtime_refill_qk_tile_at
_runtime_refill_qk_tile_dispatch
_runtime_refill_qk_tile_state
_runtime_t336_prepare_p_source
_runtime_t336_producer_tile
_runtime_t336_producer_tile_v23
_scale_pack_e4m3x4_natural_from_f16x2
_select_runtime_kv_profile
_select_smem_p4_lazy_anchor_runtime
_select_smem_p4_lazy_anchor_with_validity
_sleep_until_producer_slot_free
_smem_atomic_max_ordered_u32
_smem_p4_direct_a_word_offset
_smem_p4_n256_psf_source_word_offset
_smem_p4_p4_materialize_group_col
_smem_p4_qk_score_group_col
_softmax_exp2_pair
_softmax_exp2_pair_packed_f16x2
_stage_smem_p4_page_id_plan
_stage_smem_p4_runtime_scale_pair
_store_final_o_from_tmem
_store_p4_n256_score_half_to_direct_a
_store_prepacked_p4_n256_score_half_to_direct_a
_store_raw_trtllm_qsf_128dp_unique_owner_to_tmem_from_smem
_store_smem_p4_n256_rowsum_part
_sum_p4_n256_psf_row_parts
_tcgen05_ld_red_32x32b_x16_max_f32
_tma_gather4_cluster
_unpack_f16x2_to_f32_pair
_validate_v_packed_cache_args
_warp_uniform_f32
fused_fp4_mla_decode_ctm
kernel
packed_query_rows
run_trtllm_fp4_mla_decode_packed
run_trtllm_fp4_mla_decode_page_native
run_trtllm_fp4_mla_decode_page_native_from_raw
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_fp4_mla/fp4_mla_packed_reduce.py`

库存函数 35；已覆盖 30；未覆盖 5。

```text
DeferredPackedReduce.__enter__
DeferredPackedReduce.__exit__
DeferredPackedReduce.__init__
DeferredPackedReduce.run
_griddepcontrol_wait
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_gemm/k3_fp8w_lowm_gemm.py`

库存函数 130；已覆盖 97；未覆盖 33。

```text
K3Fp8wLowMGemmKernel.__call__
K3Fp8wLowMGemmKernel._add_addend
K3Fp8wLowMGemmKernel._launch
K3Fp8wLowMGemmKernel._push_lamport
K3Fp8wLowMGemmKernel._push_owned_rows
K3Fp8wLowMGemmKernel.call_ar_push
K3Fp8wLowMGemmKernel.call_ar_push_gated
K3Fp8wLowMGemmKernel.call_gated
K3Fp8wLowMGemmKernel.call_split_ar_push
K3Fp8wLowMGemmKernel.cta
K3Fp8wLowMGemmKernel.epilogue
K3Fp8wLowMGemmKernel.kernel
K3Fp8wLowMGemmKernel.kernel_args
K3Fp8wLowMGemmKernel.mma_warp
K3Fp8wLowMGemmKernel.quantize_activations
K3Fp8wMidMGemmKernel.__call__
K3Fp8wMidMGemmKernel._launch
K3Fp8wMidMGemmKernel._push_staged_rows
K3Fp8wMidMGemmKernel.call_ar_push
K3Fp8wMidMGemmKernel.epilogue
K3Fp8wMidMGemmKernel.kernel
K3Fp8wMidMGemmKernel.mma_warp
K3MxFp8QuantKernel.__call__
K3MxFp8QuantKernel.kernel
_FusedStoreMixin._owned_token
_FusedStoreMixin.fused_store
_check_split_inputs
_compiled_split_kernel
_gated_tactic_fits
block_exponents_of
can_run_gated
run_k3_fp8w_lowm_gemm_split_ar_push
split_source_tactic
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_gemm/k3_fp8w_lowm_mxfp8.py`

库存函数 35；已覆盖 22；未覆盖 13。

```text
K3Fp8wLowMMxGemmKernel.__call__
K3Fp8wLowMMxGemmKernel._launch
K3Fp8wLowMMxGemmKernel._push
K3Fp8wLowMMxGemmKernel.call_ar_push
K3Fp8wLowMMxGemmKernel.epilogue
K3Fp8wLowMMxGemmKernel.kernel
K3Fp8wLowMMxGemmKernel.mma_warp
K3Fp8wLowMMxGemmKernel.scale_warp
_tmem_cp_128x256b
can_stage_weights
ring_holds_weights
staged_matches_mid_m
weights_on_chip
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_gemm/k3_latent_fp8_gemm.py`

库存函数 63；已覆盖 45；未覆盖 18。

```text
K3Fp8LatentGemmKernel.__call__
K3Fp8LatentGemmKernel._load_residual_block
K3Fp8LatentGemmKernel._store_nvfp4
K3Fp8LatentGemmKernel._store_residual
K3Fp8LatentGemmKernel.epilogue
K3Fp8LatentGemmKernel.kernel
K3Fp8LatentGemmKernel.mma_warp
K3Fp8LatentGemmKernel.producer
K3Fp8LatentGemmKernel.sfb_writer
K3Fp8LatentGemmKernel.tile_coords
K3Fp8Quant1x128Kernel.__call__
K3Fp8Quant1x128Kernel.kernel
_can_run
_f32x2_to_bf16x2
_is_supported_device
_lds_v4_b32
_residual_from_bf16x2
can_run
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_gemm/mxfp8_fp8_out_gemm.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
K3Sm107MXFP8Fp8OutGemmKernel.kernel_impl
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_helix/helix_a2a_fused.py`

库存函数 45；已覆盖 31；未覆盖 14。

```text
HelixA2AFusedKernel.__call__
HelixA2AFusedKernel.__init__
HelixA2AFusedKernel.kernel
HelixA2AFusedKernel.receive
HelixA2AFusedKernel.send
_bulk_commit
_bulk_s2g
_bulk_wait_read
_ceil_div
_compiled
_fence_proxy_async_shared
_ld_nc_u8
helix_a2a_fused
supports
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_helix/k3_helix_tail.py`

库存函数 44；已覆盖 24；未覆盖 20。

```text
K3HelixTailKernel.__call__
K3HelixTailKernel.dump_rows
K3HelixTailKernel.fold_tile
K3HelixTailKernel.kernel
K3HelixTailKernel.project_and_push
K3HelixTailKernel.project_transposed
K3HelixTailKernel.receive
K3HelixTailKernel.stamp
K3HelixTailKernel.store_mx
K3HelixTailKernel.take_half
K3HelixTailKernel.vup_k_half
_fold_half_ull2
_fold_round_loads
_fold_round_sums
_fold_store
_globaltimer
_lds_f32
_st_u64
_sts_b16
_sts_f32
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_kda/_kda_kernels.py`

库存函数 38；已覆盖 18；未覆盖 20。

```text
K3KDAKernelDispatch.decode_kda
_copy_kda_replay_conv_window_kernel
_default_get_sm_version
_extract_transpose_prefill_kernel
_fused_kda_conv_prefill_kernel
_fused_kda_post_conv_kernel
_kda_conv_prefill_edges_kernel
_kda_conv_silu
_kda_conv_tap
_kda_round
_kda_store_conv_rows
_kda_widen
_pointer_table
_reset_recurrent_state_rows_kernel
_seed_kda_replay_conv_layers_kernel
_shared_layout
_zero_kda_replay_slots_kernel
fused_kda_post_conv.<locals>.normalize
plan_kda_replay_seed
seed_kda_replay_layers
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_kda/akk_inverse.py`

库存函数 19；已覆盖 18；未覆盖 1。

```text
_invert_diag_neumann
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_kda/fused_k123.py`

库存函数 25；已覆盖 7；未覆盖 18。

```text
_ak_chain_mma_A
_ak_chain_mma_B
_ak_invert_diag_neumann
_ak_invert_diag_neumann_inreg
_ak_load_C_temp
_ak_mask_packed_ltri
_ak_matmul_AB
_ak_mma
_ak_movmatrix_trans
_ak_pack_C_to_A
_ak_pack_C_to_B
_ak_pack_bf16x2
_ak_store_C_temp
_ak_store_neg_C
_ak_unpack_bf16x2_hi
_ak_unpack_bf16x2_lo
make_host_function.<locals>.host_fn
store_internal_barrier
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_kda/k3_kda_fab.py`

库存函数 65；已覆盖 53；未覆盖 12。

```text
K3KdaFabKernel.__call__
K3KdaFabKernel.gemm1
K3KdaFabKernel.kernel
K3KdaFabUmmaKernel.__call__
K3KdaFabUmmaKernel.kernel
_FabTail.control_warp
_FabTail.make_w2_tma
_FabTail.probe
_FabTail.refill
_FabTail.tail
_FabTail.tail_fa
_num_sms
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_kda/k3_l2_prefetch.py`

库存函数 9；已覆盖 7；未覆盖 2。

```text
K3L2PrefetchKernel.__call__
K3L2PrefetchKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_kda/k4_persistent.py`

库存函数 19；已覆盖 7；未覆盖 12。

```text
_chain_mma_A_inv
_chain_mma_B_inv
_invert_diag
_load_C_temp_inv
_matmul_AB_inv
_shuffle_C_to_B_inv
_store_C_temp_inv
_store_neg_C_inv
inv_internal_barrier
make_host_fn.<locals>.host_fn
make_host_fn.<locals>.host_fn_snapshot
mma_tf32_m16n8k8
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_kda/kda_mtp_decode.py`

库存函数 12；已覆盖 10；未覆盖 2。

```text
read_globaltimer
swz128
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_kda/mtp_ops.py`

库存函数 23；已覆盖 22；未覆盖 1。

```text
_compact_layout_enabled
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_mla_ctx/fp8_static_quant.py`

库存函数 7；已覆盖 4；未覆盖 3。

```text
_e4m3x2_from_bf16x2
_static_quant_e4m3_kernel
_static_quant_e4m3_rows_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_mla_ctx/k3_context_fmha.py`

库存函数 22；已覆盖 6；未覆盖 16。

```text
RubinFusedMultiHeadAttentionForward.__call__
RubinFusedMultiHeadAttentionForward.apply_exp_and_cvt
RubinFusedMultiHeadAttentionForward.calculate_skip_softmax_flag
RubinFusedMultiHeadAttentionForward.can_implement
RubinFusedMultiHeadAttentionForward.check_invalid_shape
RubinFusedMultiHeadAttentionForward.check_supported_dtypes
RubinFusedMultiHeadAttentionForward.correction_epilog
RubinFusedMultiHeadAttentionForward.correction_rescale
RubinFusedMultiHeadAttentionForward.get_skip_softmax_flag
RubinFusedMultiHeadAttentionForward.kernel
RubinFusedMultiHeadAttentionForward.kernel.<locals>.make_tmem_tensors
RubinFusedMultiHeadAttentionForward.mma_pv
RubinFusedMultiHeadAttentionForward.mma_qk
RubinFusedMultiHeadAttentionForward.softmax
RubinFusedMultiHeadAttentionForward.softmax.<locals>.softmax_loop
RubinFusedMultiHeadAttentionForward.softmax_step
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_mla_ctx/k3_context_fmha_helpers.py`

库存函数 33；已覆盖 20；未覆盖 13。

```text
FusedMask.apply_mask
FusedMask.get_leading_mask_id
FusedMask.get_masked_info
FusedMask.get_masked_leading_count
FusedMask.get_masked_trailing_count
FusedMask.get_trailing_mask_id
FusedMask.get_trip_start
FusedMask.get_unmasked_trip_count
WorkTileInfo.is_valid_tile
WorkTileInfo.tile_idx
ex2_emulation_packed_f32x2
ex2_emulation_packed_f32x2.<locals>.combine_int_frac_ex2
ex2_emulation_packed_f32x2.<locals>.polynomial_deg3_packed_f32x2
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_mla_ctx/mla_kv_fp8.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
_e4m3x2_from_bf16x2
_mla_kv_fp8_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_mla_q/k3_mla_q_fused.py`

库存函数 69；已覆盖 52；未覆盖 17。

```text
K3MlaLatentRowKernel.__call__
K3MlaLatentRowKernel.kernel
K3MlaQFusedKernel.__call__
K3MlaQFusedKernel.absorb
K3MlaQFusedKernel.kernel
K3MlaQFusedKernel.load_tile
K3MlaQFusedKernel.prefetch_rope
K3MlaQFusedKernel.q_b_tile
K3MlaQFusedKernel.q_row
K3MlaQFusedKernel.quantize_row
K3MlaQFusedKernel.row_rstd
K3MlaQFusedKernel.stamp
K3MlaQFusedKernel.store_prefix_group
K3MlaQFusedKernel.store_rope_group
K3MlaQFusedKernel.wait
_stg_bf16
smem_bytes
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_mla_q/q_norm_pdl.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_LaunchReleaseRMSNormKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_moe/die_locality.py`

库存函数 23；已覆盖 20；未覆盖 3。

```text
_SmDieProbe._launch.<locals>.kernel
_SmDieProbe._launch.<locals>.launch
die_alternating_allocations
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_moe/fc12_kernel.py`

库存函数 73；已覆盖 42；未覆盖 31。

```text
L2AtomicPersistentTileScheduler._claim_next_linear_idx
L2AtomicPersistentTileScheduler.get_published_work
L2AtomicPersistentTileScheduler.publish_next_work
L2AtomicPersistentTileScheduler.publish_next_work.<locals>.publish_to_peer
PipelineCpAsyncUmma._compute_leading_cta_rank
PipelineCpAsyncUmma._compute_peer_cta_mask
SeparatePhaseSmemStageBytes.bsfb_stage_bytes
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.__call__
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._apply_situ_epilogue
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._apply_situ_epilogue.<locals>._sigmoid
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._apply_swiglu_epilogue
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._die_local_claim
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._get_tma_atom_kind
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._is_interleaved_utccp
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._mainloop_s2t_interleaved_copies
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._sm_die
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.can_implement
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.is_valid_dtypes_and_scale_factor_vec_size
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.is_valid_layouts
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.is_valid_mma_tiler_and_cluster_shape
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.is_valid_problem_shapes
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.is_valid_problem_shapes.<locals>.is_contiguous_16b_aligned
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.kernel
derive_fc12_cta_task_stream
derive_fc12_task_shape
derive_fc1_ready_expected
fclip_xorsign
sigmoid_f32
silu_f32
sm100_tcgen05_st_32x32b_x4
validate_l2_atomic_descriptor_bounds
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_moe/fc12_runner.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
K3Fc12Runner.forward.<locals>._fc12_wrapper
K3Fc12Runner.get_valid_tactics
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_moe/k3_fused_router.py`

库存函数 24；已覆盖 23；未覆盖 1。

```text
_ldsm_x4
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_moe/k3_route_tail.py`

库存函数 69；已覆盖 54；未覆盖 15。

```text
K3RouteTailKernel.__call__
K3RouteTailKernel.kernel
K3RouteTailKernel.select_token
K3RouteTailKernel.sort
K3RouteTailKernel.team_sync
K3RouteTailKernel.topk_rounds
K3RouteTailKernel.topk_token
K3RouteWideKernel.__call__
K3RouteWideKernel.kernel
K3RouteWideKernel.place_entry
K3RouteWideKernel.scan_cells
K3RouteWideKernel.wide_sort
_lds_b32_or_zero
can_use_k3_route_tail
wide_workspace_ready
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_moe/k3_router_exact.py`

库存函数 14；已覆盖 11；未覆盖 3。

```text
K3RouterExactKernel.__call__
K3RouterExactKernel._wfrag
K3RouterExactKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_moe/kimi_k3_moe_residual.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_moe_residual_add_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_rms_norm_gated/op.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_rms_norm_gated/rms_norm_gated_kernel.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
_layer_norm_fwd_1pass_kernel
_rms_norm_gated_fwd_multirow_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_situ/op.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_spec/dspark_markov_fused.py`

库存函数 5；已覆盖 1；未覆盖 4。

```text
_argmax_combine
_markov_add_argmax_kernel
_markov_prev_embedding_kernel
fused_markov_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_spec/dspark_markov_greedy.py`

库存函数 41；已覆盖 37；未覆盖 4。

```text
DSparkMarkovGreedyKernel.__call__
DSparkMarkovGreedyKernel.kernel
_ld_nc_b32
smem_bytes
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_spec/dspark_mla_rope.py`

库存函数 6；已覆盖 2；未覆盖 4。

```text
DSparkMlaBlockRoPEKernel.__call__
DSparkMlaBlockRoPEKernel.kernel
DSparkMlaCtxNormRoPEKernel.__call__
DSparkMlaCtxNormRoPEKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_spec/embedding_gather.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_embedding_gather_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_spec/spec_decode_head.py`

库存函数 7；已覆盖 5；未覆盖 2。

```text
_helix_local_len
_spec_head_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_spec/spec_tail_kernels.py`

库存函数 33；已覆盖 14；未覆盖 19。

```text
_ctx_positions_kernel
_decode_block_table_chunk_kernel
_decode_block_table_kernel
_dflash_gen_prepare_kernel
_fused_strict_accept_kernel
_next_new_tokens_kernel
_next_new_tokens_rewind_kernel
_noise_embedding_kernel
_paged_latent_prepare_kernel
_pair_argmax
_scatter_rows_kernel
_select_accept_kernel
_store_context_latents_kernel
_store_mla_context_kernel
_strict_accept_kernel
_sum_block_counts_kernel
_vocab_shard_embedding_kernel
strict_accept
vocab_shard_embedding_applies
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_spec/vocab_parallel_kernels.py`

库存函数 18；已覆盖 5；未覆盖 13。

```text
_argmax_combine
_argmax_pairs
_gdc_wait
_global_index
_gumbel
_local_argmax_kernel
_local_sample_kernel
_markov_add_local_argmax_kernel
_markov_chain_kernel
_markov_reduce_embed_kernel
_reduce_pairs_kernel
_store_pair
_wait_pairs_written
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_state_upload/state_upload_kernel.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_small_copy_pair_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/kda_state_rows/reset.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_reset_state_rows_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/decode_comm.py`

库存函数 12；已覆盖 10；未覆盖 2。

```text
K3DecodeComm.takes_tail
wide_all_reduce
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/decode_moe.py`

库存函数 14；已覆盖 11；未覆盖 3。

```text
K3DecodeMoeLayer._wide
K3DecodeMoeLayer._wide.<locals>._routed_partial
K3DecodeMoeLayer._wide.<locals>._shared_activation
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/kernels/k3_moe_engines/k3_moe_m2_kernel.py`

库存函数 17；已覆盖 15；未覆盖 2。

```text
_add_rn
_mul_rn
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/kernels/k3_moe_engines/op.py`

库存函数 26；已覆盖 25；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/modeling.py`

库存函数 29；已覆盖 17；未覆盖 12。

```text
KimiLinearDecoderLayer._dense_mlp
KimiLinearDecoderLayer._o_allreduce
KimiLinearDecoderLayer.accepts_moe_partial
KimiLinearDecoderLayer.skip_forward
KimiLinearModel._final_norm
ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1._build_decode_moe
ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1._check_step_contract
ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1.cache_derived_state
ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1.get_preferred_kv_cache_manager_version
ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1.load_weights
ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1.post_load_weights
_text_model_config
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/decode_comm.py`

库存函数 17；已覆盖 16；未覆盖 1。

```text
K3DecodeComm.takes_tail
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/decode_moe.py`

库存函数 13；已覆盖 11；未覆盖 2。

```text
K3DecodeMoeLayer.takes
layout_gaps
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/drafter.py`

库存函数 65；已覆盖 62；未覆盖 3。

```text
K3DSparkDrafter.apply_draft_logit_transform
K3DSparkDrafter.apply_markov_chain_logits
K3DSparkDrafter.reads_managed_ctx_pages
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/drafter_mla.py`

库存函数 34；已覆盖 31；未覆盖 3。

```text
K3MLADSparkDrafter._get_cos_sin_cache
K3MLADSparkDrafter.validate_block_attention_windows
_all_reduce_add_norm
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/kernels/k3_ctx_kv/op.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/kernels/k3_drafter/op.py`

库存函数 13；已覆盖 12；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/kernels/k3_dspark_markov_chain/k3_dspark_markov_chain_kernel.py`

库存函数 10；已覆盖 5；未覆盖 5。

```text
_global_index
_markov_add_local_argmax_kernel
_markov_reduce_embed_kernel
_store_pair
argmax_combine
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/kernels/k3_markov/op.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
_
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/kernels/k3_spec_accept/op.py`

库存函数 10；已覆盖 8；未覆盖 2。

```text
_
existing_workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/modeling.py`

库存函数 34；已覆盖 21；未覆盖 13。

```text
KimiLinearDecoderLayer._dense_mlp
KimiLinearDecoderLayer.accepts_moe_partial
KimiLinearDecoderLayer.skip_forward
KimiLinearModel._final_norm
KimiLinearModel._latent_push
KimiLinearModel.kda_draft_records
ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4._build_decode_moe
ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4._check_step_contract
ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4.cache_derived_state
ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4.get_preferred_kv_cache_manager_version
ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4.load_weights
ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4.post_load_weights
_text_model_config
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/spec_shell.py`

库存函数 7；已覆盖 5；未覆盖 2。

```text
forward
slice_spec_position_ids
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/spec_worker.py`

库存函数 32；已覆盖 29；未覆盖 3。

```text
KimiK3DSparkWorker._k3_mask_row
KimiK3DSparkWorker._k3_static_arange
KimiK3DSparkWorker._trained_mask_embedding
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__dep16/modeling.py`

库存函数 192；已覆盖 83；未覆盖 109。

```text
K3KDALinearAttention._forward_impl
K3KDALinearAttention.forward
K3KDALinearAttention.forward_decode
K3KDALinearAttention.forward_verify_context_only
K3MLAAttention.__init__
K3MLAAttention.__init__.<locals>.is_fp8
K3MLAAttention.__init__.<locals>.projection
K3MLAAttention._forward_attention
K3MLAAttention._gate_and_o_proj
K3MLAAttention._project_q_by_phase
K3MLAAttention._reject_unsupported
K3MLAAttention.cache_derived_state
K3MLAAttention.create_output
K3MLAAttention.create_weights
K3MLAAttention.forward
K3MLAAttention.post_load_weights
K3MLAAttention.project_qkv_a
K3MLAAttention.q_a_norm
K3MLAAttention.run_attention
K3MLAAttention.transform_weights
K3MLARuntime.__init__
K3MLARuntime.forward
K3MegaMoeNvfp4Experts.loaded_slots
K3MoERuntime._resolve_routed_quant_config
K3MoERuntime._routed_moe_model_config
K3MoERuntime._routed_projection
K3MoERuntime._select_moe_tp_ep
K3MoERuntime.finalize_expert_load
K3MoERuntime.forward
K3MoERuntime.forward.<locals>._routed_output
K3MoERuntime.forward.<locals>._shared_output
K3MoERuntime.prepare_expert_load
KimiK3DefaultDecoderLayer.__init__
KimiK3DefaultDecoderLayer._capture_hidden_states
KimiK3DefaultDecoderLayer.forward
KimiK3DefaultDecoderLayer.skip_forward
KimiK3DefaultForCausalLM.__init__
KimiK3DefaultModelingV2.__init__
KimiK3DefaultModelingV2.forward
ModelingV2KimiK3Nvfp4Sm107Dep16.__init__
ModelingV2KimiK3Nvfp4Sm107Dep16.draft_config
ModelingV2KimiK3Nvfp4Sm107Dep16.forward
ModelingV2KimiK3Nvfp4Sm107Dep16.get_preferred_kv_cache_manager_version
ModelingV2KimiK3Nvfp4Sm107Dep16.get_preferred_transceiver_runtime
ModelingV2KimiK3Nvfp4Sm107Dep16.infer_max_seq_len
ModelingV2KimiK3Nvfp4Sm107Dep16.lm_head
ModelingV2KimiK3Nvfp4Sm107Dep16.model
ModelingV2KimiK3Nvfp4Sm107Dep16.set_guided_decoder
ModelingV2KimiK3Nvfp4Sm107Dep16.vocab_size_padded
_DeferredMoEResidual.materialize
_Fp8BlockScaleWeightReadLinear.has_fp8_qdq
_Fp8BlockScaleWeightReadLinear.has_w4a8_nvfp4_fp8
_Fp8BlockScaleWeightReadLinear.quantize_weight
_K3SharedExpert.forward
_apply_attn_res
_apply_attn_res_add_and_rmsnorm
_apply_attn_res_add_rmsnorm_fused
_apply_attn_res_and_rmsnorm
_apply_attn_res_fused
_apply_attn_res_fused.<locals>.run
_apply_attn_res_rmsnorm_fused
_convert_moe_mlps_to_fp8_weight_read
_defer_moe_residual
_expand_kv_bf16
_expand_kv_fp8
_extract_kda_extra_attrs
_fp8_q_args
_get_text_config
_has_weights
_is_fp8_block_projection
_is_kda_layer
_is_mla_layer
_k3_attn_res_pdl
_k3_attn_res_pdl_moe_add
_k3_expert_ckpt_spec
_kimi_k3_fp8_kv_layer_ids
_kv_b_strided_out
_load_nvfp4_expert
_materialize_prefix_sum
_note_attn_res_fusion
_quantize_q_fp8
_registered_layer
_resolve_kimi_situ_betas
_swap_linear_to_fp8_weight_read
_use_persistent_attn_res
_warmup_expand_kv_bf16
_warmup_expand_kv_bf16.<locals>.warmup
_warmup_expand_kv_fp8
context_cached_kv
context_chunked_prefill
context_default
create_mla_outputs
forward_absorption_generation
forward_absorption_generation.<locals>.rope_and_cache_update
forward_absorption_generation.<locals>.w_uk
forward_context_split_k_fp4
forward_context_split_k_fp4.<locals>.expand_kv
forward_context_split_k_fp4.<locals>.run_partition
gen_attention
kda_core_inplace
kv_b_takes_mxfp8
load_fp4_mla_chunked_kv_cache_mxfp8
mla_custom_op_inplace
project_context_kv
resolve_attention_quant_config
run_k3_split_k_context_partition
split_k_context_unavailable_reason
supports_k3_fp8_split_k_context
uses_k3_split_k_context
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__dep16/weights.py`

库存函数 34；已覆盖 1；未覆盖 33。

```text
KimiK3DefaultTextWeights._attention_fp8_linears
KimiK3DefaultTextWeights._finalize_weight_load
KimiK3DefaultTextWeights._load_attention_fp8
KimiK3DefaultTextWeights._load_expert_slices
KimiK3DefaultTextWeights._load_expert_slices.<locals>.checkpoint_key
KimiK3DefaultTextWeights._load_expert_slices.<locals>.drop_file_pages
KimiK3DefaultTextWeights._load_expert_slices.<locals>.ensure_prepared
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_expert
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_expert_file
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_experts_from_weights
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_split_file_expert
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_split_file_expert.<locals>.get_optional_tensor
KimiK3DefaultTextWeights._load_expert_slices.<locals>.load_split_file_expert.<locals>.get_tensor
KimiK3DefaultTextWeights._load_expert_slices.<locals>.maybe_finalize_layer
KimiK3DefaultTextWeights._load_mla_fp8_kv_b
KimiK3DefaultTextWeights._load_trunk_params
KimiK3DefaultTextWeights._load_trunk_params.<locals>.load_param
KimiK3DefaultTextWeights._trunk_parameters
KimiK3DefaultTextWeights._validate_checkpoint_keys
KimiK3DefaultTextWeights.checkpoint_name_plan
KimiK3DefaultTextWeights.load_weights
KimiK3DefaultWeights.load_draft_weights
KimiK3DefaultWeights.load_weights
_checkpoint_fp8_pair
_dequantize_fp8_block_scaled
_fp8_block_scale_key
_full_projection_shape
_gate_up_ckpt_keys
_load_fp8_projection
_materialize
_projection_shard
_shard_head_major_param
_shards_itself
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__helix8_moeep8/modeling.py`

库存函数 258；已覆盖 69；未覆盖 189。

```text
HelixGenHooks.__init__
K3HelixTail.__call__
K3HelixTail.__init__
K3HelixTail.applies
K3HelixTail.get
K3KDALinearAttention._forward_impl
K3KDALinearAttention.forward
K3KDALinearAttention.forward_decode
K3KDALinearAttention.forward_verify_context_only
K3KDALinearAttention.l2_prefetch_ranges
K3KDALinearAttention.qkvg_mx_atom
K3KDALinearAttention.qkvg_stages_weights
K3KDALinearAttention.qkvg_takes_mxfp8
K3MLAAttention.__init__
K3MLAAttention.__init__.<locals>.is_fp8
K3MLAAttention.__init__.<locals>.projection
K3MLAAttention._bind_helix_v_up_weight
K3MLAAttention._build_helix_v_up_weight
K3MLAAttention._forward_attention
K3MLAAttention._forward_side_gate
K3MLAAttention._forward_side_gate.<locals>.gate_of
K3MLAAttention._forward_side_gate.<locals>.pre_exchange
K3MLAAttention._forward_side_gate.<locals>.pre_tail
K3MLAAttention._forward_side_gate.<locals>.zero_kv_mask_fn
K3MLAAttention._gate_and_o_proj
K3MLAAttention._project_q_by_phase
K3MLAAttention._reject_unsupported
K3MLAAttention._side_gate_applies
K3MLAAttention.cache_derived_state
K3MLAAttention.create_output
K3MLAAttention.create_weights
K3MLAAttention.forward
K3MLAAttention.forward_slice
K3MLAAttention.post_load_weights
K3MLAAttention.project_qkv_a
K3MLAAttention.q_a_mx_atom
K3MLAAttention.q_a_norm
K3MLAAttention.q_a_takes_mxfp8
K3MLAAttention.run_attention
K3MLAAttention.transform_weights
K3MLARuntime.__init__
K3MLARuntime.forward
K3MoEGate.__init__
K3MoEGate.compute_logits
K3MoEGate.route
K3MoEGate.route_tail
K3MoERuntime.__init__.<locals>._shared_forward_confined
K3MoERuntime._fused_latent_input
K3MoERuntime._fused_projection_applies
K3MoERuntime._latent_up_mid_prequantized
K3MoERuntime._lu_mid_prequant_plan
K3MoERuntime._new_fc12_prezero
K3MoERuntime._own_workspace_latent_tail
K3MoERuntime._resolve_routed_quant_config
K3MoERuntime._routed_projection
K3MoERuntime._router_logits
K3MoERuntime._rp_shared_output
K3MoERuntime._rp_tail_forward
K3MoERuntime._rp_tail_target
K3MoERuntime._rp_tail_unsupported
K3MoERuntime._select_moe_tp_ep
K3MoERuntime._shared_needs_all_reduce
K3MoERuntime.cache_derived_state
K3MoERuntime.finalize_expert_load
K3MoERuntime.forward
K3MoERuntime.forward.<locals>._routed_output
K3MoERuntime.forward.<locals>._shared_output
K3MoERuntime.post_load_weights
K3MoERuntime.prepare_expert_load
K3MoERuntime.prepare_rp_tail
KimiK3DefaultDecoderLayer.__init__
KimiK3DefaultDecoderLayer._attn_mx_atom
KimiK3DefaultDecoderLayer._attn_takes_mxfp8
KimiK3DefaultDecoderLayer._capture_hidden_states
KimiK3DefaultDecoderLayer.forward
KimiK3DefaultDecoderLayer.skip_forward
KimiK3DefaultForCausalLM.__init__
KimiK3DefaultForCausalLM._setup_helix_mappings
KimiK3DefaultModelingV2.__init__
KimiK3DefaultModelingV2.forward
KimiK3Sm107ForConditionalGeneration.draft_config
KimiK3Sm107ForConditionalGeneration.draft_model
KimiK3Sm107ForConditionalGeneration.forward
KimiK3Sm107ForConditionalGeneration.get_preferred_kv_cache_manager_version
KimiK3Sm107ForConditionalGeneration.get_preferred_transceiver_runtime
KimiK3Sm107ForConditionalGeneration.infer_max_seq_len
KimiK3Sm107ForConditionalGeneration.lm_head
KimiK3Sm107ForConditionalGeneration.model
KimiK3Sm107ForConditionalGeneration.set_guided_decoder
KimiK3Sm107ForConditionalGeneration.vocab_size_padded
_DeferredMoEResidual.materialize
_DeferredRpTail.materialize
_Fp8BlockScaleWeightReadLinear._lowm_forward_scale
_Fp8BlockScaleWeightReadLinear._run_ar_push
_Fp8BlockScaleWeightReadLinear._staged_above_cutover
_Fp8BlockScaleWeightReadLinear._take_next_output
_Fp8BlockScaleWeightReadLinear.forward_gated
_Fp8BlockScaleWeightReadLinear.forward_gated_mxfp8
_Fp8BlockScaleWeightReadLinear.forward_mid_mxfp8
_Fp8BlockScaleWeightReadLinear.forward_mxfp8
_Fp8BlockScaleWeightReadLinear.forward_sigmoid
_Fp8BlockScaleWeightReadLinear.gated_applies
_Fp8BlockScaleWeightReadLinear.gated_mxfp8_applies
_Fp8BlockScaleWeightReadLinear.has_fp8_qdq
_Fp8BlockScaleWeightReadLinear.has_w4a8_nvfp4_fp8
_Fp8BlockScaleWeightReadLinear.mid_operand_tactic
_Fp8BlockScaleWeightReadLinear.mid_push_pdl_ready
_Fp8BlockScaleWeightReadLinear.packed_weight_scale
_Fp8BlockScaleWeightReadLinear.quantize_weight
_Fp8BlockScaleWeightReadLinear.takes_mid_mxfp8_input
_Fp8BlockScaleWeightReadLinear.takes_mxfp8_input
_Fp8BlockScaleWeightReadLinear.takes_staged_mxfp8_input
_K3SharedExpert.forward
_apply_attn_res
_apply_attn_res_add_and_rmsnorm
_apply_attn_res_add_rmsnorm_fused
_apply_attn_res_and_rmsnorm
_apply_attn_res_fused
_apply_attn_res_fused.<locals>.run
_apply_attn_res_rmsnorm_fused
_check_cu_result
_convert_moe_mlps_to_fp8_weight_read
_expand_kv_bf16
_expand_kv_fp8
_extract_kda_extra_attrs
_flashinfer_cute_norm
_fp8_q_args
_gate_side_stream
_get_text_config
_has_weights
_helix_gen
_helix_gen_fused_tail
_is_fp8_block_projection
_is_kda_layer
_is_mla_layer
_k3_attn_res_lamport_applies
_k3_attn_res_pdl
_k3_attn_res_pdl_lamport
_k3_attn_res_pdl_moe_add
_k3_attn_res_pdl_rp_tail
_k3_expert_ckpt_spec
_k3_latent_ar_workspace
_k3_shared_sm_scope
_kimi_k3_fp8_kv_layer_ids
_kv_b_strided_out
_load_nvfp4_expert
_materialize_prefix_sum
_note_attn_res_fusion
_q_fused_latent_row
_quantize_q_fp8
_registered_layer
_resolve_kimi_situ_betas
_rp_tail_latent_slice
_use_persistent_attn_res
_warmup_expand_kv_bf16
_warmup_expand_kv_bf16.<locals>.warmup
_warmup_expand_kv_fp8
attn_forward_gen
context_cached_kv
context_chunked_prefill
context_default
create_mla_outputs
fold_inputs_of
forward_absorption_generation
forward_absorption_generation.<locals>.rope_and_cache_update
forward_absorption_generation.<locals>.w_uk
forward_context_split_k_fp4
forward_context_split_k_fp4.<locals>.expand_kv
forward_context_split_k_fp4.<locals>.run_partition
forward_gen_q_fused
gen_attention
helix_exchange
helix_gen_zero_kv_mask
helix_tail_applies
kda_core_inplace
kv_b_takes_mxfp8
latent_row
load_fp4_mla_chunked_kv_cache_mxfp8
mla_custom_op_inplace
project_context_kv
project_helix_v_partials
q_fused_applies
q_fused_qb_exponents
q_fused_w_uk
resolve_attention_quant_config
run_k3_split_k_context_partition
split_k_context_unavailable_reason
supports_k3_fp8_split_k_context
uses_k3_split_k_context
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__helix8_moetp2ep4/modeling.py`

库存函数 258；已覆盖 9；未覆盖 249。

```text
HelixGenHooks.__init__
K3HelixTail.__call__
K3HelixTail.__init__
K3HelixTail.applies
K3HelixTail.get
K3KDALinearAttention.__init__
K3KDALinearAttention._alias_projections_into_fused
K3KDALinearAttention._build_decode_kernel_constants
K3KDALinearAttention._build_mtp_conv_weights
K3KDALinearAttention._forward_impl
K3KDALinearAttention._fuse_checkpoint_projections
K3KDALinearAttention._issue_l2_prefetch
K3KDALinearAttention._next_kda_layer
K3KDALinearAttention._output_gate
K3KDALinearAttention._prefill_recurrence
K3KDALinearAttention._prepare_prefill_state
K3KDALinearAttention._project_gate_inputs
K3KDALinearAttention._project_output
K3KDALinearAttention._project_packed_conv_input
K3KDALinearAttention._project_verify_inputs
K3KDALinearAttention._project_verify_inputs.<locals>._project_gates
K3KDALinearAttention._project_verify_inputs.<locals>._project_qkvg
K3KDALinearAttention._require_finalized
K3KDALinearAttention._store_core
K3KDALinearAttention._sync_kda_replay_conv_window
K3KDALinearAttention.finalize_decode_weights
K3KDALinearAttention.forward
K3KDALinearAttention.forward_decode
K3KDALinearAttention.forward_prefill
K3KDALinearAttention.forward_verify
K3KDALinearAttention.forward_verify_context_only
K3KDALinearAttention.forward_verify_fused
K3KDALinearAttention.l2_prefetch_ranges
K3KDALinearAttention.qkvg_mx_atom
K3KDALinearAttention.qkvg_stages_weights
K3KDALinearAttention.qkvg_takes_mxfp8
K3MLAAttention.__init__
K3MLAAttention.__init__.<locals>.is_fp8
K3MLAAttention.__init__.<locals>.projection
K3MLAAttention._bind_helix_v_up_weight
K3MLAAttention._build_helix_v_up_weight
K3MLAAttention._forward_attention
K3MLAAttention._forward_side_gate
K3MLAAttention._forward_side_gate.<locals>.gate_of
K3MLAAttention._forward_side_gate.<locals>.pre_exchange
K3MLAAttention._forward_side_gate.<locals>.pre_tail
K3MLAAttention._forward_side_gate.<locals>.zero_kv_mask_fn
K3MLAAttention._gate_and_o_proj
K3MLAAttention._project_q_by_phase
K3MLAAttention._reject_unsupported
K3MLAAttention._side_gate_applies
K3MLAAttention.cache_derived_state
K3MLAAttention.create_output
K3MLAAttention.create_weights
K3MLAAttention.forward
K3MLAAttention.forward_slice
K3MLAAttention.post_load_weights
K3MLAAttention.project_qkv_a
K3MLAAttention.q_a_mx_atom
K3MLAAttention.q_a_norm
K3MLAAttention.q_a_takes_mxfp8
K3MLAAttention.run_attention
K3MLAAttention.transform_weights
K3MLARuntime.__init__
K3MLARuntime.forward
K3MoEGate.__init__
K3MoEGate.compute_logits
K3MoEGate.route
K3MoEGate.route_tail
K3MoERuntime.__init__
K3MoERuntime.__init__.<locals>._shared_forward_confined
K3MoERuntime._fused_latent_input
K3MoERuntime._fused_projection_applies
K3MoERuntime._latent_up_mid_prequantized
K3MoERuntime._lu_mid_prequant_plan
K3MoERuntime._new_fc12_prezero
K3MoERuntime._own_workspace_latent_tail
K3MoERuntime._resolve_routed_quant_config
K3MoERuntime._routed_moe_model_config
K3MoERuntime._routed_projection
K3MoERuntime._router_logits
K3MoERuntime._rp_shared_output
K3MoERuntime._rp_tail_forward
K3MoERuntime._rp_tail_target
K3MoERuntime._rp_tail_unsupported
K3MoERuntime._select_moe_tp_ep
K3MoERuntime._shared_needs_all_reduce
K3MoERuntime.cache_derived_state
K3MoERuntime.finalize_expert_load
K3MoERuntime.forward
K3MoERuntime.forward.<locals>._routed_output
K3MoERuntime.forward.<locals>._shared_output
K3MoERuntime.post_load_weights
K3MoERuntime.prepare_expert_load
K3MoERuntime.prepare_rp_tail
KimiK3DefaultDecoderLayer.__init__
KimiK3DefaultDecoderLayer._attn_mx_atom
KimiK3DefaultDecoderLayer._attn_takes_mxfp8
KimiK3DefaultDecoderLayer._capture_hidden_states
KimiK3DefaultDecoderLayer.forward
KimiK3DefaultDecoderLayer.skip_forward
KimiK3DefaultForCausalLM.__init__
KimiK3DefaultForCausalLM._setup_helix_mappings
KimiK3DefaultModelingV2.__init__
KimiK3DefaultModelingV2.forward
KimiK3Sm107ForConditionalGeneration.draft_config
KimiK3Sm107ForConditionalGeneration.draft_model
KimiK3Sm107ForConditionalGeneration.forward
KimiK3Sm107ForConditionalGeneration.get_preferred_kv_cache_manager_version
KimiK3Sm107ForConditionalGeneration.get_preferred_transceiver_runtime
KimiK3Sm107ForConditionalGeneration.infer_max_seq_len
KimiK3Sm107ForConditionalGeneration.lm_head
KimiK3Sm107ForConditionalGeneration.model
KimiK3Sm107ForConditionalGeneration.set_guided_decoder
KimiK3Sm107ForConditionalGeneration.vocab_size_padded
_DeferredMoEResidual.materialize
_DeferredRpTail.materialize
_Fp8BlockScaleWeightReadLinear._lowm_forward_scale
_Fp8BlockScaleWeightReadLinear._lowm_gemm_applies
_Fp8BlockScaleWeightReadLinear._mid_prequantized_push_tactic
_Fp8BlockScaleWeightReadLinear._prepare_scale_layout
_Fp8BlockScaleWeightReadLinear._run_ar_push
_Fp8BlockScaleWeightReadLinear._run_lowm_gemm
_Fp8BlockScaleWeightReadLinear._runs_low_m_kernel
_Fp8BlockScaleWeightReadLinear._staged_above_cutover
_Fp8BlockScaleWeightReadLinear._take_next_ar_push
_Fp8BlockScaleWeightReadLinear._take_next_output
_Fp8BlockScaleWeightReadLinear.forward
_Fp8BlockScaleWeightReadLinear.forward_gated
_Fp8BlockScaleWeightReadLinear.forward_gated_mxfp8
_Fp8BlockScaleWeightReadLinear.forward_mid_mxfp8
_Fp8BlockScaleWeightReadLinear.forward_mxfp8
_Fp8BlockScaleWeightReadLinear.forward_prequantized
_Fp8BlockScaleWeightReadLinear.forward_sigmoid
_Fp8BlockScaleWeightReadLinear.forward_slice
_Fp8BlockScaleWeightReadLinear.from_linear
_Fp8BlockScaleWeightReadLinear.gated_applies
_Fp8BlockScaleWeightReadLinear.gated_mxfp8_applies
_Fp8BlockScaleWeightReadLinear.has_fp8_qdq
_Fp8BlockScaleWeightReadLinear.has_w4a8_nvfp4_fp8
_Fp8BlockScaleWeightReadLinear.mid_operand_tactic
_Fp8BlockScaleWeightReadLinear.mid_push_pdl_ready
_Fp8BlockScaleWeightReadLinear.packed_weight_scale
_Fp8BlockScaleWeightReadLinear.post_load_weights
_Fp8BlockScaleWeightReadLinear.prepare_lowm_gemm
_Fp8BlockScaleWeightReadLinear.prepare_output_slice
_Fp8BlockScaleWeightReadLinear.quantize_weight
_Fp8BlockScaleWeightReadLinear.takes_mid_mxfp8_input
_Fp8BlockScaleWeightReadLinear.takes_mxfp8_input
_Fp8BlockScaleWeightReadLinear.takes_prequantized_input
_Fp8BlockScaleWeightReadLinear.takes_staged_mxfp8_input
_Fp8BlockScaleWeightReadLinear.transform_weights
_K3SharedExpert.forward
_apply_attn_res
_apply_attn_res_add_and_rmsnorm
_apply_attn_res_add_rmsnorm_fused
_apply_attn_res_and_rmsnorm
_apply_attn_res_fused
_apply_attn_res_fused.<locals>.run
_apply_attn_res_rmsnorm_fused
_check_cu_result
_check_kda_shape
_convert_moe_mlps_to_fp8_weight_read
_defer_moe_residual
_expand_kv_bf16
_expand_kv_fp8
_extract_kda_extra_attrs
_flashinfer_cute_norm
_fp8_q_args
_gate_side_stream
_get_text_config
_has_weights
_helix_gen
_helix_gen_fused_tail
_is_fp8_block_projection
_is_kda_layer
_is_mla_layer
_k3_attn_res_lamport_applies
_k3_attn_res_pdl
_k3_attn_res_pdl_lamport
_k3_attn_res_pdl_moe_add
_k3_attn_res_pdl_rp_tail
_k3_expert_ckpt_spec
_k3_fc12_options
_k3_latent_ar_workspace
_k3_lowm_gemm_can_run
_k3_shared_sm_scope
_kimi_k3_fp8_kv_layer_ids
_kv_b_strided_out
_load_nvfp4_expert
_materialize_prefix_sum
_merge_row_weights
_note_attn_res_fusion
_num_sms
_o_proj_takes_mxfp8
_pdl_serves
_q_fused_latent_row
_quantize_q_fp8
_refuse_attention_dp
_registered_layer
_resolve_kimi_situ_betas
_rp_tail_latent_slice
_snapshot_conv_windows
_stage_state_rows
_stage_state_rows_with_snapshot_rows
_swap_inforward_conv_windows
_swap_linear_to_fp8_weight_read
_use_fab_fa
_use_fused_fab
_use_persistent_attn_res
_verify_kernel_single_wave
_warmup_expand_kv_bf16
_warmup_expand_kv_bf16.<locals>.warmup
_warmup_expand_kv_fp8
_writeback_state_rows
attn_forward_gen
context_cached_kv
context_chunked_prefill
context_default
create_mla_outputs
fold_inputs_of
forward_absorption_generation
forward_absorption_generation.<locals>.rope_and_cache_update
forward_absorption_generation.<locals>.w_uk
forward_context
forward_context_split_k_fp4
forward_context_split_k_fp4.<locals>.expand_kv
forward_context_split_k_fp4.<locals>.run_partition
forward_gen_q_fused
gen_attention
helix_exchange
helix_gen_zero_kv_mask
helix_tail_applies
kda_core_inplace
kv_b_takes_mxfp8
latent_row
load_fp4_mla_chunked_kv_cache_mxfp8
mla_custom_op_inplace
project_context_kv
project_helix_v_partials
q_fused_applies
q_fused_qb_exponents
q_fused_w_uk
resolve_attention_quant_config
run_k3_split_k_context_partition
split_k_context_unavailable_reason
supports_k3_fp8_split_k_context
tail_output_buffer
uses_k3_split_k_context
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__tp1/modeling.py`

库存函数 159；已覆盖 59；未覆盖 100。

```text
K3KDALinearAttention._forward_impl
K3KDALinearAttention.forward
K3KDALinearAttention.forward_decode
K3KDALinearAttention.forward_verify_context_only
K3MLAAttention.__init__
K3MLAAttention.__init__.<locals>.is_fp8
K3MLAAttention.__init__.<locals>.projection
K3MLAAttention._forward_attention
K3MLAAttention._gate_and_o_proj
K3MLAAttention._project_q_by_phase
K3MLAAttention._reject_unsupported
K3MLAAttention.cache_derived_state
K3MLAAttention.create_output
K3MLAAttention.create_weights
K3MLAAttention.forward
K3MLAAttention.post_load_weights
K3MLAAttention.project_qkv_a
K3MLAAttention.q_a_norm
K3MLAAttention.run_attention
K3MLAAttention.transform_weights
K3MLARuntime.__init__
K3MLARuntime.forward
K3MoEGate.__init__
K3MoEGate.compute_logits
K3MoEGate.route
K3MoEGate.route_tail
K3MoERuntime.__init__
K3MoERuntime._resolve_routed_quant_config
K3MoERuntime._routed_moe_model_config
K3MoERuntime._routed_projection
K3MoERuntime._router_logits
K3MoERuntime._select_moe_tp_ep
K3MoERuntime.finalize_expert_load
K3MoERuntime.forward
K3MoERuntime.forward.<locals>._routed_output
K3MoERuntime.forward.<locals>._shared_output
K3MoERuntime.prepare_expert_load
KimiK3DefaultDecoderLayer.__init__
KimiK3DefaultDecoderLayer._capture_hidden_states
KimiK3DefaultDecoderLayer.forward
KimiK3DefaultDecoderLayer.skip_forward
KimiK3DefaultForCausalLM.__init__
KimiK3DefaultModelingV2.__init__
KimiK3DefaultModelingV2.forward
ModelingV2KimiK3Nvfp4Sm107Tp1.__init__
ModelingV2KimiK3Nvfp4Sm107Tp1.draft_config
ModelingV2KimiK3Nvfp4Sm107Tp1.forward
ModelingV2KimiK3Nvfp4Sm107Tp1.get_preferred_kv_cache_manager_version
ModelingV2KimiK3Nvfp4Sm107Tp1.get_preferred_transceiver_runtime
ModelingV2KimiK3Nvfp4Sm107Tp1.infer_max_seq_len
ModelingV2KimiK3Nvfp4Sm107Tp1.lm_head
ModelingV2KimiK3Nvfp4Sm107Tp1.model
ModelingV2KimiK3Nvfp4Sm107Tp1.set_guided_decoder
ModelingV2KimiK3Nvfp4Sm107Tp1.vocab_size_padded
_Fp8BlockScaleWeightReadLinear.has_fp8_qdq
_Fp8BlockScaleWeightReadLinear.has_w4a8_nvfp4_fp8
_Fp8BlockScaleWeightReadLinear.quantize_weight
_K3SharedExpert.forward
_apply_attn_res
_apply_attn_res_add_and_rmsnorm
_apply_attn_res_add_rmsnorm_fused
_apply_attn_res_and_rmsnorm
_apply_attn_res_fused
_apply_attn_res_fused.<locals>.run
_apply_attn_res_rmsnorm_fused
_convert_moe_mlps_to_fp8_weight_read
_expand_kv_bf16
_expand_kv_fp8
_extract_kda_extra_attrs
_fp8_q_args
_get_text_config
_has_weights
_is_fp8_block_projection
_is_kda_layer
_is_mla_layer
_k3_expert_ckpt_spec
_kimi_k3_fp8_kv_layer_ids
_kv_b_strided_out
_load_nvfp4_expert
_note_attn_res_fusion
_quantize_q_fp8
_registered_layer
_resolve_kimi_situ_betas
_swap_linear_to_fp8_weight_read
_use_persistent_attn_res
_warmup_expand_kv_bf16
_warmup_expand_kv_bf16.<locals>.warmup
_warmup_expand_kv_fp8
context_cached_kv
context_chunked_prefill
context_default
create_mla_outputs
forward_absorption_generation
forward_absorption_generation.<locals>.rope_and_cache_update
forward_absorption_generation.<locals>.w_uk
gen_attention
kda_core_inplace
mla_custom_op_inplace
project_context_kv
resolve_attention_quant_config
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/mla.py`

库存函数 40；已覆盖 38；未覆盖 2。

```text
KimiMLARuntime._reduce
_spec_position_offsets
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/moe.py`

库存函数 48；已覆盖 44；未覆盖 4。

```text
DecodeMoePath.forward
DecodeMoePath.takes
KimiK3MoERuntime.forward
KimiK3MoERuntime.tail_rp_eligible
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/_fp4_mla_decode.py`

库存函数 16；已覆盖 9；未覆盖 7。

```text
_get_sm_count
_kernel_module
_reduce_module
_run_packed_decode
applies
run_decode
run_generation
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/_k3_fp4_mla_cache.py`

库存函数 23；已覆盖 12；未覆盖 11。

```text
_Fp8MlaContextScratch.matches
_get_cutedsl_persistent_v_packed_cache
_get_fp4_mla_v_packed_pool
_get_fp4_mla_v_packed_pool_base
_get_fp4_mla_v_scale_pool_base
_prepare_fp4_mla_q_buffers
_repack_cutedsl_v_packed_cache
_scatter_fp4_mla_kv_cache_2d_generation
_scatter_fp4_mla_kv_cache_2d_generation.<locals>.launch_generation_update
_v_packed_shape
can_fuse_fp4_mla_q_quant
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/k3_mla_attention.py`

库存函数 19；已覆盖 17；未覆盖 2。

```text
_mla_split_kv_plan_kernel
plan_mla_split_kv
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/k3_mla_attn_workspace.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
K3MlaAttnWorkspace.device
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/k3_mla_cache.py`

库存函数 12；已覆盖 11；未覆盖 1。

```text
fp4_mla_rope_generation
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/attention/k3_mla_state.py`

库存函数 17；已覆盖 14；未覆盖 3。

```text
K3MlaAttnState.has_cached_kv_for_mla_context
K3MlaAttnState.has_cached_kv_for_mla_context_warmup
K3MlaAttnState.is_chunked_prefill_for_mla_context
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/cache/k3_sm107_cache_manager.py`

库存函数 28；已覆盖 27；未覆盖 1。

```text
_format_ratio
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/k3_lamport_finish.py`

库存函数 5；已覆盖 1；未覆盖 4。

```text
K3LamportFinish.is_valid
K3LamportFinish.reference
lamport_sum_reference
sanitized
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/k3_lamport_push.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
K3LamportPush.is_valid
K3LamportPush.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/mnnvl_allreduce.py`

库存函数 6；已覆盖 3；未覆盖 3。

```text
K3MnnvlAllReduce.gemm_allreduce_norm
K3MnnvlAllReduce.gemm_twoshot_allreduce_norm
K3MnnvlAllReduce.supports_moe_fold
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/sm107_allreduce.py`

库存函数 18；已覆盖 13；未覆盖 5。

```text
LamportPending.__init__
LamportPending.finish_args
LamportPending.materialize
LamportWorkspaceView.uc_local
mnnvl_workspace_mpi_comm
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/gemm/_mxfp8_gemm_fp8_out.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
K3MXFP8Fp8OutRubinRunner.get_valid_tactics
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/gemm/k3_mxfp8_gemm_fp8_out.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
K3Mxfp8GemmFp8Out.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/gemm/sm107_linear.py`

库存函数 21；已覆盖 19；未覆盖 2。

```text
load_weight_shard
load_weight_shard.<locals>.maybe_convert_to_torch_tensor
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/moe/k3r_experts.py`

库存函数 44；已覆盖 37；未覆盖 7。

```text
K3Nvfp4Experts.loaded_slots
K3RoutedExperts.consumes_sort_tables
_tile_runner_class.<locals>._InputsHelper.infer_shape_num_tokens
_tile_runner_class.<locals>._InputsHelper.inputs_pre_hook
_tile_runner_class.<locals>._TileRunner.forward
_tile_runner_class.<locals>._TileRunner.get_valid_tactics
_tile_runner_class.<locals>._TileRunner.tile_sizes
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/ssm/_kda_prefill.py`

库存函数 28；已覆盖 27；未覆盖 1。

```text
_get_empty_bias_ct
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/cute_aot.py`

库存函数 17；已覆盖 12；未覆盖 5。

```text
_export
_load
arm
toolchain
toolchain.<locals>.version
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/fake_ops.py`

库存函数 53；已覆盖 3；未覆盖 50。

```text
_add_rmsnorm_fake
_bf16_to_fp32_fake
_chain_spec_sampling_mc_fake
_chain_spec_sampling_mc_slots_fake
_chain_spec_sampling_mc_slots_win_fake
_chain_spec_sampling_mc_win_fake
_ctx_residual_rmsnorm_mxfp8_f32_fake
_ctx_residual_rmsnorm_mxfp8_fake
_dflash_merge_later_keys_fake
_draft_prob1_fake
_dspark_advance_fake
_dspark_advance_par_fake
_dspark_ctx_store_fake
_dspark_drafter_inputs_fake
_dspark_drafter_inputs_refresh_fake
_dspark_indices_fake
_dspark_prev_embeddings_fake
_dspark_qk_norm_rope_fake
_dspark_refresh_block_tables_fake
_dspark_widen_add_fake
_expand_block_sampling_fake
_fc_shard_gemm_fake
_fill_padding_rows_fake
_fsel_topk_fake
_host_query.<locals>.fake
_markov_epilogue_fake
_markov_inplace_fake
_markov_prev_embed_fake
_mc_allreduce_norm_fake
_moe_finalize_fake
_mxfp8_cublaslt_gemm_fake
_mxfp8_low_m_gemm_fake
_qkv_indexer_producer_rows_fake
_returns_none
_router_half_pin_fake
_scatter_rows_fake
_select_prefill_tile_fake
_select_prefill_tile_rows_fake
_sm107_mc_allreduce_norm_fake
_softmax_mc_fake
_sparse_decode_lc_prepare_fake
_swiglu_oai_mxfp8_decode_fake
_swiglu_oai_mxfp8_fake
_swiglu_oai_vec_fake
_topp_renorm_det_fake
_topp_renorm_det_v3_fake
_tp2ctx_peer_push_fake
_widen_add_rows_fake
_widen_rows_fake
_zero_padding_rows_fake
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/attention/m3_index_score.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
IndexDecodeScoreKernel.__call__
IndexDecodeScoreKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/attention/m3_select_long_dsl.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
_SelectLongDslKernel.__call__
_SelectLongDslKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/attention/producer_tp8.py`

库存函数 25；已覆盖 24；未覆盖 1。

```text
phase_producer
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/attention/triton_sparse_decode_mq.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_gqa_sparse_decode_mq_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/embedding_firstnorm/op.py`

库存函数 7；已覆盖 5；未覆盖 2。

```text
DecodeEmbedding.__call__
DecodeEmbedding.__init__
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/fake_ops.py`

库存函数 5；已覆盖 1；未覆盖 4。

```text
_
_minimax_m3_router_fp4_outputs
_minimax_m3_sm107_dense_qk_norm_rope_fp8_fake
_register_minimax_m3_sm107_fakes.<locals>._
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/gemm/m3_lowm_splitk.py`

库存函数 44；已覆盖 28；未覆盖 16。

```text
SplitKDenseGemmKernel.__call__
SplitKDenseGemmKernel._drain_producer
SplitKDenseGemmKernel._push_tile
SplitKDenseGemmKernel._store_tile
SplitKDenseGemmKernel._swiglu_to_b
SplitKDenseGemmKernel.dma_warp
SplitKDenseGemmKernel.epilog_warp
SplitKDenseGemmKernel.kernel
SplitKDenseGemmKernel.mma_warp
_bmm_bias
_bmm_swiglu_push
_get_compiled_swiglu_push
_max_ab_stages_for
autotune_tactics
compile_swiglu_push
default_tactic
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/gemm/m3_mx_gemv_kernel.py`

库存函数 28；已覆盖 24；未覆盖 4。

```text
_bf16_round
_din_load4
_rope_bits
read_globaltimer
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m31_moe_finalize_out.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
failure
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3_fold_tail.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_fold_tail_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3_index_score.py`

库存函数 12；已覆盖 9；未覆盖 3。

```text
IndexDecodeScoreKernel.__call__
IndexDecodeScoreKernel.kernel
IndexScoreRunner.is_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3_index_score_t2sc.py`

库存函数 15；已覆盖 10；未覆盖 5。

```text
IndexDecodeScoreKernel.__call__
IndexDecodeScoreKernel.kernel
IndexDecodeScoreKernel.kernel_qown
IndexDecodeScoreKernelQown.__call__
IndexScoreRunner.is_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3_moe/m3_moe16_kernel.py`

库存函数 64；已覆盖 58；未覆盖 6。

```text
_atom_add_u64
_bar_arrive
_fence_acq_rel_gpu
_gtime
_quant_block
_smid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3_moe/m3_moe_kernel.py`

库存函数 73；已覆盖 68；未覆盖 5。

```text
_build_b1
_fence_proxy_async_global
_ld_acquire_u32
_red_release_add
_smid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3_nnt_fold.py`

库存函数 10；已覆盖 8；未覆盖 2。

```text
_build.<locals>.nnt_fold
_build.<locals>.nnt_fold_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3_select_long_dsl.py`

库存函数 15；已覆盖 12；未覆盖 3。

```text
_SelectLongDslKernel.__call__
_SelectLongDslKernel.kernel
prebuild
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3_spec_tail.py`

库存函数 5；已覆盖 2；未覆盖 3。

```text
capture_norm
chunk_norm
draft_input
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3ll/moe_engine_host.py`

库存函数 27；已覆盖 16；未覆盖 11。

```text
M3MoeLayer.__call__
M3MoeState._ext_args
_check_dl_roles
_fine_ranges
_maxnreg.<locals>.gen
ar_push_desc
ar_push_desc_g3
make_phase_plan
make_phase_plan.<locals>.counts_for
make_phase_plan_dl
push_params_v3
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3ll/moe_engine_kernel.py`

库存函数 80；已覆盖 66；未覆盖 14。

```text
_ar_in_rearm
_ar_in_reduce
_ar_push_g3
_atom_add_release
_build_b1
_gtime
_ld_v4_u32
_mbar_arrive_n
_nkb2
_pf_next
_rsqrt_approx
_smid
_st_release_u32
_st_v4_u32
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3ll/prob1/draft_prob1.py`

库存函数 13；已覆盖 1；未覆盖 12。

```text
DraftProb1.__init__
DraftProb1.create
DraftProb1.describe
DraftProb1.fault_line
DraftProb1.faults
DraftProb1.run
DraftProb1.taken_line
_Mnnvl.__init__
_Mnnvl.alloc
_op
disabled
supported
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/m3ll/tp4_moe_engine.py`

库存函数 11；已覆盖 7；未覆盖 4。

```text
engaged
is_engine
log_engaged
workspace_bytes
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/megamoe/add_rmsnorm_mxfp8.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
AddRMSNormKernel.__call__
AddRMSNormKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/megamoe/megamoe_topk_reduce_tail.py`

库存函数 19；已覆盖 13；未覆盖 6。

```text
TopkReduceAddRMSNormKernel.__call__
TopkReduceAddRMSNormKernel.kernel
TopkReduceSharedAddKernel.__call__
TopkReduceSharedAddKernel.kernel
_TopkReduceSharedAdd._load_row
_TopkReduceSharedAdd._reduce_add
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/megamoe/swiglu_mxfp8.py`

库存函数 10；已覆盖 8；未覆盖 2。

```text
SwigluMxfp8Kernel.__call__
SwigluMxfp8Kernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/moe/m3_moe_kernel.py`

库存函数 65；已覆盖 64；未覆盖 1。

```text
_build_b1
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/triton_sparse_decode.py`

库存函数 16；已覆盖 9；未覆盖 7。

```text
_dequant_nvfp4_rows
_e2m1x2_to_f16x2
_gqa_sparse_decode_kernel
_gqa_sparse_decode_pair_kernel
_merge_topk_attn_out_kernel
_rcp_approx_ftz
_store_mxfp8_head
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/kernels/triton_sparse_decode_mq.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_gqa_sparse_decode_mq_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/m31_mxfp8_cublaslt.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
declined
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/m31_weights.py`

库存函数 58；已覆盖 15；未覆盖 43。

```text
MiniMaxM31DraftWeights.__repr__
MiniMaxM31DraftWeights._targets
MiniMaxM31DraftWeights.declare
MiniMaxM31DraftWeights.dims
MiniMaxM31DraftWeights.expected_unconsumed
MiniMaxM31DraftWeights.load
MiniMaxM31DraftWeights.manifest
MiniMaxM31Weights.__repr__
MiniMaxM31Weights._targets
MiniMaxM31Weights.declare
MiniMaxM31Weights.expected_unconsumed
MiniMaxM31Weights.load
MiniMaxM31Weights.manifest
W.layer_indices
_col
_col_w
_dev
_draft_fc_norms
_draft_qkv
_expert_s2
_expert_s31
_expert_w2
_expert_w31
_f32
_fill
_gate_up_s
_gate_up_w
_gemma_fold
_heads
_index_qk_s
_index_qk_w
_materialize
_per_expert.<locals>.build
_qkv_dense_s
_qkv_dense_w
_qkv_sparse_s
_qkv_sparse_w
_release
_row
_row_s
_row_w
_swizzle
_unfused_sparse_layers
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/drafter_mxfp8.py`

库存函数 16；已覆盖 14；未覆盖 2。

```text
_unroute
engagement
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/fake_ops.py`

库存函数 8；已覆盖 2；未覆盖 6。

```text
_fp8_qkv_indexer_norm_rope_kv_insert_fake
_host_query.<locals>.fake
_nvfp4_qkv_indexer_norm_rope_kv_insert_fake
_returns_none
_sampling_from_probs_mc_fi_fake
_sparse_decode_own_t_prepare_fake
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/lc2_decode.py`

库存函数 13；已覆盖 9；未覆盖 4。

```text
install
own_t_status
reserve_own_t_workspace
serves
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/logits_widen.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
engagement
log_engagement
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/markov_glue.py`

库存函数 11；已覆盖 8；未覆盖 3。

```text
_not_taken
engagement
log_engagement
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/merge_mxfp8.py`

库存函数 10；已覆盖 8；未覆盖 2。

```text
_log_once
install
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/norm_mxfp8.py`

库存函数 14；已覆盖 8；未覆盖 6。

```text
_log_l2pf_engagement
_log_l2pf_once
install
is_site
site_decline_reason
site_handoff
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/pad_fill.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
engagement
log_engagement
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/rej_mc_fi.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
engagement
log_engagement
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/router_operand.py`

库存函数 17；已覆盖 13；未覆盖 4。

```text
_wrapped
install
is_site
site_decline_reason
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/se_side.py`

库存函数 10；已覆盖 1；未覆盖 9。

```text
_all_ranks
_log_once
_moe_backend
engagement
install
layer_decline_reason
local_decline_reason
placement
side_calls
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/sfp_mc_fi.py`

库存函数 10；已覆盖 7；未覆盖 3。

```text
_declined
engagement
log_engagement
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/shared_mxfp8.py`

库存函数 15；已覆盖 14；未覆盖 1。

```text
_wrapped
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/topp_det.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
engagement
log_engagement
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/ctx_boundary_plan.py`

库存函数 36；已覆盖 11；未覆盖 25。

```text
chunk_max_tokens
ctas_per_sm
drop_norm_decline_reason
launch_choice
launch_form
launch_stages
ms_per_1k
p2_choice
p2_ctas_per_sm
p2_supported
p2_virtual_warp
ring_slots
smem_bytes
stage_decline_reason
supported
topk_dual_choice
topk_dual_form
topk_dual_group
topk_dual_pos_row
topk_dual_release_need
topk_fits
topk_fold_decline_reason
topk_fold_plane_decline_reason
topk_smem_bytes
topk_stage_bytes
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/fake_ops.py`

库存函数 8；已覆盖 2；未覆盖 6。

```text
_ctx_boundary_normq_fake
_ctx_boundary_normq_out_fake
_ctx_boundary_normq_topk_fake
_ctx_residual_rmsnorm_mxfp8_fake
_host_query.<locals>.fake
_returns_none
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/modeling.py`

库存函数 166；已覆盖 3；未覆盖 163。

```text
MiniMaxM3Attention.__init__
MiniMaxM3Attention._a1_fused_in_bcg_takes
MiniMaxM3Attention._ctx_fused_qkv_gemm_producer
MiniMaxM3Attention._dense_forward
MiniMaxM3Attention._dispatch_attention_backend
MiniMaxM3Attention._emit_fp8_main_qkv
MiniMaxM3Attention._expect_fused_qk_norm_rope
MiniMaxM3Attention._forward_attention_core
MiniMaxM3Attention._fused_fp8_index_qk_norm_rope
MiniMaxM3Attention._fused_fp8_qkv_indexer_norm_rope_kv_insert
MiniMaxM3Attention._fused_qk_norm_rope
MiniMaxM3Attention._init_native_dense_attention
MiniMaxM3Attention._msa_attention_core
MiniMaxM3Attention._msa_backend_active
MiniMaxM3Attention._sdpa_dense_attention_core
MiniMaxM3Attention._sparse_forward
MiniMaxM3Attention._sparse_forward.<locals>._index_norm_rope
MiniMaxM3Attention._sparse_forward.<locals>._main_norm_rope
MiniMaxM3Attention._split_index_qk
MiniMaxM3Attention._split_main_qkv
MiniMaxM3Attention._triton_sparse_attention_core
MiniMaxM3Attention.apply_index_qk_norm
MiniMaxM3Attention.apply_index_qk_norm.<locals>._idx_k_norm
MiniMaxM3Attention.apply_index_qk_norm.<locals>._idx_q_norm
MiniMaxM3Attention.apply_qk_norm
MiniMaxM3Attention.apply_qk_norm.<locals>._k_norm
MiniMaxM3Attention.apply_qk_norm.<locals>._q_norm
MiniMaxM3Attention.forward
MiniMaxM3DecoderLayer.__init__
MiniMaxM3DecoderLayer._apply_next_layer_layernorm
MiniMaxM3DecoderLayer._apply_pre_feed_forward_norm
MiniMaxM3DecoderLayer.forward
MiniMaxM3DecoderLayer.forward_MoE
MiniMaxM3DecoderLayer.forward_mlp
MiniMaxM3ForCausalLM.__init__
MiniMaxM3ForCausalLM.get_preferred_kv_cache_manager_version
MiniMaxM3ForCausalLM.get_preferred_transceiver_runtime
MiniMaxM3ForCausalLM.load_weights
MiniMaxM3ForCausalLM.setup_aliases
MiniMaxM3Gate.__init__
MiniMaxM3Gate.bf16s_weight
MiniMaxM3Gate.forward
MiniMaxM3Gate.load_weights
MiniMaxM3Gate.routing_method
MiniMaxM3MoE.__init__
MiniMaxM3MoE._forward
MiniMaxM3MoE._forward.<locals>._compute_routed_output
MiniMaxM3MoE._forward.<locals>._compute_routed_output.<locals>._run
MiniMaxM3MoE._forward.<locals>._compute_shared_output
MiniMaxM3MoE._get_experts_quant_config
MiniMaxM3MoE._normq_stage_seat
MiniMaxM3MoE.forward
MiniMaxM3MoE.fp4_input_scale
MiniMaxM3MoE.normq_stage_plan
MiniMaxM3Model.__init__
MiniMaxM3Model.forward
MiniMaxM3QKVIndexerLinear.__init__
MiniMaxM3QKVIndexerLinear._shard_geometry
MiniMaxM3QKVIndexerLinear.load_five_way_weights
MiniMaxRMSNorm.__init__
MiniMaxRMSNorm.forward
MiniMaxRMSNorm.load_weights
ModelingV2MiniMaxM31FlashNvfp4Sm107Dep2ctx.__init__
ModelingV2MiniMaxM31FlashNvfp4Sm107Dep2ctx._wire_pointer_guard
ModelingV2MiniMaxM31FlashNvfp4Sm107Dep2ctx.cache_derived_state
ModelingV2MiniMaxM31FlashNvfp4Sm107Dep2ctx.forward
ModelingV2MiniMaxM31FlashNvfp4Sm107Dep2ctx.load_weights
_C1BcgSink.__call__
_C1BcgSink.allocate
_C1BcgSink.min_rows
_DroppedNorms.add
_DroppedNorms.clear
_DroppedNorms.materialize
_NormqMXFP8LinearMethod.apply
_NormqSlot.publish
_NormqSlot.take
_NormqStage.build
_SwigluMxfp8GatedMLP.forward
_TopkFoldSlot.arm
_TopkFoldSlot.park
_TopkFoldSlot.take
_a1_bcg_gemm
_a1_bcg_ptr_reader
_a1_bcg_ptr_reader.<locals>.read
_a1_bcg_ptrs
_a1_bcg_warm
_attn_core_inplace
_autotuner_tuning
_bcg_break_attention
_bcg_graphs_released
_bcg_note
_build_swiglu_oai_dense_mlp
_c1_allocate
_c1_bcg_bucket
_c1_bcg_pair
_c1_materialize
_c1_publishes
_combine_mxfp8_drop_bf16
_combine_mxfp8_sink
_ctx_moe_tail_add
_ctx_moe_tail_log
_ctx_moe_tail_request
_ctx_qkv_fused_band
_ctx_qkv_fused_cfg
_ctx_qkv_fused_count
_ctx_qkv_fused_final
_ctx_qkv_fused_runner
_ctx_qkv_fused_takes_rows
_dep2ctx_capture_blocked
_dispatch_attention_over_live_tokens
_extract_minimax_m3_attention_extra_attrs
_fold_gemma_boundary_norm_weights
_index_projection_quant_config
_is_deferred_topk_reduce
_load_index_qk_proj_weights
_load_qkv_index_proj_weights
_minimax_m3_fused_sparse_qkv_producer_fake
_minimax_m3_qkv_index_proj_fake
_minimax_m3_swiglu_oai
_moe_routed_output_is_global
_mxfp8_cublaslt_gemm
_mxfp8_cublaslt_op
_mxfp8_cublaslt_self_check
_normq_boundary
_normq_f32_op
_normq_op
_normq_stage_boundary
_normq_topk_boundary
_normq_v2_op
_normq_v2_ops
_normq_v2_out_op
_nvfp4_cache_supports_fused_write
_nvfp4_qkv_producer
_route_stage_op
_router_bf16s_log
_router_cast_kernel
_router_cast_kernel.<locals>._bf16_to_fp32_kernel
_router_cast_op
_router_input_fp32
_router_reads_bf16
_strip_language_model_prefix
_swiglu_mxfp8_op
_swiglu_noh
_takes_published_mxfp8
_topk_capture
_topk_fold_arm
_topk_fold_decline_reason
_topk_fold_hand_off
_topk_fold_log
_topk_fold_op
_topk_fold_plane_decline_reason
_unpermute_add_supported
_validate_sparse_attention_runtime_config
_wrap_dict_as_config
get_moe_layer_ids
get_sparse_disable_index_value_layer_ids
get_sparse_layer_ids
get_text_config
get_text_model_config
is_minimax_m3_vl_config
minimax_m3_attn_custom_op_inplace
minimax_m3_fused_sparse_qkv_producer
minimax_m3_qkv_index_proj
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark.py`

库存函数 81；已覆盖 13；未覆盖 68。

```text
DraftOneshotAllReduce.__init__
DraftOneshotAllReduce.forward
DraftOneshotAllReduce.uses_nccl_symmetric_memory_window
M3DSparkWorker._a1_all_ok
M3DSparkWorker._dc_arm
M3DSparkWorker._dc_flush
M3DSparkWorker._dc_keepable
M3DSparkWorker._forward_impl
M3DSparkWorker._lazy_init_ctx_buffers
M3DSparkWorker._mcag_gather
M3DSparkWorker._n3a_kv_rewind
M3DSparkWorker._nnt_fold_log
M3DSparkWorker._prebuild_a1
M3DSparkWorker._prebuild_dhead8
M3DSparkWorker._prebuild_draft_a1
M3DSparkWorker._prebuild_draft_heads
M3DSparkWorker._prebuild_mcag
M3DSparkWorker._refresh_ctx_block_tables
M3DSparkWorker._static_arange
M3DSparkWorker._tf86_note
M3DSparkWorker._tf_account
M3DSparkWorker._tf_arm
M3DSparkWorker._tf_ctx_store
M3DSparkWorker._tf_gather_takes
M3DSparkWorker._tf_join
M3DSparkWorker._tf_log
M3DSparkWorker._tf_mc
M3DSparkWorker._tf_prebuild
M3DSparkWorker._tf_prebuild_gather
M3DSparkWorker._tf_prepare_1st_drafter_inputs
M3DSparkWorker._tf_prepare_1st_drafter_inputs.<locals>.project
M3DSparkWorker._tf_project
M3DSparkWorker._tfold_end_step
M3DSparkWorker._tfold_hook_inputs
M3DSparkWorker._tfv_draw
M3DSparkWorker._tfv_prebuild
M3DSparkWorker.arm_capture_norm
M3DSparkWorker.bind_tail_fold
M3DSparkWorker.capture_hidden_states
M3DSparkWorker.prebuild
M3DSparkWorker.prepare_1st_drafter_inputs
M3DSparkWorker.tail_eligible
_DeferredCaptures.__init__
_DraftA1Logits.__init__
_DraftA1Logits.forward
_TailStep.__init__
_counted
_counted.<locals>.run
_n3a_kv_log_once
_n3a_log_once
_note_capfork_wide
_qk_norm_rope_append_logged
_s9b_count_op
_s9b_count_op.<locals>.op
_s9b_mc_op
_s9b_mc_op.<locals>.op
_s9b_mc_wide_ready
_stock_projection
build_dspark
draft_kv_override_applies
install_draft_oneshot_allreduce
install_draft_swiglu
is_standalone_dspark
keep_drafter_kv_dtype
load_dspark_draft_weights
mcag_unreachable
s10_applies
s5_max_batch_for
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark_markov_worker.py`

库存函数 48；已覆盖 3；未覆盖 45。

```text
S4WorkerMixin._apply_kv_rewind_after_draft
S4WorkerMixin._block_sampling_params
S4WorkerMixin._dhead_noslice
S4WorkerMixin._forward_impl
S4WorkerMixin._host_step_hook
S4WorkerMixin._m16_all_ranks_ok
S4WorkerMixin._m16_b1_log
S4WorkerMixin._m16_b1_setup_line
S4WorkerMixin._m16_base
S4WorkerMixin._m16_chain
S4WorkerMixin._m16_folds
S4WorkerMixin._p1_engages
S4WorkerMixin._p1_step
S4WorkerMixin._prebuild_markov16
S4WorkerMixin._prebuild_markov16.<locals>.base_for
S4WorkerMixin._prebuild_prob1
S4WorkerMixin._s4_chain
S4WorkerMixin._s4_common
S4WorkerMixin._s4_decline
S4WorkerMixin._s4_log_count
S4WorkerMixin._s4_mode
S4WorkerMixin._s4_ranks_agree
S4WorkerMixin._s4_serves
S4WorkerMixin._s4_work
S4WorkerMixin._s4l_log
S4WorkerMixin._store_block_draft_probs
S4WorkerMixin._tailbk_c_log
S4WorkerMixin._tailbk_rewind_args
S4WorkerMixin.prebuild_dhead_noslice
S4WorkerMixin.prebuild_dhead_noslice.<locals>.window
S4WorkerMixin.prebuild_s4
S4WorkerMixin.sample_draft_tokens
_M16State.__init__
_S4State.__init__
_S4State.buffer
_ShardedDraftLogits.__init__
_ShardedDraftLogits.forward
_biased_shard_logits
_capacity_reason
_head_is_sharded
_head_kind
_noslice_log
_takes_window
disabled
prob1_off_reason
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/gen_glue.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
log_engagement
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/lamport_push_gemm.py`

库存函数 24；已覆盖 8；未覆盖 16。

```text
LamportPushGemmKernel.__call__
LamportPushGemmKernel.can_implement
LamportPushGemmKernel.is_valid_dtypes_and_scale_factor_vec_size
LamportPushGemmKernel.is_valid_layouts
LamportPushGemmKernel.is_valid_mma_tiler_and_cluster_shape
LamportPushGemmKernel.is_valid_tensor_alignment
LamportPushGemmKernel.is_valid_tensor_alignment.<locals>.check_contigous_16B_alignment
LamportPushGemmKernel.kernel
LamportPushGemmKernel.narrow_tile_ok
LamportPushGemmKernel.push_wrapper
LamportPushGemmLauncher.slots
_DeviceBytes.__init__
_l2_cache_policy
_per_token_fragment_plan
_per_token_fragment_plan.<locals>.walk
read_device_i64
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_attn_lc.py`

库存函数 5；已覆盖 1；未覆盖 4。

```text
N1Slot.take
lc_applies
m3_lc_attn
one_div_applies
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_moe/op.py`

库存函数 19；已覆盖 17；未覆盖 2。

```text
M3MoeLayer.__call__
ar_push_desc
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_moe_finalize.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_moe_finalize_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_prequant.py`

库存函数 14；已覆盖 5；未覆盖 9。

```text
ArPushSlot.arm
ArPushSlot.take_pushed
ArPushSlot.take_target
MoePushSlot.arm
MoePushSlot.disarm
MoePushSlot.take_engine
MoePushSlot.take_pushed
MoePushSlot.take_shared
PrequantSlot.take
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/mx_gemv/m3_mx_gemv_kernel.py`

库存函数 30；已覆盖 20；未覆盖 10。

```text
_bf16_round
_bf16x2_word
_din_load4
_ld_volatile_s32
_push_parities
_rope_bits
_st_global_b32
_st_global_v4_b32
_swiglu_oai_bf16x2
read_globaltimer
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/mx_gemv/op.py`

库存函数 27；已覆盖 18；未覆盖 9。

```text
_note_pushopt
ensure_compiled
freeze
frozen
gu_interleave
is_compiled
m3_mx_gemv_gu_swiglu
m3_mx_gemv_swiglu
precompile
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/triton_sparse_decode.py`

库存函数 10；已覆盖 8；未覆盖 2。

```text
_log_tp2wide_lc2
_longattn_lc_decode
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/modeling.py`

库存函数 279；已覆盖 8；未覆盖 271。

```text
DecodeTarget.__init__
DecodeTarget._attention
DecodeTarget._attention_core
DecodeTarget._attn_push_target
DecodeTarget._boundary
DecodeTarget._bucket_flags
DecodeTarget._check_step_contract
DecodeTarget._decode_sparse_core
DecodeTarget._embed
DecodeTarget._embed16
DecodeTarget._embed16_raw
DecodeTarget._fullctx_decode
DecodeTarget._fullctx_plan
DecodeTarget._generic_boundary
DecodeTarget._generic_layer
DecodeTarget._layer
DecodeTarget._moe
DecodeTarget._moe_push_targets
DecodeTarget._no_bucket
DecodeTarget._nvfp4_index_scorer
DecodeTarget._prepushed
DecodeTarget._push_gemm_boundary
DecodeTarget._push_gemm_target
DecodeTarget._routed
DecodeTarget._routed_chunks
DecodeTarget._s2_tail
DecodeTarget._swiglu_mlp
DecodeTarget._unfused_sparse_attention
DecodeTarget._unfused_sparse_attention.<locals>._index
DecodeTarget._unfused_sparse_attention.<locals>._main
DecodeTarget.forward
DecodeTarget.step_args
M3DSparkStepFold.__init__
M3DSparkStepFold.gather
M3DSparkStepFold.scatter
M3DSparkStepFoldFwd.__init__
M3DSparkStepFoldFwd.begin_step
M3DSparkStepFoldFwd.fold_next_new_tokens
M3DSparkStepFoldFwd.sampler_rows_stored
M3DSparkStepFoldFwd.store_in_forward
M3Eagle3Draft.__init__
M3Eagle3Draft._linear
M3Eagle3Draft.apply_eagle3_fc
M3Eagle3Draft.ar_norm
M3Eagle3Draft.attention
M3Eagle3Draft.embed_table
M3Eagle3Draft.fc
M3Eagle3Draft.forward_layer
M3Eagle3Draft.load_weights_from_target_model
M3Eagle3Draft.mlp
M3Eagle3Draft.prebuild_linears
M3Eagle3Worker.__init__
M3Eagle3Worker._forward_draft_loop
M3Eagle3Worker._forward_impl
M3Eagle3Worker._m3_arm_draft_push
M3Eagle3Worker._m3_coded_head
M3Eagle3Worker._m3_draft_ar_norm
M3Eagle3Worker._m3_drafter_supported
M3Eagle3Worker._m3_fc_fork
M3Eagle3Worker._m3_fc_join
M3Eagle3Worker._m3_first_drafter_inputs
M3Eagle3Worker._m3_force_vec
M3Eagle3Worker._m3_fused_draft_loop
M3Eagle3Worker._m3_fused_loop_ok
M3Eagle3Worker._m3_renorm_raw_captures
M3Eagle3Worker._m3_sampled_verify_ok
M3Eagle3Worker._m3_seed_first_pass
M3Eagle3Worker._m3_structural_ok
M3Eagle3Worker._m3_target_accept
M3Eagle3Worker.arm_capture_norm
M3Eagle3Worker.capture_hidden_states
M3Eagle3Worker.capture_target
M3Eagle3Worker.draft_only_eligible
M3Eagle3Worker.prebuild
M3Eagle3Worker.prebuild_draft_head
M3Eagle3Worker.prebuild_target_head
M3Eagle3Worker.prepare_1st_drafter_inputs
M3Eagle3Worker.tail_eligible
MiniMaxM3Tp2ModelingV2.__init__
MiniMaxM3Tp2ModelingV2._build_embed_owners
MiniMaxM3Tp2ModelingV2._build_engine
MiniMaxM3Tp2ModelingV2._build_engine_agreed
MiniMaxM3Tp2ModelingV2._build_moe16_engine
MiniMaxM3Tp2ModelingV2._build_wc_engine
MiniMaxM3Tp2ModelingV2._derive_moe
MiniMaxM3Tp2ModelingV2._engine_state
MiniMaxM3Tp2ModelingV2._expert_operands
MiniMaxM3Tp2ModelingV2._lookup
MiniMaxM3Tp2ModelingV2._prebuild_engines
MiniMaxM3Tp2ModelingV2._prebuild_index_scorer
MiniMaxM3Tp2ModelingV2._prebuild_index_scorer_intree
MiniMaxM3Tp2ModelingV2._prebuild_index_scorer_t2sc
MiniMaxM3Tp2ModelingV2._prebuild_msa_proxy
MiniMaxM3Tp2ModelingV2._prepare_die_local
MiniMaxM3Tp2ModelingV2._scorer_variants
MiniMaxM3Tp2ModelingV2._select_target
MiniMaxM3Tp2ModelingV2._shared_weights
MiniMaxM3Tp2ModelingV2._try_build
MiniMaxM3Tp2ModelingV2.cache_derived_state
MiniMaxM3Tp2ModelingV2.decode_query_lens
MiniMaxM3Tp2ModelingV2.engine
MiniMaxM3Tp2ModelingV2.forward
MiniMaxM3Tp2ModelingV2.moe16_engine
MiniMaxM3Tp2ModelingV2.post_load_weights
MiniMaxM3Tp2ModelingV2.prebuild_kernels
MiniMaxM3Tp2ModelingV2.projections
MiniMaxM3Tp2ModelingV2.sm_scale
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2.__init__
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2.cache_derived_state
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2.forward
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2.load_draft_weights
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2.load_weights
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2.post_load_weights
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2.set_guided_decoder
PrefillTarget.__init__
PrefillTarget._attention
PrefillTarget._attention_core
PrefillTarget._check_step_contract
PrefillTarget._generic_boundary
PrefillTarget._layer
PrefillTarget._moe
PrefillTarget._routed
PrefillTarget._routed_chunks
PrefillTarget._swiglu_mlp
PrefillTarget._unfused_sparse_attention
PrefillTarget._unfused_sparse_attention.<locals>._index
PrefillTarget._unfused_sparse_attention.<locals>._main
PrefillTarget.forward
PrefillTarget.step_args
PushGemmTarget.__init__
PushGemmTarget.accepts
PushGemmTarget.run
RowRange.certified
RowRange.covers
_DeferredA1.__init__
_DeferredA1._wait_ovl
_DeferredA1.fold
_DeferredA1.materialize
_DeferredGemv.__init__
_DeferredGemv.materialize
_DraftForward.__call__
_DraftForward.__init__
_Mxfp8GemmState.__init__
_a1_count
_a1_defer
_a1_runner
_attn_push16_taken
_build_speculation
_can_mc
_capture_layer_output
_combine_mxfp8_allocate
_combine_mxfp8_finish
_combine_mxfp8_sink
_counted_pack_prologue
_ctxnormq_allreduce_norm
_ctxnormq_f32_op
_ctxnormq_mxfp8_gemm
_ctxnormq_op
_decode_route
_decode_router_logits
_dense_swiglu
_eagle3_capture
_fold_cfg
_fp8_index_qk_norm_rope
_gemv_variants
_gqamix_decode
_gqamix_register
_in_tree_scorer
_index_score_t2sc_requests
_install_rej_mc
_install_softmax_mc
_install_topp_det
_install_tp_gumbel
_is_vl_config
_linear
_log_once
_long_select_dsl_widths
_long_select_widths
_mc_allreduce_norm
_mc_taken
_moe_layer_ids
_moe_tactic_as_logits
_msa_attention_live
_mx_mode
_mxfp8_linear
_native_allreduce_norm
_native_dispatch
_native_finalize
_native_op
_native_swiglu
_next_norm
_note_mx_tac
_note_wide_gemvcfg
_note_wide_input9
_note_wide_push
_nvfp4_fused_write_layout
_nvfp4_layer
_pack_nvfp4_index_k
_pin_allreduce_heuristic
_plain_capture
_prebuild_rej_mc
_prebuild_softmax_mc
_prebuild_topp_det
_precompile_cfg
_prefill_allreduce
_prefill_attention_boundary
_prefill_attention_opt_ins
_producer_covered
_producer_early
_producer_layout
_producer_miss
_push_gemm_init
_push_gemm_op
_push_gemm_stock_is_pinned
_qk_norm_rope
_qkv_producer_fold
_route_ids16
_routed_experts
_router_logits
_runtime_hooks
_s2_boundary
_s2_op
_select_triggers
_sharded_head
_sp_boundary
_sp_consumer_ws
_sp_count
_sp_init
_sp_keyed
_sp_load
_sp_logits_bytes
_sp_once
_sp_ovl_take
_sp_ovl_wait
_sp_own_rows
_sp_own_sf_bytes
_sp_put
_sp_supports
_sp_unshard
_sp_ws
_sparse_decode_lc_keys
_sparse_decode_producer
_sparse_layer_ids
_sparse_qkv_producer
_spec_rows.<locals>.certifies
_split_chunk
_swq
_swq_cpp_takes
_text_model_config
_triton_finalize
_triton_finalize_rows
_tsel_op
_tsel_register
_tsel_select
_uncompiled_note
_validate_runtime
_wide_cfg_is_order_preserving
capture_missed
index_score_t2sc_schedule
is_decode_step
is_mc_finalize_capture
linear_path
long_select_dsl_keys
missed
moe_path
plan_for
require_mode
reset_logged
set_require
taken
uncovered
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/attention/m3_sparse_decode.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
M3SparseDecode.is_valid
M3SparseDecode.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/attention/m3_sparse_decode_lc.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
prepare_lc2_split_workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/gemm/m3_mxfp8_pinned_gemm.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
M3Mxfp8PinnedGemm.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/gemm/m3_mxfp8_pinned_quantize.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
M3Mxfp8PinnedQuantize.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/norm/m3_embedding_firstnorm.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
DecodeEmbedding.__init__
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/sampling/m31_softmax_mc.py`

库存函数 9；已覆盖 2；未覆盖 7。

```text
M31SoftmaxMc.__call__
M31SoftmaxMc.is_valid
M31SoftmaxMc.reference
partials_numel
prebuild
softmax_mc_op
write_ctas
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2/ops/spec/m31_markov_epilogue.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
M31MarkovChainFused.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2ctx/idx_scorer.py`

库存函数 5；已覆盖 1；未覆盖 4。

```text
_log_once
_wrapper
register
score
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2ctx/modeling.py`

库存函数 145；已覆盖 7；未覆盖 138。

```text
MiniMaxM3Attention.__init__
MiniMaxM3Attention._ctx_fused_qkv_gemm_producer
MiniMaxM3Attention._dense_forward
MiniMaxM3Attention._dispatch_attention_backend
MiniMaxM3Attention._emit_fp8_main_qkv
MiniMaxM3Attention._expect_fused_qk_norm_rope
MiniMaxM3Attention._forward_attention_core
MiniMaxM3Attention._fused_fp8_index_qk_norm_rope
MiniMaxM3Attention._fused_fp8_qkv_indexer_norm_rope_kv_insert
MiniMaxM3Attention._fused_qk_norm_rope
MiniMaxM3Attention._init_native_dense_attention
MiniMaxM3Attention._msa_attention_core
MiniMaxM3Attention._msa_backend_active
MiniMaxM3Attention._sdpa_dense_attention_core
MiniMaxM3Attention._sparse_forward
MiniMaxM3Attention._sparse_forward.<locals>._index_norm_rope
MiniMaxM3Attention._sparse_forward.<locals>._main_norm_rope
MiniMaxM3Attention._split_index_qk
MiniMaxM3Attention._split_main_qkv
MiniMaxM3Attention._triton_sparse_attention_core
MiniMaxM3Attention.apply_index_qk_norm
MiniMaxM3Attention.apply_index_qk_norm.<locals>._idx_k_norm
MiniMaxM3Attention.apply_index_qk_norm.<locals>._idx_q_norm
MiniMaxM3Attention.apply_qk_norm
MiniMaxM3Attention.apply_qk_norm.<locals>._k_norm
MiniMaxM3Attention.apply_qk_norm.<locals>._q_norm
MiniMaxM3Attention.forward
MiniMaxM3DecoderLayer.__init__
MiniMaxM3DecoderLayer._apply_next_layer_layernorm
MiniMaxM3DecoderLayer._apply_pre_feed_forward_norm
MiniMaxM3DecoderLayer._feed_forward_all_reduce_params
MiniMaxM3DecoderLayer.forward
MiniMaxM3DecoderLayer.forward_MoE
MiniMaxM3DecoderLayer.forward_mlp
MiniMaxM3ForCausalLM.__init__
MiniMaxM3ForCausalLM.get_preferred_kv_cache_manager_version
MiniMaxM3ForCausalLM.get_preferred_transceiver_runtime
MiniMaxM3ForCausalLM.load_weights
MiniMaxM3ForCausalLM.setup_aliases
MiniMaxM3Gate.__init__
MiniMaxM3Gate.forward
MiniMaxM3Gate.load_weights
MiniMaxM3Gate.routing_method
MiniMaxM3MoE.__init__
MiniMaxM3MoE._get_experts_quant_config
MiniMaxM3MoE.forward
MiniMaxM3MoE.forward.<locals>._compute_routed_output
MiniMaxM3MoE.forward.<locals>._compute_routed_output.<locals>._run
MiniMaxM3MoE.forward.<locals>._compute_shared_output
MiniMaxM3MoE.fp4_input_scale
MiniMaxM3Model.__init__
MiniMaxM3Model.forward
MiniMaxM3QKVIndexerLinear.__init__
MiniMaxM3QKVIndexerLinear._shard_geometry
MiniMaxM3QKVIndexerLinear.load_five_way_weights
MiniMaxRMSNorm.__init__
MiniMaxRMSNorm.forward
MiniMaxRMSNorm.load_weights
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2ctx.__init__
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2ctx.forward
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2ctx.load_weights
_NormqMXFP8LinearMethod.apply
_NormqSlot.publish
_NormqSlot.take
_NormqSlot.take_ovl
_SwigluMxfp8GatedMLP.forward
_a1_bcg_gemm
_a1_bcg_warm
_addnorm_note
_autotuner_tuning
_build_swiglu_oai_dense_mlp
_c1_materialize
_c1_publishes
_combine_mxfp8_drop_bf16
_combine_mxfp8_sink
_ctx_qkv_fused_band
_ctx_qkv_fused_cfg
_ctx_qkv_fused_count
_ctx_qkv_fused_final
_ctx_qkv_fused_runner
_ctx_qkv_fused_takes_rows
_dispatch_attention_over_live_tokens
_enable_draft_fc_split
_extract_minimax_m3_attention_extra_attrs
_fold_gemma_boundary_norm_weights
_gen_finalize_reason
_index_projection_quant_config
_load_index_qk_proj_weights
_load_qkv_index_proj_weights
_minimax_m3_fused_sparse_qkv_producer_fake
_minimax_m3_qkv_index_proj_fake
_minimax_m3_swiglu_oai
_moe_routed_output_is_global
_mxfp8_cublaslt_gemm
_mxfp8_cublaslt_op
_mxfp8_cublaslt_self_check
_nccl_window_probe
_normq_boundary
_normq_f32_op
_normq_op
_nvfp4_cache_supports_fused_write
_nvfp4_qkv_producer
_pin_allreduce_heuristic
_pmovl_note
_router_cast_kernel
_router_cast_kernel.<locals>._bf16_to_fp32_kernel
_router_zc_note
_sp_addout_slot
_sp_boundary
_sp_capture_rows
_sp_init
_sp_logits_note
_sp_note
_sp_oproj_target
_sp_ovl_note
_sp_own_rows
_sp_put
_sp_routed_call
_sp_routed_call.<locals>.apply
_sp_supports
_sp_unshard
_sp_workspace
_sp_ws
_strip_language_model_prefix
_swiglu_mxfp8_op
_swiglu_noh
_unpermute_add_supported
_validate_sparse_attention_runtime_config
_wrap_dict_as_config
get_moe_layer_ids
get_sparse_disable_index_value_layer_ids
get_sparse_layer_ids
get_text_config
get_text_model_config
is_minimax_m3_vl_config
minimax_m3_attn_custom_op_inplace
minimax_m3_fused_sparse_qkv_producer
minimax_m3_qkv_index_proj
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/dspark.py`

库存函数 68；已覆盖 3；未覆盖 65。

```text
DraftOneshotAllReduce.__init__
DraftOneshotAllReduce.forward
DraftOneshotAllReduce.uses_nccl_symmetric_memory_window
M3DSparkWorker._a1_all_ok
M3DSparkWorker._forward_impl
M3DSparkWorker._n3a_eligible
M3DSparkWorker._n3a_kv_rewind
M3DSparkWorker._prebuild_a1
M3DSparkWorker._prebuild_dhead4
M3DSparkWorker._prepare_attn_metadata_for_dflash
M3DSparkWorker._prepare_next_new_tokens
M3DSparkWorker._refresh_ctx_block_tables
M3DSparkWorker._rng_state_per_request
M3DSparkWorker._sample_and_accept_draft_tokens_base
M3DSparkWorker._samples_block_ancestrally
M3DSparkWorker._tailbk_f_log
M3DSparkWorker._tailbk_rng_window
M3DSparkWorker._tailbk_skips_padding_rows
M3DSparkWorker._tf86_note
M3DSparkWorker._tf_arm
M3DSparkWorker._tf_ctx_store
M3DSparkWorker._tf_flush
M3DSparkWorker._tf_gather_takes
M3DSparkWorker._tf_join
M3DSparkWorker._tf_keepable
M3DSparkWorker._tf_launch
M3DSparkWorker._tf_prebuild
M3DSparkWorker._tf_prebuild_gather
M3DSparkWorker._tf_prepare_1st_drafter_inputs
M3DSparkWorker._tf_prepare_1st_drafter_inputs.<locals>.project
M3DSparkWorker._tf_project
M3DSparkWorker._tfold_end_step
M3DSparkWorker._tfold_hook_advance
M3DSparkWorker._tfold_hook_inputs
M3DSparkWorker._zero_padding_rows
M3DSparkWorker.arm_capture_norm
M3DSparkWorker.bind_tail_fold
M3DSparkWorker.capture_hidden_states
M3DSparkWorker.prebuild
M3DSparkWorker.prepare_1st_drafter_inputs
M3DSparkWorker.tail_eligible
M3DSparkWorker.target_logits
M3EmbedTokens.__init__
M3EmbedTokens.forward
M3EmbedTokens.weight
_TailStep.__init__
_a1_log_once
_n3a_kv_log_once
_n3a_log_once
_n3p_log_once
_qk_norm_rope_append_logged
_stock_projection
_whole_store_folded
build_dspark
draft_kv_override_applies
install_draft_oneshot_allreduce
install_draft_swiglu
install_tp4_gumbel
is_standalone_dspark
keep_drafter_kv_dtype
load_dspark_draft_weights
s10_applies
s10c_c_drafter_inputs
s5_max_batch_for
tp4_ancestral_route
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/dspark_markov_worker.py`

库存函数 28；已覆盖 11；未覆盖 17。

```text
S4WorkerMixin._apply_kv_rewind_after_draft
S4WorkerMixin._forward_impl
S4WorkerMixin._host_step_hook
S4WorkerMixin._markov_grid_log
S4WorkerMixin._s4_common
S4WorkerMixin._s4_log_count
S4WorkerMixin._s4_mode
S4WorkerMixin._s4_ranks_agree
S4WorkerMixin._s4_ranks_agree_value
S4WorkerMixin._s4_serves
S4WorkerMixin._s4l_log
S4WorkerMixin.prebuild_s4
S4WorkerMixin.sample_draft_tokens
_biased_shard_logits
_head_is_sharded
disabled
ranks_agree_value
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/dspark_tailfold.py`

库存函数 21；已覆盖 9；未覆盖 12。

```text
TailGlue.end_step
_memcpy4b_off_reason
_rank
_stock_prepare
drafter_owns_noise_hook
install_seq_lens_snapshot
own_noise
recording_inputs
recording_inputs.<locals>.hook
stock_two_field_prepare
tailfold_setup_line
take_seq_lens_snapshot
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/dspark_tailfork.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
StoreOperands.args
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/fake_ops.py`

库存函数 5；已覆盖 2；未覆盖 3。

```text
_host_query.<locals>.fake
_qkv_front_fake
_returns_none
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/m3_mc_allreduce.py`

库存函数 14；已覆盖 5；未覆盖 9。

```text
McAllReduceWorkspace.create
McBuckets.__init__
McBuckets.for_tokens
_tp_comm
decline_reason
get_workspace
m3_mc_finalize_allreduce_norm
tp_agree
tp_agree_built
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/m3_moe/op.py`

库存函数 24；已覆盖 22；未覆盖 2。

```text
M3MoeLayer.__call__
moe_lag_deterministic
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/m3_prequant.py`

库存函数 14；已覆盖 6；未覆盖 8。

```text
ArPushSlot.arm
ArPushSlot.take_pushed
ArPushSlot.take_target
MoePushSlot.arm
MoePushSlot.disarm
MoePushSlot.take_engine
MoePushSlot.take_pushed
MoePushSlot.take_shared
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/m3_report.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_noop
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/m3_tail_glue.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_tail_glue_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/mx_gemv/m3_mx_gemv_kernel.py`

库存函数 54；已覆盖 19；未覆盖 35。

```text
_bf16_bits
_bf16_round
_bf16x2_word
_din_load4
_e2m1x4
_e4m3_value
_ld_global_i32
_ld_global_v2_i32
_ld_global_v4_f32
_ld_volatile_s32
_pr_absmax
_pr_add
_pr_asm
_pr_const_loads
_pr_fma
_pr_max
_pr_mul
_pr_op1
_pr_op2
_pr_op3
_pr_rcp
_pr_rsqrt
_pr_ss2
_pr_step_loads
_pr_task
_push_parities
_rope_bits
_smid
_st_async_b32
_st_global_b32
_st_global_u16
_st_global_u8
_st_global_v4_b32
_swiglu_oai_bf16x2
read_globaltimer
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/kernels/mx_gemv/op.py`

库存函数 24；已覆盖 17；未覆盖 7。

```text
ensure_compiled
freeze
frozen
is_compiled
m3_mx_gemv_swiglu
precompile
producer_args
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/modeling.py`

库存函数 221；已覆盖 21；未覆盖 200。

```text
DecodeTarget.__init__
DecodeTarget._attention
DecodeTarget._attention_core
DecodeTarget._attn_push_target
DecodeTarget._boundary
DecodeTarget._bucket_flags
DecodeTarget._check_step_contract
DecodeTarget._decode_sparse_core
DecodeTarget._defer_front
DecodeTarget._embed
DecodeTarget._front_producer_operands
DecodeTarget._fullctx_decode
DecodeTarget._fullctx_plan
DecodeTarget._generic_boundary
DecodeTarget._generic_layer
DecodeTarget._hoist_arm
DecodeTarget._hoist_layer
DecodeTarget._hoist_take
DecodeTarget._layer
DecodeTarget._moe
DecodeTarget._moe_push_targets
DecodeTarget._next_kv_prefetch
DecodeTarget._no_bucket
DecodeTarget._prepushed
DecodeTarget._routed
DecodeTarget._routed_chunks
DecodeTarget._run_front
DecodeTarget._swiglu_mlp
DecodeTarget._unfused_sparse_attention
DecodeTarget._unfused_sparse_attention.<locals>._index
DecodeTarget._unfused_sparse_attention.<locals>._main
DecodeTarget.forward
DecodeTarget.step_args
M3DSparkStepFold.__init__
M3DSparkStepFold.gather
M3DSparkStepFold.scatter
M3Eagle3Draft.__init__
M3Eagle3Draft._linear
M3Eagle3Draft.apply_eagle3_fc
M3Eagle3Draft.ar_norm
M3Eagle3Draft.attention
M3Eagle3Draft.embed_table
M3Eagle3Draft.fc
M3Eagle3Draft.forward_layer
M3Eagle3Draft.load_weights_from_target_model
M3Eagle3Draft.mlp
M3Eagle3Draft.prebuild_linears
M3Eagle3Worker.__init__
M3Eagle3Worker._forward_draft_loop
M3Eagle3Worker._forward_impl
M3Eagle3Worker._m3_arm_draft_push
M3Eagle3Worker._m3_coded_head
M3Eagle3Worker._m3_draft_ar_norm
M3Eagle3Worker._m3_drafter_supported
M3Eagle3Worker._m3_fc_fork
M3Eagle3Worker._m3_fc_join
M3Eagle3Worker._m3_first_drafter_inputs
M3Eagle3Worker._m3_force_vec
M3Eagle3Worker._m3_fused_draft_loop
M3Eagle3Worker._m3_fused_loop_ok
M3Eagle3Worker._m3_renorm_raw_captures
M3Eagle3Worker._m3_sampled_verify_ok
M3Eagle3Worker._m3_seed_first_pass
M3Eagle3Worker._m3_structural_ok
M3Eagle3Worker._m3_target_accept
M3Eagle3Worker.arm_capture_norm
M3Eagle3Worker.capture_hidden_states
M3Eagle3Worker.capture_target
M3Eagle3Worker.draft_only_eligible
M3Eagle3Worker.prebuild
M3Eagle3Worker.prebuild_draft_head
M3Eagle3Worker.prebuild_target_head
M3Eagle3Worker.prepare_1st_drafter_inputs
M3Eagle3Worker.tail_eligible
M3SpecStepFold.gather
M3SpecStepFold.scatter
MiniMaxM3Tp4ModelingV2.__init__
MiniMaxM3Tp4ModelingV2._build_embed_owners
MiniMaxM3Tp4ModelingV2._build_engine
MiniMaxM3Tp4ModelingV2._build_engine_agreed
MiniMaxM3Tp4ModelingV2._build_moe16_engine
MiniMaxM3Tp4ModelingV2._build_wc_engine
MiniMaxM3Tp4ModelingV2._derive_moe
MiniMaxM3Tp4ModelingV2._engine_state
MiniMaxM3Tp4ModelingV2._expert_operands
MiniMaxM3Tp4ModelingV2._lookup
MiniMaxM3Tp4ModelingV2._prebuild_engines
MiniMaxM3Tp4ModelingV2._prebuild_hoist
MiniMaxM3Tp4ModelingV2._prebuild_hoist.<locals>.build
MiniMaxM3Tp4ModelingV2._prebuild_index_scorer
MiniMaxM3Tp4ModelingV2._prebuild_index_scorer_t2sc
MiniMaxM3Tp4ModelingV2._prebuild_msa_proxy
MiniMaxM3Tp4ModelingV2._scorer_variants
MiniMaxM3Tp4ModelingV2._select_target
MiniMaxM3Tp4ModelingV2._shared_weights
MiniMaxM3Tp4ModelingV2._try_build
MiniMaxM3Tp4ModelingV2.cache_derived_state
MiniMaxM3Tp4ModelingV2.decode_query_lens
MiniMaxM3Tp4ModelingV2.engine
MiniMaxM3Tp4ModelingV2.forward
MiniMaxM3Tp4ModelingV2.hoist_block
MiniMaxM3Tp4ModelingV2.moe16_engine
MiniMaxM3Tp4ModelingV2.post_load_weights
MiniMaxM3Tp4ModelingV2.prebuild_kernels
MiniMaxM3Tp4ModelingV2.projections
MiniMaxM3Tp4ModelingV2.sm_scale
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp4.__init__
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp4.cache_derived_state
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp4.forward
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp4.load_draft_weights
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp4.load_weights
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp4.post_load_weights
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp4.set_guided_decoder
PrefillTarget.__init__
PrefillTarget._attention
PrefillTarget._attention_core
PrefillTarget._check_step_contract
PrefillTarget._generic_boundary
PrefillTarget._layer
PrefillTarget._moe
PrefillTarget._routed
PrefillTarget._routed_chunks
PrefillTarget._swiglu_mlp
PrefillTarget._unfused_sparse_attention
PrefillTarget._unfused_sparse_attention.<locals>._index
PrefillTarget._unfused_sparse_attention.<locals>._main
PrefillTarget.forward
PrefillTarget.step_args
RowRange.certified
_DraftForward.__call__
_DraftForward.__init__
_Mxfp8GemmState.__init__
_build_speculation
_can_mc
_ctxnormq_allreduce_norm
_ctxnormq_mxfp8_gemm
_ctxnormq_op
_decode_router_logits
_dense_swiglu
_eagle3_capture
_fp8_index_qk_norm_rope
_gemv_variants
_gqamix_decode
_gqamix_register
_index_score_t2sc_requests
_install_rej_mc
_install_softmax_mc
_install_topp_det
_is_vl_config
_linear
_log_tailbk
_long_select_dsl_widths
_long_select_widths
_mc_allreduce_norm
_moe_layer_ids
_mx_mode
_mxfp8_linear
_native_allreduce_norm
_next_norm
_note_mx_tac
_nvfp4_fused_write_layout
_nvfp4_layer
_pin_allreduce_heuristic
_plain_capture
_prebuild_rej_mc
_prebuild_softmax_mc
_prebuild_topp_det
_precompile_cfg
_prefill_allreduce
_prod_gemv
_producer_covered
_producer_layout
_producer_miss
_qk_norm_rope
_qkv_prodfold
_rej_mc_override
_routed_experts
_router_logits
_runtime_hooks
_sharded_head
_sparse_decode_producer
_sparse_layer_ids
_sparse_qkv_producer
_spec_rows.<locals>.certifies
_split_chunk
_text_model_config
_uncompiled_note
_validate_runtime
capture_missed
is_decode_step
linear_path
long_select_dsl_keys
missed
moe_path
plan_for
require_mode
reset_logged
set_require
t2sc_token_major
uncovered
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/ops/attention/m3_producer.py`

库存函数 6；已覆盖 1；未覆盖 5。

```text
M3Producer.is_valid
M3Producer.reference
_norm_rope
decode_nvfp4_token
scale_index
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/ops/attention/m3_select_long_dsl.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
M3SelectLongDsl.is_valid
M3SelectLongDsl.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/ops/attention/m3_sparse_decode.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
M3SparseDecode.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/ops/attention/m3_sparse_decode_lc.py`

库存函数 6；已覆盖 3；未覆盖 3。

```text
M3SparseDecodeLc.is_valid
M3SparseDecodeLc.reference
mxfp8_handoff_reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/ops/qkv_front.py`

库存函数 6；已覆盖 3；未覆盖 3。

```text
fits
get_workspace
log_engaged
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/dspark.py`

库存函数 48；已覆盖 4；未覆盖 44。

```text
M3DSparkWorker._a1_all_ok
M3DSparkWorker._f7_gather
M3DSparkWorker._f8_drafter_inputs
M3DSparkWorker._forward_impl
M3DSparkWorker._mcag_gather
M3DSparkWorker._n3a_eligible
M3DSparkWorker._n3a_kv_rewind
M3DSparkWorker._prebuild_a1
M3DSparkWorker._prebuild_f7
M3DSparkWorker._prebuild_mcag
M3DSparkWorker._prepare_next_new_tokens
M3DSparkWorker._refresh_ctx_block_tables
M3DSparkWorker._rng_state_per_request
M3DSparkWorker._sample_and_accept_draft_tokens_base
M3DSparkWorker._samples_block_ancestrally
M3DSparkWorker._tailbk_rng_window
M3DSparkWorker._tailbk_skips_padding_rows
M3DSparkWorker._zero_padding_rows
M3DSparkWorker.arm_capture_norm
M3DSparkWorker.arm_next_new_tokens_tail
M3DSparkWorker.capture_hidden_states
M3DSparkWorker.prebuild
M3DSparkWorker.prepare_1st_drafter_inputs
M3DSparkWorker.tail_eligible
M3DSparkWorker.target_logits
_a1_log_once
_f3_defers
_n3a_kv_log_once
_n3a_log_once
_n3p_log_once
_qk_norm_rope_append_logged
_s9b_mc_op
_s9b_mc_op.<locals>.op
build_dspark
draft_kv_override_applies
install_causal_draft
install_causal_draft.<locals>.resolve
install_draft_swiglu
is_standalone_dspark
keep_drafter_kv_dtype
load_dspark_draft_weights
s10_applies
s10c_c_drafter_inputs_and_refresh
s5_max_batch_for
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/fake_ops.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_returns_none
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_fold_tail.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
_fold_tail_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_fullctx_decode.py`

库存函数 7；已覆盖 5；未覆盖 2。

```text
_tp8_reduce_op
emits_mxfp8
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_layer/phase_producer.py`

库存函数 25；已覆盖 24；未覆盖 1。

```text
phase_producer
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_lowm_splitk.py`

库存函数 46；已覆盖 22；未覆盖 24。

```text
SplitKDenseGemmKernel.__call__
SplitKDenseGemmKernel._drain_producer
SplitKDenseGemmKernel._push_tile
SplitKDenseGemmKernel._store_tile
SplitKDenseGemmKernel._swiglu_to_b
SplitKDenseGemmKernel.dma_warp
SplitKDenseGemmKernel.epilog_warp
SplitKDenseGemmKernel.kernel
SplitKDenseGemmKernel.mma_warp
_bmm_bias
_bmm_swiglu
_bmm_swiglu_push
_check_push_rows
_fence_proxy_async_global
_get_compiled_swiglu_push
_ld_bf16_f32
_max_ab_stages_for
_silu_mul_f32
_st_f32_bf16
autotune_tactics
compile_swiglu_push
default_tactic
run_splitk_dense_push
run_splitk_dense_swiglu
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_moe/m3_moe_kernel.py`

库存函数 66；已覆盖 63；未覆盖 3。

```text
_build_b1
_gtime
_smid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_moe/op.py`

库存函数 19；已覆盖 16；未覆盖 3。

```text
M3MoeLayer.__call__
ar_push_desc
make_phase_plan.<locals>.counts_for
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_prequant.py`

库存函数 14；已覆盖 6；未覆盖 8。

```text
ArPushSlot.arm
ArPushSlot.take_pushed
ArPushSlot.take_target
MoePushSlot.arm
MoePushSlot.disarm
MoePushSlot.take_engine
MoePushSlot.take_pushed
MoePushSlot.take_shared
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/m3_report.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_noop
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/mx_gemv/m3_mx_gemv_kernel.py`

库存函数 27；已覆盖 19；未覆盖 8。

```text
_bf16_round
_bf16x2_word
_din_load4
_ld_volatile_s32
_rope_bits
_st_global_b32
_swiglu_oai_bf16x2
read_globaltimer
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/kernels/mx_gemv/op.py`

库存函数 26；已覆盖 18；未覆盖 8。

```text
ensure_compiled
freeze
frozen
gu_interleave
is_compiled
m3_mx_gemv_gu_swiglu
m3_mx_gemv_swiglu
precompile
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/modeling.py`

库存函数 224；已覆盖 38；未覆盖 186。

```text
DecodeTarget._attention
DecodeTarget._attention_core
DecodeTarget._attn_push_target
DecodeTarget._boundary
DecodeTarget._bucket_flags
DecodeTarget._check_step_contract
DecodeTarget._decode_sparse_core
DecodeTarget._dense_attention
DecodeTarget._embed
DecodeTarget._fullctx_decode
DecodeTarget._fullctx_plan
DecodeTarget._generic_boundary
DecodeTarget._generic_layer
DecodeTarget._index_scores
DecodeTarget._layer
DecodeTarget._moe_push_targets
DecodeTarget._next_kv_prefetch
DecodeTarget._no_bucket
DecodeTarget._prepushed
DecodeTarget._routed
DecodeTarget._routed_chunks
DecodeTarget._swiglu_mlp
DecodeTarget._unfused_sparse_attention
DecodeTarget._unfused_sparse_attention.<locals>._index
DecodeTarget._unfused_sparse_attention.<locals>._main
DecodeTarget.forward
DecodeTarget.step_args
M3DSparkStepFold.__init__
M3DSparkStepFold.gather
M3DSparkStepFold.scatter
M3Eagle3Draft.__init__
M3Eagle3Draft.apply_eagle3_fc
M3Eagle3Draft.ar_norm
M3Eagle3Draft.arm_push
M3Eagle3Draft.attention
M3Eagle3Draft.embed_table
M3Eagle3Draft.fc
M3Eagle3Draft.forward_layer
M3Eagle3Draft.load_weights_from_target_model
M3Eagle3Draft.mlp
M3Eagle3Draft.take_pushed
M3Eagle3Worker.__init__
M3Eagle3Worker._forward_draft_loop
M3Eagle3Worker._forward_impl
M3Eagle3Worker._m3_arm_draft_push
M3Eagle3Worker._m3_draft_ar_norm
M3Eagle3Worker._m3_drafter_supported
M3Eagle3Worker._m3_fc_bump
M3Eagle3Worker._m3_fc_fork
M3Eagle3Worker._m3_fc_join
M3Eagle3Worker._m3_first_drafter_inputs
M3Eagle3Worker._m3_first_inputs_covered
M3Eagle3Worker._m3_force_vec
M3Eagle3Worker._m3_fused_draft_loop
M3Eagle3Worker._m3_fused_loop_ok
M3Eagle3Worker._m3_join_pending
M3Eagle3Worker._m3_renorm_raw_captures
M3Eagle3Worker._m3_sampled_verify_ok
M3Eagle3Worker._m3_seed_first_pass
M3Eagle3Worker._m3_structural_ok
M3Eagle3Worker._m3_target_accept
M3Eagle3Worker.arm_capture_norm
M3Eagle3Worker.capture_hidden_states
M3Eagle3Worker.capture_target
M3Eagle3Worker.draft_only_eligible
M3Eagle3Worker.prebuild
M3Eagle3Worker.prebuild_draft_head
M3Eagle3Worker.prepare_1st_drafter_inputs
M3Eagle3Worker.tail_eligible
M3SpecStepFold.defer_next_new_tokens
M3SpecStepFold.flush_next_new_tokens
M3SpecStepFold.gather
M3SpecStepFold.scatter
MiniMaxM3Tp8ModelingV2.__init__
MiniMaxM3Tp8ModelingV2._build_embed_owners
MiniMaxM3Tp8ModelingV2._build_engine
MiniMaxM3Tp8ModelingV2._derive_moe
MiniMaxM3Tp8ModelingV2._expert_operands
MiniMaxM3Tp8ModelingV2._in_tree_scorer_variants
MiniMaxM3Tp8ModelingV2._index_geometries
MiniMaxM3Tp8ModelingV2._lookup
MiniMaxM3Tp8ModelingV2._prebuild_engines
MiniMaxM3Tp8ModelingV2._prebuild_index_scorer
MiniMaxM3Tp8ModelingV2._prebuild_msa_proxy
MiniMaxM3Tp8ModelingV2._scorer_variants
MiniMaxM3Tp8ModelingV2._select_target
MiniMaxM3Tp8ModelingV2._shared_weights
MiniMaxM3Tp8ModelingV2._try_build
MiniMaxM3Tp8ModelingV2.cache_derived_state
MiniMaxM3Tp8ModelingV2.decode_query_lens
MiniMaxM3Tp8ModelingV2.engine
MiniMaxM3Tp8ModelingV2.forward
MiniMaxM3Tp8ModelingV2.post_load_weights
MiniMaxM3Tp8ModelingV2.prebuild_kernels
MiniMaxM3Tp8ModelingV2.projections
MiniMaxM3Tp8ModelingV2.sm_scale
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp8.__init__
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp8.cache_derived_state
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp8.forward
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp8.load_draft_weights
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp8.load_weights
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp8.post_load_weights
ModelingV2MiniMaxM31FlashNvfp4Sm107Tp8.set_guided_decoder
PrefillTarget.__init__
PrefillTarget._attention
PrefillTarget._attention_core
PrefillTarget._check_step_contract
PrefillTarget._generic_boundary
PrefillTarget._layer
PrefillTarget._moe
PrefillTarget._routed
PrefillTarget._routed_chunks
PrefillTarget._swiglu_mlp
PrefillTarget._unfused_sparse_attention
PrefillTarget._unfused_sparse_attention.<locals>._index
PrefillTarget._unfused_sparse_attention.<locals>._main
PrefillTarget.forward
PrefillTarget.step_args
_DeferredGemv.__init__
_DeferredGemv.materialize
_DraftForward.__call__
_DraftForward.__init__
_Mxfp8GemmState.__init__
_build_speculation
_can_mc
_ctxnormq_allreduce_norm
_ctxnormq_mxfp8_gemm
_decode_router_logits
_dense_fp8_op_applies
_dense_qk_norm_rope_fp8
_dense_swiglu
_dummy_scorer_operands
_eagle3_capture
_fp8_index_qk_norm_rope
_gemv_variants
_gqamix_decode
_gqamix_register
_in_tree_scorer
_install_rej_mc
_install_softmax_mc
_install_topp_det
_is_vl_config
_linear
_linear_wide
_log_engaged
_log_tailbk
_mc_allreduce_norm
_moe_layer_ids
_mx_mode
_mxfp8_linear
_native_allreduce_norm
_next_norm
_note_mx_tac
_nvfp4_fused_write_layout
_nvfp4_layer
_pin_allreduce_heuristic
_pinned_mxfp8_gemm
_pinned_mxfp8_quantize
_pinned_runner
_plain_capture
_prebuild_builtin_sparse_decode
_prebuild_rej_mc
_prebuild_softmax_mc
_prebuild_topp_det
_producer_covered
_producer_layout
_producer_miss
_qk_norm_rope
_qkv_producer_fold
_routed_experts
_router_logits
_runtime_hooks
_sharded_head
_sparse_layer_ids
_sparse_qkv_producer
_split_chunk
_swq
_swq_cpp_takes
_text_model_config
_uncompiled_note
capture_missed
is_decode_step
long_select_dsl_keys
long_select_fallback
plan_for
require_mode
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/attention/m3_sparse_decode_lc8.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
_identity_table
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/gemm/m3_target_head.py`

库存函数 12；已覆盖 9；未覆盖 3。

```text
M3TargetHead.__call__
M3TargetHead.is_valid
M3TargetHead.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/moe/m3_router_tc.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
fp32_logits
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/sampling/m31_chain_spec_mc.py`

库存函数 19；已覆盖 13；未覆盖 6。

```text
M31ChainSpecMc.is_valid
M31ChainSpecMc.reference
_f3_taken
defer_window
flush_window
window_ops_built
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/sampling/m31_softmax_mc.py`

库存函数 9；已覆盖 7；未覆盖 2。

```text
M31SoftmaxMc.is_valid
M31SoftmaxMc.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/ops/sampling/m31_topp_det.py`

库存函数 16；已覆盖 9；未覆盖 7。

```text
M31ToppDet.is_valid
M31ToppDet.reference
_row_store_fits
_row_store_key
arm_row_store
disarm_row_store
rows_stored
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/modeling.py`

库存函数 169；已覆盖 156；未覆盖 13。

```text
DecodeTarget._unfused_sparse_attention.<locals>._index
DecodeTarget._unfused_sparse_attention.<locals>._main
M3Eagle3Draft.load_weights_from_target_model
M3Eagle3Worker._m3_force_vec
MiniMaxM3Tp4ModelingV2.cache_derived_state
PrefillTarget._unfused_sparse_attention.<locals>._index
PrefillTarget._unfused_sparse_attention.<locals>._main
_fp8_index_qk_norm_rope
_low_m_gemm_supports
_plain_capture
_print_values
_split_chunk
_uncompiled_note
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/comm/m3_allreduce.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
M3AllReduce.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/comm/m3_mc_allreduce_norm.py`

库存函数 7；已覆盖 1；未覆盖 6。

```text
M3McAllreduceNorm.is_valid
M3McAllreduceNorm.reference
boundary
mxfp8_rows
sf_written
swizzle_sf
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/comm/m3_mc_finalize_allreduce_norm.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
M3McFinalizeAllreduceNorm.is_valid
M3McFinalizeAllreduceNorm.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/embedding_firstnorm/op.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
DecodeEmbedding.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_attn_mega.py`

库存函数 4；已覆盖 1；未覆盖 3。

```text
_check
_load_module
m3_attn_mega
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_attn_mega_c1.py`

库存函数 7；已覆盖 3；未覆盖 4。

```text
_check
_load_module
m3_attn_c1
m3_fused_decode_attn_checks
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_index_score.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
IndexDecodeScoreKernel.__call__
IndexDecodeScoreKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_mc_allreduce.py`

库存函数 13；已覆盖 12；未覆盖 1。

```text
decline_reason
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_moe/m3_moe16_kernel.py`

库存函数 64；已覆盖 58；未覆盖 6。

```text
_atom_add_u64
_bar_arrive
_fence_acq_rel_gpu
_gtime
_quant_block
_smid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_moe/m3_moe_kernel.py`

库存函数 62；已覆盖 59；未覆盖 3。

```text
_build_b1
_gtime
_smid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_moe/op.py`

库存函数 15；已覆盖 14；未覆盖 1。

```text
M3MoeLayer.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_prequant.py`

库存函数 14；已覆盖 6；未覆盖 8。

```text
ArPushSlot.arm
ArPushSlot.take_pushed
ArPushSlot.take_target
MoePushSlot.arm
MoePushSlot.disarm
MoePushSlot.take_engine
MoePushSlot.take_pushed
MoePushSlot.take_shared
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_router_gemv.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
m3_router_gemv
m3_router_gemv_fp4
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_sampled_accept.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
sampled_accept
workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/mx_gemv/m3_mx_gemv_kernel.py`

库存函数 25；已覆盖 20；未覆盖 5。

```text
_bf16_bits
_bf16_round
_din_load4
_rope_bits
read_globaltimer
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/mxfp8_low_m_gemm.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
mxfp8_low_m_gemm
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/triton_sparse_decode.py`

库存函数 18；已覆盖 11；未覆盖 7。

```text
_dequant_nvfp4_rows
_e2m1x2_to_f16x2
_gqa_sparse_decode_kernel
_gqa_sparse_decode_pair_kernel
_merge_topk_attn_out_kernel
_rcp_approx_ftz
_store_mxfp8_head
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/triton_sparse_decode_mq.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_gqa_sparse_decode_mq_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/weights.py`

库存函数 49；已覆盖 48；未覆盖 1。

```text
_index_qk_w
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/modeling.py`

库存函数 163；已覆盖 64；未覆盖 99。

```text
DecodeTarget._attention
DecodeTarget._attention_core
DecodeTarget._boundary
DecodeTarget._decode_sparse_core
DecodeTarget._generic_boundary
DecodeTarget._layer
DecodeTarget._moe
DecodeTarget._prepushed
DecodeTarget._routed
DecodeTarget._routed_chunks
DecodeTarget._swiglu_mlp
DecodeTarget._unfused_sparse_attention.<locals>._index
DecodeTarget._unfused_sparse_attention.<locals>._main
M3Eagle3Draft.__init__
M3Eagle3Draft.apply_eagle3_fc
M3Eagle3Draft.ar_norm
M3Eagle3Draft.attention
M3Eagle3Draft.embed_table
M3Eagle3Draft.fc
M3Eagle3Draft.forward_layer
M3Eagle3Draft.load_weights_from_target_model
M3Eagle3Draft.mlp
M3Eagle3Worker.__init__
M3Eagle3Worker._forward_draft_loop
M3Eagle3Worker._forward_impl
M3Eagle3Worker._m3_fc_fork
M3Eagle3Worker._m3_first_drafter_inputs
M3Eagle3Worker._m3_force_vec
M3Eagle3Worker._m3_fused_draft_loop
M3Eagle3Worker._m3_sampled_verify_ok
M3Eagle3Worker._m3_seed_first_pass
M3Eagle3Worker._m3_structural_ok
M3Eagle3Worker._m3_target_accept
M3Eagle3Worker.draft_only_eligible
M3Eagle3Worker.prebuild
M3Eagle3Worker.prepare_1st_drafter_inputs
M3Eagle3Worker.tail_eligible
M3SpecStepFold.__init__
M3SpecStepFold.begin_step
M3SpecStepFold.bind_engine_buffers
M3SpecStepFold.bind_sampler_store
M3SpecStepFold.gather
M3SpecStepFold.sampler_rows_stored
M3SpecStepFold.scatter
MiniMaxM3Tp8ModelingV2.__init__
MiniMaxM3Tp8ModelingV2._build_embed_owners
MiniMaxM3Tp8ModelingV2._build_engine
MiniMaxM3Tp8ModelingV2._derive_moe
MiniMaxM3Tp8ModelingV2._expert_operands
MiniMaxM3Tp8ModelingV2._shared_weights
MiniMaxM3Tp8ModelingV2.cache_derived_state
MiniMaxM3Tp8ModelingV2.forward
MiniMaxM3Tp8ModelingV2.post_load_weights
MiniMaxM3Tp8ModelingV2.projections
MiniMaxM3Tp8ModelingV2.sm_scale
ModelingV2MiniMaxM3Nvfp4Sm103Tp8.__init__
ModelingV2MiniMaxM3Nvfp4Sm103Tp8.cache_derived_state
ModelingV2MiniMaxM3Nvfp4Sm103Tp8.forward
ModelingV2MiniMaxM3Nvfp4Sm103Tp8.load_draft_weights
ModelingV2MiniMaxM3Nvfp4Sm103Tp8.load_weights
ModelingV2MiniMaxM3Nvfp4Sm103Tp8.post_load_weights
PrefillTarget._attention
PrefillTarget._attention_core
PrefillTarget._check_step_contract
PrefillTarget._generic_boundary
PrefillTarget._moe
PrefillTarget._routed
PrefillTarget._routed_chunks
PrefillTarget._swiglu_mlp
PrefillTarget._unfused_sparse_attention.<locals>._index
PrefillTarget._unfused_sparse_attention.<locals>._main
PrefillTarget.forward
PrefillTarget.step_args
_DraftForward.__call__
_DraftForward.__init__
_Mxfp8GemmState.__init__
_build_speculation
_can_mc
_fp8_index_qk_norm_rope
_is_vl_config
_linear
_mc_allreduce_norm
_moe_layer_ids
_mx_mode
_mxfp8_linear
_native_allreduce_norm
_nvfp4_fused_write_layout
_nvfp4_layer
_plain_capture
_print_values
_qk_norm_rope
_routed_experts
_router_fp4
_router_logits
_runtime_hooks
_sparse_layer_ids
_split_chunk
_text_model_config
_uncompiled_note
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/embedding_firstnorm/op.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
DecodeEmbedding.__call__
DecodeEmbedding.__init__
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/m3_draft_step.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
_num_sms
draft_step
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/m3_index_score.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
IndexDecodeScoreKernel.__call__
IndexDecodeScoreKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/m3_mc_allreduce.py`

库存函数 13；已覆盖 4；未覆盖 9。

```text
McAllReduceWorkspace.flags_for
McBuckets.__init__
McBuckets.for_tokens
_bucket
_tp_comm
decline_reason
get_workspace
m3_mc_allreduce_norm
m3_mc_finalize_allreduce_norm
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/m3_moe/m3_moe_kernel.py`

库存函数 63；已覆盖 60；未覆盖 3。

```text
_build_b1
_gtime
_smid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/m3_moe/op.py`

库存函数 16；已覆盖 13；未覆盖 3。

```text
M3MoeLayer.__call__
ar_push_desc
make_phase_plan.<locals>.counts_for
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/m3_prequant.py`

库存函数 14；已覆盖 5；未覆盖 9。

```text
ArPushSlot.arm
ArPushSlot.take_pushed
ArPushSlot.take_target
MoePushSlot.arm
MoePushSlot.disarm
MoePushSlot.take_engine
MoePushSlot.take_pushed
MoePushSlot.take_shared
PrequantSlot.take
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/m3_sampled_accept.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
sampled_accept
workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/m3_shard_accept.py`

库存函数 4；已覆盖 1；未覆盖 3。

```text
_num_sms
_workspace
shard_accept
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/m3_spec_tail.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
slot_scatter
step_gather
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/mx_gemv/m3_mx_gemv_kernel.py`

库存函数 25；已覆盖 20；未覆盖 5。

```text
_bf16_bits
_bf16_round
_din_load4
_rope_bits
read_globaltimer
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/triton_sparse_decode.py`

库存函数 18；已覆盖 11；未覆盖 7。

```text
_dequant_nvfp4_rows
_e2m1x2_to_f16x2
_gqa_sparse_decode_kernel
_gqa_sparse_decode_pair_kernel
_merge_topk_attn_out_kernel
_rcp_approx_ftz
_store_mxfp8_head
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/ops/triton_sparse_decode_mq.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_gqa_sparse_decode_mq_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp8/weights.py`

库存函数 50；已覆盖 5；未覆盖 45。

```text
MiniMaxM3Tp8DraftWeights._targets
MiniMaxM3Tp8DraftWeights.declare
MiniMaxM3Tp8DraftWeights.dims
MiniMaxM3Tp8DraftWeights.expected_unconsumed
MiniMaxM3Tp8DraftWeights.load
MiniMaxM3Tp8DraftWeights.manifest
MiniMaxM3Tp8Weights._targets
MiniMaxM3Tp8Weights.declare
MiniMaxM3Tp8Weights.dims
MiniMaxM3Tp8Weights.expected_unconsumed
MiniMaxM3Tp8Weights.load
MiniMaxM3Tp8Weights.manifest
W.layer_indices
_col
_col_w
_dense_layers
_dev
_draft_fc_norms
_draft_qkv
_expert_s2
_expert_s31
_expert_w2
_expert_w31
_f32
_fill
_five_way_qkv_layers
_gate_up_s
_gate_up_w
_gemma_fold
_heads
_index_qk_w
_materialize
_per_expert.<locals>.build
_qkv_dense_s
_qkv_dense_w
_qkv_sparse_s
_qkv_sparse_w
_release
_row
_row_s
_row_w
_sparse_layers
_swizzle
_three_way_qkv_layers
_unfused_sparse_layers
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/modeling.py`

库存函数 203；已覆盖 68；未覆盖 135。

```text
DecodeTarget._attention
DecodeTarget._attention_core
DecodeTarget._attn_push_target
DecodeTarget._boundary
DecodeTarget._bucket_flags
DecodeTarget._check_step_contract
DecodeTarget._decode_sparse_core
DecodeTarget._dense_attention
DecodeTarget._embed
DecodeTarget._fullctx_decode
DecodeTarget._fullctx_plan
DecodeTarget._generic_boundary
DecodeTarget._index_scores
DecodeTarget._mc_finalize
DecodeTarget._moe
DecodeTarget._moe_push_targets
DecodeTarget._no_bucket
DecodeTarget._prepushed
DecodeTarget._routed
DecodeTarget._routed_chunks
DecodeTarget._select
DecodeTarget._unfused_sparse_attention
DecodeTarget._unfused_sparse_attention.<locals>._index
DecodeTarget._unfused_sparse_attention.<locals>._main
DecodeTarget.forward
DecodeTarget.step_args
M3Eagle3Draft.__init__
M3Eagle3Draft.apply_eagle3_fc
M3Eagle3Draft.ar_norm
M3Eagle3Draft.arm_push
M3Eagle3Draft.embed_table
M3Eagle3Draft.fc
M3Eagle3Draft.forward_layer
M3Eagle3Draft.load_weights_from_target_model
M3Eagle3Draft.mlp
M3Eagle3Worker.__init__
M3Eagle3Worker._forward_draft_loop
M3Eagle3Worker._forward_impl
M3Eagle3Worker._m3_arm_draft_push
M3Eagle3Worker._m3_draft_ar_norm
M3Eagle3Worker._m3_drafter_supported
M3Eagle3Worker._m3_fc_fork
M3Eagle3Worker._m3_first_drafter_inputs
M3Eagle3Worker._m3_force_vec
M3Eagle3Worker._m3_fused_draft_loop
M3Eagle3Worker._m3_fused_loop_ok
M3Eagle3Worker._m3_sampled_verify_ok
M3Eagle3Worker._m3_seed_first_pass
M3Eagle3Worker._m3_structural_ok
M3Eagle3Worker._m3_target_accept
M3Eagle3Worker.arm_capture_norm
M3Eagle3Worker.capture_hidden_states
M3Eagle3Worker.capture_target
M3Eagle3Worker.draft_only_eligible
M3Eagle3Worker.prebuild
M3Eagle3Worker.prebuild_ugpu_head
M3Eagle3Worker.prepare_1st_drafter_inputs
M3Eagle3Worker.tail_eligible
M3SpecStepFold.gather
M3SpecStepFold.scatter
MiniMaxM3Tp2ModelingV2.__init__
MiniMaxM3Tp2ModelingV2._build_embed_owners
MiniMaxM3Tp2ModelingV2._build_engine
MiniMaxM3Tp2ModelingV2._derive_moe
MiniMaxM3Tp2ModelingV2._expert_operands
MiniMaxM3Tp2ModelingV2._in_tree_scorer_variants
MiniMaxM3Tp2ModelingV2._index_geometries
MiniMaxM3Tp2ModelingV2._prebuild_index_scorer
MiniMaxM3Tp2ModelingV2._scorer_variants
MiniMaxM3Tp2ModelingV2._select_target
MiniMaxM3Tp2ModelingV2._shared_weights
MiniMaxM3Tp2ModelingV2.cache_derived_state
MiniMaxM3Tp2ModelingV2.decode_query_lens
MiniMaxM3Tp2ModelingV2.forward
MiniMaxM3Tp2ModelingV2.post_load_weights
MiniMaxM3Tp2ModelingV2.prebuild_kernels
MiniMaxM3Tp2ModelingV2.projections
MiniMaxM3Tp2ModelingV2.sm_scale
ModelingV2MiniMaxM3Nvfp4Sm107Tp2.__init__
ModelingV2MiniMaxM3Nvfp4Sm107Tp2.cache_derived_state
ModelingV2MiniMaxM3Nvfp4Sm107Tp2.forward
ModelingV2MiniMaxM3Nvfp4Sm107Tp2.load_draft_weights
ModelingV2MiniMaxM3Nvfp4Sm107Tp2.load_weights
ModelingV2MiniMaxM3Nvfp4Sm107Tp2.post_load_weights
ModelingV2MiniMaxM3Nvfp4Sm107Tp2.set_guided_decoder
PrefillTarget._attention
PrefillTarget._attention_core
PrefillTarget._check_step_contract
PrefillTarget._generic_boundary
PrefillTarget._moe
PrefillTarget._routed
PrefillTarget._routed_chunks
PrefillTarget._unfused_sparse_attention
PrefillTarget._unfused_sparse_attention.<locals>._index
PrefillTarget._unfused_sparse_attention.<locals>._main
PrefillTarget.forward
PrefillTarget.step_args
_DraftForward.__call__
_DraftForward.__init__
_build_speculation
_can_mc
_decode_routed_unfinalized
_decode_router_logits
_dense_fp8_op_applies
_dense_qk_norm_rope_fp8
_dummy_scorer_operands
_eagle3_capture
_fp8_index_qk_norm_rope
_gemv_variants
_in_tree_scorer
_linear
_mc_allreduce_norm
_msa_attention_live
_mxfp8_linear
_native_allreduce_norm
_note_mx_tac
_nvfp4_fused_write_layout
_nvfp4_layer
_plain_capture
_prebuild_builtin_sparse_decode
_prefill_attention_boundary
_print_values
_producer_covered
_producer_layout
_producer_miss
_qk_norm_rope
_routed_experts
_router_logits
_router_tc_fp4
_sparse_qkv_producer
_split_chunk
_triton_finalize
_triton_finalize_rows
_uncompiled_note
capture_missed
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/embedding_firstnorm/op.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
DecodeEmbedding.__call__
DecodeEmbedding.__init__
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_draft_step.py`

库存函数 8；已覆盖 4；未覆盖 4。

```text
_draft_step_split
_num_sms
draft_step
head_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_fullctx_decode.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
_check_layout
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_lowm_direct.py`

库存函数 19；已覆盖 11；未覆盖 8。

```text
DirectDenseGemmKernel.__call__
DirectDenseGemmKernel.kernel
_default_rows_per_block
_get_compiled_direct_kernel
autotune_tactics
default_tactic
prefer_direct_bf16_gemm_sm100
run_direct_dense
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_mc_allreduce.py`

库存函数 12；已覆盖 3；未覆盖 9。

```text
McBuckets.__init__
McBuckets.for_tokens
_bucket
_capture_args
_tp_comm
decline_reason
get_workspace
m3_mc_allreduce_norm
m3_mc_finalize_allreduce_norm
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_moe/op.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
_shared_views
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_moe_finalize.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_moe_finalize_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_report.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_noop
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_sampled_accept.py`

库存函数 6；已覆盖 1；未覆盖 5。

```text
SampledAcceptWorkspace.__init__
SampledAcceptWorkspace.create
_ilp_walk
_launch
sampled_accept
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_select_long_dsl.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
prebuild
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_shard_accept.py`

库存函数 6；已覆盖 1；未覆盖 5。

```text
_launch
_num_sms
_workspace
create_workspace
shard_accept
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_spec_tail.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
capture_norm
chunk_norm
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/m3_tokmap_guard.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
_no_missed
_no_taken
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/moe/m3_moe_engine.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_shared_expert
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/mx_gemv/op.py`

库存函数 21；已覆盖 16；未覆盖 5。

```text
ensure_compiled
freeze
frozen
is_compiled
precompile
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/ops/spec/m3_spec_offsets.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
M3SpecOffsets.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp2/weights.py`

库存函数 49；已覆盖 5；未覆盖 44。

```text
MiniMaxM3Tp2DraftWeights._targets
MiniMaxM3Tp2DraftWeights.declare
MiniMaxM3Tp2DraftWeights.dims
MiniMaxM3Tp2DraftWeights.expected_unconsumed
MiniMaxM3Tp2DraftWeights.load
MiniMaxM3Tp2DraftWeights.manifest
MiniMaxM3Tp2Weights._targets
MiniMaxM3Tp2Weights.declare
MiniMaxM3Tp2Weights.dims
MiniMaxM3Tp2Weights.expected_unconsumed
MiniMaxM3Tp2Weights.load
MiniMaxM3Tp2Weights.manifest
W.layer_indices
_col
_col_w
_dense_layers
_dev
_draft_fc_norms
_draft_qkv
_expert_s2
_expert_s31
_expert_w2
_expert_w31
_f32
_fill
_five_way_qkv_layers
_gate_up_s
_gate_up_w
_gemma_fold
_index_qk_w
_materialize
_per_expert.<locals>.build
_qkv_dense_s
_qkv_dense_w
_qkv_sparse_s
_qkv_sparse_w
_release
_row
_row_s
_row_w
_sparse_layers
_swizzle
_three_way_qkv_layers
_unfused_sparse_layers
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/modeling.py`

库存函数 176；已覆盖 58；未覆盖 118。

```text
DecodeTarget._attention
DecodeTarget._attention_core
DecodeTarget._attn_push_target
DecodeTarget._boundary
DecodeTarget._bucket_flags
DecodeTarget._check_step_contract
DecodeTarget._decode_sparse_core
DecodeTarget._embed
DecodeTarget._fullctx_decode
DecodeTarget._fullctx_plan
DecodeTarget._generic_boundary
DecodeTarget._layer
DecodeTarget._moe
DecodeTarget._moe_push_targets
DecodeTarget._next_kv_prefetch
DecodeTarget._no_bucket
DecodeTarget._prepushed
DecodeTarget._routed
DecodeTarget._routed_chunks
DecodeTarget._swiglu_mlp
DecodeTarget._unfused_sparse_attention
DecodeTarget._unfused_sparse_attention.<locals>._index
DecodeTarget._unfused_sparse_attention.<locals>._main
DecodeTarget.step_args
M3Eagle3Draft.__init__
M3Eagle3Draft.apply_eagle3_fc
M3Eagle3Draft.ar_norm
M3Eagle3Draft.attention
M3Eagle3Draft.embed_table
M3Eagle3Draft.fc
M3Eagle3Draft.forward_layer
M3Eagle3Draft.load_weights_from_target_model
M3Eagle3Draft.mlp
M3Eagle3Worker.__init__
M3Eagle3Worker._forward_draft_loop
M3Eagle3Worker._forward_impl
M3Eagle3Worker._m3_arm_draft_push
M3Eagle3Worker._m3_coded_head
M3Eagle3Worker._m3_draft_ar_norm
M3Eagle3Worker._m3_drafter_supported
M3Eagle3Worker._m3_fc_fork
M3Eagle3Worker._m3_first_drafter_inputs
M3Eagle3Worker._m3_force_vec
M3Eagle3Worker._m3_fused_draft_loop
M3Eagle3Worker._m3_fused_loop_ok
M3Eagle3Worker._m3_sampled_verify_ok
M3Eagle3Worker._m3_seed_first_pass
M3Eagle3Worker._m3_structural_ok
M3Eagle3Worker._m3_target_accept
M3Eagle3Worker.capture_hidden_states
M3Eagle3Worker.capture_target
M3Eagle3Worker.draft_only_eligible
M3Eagle3Worker.prebuild
M3Eagle3Worker.prebuild_draft_head
M3Eagle3Worker.prebuild_target_head
M3Eagle3Worker.prepare_1st_drafter_inputs
M3Eagle3Worker.tail_eligible
M3SpecStepFold.gather
M3SpecStepFold.scatter
MiniMaxM3Tp4ModelingV2.__init__
MiniMaxM3Tp4ModelingV2._build_embed_owners
MiniMaxM3Tp4ModelingV2._build_engine
MiniMaxM3Tp4ModelingV2._build_moe16_engine
MiniMaxM3Tp4ModelingV2._derive_moe
MiniMaxM3Tp4ModelingV2._expert_operands
MiniMaxM3Tp4ModelingV2._shared_weights
MiniMaxM3Tp4ModelingV2.cache_derived_state
MiniMaxM3Tp4ModelingV2.forward
MiniMaxM3Tp4ModelingV2.post_load_weights
MiniMaxM3Tp4ModelingV2.projections
MiniMaxM3Tp4ModelingV2.sm_scale
ModelingV2MiniMaxM3Nvfp4Sm107Tp4.__init__
ModelingV2MiniMaxM3Nvfp4Sm107Tp4.cache_derived_state
ModelingV2MiniMaxM3Nvfp4Sm107Tp4.forward
ModelingV2MiniMaxM3Nvfp4Sm107Tp4.load_draft_weights
ModelingV2MiniMaxM3Nvfp4Sm107Tp4.load_weights
ModelingV2MiniMaxM3Nvfp4Sm107Tp4.post_load_weights
ModelingV2MiniMaxM3Nvfp4Sm107Tp4.set_guided_decoder
PrefillTarget._attention
PrefillTarget._attention_core
PrefillTarget._check_step_contract
PrefillTarget._generic_boundary
PrefillTarget._moe
PrefillTarget._routed
PrefillTarget._routed_chunks
PrefillTarget._swiglu_mlp
PrefillTarget._unfused_sparse_attention
PrefillTarget._unfused_sparse_attention.<locals>._index
PrefillTarget._unfused_sparse_attention.<locals>._main
PrefillTarget.forward
PrefillTarget.step_args
_DraftForward.__call__
_DraftForward.__init__
_build_speculation
_can_mc
_dense_swiglu
_eagle3_capture
_fp8_index_qk_norm_rope
_linear
_mc_allreduce_norm
_mxfp8_linear
_native_allreduce_norm
_note_mx_tac
_nvfp4_fused_write_layout
_nvfp4_layer
_plain_capture
_print_values
_producer_covered
_producer_layout
_producer_miss
_qk_norm_rope
_routed_experts
_sharded_head
_sparse_qkv_producer
_split_chunk
_uncompiled_note
capture_missed
plan_for
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/attention/m3_sparse_decode_lc.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
_identity_table
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/embedding_firstnorm/op.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
DecodeEmbedding.__call__
DecodeEmbedding.__init__
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/m3_draft_lowm.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
arm_push
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/m3_draft_step.py`

库存函数 7；已覆盖 3；未覆盖 4。

```text
_num_sms
draft_step
head_supported
workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/m3_fullctx_decode.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
_check_layout
emits_mxfp8
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/m3_index_score_t2sc.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
IndexDecodeScoreKernel.__call__
IndexDecodeScoreKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/m3_mc_allreduce.py`

库存函数 12；已覆盖 3；未覆盖 9。

```text
McBuckets.__init__
McBuckets.for_tokens
_bucket
_capture_args
_tp_comm
decline_reason
get_workspace
m3_mc_allreduce_norm
m3_mc_finalize_allreduce_norm
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/m3_moe/m3_moe16_kernel.py`

库存函数 64；已覆盖 58；未覆盖 6。

```text
_atom_add_u64
_bar_arrive
_fence_acq_rel_gpu
_gtime
_quant_block
_smid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/m3_report.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_noop
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/m3_tokmap_guard.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
_no_missed
_no_taken
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/moe/m3_moe_engine.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_shared_expert
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/mx_gemv/op.py`

库存函数 22；已覆盖 21；未覆盖 1。

```text
m3_mx_gemv_swiglu
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/ops/spec/m3_spec_offsets.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
M3SpecOffsets.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp4/weights.py`

库存函数 49；已覆盖 5；未覆盖 44。

```text
MiniMaxM3Tp4DraftWeights._targets
MiniMaxM3Tp4DraftWeights.declare
MiniMaxM3Tp4DraftWeights.dims
MiniMaxM3Tp4DraftWeights.expected_unconsumed
MiniMaxM3Tp4DraftWeights.load
MiniMaxM3Tp4DraftWeights.manifest
MiniMaxM3Tp4Weights._targets
MiniMaxM3Tp4Weights.declare
MiniMaxM3Tp4Weights.dims
MiniMaxM3Tp4Weights.expected_unconsumed
MiniMaxM3Tp4Weights.load
MiniMaxM3Tp4Weights.manifest
W.layer_indices
_col
_col_w
_dense_layers
_dev
_draft_fc_norms
_draft_qkv
_expert_s2
_expert_s31
_expert_w2
_expert_w31
_f32
_fill
_five_way_qkv_layers
_gate_up_s
_gate_up_w
_gemma_fold
_index_qk_w
_materialize
_per_expert.<locals>.build
_qkv_dense_s
_qkv_dense_w
_qkv_sparse_s
_qkv_sparse_w
_release
_row
_row_s
_row_w
_sparse_layers
_swizzle
_three_way_qkv_layers
_unfused_sparse_layers
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/modeling.py`

库存函数 195；已覆盖 57；未覆盖 138。

```text
DecodeTarget._attention
DecodeTarget._attention_core
DecodeTarget._attn_push_target
DecodeTarget._boundary
DecodeTarget._bucket_flags
DecodeTarget._check_step_contract
DecodeTarget._decode_sparse_core
DecodeTarget._dense_attention
DecodeTarget._embed
DecodeTarget._fullctx_decode
DecodeTarget._fullctx_plan
DecodeTarget._generic_boundary
DecodeTarget._index_scores
DecodeTarget._layer
DecodeTarget._moe
DecodeTarget._moe_push_targets
DecodeTarget._next_kv_prefetch
DecodeTarget._no_bucket
DecodeTarget._prepushed
DecodeTarget._routed
DecodeTarget._routed_chunks
DecodeTarget._select
DecodeTarget._swiglu_mlp
DecodeTarget._unfused_sparse_attention
DecodeTarget._unfused_sparse_attention.<locals>._index
DecodeTarget._unfused_sparse_attention.<locals>._main
DecodeTarget.forward
DecodeTarget.step_args
M3Eagle3Draft.__init__
M3Eagle3Draft.apply_eagle3_fc
M3Eagle3Draft.ar_norm
M3Eagle3Draft.arm_push
M3Eagle3Draft.attention
M3Eagle3Draft.embed_table
M3Eagle3Draft.fc
M3Eagle3Draft.forward_layer
M3Eagle3Draft.load_weights_from_target_model
M3Eagle3Draft.mlp
M3Eagle3Worker.__init__
M3Eagle3Worker._forward_draft_loop
M3Eagle3Worker._forward_impl
M3Eagle3Worker._m3_arm_draft_push
M3Eagle3Worker._m3_draft_ar_norm
M3Eagle3Worker._m3_drafter_supported
M3Eagle3Worker._m3_fc_bump
M3Eagle3Worker._m3_fc_fork
M3Eagle3Worker._m3_first_drafter_inputs
M3Eagle3Worker._m3_force_vec
M3Eagle3Worker._m3_fused_draft_loop
M3Eagle3Worker._m3_fused_loop_ok
M3Eagle3Worker._m3_join_pending
M3Eagle3Worker._m3_sampled_verify_ok
M3Eagle3Worker._m3_seed_first_pass
M3Eagle3Worker._m3_structural_ok
M3Eagle3Worker._m3_target_accept
M3Eagle3Worker.capture_hidden_states
M3Eagle3Worker.draft_only_eligible
M3Eagle3Worker.prebuild
M3Eagle3Worker.prepare_1st_drafter_inputs
M3Eagle3Worker.tail_eligible
M3SpecStepFold.gather
M3SpecStepFold.scatter
MiniMaxM3Tp8ModelingV2.__init__
MiniMaxM3Tp8ModelingV2._build_embed_owners
MiniMaxM3Tp8ModelingV2._build_engine
MiniMaxM3Tp8ModelingV2._derive_moe
MiniMaxM3Tp8ModelingV2._expert_operands
MiniMaxM3Tp8ModelingV2._in_tree_scorer_variants
MiniMaxM3Tp8ModelingV2._index_geometries
MiniMaxM3Tp8ModelingV2._prebuild_index_scorer
MiniMaxM3Tp8ModelingV2._scorer_variants
MiniMaxM3Tp8ModelingV2._select_target
MiniMaxM3Tp8ModelingV2._shared_weights
MiniMaxM3Tp8ModelingV2.cache_derived_state
MiniMaxM3Tp8ModelingV2.decode_query_lens
MiniMaxM3Tp8ModelingV2.forward
MiniMaxM3Tp8ModelingV2.post_load_weights
MiniMaxM3Tp8ModelingV2.prebuild_kernels
MiniMaxM3Tp8ModelingV2.projections
MiniMaxM3Tp8ModelingV2.sm_scale
ModelingV2MiniMaxM3Nvfp4Sm107Tp8.__init__
ModelingV2MiniMaxM3Nvfp4Sm107Tp8.cache_derived_state
ModelingV2MiniMaxM3Nvfp4Sm107Tp8.forward
ModelingV2MiniMaxM3Nvfp4Sm107Tp8.load_draft_weights
ModelingV2MiniMaxM3Nvfp4Sm107Tp8.load_weights
ModelingV2MiniMaxM3Nvfp4Sm107Tp8.post_load_weights
ModelingV2MiniMaxM3Nvfp4Sm107Tp8.set_guided_decoder
PrefillTarget._attention
PrefillTarget._attention_core
PrefillTarget._check_step_contract
PrefillTarget._generic_boundary
PrefillTarget._moe
PrefillTarget._routed
PrefillTarget._routed_chunks
PrefillTarget._swiglu_mlp
PrefillTarget._unfused_sparse_attention
PrefillTarget._unfused_sparse_attention.<locals>._index
PrefillTarget._unfused_sparse_attention.<locals>._main
PrefillTarget.forward
PrefillTarget.step_args
_DeferredGemv.__init__
_DeferredGemv.materialize
_DraftForward.__call__
_DraftForward.__init__
_build_speculation
_can_mc
_decode_router_logits
_dense_fp8_op_applies
_dense_qk_norm_rope_fp8
_dense_swiglu
_dummy_scorer_operands
_eagle3_capture
_fp8_index_qk_norm_rope
_gemv_variants
_in_tree_scorer
_linear
_log_engaged
_mc_allreduce_norm
_mxfp8_linear
_native_allreduce_norm
_note_mx_tac
_nvfp4_fused_write_layout
_nvfp4_layer
_plain_capture
_prebuild_builtin_sparse_decode
_print_values
_producer_covered
_producer_layout
_producer_miss
_qk_norm_rope
_qkv_producer_fold
_routed_experts
_router_logits
_sharded_head
_sparse_qkv_producer
_split_chunk
_uncompiled_note
capture_missed
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/attention/m3_sparse_decode_lc8.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
_identity_table
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/embedding_firstnorm/op.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
DecodeEmbedding.__call__
DecodeEmbedding.__init__
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_draft_step.py`

库存函数 9；已覆盖 5；未覆盖 4。

```text
_draft_step_split
_num_sms
draft_step
head_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_fullctx_decode.py`

库存函数 8；已覆盖 5；未覆盖 3。

```text
_check_layout
_tp8_reduce_op
emits_mxfp8
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_mc_allreduce.py`

库存函数 15；已覆盖 6；未覆盖 9。

```text
McBuckets.__init__
McBuckets.for_tokens
_bucket
_capture_args
_tp_comm
decline_reason
get_workspace
m3_mc_allreduce_norm
m3_mc_finalize_allreduce_norm
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_moe/op.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
_shared_views
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_report.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_noop
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_sampled_accept.py`

库存函数 7；已覆盖 1；未覆盖 6。

```text
SampledAcceptWorkspace.__init__
SampledAcceptWorkspace.create
_ilp_walk
_op_args
one_round_k
sampled_accept
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_select_long_dsl.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
prebuild
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_spec_tail.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
capture_norm
chunk_norm
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/m3_tokmap_guard.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
_no_missed
_no_taken
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/moe/m3_moe_engine.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_shared_expert
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/mx_gemv/op.py`

库存函数 23；已覆盖 18；未覆盖 5。

```text
ensure_compiled
freeze
frozen
is_compiled
precompile
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/ops/spec/m3_spec_offsets.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
M3SpecOffsets.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_107__tp8/weights.py`

库存函数 50；已覆盖 5；未覆盖 45。

```text
MiniMaxM3Tp8DraftWeights._targets
MiniMaxM3Tp8DraftWeights.declare
MiniMaxM3Tp8DraftWeights.dims
MiniMaxM3Tp8DraftWeights.expected_unconsumed
MiniMaxM3Tp8DraftWeights.load
MiniMaxM3Tp8DraftWeights.manifest
MiniMaxM3Tp8Weights._targets
MiniMaxM3Tp8Weights.declare
MiniMaxM3Tp8Weights.dims
MiniMaxM3Tp8Weights.expected_unconsumed
MiniMaxM3Tp8Weights.load
MiniMaxM3Tp8Weights.manifest
W.layer_indices
_col
_col_w
_dense_layers
_dev
_draft_fc_norms
_draft_qkv
_expert_s2
_expert_s31
_expert_w2
_expert_w31
_f32
_fill
_five_way_qkv_layers
_gate_up_s
_gate_up_w
_gemma_fold
_heads
_index_qk_w
_materialize
_per_expert.<locals>.build
_qkv_dense_s
_qkv_dense_w
_qkv_sparse_s
_qkv_sparse_w
_release
_row
_row_s
_row_w
_sparse_layers
_swizzle
_three_way_qkv_layers
_unfused_sparse_layers
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/modeling_support.py`

库存函数 11；已覆盖 4；未覆盖 7。

```text
_build_speculation
_eagle3_capture
_fp8_index_qk_norm_rope
_mxfp8_linear
_nvfp4_fused_write_layout
_nvfp4_layer
_split_chunk
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/engine.py`

库存函数 9；已覆盖 3；未覆盖 6。

```text
arm_target_paths
create_msa_attention
indexer_prewarm
require_msa_module
use_family_backend
warmup_scope
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/kernels/msa_utils.py`

库存函数 12；已覆盖 6；未覆盖 6。

```text
build_kv_page_indices
check_decode_span_shape
is_msa_layer
msa_paged_kv
write_msa_main_kv
write_msa_phase_kv
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/kernels/nvfp4_index_k.py`

库存函数 12；已覆盖 7；未覆盖 5。

```text
compact_step_pages
decode_prewait_prefetch
gather_fp8
is_nvfp4_index_cache
set_decode_prewait_prefetch
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/msa_backend.py`

库存函数 65；已覆盖 3；未覆盖 62。

```text
MiniMaxM31MsaSparseAttention.__init__
MiniMaxM31MsaSparseAttention._validate_fmha_pair
MiniMaxM31MsaSparseAttention._validate_msa_preconditions
MiniMaxM31MsaSparseAttention.sparse_attn_predict
MiniMaxM31MsaSparseAttention.sparse_kv_predict
MiniMaxM31MsaSparseAttention.support_fused_rope
MiniMaxM31MsaSparseAttention.update_quant_config
MiniMaxM31MsaSparseAttention.write_layer_caches
MiniMaxM31MsaSparseAttentionMetadata.__post_init__
MiniMaxM31MsaSparseAttentionMetadata._alloc_msa_proxy_scratch
MiniMaxM31MsaSparseAttentionMetadata._build_msa_fields
MiniMaxM31MsaSparseAttentionMetadata._build_step_plans
MiniMaxM31MsaSparseAttentionMetadata._build_step_plans.<locals>.build_proxy_plans
MiniMaxM31MsaSparseAttentionMetadata._build_step_plans.<locals>.plan_for
MiniMaxM31MsaSparseAttentionMetadata._check_beam_width
MiniMaxM31MsaSparseAttentionMetadata._check_capture_is_pure_decode
MiniMaxM31MsaSparseAttentionMetadata._create_msa_buffers
MiniMaxM31MsaSparseAttentionMetadata._cutedsl_indexer_supported
MiniMaxM31MsaSparseAttentionMetadata._ensure_msa_decode_scratch_buffers
MiniMaxM31MsaSparseAttentionMetadata._ensure_prefill_n_valid_buffer
MiniMaxM31MsaSparseAttentionMetadata._ensure_proxy_plan
MiniMaxM31MsaSparseAttentionMetadata._fill_tail_in_forward
MiniMaxM31MsaSparseAttentionMetadata._first_plan_read
MiniMaxM31MsaSparseAttentionMetadata._kv_update_with_tail
MiniMaxM31MsaSparseAttentionMetadata._msa_attn_plan_rows
MiniMaxM31MsaSparseAttentionMetadata._msa_index_kv_dtype
MiniMaxM31MsaSparseAttentionMetadata._msa_kv_lens_may_change
MiniMaxM31MsaSparseAttentionMetadata._msa_main_kv_is_fp8
MiniMaxM31MsaSparseAttentionMetadata._msa_max_decode_query_len
MiniMaxM31MsaSparseAttentionMetadata._msa_max_decode_tokens
MiniMaxM31MsaSparseAttentionMetadata._msa_proxy_plan_kwargs
MiniMaxM31MsaSparseAttentionMetadata._msa_proxy_plan_rows
MiniMaxM31MsaSparseAttentionMetadata._msa_runs_no_fmha
MiniMaxM31MsaSparseAttentionMetadata._set_decode_span
MiniMaxM31MsaSparseAttentionMetadata._stage_host_lengths
MiniMaxM31MsaSparseAttentionMetadata._stage_host_lengths.<locals>.as_pinned_int32
MiniMaxM31MsaSparseAttentionMetadata._stage_msa_short_table
MiniMaxM31MsaSparseAttentionMetadata._validate_decode_kernel_support
MiniMaxM31MsaSparseAttentionMetadata.msa_decode_query_len
MiniMaxM31MsaSparseAttentionMetadata.msa_decode_span
MiniMaxM31MsaSparseAttentionMetadata.msa_idx_k_cache
MiniMaxM31MsaSparseAttentionMetadata.msa_kv_lens_cpu
MiniMaxM31MsaSparseAttentionMetadata.msa_max_blocks_per_seq
MiniMaxM31MsaSparseAttentionMetadata.msa_max_kv_len
MiniMaxM31MsaSparseAttentionMetadata.msa_prefill_dense_plan
MiniMaxM31MsaSparseAttentionMetadata.msa_prefill_gqa_plan
MiniMaxM31MsaSparseAttentionMetadata.msa_prefill_n_valid_blocks
MiniMaxM31MsaSparseAttentionMetadata.msa_prefill_proxy_kv_indices
MiniMaxM31MsaSparseAttentionMetadata.msa_prefill_proxy_plan
MiniMaxM31MsaSparseAttentionMetadata.msa_proxy_max_score_view
MiniMaxM31MsaSparseAttentionMetadata.msa_qo_lens_cpu
MiniMaxM31MsaSparseAttentionMetadata.msa_qo_offset_cpu
MiniMaxM31MsaSparseAttentionMetadata.msa_subpage_rows
MiniMaxM31MsaSparseAttentionMetadata.msa_worst_case_max_k_tiles
MiniMaxM31MsaSparseAttentionMetadata.msa_write_idx_k
MiniMaxM31MsaSparseAttentionMetadata.on_update_kv_lens
MiniMaxM31MsaSparseAttentionMetadata.prepare
_cache_device
_fill_unused_slots
_host_prefix_sum
_log_mix_reader_taken
_worst_case_proxy_max_k_tiles
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/msa_indexer.py`

库存函数 33；已覆盖 16；未覆盖 17。

```text
MsaIndexer.select_prefill_hooked
_combined_topk_table
_count_prefill_scorer
_cutedsl_score
_dep_vllm_topk_decline
_empty_topk_table
_plan_takes_maxscore_tile
_proxy_max_score
arm_prefill_context_scorer
arm_proxy_ms_reuse
arm_sm107_msa_paths
cutedsl_score_runner
prefill_context_scorer_accepts
prefill_context_scorer_armed
prewarm_prefill_context_scorer
prewarm_split_proxy_variants
proxy_plan_kwargs
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/msa/msa_prefill.py`

库存函数 31；已覆盖 13；未覆盖 18。

```text
M31MsaPrefillFmha._is_available
M31MsaPrefillFmha._is_supported
M31MsaPrefillFmha.run_context
_aligned_nvfp4_dequant_scales
_combine_takes
_csr_builder_takes_order
_k2q_lpt_parts
_overlaps
_overlaps.<locals>.span
_prefill_split_workspace
_run_tp2ctx_bf16x
_sparse_func_takes_out
_tp2ctx_e4m3x_count
_tp2ctx_e4m3x_ok
arm_prefill_fork
run_msa_prefill_gqa
run_msa_prefill_gqa.<locals>.rows_of
run_msa_sparse_gqa
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/activation/m3_shared_swiglu_mxfp8.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
_rcp_approx_ftz
_swq_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/add_rmsnorm.py`

库存函数 15；已覆盖 7；未覆盖 8。

```text
M31FusedAddRmsnorm.__call__
_cute
_plain_forward
install_plain
is_plain_site
l2pf_ranges
takes_prefetch
uninstall_plain
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/attention/drafter.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
DrafterFullCtxBase._shape
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/attention/token_map.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
_LogSink.info
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/comm/m3_mc_allreduce_norm.py`

库存函数 7；已覆盖 1；未覆盖 6。

```text
M3McAllreduceNorm.is_valid
M3McAllreduceNorm.reference
boundary
mxfp8_rows
sf_written
swizzle_sf
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/comm/tp2ctx_ipc.py`

库存函数 26；已覆盖 14；未覆盖 12。

```text
Tp2ctxIpcWorkspace._router_full
Tp2ctxIpcWorkspace.input_slot
Tp2ctxIpcWorkspace.owned_slot_ptrs
Tp2ctxIpcWorkspace.peer_sf_slice
Tp2ctxIpcWorkspace.reduce_scatter_owned
Tp2ctxIpcWorkspace.route_owned_full_shape
Tp2ctxIpcWorkspace.router_owned_input
Tp2ctxIpcWorkspace.supports
_nccl_fallback_bytes
enabled
get_workspace
router_zc_enabled
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/dflash_merge.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
install
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/dspark_prefill_store.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
engagement
install
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/dspark/strict_accept.py`

库存函数 24；已覆盖 20；未覆盖 4。

```text
_
is_compiled_kv_rewind
precompile
supports_kv_rewind
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/embedding/decode_embedding.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
call_decode_embedding
init_decode_embedding
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/flagfree_gather.py`

库存函数 25；已覆盖 1；未覆盖 24。

```text
FlagFreeGather.__init__
FlagFreeGather._expected
FlagFreeGather._gen
FlagFreeGather._log_once
FlagFreeGather._serve
FlagFreeGather._stock
FlagFreeGather.describe
FlagFreeGather.fc_output
FlagFreeGather.self_check
FlagFreeGather.verify_logits
_Gather.__init__
_Gather._args
_Gather._shard_dtype
_Gather.build
_Gather.compile_all
_Gather.describe
_Gather.run
_Gather.takes
_Gather.why_not
_agree
_arg
_i32
attach
pick_grid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/gemm/locality.py`

库存函数 17；已覆盖 12；未覆盖 5。

```text
LocalityMap.logical_map
LocalityMap.replace_param_
LocalityMap.sm_table
LocalityMap.sms
LocalityMap.split_rows
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/gemm/m31_fc_shard.py`

库存函数 10；已覆盖 4；未覆盖 6。

```text
FcShard.__call__
FcShard.applies
FcShard.build
FcShard.gather
FcShard.partial
FcShard.set_fast_gather
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/gemm/m3_dense_swiglu.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
M3DenseSwiglu.is_valid
M3DenseSwiglu.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/gemm/prequant.py`

库存函数 14；已覆盖 6；未覆盖 8。

```text
ArPushSlot.arm
ArPushSlot.take_pushed
ArPushSlot.take_target
MoePushSlot.arm
MoePushSlot.disarm
MoePushSlot.take_engine
MoePushSlot.take_pushed
MoePushSlot.take_shared
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/gumbel_draft.py`

库存函数 8；已覆盖 4；未覆盖 4。

```text
_not_taken
engagement
install
log_engagement
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/m3_fsel_topk.py`

库存函数 10；已覆盖 6；未覆盖 4。

```text
M3FselTopk.is_valid
log_engaged
ms4_available
ms4_variant
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/moe/m3_router_gemv.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
M3RouterGemv.__call__
M3RouterGemv.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/moe/megamoe_local/local_impl.py`

库存函数 28；已覆盖 1；未覆盖 27。

```text
MegaMoEDeferredTopkReduce.add_into
MegaMoEDeferredTopkReduce.add_rmsnorm_into
TrtllmCutedslMegaMoeLocalNvfp4Impl.__init__
TrtllmCutedslMegaMoeLocalNvfp4Impl._route_quant_decline_reason
TrtllmCutedslMegaMoeLocalNvfp4Impl._route_quant_empty
TrtllmCutedslMegaMoeLocalNvfp4Impl._route_quant_port_quantizer
TrtllmCutedslMegaMoeLocalNvfp4Impl._run_moe
TrtllmCutedslMegaMoeLocalNvfp4Impl._stage_inputs
TrtllmCutedslMegaMoeLocalNvfp4Impl._staged_route_quant
TrtllmCutedslMegaMoeLocalNvfp4Impl._supports_load_balancer
TrtllmCutedslMegaMoeLocalNvfp4Impl._take_pad_mask
TrtllmCutedslMegaMoeLocalNvfp4Impl._take_prestaged_route_quant
TrtllmCutedslMegaMoeLocalNvfp4Impl.can_implement
TrtllmCutedslMegaMoeLocalNvfp4Impl.cancel_deferred_topk_reduce
TrtllmCutedslMegaMoeLocalNvfp4Impl.pad_mask_decline_reason
TrtllmCutedslMegaMoeLocalNvfp4Impl.prestage_route_quant
TrtllmCutedslMegaMoeLocalNvfp4Impl.request_deferred_topk_reduce
TrtllmCutedslMegaMoeLocalNvfp4Impl.route_beside_shared
TrtllmCutedslMegaMoeLocalNvfp4Impl.set_pad_mask
TrtllmCutedslMegaMoeLocalNvfp4Impl.tail_decline_reason
TrtllmCutedslMegaMoeLocalNvfp4Impl.tail_static_decline_reason
TrtllmCutedslMegaMoeLocalNvfp4Impl.try_fused_route_quant
_flashinfer_fp4_quantize_tail_args
_flashinfer_nvfp4_quantizer
_load_flashinfer_nvfp4_quantizer
_route_quant_key
is_registered
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/moe/phase_plan.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
make_phase_plan_tp8.<locals>.counts_for
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/moe/runtime.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
ar_push_desc
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/native_swiglu.py`

库存函数 9；已覆盖 3；未覆盖 6。

```text
_once
engagement
install
install.<locals>.apply
serves
swiglu_oai_torch
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/prefill_pad_rows.py`

库存函数 10；已覆盖 4；未覆盖 6。

```text
PrefillPadRows.allocated
PrefillPadRows.end
PrefillPadRows.hand_over
band_straddles
moe_backend
wire
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/prob1_worker.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
log_fault
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/ops/router_bf16_split.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
parse_terms
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/reporting.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
capture_missed
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/runtime_hooks_compat.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
requested
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/tile_select.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
TileSelect.register
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/kernels/moe/blockscaled_contiguous_grouped_gemm_finalize_fusion.py`

库存函数 21；已覆盖 8；未覆盖 13。

```text
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.__call__
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.__call__.<locals>._make_tma_b
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel._expert_count_tile_info
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel._get_tma_atom_kind
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.is_valid_dtypes_and_scale_factor_vec_size
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.is_valid_layouts
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.is_valid_mma_tiler_and_cluster_shape
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.is_valid_tensor_alignment
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.is_valid_tensor_alignment.<locals>.check_contigous_16B_alignment
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.kernel
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.wrapper
cvt_sf_MKL_to_M32x4xrm_K4xrk_L
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/activation/silu_and_mul_rows.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_silu_and_mul_rows_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_canonical_selection.py`

库存函数 12；已覆盖 4；未覆盖 8。

```text
_canonical_rows_kernel
_count
_kth_value
_load_window
_split_count_kernel
_split_kth_kernel
_split_write_kernel
_write_block
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_nvfp4_pages_to_bf16.py`

库存函数 7；已覆盖 5；未覆盖 2。

```text
_dequantize_head_page
_nvfp4_pages_to_bf16_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_paged_index_scores.py`

库存函数 15；已覆盖 13；未覆盖 2。

```text
_qsa_index_scores_group_kernel
_qsa_index_scores_visible_split_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_paged_index_store.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
_qsa_paged_index_store_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_paged_kv_store.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
_bits
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_paged_kv_store_nvfp4.py`

库存函数 10；已覆盖 6；未覆盖 4。

```text
_e2m1x2
_qsa_paged_kv_store_nvfp4_kernel
_quantize_row
_rcp_approx
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_paged_sparse_gqa_decode.py`

库存函数 18；已覆盖 14；未覆盖 4。

```text
_MergeKernel.__call__
_MergeKernel.kernel
_SplitKKernel.__call__
_SplitKKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_paged_sparse_gqa_nvfp4.py`

库存函数 16；已覆盖 9；未覆盖 7。

```text
_attend_tile
_e2m1_to_f32
_load_halves
_nvfp4_halves
_nvfp4_merge_splitk_kernel
_nvfp4_sparse_gqa_prefill_kernel
_nvfp4_sparse_gqa_splitk_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_paged_sparse_gqa_nvfp4_decode.py`

库存函数 47；已覆盖 38；未覆盖 9。

```text
_MergeKernel.__call__
_MergeKernel.kernel
_SplitKernel.__call__
_SplitKernel._issue
_SplitKernel.kernel
_cp16
_e4m3x2
_lds_u16
_stg4
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_paged_sparse_gqa_nvfp4_verify.py`

库存函数 39；已覆盖 21；未覆盖 18。

```text
_RowMergeKernel.__call__
_RowMergeKernel.kernel
_VerifyKernel.__call__
_VerifyKernel.__call__.<locals>.pv
_VerifyKernel.__call__.<locals>.qk
_VerifyKernel._decode_row
_VerifyKernel._pass_rows
_VerifyKernel._pass_tiles
_VerifyKernel._resolve
_VerifyKernel._union
_VerifyKernel.dequant_tiles
_VerifyKernel.epilogue
_VerifyKernel.issue_qk
_VerifyKernel.kernel
_VerifyKernel.load_tiles
_VerifyKernel.mma_tiles
_VerifyKernel.softmax_tiles
_VerifyKernel.stage_q
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_rollback_planes.py`

库存函数 10；已覆盖 8；未覆盖 2。

```text
_gather_kernel
_restore_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_rowtile_attention_cute.py`

库存函数 30；已覆盖 12；未覆盖 18。

```text
_RowtileAttentionKernel.__call__
_RowtileAttentionKernel.__call__.<locals>.pv
_RowtileAttentionKernel.__call__.<locals>.qk
_RowtileAttentionKernel.advance
_RowtileAttentionKernel.box_offset
_RowtileAttentionKernel.copy_q
_RowtileAttentionKernel.gemm_pv
_RowtileAttentionKernel.gemm_qk
_RowtileAttentionKernel.issue_q
_RowtileAttentionKernel.issue_tile
_RowtileAttentionKernel.item_info
_RowtileAttentionKernel.kernel
_RowtileAttentionKernel.load_role
_RowtileAttentionKernel.mma_role
_RowtileAttentionKernel.publish
_RowtileAttentionKernel.read_item
_RowtileAttentionKernel.release_item
_RowtileAttentionKernel.softmax_role
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_sparse_gqa_rowtile.py`

库存函数 21；已覆盖 9；未覆盖 12。

```text
_RowBitsKernel.__call__
_RowBitsKernel.kernel
_bits_below
_compress_bits
_group_rows
_or_combine
_popc64
_rowtile_group_kernel
_rowtile_lone_kernel
_rowtile_lone_row
_rowtile_slots_kernel
_rowtile_union_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qsa_topk_expand_decode.py`

库存函数 26；已覆盖 14；未覆盖 12。

```text
_TopkExpandKernel.__call__
_TopkExpandKernel.kernel
_TopkExpandWideKernel.__call__
_TopkExpandWideKernel._cut
_TopkExpandWideKernel._emit
_TopkExpandWideKernel._mark
_TopkExpandWideKernel._narrow
_TopkExpandWideKernel._passes
_TopkExpandWideKernel._select_listed
_TopkExpandWideKernel._select_window
_TopkExpandWideKernel._slab
_TopkExpandWideKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qwen4_exp_attn_decode_prologue.py`

库存函数 11；已覆盖 3；未覆盖 8。

```text
_attn_decode_prologue_kernel
_compress_factors
_compress_norm_rope
_norm_rope_apply
_norm_rope_tile
_norm_weights
_quantize_tile
_rope_factors
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/qwen4_exp_qk_norm_rope.py`

库存函数 47；已覆盖 43；未覆盖 4。

```text
_NormRopeRing.__call__
_NormRopeRing.kernel
_norm_rope_tile
_qwen4_exp_qk_norm_rope_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/spec_rollback_commit.py`

库存函数 9；已覆盖 7；未覆盖 2。

```text
_commit_kernel
_stage_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/attention/verify_step_setup.py`

库存函数 5；已覆盖 2；未覆盖 3。

```text
VerifyStepSetup.reference
_setup_kernel
_slot
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/gemm/bf16_gemv_pdl.py`

库存函数 13；已覆盖 11；未覆盖 2。

```text
_GemvKernel.__call__
_GemvKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/gemm/fp8_gemv.py`

库存函数 70；已覆盖 56；未覆盖 14。

```text
Fp8GemvConv1d.reference
_Fp8Gemv.__call__
_Fp8Gemv._compute
_Fp8Gemv._compute_w4
_Fp8Gemv._issue_w
_Fp8Gemv._issue_x
_Fp8Gemv._prefetch_w
_Fp8Gemv.kernel
_cluster_arrive_relaxed
_cluster_arrive_release
_cluster_wait
_cluster_wait_acquire
_mapa
_st_dsmem_f32
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/gemm/fp8_per_tensor_mm.py`

库存函数 8；已覆盖 5；未覆盖 3。

```text
Fp8PerTensorMm.is_valid
_amax_kernel
_quant_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/gemm/mid_rows_mm.py`

库存函数 49；已覆盖 42；未覆盖 7。

```text
Fp8RowQuant.reference
MidRowsMm.reference
_MidMm.__call__
_MidMm.kernel
_RowQuant.__call__
_RowQuant.kernel
_bf16x2
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/gemm/nvfp4_gemv.py`

库存函数 38；已覆盖 34；未覆盖 4。

```text
Nvfp4SplitGemvConv1d.reference
_Nvfp4Gemv.__call__
_Nvfp4Gemv._compute
_Nvfp4Gemv.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/hc/qwen4_exp_hc_decode_first_cute.py`

库存函数 10；已覆盖 8；未覆盖 2。

```text
_FirstKernel.__call__
_FirstKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/hc/qwen4_exp_hc_decode_mix_fp4_cute.py`

库存函数 15；已覆盖 13；未覆盖 2。

```text
_MixKernel.__call__
_MixKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/hc/qwen4_exp_hc_decode_rows_tc.py`

库存函数 27；已覆盖 17；未覆盖 10。

```text
Qwen4ExpHcDecodeRowsFirst.is_valid
Qwen4ExpHcDecodeRowsFirst.reference
Qwen4ExpHcDecodeRowsMix.is_valid
Qwen4ExpHcDecodeRowsMix.reference
_First2.__call__
_First2.kernel
_Mix2.__call__
_Mix2.kernel
_reference_first
_st_dsmem_f32
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/hc/qwen4_exp_hc_first_rescale_tc.py`

库存函数 19；已覆盖 16；未覆盖 3。

```text
HcFirstRescaleKernel.__call__
HcFirstRescaleKernel.kernel
_combine_act_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/hc/qwen4_exp_hc_inject_norm.py`

库存函数 28；已覆盖 25；未覆盖 3。

```text
_FoldNormKernel.__call__
_FoldNormKernel.kernel
_hc_inject_norm_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/hc/qwen4_exp_hc_inject_rstd.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_hc_inject_rstd_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/hc/qwen4_exp_hc_lowrank_act.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_hc_lowrank_act_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/hc/qwen4_exp_hc_up_gate_mean.py`

库存函数 19；已覆盖 17；未覆盖 2。

```text
HcUpGateMeanKernel.__call__
HcUpGateMeanKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/hc/qwen4_exp_hc_up_gate_mean_rstd.py`

库存函数 20；已覆盖 18；未覆盖 2。

```text
HcUpGateMeanRstdKernel.__call__
HcUpGateMeanRstdKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/ple/ple_gate_conv_inject_norm.py`

库存函数 14；已覆盖 8；未覆盖 6。

```text
_injected_stream
_ple_conv_kernel
_ple_conv_rows_kernel
_ple_gate_kernel
_ple_residual_norm_kernel
_rms
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/ple/ple_ngram_embed.py`

库存函数 8；已覆盖 4；未覆盖 4。

```text
_ple_ngram_embed_kernel
_ple_ngram_embed_rows_kernel
_ple_ngram_publish_kernel
_seq_of_rows
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/quantization/bf16_to_fp32.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_bf16_to_fp32_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/ssm/causal_conv1d_update.py`

库存函数 22；已覆盖 19；未覆盖 3。

```text
_DecodeConvKernel.__call__
_DecodeConvKernel.kernel
_DecodeConvKernel.kernel.<locals>.elem
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/ssm/flashinfer_chunk_gated_delta_rule.py`

库存函数 10；已覆盖 8；未覆盖 2。

```text
_gather_carried_states_kernel
_zero_fresh_states_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/ssm/gdn_decode_gated_norm_pdl.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
_GdnKernel.__call__
_GdnKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/ssm/gdn_decode_rows_gated_norm.py`

库存函数 34；已覆盖 32；未覆盖 2。

```text
_GdnRowsKernel.__call__
_GdnRowsKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/ssm/gdn_prefill_conv_post.py`

库存函数 29；已覆盖 25；未覆盖 4。

```text
_ConvPassKernel.__call__
_ConvPassKernel.kernel
_gdn_gates_kernel
_gdn_prefill_conv_publish_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/fake_ops.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
_
_host_query.<locals>.fake
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/modeling.py`

库存函数 208；已覆盖 203；未覆盖 5。

```text
ModelingV2Qwen38FlashNextFp8Sm10xDefault.get_preferred_kv_cache_manager_version
PrefillTarget._attention_core_graph
PrefillTarget._attn_dispatch
PrefillTarget._gdn_core_graph
_select
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/draft_continue_rows__decode.py`

库存函数 11；已覆盖 8；未覆盖 3。

```text
_continue_select_kernel
_copy_row
_rows_gather_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qsa_fp8_pages_to_bf16__prefill.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_fp8_pages_to_bf16_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qsa_index_scores_cute__prefill.py`

库存函数 24；已覆盖 8；未覆盖 16。

```text
QsaIndexScoresPrefill.is_valid
QsaIndexScoresPrefill.reference
_IndexScoresKernel.__call__
_IndexScoresKernel.__call__.<locals>.make_mma
_IndexScoresKernel.copy_keys
_IndexScoresKernel.epilogue_role
_IndexScoresKernel.group_end
_IndexScoresKernel.group_starts
_IndexScoresKernel.group_tiles
_IndexScoresKernel.head_sum
_IndexScoresKernel.item_geometry
_IndexScoresKernel.item_meta
_IndexScoresKernel.kernel
_IndexScoresKernel.load_role
_IndexScoresKernel.mma_role
_IndexScoresKernel.tile_page
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qsa_index_scores_groups_cute__decode.py`

库存函数 11；已覆盖 6；未覆盖 5。

```text
QsaIndexScoresGroups.is_valid
QsaIndexScoresGroups.reference
_IdxGroupsKernel.__call__
_IdxGroupsKernel._issue
_IdxGroupsKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qsa_index_split_plan__decode.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
_split_plan_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qsa_paged_index_store.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
_qsa_paged_index_store_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qsa_sparse_gqa_grouped_cute__decode.py`

库存函数 70；已覆盖 46；未覆盖 24。

```text
_BalPlanKernel.__call__
_BalPlanKernel.kernel
_GroupedKernel.__call__
_GroupedKernel.__call__.<locals>.pv
_GroupedKernel.__call__.<locals>.qk
_GroupedKernel._bal_schedule
_GroupedKernel._hash_insert
_GroupedKernel._issue_pv
_GroupedKernel._pass_rows
_GroupedKernel._pass_tiles
_GroupedKernel._resolve
_GroupedKernel._union
_GroupedKernel._widen_row
_GroupedKernel.epilogue
_GroupedKernel.fixup
_GroupedKernel.issue_qk
_GroupedKernel.kernel
_GroupedKernel.load_tiles
_GroupedKernel.mma_tiles
_GroupedKernel.softmax_tiles
_GroupedKernel.stage_q
_GroupedKernel.widen_tiles
_RowMergeKernel.__call__
_RowMergeKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qsa_sparse_gqa_rowtile__prefill.py`

库存函数 39；已覆盖 29；未覆盖 10。

```text
_BlockPlanKernel.__call__
_BlockPlanKernel.kernel
_GroupPlanKernel.__call__
_GroupPlanKernel.__init__
_GroupPlanKernel.kernel
_group_plan
_red_or_shared
_rowtile_lone_kernel
_rowtile_lone_row
_word_pages
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qsa_sparse_gqa_splitk_cute__decode.py`

库存函数 49；已覆盖 42；未覆盖 7。

```text
_MergeKernel.__call__
_MergeKernel.kernel
_SplitKernel.__call__
_SplitKernel._epilogue
_SplitKernel._issue
_SplitKernel.kernel
_sigmoid
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qsa_spec_step_metadata__decode.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
_spec_step_metadata_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qsa_topk_expand_cluster__decode.py`

库存函数 17；已覆盖 12；未覆盖 5。

```text
_TopkExpandClusterKernel.__call__
_TopkExpandClusterKernel._exclusive_scan
_TopkExpandClusterKernel._narrow
_TopkExpandClusterKernel._pass
_TopkExpandClusterKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qsa_topk_rows_cute__prefill.py`

库存函数 15；已覆盖 5；未覆盖 10。

```text
QsaTopkRows.is_valid
QsaTopkRows.reference
_TopkRowsKernel.__call__
_TopkRowsKernel._compact
_TopkRowsKernel._excl_scan
_TopkRowsKernel._insert
_TopkRowsKernel._load
_TopkRowsKernel._load_full
_TopkRowsKernel._narrow
_TopkRowsKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qsa_triton_kernels__stable_jit.py`

库存函数 23；已覆盖 19；未覆盖 4。

```text
_qsa_kv_store_stable_kernel
_qsa_merge_stable_kernel
_qsa_prefill_pair_stable_kernel
_qsa_splitk_stable_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qwen4_exp_attn_decode_prologue__decode.py`

库存函数 12；已覆盖 7；未覆盖 5。

```text
_attn_decode_prologue_kernel
_compress_norm_rope
_norm_rope_apply
_norm_weights
_rope_factors
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/spec_rollback_begin__verify.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
_begin_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/vision_rope_2d_packed_qk.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_rotary_neox_inplace_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/comm/ce_exchange__sp_pair.py`

库存函数 12；已覆盖 11；未覆盖 1。

```text
CeExchange.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/comm/hc_allreduce_inject_norm__tp_sum.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
HcAllreduceInjectNorm.is_valid
HcAllreduceInjectNorm.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/comm/lamport_allgather__index_split.py`

库存函数 6；已覆盖 3；未覆盖 3。

```text
LamportAllgather.reference
_dst_words
_lamport_allgather_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/comm/mnnvl_allreduce__tp_sum.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
MnnvlTpSum.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/comm/moe_a2a__dp_dispatch_combine.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
MoeA2aCombine.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/embedding/embed_rows__decode.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_embed_rows_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/hc/qwen4_exp_hc_decode_rows_tc__decode.py`

库存函数 26；已覆盖 22；未覆盖 4。

```text
_First2.__call__
_First2.kernel
_Mix2.__call__
_Mix2.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/hc/qwen4_exp_hc_decode_tc_first__decode.py`

库存函数 19；已覆盖 16；未覆盖 3。

```text
HcDecodeTcFirstKernel.__call__
HcDecodeTcFirstKernel.kernel
_mapa
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/moe/blockwise_gemm_grouped.py`

库存函数 20；已覆盖 13；未覆盖 7。

```text
BlockwiseGemmGrouped.reference
BlockwiseGemmGroupedQuant.reference
Sm100BlockwiseGemmKernel.__call__
Sm100BlockwiseGemmKernel.kernel
Sm100BlockwiseGemmKernel.wrapper_grouped
Sm100BlockwiseGemmKernel.wrapper_grouped_quant
_grouped_reference
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/moe/moe_front_router_quant__decode.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
_moe_front_kernel
_scales
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/moe/moe_prefill_grouped__prefill.py`

库存函数 22；已覆盖 13；未覆盖 9。

```text
MoePrefillGrouped.reference
_act_kernel
_act_q_kernel
_count_kernel
_finalize_kernel
_finalize_pipe_kernel
_layout_kernel
_permute_kernel
_requant
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/moe/mtp_moe_combine_inject__decode.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
_combine_inject_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/ssm/causal_conv1d_token_major__verify.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
_verify_conv_seqs_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/ssm/flashinfer_chunk_gated_delta_rule.py`

库存函数 19；已覆盖 17；未覆盖 2。

```text
_gather_carried_states_kernel
_zero_fresh_states_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/ssm/gdn_cached_replay_cute__verify.py`

库存函数 92；已覆盖 83；未覆盖 9。

```text
_Replay.__call__
_Replay.kernel
_bits_f32
_bulk_g2s
_f32_bits
_ldg_bf16p
_ldg_f32
_mbar_arrive
_stg_b16
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/ssm/gdn_in_proj_conv__verify.py`

库存函数 20；已覆盖 18；未覆盖 2。

```text
_InProjConvKernel.__call__
_InProjConvKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/ssm/gdn_inforward_tail.py`

库存函数 9；已覆盖 5；未覆盖 4。

```text
GdnInforwardConvSwap.reference
GdnInforwardTail.reference
_conv_swap_kernel
_gdn_tail_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/vision/triton_pos_embed_interpolate.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
_bilinear_pos_embed_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/weights.py`

库存函数 85；已覆盖 83；未覆盖 2。

```text
_close_ngram_table
memory_kind
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_10x__default/modeling.py`

库存函数 160；已覆盖 159；未覆盖 1。

```text
ModelingV2Qwen38FlashNextNvfp4Sm10xDefault.get_preferred_kv_cache_manager_version
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_10x__default/ops/attention/qsa_fp8_pages_to_bf16__prefill.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_fp8_pages_to_bf16_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_10x__default/ops/attention/qsa_sparse_gqa_rowtile__prefill.py`

库存函数 16；已覆盖 9；未覆盖 7。

```text
_RowBitsKernel.__call__
_RowBitsKernel.kernel
_compress_bits
_or_combine
_rowtile_lone_kernel
_rowtile_lone_row
_rowtile_plan_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_10x__default/ops/attention/qwen4_exp_attn_decode_prologue__decode.py`

库存函数 12；已覆盖 7；未覆盖 5。

```text
_attn_decode_prologue_kernel
_compress_norm_rope
_norm_rope_apply
_norm_weights
_rope_factors
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_10x__default/ops/attention/vision_rope_2d_packed_qk.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
_rotary_neox_inplace_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_10x__default/ops/hc/qwen4_exp_hc_decode_tc_first__decode.py`

库存函数 14；已覆盖 12；未覆盖 2。

```text
HcDecodeTcFirstKernel.__call__
HcDecodeTcFirstKernel.kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_10x__default/ops/ssm/gdn_cached_replay_cute__verify.py`

库存函数 70；已覆盖 67；未覆盖 3。

```text
_Replay.__call__
_Replay.kernel
_bulk_g2s
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_10x__default/ops/vision/triton_pos_embed_interpolate.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
_bilinear_pos_embed_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_10x__default/weights.py`

库存函数 92；已覆盖 90；未覆盖 2。

```text
_close_ngram_table
memory_kind
```

### `tensorrt_llm/_torch/models/__init__.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
__dir__
```

### `tensorrt_llm/_torch/models/checkpoints/base_checkpoint_loader.py`

库存函数 13；已覆盖 5；未覆盖 8。

```text
BaseCheckpointLoader.build_checkpoint_catalog
BaseCheckpointLoader.checkpoint_format
BaseCheckpointLoader.cleanup
BaseCheckpointLoader.config_loader
BaseCheckpointLoader.get_default_config_loader
BaseCheckpointLoader.get_default_weight_loader
BaseCheckpointLoader.weight_loader
BaseCheckpointLoader.weight_mapper
```

### `tensorrt_llm/_torch/models/checkpoints/base_config_loader.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
BaseConfigLoader.load
```

### `tensorrt_llm/_torch/models/checkpoints/base_weight_loader.py`

库存函数 20；已覆盖 18；未覆盖 2。

```text
BaseWeightLoader.load_weights
ConsumableWeightsDict.values
```

### `tensorrt_llm/_torch/models/checkpoints/base_weight_mapper.py`

库存函数 23；已覆盖 15；未覆盖 8。

```text
BaseWeightMapper.abort_update_weights
BaseWeightMapper.apply_callbacks
BaseWeightMapper.begin_update_weights
BaseWeightMapper.finalize_update_weights
BaseWeightMapper.handle_special_instance_module
BaseWeightMapper.map_mtp_module_name
BaseWeightMapper.map_weights
BaseWeightMapper.preprocess_weights
```

### `tensorrt_llm/_torch/models/checkpoints/hf/exaone_moe_weight_mapper.py`

库存函数 6；已覆盖 3；未覆盖 3。

```text
ExaoneMoeWeightMapper._renames_scales_to_inv
ExaoneMoeWeightMapper.preprocess_weights
ExaoneMoeWeightMapper.preprocess_weights.<locals>.rename
```

### `tensorrt_llm/_torch/models/checkpoints/hf/musespark_open_weight_mapper.py`

库存函数 27；已覆盖 18；未覆盖 9。

```text
WeightAccounting.summary
_local_expert_range
_slice_experts
_unaccounted_keys
build_expert_weights
load_layer
load_musespark_open_audio_weights
load_musespark_open_omni_weights
load_musespark_open_weights
```

### `tensorrt_llm/_torch/models/checkpoints/hf/weight_loader.py`

库存函数 43；已覆盖 42；未覆盖 1。

```text
_LazySafetensorsWeights.clear
```

### `tensorrt_llm/_torch/models/checkpoints/memory_init.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
_allocate_buffer_on_cuda
```

### `tensorrt_llm/_torch/models/checkpoints/mistral/checkpoint_loader.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
MistralCheckpointLoader.get_default_config_loader
```

### `tensorrt_llm/_torch/models/checkpoints/mistral/config_loader.py`

库存函数 12；已覆盖 11；未覆盖 1。

```text
_remap_mistral_audio_args
```

### `tensorrt_llm/_torch/models/checkpoints/mistral/tokenizer.py`

库存函数 39；已覆盖 29；未覆盖 10。

```text
MistralTokenizer.all_special_ids
MistralTokenizer.all_special_tokens_extended
MistralTokenizer.batch_encode_plus
MistralTokenizer.clean_up_tokenization
MistralTokenizer.get_added_vocab
MistralTokenizer.get_chat_template
MistralTokenizer.hf_decode_incrementally
MistralTokenizer.is_fast
MistralTokenizer.pad_token
MistralTokenizer.sep_token
```

### `tensorrt_llm/_torch/models/deepseek_v41_processor.py`

库存函数 26；已覆盖 23；未覆盖 3。

```text
DeepseekV41InputProcessor.dtype
DeepseekV41InputProcessor.get_text_with_mm_placeholders
DeepseekV41InputProcessor.model_path
```

### `tensorrt_llm/_torch/models/deepseekv41_decoder_graph.py`

库存函数 22；已覆盖 21；未覆盖 1。

```text
_PassGraph._static_inputs
```

### `tensorrt_llm/_torch/models/hf_parameter_utils.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
get_parameter_dtype
```

### `tensorrt_llm/_torch/models/modeling_bart.py`

库存函数 36；已覆盖 33；未覆盖 3。

```text
_bart_decoder_hidden_size
_bart_encoder_hidden_size
_bart_head_dim
```

### `tensorrt_llm/_torch/models/modeling_clip.py`

库存函数 13；已覆盖 5；未覆盖 8。

```text
CLIPVisionModel.__init__
CLIPVisionModel.device
CLIPVisionModel.dtype
CLIPVisionModel.forward
CLIPVisionModel.load_weights
CLIPVisionModel.prepare_attn_metadata
CLIPVisionTransformer.__init__
CLIPVisionTransformer.forward
```

### `tensorrt_llm/_torch/models/modeling_cosmos3.py`

库存函数 5；已覆盖 2；未覆盖 3。

```text
Cosmos3Model.__init__
Cosmos3Model.llm_checkpoint_dir
Cosmos3Model.load_weights
```

### `tensorrt_llm/_torch/models/modeling_cosmos3_edge.py`

库存函数 49；已覆盖 44；未覆盖 5。

```text
Cosmos3EdgeInputProcessor.get_preferred_media_io_kwargs
Cosmos3EdgeSiglip.device
Cosmos3EdgeSiglip.get_encoder_attention_metadata_capacity
Cosmos3EdgeVisionEncoder.llm_checkpoint_dir
ceil_by_factor
```

### `tensorrt_llm/_torch/models/modeling_deepseekv3.py`

库存函数 59；已覆盖 54；未覆盖 5。

```text
DeepseekV3Gate.get_experts_per_token
DeepseekV3MTPHead.get_last_token_states
DeepseekV3WeightLoader.load_weights.<locals>.requantize_weight_with_new_scale
moe_reduce_add_shared_output
weight_dequant_kernel
```

### `tensorrt_llm/_torch/models/modeling_deepseekv4.py`

库存函数 113；已覆盖 92；未覆盖 21。

```text
DeepseekV4DecoderLayer._compute_mlp_tp_size
DeepseekV4DecoderLayer._make_engram
DeepseekV4ForCausalLM.register_cuda_graph_pre_replay_hooks.<locals>._refresh
DeepseekV4Gate.apply
DeepseekV4Gate.get_experts_per_token
DeepseekV4Linear.__init__
DeepseekV4Linear.apply_linear
DeepseekV4MoE.forward.<locals>._overlap_shared
DeepseekV4Model.__pp_init__
DeepseekV4Model._compute_engram_hashes
DeepseekV4Model._engram_after_attention_callback
DeepseekV4Model._engram_precompute_start_layer
DeepseekV4Model._enter_bounded_replay
DeepseekV4Model._enter_remote_tail_boundary
DeepseekV4Model._exit_bounded_replay
DeepseekV4Model._precompute_engram
DeepseekV4Model._precompute_engram_kv
DeepseekV4WeightLoader._load_weights_impl.<locals>.load_kv_b_proj_and_k_b_proj_trans_dequant
DeepseekV4WeightLoader._load_weights_impl.<locals>.requantize_weight_with_new_scale
moe_reduce_add_shared_output
weight_dequant_kernel
```

### `tensorrt_llm/_torch/models/modeling_deepseekv41.py`

库存函数 145；已覆盖 138；未覆盖 7。

```text
DeepseekV41DecoderLayer._moe_supports_unfinalized
DeepseekV41Engram._load_from_state_dict
DeepseekV41ForCausalLM.prepare_breakable_cuda_graph_replay
DeepseekV41Model._enter_remote_tail_boundary.<locals>.prepare_destination
DeepseekV41Model._precompute_engram_exchange
_candidate_inert_up_to_positions
_candidate_source_ratio
```

### `tensorrt_llm/_torch/models/modeling_deepseekv41_multimodal.py`

库存函数 27；已覆盖 24；未覆盖 3。

```text
DeepseekV41ForConditionalGeneration.get_preferred_kv_cache_manager_version
DeepseekV41ForConditionalGeneration.get_preferred_transceiver_runtime
DeepseekV41ForConditionalGeneration.mm_token_ids
```

### `tensorrt_llm/_torch/models/modeling_dflash.py`

库存函数 62；已覆盖 61；未覆盖 1。

```text
DFlashForCausalLM.reads_helix_sharded_ctx_pages
```

### `tensorrt_llm/_torch/models/modeling_dspark.py`

库存函数 158；已覆盖 146；未覆盖 12。

```text
DSv4DSparkDraftModel.forward_embed
DSv4DSparkForCausalLM.embed_tokens
DSv4DSparkForCausalLM.forward
K3DsparkDecoderLayer.forward
K3DsparkMLA.forward
K3DsparkModel.forward
MLADSparkForCausalLM._init_rope
MiniMaxDSparkAttention.__init__
MiniMaxDSparkBackbone.__init__
MiniMaxDSparkDecoderLayer.forward
MiniMaxDSparkForCausalLM.load_weights
MiniMaxDSparkModel.forward
```

### `tensorrt_llm/_torch/models/modeling_exaone4_5.py`

库存函数 10；已覆盖 7；未覆盖 3。

```text
Exaone4_5InputProcessor.get_mrope_config
Exaone4_5_ForConditionalGeneration.infer_max_seq_len
Exaone4_5_ForConditionalGeneration.load_weights
```

### `tensorrt_llm/_torch/models/modeling_exaone_moe.py`

库存函数 23；已覆盖 19；未覆盖 4。

```text
ExaoneMoeMTP.__init__
ExaoneMoeMTP.forward
ExaoneMoeMTP.forward.<locals>.norm_embeds
ExaoneMoeMTP.forward.<locals>.norm_hidden
```

### `tensorrt_llm/_torch/models/modeling_gemma3.py`

库存函数 22；已覆盖 21；未覆盖 1。

```text
_write_mask
```

### `tensorrt_llm/_torch/models/modeling_gemma3vl.py`

库存函数 23；已覆盖 3；未覆盖 20。

```text
Gemma3InputProcessor.__init__
Gemma3InputProcessor._preprocess
Gemma3InputProcessor.call_with_text_prompt
Gemma3InputProcessor.config
Gemma3InputProcessor.dtype
Gemma3InputProcessor.model_path
Gemma3InputProcessor.processor
Gemma3InputProcessor.tokenizer
Gemma3MultiModalProjector.__init__
Gemma3MultiModalProjector.forward
Gemma3MultiModalProjector.load_weights
Gemma3VLM.__init__
Gemma3VLM._get_image_features
Gemma3VLM.forward
Gemma3VLM.get_sub_model_config
Gemma3VLM.infer_max_seq_len
Gemma3VLM.load_weights
Gemma3VLM.mm_token_ids
Gemma3VLM.post_config
Gemma3VLM.vocab_size_padded
```

### `tensorrt_llm/_torch/models/modeling_gemma4_audio.py`

库存函数 30；已覆盖 12；未覆盖 18。

```text
Gemma4AudioAttention._convert_to_block
Gemma4AudioAttention._extract_block_context
Gemma4AudioAttention._rel_shift
Gemma4AudioAttention.forward
Gemma4AudioCausalConv1d.forward
Gemma4AudioCausalConv1d.left_pad
Gemma4AudioClippableLinear.forward
Gemma4AudioFeedForward.forward
Gemma4AudioLayer.forward
Gemma4AudioLightConv1d.forward
Gemma4AudioModel._build_chunked_local_mask
Gemma4AudioModel._convert_4d_mask_to_blocked_5d
Gemma4AudioModel.forward
Gemma4AudioModel.max_future_horizon
Gemma4AudioModel.max_past_horizon
Gemma4AudioRelPositionalEncoding.forward
Gemma4AudioSubSampleConvProjection.forward
Gemma4AudioSubSampleConvProjectionLayer.forward
```

### `tensorrt_llm/_torch/models/modeling_gemma4_unified.py`

库存函数 35；已覆盖 25；未覆盖 10。

```text
Gemma4UnifiedForConditionalGeneration._get_audio_features
Gemma4UnifiedForConditionalGeneration._get_image_features
Gemma4UnifiedInputProcessor.__init__
Gemma4UnifiedProcessorShim.replace_audio_token
Gemma4UnifiedVideoProcessor.aspect_ratio_preserving_resize
Gemma4UnifiedVisionEmbedder.__init__
Gemma4UnifiedVisionEmbedder.forward
Gemma4UnifiedVisionEmbedder.load_weights
load_gemma4_unified_processor
load_gemma4_unified_processor.<locals>._component_kwargs
```

### `tensorrt_llm/_torch/models/modeling_gemma4mm.py`

库存函数 56；已覆盖 55；未覆盖 1。

```text
Gemma4InputProcessor.get_num_tokens_per_audio
```

### `tensorrt_llm/_torch/models/modeling_glm.py`

库存函数 29；已覆盖 28；未覆盖 1。

```text
Glm4MoE._get_experts_quant_config
```

### `tensorrt_llm/_torch/models/modeling_glm5_next.py`

库存函数 73；已覆盖 71；未覆盖 2。

```text
Glm5NextForCausalLM.num_hidden_layers
Glm5NextMTPHead.get_last_token_states
```

### `tensorrt_llm/_torch/models/modeling_glm5_next_vision.py`

库存函数 71；已覆盖 53；未覆盖 18。

```text
Glm5NextInputProcessor.config
Glm5NextInputProcessor.dtype
Glm5NextInputProcessor.get_mm_token_ids
Glm5NextInputProcessor.get_num_tokens_per_video
Glm5NextInputProcessor.get_preferred_media_io_kwargs
Glm5NextInputProcessor.get_vocab_size
Glm5NextInputProcessor.model_path
Glm5NextVLM.embedding_dim
Glm5NextVLM.embedding_dtype
Glm5NextVLM.encode_multimodal_inputs
Glm5NextVLM.load_draft_weights
Glm5NextVLM.multimodal_data_device_paths
Glm5NextVLM.multimodal_token_ids
Glm5NextVisionModelBase.enable_blocks_cuda_graph
Glm5NextVisionModelBase.forward
Glm5NextVisionModelBase.load_weights
Glm5NextVisionModelBase.mm_encoder_groups
_flatten_video_grid_thw
```

### `tensorrt_llm/_torch/models/modeling_gpt_oss.py`

库存函数 25；已覆盖 22；未覆盖 3。

```text
GptOssForCausalLM.load_nvfp4_weights
GptOssForCausalLM.load_nvfp4_weights.<locals>.deinterleave
MLPBlock.swiglu
```

### `tensorrt_llm/_torch/models/modeling_hunyuan_moe.py`

库存函数 16；已覆盖 1；未覆盖 15。

```text
HunYuanAttention.__init__
HunYuanAttention.apply_qk_norm
HunYuanAttention.apply_qk_norm.<locals>.k_l2norm
HunYuanAttention.apply_qk_norm.<locals>.q_l2norm
HunYuanAttention.apply_rope
HunYuanAttention.forward
HunYuanDecoderLayer.__init__
HunYuanDecoderLayer.forward
HunYuanMoEV1ForCausalLM.__init__
HunYuanMoEV1ForCausalLM.forward
HunYuanMoEV1ForCausalLM.load_weights
HunYuanMoEV1ForCausalLM.load_weights.<locals>.filter_weights
HunYuanModel.__init__
HunYuanModel.forward
HunyuanMoE.forward
```

### `tensorrt_llm/_torch/models/modeling_kimi_k25.py`

库存函数 61；已覆盖 31；未覆盖 30。

```text
EncoderLayer.forward
KimiK25ForConditionalGeneration.load_draft_weights
KimiK25ForConditionalGeneration.multimodal_data_device_paths
KimiK25ForConditionalGeneration.set_guided_decoder
KimiK25ForConditionalGeneration.vocab_size_padded
KimiK25InputProcessor.config
KimiK25InputProcessor.dtype
KimiK25InputProcessor.get_mm_token_ids
KimiK25InputProcessor.get_num_tokens_per_image
KimiK25InputProcessor.get_num_tokens_per_video
KimiK25InputProcessor.get_vocab_size
KimiK25InputProcessor.model_path
KimiK25InputProcessor.processor
KimiK25VisionAttention.forward
KimiK25VisionModel._extract_features
KimiK25VisionModel.forward
Learnable2DPosEmb.forward
MoonViT3dEncoder.forward
MoonViT3dEncoder.prepare_attn_metadata
PatchEmbed3d.forward
PatchMergerMLP.forward
Rope2D._precompute
Rope2D.get_freqs_cis
_apply_rope_2d
_decode_video_to_chunks
_format_video_timestamp
_frames_to_chunks
_gelu_tanh
_get_rope_shape_impl
_tpool_patch_merger
```

### `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py`

库存函数 11；已覆盖 8；未覆盖 3。

```text
K3EncoderLayer.forward
K3PatchEmbed3d.forward
PatchMergerMLPV2.forward
```

### `tensorrt_llm/_torch/models/modeling_kimi_linear.py`

库存函数 131；已覆盖 121；未覆盖 10。

```text
KimiK3HelixA2A.__init__
KimiK3HelixA2A.all_gather
KimiK3HelixA2A.reduce_scatter
KimiLinearDecoderLayer.skip_forward
KimiLinearForCausalLM._load_expert_slices.<locals>.load_split_file_expert.<locals>.get_optional_tensor
KimiLinearModel.prewarm_decode_kernels.<locals>.need
_Fp8BlockScaleWeightReadLinear.has_fp8_qdq
_Fp8BlockScaleWeightReadLinear.has_w4a8_nvfp4_fp8
_Fp8BlockScaleWeightReadLinear.supports_prequantized_input
_multimem_allgather
```

### `tensorrt_llm/_torch/models/modeling_lilicorr.py`

库存函数 21；已覆盖 20；未覆盖 1。

```text
LiLiCorrForCausalLM.select_lilicorr_path
```

### `tensorrt_llm/_torch/models/modeling_llama.py`

库存函数 49；已覆盖 39；未覆盖 10。

```text
Llama4Attention.apply_qk_norm
Llama4Attention.apply_qk_norm.<locals>.k_l2norm
Llama4Attention.apply_qk_norm.<locals>.q_l2norm
Llama4InputProcessor._attach_multimodal_embeddings_impl
Llama4InputProcessor.dtype
Llama4InputProcessor.get_mm_token_ids
Llama4InputProcessor.model_path
Llama4VisionEncoder.forward
MistralForCausalLM.__init__
load_sharded_checkpoint
```

### `tensorrt_llm/_torch/models/modeling_llama_min_latency.py`

库存函数 19；已覆盖 2；未覆盖 17。

```text
Llama4MinLatencyAttention.__init__
Llama4MinLatencyAttention._forward_nope
Llama4MinLatencyDecoderLayer.__init__
Llama4MinLatencyDecoderLayer.forward
Llama4MinLatencyFusedMoE.forward
Llama4MinLatencyGatedMLP.__init__
Llama4MinLatencyGatedMLP.cache_derived_state
Llama4MinLatencyGatedMLP.forward
Llama4MinLatencyGatedMLP.post_load_weights
Llama4MinLatencyLinear.__init__
Llama4MinLatencyLinear.apply_linear
Llama4MinLatencyLinear.load_weights
Llama4MinLatencyLinear.set_position_ids
Llama4MinLatencyMoE.cache_derived_state
Llama4MinLatencyMoE.compute_routed_output
Llama4MinLatencyMoE.forward
Llama4MinLatencyMoE.post_load_weights
```

### `tensorrt_llm/_torch/models/modeling_llava_next.py`

库存函数 29；已覆盖 6；未覆盖 23。

```text
LlavaNextInputProcessor.__init__
LlavaNextInputProcessor.call_with_text_prompt
LlavaNextInputProcessor.dtype
LlavaNextInputProcessor.expand_prompt_token_ids_for_mm
LlavaNextInputProcessor.get_mm_token_ids
LlavaNextInputProcessor.get_text_with_mm_placeholders
LlavaNextInputProcessor.model_path
LlavaNextInputProcessor.processor
LlavaNextModel.__init__
LlavaNextModel.forward
LlavaNextModel.infer_max_seq_len
LlavaNextModel.load_weights
LlavaNextModel.load_weights.<locals>.filter_weights
LlavaNextModel.mm_token_ids
LlavaNextModel.post_config
LlavaNextModel.vocab_size_padded
LlavaNextVisionModel.__init__
LlavaNextVisionModel._pad_for_batching
LlavaNextVisionModel.forward
LlavaNextVisionModel.load_weights
LlavaNextVisionModel.load_weights.<locals>.filter_weights
LlavaNextVisionModel.pack_image_features
LlavaNextVisionModel.post_config
```

### `tensorrt_llm/_torch/models/modeling_minicpmv4_6.py`

库存函数 50；已覆盖 32；未覆盖 18。

```text
MiniCPMV4_6InputProcessor.__init__
MiniCPMV4_6InputProcessor._mm_token_ids
MiniCPMV4_6InputProcessor._postprocess
MiniCPMV4_6InputProcessor._preprocess
MiniCPMV4_6InputProcessor.call_with_text_prompt
MiniCPMV4_6InputProcessor.config
MiniCPMV4_6InputProcessor.dtype
MiniCPMV4_6InputProcessor.get_num_tokens_per_image
MiniCPMV4_6InputProcessor.get_num_tokens_per_video
MiniCPMV4_6InputProcessor.get_vocab_size
MiniCPMV4_6InputProcessor.model_path
MiniCPMV4_6InputProcessor.processor
MiniCPMV4_6InputProcessor.tokenizer
MiniCPMV4_6Model.infer_max_seq_len
MiniCPMV4_6Model.load_weights
MiniCPMV4_6Model.multimodal_data_device_paths
MiniCPMV4_6Model.vocab_size_padded
MiniCPMV4_6VisionModel.load_weights
```

### `tensorrt_llm/_torch/models/modeling_minimaxm2.py`

库存函数 16；已覆盖 5；未覆盖 11。

```text
MiniMaxM2Attention.__init__
MiniMaxM2Attention.apply_qk_norm
MiniMaxM2Attention.apply_rope
MiniMaxM2DecoderLayer.__init__
MiniMaxM2ForCausalLM.__init__
MiniMaxM2Model.__init__
MiniMaxM2Model.forward
MiniMaxRMSNorm.__init__
MiniMaxRMSNorm.forward
MiniMaxRMSNorm.load_weights
_EScoreCorrectionBiasHolder.load_weights
```

### `tensorrt_llm/_torch/models/modeling_minimaxm3.py`

库存函数 85；已覆盖 75；未覆盖 10。

```text
MiniMaxM3Attention._sdpa_dense_attention_core
MiniMaxM3Attention._split_index_qk
MiniMaxM3Attention._triton_sparse_attention_core
MiniMaxM3Attention.apply_index_qk_norm
MiniMaxM3Attention.apply_index_qk_norm.<locals>._idx_k_norm
MiniMaxM3Attention.apply_index_qk_norm.<locals>._idx_q_norm
MiniMaxRMSNorm.__init__
MiniMaxRMSNorm.forward
MiniMaxRMSNorm.load_weights
_minimax_m3_swiglu_oai
```

### `tensorrt_llm/_torch/models/modeling_minimaxm3_vl.py`

库存函数 73；已覆盖 62；未覆盖 11。

```text
MiniMaxM3VLInputProcessor.config
MiniMaxM3VLInputProcessor.get_vocab_size
MiniMaxM3VLInputProcessor.model_path
MiniMaxM3VLInputProcessor.use_fast
_find_placeholder_runs
_flatten_grids
extract_multimodal_data_from_params
extract_multimodal_data_from_params.<locals>._cat
prepare_multimodal_inputs_embeds
prepare_multimodal_inputs_embeds.<locals>._run_tower
prepare_multimodal_inputs_embeds.<locals>._splice
```

### `tensorrt_llm/_torch/models/modeling_mistral.py`

库存函数 84；已覆盖 79；未覆盖 5。

```text
Mistral3MultiModalProjector.load_weights
Mistral3PatchMerger.load_weights
Mistral3VLM.mm_token_ids
MistralHFInputProcessor.model_path
MistralNativeInputProcessor.requires_input_placeholders
```

### `tensorrt_llm/_torch/models/modeling_mixtral.py`

库存函数 11；已覆盖 2；未覆盖 9。

```text
MixtralAttention.__init__
MixtralAttention.forward
MixtralDecoderLayer.__init__
MixtralDecoderLayer.forward
MixtralForCausalLM.__init__
MixtralForCausalLM.load_weights
MixtralModel.__init__
MixtralModel.forward
_unfuse_mixtral_moe_weights
```

### `tensorrt_llm/_torch/models/modeling_multimodal_encoder.py`

库存函数 23；已覆盖 8；未覆盖 15。

```text
MultimodalEncoderMixin._build_blocks_graph_runner
VisionTower.__init__
VisionTower.config
VisionTower.device
VisionTower.dtype
VisionTower.feature_select
VisionTower.forward
VisionTower.hidden_size
VisionTower.num_patches
VisionTowerDynamicS2.__init__
VisionTowerDynamicS2.forward
VisionTowerS2.__init__
VisionTowerS2.forward
VisionTowerS2.forward_feature
VisionTowerS2.hidden_size
```

### `tensorrt_llm/_torch/models/modeling_multimodal_mixin.py`

库存函数 55；已覆盖 51；未覆盖 4。

```text
MultimodalModelMixin.embedding_dtype
MultimodalModelMixin.encode_multimodal_inputs
MultimodalModelMixin.get_language_model_extra_forward_kwargs
MultimodalModelMixin.text_embedding_layer
```

### `tensorrt_llm/_torch/models/modeling_multimodal_utils.py`

库存函数 35；已覆盖 19；未覆盖 16。

```text
_lookup_vision_pixels_kernel
dynamic_preprocess
dynamic_preprocess_dispatch
dynamic_preprocess_torch
dynamic_s2_preprocess
dynamic_s2_preprocess_dispatch
dynamic_s2_preprocess_torch
find_closest_aspect_ratio
merge_chessboard
merge_features_for_dynamic_s2
multiscale_forward
preprocess_dispatch
s2_batched_forward
s2_merge_chessboard
s2_split_chessboard
split_chessboard
```

### `tensorrt_llm/_torch/models/modeling_musespark_open.py`

库存函数 58；已覆盖 50；未覆盖 8。

```text
MuseSparkOpenAttention.__init__
MuseSparkOpenAttention.apply_qk_norm
MuseSparkOpenAttention.forward
MuseSparkOpenDecoderLayer.__init__
MuseSparkOpenForCausalLM.__init__
MuseSparkOpenForCausalLM.infer_max_seq_len
MuseSparkOpenForCausalLM.load_weights
MuseSparkOpenModel.__init__
```

### `tensorrt_llm/_torch/models/modeling_musespark_open_omni.py`

库存函数 54；已覆盖 44；未覆盖 10。

```text
MuseSparkForConditionalGeneration.audio_encoder
MuseSparkForConditionalGeneration.draft_config
MuseSparkForConditionalGeneration.draft_model
MuseSparkForConditionalGeneration.encode_multimodal_inputs
MuseSparkForConditionalGeneration.load_draft_weights
MuseSparkForConditionalGeneration.multimodal_data_device_paths
MuseSparkForConditionalGeneration.spec_worker
MuseSparkForConditionalGeneration.use_separate_draft_kv_cache
MuseSparkForConditionalGeneration.vision_tower
MuseSparkOpenVisionEncoder.__init__
```

### `tensorrt_llm/_torch/models/modeling_nemotron_h.py`

库存函数 32；已覆盖 31；未覆盖 1。

```text
NemotronHForCausalLM._normalize_puzzle_config.<locals>._is_moe
```

### `tensorrt_llm/_torch/models/modeling_nemotron_h_multimodal.py`

库存函数 107；已覆盖 103；未覆盖 4。

```text
NemotronHMultimodalInputProcessor.model_path
NemotronHMultimodalModel.encode_multimodal_inputs
NemotronHVisionEncoder.apply_evs_per_video
NemotronHVisionEncoder.enable_radio_cuda_graph
```

### `tensorrt_llm/_torch/models/modeling_nemotron_nas.py`

库存函数 13；已覆盖 8；未覆盖 5。

```text
LinearAttention.__init__
LinearAttention.forward
LinearMLP.__init__
LinearMLP.forward
_create_linear_from_configs
```

### `tensorrt_llm/_torch/models/modeling_parakeet.py`

库存函数 39；已覆盖 38；未覆盖 1。

```text
ParakeetExtractor.audio_length
```

### `tensorrt_llm/_torch/models/modeling_phi4mm.py`

库存函数 34；已覆盖 2；未覆盖 32。

```text
HFPhi4MultimodalEncoder.__init__
HFPhi4MultimodalEncoder._batch_infer_audio_embeds
HFPhi4MultimodalEncoder._batch_infer_image_embeds
HFPhi4MultimodalEncoder._encoding_batch_request
HFPhi4MultimodalEncoder.forward
NoOp.forward
Phi4MMForCausalLM.__init__
Phi4MMForCausalLM.forward
Phi4MMForCausalLM.infer_max_seq_len
Phi4MMForCausalLM.load_weights
Phi4MMForCausalLM.lora_config
Phi4MMForCausalLM.lora_request
Phi4MMForCausalLM.mm_token_ids
Phi4MMForCausalLM.multimodal_data_device_paths
Phi4MMForCausalLM.post_config
Phi4MMForCausalLM.vocab_size_padded
Phi4MMInputProcessor.__init__
Phi4MMInputProcessor._post_process
Phi4MMInputProcessor.call_with_text_prompt
Phi4MMInputProcessor.config
Phi4MMInputProcessor.dtype
Phi4MMInputProcessor.get_mm_token_ids
Phi4MMInputProcessor.get_num_tokens_per_image
Phi4MMInputProcessor.model_path
Phi4MMInputProcessor.processor
Phi4MMInputProcessor.tokenizer
_load_phi4mm_classes
_reshape_attention_masks
dynamic_preprocess
image_preprocess
siglip_embedding_forward
vision_encoder_forward
```

### `tensorrt_llm/_torch/models/modeling_pplx_decider.py`

库存函数 8；已覆盖 5；未覆盖 3。

```text
PplxDeciderForDecision.cache_derived_state
PplxDeciderForDecision.setup_aliases
PplxDeciderGatedDeltaNet.forward
```

### `tensorrt_llm/_torch/models/modeling_qwen.py`

库存函数 12；已覆盖 7；未覆盖 5。

```text
Qwen2ForProcessRewardModel.__init__
Qwen2ForProcessRewardModel.forward
Qwen2ForRewardModel.__init__
Qwen2ForRewardModel.forward
QwenYarnAttention.__init__
```

### `tensorrt_llm/_torch/models/modeling_qwen2vl.py`

库存函数 84；已覆盖 65；未覆盖 19。

```text
Qwen2VLInputProcessorBase._dummy_mm_data_for_size
Qwen2VLInputProcessorBase.get_mm_encoder_attention_metadata_capacity
Qwen2VLInputProcessorBase.model_path
Qwen2VLModel.__init__
Qwen2VLModel.load_weights
Qwen2VLModel.multimodal_data_device_paths
Qwen2VLModelBase.apply_llm_torch_compile
Qwen2VLModelBase.encode_multimodal_inputs
Qwen2VLModelBase.forward
Qwen2VLModelBase.get_multimodal_item_partitions
Qwen2VLModelBase.infer_max_seq_len
Qwen2VLModelBase.init_mrope_embedding
Qwen2VLModelBase.load_weights
Qwen2VLModelBase.mm_token_ids
Qwen2VLModelBase.prepare_mrope_config
Qwen2VLModelBase.vocab_size_padded
Qwen2_5_VLModel.__init__
Qwen2_5_VLModel.load_weights
Qwen2_5_VLModel.multimodal_data_device_paths
```

### `tensorrt_llm/_torch/models/modeling_qwen3_next.py`

库存函数 96；已覆盖 90；未覆盖 6。

```text
Qwen3NextMTPHead.get_last_token_states
Qwen3NextSparseMoeBlock._tune_fused_shared_expert
Qwen3NextSparseMoeBlock.forward.<locals>.shared_fn
_Qwen3NextDecoderLayerBase._attach_expert_prefetch
_Qwen3NextDecoderLayerBase.mixer_forward
_post_moe_norm_fold
```

### `tensorrt_llm/_torch/models/modeling_qwen3vl.py`

库存函数 71；已覆盖 70；未覆盖 1。

```text
_bilinear_pos_embed_kernel
```

### `tensorrt_llm/_torch/models/modeling_qwen4_exp.py`

库存函数 41；已覆盖 37；未覆盖 4。

```text
Qwen4ExpAttention.is_qsa
Qwen4ExpDecoderLayer.skip_forward
Qwen4ExpForConditionalGeneration.get_preferred_kv_cache_manager_version
Qwen4ExpModel._ensure_ple_pools
```

### `tensorrt_llm/_torch/models/modeling_qwen_moe.py`

库存函数 8；已覆盖 1；未覆盖 7。

```text
Qwen2MoeForCausalLM.__init__
QwenMoE.forward
QwenMoeAttention.__init__
QwenMoeDecoderLayer.__init__
QwenMoeDecoderLayer.forward
QwenMoeModel.__init__
QwenMoeModel.forward
```

### `tensorrt_llm/_torch/models/modeling_radio.py`

库存函数 51；已覆盖 45；未覆盖 6。

```text
RADIOVisionModelBase.blocks
RADIOVisionModelBase.embed_dim
RADIOVisionModelBase.get_nearest_supported_resolution
RADIOVisionModelBase.max_resolution
RADIOVisionModelBase.num_cls_tokens
RADIOVisionModelBase.preferred_resolution
```

### `tensorrt_llm/_torch/models/modeling_siglip.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
SiglipVisionModel.device
SiglipVisionModel.dtype
```

### `tensorrt_llm/_torch/models/modeling_speculative.py`

库存函数 90；已覆盖 86；未覆盖 4。

```text
Eagle3ForCausalLM.forward
MistralLarge3EagleForCausalLM.forward
PARDForCausalLM.forward
VanillaMarkov.apply_step_logits
```

### `tensorrt_llm/_torch/models/modeling_step3p7.py`

库存函数 62；已覆盖 58；未覆盖 4。

```text
Step3p7ForCausalLM.get_ignored_key_prefixes
Step3p7MoE._clamp_weight_device
Step3p7MoE._local_expert_ids
_nvfp4_stack_dequant
```

### `tensorrt_llm/_torch/models/modeling_step3p7vl.py`

库存函数 57；已覆盖 50；未覆盖 7。

```text
Step3p7VLForConditionalGeneration.epilogue
Step3p7VLForConditionalGeneration.multimodal_data_device_paths
Step3p7VLForConditionalGeneration.spec_worker
Step3p7VLForConditionalGeneration.vocab_size_padded
Step3p7VLInputProcessor.config
Step3p7VLInputProcessor.dtype
Step3p7VLInputProcessor.processor
```

### `tensorrt_llm/_torch/models/modeling_utils.py`

库存函数 87；已覆盖 85；未覆盖 2。

```text
DecoderModel.forward
_is_builtin_draft_model_builder
```

### `tensorrt_llm/_torch/models/modeling_whisper.py`

库存函数 42；已覆盖 41；未覆盖 1。

```text
WhisperInputProcessor.get_mm_special_token_ids
```

### `tensorrt_llm/_torch/models/multimodal_encoder_graph.py`

库存函数 27；已覆盖 22；未覆盖 5。

```text
EncoderGraphIOAdapter.allocate_inputs
EncoderGraphIOAdapter.collect_outputs
EncoderGraphIOAdapter.copy_inputs
EncoderMetadataProvider.build
EncoderMetadataProvider.refresh_in_place
```

### `tensorrt_llm/_torch/models/musespark_open_boundary.py`

库存函数 11；已覆盖 7；未覆盖 4。

```text
_ClusterBoundaryKernel.__call__
_ClusterBoundaryKernel.kernel
_ClusterBoundaryKernel.load
_ClusterBoundaryKernel.reduce
```

### `tensorrt_llm/_torch/models/musespark_open_finalize_ar.py`

库存函数 47；已覆盖 45；未覆盖 2。

```text
_FinalizeAllReduceKernel.__call__
_FinalizeAllReduceKernel.kernel
```

### `tensorrt_llm/_torch/models/musespark_open_kernels.py`

库存函数 23；已覆盖 15；未覆盖 8。

```text
_attention_norm_gate_kernel
_full_routing_kernel
_gated_residual_after_rmsnorm_kernel
_post_expert_norm_combine_kernel
_router_projection_kernel
_router_tensorcore_kernel
_router_tensorcore_reduce_kernel
_routing_weights_kernel
```

### `tensorrt_llm/_torch/models/musespark_open_vision.py`

库存函数 34；已覆盖 27；未覆盖 7。

```text
MuseSparkOpenNaViTVisionTower._max_seq_len_for
MuseSparkOpenNaViTVisionTower.attn_max_seq_len
MuseSparkOpenNaViTVisionTower.device
MuseSparkOpenNaViTVisionTower.dtype
MuseSparkOpenNaViTVisionTower.full_attention_layer_indices
MuseSparkOpenNaViTVisionTower.prepare_attn_metadata
vision_layer_is_full_attention
```

### `tensorrt_llm/_torch/modules/context_logits.py`

库存函数 11；已覆盖 10；未覆盖 1。

```text
ContextLogitsProjector.project
```

### `tensorrt_llm/_torch/modules/decoder_layer.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
DecoderLayer.forward
```

### `tensorrt_llm/_torch/modules/draft_experts_nvfp4.py`

库存函数 11；已覆盖 10；未覆盖 1。

```text
experts_unquantized
```

### `tensorrt_llm/_torch/modules/dwdp/dense/layout.py`

库存函数 27；已覆盖 24；未覆盖 3。

```text
_signed_64
checksum_bits
device_checksum
```

### `tensorrt_llm/_torch/modules/dwdp/dense/manager.py`

库存函数 26；已覆盖 21；未覆盖 5。

```text
ExpertPrefetcher.start_forward
StandaloneTransfer.copy_stream
StandaloneTransfer.mode
_free_group
new_copy_stream
```

### `tensorrt_llm/_torch/modules/dwdp/dense/plan.py`

库存函数 19；已覆盖 7；未覆盖 12。

```text
DenseCopyLauncher.__call__
DenseCopyLauncher.__init__
DenseCopyLauncher.__init__.<locals>.prepare
DenseCopyLauncher.mode
DenseCopyLauncher.release
_copy_views
_descriptor_views
_load_cudart
_memcpy_batch
batched_copies
build_layer_plan
gather_launch_args
```

### `tensorrt_llm/_torch/modules/dwdp/dense/repoint.py`

库存函数 21；已覆盖 18；未覆盖 3。

```text
_layer_ranges
_listing
_missing
```

### `tensorrt_llm/_torch/modules/dwdp/dense/runtime.py`

库存函数 47；已覆盖 43；未覆盖 4。

```text
DenseDWDP._phase_text
DenseDWDP.covers
DenseDWDP.local_bytes_outside_torch
DenseDWDP.segments
```

### `tensorrt_llm/_torch/modules/dwdp/dense/setup.py`

库存函数 40；已覆盖 33；未覆盖 7。

```text
DenseTransfer.copy_stream
DenseTransfer.use_batched_prefetch
DenseTransfer.use_gather
DenseTransfer.use_tma
_copy_launcher
_device_mem_get_info
_device_mem_get_info.<locals>.mem_get_info
```

### `tensorrt_llm/_torch/modules/dwdp/dense/store.py`

库存函数 37；已覆盖 28；未覆盖 9。

```text
DenseShardStore.allocate
DenseShardStore.chunk_bases
DenseShardStore.export
DenseShardStore.import_peers
DenseShardStore.local_bytes_outside_torch
DenseShardStore.release
DenseShardStore.write_local_rows
FabricShardStore.local_bytes_outside_torch
_describe_payload
```

### `tensorrt_llm/_torch/modules/dwdp/dense/types.py`

库存函数 66；已覆盖 51；未覆盖 15。

```text
DenseLayout.covers
DenseRootsProvider.dense_dwdp_roots
DeviceOps.current_stream
DeviceOps.do_multi_stream
DeviceOps.is_current_stream_capturing
DeviceOps.new_event
DeviceOps.nvtx_range
DeviceOps.stream
DeviceOps.synchronize
EventLike.record
LayerCopyPlan.is_empty
LayerCopyPlan.nbytes
SegmentLayout.key
StreamLike.wait_event
TensorBinding.owner
```

### `tensorrt_llm/_torch/modules/dwdp/page_pool.py`

库存函数 18；已覆盖 11；未覆盖 7。

```text
PagePool.__enter__
PagePool.__exit__
PagePool.device_id
PagePool.get_page_handle
PagePool.granularity
PagePool.slot_size
PagePool.unmap_pages
```

### `tensorrt_llm/_torch/modules/dwdp/rendezvous.py`

库存函数 14；已覆盖 11；未覆盖 3。

```text
GroupComm.Get_rank
GroupComm.Get_size
GroupComm.allgather
```

### `tensorrt_llm/_torch/modules/dwdp/specs.py`

库存函数 30；已覆盖 25；未覆盖 5。

```text
MnnvlHandleSet.layer_indices
MnnvlHandleSet.weight_names
PageAlignedLayout.mnnvl_pages
PageAlignedLayout.remote_pages
WeightSpec.chunk_bytes
```

### `tensorrt_llm/_torch/modules/dwdp/transport.py`

库存函数 22；已覆盖 14；未覆盖 8。

```text
DWDPTransport.__enter__
DWDPTransport.__exit__
DWDPTransport.device_id
DWDPTransport.dwdp_rank
DWDPTransport.dwdp_size
DWDPTransport.granularity
_pidfd_getfd
_pidfd_open
```

### `tensorrt_llm/_torch/modules/dwdp/vmm.py`

库存函数 35；已覆盖 16；未覆盖 19。

```text
VARegion.__del__
VARegion.__enter__
VARegion.__exit__
VARegion.__init__
VARegion.map
VARegion.release
VARegion.size
VARegion.unmap_all
VARegion.va
VMMHandle.__del__
VMMHandle.__enter__
VMMHandle.__exit__
VMMHandle.__init__
VMMHandle.device_id
VMMHandle.handle
VMMHandle.release
VMMHandle.size
_tensor_from_ptr_internal
align_down
```

### `tensorrt_llm/_torch/modules/dwdp/weight_buffer.py`

库存函数 31；已覆盖 22；未覆盖 9。

```text
WeightBuffer.__enter__
WeightBuffer.__exit__
WeightBuffer._create_tensor_view
WeightBuffer.debug_info
WeightBuffer.dwdp_size
WeightBuffer.get_data_offset
WeightBuffer.granularity
WeightBuffer.page_pool
WeightBuffer.pool_page_size
```

### `tensorrt_llm/_torch/modules/dwdp/weight_manager.py`

库存函数 26；已覆盖 20；未覆盖 6。

```text
DWDPWeightManager.dwdp_rank
DWDPWeightManager.dwdp_size
DWDPWeightManager.is_moe_layer
DWDPWeightManager.local_bytes_outside_torch
DWDPWeightManager.use_batched_prefetch
DWDPWeightManager.weight_buffer
```

### `tensorrt_llm/_torch/modules/embedding.py`

库存函数 21；已覆盖 20；未覆盖 1。

```text
_embedding_gather_kernel
```

### `tensorrt_llm/_torch/modules/engram/_node_share.py`

库存函数 13；已覆盖 11；未覆盖 2。

```text
NodeShare.close
_forget_in_child
```

### `tensorrt_llm/_torch/modules/engram/adp_exchange.py`

库存函数 17；已覆盖 10；未覆盖 7。

```text
_bf16x2
_e2m1x4_to_e4m3x4
_e4m3x2_to_f32
_hash_exchange_kernel
_mul_rn
_owner_push_kernel
_receive_kernel
```

### `tensorrt_llm/_torch/modules/engram/engram.py`

库存函数 69；已覆盖 52；未覆盖 17。

```text
CompressedTokenizer.__call__
CompressedTokenizer._compress
Engram._make_kv_projection
Engram._make_multi_head_embedding
Engram._make_short_conv
Engram.forward
Engram.precompute
MultiHeadEmbedding.__init__
MultiHeadEmbedding.forward
NgramHashMapping._get_ngram_hashes
NgramHashMapping._get_ngram_hashes.<locals>.shift_k
NgramHashMapping.hash
NgramHashMapping.hash_single_layer
ShardedFp8MultiHeadEmbedding._release_shared
ShortConv.__init__
ShortConv.forward
ShortConv.load_weights
```

### `tensorrt_llm/_torch/modules/engram/functional.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
_engram_gate_kernel
_history_scatter_kernel
```

### `tensorrt_llm/_torch/modules/expert_l2_prefetch.py`

库存函数 16；已覆盖 12；未覆盖 4。

```text
_globaltimer
_issue
_prefetch_kernel
_record_kernel
```

### `tensorrt_llm/_torch/modules/fla/cache_manager.py`

库存函数 85；已覆盖 81；未覆盖 4。

```text
GDNMtpReplayState._allocate_history
GDNMtpReplayState._history_buffers
GDNMtpReplayState.commit_all_layers
GDNReplayState.uses_replay
```

### `tensorrt_llm/_torch/modules/fla/cached_replay.py`

库存函数 10；已覆盖 7；未覆盖 3。

```text
_cached_replay_kernel
_cached_replay_layered_commit_kernel
_gdc_wait_with_memory_clobber
```

### `tensorrt_llm/_torch/modules/fla/chunk_delta_h.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
chunk_gated_delta_rule_fwd_kernel_h_blockdim64
```

### `tensorrt_llm/_torch/modules/fla/chunk_o.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
chunk_fwd_kernel_o
```

### `tensorrt_llm/_torch/modules/fla/chunk_scaled_dot_kkt.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
chunk_scaled_dot_kkt_fwd_kernel
```

### `tensorrt_llm/_torch/modules/fla/cumsum.py`

库存函数 6；已覆盖 2；未覆盖 4。

```text
chunk_local_cumsum_scalar_kernel
chunk_local_cumsum_vector
chunk_local_cumsum_vector.<locals>.grid
chunk_local_cumsum_vector_kernel
```

### `tensorrt_llm/_torch/modules/fla/flashinfer_chunk.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
indexed_state_io_available
```

### `tensorrt_llm/_torch/modules/fla/fused_recurrent.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
FusedRecurrentUpdateFunction.backward
fused_recurrent_gated_delta_rule_update_fwd_kernel
```

### `tensorrt_llm/_torch/modules/fla/fused_sigmoid_gating_recurrent.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
_current_cu_stream
fused_sigmoid_gating_delta_rule_update_kernel
```

### `tensorrt_llm/_torch/modules/fla/fused_state_io.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
_cast_scatter_fp32_vk_to_vk_kernel
_gather_cast_vk_to_fp32_vk_kernel
```

### `tensorrt_llm/_torch/modules/fla/inforward_snapshot.py`

库存函数 19；已覆盖 18；未覆盖 1。

```text
InforwardSnapshotHost.inforward_snapshot_max_tail
```

### `tensorrt_llm/_torch/modules/fla/l2norm.py`

库存函数 8；已覆盖 2；未覆盖 6。

```text
L2Norm.__init__
L2Norm.forward
L2NormFunction.forward
l2norm
l2norm_fwd_kernel
l2norm_fwd_kernel1
```

### `tensorrt_llm/_torch/modules/fla/layernorm_gated.py`

库存函数 12；已覆盖 3；未覆盖 9。

```text
LayerNorm.__init__
LayerNorm.forward
LayerNorm.reset_parameters
RMSNorm.__init__
RMSNorm.forward
RMSNorm.reset_parameters
_layer_norm_fwd_1pass_kernel
rms_norm_ref
rmsnorm_fn
```

### `tensorrt_llm/_torch/modules/fla/replay_promotion.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_promote_conv_and_advance_replay_kernel
```

### `tensorrt_llm/_torch/modules/fla/solve_tril.py`

库存函数 4；已覆盖 1；未覆盖 3。

```text
merge_16x16_to_32x32_inverse_kernel
merge_16x16_to_64x64_inverse_kernel
solve_tril_16x16_kernel
```

### `tensorrt_llm/_torch/modules/fla/utils.py`

库存函数 23；已覆盖 14；未覆盖 9。

```text
assert_close
checkpoint
checkpoint.<locals>.wrapper
get_abs_err
get_err_ratio
get_multiprocessor_count
require_version
require_version.<locals>.decorator
require_version.<locals>.decorator.<locals>.wrapper
```

### `tensorrt_llm/_torch/modules/fla/wy_fast.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
recompute_w_u_fwd_kernel
```

### `tensorrt_llm/_torch/modules/gemma4/clustered_logits.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
_clustered_logits_kernel
_select_kernel
```

### `tensorrt_llm/_torch/modules/gemma4/fused_qkv.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
_gemma4_qkv_norm_rope_quant_kernel
_store_head
```

### `tensorrt_llm/_torch/modules/kimi_k3_attn_res/kimi_k3_attn_res.py`

库存函数 11；已覆盖 1；未覆盖 10。

```text
KimiK3AttnResidualOp.__init__
KimiK3AttnResidualOp._attn_res_fwd_optimized
KimiK3AttnResidualOp._attn_res_fwd_reference
KimiK3AttnResidualOp._validate_inputs
KimiK3AttnResidualOp.copy_weights_from
KimiK3AttnResidualOp.forward
KimiK3AttnResidualOp.forward_hf_layout
KimiK3AttnResidualOp.kernel_source
KimiK3AttnResidualOp.precompile
attn_res_fwd_chunked_reference
```

### `tensorrt_llm/_torch/modules/kimi_kda/_kda_kernels.py`

库存函数 23；已覆盖 20；未覆盖 3。

```text
_copy_kda_replay_conv_window_kernel
_default_get_sm_version
_record_kda_replay_acceptance_kernel
```

### `tensorrt_llm/_torch/modules/kimi_kda/cache_manager.py`

库存函数 96；已覆盖 95；未覆盖 1。

```text
KDAReplayState.on_cache_stream
```

### `tensorrt_llm/_torch/modules/layer_norm.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
LayerNorm.skip_forward
```

### `tensorrt_llm/_torch/modules/linear.py`

库存函数 246；已覆盖 206；未覆盖 40。

```text
FP8QDQLinearMethod.apply_linear_allreduce_norm
Linear._run_bf16_linear_locality_domain
Linear._run_linear_locality_domain
Linear._run_nvfp4_linear_locality_domain
Linear._save_locality_domain_reload_metadata
Linear._split_bf16_weights_for_locality_domain
Linear._split_nvfp4_weights_for_locality_domain
Linear._split_weights_for_locality_domain
Linear.apply_linear_allreduce
Linear.has_mxfp8
Linear.has_w4a16_awq
Linear.has_w4a8_mxfp4_fp8
LinearMethodBase.apply
LinearMethodBase.apply_linear_allreduce
LinearMethodBase.apply_slice
LinearMethodBase.create_weights
LinearMethodBase.load_weight_scales
LinearMethodBase.load_weights_fused_gate_up_linear
LinearMethodBase.load_weights_fused_qkv_linear
LinearMethodBase.load_weights_vanilla
LinearMethodBase.prefers_bf16_swiglu_input
MXFP8LinearMethod._store_fused_scales
MXFP8LinearMethod._store_scale_shard
MarlinNVFP4LinearMethod.apply_linear_allreduce
MarlinNVFP4LinearMethod.get_tp_alignment
NVFP4ARCLinearMethod._input_prepare
NVFP4ARCLinearMethod.create_weights
NVFP4ARCLinearMethod.load_weights
NVFP4LinearMethod.apply_linear_allreduce
W4A16_AWQ_LinearMethod.load_weights_fused_gate_up_linear
W4A16_AWQ_LinearMethod.load_weights_fused_qkv_linear
W4A8MXFP4FP8LinearMethod.get_tp_alignment
W4A8NVFP4FP8LinearMethod.get_tp_alignment
W4A8NVFP4FP8LinearMethod.load_weights_fused_qkv_linear
W4A8_AWQ_LinearMethod.get_tp_alignment
W4A8_AWQ_LinearMethod.load_weights_fused_gate_up_linear
W4A8_AWQ_LinearMethod.load_weights_fused_qkv_linear
WeightOnlyQuantLinearMethod.get_tp_alignment
WeightOnlyQuantLinearMethod.load_weights_fused_gate_up_linear
WeightOnlyQuantLinearMethod.load_weights_fused_qkv_linear
```

### `tensorrt_llm/_torch/modules/logits_cast.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_widen_to_fp32_kernel
```

### `tensorrt_llm/_torch/modules/low_m_mma_gemm.py`

库存函数 24；已覆盖 20；未覆盖 4。

```text
_SharedExpertLaunchRunner.unique_id
_shared_expert_bucket
_shared_expert_tuning_config
raced_shared_expert
```

### `tensorrt_llm/_torch/modules/mamba/cache_manager.py`

库存函数 51；已覆盖 50；未覆盖 1。

```text
ReplayHistory.uses_replay
```

### `tensorrt_llm/_torch/modules/mamba/causal_conv1d_triton.py`

库存函数 6；已覆盖 2；未覆盖 4。

```text
_causal_conv1d_fwd_kernel
_causal_conv1d_update_kernel
causal_conv1d_fn
causal_conv1d_fn.<locals>.grid
```

### `tensorrt_llm/_torch/modules/mamba/fuse_elementwise_ops.py`

库存函数 11；已覆盖 6；未覆盖 5。

```text
_extract_transpose_prefill_kernel
_fused_conv_output_transpose_kernel
_fused_gdn_post_conv_kernel
_pack_gdn_decode_qkv_kernel
_ssd_output_transpose_kernel
```

### `tensorrt_llm/_torch/modules/mamba/fused_gdn_conv.py`

库存函数 6；已覆盖 1；未覆盖 5。

```text
_conv_silu_tile
_fused_gdn_conv_post_conv_kernel
_gdn_conv_window_kernel
_head_columns
_store_head_tile
```

### `tensorrt_llm/_torch/modules/mamba/gdn_inforward_split.py`

库存函数 11；已覆盖 7；未覆盖 4。

```text
_split_conv_state_kernel
_split_state_rows_kernel
_tail_scatter_kernel
_tail_token_gather_kernel
```

### `tensorrt_llm/_torch/modules/mamba/gdn_mixer.py`

库存函数 67；已覆盖 58；未覆盖 9。

```text
Qwen3NextGatedDeltaNet._fused_verify
Qwen3NextGatedDeltaNet._fused_verify_inproj
Qwen3NextGatedDeltaNet._fused_verify_replay_norm
Qwen3NextGatedDeltaNet._resident_verify_inproj
Qwen3NextGatedDeltaNet._resident_verify_inproj.<locals>._ba
Qwen3NextGatedDeltaNet._resident_verify_inproj.<locals>._qkvz
_reset_gdn_states_all_layers_kernel
_verify_conv_launch
fused_gdn_gating_kernel
```

### `tensorrt_llm/_torch/modules/mamba/gdn_prefill_conv.py`

库存函数 15；已覆盖 4；未覆盖 11。

```text
_conv_post_chunk
_gates
_gdn_conv_post_kernel
_gdn_conv_state_update_kernel
_inv_norm
_l2_normalize
_seq_state
_silu
_state_value
_tile_rows
_window_tap
```

### `tensorrt_llm/_torch/modules/mamba/layernorm_gated.py`

库存函数 15；已覆盖 11；未覆盖 4。

```text
_gated_rows_load
_gated_rows_store
_layer_norm_fwd_1pass_kernel
_rms_norm_gated_fwd_multirow_kernel
```

### `tensorrt_llm/_torch/modules/mamba/mamba2_metadata.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
_cu_seqlens_triton_kernel
```

### `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py`

库存函数 14；已覆盖 12；未覆盖 2。

```text
_
_flashinfer_selective_state_update_op
```

### `tensorrt_llm/_torch/modules/mamba/recurrent_state_cache.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_reset_recurrent_state_rows_kernel
```

### `tensorrt_llm/_torch/modules/mamba/replay_metadata.py`

库存函数 12；已覆盖 10；未覆盖 2。

```text
_prepare_gdn_replay_work_items_kernel
_snapshot_gdn_mtp_replay_work_items_kernel
```

### `tensorrt_llm/_torch/modules/mamba/replay_selective_state_update.py`

库存函数 22；已覆盖 9；未覆盖 13。

```text
_bitrev32
_dynamic_precompute_kernel
_ensure_tma_allocator.<locals>._alloc_fn
_gdc_wait_with_memory_clobber
_persistent_main_impl
_persistent_main_kernel
_persistent_rectangle_impl
_rectangle_precompute_impl
_replay_precompute_impl
_stochastic_round_fp16x2
_stochastic_round_fp8x4_e4m3
_stochastic_round_int16_packed
_stochastic_round_int8_packed
```

### `tensorrt_llm/_torch/modules/mamba/selective_state_update.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
_selective_scan_update_kernel
```

### `tensorrt_llm/_torch/modules/mamba/ssd_bmm.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_bmm_chunk_fwd_kernel
```

### `tensorrt_llm/_torch/modules/mamba/ssd_chunk_scan.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_chunk_scan_fwd_kernel
```

### `tensorrt_llm/_torch/modules/mamba/ssd_chunk_state.py`

库存函数 6；已覆盖 3；未覆盖 3。

```text
_chunk_cumsum_fwd_kernel
_chunk_state_fwd_kernel
_chunk_state_varlen_kernel
```

### `tensorrt_llm/_torch/modules/mamba/ssd_combined.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
_mamba_chunk_scan_flashinfer_fwd
```

### `tensorrt_llm/_torch/modules/mamba/ssd_state_passing.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_state_passing_fwd_kernel
```

### `tensorrt_llm/_torch/modules/mhc/hyper_connection.py`

库存函数 36；已覆盖 34；未覆盖 2。

```text
HCHead.skip_forward
_collapse_kernel
```

### `tensorrt_llm/_torch/modules/mhc/mhc_lowm.py`

库存函数 19；已覆盖 18；未覆盖 1。

```text
MoeUnfinalized.shape
```

### `tensorrt_llm/_torch/modules/mxfp8_draft_head.py`

库存函数 16；已覆盖 10；未覆盖 6。

```text
MXFP8DraftHead.__init__
MXFP8DraftHead.supports
_decode_keys_kernel
build_mxfp8_draft_head
exchange_argmax_keys
is_supported
```

### `tensorrt_llm/_torch/modules/mxfp8_swapab.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
MXFP8SwapABGemmRunner.unique_id
```

### `tensorrt_llm/_torch/modules/q35_mtp_glue.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_q35_mtp_input_kernel
```

### `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection.py`

库存函数 15；已覆盖 14；未覆盖 1。

```text
Qwen4ExpHyperConnection.extra_repr
```

### `tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py`

库存函数 12；已覆盖 6；未覆盖 6。

```text
_
_hc_combine_kernel
_hc_combine_norm_kernel
_hc_gate_mix_kernel
_hc_moe_finalize_combine_norm_kernel
_hc_silu_kernel
```

### `tensorrt_llm/_torch/modules/qwen4_exp/ple.py`

库存函数 40；已覆盖 39；未覆盖 1。

```text
_gather_pinned_embedding_kernel
```

### `tensorrt_llm/_torch/modules/qwen4_exp/ple_kernels.py`

库存函数 19；已覆盖 11；未覆盖 8。

```text
_
_decode_short_conv_kernel
_gate_value_kernel
_ngram_hash_kernel
_packed_short_conv_kernel
_packed_short_conv_state_kernel
_round_bf16_to_fp32
_short_conv_state_kernel
```

### `tensorrt_llm/_torch/modules/rms_norm_fp8_per_token.py`

库存函数 7；已覆盖 2；未覆盖 5。

```text
_
_add_rms_norm_fp8_per_token_kernel
_quantize_fp8_per_token
_rms_norm_gated_fp8_per_token_kernel
rms_norm_gated_fp8_per_token_eligible
```

### `tensorrt_llm/_torch/modules/situ.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
situ_and_mul_kernel
```

### `tensorrt_llm/_torch/modules/swiglu.py`

库存函数 5；已覆盖 2；未覆盖 3。

```text
scale_and_clamp
silu_and_mul_2in_kernel
silu_and_mul_kernel
```

### `tensorrt_llm/_torch/modules/triton_linear.py`

库存函数 22；已覆盖 16；未覆盖 6。

```text
TritonFP8QDQLinearMethod.load_weights_fused_gate_up_linear
TritonFP8QDQLinearMethod.load_weights_fused_qkv_linear
TritonMXFP4LinearMethod.load_weights_fused_gate_up_linear
TritonMXFP4LinearMethod.load_weights_fused_qkv_linear
TritonUnquantizedLinearMethod.load_weights_fused_gate_up_linear
TritonUnquantizedLinearMethod.load_weights_fused_qkv_linear
```

### `tensorrt_llm/_torch/modules/xielu.py`

库存函数 7；已覆盖 5；未覆盖 2。

```text
XIELU.extra_repr
XIELU.post_load_weights
```

### `tensorrt_llm/_torch/moe/custom_ops/cute_dsl_megamoe_custom_op.py`

库存函数 77；已覆盖 70；未覆盖 7。

```text
Sm100MegaMoENvfp4Runner._autotuner_inputs_pre_hook
_
_evict_latched_tuning_workspaces
_megamoe_autotune_num_tokens
_megamoe_ugpu_die_smid_mask
get_megamoe_profiling_scratch
synthesize_profiling_topk
```

### `tensorrt_llm/_torch/moe/expert_statistic.py`

库存函数 11；已覆盖 6；未覆盖 5。

```text
ExpertStatistic.__init__
ExpertStatistic._maybe_add_info
ExpertStatistic._set_iter
ExpertStatistic._set_layer
ExpertStatistic._should_record
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/allgather_reducescatter.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
AllGatherReduceScatter.carries_unrouted_rows
AllGatherReduceScatter.is_workload_feasible
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/base.py`

库存函数 18；已覆盖 5；未覆盖 13。

```text
CheckpointBlockable.checkpoint_blocked_reason
CheckpointableCommunication.checkpoint_prepare
CheckpointableCommunication.checkpoint_resource_key
CheckpointableCommunication.checkpoint_restore
Communication.carries_unrouted_rows
Communication.checkpoint_blocked_reason
Communication.combine
Communication.combine_fuses_addend
Communication.dispatch
Communication.get_eplb_gathered_statistics
Communication.is_platform_supported
Communication.is_workload_feasible
Communication.supports_finalize_before_combine
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/deep_ep_low_latency.py`

库存函数 14；已覆盖 13；未覆盖 1。

```text
DeepEPLowLatency.supports_low_precision_combine
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/nccl_ep.py`

库存函数 15；已覆盖 14；未覆盖 1。

```text
NcclEP.is_workload_feasible
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/nvlink_two_sided.py`

库存函数 21；已覆盖 18；未覆盖 3。

```text
MnnvlMoe.compute_target_rank_id
MnnvlMoe.mnnvl_moe_alltoallv_prepare
MnnvlMoe.mnnvl_moe_expert_static_allgather
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/pull_combine.py`

库存函数 8；已覆盖 4；未覆盖 4。

```text
_globaltimer
_handshake_kernel
_pull_reduce_kernel
_trap
```

### `tensorrt_llm/_torch/moe/fused_moe/configurable_moe.py`

库存函数 48；已覆盖 46；未覆盖 2。

```text
ConfigurableMoE.can_implement
ConfigurableMoE.process_weights_after_loading
```

### `tensorrt_llm/_torch/moe/fused_moe/cutedsl/fused_fc12/mxfp8.py`

库存函数 12；已覆盖 11；未覆盖 1。

```text
TrtllmCutedslFusedFc12Mxfp8Impl.supports_moe_output_in_alltoall_workspace
```

### `tensorrt_llm/_torch/moe/fused_moe/cutedsl/fused_fc12/nvfp4.py`

库存函数 16；已覆盖 14；未覆盖 2。

```text
TrtllmCutedslFusedFc12Nvfp4Impl.sm_reserve_ceiling
TrtllmCutedslFusedFc12Nvfp4Impl.supports_moe_output_in_alltoall_workspace
```

### `tensorrt_llm/_torch/moe/fused_moe/cutedsl/grouped_gemm/__init__.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
cutedsl_grouped_gemm_leaf
```

### `tensorrt_llm/_torch/moe/fused_moe/cutedsl/runners.py`

库存函数 31；已覆盖 27；未覆盖 4。

```text
CuteDslFusedMoEBF16InputsHelper.infer_shape_num_tokens
CuteDslFusedMoEBF16InputsHelper.inputs_pre_hook
CuteDslFusedMoEBF16Runner.get_valid_tactics
CuteDslFusedMoENvfp4InputsHelper.spread_num_tokens_per_expert
```

### `tensorrt_llm/_torch/moe/fused_moe/cutlass/w4a16_nvfp4.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
TrtllmCutlassW4a16Nvfp4Impl._get_quant_method
```

### `tensorrt_llm/_torch/moe/fused_moe/deep_ep_utils.py`

库存函数 17；已覆盖 14；未覆盖 3。

```text
VariableLengthLowLatencyBuffer.clean_low_latency_buffer
VariableLengthLowLatencyBuffer.low_latency_combine_low_precision
VariableLengthLowLatencyBuffer.low_latency_dispatch_fp4
```

### `tensorrt_llm/_torch/moe/fused_moe/fused_moe_deepgemm.py`

库存函数 23；已覆盖 18；未覆盖 5。

```text
_fused_expand_group_quant_fp8
_masked_index_copy_group_quant_fp8
_preprocess_after_permute_kernel
fused_gather_finalize_kernel
masked_index_gather_kernel
```

### `tensorrt_llm/_torch/moe/fused_moe/fused_moe_triton.py`

库存函数 49；已覆盖 34；未覆盖 15。

```text
TritonFP8QDQFusedMoEMethod.__init__
TritonFP8QDQFusedMoEMethod.apply
TritonFP8QDQFusedMoEMethod.create_weights
TritonFP8QDQFusedMoEMethod.load_expert_w2_weight_scale_fp8
TritonFP8QDQFusedMoEMethod.load_expert_w3_w1_weight_scale_fp8_qdq
TritonFP8QDQFusedMoEMethod.load_expert_weights_to_dst
TritonFP8QDQFusedMoEMethod.load_quant_scales
TritonFP8QDQFusedMoEMethod.setup_quant_scales
TritonMXFP4FusedMoEMethod._permute_mxfp4_quantize
TritonUnquantizedFusedMoEMethod.load_expert_w2_weight
TritonUnquantizedFusedMoEMethod.load_expert_w3_w1_weight
TritonUnquantizedFusedMoEMethod.load_expert_weights_to_dst
_routing_clear_bitmatrix
_routing_shift_bitmatrix_range
swiglu_torch
```

### `tensorrt_llm/_torch/moe/fused_moe/fused_moe_triton_fp8_block_scale.py`

库存函数 25；已覆盖 11；未覆盖 14。

```text
_fused_moe_bf16act_fp8w_kernel
_fused_moe_fp8act_fp8w_kernel
_gated_activation_kernel
_int32_add
_moe_align_small_kernel
_moe_expert_ids_kernel
_moe_histogram_kernel
_moe_prefix_kernel
_moe_scatter_kernel
_moe_topk_sum_kernel
_per_token_group_quant_fp8_kernel
_silu_ieee
_swiglu_quant_fp8_kernel
_write_zeros_to_output
```

### `tensorrt_llm/_torch/moe/fused_moe/impl_base.py`

库存函数 18；已覆盖 13；未覆盖 5。

```text
MoEImplBase._get_quant_method
MoEImplBase.can_implement
MoEImplBase.get_workspaces
MoEImplBase.quantize_input
MoEImplBase.run_moe
```

### `tensorrt_llm/_torch/moe/fused_moe/impl_blocks.py`

库存函数 29；已覆盖 23；未覆盖 6。

```text
MoEExecutionContractMixin.supports_moe_output_in_alltoall_workspace
MoEExecutionContractMixin.uses_locality_domain
MoEWeightOwnerMixin._get_quant_method
MoEWeightOwnerMixin.has_fp8_qdq
MoEWeightOwnerMixin.has_nvfp4_activation_quantization
MoEWeightOwnerMixin.has_w4a8_nvfp4_fp8
```

### `tensorrt_llm/_torch/moe/fused_moe/impl_contract.py`

库存函数 39；已覆盖 34；未覆盖 5。

```text
MoEEligibility.__bool__
MoEEplbBinding.eplb_enabled
MoEProblem.is_fully_specified
MoERejection.to_dict
MoEResolutionReport.to_dict
```

### `tensorrt_llm/_torch/moe/fused_moe/impl_identity.py`

库存函数 27；已覆盖 22；未覆盖 5。

```text
MoEImplQuery.__str__
MoEImplQuery.as_impl_id
MoEImplQuery.is_exact
MoEImplRegistry.__len__
MoEImplRegistry.field_of
```

### `tensorrt_llm/_torch/moe/fused_moe/interface.py`

库存函数 37；已覆盖 31；未覆盖 6。

```text
MoE.can_implement
MoE.can_use_deep_ep_direct_metadata
MoE.forward_impl
MoE.has_comm_strategy
MoE.quantize_input
MoE.run_moe
```

### `tensorrt_llm/_torch/moe/fused_moe/marlin/__init__.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
marlin_leaf
```

### `tensorrt_llm/_torch/moe/fused_moe/marlin/base.py`

库存函数 11；已覆盖 10；未覆盖 1。

```text
_sum_topk_kernel
```

### `tensorrt_llm/_torch/moe/fused_moe/mega_moe/mega_moe_cute_dsl.py`

库存函数 62；已覆盖 54；未覆盖 8。

```text
TrtllmCutedslMegaMoeNvfp4Impl._launch_megamoe_kernel.<locals>._deferred_scratch_factory
TrtllmCutedslMegaMoeNvfp4Impl.can_fuse_shared_expert
TrtllmCutedslMegaMoeNvfp4Impl.decline_shared_expert
TrtllmCutedslMegaMoeNvfp4Impl.experts_of_slots
TrtllmCutedslMegaMoeNvfp4Impl.pre_reload_weights
TrtllmCutedslMegaMoeNvfp4Impl.slots_of_experts
_replica_slots
_warn_shared_expert_fusion_ignored
```

### `tensorrt_llm/_torch/moe/fused_moe/mega_moe/mega_moe_deepgemm.py`

库存函数 26；已覆盖 25；未覆盖 1。

```text
_quantize_bf16_to_fp8_ue8m0.<locals>._call
```

### `tensorrt_llm/_torch/moe/fused_moe/mega_moe/shared_expert.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
_mxfp8_quant_kernel
mxfp8_quantize_megamoe
```

### `tensorrt_llm/_torch/moe/fused_moe/mega_moe/ugpu_weights.py`

库存函数 31；已覆盖 13；未覆盖 18。

```text
_CudaVmm.__init__
_CudaVmm._properties
_CudaVmm.alias
_CudaVmm.allow_access
_CudaVmm.create
_CudaVmm.free_address
_CudaVmm.granularity
_CudaVmm.locality
_CudaVmm.map
_CudaVmm.release
_CudaVmm.reserve
_CudaVmm.synchronize
_CudaVmm.unmap
_RawDeviceBytes.__init__
_driver_result
_smids_of
die_smid_mask
localize_megamoe_weights
```

### `tensorrt_llm/_torch/moe/fused_moe/moe_comm_timeout_guard.py`

库存函数 19；已覆盖 18；未覆盖 1。

```text
MoECommTimeoutProxy.set_timeout_seconds
```

### `tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py`

库存函数 83；已覆盖 71；未覆盖 12。

```text
BaseMoeLoadBalancer.add_layer
BaseMoeLoadBalancer.end_iter
BaseMoeLoadBalancer.finalize_model
BaseMoeLoadBalancer.is_static_routing
BaseMoeLoadBalancer.register_weight_slots_after_to_cuda
BaseMoeLoadBalancer.set_repeated_for_next_layer
BaseMoeLoadBalancer.set_warm_up_iter_count
BaseMoeLoadBalancer.shutdown
BaseMoeLoadBalancer.start_iter
MoeLoadBalancer.__repr__
MoeLoadBalancer.set_repeated_for_next_layer
SingleLayerMoeLoadBalancer.get_layer_idx
```

### `tensorrt_llm/_torch/moe/fused_moe/moe_op_backend.py`

库存函数 14；已覆盖 9；未覆盖 5。

```text
MoEOpBackend.fp4_quantize
MoEOpBackend.mxfp8_quantize
MoEOpBackend.run_bf16_moe
MoEOpBackend.run_fp4_block_scale_moe
MoEOpBackend.run_fp8_block_scale_moe
```

### `tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py`

库存函数 34；已覆盖 33；未覆盖 1。

```text
MoEScheduler.forward
```

### `tensorrt_llm/_torch/moe/fused_moe/moe_warmup_placement.py`

库存函数 61；已覆盖 49；未覆盖 12。

```text
WarmupPlacementController._cost_inputs
WarmupPlacementController.active
WarmupPlacementController.counters
WarmupPlacementController.reset_schedule
_backend_name
_context_rows
_count_kernel
cost_coeffs
count_routing
expert_cost
register_controller
set_live_rows
```

### `tensorrt_llm/_torch/moe/fused_moe/nccl_ep_utils.py`

库存函数 13；已覆盖 12；未覆盖 1。

```text
destroy_all_nccl_ep_contexts
```

### `tensorrt_llm/_torch/moe/fused_moe/per_layer_dynamic_eplb/arena.py`

库存函数 53；已覆盖 7；未覆盖 46。

```text
RebalanceLiveArena.assert_identity
RebalanceLiveArena.publish_ready_generation
RebalanceLiveArena.ready_generation
RebalanceLiveArena.slot_count
RebalanceLiveArena.terminal_flags_tensor
SharedSlotArenaProvider.register_producer
SharedSlotArenaProvider.tekit_alias
_MulticastSpec.__init__
_RawPointer.__init__
_SharedLayer.__init__
_SharedSlotPool.__init__
_SharedSlotPool._cleanup_manifest
_SharedSlotPool._close_producers
_SharedSlotPool._collective_phase
_SharedSlotPool._create_page
_SharedSlotPool._create_sets
_SharedSlotPool._flat
_SharedSlotPool._on_device
_SharedSlotPool.build_layer
_SharedSlotPool.close_collectively
_SharedSlotPool.register_producer
_ViewSpec.__init__
_ViewSpec.data_ptr
_ViewSpec.element_size
_ViewSpec.stride
_Vmm.__init__
_Vmm._release_handles
_Vmm.create
_Vmm.free_reservations
_Vmm.import_
_Vmm.import_multicast
_Vmm.map
_Vmm.multicast_properties
_Vmm.record_multicast_bind
_Vmm.register_created_multicast
_Vmm.release_created_handles
_Vmm.release_imported_handles
_Vmm.reserve
_Vmm.unbind_all
_Vmm.unmap_all
_Vmm.zero
_align_up
_create_multicast
_plane_dtypes
_plane_view
allocate_rebalance_arena
```

### `tensorrt_llm/_torch/moe/fused_moe/per_layer_dynamic_eplb/layer.py`

库存函数 27；已覆盖 17；未覆盖 10。

```text
PerLayerDynamicMoeLayerLoadBalancer.done_set_cpu_stage
PerLayerDynamicMoeLayerLoadBalancer.done_wait_gpu_stage
PerLayerDynamicMoeLayerLoadBalancer.get_layer_idx
PerLayerDynamicMoeLayerLoadBalancer.get_repeat_count
PerLayerDynamicMoeLayerLoadBalancer.is_static_routing
PerLayerDynamicMoeLayerLoadBalancer.start_wait_gpu_stage
PerLayerDynamicMoeLayerLoadBalancer.update_local_statistic
PerLayerDynamicMoeLayerLoadBalancer.update_statistic_with_gathered_statistic
PerLayerDynamicMoeLayerLoadBalancer.update_statistic_with_global_ids
PerLayerDynamicMoeLayerLoadBalancer.update_statistic_with_local_ids
```

### `tensorrt_llm/_torch/moe/fused_moe/per_layer_dynamic_eplb/manager.py`

库存函数 19；已覆盖 14；未覆盖 5。

```text
PerLayerDynamicMoeLoadBalancer.finalize_model
PerLayerDynamicMoeLoadBalancer.get_copy_stream
PerLayerDynamicMoeLoadBalancer.register_weight_slots_after_to_cuda
PerLayerDynamicMoeLoadBalancer.set_repeated_for_next_layer
PerLayerDynamicMoeLoadBalancer.set_warm_up_iter_count
```

### `tensorrt_llm/_torch/moe/fused_moe/per_layer_dynamic_eplb/planner.py`

库存函数 16；已覆盖 12；未覆盖 4。

```text
RebalanceSlotSchedulerGroup.__init__
RebalanceSlotSchedulerGroup.close
RebalanceSlotSchedulerGroup.plan
build_rebalance_slot_scheduler_group
```

### `tensorrt_llm/_torch/moe/fused_moe/quantization.py`

库存函数 298；已覆盖 279；未覆盖 19。

```text
DeepSeekFP8BlockScalesFusedMoEMethod._prepare_shared_weights_for_finalization
DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm._prepare_shared_weights_for_finalization
FusedMoEMethodBase.apply
FusedMoEMethodBase.post_load_weights
FusedMoEMethodBase.setup_quant_scales
MXFP4WeightFusedMoEMethod.load_expert_w2_weight_scale_mxfp4
MXFP4WeightFusedMoEMethod.load_expert_w3_w1_weight_scale_mxfp4
MXFP4WeightFusedMoEMethod.setup_quant_scales
MXFP8CuteDslFusedMoEMethod._interleave_w3_w1_weight_scale
NVFP4FusedMoEMethod.finalize_streamed_expert
NVFP4FusedMoEMethod.load_expert_fc2_input_scale_nvfp4
NVFP4FusedMoEMethod.load_expert_fc31_input_scale_nvfp4
NVFP4FusedMoEMethod.load_expert_w2_weight_scale_nvfp4
NVFP4FusedMoEMethod.load_expert_w3_w1_weight_scale_nvfp4
NVFP4MegaMoECuteDslMethod.fc1_sf_flat_size
NVFP4MegaMoECuteDslMethod.fc2_sf_flat_size
NVFP4TRTLLMGenFusedMoEMethod._round_up
W4A16NVFP4CutlassFusedMoEMethod.dequant_active_experts_to_hp
W4A8MXFP4MXFP8MegaMoEDeepGemmMethod._iter_fused_gate_up_expert_weights
```

### `tensorrt_llm/_torch/moe/fused_moe/routing.py`

库存函数 98；已覆盖 91；未覆盖 7。

```text
BaseMoeRoutingMethod.apply
BaseMoeRoutingMethod.apply_with_aux
BaseMoeRoutingMethod.serves_with_sort
BasePerfectRouterPlanner.create_logits
BasePerfectRouterPlanner.prepare_routing_method_in_place
SqrtSoftplusMoeRoutingMethod.get_scores
SqrtSoftplusMoeRoutingMethod.requires_separated_routing
```

### `tensorrt_llm/_torch/moe/fused_moe/triton_dequant_nvfp4.py`

库存函数 6；已覆盖 2；未覆盖 4。

```text
_dequant_nvfp4_active_kernel
_dequant_nvfp4_linear_kernel
build_active_expert_mask
dequant_nvfp4_active_triton
```

### `tensorrt_llm/_torch/moe/fused_moe/trtllm_gen/base.py`

库存函数 17；已覆盖 16；未覆盖 1。

```text
TrtllmGenFusedMoEBase._situ_tp_weight_alignment
```

### `tensorrt_llm/_torch/moe/fused_moe/trtllm_gen/decode_front.py`

库存函数 43；已覆盖 36；未覆盖 7。

```text
_fence_acq_rel_gpu
_nvfp4_quant_row
_quant_rows_kernel
_rcp_approx
_route_tables_kernel
_route_token
_router_gemm_kernel
```

### `tensorrt_llm/_torch/moe/fused_moe/trtllm_gen/decode_moe.py`

库存函数 21；已覆盖 20；未覆盖 1。

```text
precedes
```

### `tensorrt_llm/_torch/moe/fused_moe/trtllm_gen/fp4_block_scale.py`

库存函数 19；已覆盖 17；未覆盖 2。

```text
TRTLLMGenFp4BlockScaleBase.ran_on_routing_tables
TRTLLMGenNvfp4Base._situ_tp_weight_alignment
```

### `tensorrt_llm/_torch/moe/fused_moe/trtllm_gen/trtllm_bf16.py`

库存函数 10；已覆盖 9；未覆盖 1。

```text
TrtllmTrtllmGenBf16Impl.supports_unfinalized_output
```

### `tensorrt_llm/_torch/moe/fused_moe/trtllm_gen/trtllm_w4a8_nvfp4_fp8.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
TrtllmTrtllmGenW4a8Nvfp4Fp8Impl.supports_unfinalized_output
```

### `tensorrt_llm/_torch/moe/fused_shared_expert.py`

库存函数 5；已覆盖 2；未覆盖 3。

```text
_
_fused_sigmoid_gate_mul_add_op
_sigmoid_gate_mul_add_kernel
```

### `tensorrt_llm/_torch/moe/kernels/blackwell/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py`

库存函数 28；已覆盖 16；未覆盖 12。

```text
BlockScaledContiguousGatherGroupedGemmKernel.__call__
BlockScaledContiguousGatherGroupedGemmKernel.__call__.<locals>._make_tma_b
BlockScaledContiguousGatherGroupedGemmKernel._apply_relu2_epilogue
BlockScaledContiguousGatherGroupedGemmKernel._apply_situ_epilogue
BlockScaledContiguousGatherGroupedGemmKernel._apply_situ_epilogue.<locals>._sigmoid
BlockScaledContiguousGatherGroupedGemmKernel._apply_swiglu_epilogue
BlockScaledContiguousGatherGroupedGemmKernel._expert_count_tile_info
BlockScaledContiguousGatherGroupedGemmKernel._get_tma_atom_kind
BlockScaledContiguousGatherGroupedGemmKernel.kernel
BlockScaledContiguousGatherGroupedGemmKernel.wrapper
cvt_sf_M32x4xrm_K4xrk_L_to_MKL
cvt_sf_MKL_to_M32x4xrm_K4xrk_L
```

### `tensorrt_llm/_torch/moe/kernels/blackwell/blockscaled_contiguous_grouped_gemm.py`

库存函数 19；已覆盖 14；未覆盖 5。

```text
Sm100BlockScaledContiguousGroupedGemmKernel.__call__
Sm100BlockScaledContiguousGroupedGemmKernel._get_tma_atom_kind
Sm100BlockScaledContiguousGroupedGemmKernel.kernel
Sm100BlockScaledContiguousGroupedGemmKernel.wrapper
cvt_sf_MKL_to_M32x4xrm_K4xrk_L
```

### `tensorrt_llm/_torch/moe/kernels/blackwell/blockscaled_contiguous_grouped_gemm_finalize_fusion.py`

库存函数 20；已覆盖 13；未覆盖 7。

```text
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.__call__
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.__call__.<locals>._make_tma_b
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel._expert_count_tile_info
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel._get_tma_atom_kind
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.kernel
Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.wrapper
cvt_sf_MKL_to_M32x4xrm_K4xrk_L
```

### `tensorrt_llm/_torch/moe/kernels/blackwell/blockscaled_contiguous_grouped_gemm_swiglu_fusion.py`

库存函数 20；已覆盖 9；未覆盖 11。

```text
Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel.__call__
Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel._get_tma_atom_kind
Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel.can_implement
Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel.is_valid_dtypes_and_scale_factor_vec_size
Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel.is_valid_layouts
Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel.is_valid_mma_tiler_and_cluster_shape
Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel.is_valid_tensor_alignment
Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel.is_valid_tensor_alignment.<locals>.check_contigous_16B_alignment
Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel.kernel
Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel.wrapper
cvt_sf_MKL_to_M32x4xrm_K4xrk_L
```

### `tensorrt_llm/_torch/moe/kernels/blackwell/dsv4_route_quant.py`

库存函数 41；已覆盖 27；未覆盖 14。

```text
DSv4RouteNvfp4QuantKernel.__call__
DSv4RouteNvfp4QuantKernel.clear_tail
DSv4RouteNvfp4QuantKernel.counts_row
DSv4RouteNvfp4QuantKernel.home_first_owner
DSv4RouteNvfp4QuantKernel.kernel
DSv4RouteNvfp4QuantKernel.logit
DSv4RouteNvfp4QuantKernel.pick
DSv4RouteNvfp4QuantKernel.quantize
DSv4RouteNvfp4QuantKernel.route
DSv4RouteNvfp4QuantKernel.route_warp
DSv4RouteNvfp4QuantKernel.route_with_tables
DSv4RouteNvfp4QuantKernel.score
DSv4RouteNvfp4QuantKernel.score_logit
DSv4RouteNvfp4QuantKernel.store_picks
```

### `tensorrt_llm/_torch/moe/kernels/blackwell/kimi_k3_router_logits.py`

库存函数 13；已覆盖 9；未覆盖 4。

```text
KimiK3RouterLogitsKernel.__call__
KimiK3RouterLogitsKernel._wfrag
KimiK3RouterLogitsKernel.kernel
precompile
```

### `tensorrt_llm/_torch/moe/kernels/blackwell/kimi_k3_routing.py`

库存函数 35；已覆盖 28；未覆盖 7。

```text
NoauxTcRouteSortKernel.__call__
NoauxTcRouteSortKernel.kernel
NoauxTcRouteSortKernel.select_token
NoauxTcRouteSortKernel.sort
NoauxTcRouteSortKernel.team_sync
NoauxTcRouteSortKernel.topk_rounds
NoauxTcRouteSortKernel.topk_token
```

### `tensorrt_llm/_torch/moe/kernels/blackwell/megamoe_stage_inputs.py`

库存函数 7；已覆盖 4；未覆盖 3。

```text
MegaMoEStageInputsKernel.__call__
MegaMoEStageInputsKernel.copy_row
MegaMoEStageInputsKernel.kernel
```

### `tensorrt_llm/_torch/moe/kernels/blackwell/moe_as_dense_gemm/fc1.py`

库存函数 21；已覆盖 16；未覆盖 5。

```text
Sm100BlockScaledPersistentDenseGemmKernel.__call__
Sm100BlockScaledPersistentDenseGemmKernel.kernel
Sm100BlockScaledPersistentDenseGemmKernel.wrapper
cvt_sf_M32x4xrm_K4xrk_L_to_MKL
cvt_sf_MKL_to_M32x4xrm_K4xrk_L
```

### `tensorrt_llm/_torch/moe/kernels/blackwell/moe_as_dense_gemm/fc2.py`

库存函数 19；已覆盖 15；未覆盖 4。

```text
Sm100BlockScaledPersistentDenseGemmKernel.__call__
Sm100BlockScaledPersistentDenseGemmKernel.kernel
Sm100BlockScaledPersistentDenseGemmKernel.wrapper
cvt_sf_MKL_to_M32x4xrm_K4xrk_L
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/api.py`

库存函数 15；已覆盖 10；未覆盖 5。

```text
Desc.__len__
KernelClass.aot_compile
KernelClass.name
KernelComponent.impl_desc_require
KernelComponent.problem_desc_require
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain/one_round_router.py`

库存函数 36；已覆盖 15；未覆盖 21。

```text
_OneRoundRouter.__extract_mlir_values__
_OneRoundRouter.__new_from_mlir_values__
_OneRoundRouter._kernel
_OneRoundRouter._push
_OneRoundRouter._push_row
_OneRoundRouter._sort
_OneRoundRouter._wait_flag
_OneRoundRouter._write_pool_row
_OneRoundRouter.fc1_topk_scores_tensor
_OneRoundRouter.launch_router
_OneRoundRouter.local_expert_sizes
_OneRoundRouter.metadata_ready_target
_OneRoundRouter.plane_activation_pointer
_OneRoundRouter.plane_activation_sf_pointer
_OneRoundRouter.pool_expert_base_tensor
_OneRoundRouter.router_grid_cta_count
_OneRoundRouter.router_smem_workspace
_OneRoundRouter.token_src_metadata_pointer
_OneRoundRouter.token_src_metadata_tensor
_OneRoundRouter.wait_for_metadata_ready
_OneRoundRouter.wait_for_sizes_ready
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain/symmetric_buffer.py`

库存函数 14；已覆盖 10；未覆盖 4。

```text
SymmetricBufferDevice.map
SymmetricBufferDevice.map_pointer
_SymmetricBufferHostAdapter.__extract_mlir_values__
_SymmetricBufferHostAdapter.__new_from_mlir_values__
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain/token_comm.py`

库存函数 136；已覆盖 58；未覆盖 78。

```text
TokenCommNonDeterministic.__extract_mlir_values__
TokenCommNonDeterministic.__new_from_mlir_values__
TokenCommNonDeterministic._adaptive_pace
TokenCommNonDeterministic._issue_combine_read
TokenCommNonDeterministic._issue_token_group
TokenCommNonDeterministic._stateless_pace
TokenCommNonDeterministic._token_back_multi_stage
TokenCommNonDeterministic._token_in_multi_stage
TokenCommNonDeterministic.assign_device_members
TokenCommNonDeterministic.combine_landing_tensor
TokenCommNonDeterministic.combine_output_token_bytes
TokenCommNonDeterministic.expert_count_padded
TokenCommNonDeterministic.expert_count_with_trash
TokenCommNonDeterministic.fc1_activation_sf_tensor
TokenCommNonDeterministic.fc1_activation_tensor
TokenCommNonDeterministic.fc1_ready_counter_pointer
TokenCommNonDeterministic.fc1_topk_scores_tensor
TokenCommNonDeterministic.fc2_activation_sf_tensor
TokenCommNonDeterministic.fc2_activation_tensor
TokenCommNonDeterministic.fc2_done_counter_tensor
TokenCommNonDeterministic.launch_router
TokenCommNonDeterministic.local_expert_sizes
TokenCommNonDeterministic.message_index_tensor
TokenCommNonDeterministic.metadata_ready_target
TokenCommNonDeterministic.name_token
TokenCommNonDeterministic.next_token
TokenCommNonDeterministic.pool_expert_base_tensor
TokenCommNonDeterministic.pre_reduced_activation_sf_tensor
TokenCommNonDeterministic.pre_reduced_activation_tensor
TokenCommNonDeterministic.pull_source_addresses
TokenCommNonDeterministic.relay_landing_tensor
TokenCommNonDeterministic.reset_tail
TokenCommNonDeterministic.router_data_cta_count
TokenCommNonDeterministic.router_elements_per_lane
TokenCommNonDeterministic.router_grid_cta_count
TokenCommNonDeterministic.router_push_cta_count
TokenCommNonDeterministic.router_smem_workspace
TokenCommNonDeterministic.router_tokens_per_cta
TokenCommNonDeterministic.router_warps_per_cta
TokenCommNonDeterministic.sf_k_group_count
TokenCommNonDeterministic.token_back
TokenCommNonDeterministic.token_back_pushes
TokenCommNonDeterministic.token_in
TokenCommNonDeterministic.token_src_metadata_pointer
TokenCommNonDeterministic.token_src_metadata_tensor
TokenCommNonDeterministic.wait_for_pool_record
TokenCommNonDeterministic.wait_for_sizes_ready
_MetadataPushRouter.__extract_mlir_values__
_MetadataPushRouter.__new_from_mlir_values__
_MetadataPushRouter._broadcast_sizes_to_peers
_MetadataPushRouter._broadcast_sizes_to_peers.<locals>.finalize
_MetadataPushRouter._build_histogram
_MetadataPushRouter._compute_push_tables
_MetadataPushRouter._dump_contiguous_router_output
_MetadataPushRouter._dump_router_output_by_expert
_MetadataPushRouter._expert_of_route
_MetadataPushRouter._load_router_inputs
_MetadataPushRouter._push_route_record
_MetadataPushRouter._router_extra_cta_body
_MetadataPushRouter._router_helper_multiple_ctas
_MetadataPushRouter._router_helper_single_cta
_MetadataPushRouter._router_kernel
_MetadataPushRouter._router_multiple_ctas
_MetadataPushRouter._router_push_metadata
_MetadataPushRouter._router_single_cta
_MetadataPushRouter._router_value_coordinate
_MetadataPushRouter._sort_router_elements
_MetadataPushRouter.fc1_activation_per_token_scales_tensor
_MetadataPushRouter.fc1_topk_scores_tensor
_MetadataPushRouter.launch_router
_MetadataPushRouter.local_expert_sizes
_MetadataPushRouter.pool_expert_base_tensor
_MetadataPushRouter.router_smem_workspace
_MetadataPushRouter.sizes_tensor
_MetadataPushRouter.token_src_metadata_pointer
_MetadataPushRouter.token_src_metadata_tensor
_MetadataPushRouter.wait_for_metadata_ready
_MetadataPushRouter.wait_for_sizes_ready
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/token_protocol.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
TokenSrcMetadata.is_last_copy
TokenSrcMetadata.store
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/device_workspace.py`

库存函数 32；已覆盖 17；未覆盖 15。

```text
DeviceWorkspace._epoch_ptr
DeviceWorkspace.advance_epoch
DeviceWorkspace.assign_device_members
DeviceWorkspace.byte_alignment
DeviceWorkspace.cosize
DeviceWorkspace.finalized
DeviceWorkspace.nbytes
DeviceWorkspace.ptr
DeviceWorkspace.regions
DeviceWorkspace.reset_next_tail_shared
DeviceWorkspace.reset_tail
DeviceWorkspace.reset_tail_space
DeviceWorkspace.tail_reset_bytes
DeviceWorkspace.tensor
DeviceWorkspace.zero_on_allocate_bytes
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/flag_batch.py`

库存函数 8；已覆盖 1；未覆盖 7。

```text
GpuAsyncReleaseFlagBatchTracker._make
GpuAsyncReleaseFlagBatchTracker.accumulate
GpuAsyncReleaseFlagBatchTracker.fire
GpuReleaseFlagBatchTracker._make
GpuReleaseFlagBatchTracker.accumulate
GpuReleaseFlagBatchTracker.fire
SysReleaseFlagBatchTracker.fire
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/ptx_helpers.py`

库存函数 35；已覆盖 17；未覆盖 18。

```text
cp_reduce_async_bulk_add_bf16_s2g
cvt_f32_to_ue8m0_raw_i32
cvt_f32x4_to_f8x4_scaled4_pack_i32
cvt_f32x4_to_f8x4_scaled_pack_i32
cvt_rn_bf16x2_f32x2
f32x2_pack
fma_rn_f32x2_bf16x2
ldg_v4_b32
lds128_v4_b32
read_globaltimer_ns
read_smid
red_add_relaxed_sys_f32
red_add_relaxed_sys_v2_bf16x2
red_add_release_sys_s32
red_async_add_release_sys_u32
stg_e8m0_from_f32
stg_e8m0x8_from_f32
stg_v4_b32
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/route_reset.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
ClearInvalidRoutes.__call__
ClearInvalidRoutes._kernel
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/smem_workspace.py`

库存函数 33；已覆盖 26；未覆盖 7。

```text
SmemWorkspace.__extract_mlir_values__
SmemWorkspace.__new_from_mlir_values__
SmemWorkspace.assert_fits_arena
SmemWorkspace.nbytes
SmemWorkspace.ptr
SmemWorkspace.region
SmemWorkspace.tensor
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/software_sync.py`

库存函数 15；已覆盖 7；未覆盖 8。

```text
NvlinkBarrier.arrive_and_wait
NvlinkBarrier.assign_device_members
NvlinkBarrier.finalize
SoftwareGridSync.__extract_mlir_values__
SoftwareGridSync.__new_from_mlir_values__
SoftwareGridSync._cta_rendezvous
SoftwareGridSync.assign_device_members
SoftwareGridSync.sync
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/tmem_transpose.py`

库存函数 18；已覆盖 1；未覆盖 17。

```text
_TmemTranspose16x32Core.__init__
_TmemTranspose16x32Core._rmem_copy_view
_TmemTranspose16x32Core.from_r1_perm_until_last_store
_TmemTranspose16x32Core.load_subtile_raw_acc
_TmemTranspose16x32Core.r1_load
_TmemTranspose16x32Core.r1_perm
_TmemTranspose16x32Core.r1_store
_TmemTranspose16x32Core.r2_load
_TmemTranspose16x32Core.r2_store
_TmemTranspose16x32Core.r3_load_bot
_TmemTranspose16x32Core.r3_load_top
_TmemTranspose16x32Core.r3_perm
_TmemTranspose16x32Core.r3_store
_TmemTranspose16x32Core.r4_load_bot
_TmemTranspose16x32Core.r4_load_top
_TmemTranspose16x32Core.r4_perm
_TmemTranspose16x32Core.r4_store
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/topk_reduce.py`

库存函数 15；已覆盖 6；未覆盖 9。

```text
TopkReduce.__call__
TopkReduce._fma
TopkReduce._fmul
TopkReduce._reduce_bf16_internode
TopkReduce._reduce_bf16_ll
TopkReduce._reduce_cell
TopkReduce._reduce_persistent
TopkReduce._reduce_static
cvt_e2m1_to_fp32_optimal_ptx
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/utils.py`

库存函数 12；已覆盖 11；未覆盖 1。

```text
product
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/custom_mix_cga_helpers.py`

库存函数 14；已覆盖 10；未覆盖 4。

```text
PipelineTmaUmmaMixedCga.consumer_release
PipelineTmaUmmaMixedCga.create
PipelineTmaUmmaMixedCga.producer_acquire
PipelineTmaUmmaMixedCga.producer_commit
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_epilogue.py`

库存函数 121；已覆盖 74；未覆盖 47。

```text
Fc2OutputRouter.get_data_dst
Fc2OutputRouter.get_sf_dst
Fc2OutputRouter.prefetch
QuantImpl.__call__
QuantImpl._amax_lane
QuantImpl._amax_thread_bf16
QuantImpl._amax_thread_bf16.<locals>.max_abs
QuantImpl._amax_thread_fp32
QuantImpl._amax_thread_fp32.<locals>.max_abs
QuantImpl._concat_blocks_ssa
QuantImpl._concat_i32_blocks_ssa
QuantImpl._e8m0
QuantImpl._enc_mxfp8
QuantImpl._enc_nvfp4
QuantImpl._scale_to_data_ssa
QuantImpl._store_packed_blocks
QuantImpl._store_packed_data
QuantImpl._values_to_ssa
QuantImpl.mx_quant_regs_in_pair_threads_impl
QuantImpl.mxfp8_combine_quant_regs_in_thread_impl
QuantImpl.mxfp8_combine_quant_threads_with_the_same_reg_impl
QuantImpl.nvfp4_combine_quant_regs_in_thread_impl
QuantImpl.nvfp4_combine_quant_threads_with_the_same_reg_impl
QuantImpl.nvfp4_quant_impl
SwapABFc1Epilogue.__call__
SwapABFc1Epilogue._done_flag_address
SwapABFc1Epilogue.alpha_swiglu_clamp
SwapABFc1Epilogue.alpha_swiglu_clamp.<locals>._sigmoid
SwapABFc1Epilogue.fc1_quant
SwapABFc1Epilogue.prefetch_act_scales
SwapABFc1Epilogue.run_subtile
SwapABFc1Epilogue.signal_fc1_done
SwapABFc1Epilogue.signal_fc1_done_now
SwapABFc1Epilogue.stage_act_scales
SwapABFc2Epilogue.__call__
SwapABFc2Epilogue._make_output_router
SwapABFc2Epilogue.run_subtile
SwapABFc2Epilogue.signal_fc2_done
SwapABGatedActEpilogue.run
TmemTranspose32x64B16Movm.__call__
TmemTranspose32x64B16Movm.__init__
_TmemTranspose16x32Core.r1_load
_TmemTranspose16x32Core.r4_store
fc2_redg_post_f2fp_reorder
fc2_redg_store_function
make_fc2_redg_cta_store_out_mapping
make_fc2_redg_process_pipeline
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_extension.py`

库存函数 8；已覆盖 2；未覆盖 6。

```text
BlockScaledSwapAbFc12Extension.__extract_mlir_values__
BlockScaledSwapAbFc12Extension.__new_from_mlir_values__
BlockScaledSwapAbFc12Extension.__new_from_mlir_values__.<locals>.rebuild
BlockScaledSwapAbFc12Extension.get_gmem_tensor
BlockScaledSwapAbFc12Extension.prepare_work_tile
BlockScaledSwapAbFc12Extension.wait_for_input
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_kernel.py`

库存函数 13；已覆盖 8；未覆盖 5。

```text
BlockScaledSwapAbFc12Kernel.__call__
BlockScaledSwapAbFc12Kernel.__call__.<locals>.rewrite_tensor_shape
BlockScaledSwapAbFc12Kernel._kernel
BlockScaledSwapAbFc12Kernel.aot_compile
BlockScaledSwapAbFc12Kernel.name
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_mainloop.py`

库存函数 28；已覆盖 18；未覆盖 10。

```text
BlockScaledSwapAbFc12Mainloop._is_shared_tile
BlockScaledSwapAbFc12Mainloop._load_shared_token_tile
BlockScaledSwapAbFc12Mainloop._load_shared_weight_tile
BlockScaledSwapAbFc12Mainloop.assign_device_members
BlockScaledSwapAbFc12Mainloop.create_ab_pipeline
BlockScaledSwapAbFc12Mainloop.create_acc_pipeline
BlockScaledSwapAbFc12Mainloop.create_tmem_allocator
BlockScaledSwapAbFc12Mainloop.run_mma
BlockScaledSwapAbFc12Mainloop.run_tma_a
BlockScaledSwapAbFc12Mainloop.run_tma_b
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py`

库存函数 26；已覆盖 10；未覆盖 16。

```text
BlockScaledSwapAbMegaMoeKernel.__call__
BlockScaledSwapAbMegaMoeKernel.__call__.<locals>.rewrite_tensor_shape
BlockScaledSwapAbMegaMoeKernel._aot_symbol_name
BlockScaledSwapAbMegaMoeKernel._kernel
BlockScaledSwapAbMegaMoeKernel._shared_expert_tensors
BlockScaledSwapAbMegaMoeKernel._shared_expert_tensors.<locals>.k_major
BlockScaledSwapAbMegaMoeKernel._shared_expert_tensors.<locals>.scale_factors
BlockScaledSwapAbMegaMoeKernel._shared_name_suffix
BlockScaledSwapAbMegaMoeKernel.aot_compile
BlockScaledSwapAbMegaMoeKernel.aot_compile.<locals>.fake_tensor
BlockScaledSwapAbMegaMoeKernel.load_compiled
BlockScaledSwapAbMegaMoeKernel.name
BlockScaledSwapAbMegaMoeKernel.name.<locals>.dtype_name
BlockScaledSwapAbMegaMoeKernel.skip_topk_reduce_region
BlockScaledSwapAbMegaMoeKernel.topk_reduce_input_region
_internode_component
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_epilogue.py`

库存函数 114；已覆盖 55；未覆盖 59。

```text
Fc2OutputRouter.get_data_dst
Fc2OutputRouter.get_sf_dst
Fc2OutputRouter.prefetch
Fc2OutputRouter.sf_mapping
Fc2OutputRouter.sf_output
Fc2OutputRouter.sf_token_base
QuantImpl.__call__
QuantImpl._amax_lane
QuantImpl._amax_thread_bf16
QuantImpl._amax_thread_bf16.<locals>.max_abs
QuantImpl._amax_thread_fp32
QuantImpl._amax_thread_fp32.<locals>.max_abs
QuantImpl._concat_blocks_ssa
QuantImpl._concat_i32_blocks_ssa
QuantImpl._e8m0
QuantImpl._enc_mxfp8
QuantImpl._enc_nvfp4
QuantImpl._scale_to_data_ssa
QuantImpl._store_packed_blocks
QuantImpl._store_packed_data
QuantImpl._values_to_ssa
QuantImpl.mx_quant_regs_in_pair_threads_impl
QuantImpl.mxfp8_combine_quant_regs_in_thread_impl
QuantImpl.mxfp8_combine_quant_threads_with_the_same_reg_impl
QuantImpl.nvfp4_combine_quant_regs_in_thread_impl
QuantImpl.nvfp4_combine_quant_threads_with_the_same_reg_impl
QuantImpl.nvfp4_quant_impl
QuantImpl.scale_dtype
SwapABFc1Epilogue.__call__
SwapABFc1Epilogue.alpha_swiglu_clamp
SwapABFc1Epilogue.alpha_swiglu_clamp.<locals>._sigmoid
SwapABFc1Epilogue.fc1_quant
SwapABFc1Epilogue.run_subtile
SwapABFc1Epilogue.signal_fc1_done
SwapABFc2Epilogue.__call__
SwapABFc2Epilogue._make_output_router
SwapABFc2Epilogue.run_subtile
SwapABFc2Epilogue.signal_fc2_done
SwapABGatedActEpilogue.fc2_tma_staged_smem_layout
SwapABGatedActEpilogue.fc2_tma_staged_smem_spec
SwapABGatedActEpilogue.run
TmemTranspose32x64B16Movm.__call__
TmemTranspose32x64B16Movm.__init__
TmemTranspose32x64B16Movm._rmem_copy_view
TmemTranspose32x64B16Movm._tmem_layout
_TmemTranspose16x32Core.r1_load
_TmemTranspose16x32Core.r4_store
fc2_redg_post_f2fp_reorder
fc2_redg_store_function
fc2_stg_post_f2fp_reorder
fc2_stg_store_function
fc2_stg_tmem_acc_load
fc2_tma_store_function
make_fc2_redg_cta_store_out_mapping
make_fc2_redg_process_pipeline
make_fc2_stg_cta_store_out_mapping
make_fc2_stg_cta_store_out_mapping.<locals>.stg_sf_mapping
make_fc2_stg_process_pipeline
make_fc2_tma_process_pipeline
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_extension.py`

库存函数 11；已覆盖 2；未覆盖 9。

```text
BlockScaledSwapAbFc12Extension.__extract_mlir_values__
BlockScaledSwapAbFc12Extension.__new_from_mlir_values__
BlockScaledSwapAbFc12Extension.__new_from_mlir_values__.<locals>.rebuild
BlockScaledSwapAbFc12Extension._half_cta_tiles
BlockScaledSwapAbFc12Extension.get_gmem_tensor
BlockScaledSwapAbFc12Extension.local_tile_m
BlockScaledSwapAbFc12Extension.prepare_work_tile
BlockScaledSwapAbFc12Extension.wait_for_input
BlockScaledSwapAbFc12Extension.weight_half
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop.py`

库存函数 19；已覆盖 12；未覆盖 7。

```text
BlockScaledSwapAbFc12Mainloop.assign_device_members
BlockScaledSwapAbFc12Mainloop.create_ab_pipeline
BlockScaledSwapAbFc12Mainloop.create_acc_pipeline
BlockScaledSwapAbFc12Mainloop.create_tmem_allocator
BlockScaledSwapAbFc12Mainloop.run_mma
BlockScaledSwapAbFc12Mainloop.run_tma_a
BlockScaledSwapAbFc12Mainloop.run_tma_b
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop_gen_specialized.py`

库存函数 23；已覆盖 12；未覆盖 11。

```text
BlockScaledSwapAbFc12MainloopGenSpecialized.assign_device_members
BlockScaledSwapAbFc12MainloopGenSpecialized.create_ab_pipeline
BlockScaledSwapAbFc12MainloopGenSpecialized.create_acc_pipeline
BlockScaledSwapAbFc12MainloopGenSpecialized.create_tmem_allocator
BlockScaledSwapAbFc12MainloopGenSpecialized.create_token_pipeline
BlockScaledSwapAbFc12MainloopGenSpecialized.create_weight_pipeline
BlockScaledSwapAbFc12MainloopGenSpecialized.run_mma
BlockScaledSwapAbFc12MainloopGenSpecialized.run_mma_asymmetric
BlockScaledSwapAbFc12MainloopGenSpecialized.run_tma_a
BlockScaledSwapAbFc12MainloopGenSpecialized.run_tma_b
BlockScaledSwapAbFc12MainloopGenSpecialized.run_tma_gather_b1
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py`

库存函数 28；已覆盖 12；未覆盖 16。

```text
BlockScaledSwapAbMegaMoeKernel.__call__
BlockScaledSwapAbMegaMoeKernel.__call__.<locals>.rewrite_tensor_shape
BlockScaledSwapAbMegaMoeKernel._aot_symbol_name
BlockScaledSwapAbMegaMoeKernel._build_comm_cta_and_smem
BlockScaledSwapAbMegaMoeKernel._combine_cta_ordinal
BlockScaledSwapAbMegaMoeKernel._compute_cluster_ordinal
BlockScaledSwapAbMegaMoeKernel._dispatch_cta_ordinal
BlockScaledSwapAbMegaMoeKernel._is_dispatch_cta
BlockScaledSwapAbMegaMoeKernel._kernel
BlockScaledSwapAbMegaMoeKernel._kernel_cta_specialized
BlockScaledSwapAbMegaMoeKernel._validate_token_tile_ready_contract
BlockScaledSwapAbMegaMoeKernel.aot_compile
BlockScaledSwapAbMegaMoeKernel.aot_compile.<locals>.fake_tensor
BlockScaledSwapAbMegaMoeKernel.load_compiled
BlockScaledSwapAbMegaMoeKernel.name
BlockScaledSwapAbMegaMoeKernel.name.<locals>.dtype_name
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py`

库存函数 21；已覆盖 10；未覆盖 11。

```text
BlockScaledSwapAbGenphaseMoeKernel.__call__
BlockScaledSwapAbGenphaseMoeKernel.__call__.<locals>.rewrite_tensor_shape
BlockScaledSwapAbGenphaseMoeKernel._aot_symbol_name
BlockScaledSwapAbGenphaseMoeKernel._die_from_smid_mask
BlockScaledSwapAbGenphaseMoeKernel._kernel
BlockScaledSwapAbGenphaseMoeKernel._prefetch_first_waves
BlockScaledSwapAbGenphaseMoeKernel.aot_compile
BlockScaledSwapAbGenphaseMoeKernel.aot_compile.<locals>.fake_tensor
BlockScaledSwapAbGenphaseMoeKernel.load_compiled
BlockScaledSwapAbGenphaseMoeKernel.name
BlockScaledSwapAbGenphaseMoeKernel.name.<locals>.dtype_name
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/genphase_tail.py`

库存函数 9；已覆盖 2；未覆盖 7。

```text
GenphaseTail.arrive
GenphaseTail.cross_rank_barrier
GenphaseTail.previous_done_observation
GenphaseTail.read_epoch
GenphaseTail.reset_tail_space
GenphaseTail.signal_done
GenphaseTail.wait_previous_done
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/token_comm_gen_specialized.py`

库存函数 122；已覆盖 61；未覆盖 61。

```text
GenphaseTokenComm._block_row_source
GenphaseTokenComm._block_row_source_plain
GenphaseTokenComm._call_tag
GenphaseTokenComm._canonicalize_wire_indices
GenphaseTokenComm._claim_next_block
GenphaseTokenComm._classify_tokens
GenphaseTokenComm._is_unowned_expert
GenphaseTokenComm._kernel
GenphaseTokenComm._leave
GenphaseTokenComm._ll_half_address
GenphaseTokenComm._ll_parity
GenphaseTokenComm._payload_plane_by_dense_row
GenphaseTokenComm._place_expert_at_slot
GenphaseTokenComm._preload_first_row
GenphaseTokenComm._publish_padded_prefix
GenphaseTokenComm._publish_payload_flags
GenphaseTokenComm._publish_payload_once
GenphaseTokenComm._publish_refine_block_prefix
GenphaseTokenComm._publish_source_segment_bases
GenphaseTokenComm._push_one_plane
GenphaseTokenComm._push_sf_ll
GenphaseTokenComm._rank_by_counting_in_warp
GenphaseTokenComm._rank_by_load_histogram
GenphaseTokenComm._rank_experts_by_load
GenphaseTokenComm._refine_one_block
GenphaseTokenComm._refine_one_block_plain
GenphaseTokenComm._resolve_block_expert
GenphaseTokenComm._run_exchange
GenphaseTokenComm._run_helper
GenphaseTokenComm._run_pusher
GenphaseTokenComm._run_refine
GenphaseTokenComm._run_sort
GenphaseTokenComm._scatter_block_atoms
GenphaseTokenComm._scatter_block_atoms_plain
GenphaseTokenComm._stage_block_rows
GenphaseTokenComm._stage_block_rows_plain
GenphaseTokenComm._stage_refine_expert_table
GenphaseTokenComm._wire_index_view
GenphaseTokenComm.canonical_activation_sf_tensor
GenphaseTokenComm.dense_activation_sf_tensor
GenphaseTokenComm.dense_activation_tensor
GenphaseTokenComm.dense_sf_ll_tensor
GenphaseTokenComm.expert_sizes_tensor
GenphaseTokenComm.fc1_topk_scores_tensor
GenphaseTokenComm.gather_index_tensor
GenphaseTokenComm.launch
GenphaseTokenComm.plain_activation_sf_tensor
GenphaseTokenComm.plain_activation_tensor
GenphaseTokenComm.pusher_global_warp_count
GenphaseTokenComm.pusher_tokens_per_warp
GenphaseTokenComm.scheduler_expert_sizes_tensor
GenphaseTokenComm.sf_ll_words_per_lane
GenphaseTokenComm.sf_ready_target
GenphaseTokenComm.slot_to_expert_tensor
GenphaseTokenComm.smem_bytes
GenphaseTokenComm.smem_workspace
GenphaseTokenComm.token_src_metadata_tensor
GenphaseTokenComm.topk_index_inbox_tensor
GenphaseTokenComm.topk_score_inbox_tensor
GenphaseTokenComm.worst_case_refine_blocks
_int32_bits_to_float32
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mxfp8_fc12/fwd_glu/glu_mxfp8_fc12_epilogue.py`

库存函数 77；已覆盖 20；未覆盖 57。

```text
Fc2OutputDest.__post_init__
Fc2OutputDest.resolve_token_row
Fc2OutputRouter.get_data_addr
Fc2OutputRouter.get_data_addr_from_coord
Fc2OutputRouter.get_sf_addr
Fc2OutputRouter.prefetch
Fc2TmaTileRouter.get_data_tile
GluMxfp8Epilogue._acc_pipeline_consumer_release
GluMxfp8Epilogue._advance_fc2_tmem_tensor
GluMxfp8Epilogue._begin_fc2_task_tile
GluMxfp8Epilogue._c_store_bar_id
GluMxfp8Epilogue._drain_bulk_stores
GluMxfp8Epilogue._fc1_subtile_compute_store
GluMxfp8Epilogue._fc1_subtile_load
GluMxfp8Epilogue._fc2_lean_mxfp8_store
GluMxfp8Epilogue._fc2_subtile_cols
GluMxfp8Epilogue._fc2_subtile_load
GluMxfp8Epilogue._finish_fc2_task_tile
GluMxfp8Epilogue._get_task_tile
GluMxfp8Epilogue._issue_fc2_reduce_coalesced
GluMxfp8Epilogue._issue_fc2_tma_store
GluMxfp8Epilogue._make_fc2_output_router
GluMxfp8Epilogue._run_fc1_subtile
GluMxfp8Epilogue._run_fc1_subtiles
GluMxfp8Epilogue._run_fc1_task_tile
GluMxfp8Epilogue._run_fc2_subtile
GluMxfp8Epilogue._run_fc2_task_tile
GluMxfp8Epilogue._situ_act
GluMxfp8Epilogue._situ_act.<locals>._sigmoid
GluMxfp8Epilogue._stg_sf_fc1
GluMxfp8Epilogue._stg_sf_fc2
GluMxfp8Epilogue._stg_sf_fc2_blocked
GluMxfp8Epilogue._stg_sf_fc2_flat
GluMxfp8Epilogue._stg_sf_row_group
GluMxfp8Epilogue._store_fc1_c_subtile
GluMxfp8Epilogue._subtile_fc2_tmem_tensor
GluMxfp8Epilogue._swiglu_act
GluMxfp8Epilogue._validate_tile_shape
GluMxfp8Epilogue._write_sf_fc2_buffer
GluMxfp8Epilogue.c_bytes_per_stage
GluMxfp8Epilogue.c_smem_layout_one_stage
GluMxfp8Epilogue.cta_tile_n
GluMxfp8Epilogue.epi_tile_c
GluMxfp8Epilogue.fc1_c_scratch_offset
GluMxfp8Epilogue.fc2_reduce_smem_layout
GluMxfp8Epilogue.fc2_reduce_staging_bytes
GluMxfp8Epilogue.fc2_tma_smem_layout_one_stage
GluMxfp8Epilogue.fc2_tma_staged_smem_layout
GluMxfp8Epilogue.fc2_tma_stages
GluMxfp8Epilogue.fc2_tma_staging_bytes
GluMxfp8Epilogue.fc2_tma_tile
GluMxfp8Epilogue.fc2_use_ublk
GluMxfp8Epilogue.run
GluMxfp8Epilogue.staged_c_smem_layout
GluMxfp8Epilogue.subtile_cnt
GluMxfp8Epilogue.tma_store_fc1_output
make_fc2_ublk_store_out_mapping
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mxfp8_fc12/fwd_glu/glu_mxfp8_fc12_epilogue_breuse.py`

库存函数 9；已覆盖 4；未覆盖 5。

```text
GluMxfp8BreuseEpilogue._c_store_bar_id
GluMxfp8BreuseEpilogue._run_fc1_subtiles
GluMxfp8BreuseEpilogue._run_fc2_task_tile
GluMxfp8BreuseEpilogue._run_fc2_task_tile_drain
GluMxfp8BreuseEpilogue.staged_c_smem_layout
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mxfp8_fc12/fwd_glu/glu_mxfp8_fc12_extension.py`

库存函数 23；已覆盖 3；未覆盖 20。

```text
GluMxFp8Fc12SchedExtension.__extract_mlir_values__
GluMxFp8Fc12SchedExtension.__new_from_mlir_values__
GluMxFp8Fc12SchedExtension.__new_from_mlir_values__.<locals>.rebuild
GluMxFp8Fc12SchedExtension._counter_slot
GluMxFp8Fc12SchedExtension._discrete_weight_descriptor
GluMxFp8Fc12SchedExtension._done_counter_slot
GluMxFp8Fc12SchedExtension.get_gmem_tensor
GluMxFp8Fc12SchedExtension.prepare_work_tile
GluMxFp8Fc12SchedExtension.wait_for_input
MixedCgaPipeline.consumer_release
MixedCgaPipeline.producer_acquire
MixedCgaPipeline.producer_commit
OneShotSchedConsumer.__extract_mlir_values__
OneShotSchedConsumer.__init__
OneShotSchedConsumer.__new_from_mlir_values__
OneShotSchedConsumer.consume_work
copy_discrete_weight_descriptor_pointer
discrete_weight_descriptor_workspace_size
make_mixed_cga_tma_umma_pipeline
raw_discrete_weight_descriptor_pointer
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mxfp8_fc12/fwd_glu/glu_mxfp8_fc12_kernel.py`

库存函数 33；已覆盖 15；未覆盖 18。

```text
Sm107Mxfp8GluFc12Kernel.__call__
Sm107Mxfp8GluFc12Kernel.__call__.<locals>.make_tma_b_maybe_pair
Sm107Mxfp8GluFc12Kernel._initialize_discrete_weight_descriptors
Sm107Mxfp8GluFc12Kernel.get_weight_descriptor_workspace_offset_in_bytes
Sm107Mxfp8GluFc12Kernel.get_weight_descriptor_workspace_size_in_bytes
Sm107Mxfp8GluFc12Kernel.kernel
Sm107Mxfp8GluFc12Kernel.load_balance_counter_is_undersized
Sm107Mxfp8GluFc12Kernel.load_balance_counter_is_undersized.<locals>._flatten
Sm107Mxfp8GluFc12Kernel.make_weight_descriptor_workspace_pointer
Sm107Mxfp8GluFc12Kernel.one_shot_tile_count
Sm107Mxfp8GluFc12Kernel.sched_ext_fc1_counter_cumul_scale
Sm107Mxfp8GluFc12Kernel.sched_ext_fc1_peek_threshold
Sm107Mxfp8GluFc12Kernel.token_comm_hook_dispatch_warp_body
Sm107Mxfp8GluFc12Kernel.token_comm_hook_fc1_tma_b_predispatch_spin
Sm107Mxfp8GluFc12Kernel.token_comm_hook_kernel_tail
Sm107Mxfp8GluFc12Kernel.token_comm_hook_sched_warp_pre_init_wait
Sm107Mxfp8GluFc12Kernel.token_comm_hook_token_back_warp_body
_int64_pointer_array_tensor
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mxfp8_fc12/helpers/utils.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
swiglu_act
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/base.py`

库存函数 21；已覆盖 9；未覆盖 12。

```text
SchedulerBase.assign_device_members
SchedulerBase.create_scheduler_pipelines
SchedulerBase.gen_next_work
SchedulerBase.get_grid_shape
SchedulerBase.problem_desc_require
SchedulerBase.produce_tail
SchedulerBase.publish_work
SchedulerBase.register_device_workspace
SchedulerConsumer.consume_work
SchedulerWorkTileBase.from_rmem
SchedulerWorkTileBase.is_valid_tile
SchedulerWorkTileBase.to_rmem
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/fc12_mapping.py`

库存函数 59；已覆盖 54；未覆盖 5。

```text
NonSwapAbFc12WorkTileInfo.peek_ready
NonSwapAbFc12WorkTileInfo.valid_tokens_in_cluster_tile
_owner_owns_slot
reset_phase_interleaved_cursors_for_owner
rewind_fc12_mapping_cursor
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/fc12_scheduler.py`

库存函数 44；已覆盖 32；未覆盖 12。

```text
BlackwellFusedFc12Scheduler.assign_device_members
BlackwellFusedFc12Scheduler.clc_ticket_count
BlackwellFusedFc12Scheduler.gen_next_work
BlackwellFusedFc12Scheduler.initialize_fallback_group
BlackwellFusedFc12Scheduler.produce_tail
PhaseInterleavedFc12Scheduler._wait_for_fc1_prologue_claims
PhaseInterleavedFc12Scheduler.assign_device_members
PhaseInterleavedFc12Scheduler.cluster_uniform_counter_ready
PhaseInterleavedFc12Scheduler.gen_next_linear1_work
PhaseInterleavedFc12Scheduler.gen_next_linear1_work_unless_counter_ready
PhaseInterleavedFc12Scheduler.gen_next_work
PhaseInterleavedFc12Scheduler.initialize_fallback_group
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/non_clc_mixed_cga.py`

库存函数 19；已覆盖 11；未覆盖 8。

```text
NonClcMixedCgaSchedulerWorker._preferred_cluster_cta_coord
NonClcMixedCgaSchedulerWorker.assign_device_members
NonClcMixedCgaSchedulerWorker.claim_next_work
NonClcMixedCgaSchedulerWorker.cluster_uniform_counter_ready
NonClcMixedCgaSchedulerWorker.cluster_uniform_counter_ready_or_claim
NonClcMixedCgaSchedulerWorker.finish_claim_channel
NonClcMixedCgaSchedulerWorker.get_atomic_counter_pointer
NonClcMixedCgaSchedulerWorker.initialize_fallback_group
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/shared_fc12_scheduler.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
SharedExpertTileSchedule.publish
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/work_id_claim.py`

库存函数 31；已覆盖 17；未覆盖 14。

```text
ClcMixedCgaWorkIdState.__extract_mlir_values__
ClcMixedCgaWorkIdState.__init__
ClcMixedCgaWorkIdState.__new_from_mlir_values__
ClcMixedCgaWorkIdState.__new_from_mlir_values__.<locals>.rebuild
ClusterLaunchControlWorkIdState.__extract_mlir_values__
ClusterLaunchControlWorkIdState.__init__
ClusterLaunchControlWorkIdState.__new_from_mlir_values__
ClusterLaunchControlWorkIdState.__new_from_mlir_values__.<locals>.rebuild
GridWorkId.__extract_mlir_values__
GridWorkId.__new_from_mlir_values__
_claim_clc_mixed_cga_work_id
_claim_cluster_launch_control_work_id
broadcast_atomic_counter_ready
broadcast_atomic_counter_ready_or_claim
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/quant_def.py`

库存函数 19；已覆盖 17；未覆盖 2。

```text
CombineFormat.__str__
CombineFormat.name
```

### `tensorrt_llm/_torch/moe/kernels/megamoe_scheduler/cuda_scheduler/runtime.py`

库存函数 20；已覆盖 2；未覆盖 18。

```text
CudaPhysicalSlotScheduler.__init__
CudaPhysicalSlotScheduler.bind_gpu_direct
CudaPhysicalSlotScheduler.check_status
CudaPhysicalSlotScheduler.connect_peer_bases
CudaPhysicalSlotScheduler.reset_status
CudaSchedulerConfig.__post_init__
CudaSchedulerConfig.global_expert_count
CudaSchedulerConfig.group_sizes
CudaSchedulerConfig.helper_count
CudaSchedulerConfig.home_experts_per_rank
CudaSchedulerConfig.local_physical_slot_count
CudaSchedulerConfig.route_count
CudaSchedulerConfig.validate
CudaSchedulerConfig.world_size
CudaSchedulerOutputs.__init__
_stream_handle
recommend_cuda_scheduler_ctas
symmetric_buffer_ints
```

### `tensorrt_llm/_torch/moe/kernels/megamoe_scheduler/sami/geometry.py`

库存函数 11；已覆盖 10；未覆盖 1。

```text
BundleLayout.plane_count
```

### `tensorrt_llm/_torch/moe/kernels/megamoe_scheduler/sami/hierarchical.py`

库存函数 16；已覆盖 5；未覆盖 11。

```text
HierarchicalLiveWeightArena.bind
HierarchicalLiveWeightArenaProvider.build_hierarchical_live_arena
HierarchicalSamiWeightBroadcast.__init__
HierarchicalSamiWeightBroadcast._validate_reuse_authority
HierarchicalSamiWeightBroadcast._validate_stream_context
HierarchicalSamiWeightBroadcast._validate_tma_warp_capacity
HierarchicalSamiWeightBroadcast.bind_generation_reuse_authority
HierarchicalSamiWeightBroadcast.bind_scheduler
HierarchicalSamiWeightBroadcast.close
HierarchicalSamiWeightBroadcast.last_mode_counts
_stream_handle
```

### `tensorrt_llm/_torch/moe/kernels/megamoe_shared_fc12.py`

库存函数 23；已覆盖 22；未覆盖 1。

```text
shared_fc12_cache_info
```

### `tensorrt_llm/_torch/moe/kernels/rubin/custom_pipeline.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
PipelineCpAsyncUmma._compute_is_leader_cta
```

### `tensorrt_llm/_torch/moe/kernels/rubin/rubin_contiguous_gather_grouped_blockscaled_gemm_act_fusion.py`

库存函数 61；已覆盖 21；未覆盖 40。

```text
Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel.__call__
Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel._apply_relu2_epilogue
Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel._apply_situ_epilogue
Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel._apply_situ_epilogue.<locals>._sigmoid
Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel._apply_swiglu_epilogue
Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel._get_tma_atom_kind
Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel._is_interleaved_utccp
Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel._mainloop_s2t_interleaved_copies
Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel.epilog_smem_copy_and_partition
Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel.kernel
Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel.mainloop_s2t_copy_and_partition
Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel.wrapper
create_mask
create_scale_factor_tensor
create_scale_factor_tensor.<locals>.ceil_div
create_scale_factor_tensor_unswizzled
create_scale_factor_tensor_unswizzled.<locals>.ceil_div
create_sf_layout_tensor
create_sf_layout_tensor.<locals>.ceil_div
create_tensors
create_token_id_mapping_tensor
cvt_sf_M32x4xrm_K4xrk_L_to_MKL
cvt_sf_MKL_to_M32x4xrm_K4xrk_L
fclip_xorsign
main
parse_benchmark_arg
parse_comma_separated_ints
read_benchmark_file
run
run.<locals>.apply_quantization_scale
run.<locals>.ceil_div
run.<locals>.compute_scale_factor
run.<locals>.generate_tensors
run.<locals>.simulate_f8_quantization
run.<locals>.simulate_nvfp4_quantization
run.<locals>.unswizzle_kernel_sfc
sigmoid_f32
silu_f32
store_output_halves
store_sfc_word
```

### `tensorrt_llm/_torch/moe/kernels/rubin/rubin_contiguous_gather_grouped_gemm_swiglu_fusion.py`

库存函数 27；已覆盖 6；未覆盖 21。

```text
Sm107ContiguousGatherGroupedGemmSwigluFusionKernel.__call__
Sm107ContiguousGatherGroupedGemmSwigluFusionKernel._get_tma_atom_kind
Sm107ContiguousGatherGroupedGemmSwigluFusionKernel.can_implement
Sm107ContiguousGatherGroupedGemmSwigluFusionKernel.epilog_smem_copy_and_partition
Sm107ContiguousGatherGroupedGemmSwigluFusionKernel.is_valid_dtypes
Sm107ContiguousGatherGroupedGemmSwigluFusionKernel.is_valid_layouts
Sm107ContiguousGatherGroupedGemmSwigluFusionKernel.is_valid_mma_tiler_and_cluster_shape
Sm107ContiguousGatherGroupedGemmSwigluFusionKernel.is_valid_mma_tiler_and_cluster_shape.<locals>._is_power_of_2
Sm107ContiguousGatherGroupedGemmSwigluFusionKernel.is_valid_tensor_alignment
Sm107ContiguousGatherGroupedGemmSwigluFusionKernel.is_valid_tensor_alignment.<locals>.check_contigous_16B_alignment
Sm107ContiguousGatherGroupedGemmSwigluFusionKernel.kernel
Sm107ContiguousGatherGroupedGemmSwigluFusionKernel.wrapper
create_mask
create_tensors
create_token_id_mapping_tensor
main
parse_benchmark_arg
parse_comma_separated_ints
read_benchmark_file
run
run.<locals>.generate_tensors
```

### `tensorrt_llm/_torch/moe/kernels/rubin/rubin_contiguous_grouped_blockscaled_gemm_finalize_fusion.py`

库存函数 36；已覆盖 17；未覆盖 19。

```text
Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel.__call__
Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel._mainloop_s2t_interleaved_copies
Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel.kernel
Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel.wrapper
blk_reduce_fp16
blk_reduce_fp32
create_fused_finalize_tensors
create_mask
create_scale_factor_tensor
create_scale_factor_tensor.<locals>.ceil_div
create_tensors
cvt_sf_MKL_to_M32x4xrm_K4xrk_L
main
parse_benchmark_arg
parse_comma_separated_ints
read_benchmark_file
run
run.<locals>.generate_tensors
verify_reference_result
```

### `tensorrt_llm/_torch/moe/kernels/rubin/rubin_contiguous_grouped_blockscaled_gemm_fused_fc12.py`

库存函数 69；已覆盖 46；未覆盖 23。

```text
L2AtomicPersistentTileScheduler._claim_next_linear_idx
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.__call__
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._apply_situ_epilogue
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._apply_situ_epilogue.<locals>._sigmoid
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._apply_swiglu_epilogue
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._get_tma_atom_kind
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._is_interleaved_utccp
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._mainloop_s2t_interleaved_copies
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.can_implement
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.is_valid_dtypes_and_scale_factor_vec_size
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.is_valid_layouts
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.is_valid_mma_tiler_and_cluster_shape
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.is_valid_problem_shapes
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.is_valid_problem_shapes.<locals>.is_contiguous_16b_aligned
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.kernel
derive_fc12_cta_task_stream
derive_fc12_task_shape
derive_fc1_ready_expected
fclip_xorsign
sigmoid_f32
silu_f32
sm100_tcgen05_st_32x32b_x4
validate_l2_atomic_descriptor_bounds
```

### `tensorrt_llm/_torch/moe/kernels/rubin/rubin_contiguous_grouped_blockscaled_gemm_fused_fc12_mxfp8.py`

库存函数 60；已覆盖 46；未覆盖 14。

```text
L2AtomicPersistentTileScheduler._claim_next_linear_idx
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.__call__
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._get_tma_atom_kind
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._is_interleaved_utccp
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._mainloop_s2t_interleaved_copies
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.kernel
Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.mma_k_shape_for_dtype
derive_fc12_cta_task_stream
derive_fc12_task_shape
derive_fc1_ready_expected
fclip_xorsign
sigmoid_f32
silu_f32
sm100_tcgen05_st_32x32b_x4
```

### `tensorrt_llm/_torch/moe/kernels/rubin/rubin_contiguous_grouped_gemm_finalize_fusion.py`

库存函数 27；已覆盖 5；未覆盖 22。

```text
Sm107ContiguousGroupedGemmFinalizeFusionKernel.__call__
Sm107ContiguousGroupedGemmFinalizeFusionKernel.can_implement
Sm107ContiguousGroupedGemmFinalizeFusionKernel.is_valid_dtypes
Sm107ContiguousGroupedGemmFinalizeFusionKernel.is_valid_layouts
Sm107ContiguousGroupedGemmFinalizeFusionKernel.is_valid_mma_tiler_and_cluster_shape
Sm107ContiguousGroupedGemmFinalizeFusionKernel.is_valid_mma_tiler_and_cluster_shape.<locals>._is_power_of_2
Sm107ContiguousGroupedGemmFinalizeFusionKernel.is_valid_tensor_alignment
Sm107ContiguousGroupedGemmFinalizeFusionKernel.is_valid_tensor_alignment.<locals>.check_contigous_16B_alignment
Sm107ContiguousGroupedGemmFinalizeFusionKernel.kernel
Sm107ContiguousGroupedGemmFinalizeFusionKernel.wrapper
blk_reduce_fp16
blk_reduce_fp32
create_fused_finalize_tensors
create_mask
create_tensors
main
parse_benchmark_arg
parse_comma_separated_ints
read_benchmark_file
run
run.<locals>.generate_tensors
verify_reference_result
```

### `tensorrt_llm/_torch/moe/kernels/rubin/utils.py`

库存函数 23；已覆盖 2；未覆盖 21。

```text
_Pointer.__add__
_Pointer.__c_pointers__
_Pointer.__get_mlir_types__
_Pointer.__init__
_Pointer.__new_from_mlir_values__
_Pointer.__repr__
_Pointer.__str__
_Pointer.align
_Pointer.dtype
_Pointer.memspace
_Pointer.mlir_type
_Pointer.size_in_bytes
_Pointer.verify
atomic_add_func
fmin
is_power_of_2
make_ptr
sigmoid_f32
silu_f32
vectorized_atomic_add_bf16x8
vectorized_atomic_add_fp32x2
```

### `tensorrt_llm/_torch/peft/lora/adapter_slot_manager.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
AdapterSlotManager.remove_task
```

### `tensorrt_llm/_torch/peft/lora/cuda_graph_lora_params.py`

库存函数 16；已覆盖 15；未覆盖 1。

```text
CudaGraphLoraParams.LoraLayerInfo.is_enabled
```

### `tensorrt_llm/_torch/peft/lora/layer.py`

库存函数 26；已覆盖 19；未覆盖 7。

```text
LoraLayer.prepare_grouped_gemm_buffers
LoraModuleType.__str__
LoraModuleType.is_attention
LoraModuleType.is_mamba
LoraModuleType.is_mlp
LoraModuleType.is_moe
MoeLoraLayer.forward
```

### `tensorrt_llm/_torch/peft/lora/loaders.py`

库存函数 20；已覆盖 19；未覆盖 1。

```text
norm_dora_magnitude
```

### `tensorrt_llm/_torch/peft/lora/manager.py`

库存函数 33；已覆盖 25；未覆盖 8。

```text
LoraManager._generate_uid
LoraManager.input_buffers
LoraManager.is_adapter_in_cpu_cache
LoraManager.load_from_hf.<locals>.interleave_fused_lora_weights_for_tp
LoraManager.lora_weights_pointers_list
LoraManager.num_lora_adapters
LoraManager.save_lora_weights_to_bin
LoraManager.save_lora_weights_to_bin.<locals>.save_val
```

### `tensorrt_llm/_torch/pyexecutor/breakable_cuda_graph/breakable_cuda_graph.py`

库存函数 28；已覆盖 26；未覆盖 2。

```text
BreakableCUDAGraph.pool
get_current_replay_token
```

### `tensorrt_llm/_torch/pyexecutor/config_utils.py`

库存函数 49；已覆盖 48；未覆盖 1。

```text
get_glm5_next_num_attention_layers
```

### `tensorrt_llm/_torch/pyexecutor/connectors/kv_cache_connector.py`

库存函数 85；已覆盖 71；未覆盖 14。

```text
KvCacheConnectorScheduler.build_connector_meta
KvCacheConnectorScheduler.get_num_new_matched_tokens
KvCacheConnectorScheduler.release_prefix_reservation
KvCacheConnectorScheduler.request_finished
KvCacheConnectorScheduler.reserve_prefix
KvCacheConnectorScheduler.update_state_after_alloc
KvCacheConnectorWorker._clear_connector_meta
KvCacheConnectorWorker.get_finished
KvCacheConnectorWorker.get_finished_prefix_loads
KvCacheConnectorWorker.register_kv_caches
KvCacheConnectorWorker.save_kv_layer
KvCacheConnectorWorker.start_load_kv
KvCacheConnectorWorker.wait_for_layer_load
KvCacheConnectorWorker.wait_for_save
```

### `tensorrt_llm/_torch/pyexecutor/connectors/mooncake_store/addressing.py`

库存函数 14；已覆盖 12；未覆盖 2。

```text
PageAddressing.layout
PageAddressing.num_slots
```

### `tensorrt_llm/_torch/pyexecutor/connectors/mooncake_store/keys.py`

库存函数 10；已覆盖 9；未覆盖 1。

```text
BlockHashChain.tokens_per_block
```

### `tensorrt_llm/_torch/pyexecutor/connectors/mooncake_store/ledger.py`

库存函数 9；已覆盖 4；未覆盖 5。

```text
SegmentRecord.gib
_capacity_lines
_placement_lines
_roles_by_host
format_pool_report
```

### `tensorrt_llm/_torch/pyexecutor/connectors/mooncake_store/master.py`

库存函数 24；已覆盖 21；未覆盖 3。

```text
_claim_is_live
_log_tail
_write_claim
```

### `tensorrt_llm/_torch/pyexecutor/connectors/mooncake_store/worker.py`

库存函数 25；已覆盖 23；未覆盖 2。

```text
MooncakeStoreConnectorWorker.config
MooncakeStoreConnectorWorker.register_kv_caches
```

### `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py`

库存函数 137；已覆盖 134；未覆盖 3。

```text
CUDAGraphRunner._get_or_create_padding_dummy.<locals>.rollback
CUDAGraphRunner.register_sample_type_resolver
CUDAGraphRunner.take_sampling_modes
```

### `tensorrt_llm/_torch/pyexecutor/device_copy.py`

库存函数 78；已覆盖 74；未覆盖 4。

```text
DeviceCopier.merges
PrologueOp.__call__
_graph_scatter_kernel
_parallel_scatter_kernel
```

### `tensorrt_llm/_torch/pyexecutor/dwdp.py`

库存函数 31；已覆盖 28；未覆盖 3。

```text
DwdpManager.is_enabled
DwdpManager.weight_manager
new_expert_copy_stream
```

### `tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/graphs.py`

库存函数 48；已覆盖 45；未覆盖 3。

```text
DecoderCudaGraphs._decode_sm_count
DecoderCudaGraphs.register_sample_type_resolver
ForwardWarmup.__call__
```

### `tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/inputs.py`

库存函数 70；已覆盖 66；未覆盖 4。

```text
DecoderInputPreparer._prepare_helix_state.<locals>._helix_local_len_host
DecoderInputPreparer._prepare_helix_state.<locals>._helix_pack_extend
DecoderInputPreparer._ragged_gather_indices
DecoderInputPreparer._set_locality_domain_splits
```

### `tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/runner.py`

库存函数 60；已覆盖 54；未覆盖 6。

```text
DecoderRunner.has_previous_device_draft
DecoderRunner.previous_request_ids
DecoderRunner.previous_verify_lens
DecoderRunner.ragged_verify_token_buckets
DecoderRunner.register_sample_type_resolver
DecoderRunner.replays_final_windows_only
```

### `tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/spec_overlap_inputs.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_stage_kernel
```

### `tensorrt_llm/_torch/pyexecutor/executor_request_queue.py`

库存函数 41；已覆盖 40；未覆盖 1。

```text
ExecutorRequestQueue.enqueue_benchmark_stage_vote_request
```

### `tensorrt_llm/_torch/pyexecutor/gpu_keepalive.py`

库存函数 25；已覆盖 23；未覆盖 2。

```text
_SelftestChild.kill
_spin_kernel
```

### `tensorrt_llm/_torch/pyexecutor/grammar_matcher.py`

库存函数 20；已覆盖 15；未覆盖 5。

```text
GrammarMatcher.accept_token
GrammarMatcher.fill_next_token_bitmask
GrammarMatcher.is_terminated
GrammarMatcher.rollback
GrammarMatcherFactory.create
```

### `tensorrt_llm/_torch/pyexecutor/guided_decoder.py`

库存函数 48；已覆盖 45；未覆盖 3。

```text
GuidedDecoder.rollback_draft_tokens
GuidedDecoder.rollback_rejected_tokens
GuidedRequests.__iter__
```

### `tensorrt_llm/_torch/pyexecutor/hang_detector.py`

库存函数 31；已覆盖 30；未覆盖 1。

```text
_best_effort_flush_streams
```

### `tensorrt_llm/_torch/pyexecutor/hang_diagnostics.py`

库存函数 11；已覆盖 7；未覆盖 4。

```text
_CudaEvent.query
_CudaEvent.record
_CudaEventFactory.__call__
_StatusProviderRegistry.register_status_provider
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`

库存函数 379；已覆盖 372；未覆盖 7。

```text
KVCacheManagerV2._get_buffer_views_for_invalid_value_check
KVCacheManagerV2._stats_life_cycle_window_size
KVCacheManagerV2.any_remote_level_failed
KVCacheManagerV2.get_locality_domain
KVCacheManagerV2.get_max_resource_count
KVCacheManagerV2.get_needed_resource_to_completion
_disk_tier_init_error
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/mamba_cache_manager/__init__.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
__dir__
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/mamba_cache_manager/common.py`

库存函数 38；已覆盖 21；未覆盖 17。

```text
BaseMambaCacheManager.get_conv_states
BaseMambaCacheManager.get_ssm_states
BaseMambaCacheManager.get_state_indices
BaseMambaCacheManager.is_speculative
BaseMambaCacheManager.mamba_layer_cache
MambaLayerCache.at_layer_idx
MambaState.bind
MambaState.intermediate_conv
MambaState.intermediate_indices
MambaState.intermediate_ssm
MambaState.make_layer_cache
MambaState.reset_slots
MambaState.shutdown
MambaState.update
_PrefixReuseDiagnostics._get_num_reusable_tokens_before_hybrid_pruning
_PrefixReuseDiagnostics._get_num_reusable_tokens_before_pruning
_promote_mamba_state_kernel
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/mamba_cache_manager/legacy.py`

库存函数 113；已覆盖 60；未覆盖 53。

```text
CppMambaHybridCacheManager._commit_gdn_cached_replay_history_layers
CppMambaHybridCacheManager._prepare_resources
CppMambaHybridCacheManager.free_resources
CppMambaHybridCacheManager.update_resources
CppMambaHybridCacheManager.use_gdn_cached_replay_all_layer_commit
MambaCacheManager.filter_ctx_requests_by_capacity
MambaCacheManager.free_resources
MambaCacheManager.get_conv_states
MambaCacheManager.get_intermediate_conv_states
MambaCacheManager.get_intermediate_ssm_states
MambaCacheManager.get_mamba_ssm_cache_dtype
MambaCacheManager.get_mamba_ssm_rand_seed
MambaCacheManager.get_max_resource_count
MambaCacheManager.get_needed_resource_to_completion
MambaCacheManager.get_replay_cache_buf_idx
MambaCacheManager.get_replay_old_B
MambaCacheManager.get_replay_old_dA_cumsum
MambaCacheManager.get_replay_old_dt
MambaCacheManager.get_replay_old_x
MambaCacheManager.get_replay_prev_num_accepted_tokens
MambaCacheManager.get_replay_state_update_metadata
MambaCacheManager.get_ssm_states
MambaCacheManager.mamba_cache_free_blocks
MambaCacheManager.mamba_cache_index
MambaCacheManager.on_state_transfer_complete
MambaCacheManager.seed_kda_replay_caches_for_disagg_gen
MambaCacheManager.update_mamba_states
MambaCacheManager.use_replay_state_update
MixedMambaHybridCacheManager._maybe_promote_drafter_states
MixedMambaHybridCacheManager.free_resources
MixedMambaHybridCacheManager.update_resources
PythonMambaCacheManager.filter_ctx_requests_by_capacity
PythonMambaCacheManager.get_conv_states
PythonMambaCacheManager.get_intermediate_conv_states
PythonMambaCacheManager.get_intermediate_ssm_states
PythonMambaCacheManager.get_mamba_ssm_cache_dtype
PythonMambaCacheManager.get_needed_resource_to_completion
PythonMambaCacheManager.get_replay_cache_buf_idx
PythonMambaCacheManager.get_replay_old_B
PythonMambaCacheManager.get_replay_old_dA_cumsum
PythonMambaCacheManager.get_replay_old_dt
PythonMambaCacheManager.get_replay_old_x
PythonMambaCacheManager.get_replay_prev_num_accepted_tokens
PythonMambaCacheManager.get_ssm_states
PythonMambaCacheManager.on_state_transfer_complete
PythonMambaCacheManager.shutdown.<locals>._drop
_LegacyMambaHybridCacheManager._reset_context_mamba_slots
_LegacyMambaHybridCacheManager.get_intermediate_conv_states
_LegacyMambaHybridCacheManager.get_intermediate_ssm_states
_LegacyMambaHybridCacheManager.get_mamba_ssm_cache_dtype
_LegacyMambaHybridCacheManager.is_speculative
_LegacyMambaHybridCacheManager.use_kda_replay_update
_LegacyMambaHybridCacheManager.use_replay_state_update
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/mixed_precision_staging.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
MixedPrecisionKVStaging.get_attention_op_kv_page_offset
MixedPrecisionKVStaging.get_batch_cache_indices
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/regions.py`

库存函数 17；已覆盖 16；未覆盖 1。

```text
RegionProvider.declare_regions
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/remote_donor.py`

库存函数 48；已覆盖 45；未覆盖 3。

```text
DonorTransport.options_from
DonorTransport.start
_nvml_gpu_index
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/sharing/_lender.py`

库存函数 85；已覆盖 84；未覆盖 1。

```text
_StagingLease.__repr__
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/sharing/_manager.py`

库存函数 44；已覆盖 39；未覆盖 5。

```text
_ManagerHooks._on_caches_closed
_ManagerHooks._on_free
_ManagerHooks._on_reset
_ManagerHooks._on_shrink
_ManagerHooks._on_shutdown
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/sharing/_types.py`

库存函数 19；已覆盖 9；未覆盖 10。

```text
Lease.failure
Lease.mark_arrived
Lease.poll
Lease.release
Lender.hold_regions
Lender.lend_read
Lender.lend_write
Lender.regions
Lender.resume_bounds
RegionHold.release
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache_events.py`

库存函数 38；已覆盖 33；未覆盖 5。

```text
EventPublisher.publish
EventPublisher.shutdown
NullEventPublisher.publish
StreamingKVCacheEventManager.get_latest_events
StreamingKVCacheEventManager.get_source_descriptor
```

### `tensorrt_llm/_torch/pyexecutor/llm_request.py`

库存函数 95；已覆盖 94；未覆盖 1。

```text
_restore_trimmed_prompt
```

### `tensorrt_llm/_torch/pyexecutor/model_engine.py`

库存函数 73；已覆盖 67；未覆盖 6。

```text
ModelEngine.invalidate_multimodal_encoder_cache
ModelEngine.make_graph_batch
ModelEngine.ragged_verify_token_buckets
ModelEngine.register_sample_in_graph_callable
ModelEngine.register_sample_type_resolver
ModelEngine.replays_final_windows_only
```

### `tensorrt_llm/_torch/pyexecutor/model_loader.py`

库存函数 48；已覆盖 46；未覆盖 2。

```text
ModelLoader.abort_update_weights
ModelLoader.finalize_update_weights
```

### `tensorrt_llm/_torch/pyexecutor/py_executor.py`

库存函数 356；已覆盖 338；未覆盖 18。

```text
PyExecutor.__enter__
PyExecutor.__exit__
PyExecutor._apply_profile_start_config
PyExecutor._apply_profile_stop_config
PyExecutor._broadcast_profile_start
PyExecutor._broadcast_profile_stop
PyExecutor._build_step_scope_label
PyExecutor._initialize_execution.<locals>.on_detected
PyExecutor._sample_in_graph
PyExecutor._sync_and_process_resource_governor_queue
PyExecutor._update_sampler_state_for_disagg_gen_request.<locals>.fail_request
PyExecutor.abort_sleep_transition
PyExecutor.abort_wakeup_transition
PyExecutor.fail_sleep_wakeup_transition
PyExecutor.get_request_admission_state
PyExecutor.set_resource_governor_queue
PyExecutor.start_profile
PyExecutor.stop_profile
```

### `tensorrt_llm/_torch/pyexecutor/replay_inputs.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_replay_inputs_kernel
```

### `tensorrt_llm/_torch/pyexecutor/resource_manager.py`

库存函数 155；已覆盖 126；未覆盖 29。

```text
BaseResourceManager.get_max_resource_count
BaseResourceManager.get_needed_resource_to_completion
BaseResourceManager.get_request_kv_block_budget
BaseResourceManager.kv_block_budget_applies
BlockManager._free_blocks
BlockManager.free_resources
BlockManager.rewind_cache
KVCacheCompressionManager.create_cold_page_codec
KVCacheCompressionManager.decode_cold_pages
KVCacheCompressionManager.encode_cold_pages
KVCacheCompressionManager.mixed_precision_high_window
KVCacheManager.batch_block_table_key
KVCacheManager.calculate_cache_size_per_token
KVCacheManager.extend_capacity_for_tokens
KVCacheManager.get_last_block_id
KVCacheManager.get_max_resource_count
KVCacheManager.get_num_front_blocks_removed
KVCacheManager.get_num_tokens
KVCacheManager.get_pool_configuration
KVCacheManager.pin_blocks
KVCacheManager.spec_recompute_tail
KVCacheManager.truncate_blocks
PeftCacheManager.get_max_resource_count
PeftCacheManager.get_needed_resource_to_completion
ResourceManager.__call__
ResourceManager.register_resource_manager
ResourceManager.reorder_pipeline
SlotManager.fill_slot_id_tensor
compute_page_count
```

### `tensorrt_llm/_torch/pyexecutor/runtime_hooks.py`

库存函数 15；已覆盖 2；未覆盖 13。

```text
DecodePrologue.__call__
ExecutorEvents.after_warmup
ExecutorEvents.at_shutdown
ExecutorEvents.before_overlap_loop
KvBalancedAdmissionPlanner.plan
KvBalancedAdmissionPlanner.window
MsaKvUpdate.__call__
SpecStepFold.begin_step
SpecStepFold.bind_engine_buffers
SpecStepFold.bind_sampler_store
SpecStepFold.gather
SpecStepFold.sampler_rows_stored
SpecStepFold.scatter
```

### `tensorrt_llm/_torch/pyexecutor/sampler/beam_search.py`

库存函数 36；已覆盖 35；未覆盖 1。

```text
_cba_step_math.<locals>._take
```

### `tensorrt_llm/_torch/pyexecutor/sampler/logprobs.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
get_logprobs_from_request
```

### `tensorrt_llm/_torch/pyexecutor/sampler/ops/flashinfer.py`

库存函数 22；已覆盖 16；未覆盖 6。

```text
_FlashInferUnavailable.__getattr__
sampling_batch_spec_dec_one_model_for_rejection
top_k_mask_logits_op
top_k_sampling_from_probs_op
top_k_top_p_sampling_from_logits_op
top_k_top_p_sampling_from_probs_op
```

### `tensorrt_llm/_torch/pyexecutor/sampler/ops/triton.py`

库存函数 17；已覆盖 7；未覆盖 10。

```text
_compact_stop_history_kernel
_compact_stop_match_kernel
_finish_reasons_kernel
_raw_logprobs_direct_topk_kernel
_raw_logprobs_finish_kernel
_raw_logprobs_normalize_kernel
_raw_logprobs_partials_kernel
_raw_logprobs_rank_kernel
_selected_raw_logprobs_kernel
_store_selected_logprobs_kernel
```

### `tensorrt_llm/_torch/pyexecutor/sampler/ops/vanilla.py`

库存函数 18；已覆盖 15；未覆盖 3。

```text
Fusions._gather_log_softmax_impl
Fusions.gather_log_softmax_with_output
occurrence_penalized_logits
```

### `tensorrt_llm/_torch/pyexecutor/sampler/penalties.py`

库存函数 22；已覆盖 17；未覆盖 5。

```text
SpecMetadataLike.is_ragged_verify
_apply_penalties_impl
_unpack_prompt_mask
_update_penalty_counts_impl
seed_prompt
```

### `tensorrt_llm/_torch/pyexecutor/sampler/sampler.py`

库存函数 61；已覆盖 53；未覆盖 8。

```text
Sampler.sample_async
Sampler.sample_in_graph
Sampler.update_requests
Sampler.validate_request
TorchSampler._fast_sample_in_graph
TorchSampler._sample_batched_by_strategy.<locals>._select_steps
TorchSampler.current_sample_type
TorchSampler.sample_in_graph
```

### `tensorrt_llm/_torch/pyexecutor/sampler/sampler_common.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
RequestSeeds.index_select
```

### `tensorrt_llm/_torch/pyexecutor/sampler/sampler_features.py`

库存函数 37；已覆盖 36；未覆盖 1。

```text
_StepIndexTranslator._compute_index_map
```

### `tensorrt_llm/_torch/pyexecutor/sampler/token_ban.py`

库存函数 21；已覆盖 19；未覆盖 2。

```text
TokenBanHandler.apply_ban_list
TokenBanHandler.generate_ban_list
```

### `tensorrt_llm/_torch/pyexecutor/sampler/top_p_decay.py`

库存函数 10；已覆盖 8；未覆盖 2。

```text
TopPDecayHandler._update_store_for_new_requests
TopPDecayHandler.active
```

### `tensorrt_llm/_torch/pyexecutor/scheduled_encoder_executor.py`

库存函数 41；已覆盖 38；未覆盖 3。

```text
ScheduledEncoderExecutor._on_hang_detected
ScheduledEncoderExecutor.get_latest_iteration_stats
ScheduledEncoderExecutor.get_latest_kv_cache_events
```

### `tensorrt_llm/_torch/pyexecutor/scheduler/adp_router.py`

库存函数 57；已覆盖 54；未覆盖 3。

```text
ADPRouter.create_rank_state
ADPRouter.route_requests
_pending_prefill_tokens
```

### `tensorrt_llm/_torch/pyexecutor/scheduler/ctx_decoder_batch.py`

库存函数 8；已覆盖 1；未覆盖 7。

```text
CtxDecoderBatcher.__init__
CtxDecoderBatcher.after_schedule
CtxDecoderBatcher.before_schedule
hold_chunk
hold_pass
holdable
window_rows
```

### `tensorrt_llm/_torch/pyexecutor/scheduler/gen_preemption.py`

库存函数 32；已覆盖 31；未覆盖 1。

```text
GenPreemption.stats_line
```

### `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py`

库存函数 124；已覆盖 109；未覆盖 15。

```text
CapacityScheduler.schedule_request
MicroBatchScheduler.schedule
MultimodalScheduler.can_schedule
MultimodalScheduler.scheduling_state_range
MultimodalScheduler.set_disagg_transfer_admission_controller
PyMicroBatchScheduler._can_be_scheduled
RequestScheduler.can_schedule
RequestScheduler.schedule_request
RequestScheduler.scheduling_state_range
ScheduledRequests.append_encoder_request
ScheduledRequests.is_generation_only
SchedulerPolicyBase.schedule
SimpleScheduler.can_schedule
SimpleScheduler.scheduling_state_range
SimpleUnifiedScheduler.scheduling_state_range
```

### `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py`

库存函数 83；已覆盖 80；未覆盖 3。

```text
KVCacheV2Scheduler._ctx_decoder_hold_chunk
KVCacheV2Scheduler.plan_context_pass
KVCacheV2Scheduler.set_ctx_decoder_hold
```

### `tensorrt_llm/_torch/pyexecutor/scheduler/waiting_queue.py`

库存函数 71；已覆盖 60；未覆盖 11。

```text
WaitingQueue.__bool__
WaitingQueue.__iter__
WaitingQueue.__len__
WaitingQueue.add_request
WaitingQueue.add_requests
WaitingQueue.peek_request
WaitingQueue.peek_requests
WaitingQueue.pop_request
WaitingQueue.prepend_request
WaitingQueue.prepend_requests
WaitingQueue.remove_by_ids
```

### `tensorrt_llm/_torch/pyexecutor/seq_slot_manager.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
SeqSlotManager.get_needed_resource_to_completion
```

### `tensorrt_llm/_torch/pyexecutor/steady_spec_prepare.py`

库存函数 13；已覆盖 12；未覆盖 1。

```text
_steady_stage_kernel
```

### `tensorrt_llm/_torch/pyexecutor/step_inputs.py`

库存函数 30；已覆盖 27；未覆盖 3。

```text
_graph_scatter_kernel
_parallel_scatter_kernel
_step_inputs_scatter_fake
```

### `tensorrt_llm/_torch/route_capture.py`

库存函数 35；已覆盖 24；未覆盖 11。

```text
RouteCapture._capture
RouteCapture._ensure_buffer
RouteCapture._incremental_populate
RouteCapture.capture
RouteCapture.commit
RouteCapture.set_iter
RouteCopier.__init__
RouteCopier._slot_pending
RouteCopier.drain
RouteCopier.stage
RouteCopier.wait_before_overwrite
```

### `tensorrt_llm/_torch/speculative/dflash.py`

库存函数 73；已覆盖 72；未覆盖 1。

```text
DFlashWorker._argmax_block_tokens
```

### `tensorrt_llm/_torch/speculative/draft_target.py`

库存函数 10；已覆盖 9；未覆盖 1。

```text
DraftTargetOneModelWorker.max_draft_len
```

### `tensorrt_llm/_torch/speculative/drafter.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
Drafter.prepare_draft_tokens
```

### `tensorrt_llm/_torch/speculative/dspark.py`

库存函数 61；已覆盖 59；未覆盖 2。

```text
DSparkWorker._markov_block_tokens
DSv4DSparkWorker.batch_slot_view
```

### `tensorrt_llm/_torch/speculative/dspark_ring_metadata.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_ring_metadata_kernel
```

### `tensorrt_llm/_torch/speculative/eagle3.py`

库存函数 44；已覆盖 38；未覆盖 6。

```text
Eagle3OneModelDynamicTreeResourceManager.get_max_resource_count
Eagle3OneModelDynamicTreeResourceManager.get_needed_resource_to_completion
Eagle3OneModelWorker._process_generation_logits
Eagle3OneModelWorker._topk_kernel
Eagle3ResourceManager.get_max_resource_count
Eagle3ResourceManager.get_needed_resource_to_completion
```

### `tensorrt_llm/_torch/speculative/eagle3_dynamic_tree.py`

库存函数 25；已覆盖 24；未覆盖 1。

```text
_gather_repack_step0_kernel
```

### `tensorrt_llm/_torch/speculative/fused_accept.py`

库存函数 10；已覆盖 6；未覆盖 4。

```text
_sample_accept_kernel
_step0_epilogue_kernel
_strict_accept_kernel
_strict_accept_rows
```

### `tensorrt_llm/_torch/speculative/greedy_argmax.py`

库存函数 9；已覆盖 6；未覆盖 3。

```text
_argmax_chunk_kernel
_argmax_final_kernel
_combine
```

### `tensorrt_llm/_torch/speculative/interface.py`

库存函数 168；已覆盖 155；未覆盖 13。

```text
AuxiliarySpeculativeStateHandler.commit_speculative_states
SpecMetadata._penalty_value
SpecMetadata._request_prompt_tokens
SpecMetadata.batch_uses_penalty
SpecWorkerBase._draft_sampler_greedy
SpecWorkerBase._forward_impl
SpecWorkerBase._fused_draft_lm_head
SpecWorkerBase._reject_watermark
SpecWorkerBase._sample_and_accept_in_one_launch
SpecWorkerBase.draft_kv_cache_context.<locals>.patch_locality_metadata
SpecWorkerBase.max_draft_len
SpecWorkerBase.target_vocab_parallel_head
SpeculativeDecodingMode.needs_kv_cache_rewind
```

### `tensorrt_llm/_torch/speculative/kept_rows.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
_gather_rows2_kernel
_kept_row_ids_kernel
```

### `tensorrt_llm/_torch/speculative/mtp.py`

库存函数 29；已覆盖 24；未覆盖 5。

```text
MTPHiddenStatesManager.get_needed_resource_to_completion
MTPWorker.max_draft_len
MTPWorker.process_generation_logits
MTPWorker.skip_forward
MTPWorker.topk_kernel
```

### `tensorrt_llm/_torch/speculative/mtp_dynamic_tree.py`

库存函数 33；已覆盖 2；未覆盖 31。

```text
MTPEagleDynamicTreeResourceManager.__init__
MTPEagleDynamicTreeResourceManager.add_dummy_requests
MTPEagleDynamicTreeResourceManager.free_resources
MTPEagleDynamicTreeResourceManager.get_max_resource_count
MTPEagleDynamicTreeResourceManager.get_needed_resource_to_completion
MTPEagleDynamicTreeResourceManager.mtp_past_hidden_states_pool
MTPEagleDynamicTreeResourceManager.mtp_past_tokens_pool
MTPEagleDynamicTreeResourceManager.prepare_resources
MTPEagleDynamicTreeResourceManager.sa_manager
MTPEagleDynamicTreeResourceManager.shutdown
MTPEagleDynamicTreeResourceManager.slot_manager
MTPEagleDynamicTreeResourceManager.update_resources
MTPEagleDynamicTreeWorker.__init__
MTPEagleDynamicTreeWorker._accepted_leaf_intermediate_positions
MTPEagleDynamicTreeWorker._apply_spec_metadata
MTPEagleDynamicTreeWorker._ensure_spec_dec_state_restored
MTPEagleDynamicTreeWorker._ensure_spec_tree_manager
MTPEagleDynamicTreeWorker._forward_draft_loop
MTPEagleDynamicTreeWorker._forward_impl
MTPEagleDynamicTreeWorker._prepare_attn_metadata_for_spec_dec
MTPEagleDynamicTreeWorker._prepare_draft_layer_metadata
MTPEagleDynamicTreeWorker._refresh_blackwell_tree_mask_metadata
MTPEagleDynamicTreeWorker._repack_mask_padded_to_packed
MTPEagleDynamicTreeWorker._restore_attn_metadata_from_spec_dec
MTPEagleDynamicTreeWorker.prepare_position_ids_and_last_tokens
MTPEagleDynamicTreeWorker.prepare_tree_mask_and_position_offset
MTPEagleDynamicTreeWorker.resampling_final_draft_tokens
MTPEagleDynamicTreeWorker.sample
MTPEagleDynamicTreeWorker.sample_and_accept_draft_tokens
MTPEagleDynamicTreeWorker.update_draft_tokens_and_scores
MTPEagleDynamicTreeWorker.update_hidden_states
```

### `tensorrt_llm/_torch/speculative/ngram.py`

库存函数 11；已覆盖 7；未覆盖 4。

```text
NGramPoolManager._print_line
NGramPoolManager.get_max_resource_count
NGramPoolManager.get_needed_resource_to_completion
NGramPoolManager.print_pool
```

### `tensorrt_llm/_torch/speculative/pard.py`

库存函数 13；已覆盖 12；未覆盖 1。

```text
PARDWorker.max_draft_len
```

### `tensorrt_llm/_torch/speculative/sa_worker.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
SAWorker.max_draft_len
```

### `tensorrt_llm/_torch/speculative/save_hidden_state.py`

库存函数 12；已覆盖 10；未覆盖 2。

```text
SaveHiddenStatesResourceManager.get_max_resource_count
SaveHiddenStatesResourceManager.get_needed_resource_to_completion
```

### `tensorrt_llm/_torch/speculative/sharded_sampling.py`

库存函数 7；已覆盖 4；未覆盖 3。

```text
_gumbel_scores
_sharded_argmax_kernel
prefer_argmax
```

### `tensorrt_llm/_torch/speculative/spec_sampler_base.py`

库存函数 18；已覆盖 17；未覆盖 1。

```text
SpecSampler._device_slot_ids
```

### `tensorrt_llm/_torch/speculative/spec_store_scatter.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_spec_store_scatter_kernel
```

### `tensorrt_llm/_torch/speculative/spec_tail_kernels.py`

库存函数 8；已覆盖 4；未覆盖 4。

```text
_decode_block_table_kernel
_paged_latent_prepare_kernel
_store_context_latents_kernel
_strict_accept_kernel
```

### `tensorrt_llm/_torch/speculative/spec_tree_manager.py`

库存函数 13；已覆盖 12；未覆盖 1。

```text
SpecTreeManager.dump_tree_info
```

### `tensorrt_llm/_torch/speculative/speculation_gate.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
SpeculationGate.reset
```

### `tensorrt_llm/_torch/speculative/suffix_automaton.py`

库存函数 17；已覆盖 15；未覆盖 2。

```text
SuffixAutomatonManager.get_max_resource_count
SuffixAutomatonManager.get_needed_resource_to_completion
```

### `tensorrt_llm/_torch/speculative/vocab_parallel.py`

库存函数 41；已覆盖 25；未覆盖 16。

```text
VocabParallelHead.shard_logits
VocabShardedLogits.device
VocabShardedLogits.gather
_global_index
_gumbel
_gumbel_fine
_local_argmax_kernel
_local_sample_kernel
_markov_add_local_argmax_kernel
_markov_reduce_embed_kernel
_markov_sample_local_kernel
_markov_sample_normalize_kernel
_markov_sample_reduce_kernel
_reduce_pairs_kernel
_store_pair
argmax_combine
```

### `tensorrt_llm/_torch/utils.py`

库存函数 65；已覆盖 59；未覆盖 6。

```text
MxFp8QuantizedTensor.dtype
MxFp8QuantizedTensor.numel
MxFp8QuantizedTensor.split
_
infer_output_m_shape
tensor_to_str
```

### `tensorrt_llm/_torch/virtual_memory.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
maybe_scope
```

### `tensorrt_llm/_torch/visual_gen/attention/attention.py`

库存函数 22；已覆盖 21；未覆盖 1。

```text
Attention.apply_packed_qk_norm_rope_fp8
```

### `tensorrt_llm/_torch/visual_gen/attention/backends/cute_dsl/fmha.py`

库存函数 20；已覆盖 17；未覆盖 3。

```text
CuTeDSLAttention._smooth_qk
CuTeDSLAttention.support_fused_qkv
CuTeDSLAttention.support_lse
```

### `tensorrt_llm/_torch/visual_gen/attention/backends/flash_attn4.py`

库存函数 14；已覆盖 12；未覆盖 2。

```text
_install_flash_attn_tile_scheduler_compatibility.<locals>.is_valid_tile
_install_flash_attn_tile_scheduler_compatibility.<locals>.tile_idx
```

### `tensorrt_llm/_torch/visual_gen/attention/backends/flashinfer.py`

库存函数 14；已覆盖 13；未覆盖 1。

```text
FlashInferAttention.support_lse
```

### `tensorrt_llm/_torch/visual_gen/attention/backends/interface.py`

库存函数 10；已覆盖 6；未覆盖 4。

```text
AttentionBackend.forward
AttentionBackend.forward_with_lse
AttentionBackend.preferred_layout
AttentionBackend.support_lse
```

### `tensorrt_llm/_torch/visual_gen/attention/backends/parallel.py`

库存函数 39；已覆盖 35；未覆盖 4。

```text
Attention2DAttention.support_lse
RingAttention.support_lse
UlyssesAttention.support_fused_qkv
UlyssesAttention.support_kv_cache_page_size
```

### `tensorrt_llm/_torch/visual_gen/attention/backends/sparse/sol/backend.py`

库存函数 15；已覆盖 13；未覆盖 2。

```text
SOLCuTeDSLAttention.support_fused_qkv
SOLCuTeDSLAttention.support_lse
```

### `tensorrt_llm/_torch/visual_gen/attention/backends/sparse/sol/kernels.py`

库存函数 15；已覆盖 11；未覆盖 4。

```text
_block_pool_kernel
_block_statistics_kernel
_block_thresholds_kernel
_select_exact_blocks_kernel
```

### `tensorrt_llm/_torch/visual_gen/attention/backends/sparse/vsa/backend.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
VSACuTeDSLAttention.support_fused_qkv
VSATrtllmAttention.support_fused_qkv
```

### `tensorrt_llm/_torch/visual_gen/attention/backends/sparse/vsa/kernels.py`

库存函数 11；已覆盖 8；未覆盖 3。

```text
_blend_coarse_fine_kernel
_sort_rows_kernel
_tile_and_pool_cubes_kernel
```

### `tensorrt_llm/_torch/visual_gen/cache/block_causal_kv_cache.py`

库存函数 61；已覆盖 60；未覆盖 1。

```text
BlockCausalKVCacheManager.pin_tokens
```

### `tensorrt_llm/_torch/visual_gen/cache/rollout_kv_cache_interface.py`

库存函数 15；已覆盖 6；未覆盖 9。

```text
RolloutKVCacheInterface.__enter__
RolloutKVCacheInterface.__exit__
RolloutKVCacheInterface.close
RolloutKVCacheInterface.commit
RolloutKVCacheInterface.is_open
RolloutKVCacheInterface.open
RolloutKVCacheInterface.plan
RolloutKVCacheInterface.shutdown
RolloutKVCacheInterface.spec
```

### `tensorrt_llm/_torch/visual_gen/cache/teacache_accelerator.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
TeaCacheAccelerator.unwrap
```

### `tensorrt_llm/_torch/visual_gen/config.py`

库存函数 25；已覆盖 21；未覆盖 4。

```text
DiffusionModelConfig.cache_dit
DiffusionModelConfig.seacache
DiffusionPipelineConfig._convert_quantization_metadata
DiffusionPipelineConfig.get_quant_config
```

### `tensorrt_llm/_torch/visual_gen/cuda_graph_runner.py`

库存函数 16；已覆盖 15；未覆盖 1。

```text
ExtraKeyFn.__call__
```

### `tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/attention/fmha.py`

库存函数 22；已覆盖 5；未覆盖 17。

```text
BlackwellFusedMultiHeadAttentionForward.__call__
BlackwellFusedMultiHeadAttentionForward.apply_exp_and_cvt
BlackwellFusedMultiHeadAttentionForward.calculate_skip_softmax_flag
BlackwellFusedMultiHeadAttentionForward.can_implement
BlackwellFusedMultiHeadAttentionForward.check_invalid_shape
BlackwellFusedMultiHeadAttentionForward.check_supported_dtypes
BlackwellFusedMultiHeadAttentionForward.correction_epilog
BlackwellFusedMultiHeadAttentionForward.correction_rescale
BlackwellFusedMultiHeadAttentionForward.get_skip_softmax_flag
BlackwellFusedMultiHeadAttentionForward.kernel
BlackwellFusedMultiHeadAttentionForward.kernel.<locals>.make_tmem_tensors
BlackwellFusedMultiHeadAttentionForward.kv_producer_update_tx_acquire_and_advance
BlackwellFusedMultiHeadAttentionForward.mma_pv
BlackwellFusedMultiHeadAttentionForward.mma_qk
BlackwellFusedMultiHeadAttentionForward.softmax
BlackwellFusedMultiHeadAttentionForward.softmax.<locals>.softmax_loop
BlackwellFusedMultiHeadAttentionForward.softmax_step
```

### `tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/attention/fmha_blockscaled.py`

库存函数 26；已覆盖 6；未覆盖 20。

```text
BlackwellFusedMultiHeadBlockScaledAttentionForward.__call__
BlackwellFusedMultiHeadBlockScaledAttentionForward.apply_exp_and_cvt
BlackwellFusedMultiHeadBlockScaledAttentionForward.calculate_skip_softmax_flag
BlackwellFusedMultiHeadBlockScaledAttentionForward.can_implement
BlackwellFusedMultiHeadBlockScaledAttentionForward.check_invalid_shape
BlackwellFusedMultiHeadBlockScaledAttentionForward.check_supported_dtypes
BlackwellFusedMultiHeadBlockScaledAttentionForward.correction_epilog
BlackwellFusedMultiHeadBlockScaledAttentionForward.correction_rescale
BlackwellFusedMultiHeadBlockScaledAttentionForward.get_skip_softmax_flag
BlackwellFusedMultiHeadBlockScaledAttentionForward.kernel
BlackwellFusedMultiHeadBlockScaledAttentionForward.kernel.<locals>.make_tmem_tensors
BlackwellFusedMultiHeadBlockScaledAttentionForward.kv_producer_update_tx_acquire_and_advance
BlackwellFusedMultiHeadBlockScaledAttentionForward.mma_pv
BlackwellFusedMultiHeadBlockScaledAttentionForward.mma_qk
BlackwellFusedMultiHeadBlockScaledAttentionForward.softmax
BlackwellFusedMultiHeadBlockScaledAttentionForward.softmax.<locals>.softmax_loop
BlackwellFusedMultiHeadBlockScaledAttentionForward.softmax_step
compact_fp4_data
create_scale_factor_tensor
create_scale_factor_tensor.<locals>.ceil_div
```

### `tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/sol_attn/common/layout_utils.py`

库存函数 6；已覆盖 1；未覆盖 5。

```text
_accumulator_frga_layout
_accumulator_mn_layout
reshape_acc_to_frgA
reshape_acc_to_mn
transpose_view
```

### `tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/sol_attn/common/selector.py`

库存函数 8；已覆盖 2；未覆盖 6。

```text
_mask_word
_test_exact_bit
sol_attn_bfind_b32
sol_attn_mask_word_constexpr
sol_attn_set_exact_bit
sol_attn_test_exact_bit_limited_words
```

### `tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/sol_attn/interface.py`

库存函数 11；已覆盖 10；未覆盖 1。

```text
get_sol_attn_backend
```

### `tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/sol_attn/preprocess.py`

库存函数 10；已覆盖 4；未覆盖 6。

```text
_diag_threshold_kernel
_exact_fused_threshold_kernel
_pool_query_kernel
_reduce_kc_kernel
_reduce_kc_stats_kernel
_reduce_vc_kernel
```

### `tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/sol_attn_backend.py`

库存函数 10；已覆盖 9；未覆盖 1。

```text
_cute_runtime_available
```

### `tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/video_sparse_attention/block_sparse_attn_dsl_fwd.py`

库存函数 34；已覆盖 6；未覆盖 28。

```text
VideoSparseAttentionForwardGroup2QInterleaveKV.__call__
VideoSparseAttentionForwardGroup2QInterleaveKV.cal_local_max
VideoSparseAttentionForwardGroup2QInterleaveKV.cal_local_sum
VideoSparseAttentionForwardGroup2QInterleaveKV.correction
VideoSparseAttentionForwardGroup2QInterleaveKV.correction_loop
VideoSparseAttentionForwardGroup2QInterleaveKV.epilogue
VideoSparseAttentionForwardGroup2QInterleaveKV.epilogue.<locals>._get_gO_cpy
VideoSparseAttentionForwardGroup2QInterleaveKV.epilogue.<locals>._get_sO_cpy
VideoSparseAttentionForwardGroup2QInterleaveKV.exp2f
VideoSparseAttentionForwardGroup2QInterleaveKV.kernel
VideoSparseAttentionForwardGroup2QInterleaveKV.load
VideoSparseAttentionForwardGroup2QInterleaveKV.load.<locals>._get_gK_cpy
VideoSparseAttentionForwardGroup2QInterleaveKV.load.<locals>._get_gQ_cpy
VideoSparseAttentionForwardGroup2QInterleaveKV.load.<locals>._get_gV_cpy
VideoSparseAttentionForwardGroup2QInterleaveKV.load.<locals>._get_sK_cpy
VideoSparseAttentionForwardGroup2QInterleaveKV.load.<locals>._get_sQ_cpy
VideoSparseAttentionForwardGroup2QInterleaveKV.load.<locals>._get_sV_cpy
VideoSparseAttentionForwardGroup2QInterleaveKV.load.<locals>._load_K
VideoSparseAttentionForwardGroup2QInterleaveKV.load.<locals>._load_Q
VideoSparseAttentionForwardGroup2QInterleaveKV.load.<locals>._load_V
VideoSparseAttentionForwardGroup2QInterleaveKV.mask_then_cal_local_max
VideoSparseAttentionForwardGroup2QInterleaveKV.mask_then_cal_local_max.<locals>.clamp
VideoSparseAttentionForwardGroup2QInterleaveKV.mma
VideoSparseAttentionForwardGroup2QInterleaveKV.softmax
VideoSparseAttentionForwardGroup2QInterleaveKV.softmax.<locals>._get_tP_store
VideoSparseAttentionForwardGroup2QInterleaveKV.softmax_loop
VideoSparseAttentionForwardGroup2QInterleaveKV.softmax_step
VideoSparseAttentionForwardGroup2QInterleaveKV.update_row_max
```

### `tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/video_sparse_attention/ptx.py`

库存函数 7；已覆盖 1；未覆盖 6。

```text
__cvta_generic_to_shared
atomicAdd_f32
atomicMax_f32
exp2f
fma
warp_reduction_fmax
```

### `tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/video_sparse_attention/scheduler.py`

库存函数 17；已覆盖 14；未覆盖 3。

```text
TestScheduler.__call__
TestScheduler.__init__
TestScheduler.kernel
```

### `tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/helpers/fmha_helpers.py`

库存函数 28；已覆盖 18；未覆盖 10。

```text
FmhaStaticTileScheduler.check_valid_work_for_seqlen_q
FusedMask.apply_mask
FusedMask.get_leading_mask_id
FusedMask.get_masked_info
FusedMask.get_masked_leading_count
FusedMask.get_masked_trailing_count
FusedMask.get_trailing_mask_id
FusedMask.get_trip_start
FusedMask.get_unmasked_trip_count
ex2_emulation_packed_f32x2.<locals>.polynomial_deg3_packed_f32x2
```

### `tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/helpers/static_persistent_tile_scheduler.py`

库存函数 24；已覆盖 3；未覆盖 21。

```text
PersistentTileSchedulerParams.__extract_mlir_values__
PersistentTileSchedulerParams.__init__
PersistentTileSchedulerParams.__new_from_mlir_values__
PersistentTileSchedulerParams.get_grid_shape
StaticPersistentRuntimeTileScheduler.__init__
StaticPersistentRuntimeTileScheduler.__new_from_mlir_values__
StaticPersistentRuntimeTileScheduler._get_current_work_for_linear_idx
StaticPersistentRuntimeTileScheduler.create
StaticPersistentTileScheduler.__extract_mlir_values__
StaticPersistentTileScheduler.__init__
StaticPersistentTileScheduler.__new_from_mlir_values__
StaticPersistentTileScheduler._get_cluster_work_idx_with_fastdivmod
StaticPersistentTileScheduler._get_current_work_for_linear_idx
StaticPersistentTileScheduler.advance_to_next_work
StaticPersistentTileScheduler.create
StaticPersistentTileScheduler.get_current_work
StaticPersistentTileScheduler.get_grid_shape
StaticPersistentTileScheduler.initial_work_tile_info
StaticPersistentTileScheduler.num_tiles_executed
WorkTileInfo.is_valid_tile
WorkTileInfo.tile_idx
```

### `tensorrt_llm/_torch/visual_gen/executor.py`

库存函数 55；已覆盖 51；未覆盖 4。

```text
DiffusionExecutor._log_cuda_peak_memory
DiffusionExecutor._reset_cuda_peak_memory_stats
DiffusionRemoteClient.get_current_iteration_snapshot
DiffusionRemoteClient.get_iteration_stats
```

### `tensorrt_llm/_torch/visual_gen/kernels/blackwell/conv/dense_blockscaled_implicit_gemm_fprop.py`

库存函数 46；已覆盖 35；未覆盖 11。

```text
PipelineTmaCpAsyncUmma.producer_acquire_cpasync
PipelineTmaCpAsyncUmma.producer_commit_cpasync
Sm100BlockScaledPersistentDenseImplicitGemmKernel.__call__
Sm100BlockScaledPersistentDenseImplicitGemmKernel.__call__.<locals>._make_sfb_tma
Sm100BlockScaledPersistentDenseImplicitGemmKernel.cluster_specific_kernel
Sm100BlockScaledPersistentDenseImplicitGemmKernel.is_valid_dtypes_and_scale_factor_vec_size
Sm100BlockScaledPersistentDenseImplicitGemmKernel.is_valid_layouts
Sm100BlockScaledPersistentDenseImplicitGemmKernel.is_valid_tensor_alignment
Sm100BlockScaledPersistentDenseImplicitGemmKernel.is_valid_tensor_alignment.<locals>.check_contigous_16B_alignment
Sm100BlockScaledPersistentDenseImplicitGemmKernel.kernel
_parse_comma_separated_ints
```

### `tensorrt_llm/_torch/visual_gen/kernels/blackwell/conv/dense_gemm_persistent_dynamic.py`

库存函数 25；已覆盖 5；未覆盖 20。

```text
PersistentDenseGemmKernel.__call__
PersistentDenseGemmKernel._create_tiled_mma
PersistentDenseGemmKernel._setup_attributes
PersistentDenseGemmKernel.can_implement
PersistentDenseGemmKernel.check_epilog_store_option
PersistentDenseGemmKernel.check_supported_dtypes
PersistentDenseGemmKernel.check_tensor_alignment
PersistentDenseGemmKernel.check_tensor_alignment.<locals>.check_contiguous_16B_alignment
PersistentDenseGemmKernel.kernel
_compute_stages
_parse_comma_separated_ints
benchmark_nsight
benchmark_nsight.<locals>._details
benchmark_nsight.<locals>.compute_tflops
bmm
compile_bmm
prepare_parser
prepare_tensors
run
run.<locals>.generate_tensors
```

### `tensorrt_llm/_torch/visual_gen/kernels/blackwell/conv/dense_gemm_persistent_dynamic_preferred_cluster.py`

库存函数 12；已覆盖 1；未覆盖 11。

```text
PersistentDenseGemmKernelDynamicPreferredCluster.__call__
PersistentDenseGemmKernelDynamicPreferredCluster._create_tiled_mma
PersistentDenseGemmKernelDynamicPreferredCluster._setup_attributes
PersistentDenseGemmKernelDynamicPreferredCluster.cluster_specific_kernel
PersistentDenseGemmKernelDynamicPreferredCluster.kernel
_parse_comma_separated_ints
bmm
compile_bmm
prepare_parser
run
run.<locals>.generate_tensors
```

### `tensorrt_llm/_torch/visual_gen/kernels/blackwell/pertoken_adaln/pertoken_adaln.py`

库存函数 17；已覆盖 13；未覆盖 4。

```text
PerTokenAdaLN.__call__
PerTokenAdaLN.kernel
PerTokenAdaLN.kernel.<locals>.copy_if
PerTokenAdaLN.kernel.<locals>.slice_if
```

### `tensorrt_llm/_torch/visual_gen/memory_monitor.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
WorkerMemoryMonitor.report_error
```

### `tensorrt_llm/_torch/visual_gen/models/cosmos3/guardrails.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
check_video_safety
```

### `tensorrt_llm/_torch/visual_gen/models/cosmos3/modules.py`

库存函数 14；已覆盖 4；未覆盖 10。

```text
ConvLayerNorm.__init__
ConvLayerNorm.forward
NormConvTranspose1d.__init__
NormConvTranspose1d.forward
SConvTranspose1d.__init__
SConvTranspose1d.forward
apply_parametrization_norm
get_norm_module
pad1d
unpad1d
```

### `tensorrt_llm/_torch/visual_gen/models/cosmos3/pipeline_cosmos3.py`

库存函数 75；已覆盖 74；未覆盖 1。

```text
tqdm
```

### `tensorrt_llm/_torch/visual_gen/models/cosmos3/sound_tokenizer.py`

库存函数 19；已覆盖 16；未覆盖 3。

```text
OobleckDecoderBlock.remove_weight_norm
TrimPadding.__init__
TrimPadding.forward
```

### `tensorrt_llm/_torch/visual_gen/models/cosmos3/step_precision.py`

库存函数 11；已覆盖 10；未覆盖 1。

```text
linear_runs_high_precision
```

### `tensorrt_llm/_torch/visual_gen/models/cosmos3/transformer_cosmos3.py`

库存函数 71；已覆盖 68；未覆盖 3。

```text
Cosmos3CausalAttention.forward
Cosmos3VFMTransformer.device
TimestepEmbedder._init_weights
```

### `tensorrt_llm/_torch/visual_gen/models/flux/attention.py`

库存函数 22；已覆盖 21；未覆盖 1。

```text
Flux2ParallelSelfAttention._combine_split_projection
```

### `tensorrt_llm/_torch/visual_gen/models/flux/pipeline_flux2.py`

库存函数 42；已覆盖 40；未覆盖 2。

```text
Flux2Pipeline._compute_flux2_timestep_embedding
Flux2Pipeline._tokenize_qwen3
```

### `tensorrt_llm/_torch/visual_gen/models/flux/transformer_flux.py`

库存函数 30；已覆盖 29；未覆盖 1。

```text
CombinedTimestepTextProjEmbeddings.forward
```

### `tensorrt_llm/_torch/visual_gen/models/glm_image/pipeline_glm_image.py`

库存函数 29；已覆盖 25；未覆盖 4。

```text
GlmImagePipeline.default_generation_params
GlmImagePipeline.default_warmup_num_frames
GlmImagePipeline.default_warmup_resolutions
GlmImagePipeline.infer
```

### `tensorrt_llm/_torch/visual_gen/models/hunyuan_video1_5/pipeline_hunyuan_video1_5.py`

库存函数 24；已覆盖 23；未覆盖 1。

```text
HunyuanVideo15Pipeline._prepare_mask
```

### `tensorrt_llm/_torch/visual_gen/models/hunyuan_video1_5/timestep_embedding.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
CombinedTimestepGuidanceTextProjEmbeddings.__init__
CombinedTimestepGuidanceTextProjEmbeddings.forward
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/ltx2_core/audio_vae/resnet.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
ResBlock2.__init__
ResBlock2.forward
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/ltx2_core/guiders.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
MultiModalGuider.calculate
MultiModalGuider.should_skip_step
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/ltx2_core/patchifier.py`

库存函数 15；已覆盖 13；未覆盖 2。

```text
AudioPatchifier.get_token_count
AudioPatchifier.patch_size
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/ltx2_core/upsampler.py`

库存函数 14；已覆盖 13；未覆盖 1。

```text
BlurDownsample._apply_2d
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/ltx2_core/video_vae/convolution.py`

库存函数 11；已覆盖 3；未覆盖 8。

```text
CausalConv3d.weight
DualConv3d.__init__
DualConv3d.forward
DualConv3d.forward_with_2d
DualConv3d.forward_with_3d
DualConv3d.reset_parameters
DualConv3d.weight
make_linear_nd
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/ltx2_core/video_vae/resnet.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
ResnetBlock3D._feed_spatial_noise
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/ltx2_core/video_vae/tiling.py`

库存函数 10；已覆盖 9；未覆盖 1。

```text
compute_rectangular_mask_1d
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/ltx2_core/video_vae/video_vae.py`

库存函数 22；已覆盖 19；未覆盖 3。

```text
decode_video
decode_video.<locals>.convert_to_uint8
get_video_chunks_number
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/pipeline_ltx2.py`

库存函数 57；已覆盖 49；未覆盖 8。

```text
LTX2Pipeline._compute_ltx2_timestep_embedding
LTX2Pipeline._enhance_prompt
LTX2Pipeline.infer
LTX2TeaCacheExtractor.__call__
LTX2TeaCacheExtractor.__call__.<locals>.postprocess
LTX2TeaCacheExtractor.__call__.<locals>.run_blocks
_LTX2CUDAGraphRunner._perturbation_fingerprint
_read_safetensors_config
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/pipeline_ltx2_two_stages.py`

库存函数 45；已覆盖 43；未覆盖 2。

```text
_count_saved_lora_weight_tensors
_requantize_fp4_weight
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/transformer_ltx2.py`

库存函数 55；已覆盖 53；未覆盖 2。

```text
BasicAVTransformerBlock._get_ada_values
BasicAVTransformerBlock._get_av_ca_ada_values
```

### `tensorrt_llm/_torch/visual_gen/models/ltx23/pipeline_ltx23.py`

库存函数 25；已覆盖 23；未覆盖 2。

```text
LTX23Pipeline.extra_param_specs
LTX23Pipeline.infer
```

### `tensorrt_llm/_torch/visual_gen/models/ltx23/pipeline_ltx23_retake.py`

库存函数 21；已覆盖 19；未覆盖 2。

```text
LTX23RetakePipeline.extra_param_specs
LTX23RetakePipeline.ref_slot_specs
```

### `tensorrt_llm/_torch/visual_gen/models/minimax_h3/fused_adaln.py`

库存函数 8；已覆盖 5；未覆盖 3。

```text
_ld
_minimax_h3_adaln_kernel
_st
```

### `tensorrt_llm/_torch/visual_gen/models/minimax_h3/fused_rope.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
_minimax_h3_qk_norm_rope_kernel
```

### `tensorrt_llm/_torch/visual_gen/models/minimax_h3/pipeline_minimax_h3.py`

库存函数 59；已覆盖 57；未覆盖 2。

```text
MiniMaxH3Pipeline.default_warmup_num_frames
MiniMaxH3Pipeline.default_warmup_resolutions
```

### `tensorrt_llm/_torch/visual_gen/models/minimax_h3/ref2va.py`

库存函数 13；已覆盖 11；未覆盖 2。

```text
_normalize_video_condition
_sample_video_condition_frames
```

### `tensorrt_llm/_torch/visual_gen/models/minimax_h3/vsa.py`

库存函数 6；已覆盖 4；未覆盖 2。

```text
get_h3_vsa_context
set_h3_vsa_context
```

### `tensorrt_llm/_torch/visual_gen/models/modeling.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
BaseDiffusionModel.forward
```

### `tensorrt_llm/_torch/visual_gen/models/qwen_image/pipeline_qwen_image.py`

库存函数 30；已覆盖 28；未覆盖 2。

```text
QwenImagePipeline._compute_qwen_image_timestep_embedding
QwenImagePipeline._extract_masked_hidden
```

### `tensorrt_llm/_torch/visual_gen/models/qwen_image/pipeline_qwen_image_edit.py`

库存函数 16；已覆盖 2；未覆盖 14。

```text
QwenImageEditPlusPipeline._encode_edit_prompt
QwenImageEditPlusPipeline._encode_vae_image
QwenImageEditPlusPipeline._get_qwen_edit_prompt_embeds
QwenImageEditPlusPipeline._prepare_edit_latents
QwenImageEditPlusPipeline._preprocess_edit_images
QwenImageEditPlusPipeline._run_warmup
QwenImageEditPlusPipeline.default_generation_params
QwenImageEditPlusPipeline.default_warmup_resolutions
QwenImageEditPlusPipeline.forward
QwenImageEditPlusPipeline.load_standard_components
QwenImageEditPlusPipeline.ref_slot_specs
QwenImageEditPlusPipeline.warmup_cache_key
_calculate_dimensions
_retrieve_latents
```

### `tensorrt_llm/_torch/visual_gen/models/wan/dup_up3d.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
_dup_up3d_kernel
```

### `tensorrt_llm/_torch/visual_gen/models/wan/fp4_fused_quant.py`

库存函数 10；已覆盖 7；未覆盖 3。

```text
_build_kernel.<locals>._e2m1_code
_build_kernel.<locals>._rmsnorm_silu_nvfp4_quant_kernel
_build_kernel.<locals>._silu_nvfp4_quant_kernel
```

### `tensorrt_llm/_torch/visual_gen/models/wan/kernels/blackwell/fused_conv.py`

库存函数 60；已覆盖 40；未覆盖 20。

```text
Wino23Conv.__call__
Wino23Conv._cta_main
Wino23Conv._epi_butterfly
Wino23Conv._epi_drain
Wino23Conv._epilogue
Wino23Conv._epilogue_block
Wino23Conv._mma_acc_taps
Wino23Conv._mma_block
Wino23Conv._mma_cout_half
Wino23Conv._mma_first_chunk
Wino23Conv._mma_last_chunk
Wino23Conv._mma_middle_chunks
Wino23Conv._mma_split_chunk
Wino23Conv._mma_warp
Wino23Conv._prefetch_descriptors
Wino23Conv._store_outputs
Wino23Conv._transform_warps
Wino23Conv._u_producer
Wino23Conv._x_producer
Wino23Conv.kernel
```

### `tensorrt_llm/_torch/visual_gen/models/wan/kernels/frame_ops.py`

库存函数 26；已覆盖 19；未覆盖 7。

```text
AvgDownAddKernel._load_window_taps
AvgDownAddKernel.kernel
AvgDownAddKernel.wrapper
InterleaveBiasKernel.kernel
InterleaveBiasKernel.wrapper
UnpatchifyClampKernel.kernel
UnpatchifyClampKernel.wrapper
```

### `tensorrt_llm/_torch/visual_gen/models/wan/kernels/norm.py`

库存函数 53；已覆盖 42；未覆盖 11。

```text
VaeNormKernel._begin_position
VaeNormKernel._fill_tables
VaeNormKernel._frame_step
VaeNormKernel._halo_targets
VaeNormKernel._load_cache_into_ring
VaeNormKernel._next_position
VaeNormKernel._prefetch_next_frame
VaeNormKernel._push_halo
VaeNormKernel._store_pad_rows
VaeNormKernel.kernel
VaeNormKernel.wrapper
```

### `tensorrt_llm/_torch/visual_gen/models/wan/kernels/spatial_winograd.py`

库存函数 48；已覆盖 39；未覆盖 9。

```text
InputTransformKernel._frame_buffer
InputTransformKernel._register_window
InputTransformKernel._stage_window
InputTransformKernel.kernel
InputTransformKernel.wrapper
OutputTransformKernel._store_tile
OutputTransformKernel.kernel
OutputTransformKernel.wrapper
_Scaled.__add__
```

### `tensorrt_llm/_torch/visual_gen/models/wan/kernels/upsample_gather.py`

库存函数 15；已覆盖 13；未覆盖 2。

```text
UpsampleGatherKernel.kernel
UpsampleGatherKernel.wrapper
```

### `tensorrt_llm/_torch/visual_gen/models/wan/parallel_wino_vae.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
ParallelWinoVae.encode_mode
```

### `tensorrt_llm/_torch/visual_gen/models/wan/utils_wan.py`

库存函数 14；已覆盖 13；未覆盖 1。

```text
WanPerTokenAdaLN.self_attention_gate
```

### `tensorrt_llm/_torch/visual_gen/models/wan/vae_loader.py`

库存函数 12；已覆盖 11；未覆盖 1。

```text
_load_nvfp4_wan_vae
```

### `tensorrt_llm/_torch/visual_gen/models/wan/wan_vae.py`

库存函数 72；已覆盖 70；未覆盖 2。

```text
WanVAE.forward
dequant_fp4_conv_weight
```

### `tensorrt_llm/_torch/visual_gen/models/wan/wino_vae.py`

库存函数 116；已覆盖 111；未覆盖 5。

```text
LazyDiagonalGaussianDistribution.var
WinoVae.set_pdl
WinoVaeAttentionBlock._forward_gathered
WinoVaeCausalConv3d._single_frame_spatial
WinoVaeUpBlock.forward
```

### `tensorrt_llm/_torch/visual_gen/models/wan/wino_vae_shard.py`

库存函数 33；已覆盖 29；未覆盖 4。

```text
Shard.first
Shard.last
Shard.pad_for_downsample
Shard.retire
```

### `tensorrt_llm/_torch/visual_gen/modules/vae/nvfp4_conv_autotuner.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
_launch_fallback_tactic
```

### `tensorrt_llm/_torch/visual_gen/modules/vae/parallel_vae_interface.py`

库存函数 14；已覆盖 10；未覆盖 4。

```text
ParallelVAEBase._decode_impl
ParallelVAEBase._encode_impl
ParallelVAEBase._parallelize_modules
ParallelVAEBase.make_spec
```

### `tensorrt_llm/_torch/visual_gen/parallel/token_sharded_modules.py`

库存函数 19；已覆盖 18；未覆盖 1。

```text
TokenShardedAdapter.prepare
```

### `tensorrt_llm/_torch/visual_gen/pipeline.py`

库存函数 61；已覆盖 50；未覆盖 11。

```text
BasePipeline._init_transformer
BasePipeline._rescale_noise_cfg
BasePipeline._run_warmup
BasePipeline.default_offload_stages
BasePipeline.default_warmup_num_frames
BasePipeline.default_warmup_resolutions
BasePipeline.forward
BasePipeline.infer
BasePipeline.load_standard_components
BasePipeline.load_weights
BasePipeline.vae_adapter_class
```

### `tensorrt_llm/_torch/visual_gen/triton_kernels/bilateral.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
_bilateral_kernel
```

### `tensorrt_llm/_torch/visual_gen/triton_kernels/canny.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
_grad_kernel
_nms_kernel
```

### `tensorrt_llm/_torch/visual_gen/triton_kernels/resize.py`

库存函数 14；已覆盖 10；未覆盖 4。

```text
_area3_kernel
_area_kernel
_cubic_kernel
_linear_kernel
```

### `tensorrt_llm/_torch/visual_gen/utils.py`

库存函数 15；已覆盖 14；未覆盖 1。

```text
SequenceSharder.group
```

### `tensorrt_llm/_torch/weight_sharing/artifact_identity.py`

库存函数 11；已覆盖 10；未覆盖 1。

```text
_raise_walk_error
```

### `tensorrt_llm/_torch/weight_sharing/post_transform_profiles.py`

库存函数 28；已覆盖 26；未覆盖 2。

```text
RootModelIdentity.class_name
RootModelIdentity.matches
```

### `tensorrt_llm/_torch/weight_sharing/weight_manifest.py`

库存函数 30；已覆盖 28；未覆盖 2。

```text
SkippedTensor.from_dict
SkippedTensor.to_dict
```

### `tensorrt_llm/_utils.py`

库存函数 139；已覆盖 100；未覆盖 39。

```text
EnergyMonitor.get_current_energy
QuantModeWrapper.__getitem__
QuantModeWrapper.__repr__
QuantModeWrapper.__str__
_clean_extra_attrs
_setup_gc_nvtx_profiling.<locals>.gc_callback
_setup_gc_nvtx_profiling.<locals>.gc_cleanup
bf16_array
binding_dtype_size
copy_torch_to_numpy
default_gpus_per_node
dim_resolve_negative
dim_to_trt_axes
get_extra_attr
get_init_params
has_extra_attr
mpi_send
np_dtype_to_torch
numpy_fp32_to_bf16
numpy_to_dtype
numpy_to_torch
pad_vocab_size
run_once.<locals>.wrapper
set_extra_attr
set_obj_attrs
set_torch_comm
str_dtype_to_np
to_dict
to_json_file
to_json_string
torch_comm
torch_dtype_to_np
torch_to_numpy
torch_version
trace_func
trace_func.<locals>.wrapper
trace_func.<locals>.wrapper.<locals>.globaltrace
trace_func.<locals>.wrapper.<locals>.localtrace
trt_axes_to_dim
```

### `tensorrt_llm/bench/benchmark/__init__.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
generate_json_report
```

### `tensorrt_llm/bench/benchmark/utils/asynchronous.py`

库存函数 16；已覆盖 15；未覆盖 1。

```text
LlmManager.iteration_worker
```

### `tensorrt_llm/bench/benchmark/utils/processes.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
IterationWriter.__del__
```

### `tensorrt_llm/bench/benchmark/visual_gen.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
_run_concurrent
_run_concurrent.<locals>._generate_one
```

### `tensorrt_llm/bench/dataclasses/configuration.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
DecodingConfig.decoding_mode_validator
```

### `tensorrt_llm/bench/dataclasses/reporting.py`

库存函数 24；已覆盖 17；未覆盖 7。

```text
ReportUtility._query_gpu_info
ReportUtility.convert_to_s
ReportUtility.get_output_tokens
ReportUtility.get_request_info
ReportUtility.per_user_generation_token_throughput_s
ReportUtility.report_statistics
StatsKeeper.register_request
```

### `tensorrt_llm/bench/dataclasses/statistics.py`

库存函数 25；已覆盖 18；未覆盖 7。

```text
BenchmarkStatistics.generation_tokens
BenchmarkStatistics.output_tps_per_w
BenchmarkStatistics.per_user_generation_token_throughput_ns
BenchmarkStatistics.per_user_time_per_output_token_ns
BenchmarkStatistics.per_user_time_to_first_token_ns
BenchmarkStatistics.total_generation_time_ns
BenchmarkStatistics.total_gpu_power
```

### `tensorrt_llm/bench/dataset/prepare_synthetic_data.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
token_unif_dist
```

### `tensorrt_llm/bench/dataset/utils.py`

库存函数 10；已覆盖 5；未覆盖 5。

```text
generate_multi_turn_dataset
generate_multimodal_dataset
get_exponential_dist_delays
get_list_of_delays
get_unif_dist_lengths
```

### `tensorrt_llm/bench/tuning/dataclasses.py`

库存函数 19；已覆盖 13；未覆盖 6。

```text
NemotronHybridConfig.from_hf
NemotronHybridConfig.set_mamba_ssm_cache_dtype
Qwen3HybridConfig.cache_memory_fraction
Qwen3HybridConfig.extra_model_cache_in_gb
Qwen3HybridConfig.from_hf
Qwen3HybridConfig.set_mamba_ssm_cache_dtype
```

### `tensorrt_llm/bench/utils/__init__.py`

库存函数 11；已覆盖 1；未覆盖 10。

```text
_MethodFunctionAdapter.__call__
_MethodFunctionAdapter.__get__
_MethodFunctionAdapter.__init__
command_logger
command_logger.<locals>.inner_wrapper
command_logger.<locals>.inner_wrapper.<locals>.wrapped
detect_methods.<locals>.apply_wrapper
process_error_check
process_error_check.<locals>.runtime_check
run_process
```

### `tensorrt_llm/bench/utils/data.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
update_metadata_for_multimodal
```

### `tensorrt_llm/commands/serve.py`

库存函数 64；已覆盖 62；未覆盖 2。

```text
_report_observed_child_failure
_serve_disagg_fleet
```

### `tensorrt_llm/commands/utils.py`

库存函数 8；已覆盖 4；未覆盖 4。

```text
get_model_path
get_visual_gen_model_type
get_visual_gen_num_gpus
has_registered_llm_architecture
```

### `tensorrt_llm/evaluate/audio_asr.py`

库存函数 20；已覆盖 16；未覆盖 4。

```text
AudioASREvaluator._make_inputs
AudioASREvaluator.generate_samples
_format_worst_sample_report
_truncate_text
```

### `tensorrt_llm/evaluate/cnn_dailymail.py`

库存函数 4；已覆盖 3；未覆盖 1。

```text
CnnDailymail.command
```

### `tensorrt_llm/evaluate/covost2.py`

库存函数 12；已覆盖 3；未覆盖 9。

```text
CoVoST2.__init__
CoVoST2._dump_samples
CoVoST2._ensure_bleu
CoVoST2._sacrebleu_tokenizer
CoVoST2.command
CoVoST2.compute_score
CoVoST2.do_apply_chat_template
CoVoST2.generate_samples
_split_lang_pair
```

### `tensorrt_llm/evaluate/interface.py`

库存函数 10；已覆盖 7；未覆盖 3。

```text
Evaluator.command
Evaluator.compute_score
Evaluator.generate_samples
```

### `tensorrt_llm/evaluate/json_mode_eval.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
JsonModeEval.command
```

### `tensorrt_llm/evaluate/lm_eval.py`

库存函数 60；已覆盖 36；未覆盖 24。

```text
AIME2025.__init__
AIME2025.command
AIME2026.__init__
AIME2026.command
GPQADiamond.command
GPQAExtended.__init__
GPQAExtended.command
GPQAMain.__init__
GPQAMain.command
GSM8K.command
GSM8KInferenceX.command
LmEvalEvaluator.command_harness
LmEvalEvaluator.compute_score
LmEvalEvaluator.generate_samples
LmEvalWrapper._loglikelihood_tokens
LmEvalWrapper.eot_token_id
LmEvalWrapper.loglikelihood_rolling
LmEvalWrapper.tok_encode
LongBenchV1._flatten_task_dict
LongBenchV1.command
MMMU.command
MMMUPro.__init__
MMMUPro.command
MMMUPro.command_harness
```

### `tensorrt_llm/evaluate/lm_eval_tasks/mmmu_pro/utils.py`

库存函数 5；已覆盖 1；未覆盖 4。

```text
_doc_to_text
doc_to_image
doc_to_text
process_results
```

### `tensorrt_llm/evaluate/longbench_v2.py`

库存函数 13；已覆盖 6；未覆盖 7。

```text
LongBenchV2._calculate_metrics
LongBenchV2._save_results
LongBenchV2._truncate_prompt
LongBenchV2.command
LongBenchV2.compute_score
LongBenchV2.evaluate
LongBenchV2.generate_samples
```

### `tensorrt_llm/evaluate/mmlu.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
MMLU.command
MMLU.dowload_dataset
```

### `tensorrt_llm/evaluate/nemo_skills_eval.py`

库存函数 63；已覆盖 2；未覆盖 61。

```text
AALCR._is_correct_judgement
AALCR._judge_fill
AALCR.command
ArenaHard._aggregate_arena
ArenaHard._fallback_win_rate
ArenaHard._judge_score
ArenaHard.command
ArenaHard.evaluate
GPQANemoSkills.command
HLE._is_correct_judgement
HLE._judge_fill
HLE.command
IFBench.command
NemoSkillsEvaluator.__init__
NemoSkillsEvaluator._aggregate
NemoSkillsEvaluator._aggregate_ifbench
NemoSkillsEvaluator._aggregate_ifbench.<locals>._level
NemoSkillsEvaluator._aggregate_repeats
NemoSkillsEvaluator._batched_generate
NemoSkillsEvaluator._bound_generation_for_judge
NemoSkillsEvaluator._build_prompt
NemoSkillsEvaluator._dump_records
NemoSkillsEvaluator._external_judge_generate
NemoSkillsEvaluator._external_judge_generate.<locals>._one
NemoSkillsEvaluator._generation_text
NemoSkillsEvaluator._judge_generate
NemoSkillsEvaluator._judge_sampling_params
NemoSkillsEvaluator._load_rows
NemoSkillsEvaluator._prompt_string
NemoSkillsEvaluator._resolve_dataset_path
NemoSkillsEvaluator._score_correctness
NemoSkillsEvaluator.command_harness
NemoSkillsEvaluator.compute_score
NemoSkillsEvaluator.evaluate
NemoSkillsEvaluator.generate_samples
SciCode._aggregate_scicode
SciCode._generate_one
SciCode._grade_and_aggregate
SciCode._patch_grader_for_py312
SciCode._patch_grader_for_py312.<locals>._DropScipyPinWarning.filter
SciCode._solve_problem
SciCode.command
SciCode.evaluate
SciCode.evaluate.<locals>._sampling_for_repeat
_SelfJudgeEvaluator._aggregate_selfjudge
_SelfJudgeEvaluator._is_correct_judgement
_SelfJudgeEvaluator._judge_fill
_SelfJudgeEvaluator.evaluate
_default_ns_acc_bench_infra
_ensure_sandbox_server
_ensure_sandbox_server.<locals>._is_loopback
_ensure_sandbox_server.<locals>._up
_ns_acc_bench_infra_root
_redirect_ifbench_grader
_redirect_ifbench_grader.<locals>._SubprocessShim.__getattr__
_redirect_ifbench_grader.<locals>._SubprocessShim.run
_redirect_scicode_data
_redirect_scicode_data.<locals>.execute_code
_require_nemo_skills
_resolve_external_judge
autowire_nemo_skills_infra
```

### `tensorrt_llm/evaluate/visual_gen/evaluators.py`

库存函数 11；已覆盖 7；未覆盖 4。

```text
LPIPSVideoEvaluator.__init__
LPIPSVideoEvaluator.close
VideoEvaluator.close
VideoEvaluator.evaluate
```

### `tensorrt_llm/evaluate/visual_gen/generators.py`

库存函数 6；已覆盖 1；未覆盖 5。

```text
VisualGenGeneratorBackend.__init__
VisualGenGeneratorBackend.close
VisualGenGeneratorBackend.default_params
VisualGenGeneratorBackend.generate
VisualGenGeneratorBackend.shutdown
```

### `tensorrt_llm/evaluate/visual_gen/qwen_image_bench.py`

库存函数 22；已覆盖 14；未覆盖 8。

```text
QwenImageBenchEvaluator.__enter__
QwenImageBenchEvaluator.__exit__
QwenImageBenchEvaluator._evaluate_one
QwenImageBenchEvaluator.close
QwenImageBenchEvaluator.evaluate_batch
QwenImageBenchEvaluator.start
build_user_prompt
make_qwen_image_bench_input
```

### `tensorrt_llm/executor/base_worker.py`

库存函数 73；已覆盖 63；未覆盖 10。

```text
BaseWorker._kv_cache_load_serializer
BaseWorker.fetch_kv_cache_load
BaseWorker.fetch_openengine_discovery
BaseWorker.get_cache_transceiver
BaseWorker.get_kv_cache_capacity
BaseWorker.get_kv_cache_load
BaseWorker.get_openengine_discovery
BaseWorker.start
BaseWorker.start_profile
BaseWorker.stop_profile
```

### `tensorrt_llm/executor/executor.py`

库存函数 37；已覆盖 12；未覆盖 25。

```text
CppExecutorError.__init__
CppExecutorError.__str__
GenerationExecutor._create_ray_executor
GenerationExecutor.abort_request
GenerationExecutor.aget_kv_events
GenerationExecutor.aget_stats
GenerationExecutor.check_health
GenerationExecutor.generate
GenerationExecutor.get_cache_transceiver
GenerationExecutor.get_data_transceiver_state
GenerationExecutor.get_disaggregated_params
GenerationExecutor.get_engine_config
GenerationExecutor.get_kv_cache_capacity
GenerationExecutor.get_kv_cache_load
GenerationExecutor.get_kv_events
GenerationExecutor.get_openengine_discovery
GenerationExecutor.get_startup_metrics
GenerationExecutor.get_stats
GenerationExecutor.resource_governor_queue
GenerationExecutor.shutdown
GenerationExecutor.start_profile
GenerationExecutor.stop_profile
GenerationExecutor.submit
GenerationExecutor.wait_first_completed
empty_async_iterable
```

### `tensorrt_llm/executor/ipc.py`

库存函数 35；已覆盖 32；未覆盖 3。

```text
FusedIpcQueue.print_fuse_stats
FusedIpcQueue.setup_sender.<locals>.send_task
ZeroMqQueue.drain
```

### `tensorrt_llm/executor/postproc_worker.py`

库存函数 10；已覆盖 2；未覆盖 8。

```text
PostprocWorker.__init__
PostprocWorker._batched_put
PostprocWorker._mainloop
PostprocWorker._mainloop.<locals>.handle_single_input
PostprocWorker.default_record_creator
PostprocWorker.start
PostprocWorker.start.<locals>.main
postproc_worker_main
```

### `tensorrt_llm/executor/postprocessor_hook.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
PostProcessorHook.__call__
```

### `tensorrt_llm/executor/prompt_tail.py`

库存函数 5；已覆盖 2；未覆盖 3。

```text
PromptTailPolicy._report
_is_set
request_trim_blocker
```

### `tensorrt_llm/executor/proxy.py`

库存函数 65；已覆盖 61；未覆盖 4。

```text
GenerationExecutorProxy.__enter__
GenerationExecutorProxy.__exit__
GenerationExecutorProxy.get_kv_cache_load
GenerationExecutorProxy.get_openengine_discovery
```

### `tensorrt_llm/executor/ray/executor.py`

库存函数 24；已覆盖 3；未覆盖 21。

```text
RayExecutor.__init__
RayExecutor._get_placement_group
RayExecutor._get_placement_group.<locals>._get_default
RayExecutor._get_placement_group.<locals>._get_from_env
RayExecutor._get_placement_group.<locals>._get_from_placement_config
RayExecutor._get_worker_ready_futures
RayExecutor._wait_for_cluster_resource_release
RayExecutor.abort_all_requests
RayExecutor.abort_request
RayExecutor.call_all_ray_workers
RayExecutor.collective_rpc
RayExecutor.collective_rpc_async
RayExecutor.create_workers
RayExecutor.get_engine_config
RayExecutor.get_startup_metrics
RayExecutor.init_workers_async
RayExecutor.init_workers_sync
RayExecutor.report_device_ids
RayExecutor.setup_engine_remote
RayExecutor.setup_engine_remote_async
RayExecutor.start
```

### `tensorrt_llm/executor/ray/gpu_worker.py`

库存函数 28；已覆盖 2；未覆盖 26。

```text
RayGPUWorker.__del__
RayGPUWorker.__enter__
RayGPUWorker.__init__
RayGPUWorker._configure_device
RayGPUWorker._shutdown_rank_local
RayGPUWorker.enqueue_request
RayGPUWorker.setup_engine
RayGPUWorker.sleep
RayGPUWorker.start
RayGPUWorker.wakeup
RayWorkerWrapper.__repr__
RayWorkerWrapper._create_tcp_store
RayWorkerWrapper._inject_worker_extension
RayWorkerWrapper.abort_request
RayWorkerWrapper.call_worker_method
RayWorkerWrapper.enqueue_request
RayWorkerWrapper.ensure_distributed_setup
RayWorkerWrapper.ensure_distributed_setup.<locals>.wrapper
RayWorkerWrapper.has_setup_distributed_env_and_worker
RayWorkerWrapper.physical_to_local_id
RayWorkerWrapper.report_device_id
RayWorkerWrapper.setup_distributed_env_and_worker
RayWorkerWrapper.setup_tcp_store
RayWorkerWrapper.shutdown
RayWorkerWrapper.submit
resolve_obj_by_qualname
```

### `tensorrt_llm/executor/ray/utils.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
control_action_decorator.<locals>.decorator.<locals>.wrapper
unwrap_ray_errors
```

### `tensorrt_llm/executor/request.py`

库存函数 22；已覆盖 17；未覆盖 5。

```text
LoRARequest.name
PromptAdapterRequest.__post_init__
PromptAdapterRequest.adapter_id
PromptAdapterRequest.local_path
PromptAdapterRequest.name
```

### `tensorrt_llm/executor/result.py`

库存函数 61；已覆盖 57；未覆盖 4。

```text
GenerationResult.__hash__
GenerationResult._repr_fields
GenerationResultBase._get_decoder_output_prefix_logprobs
IterationResult.get_results
```

### `tensorrt_llm/executor/rpc/rpc_client.py`

库存函数 32；已覆盖 30；未覆盖 2。

```text
RPCClient._handle_reader_exception
RPCClient.get_server_attr
```

### `tensorrt_llm/executor/rpc/rpc_server.py`

库存函数 24；已覆盖 23；未覆盖 1。

```text
RPCServer.get_attr
```

### `tensorrt_llm/executor/rpc_proxy.py`

库存函数 25；已覆盖 11；未覆盖 14。

```text
GenerationExecutorRpcProxy.__enter__
GenerationExecutorRpcProxy.__exit__
GenerationExecutorRpcProxy.abort_request
GenerationExecutorRpcProxy.aget_kv_events
GenerationExecutorRpcProxy.aget_stats
GenerationExecutorRpcProxy.get_data_transceiver_state
GenerationExecutorRpcProxy.get_engine_config
GenerationExecutorRpcProxy.get_kv_cache_capacity
GenerationExecutorRpcProxy.get_kv_cache_load
GenerationExecutorRpcProxy.get_kv_events
GenerationExecutorRpcProxy.get_openengine_discovery
GenerationExecutorRpcProxy.get_startup_metrics
GenerationExecutorRpcProxy.start_profile
GenerationExecutorRpcProxy.stop_profile
```

### `tensorrt_llm/executor/rpc_worker.py`

库存函数 9；已覆盖 7；未覆盖 2。

```text
RpcWorker.__enter__
RpcWorker.__exit__
```

### `tensorrt_llm/executor/rpc_worker_mixin.py`

库存函数 18；已覆盖 8；未覆盖 10。

```text
RpcWorkerMixin.fetch_kv_cache_events_async
RpcWorkerMixin.fetch_kv_cache_load_async
RpcWorkerMixin.fetch_openengine_discovery_async
RpcWorkerMixin.fetch_stats_async
RpcWorkerMixin.init_postproc_workers
RpcWorkerMixin.init_postproc_workers.<locals>._collect_postproc_outputs
RpcWorkerMixin.init_postproc_workers.<locals>._make_on_postproc_worker_done
RpcWorkerMixin.init_postproc_workers.<locals>._make_on_postproc_worker_done.<locals>._on_postproc_worker_done
RpcWorkerMixin.init_rpc_worker
RpcWorkerMixin.shutdown_postproc_workers
```

### `tensorrt_llm/executor/utils.py`

库存函数 29；已覆盖 18；未覆盖 11。

```text
IntraProcessQueue.__init__
IntraProcessQueue.close
IntraProcessQueue.drain
IntraProcessQueue.get
IntraProcessQueue.poll
IntraProcessQueue.put
ProcessPoolExecutorSession.__init__
ProcessPoolExecutorSession.shutdown
ProcessPoolExecutorSession.submit
ProcessPoolExecutorSession.submit_sync
print_alive_threads
```

### `tensorrt_llm/executor/worker.py`

库存函数 13；已覆盖 11；未覆盖 2。

```text
worker_main.<locals>._print_stacks
worker_main.<locals>.send_profile_ack
```

### `tensorrt_llm/functional.py`

库存函数 28；已覆盖 17；未覆盖 11。

```text
AllReduceParams.has_affine
AllReduceParams.has_bias
AllReduceParams.has_scale
PositionEmbeddingType.choices
PositionEmbeddingType.is_alibi
RopeEmbeddingUtils.create_fake_weight
RopeEmbeddingUtils.create_sinusoidal_positions
RopeEmbeddingUtils.create_sinusoidal_positions_for_cogvlm_attention_plugin
RopeEmbeddingUtils.create_sinusoidal_positions_long_rope_for_attention_plugin
RopeEmbeddingUtils.create_sinusoidal_positions_long_rope_for_attention_plugin.<locals>._calc_mscale
RopeEmbeddingUtils.create_sinusoidal_positions_long_rope_for_attention_plugin.<locals>._compute_sinusoidal_positions
```

### `tensorrt_llm/grpc/openengine/_generated/openengine_pb2_grpc.py`

库存函数 26；已覆盖 4；未覆盖 22。

```text
Control.Abort
Control.GetKvEventSources
Control.GetLoad
Control.GetModelInfo
Control.GetServerInfo
Control.Health
Control.ListLoras
Control.LoadLora
Control.SubscribeKvEvents
Control.UnloadLora
ControlServicer.Abort
ControlServicer.GetKvEventSources
ControlServicer.GetLoad
ControlServicer.GetModelInfo
ControlServicer.GetServerInfo
ControlServicer.Health
ControlServicer.ListLoras
ControlServicer.LoadLora
ControlServicer.SubscribeKvEvents
ControlServicer.UnloadLora
Inference.Generate
InferenceServicer.Generate
```

### `tensorrt_llm/grpc/openengine/server.py`

库存函数 19；已覆盖 18；未覆盖 1。

```text
OpenEngineServer.wait_for_termination
```

### `tensorrt_llm/grpc/smg/request_manager.py`

库存函数 9；已覆盖 8；未覆盖 1。

```text
GrpcRequestManager.get_num_unfinished_requests
```

### `tensorrt_llm/grpc/smg/server.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
launch_smg_server.<locals>.serve_grpc_async.<locals>.signal_handler
```

### `tensorrt_llm/grpc/smg/servicer.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
TrtllmServiceServicer.Embed
TrtllmServiceServicer._error_response
```

### `tensorrt_llm/inputs/evs.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
compute_retention_mask
```

### `tensorrt_llm/inputs/media_io.py`

库存函数 75；已覆盖 68；未覆盖 7。

```text
BaseMediaIO.load_base64
BaseMediaIO.load_bytes
BaseMediaIO.load_file
_DecodedMemoryReservation.check_cancelled
_DecodedMemoryReservation.commit
_DecodedMemoryReservation.release
_DecodedMemoryReservation.shrink
```

### `tensorrt_llm/inputs/multimodal.py`

库存函数 59；已覆盖 57；未覆盖 2。

```text
MultimodalInput.to_tensor
int32_to_hexdigest
```

### `tensorrt_llm/inputs/multimodal_data.py`

库存函数 15；已覆盖 13；未覆盖 2。

```text
BaseModalityData.update_hash
ContentHasher.update
```

### `tensorrt_llm/inputs/prefix_token_cache.py`

库存函数 26；已覆盖 25；未覆盖 1。

```text
OffsetTokenizer.__call__
```

### `tensorrt_llm/inputs/registry.py`

库存函数 82；已覆盖 61；未覆盖 21。

```text
BaseMultimodalDummyInputsBuilder.config
BaseMultimodalDummyInputsBuilder.get_max_mm_encoder_output_embeddings
BaseMultimodalDummyInputsBuilder.get_mm_encoder_attention_metadata_capacity
BaseMultimodalDummyInputsBuilder.get_preferred_media_io_kwargs
BaseMultimodalDummyInputsBuilder.model_path
BaseMultimodalDummyInputsBuilder.tokenizer
BaseMultimodalInputProcessor._attach_multimodal_embeddings_impl
BaseMultimodalInputProcessor.call_with_text_prompt
BaseMultimodalInputProcessor.config
BaseMultimodalInputProcessor.dtype
BaseMultimodalInputProcessor.export_decode_state
BaseMultimodalInputProcessor.get_prefix_token_cache
BaseMultimodalInputProcessor.get_vocab_size
BaseMultimodalInputProcessor.processor
BaseMultimodalInputProcessor.restore_decode_state
BaseMultimodalInputProcessor.tokenizer
DefaultInputProcessor.get_prefix_token_cache
InputProcessor.__call__
MultimodalPlaceholderRegistry.__str__
_InputProcessorCallable.__call__
_warn_prompt_token_cache_ignored
```

### `tensorrt_llm/inputs/utils.py`

库存函数 44；已覆盖 37；未覆盖 7。

```text
ALL_SUPPORTED_AUDIO_MODELS
ALL_SUPPORTED_IMAGE_MODELS
ALL_SUPPORTED_MULTIMODAL_MODELS
ALL_SUPPORTED_VIDEO_MODELS
MultimodalDataTracker.placeholder_modalities
encode_base64_content_from_url
load_audio
```

### `tensorrt_llm/llmapi/_llmapi_process_guard.py`

库存函数 12；已覆盖 5；未覆盖 7。

```text
_arm_parent_death_signal
_exec_payload
_main
_positive_integer
_publish
_remove_ownerless_files
_run.<locals>.request_stop
```

### `tensorrt_llm/llmapi/llm.py`

库存函数 79；已覆盖 74；未覆盖 5。

```text
BaseLLM.__getstate__
BaseLLM._normalize_token_ids
BaseLLM.start_profile
BaseLLM.stop_profile
RequestOutput.__init__
```

### `tensorrt_llm/llmapi/llm_args.py`

库存函数 323；已覆盖 296；未覆盖 27。

```text
AgentTreeConfig._to_pybind
AutoDecodingConfig.supports_backend
BaseSparseAttentionConfig.supports_backend
BaseSparseAttentionConfig.to_sparse_params
DecodingBaseConfig.num_capture_layers
DeepSeekSparseAttentionConfig.supports_backend
DeepSeekV4SparseAttentionConfig.supports_backend
EncodeCudaGraphConfig._generate_cuda_graph_num_tokens
EncodeCudaGraphConfig._generate_cuda_graph_seq_lens
MTPDecodingConfig.num_capture_layers
MiniMaxM3SparseAttentionConfig.get_indices_block_size
MiniMaxM3SparseAttentionConfig.supports_backend
PrometheusMetricsConfig.validate_histogram_buckets
PybindMirror._to_pybind
QSASparseAttentionConfig.get_indices_block_size
QSASparseAttentionConfig.supports_backend
RayPlacementConfig.validate_ray_placement
RocketSparseAttentionConfig.get_indices_block_size
RocketSparseAttentionConfig.supports_backend
SeqLenAwareSparseAttentionConfig.get_indices_block_size
SkipSoftmaxAttentionConfig.supports_backend
TorchLlmArgs.warn_on_unstable_feature_usage
_MoeLoadBalancerConfigBase.ep_rank
_MoeLoadBalancerConfigBase.ep_size
_ParallelConfig.devices
_check_removed_sampler_type
_get_trtllm_json_schema_extra
```

### `tensorrt_llm/llmapi/mpi_session.py`

库存函数 68；已覆盖 52；未覆盖 16。

```text
MPINodeState.is_initialized
MpiCommSession.__reduce__
MpiCommSession.abort
MpiCommSession.get_comm
MpiPoolSession.__reduce__
MpiPoolSession.abort
MpiPoolSession.get_comm
MpiSession.abort
MpiSession.is_comm_session
MpiSession.release_exit_joins
MpiSession.shutdown
MpiSession.submit
MpiSession.submit_sync
RemoteMpiCommSessionClient.abort
RemoteMpiCommSessionClient.shutdown_abort
find_free_port
```

### `tensorrt_llm/llmapi/reasoning_parser.py`

库存函数 57；已覆盖 55；未覆盖 2。

```text
BaseReasoningParser.parse
BaseReasoningParser.parse_delta
```

### `tensorrt_llm/llmapi/rlhf_utils.py`

库存函数 10；已覆盖 4；未覆盖 6。

```text
WorkerExtension.check_weights_updated
WorkerExtension.reset_prefix_cache
WorkerExtension.start_profile
WorkerExtension.stop_profile
WorkerExtension.update_weights
WorkerExtension.wait_for_engine_idle
```

### `tensorrt_llm/llmapi/serialization.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
load
```

### `tensorrt_llm/llmapi/thinking_budget.py`

库存函数 12；已覆盖 10；未覆盖 2。

```text
_ChainedLogitsProcessor.__call__
_ChainedLogitsProcessor.__init__
```

### `tensorrt_llm/llmapi/tracer.py`

库存函数 9；已覆盖 4；未覆盖 5。

```text
VizTracer.register_exit
VizTracer.start
log_sparse
log_sparse.<locals>.wrapper
set_global_tracer
```

### `tensorrt_llm/llmapi/tracing.py`

库存函数 16；已覆盖 12；未覆盖 4。

```text
add_event
inject_trace_headers
insufficient_request_metrics_warning
log_tracing_disabled_warning
```

### `tensorrt_llm/llmapi/utils.py`

库存函数 67；已覆盖 49；未覆盖 18。

```text
ApiParamTagger.__call__
ApiParamTagger._amend_pydantic_field_description_with_tags
ApiParamTagger._process_pydantic_model
AsyncQueue.full
GpuArch.get_arch
GpuArch.is_post_ampere
GpuArch.is_post_hopper
_SyncQueue.full
download_hf_pretrained_config
file_with_glob_exists
file_with_suffix_exists
get_current_location
get_directory_size_in_gb
get_gpu_arch
is_directory_empty
set_docstring
set_docstring.<locals>.decorator
tag_llm_params
```

### `tensorrt_llm/logger.py`

库存函数 28；已覆盖 26；未覆盖 2。

```text
Logger.critical_once
set_level
```

### `tensorrt_llm/mapping.py`

库存函数 63；已覆盖 51；未覆盖 12。

```text
Mapping.build_mesh
Mapping.cp_group_pg
Mapping.moe_tp_group_pg
Mapping.pp_group_pg
Mapping.tp_group_pg
MappingBase.ep_experts
MappingBase.has_moe_cluster
MappingBase.has_moe_ep
MappingBase.has_moe_tp
MappingBase.moe_cluster_group
MpiTopology.moe_cluster_group
MpiTopology.moe_tp_group
```

### `tensorrt_llm/media/encoding.py`

库存函数 23；已覆盖 22；未覆盖 1。

```text
_VideoEncoder.encode_video
```

### `tensorrt_llm/metrics/perf.py`

库存函数 18；已覆盖 17；未覆盖 1。

```text
PerfMetricsRecord.categories
```

### `tensorrt_llm/models/convert_utils.py`

库存函数 22；已覆盖 3；未覆盖 19。

```text
apply_smoothing
dup_kv_bias
dup_kv_weight
get_bias
get_tllm_linear_weight
get_weight
get_weight_and_bias
has_safetensors
infer_dtype
iterate_shard_files
retrieved_layer_index_from_name
smooth_gemm
smooth_gemm_fc1_gate
split
split_matrix_tp
split_qkv_bias_tp
split_qkv_tp
weight_only_quantize
weight_only_quantize_dict
```

### `tensorrt_llm/models/modeling_utils.py`

库存函数 38；已覆盖 7；未覆盖 31。

```text
Gemma2ConfigGroup.keys
Gemma3ConfigGroup.keys
LayerQuantConfig._get_modelopt_qformat
LayerQuantConfig._get_quant_cfg
LayerQuantConfig.auto_quant_list
LayerQuantConfig.auto_quant_mode
LayerQuantConfig.from_dict
LayerQuantConfig.layer_quant_mode
LayerQuantConfig.model_post_init
LayerQuantConfig.quant_mode
PretrainedConfig.__init__
PretrainedConfig._get_quant_cfg
PretrainedConfig.for_each_rank
PretrainedConfig.from_checkpoint
PretrainedConfig.from_dict
PretrainedConfig.from_json_file
PretrainedConfig.get_config_group
PretrainedConfig.has_config_group
PretrainedConfig.kv_dtype
PretrainedConfig.quant_algo
PretrainedConfig.quant_mode
PretrainedConfig.set_if_not_exist
PretrainedConfig.set_rank
PretrainedConfig.to_dict
PretrainedConfig.to_json_file
PretrainedConfig.to_layer_quant_config
QuantConfig._get_modelopt_kv_cache_dtype
QuantConfig._get_modelopt_qformat
QuantConfig._get_quant_cfg
QuantConfig.from_dict
SpeculativeDecodingMode.from_arguments
```

### `tensorrt_llm/profiler.py`

库存函数 20；已覆盖 16；未覆盖 4。

```text
Timer.summary
print_device_memory_usage
print_host_memory_usage
summary
```

### `tensorrt_llm/quantization/mode.py`

库存函数 45；已覆盖 34；未覆盖 11。

```text
QuantMode.has_act_and_weight_quant
QuantMode.has_act_or_weight_quant
QuantMode.has_mxfp4
QuantMode.has_nvfp4_plus
QuantMode.has_weight_quant
QuantMode.is_int8_weight_only_per_group
QuantMode.is_qserve_w4a8
QuantMode.set_fp8_qdq
QuantMode.set_fp8_rowwise
QuantMode.to_dict
QuantMode.use_qserve
```

### `tensorrt_llm/quantization/utils/fp8_quantize.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_fp8_1x128_quantize_kernel
```

### `tensorrt_llm/quantization/utils/fp8_utils.py`

库存函数 23；已覆盖 18；未覆盖 5。

```text
_per_token_quant_and_transform_kernel
_resmooth_kernel
_silu_and_mul_post_quant_kernel
get_m_alignment_for_contiguous_layout
per_token_cast_to_fp8_e8m0
```

### `tensorrt_llm/quantization/utils/marlin_utils.py`

库存函数 9；已覆盖 7；未覆盖 2。

```text
marlin_make_workspace
marlin_permute_bias
```

### `tensorrt_llm/sampling_params.py`

库存函数 26；已覆盖 24；未覆盖 2。

```text
BatchedLogitsProcessor.__call__
LogitsProcessor.__call__
```

### `tensorrt_llm/scaffolding/controller.py`

库存函数 29；已覆盖 15；未覆盖 14。

```text
BestOfNController.__init__
BestOfNController.clone
BestOfNController.process
BestOfNController.select_best
Controller.process
NativeChatController.__init__
NativeChatController.process
NativeRewardController.__init__
NativeRewardController.process
PRMController.__init__
PRMController._calc_last_token_score
PRMController._calc_steps_score
PRMController.process
ParallelProcess.from_generators
```

### `tensorrt_llm/scaffolding/execution_scope.py`

库存函数 4；已覆盖 2；未覆盖 2。

```text
ExecutionScope.branch_path_list
ExecutionScope.with_branch_path
```

### `tensorrt_llm/scaffolding/load_generation_strategy.py`

库存函数 24；已覆盖 11；未覆盖 13。

```text
ConcurrentStrategy.__str__
ConstantRateStrategy.__str__
LoadGenerationStrategy.__str__
LoadGenerationStrategy._generate_request_times
PoissonRateStrategy.__str__
PoissonWarmupStrategy.__str__
PoissonWarmupStrategy._compute_arrival_times
PoissonWarmupStrategy._generate_request_times
PoissonWarmupStrategy.get_semaphore
UniformWarmupStrategy.__str__
UniformWarmupStrategy._compute_arrival_times
UniformWarmupStrategy._generate_request_times
UniformWarmupStrategy.get_semaphore
```

### `tensorrt_llm/scaffolding/math_utils.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
extract_answer_with_regex
```

### `tensorrt_llm/scaffolding/result.py`

库存函数 11；已覆盖 8；未覆盖 3。

```text
ScaffoldingResult.__aiter__
ScaffoldingResult.__anext__
ScaffoldingResult.__await__
```

### `tensorrt_llm/scaffolding/scaffolding_llm.py`

库存函数 26；已覆盖 24；未覆盖 2。

```text
ScaffoldingLlm.__enter__
ScaffoldingLlm.__exit__
```

### `tensorrt_llm/scaffolding/task.py`

库存函数 28；已覆盖 13；未覆盖 15。

```text
AssistantMessage.__str__
AssistantMessage.to_dict
ChatTask.add_messages
ChatTask.last_assistant_content
OpenAIToolDescription.to_dict
RoleMessage.__repr__
RoleMessage.__str__
StreamGenerationTask.create_from_generation_task
Task.create_from_prompt
Task.create_scaffolding_output
Task.create_scaffolding_output_stream
ToolDescription.__init__
ToolDescription.to_dict
ToolMessage.__str__
ToolMessage.to_dict
```

### `tensorrt_llm/scaffolding/task_collection.py`

库存函数 85；已覆盖 25；未覆盖 60。

```text
ChatTokenCounter.__init__
ChatTokenCounter.after_yield
ChatTokenCounter.before_yield
ChatTokenCounter.get_global_info
ExecutionTracer.__init__
ExecutionTracer._build_yield_event
ExecutionTracer._extract_tool_calls
ExecutionTracer._get_branch_path
ExecutionTracer._get_conversation_id
ExecutionTracer._is_task_already_traced
ExecutionTracer._mark_task_tracing_end
ExecutionTracer._mark_task_tracing_start
ExecutionTracer._message_event_from_role_message
ExecutionTracer._serialize_mcp_args_for_trace
ExecutionTracer._serialize_role_message_for_trace
ExecutionTracer._tool_call_event_for_tool_message
ExecutionTracer._tool_call_meta_by_id_from_messages
ExecutionTracer._tool_calls_detail_from_message
ExecutionTracer.after_yield
ExecutionTracer.before_yield
ExecutionTracer.export_trace
ExecutionTracer.on_parallel_end
ExecutionTracer.on_parallel_start
GenerationTokenCounter.__init__
GenerationTokenCounter.after_yield
GenerationTokenCounter.before_yield
QueryCollector.__init__
QueryCollector.after_yield
QueryCollector.get_global_info
TaskCollection.before_yield
TaskCollection.get_global_info
TaskMetricsCollector.__init__
TaskMetricsCollector._compute_stats
TaskMetricsCollector._is_task_already_profiled
TaskMetricsCollector._mark_task_profiling_end
TaskMetricsCollector._mark_task_profiling_start
TaskMetricsCollector._print_task_info
TaskMetricsCollector._print_task_start
TaskMetricsCollector._serialize_mcp_args
TaskMetricsCollector._serialize_message
TaskMetricsCollector._should_process_task
TaskMetricsCollector.after_yield
TaskMetricsCollector.before_yield
TaskMetricsCollector.export_to_json
TaskMetricsCollector.get_all_records
TaskMetricsCollector.get_global_info
TaskMetricsCollector.get_statistics
TaskMetricsCollector.print_summary
TaskMetricsCollector.reset
TaskTimer.__init__
TaskTimer.after_yield
TaskTimer.before_yield
TaskTimer.get_global_info
_balance_system_tokens
_collect_tokenizable_events
tokenize_trace_scope
tokenize_trace_scope.<locals>.decorator
tokenize_trace_scope.<locals>.decorator.<locals>.new_process
tokenize_trace_scope.<locals>.decorator.<locals>.new_process.<locals>.wrapper
with_execution_tracing
```

### `tensorrt_llm/scaffolding/utils.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
_TaggedSystemPrompt.__reduce__
_TaggedSystemPrompt.format_map
```

### `tensorrt_llm/scaffolding/worker.py`

库存函数 53；已覆盖 29；未覆盖 24。

```text
ApiaryMCPWorker._ConnState.__init__
ApiaryMCPWorker._ToolCall.__init__
ApiaryMCPWorker._ToolCall.set_result
ApiaryMCPWorker.__init__
ApiaryMCPWorker._conn_loop
ApiaryMCPWorker._ensure_primitives
ApiaryMCPWorker._get_conn
ApiaryMCPWorker._root_request_id
ApiaryMCPWorker.async_shutdown
ApiaryMCPWorker.call_handler
ApiaryMCPWorker.on_scope_end
ApiaryMCPWorker.release_connection
ApiaryMCPWorker.set_scope_params
ApiaryMCPWorker.shutdown
MCPWorker.init_with_urls
OpenaiWorker._reasoning_content_from_thinking_blocks
OpenaiWorker.send_kv_cache_truncate_tokens
OpenaiWorker.tokenize_handler
TRTLLMWorker.stream_generation_handler
TRTLLMWorker.streaming_generate_helper
Worker.__enter__
Worker.__exit__
Worker.register_task_handler
add_attr_if_not_none
```

### `tensorrt_llm/serve/_frontend_processes.py`

库存函数 14；已覆盖 13；未覆盖 1。

```text
_ephemeral_port_range
```

### `tensorrt_llm/serve/_visual_gen_errors.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
_ErrorResponseFactory.__call__
```

### `tensorrt_llm/serve/anthropic_protocol.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
current_timestamp
```

### `tensorrt_llm/serve/cluster_storage.py`

库存函数 92；已覆盖 82；未覆盖 10。

```text
ClusterStorage.__init__
ClusterStorage.delete
ClusterStorage.expire
ClusterStorage.get
ClusterStorage.get_prefix
ClusterStorage.set
ClusterStorage.unwatch
ClusterStorage.watch
HttpClusterStorageClient.unwatch
HttpClusterStorageClient.watch
```

### `tensorrt_llm/serve/coordinator_server.py`

库存函数 19；已覆盖 17；未覆盖 2。

```text
CoordinatorServer.renew
serve_coordinator
```

### `tensorrt_llm/serve/disagg_auto_scaling.py`

库存函数 32；已覆盖 31；未覆盖 1。

```text
get_local_ip
```

### `tensorrt_llm/serve/disagg_coordinator.py`

库存函数 69；已覆盖 58；未覆盖 11。

```text
CoordinatorClient.enable_output_length_prediction_in_select
CoordinatorClient.proxy_cluster_storage_request
DisaggCoordinator.cluster_info
DisaggCoordinator.ctx_router
DisaggCoordinator.gen_router
DisaggCoordinator.get_disagg_request_id
DisaggCoordinator.is_ready
DisaggCoordinator.predict_output_length
DisaggCoordinator.record_output_lengths
DisaggCoordinator.start
DisaggCoordinator.stop
```

### `tensorrt_llm/serve/harmony_adapter.py`

库存函数 46；已覆盖 41；未覆盖 5。

```text
HarmonyAdapter.get_stream_debug_info
HarmonyAdapter.openai_to_harmony_prompt
HarmonyAdapter.stream_harmony_tokens_to_openai_deltas
HarmonyStreamState.finalize_request
HarmonyStreamState.get_debug_info
```

### `tensorrt_llm/serve/metadata_server.py`

库存函数 16；已覆盖 11；未覆盖 5。

```text
JsonDictionary.remove
RemoteDictionary.get
RemoteDictionary.keys
RemoteDictionary.put
RemoteDictionary.remove
```

### `tensorrt_llm/serve/openai_client.py`

库存函数 27；已覆盖 20；未覆盖 7。

```text
OpenAIClient._finish_request
OpenAIClient._send_request
OpenAIClient.check_ready
OpenAIClient.get_json
OpenAIClient.post_json
OpenAIClient.shutdown
_SerializableRequest.model_dump
```

### `tensorrt_llm/serve/openai_disagg_server.py`

库存函数 35；已覆盖 31；未覆盖 4。

```text
OpenAIDisaggServer.__init__.<locals>.validation_exception_handler
OpenAIDisaggServer._create_client.<locals>.disagg_id_generator
OpenAIDisaggServer.anthropic_count_tokens
OpenAIDisaggServer.version
```

### `tensorrt_llm/serve/openai_protocol.py`

库存函数 55；已覆盖 53；未覆盖 2。

```text
CompletionRequest.check_thinking_token_budget
ResponsesRequest.check_thinking_token_budget
```

### `tensorrt_llm/serve/openai_server.py`

库存函数 134；已覆盖 124；未覆盖 10。

```text
OpenAIServer._apply_shared_clock_offset
OpenAIServer._synthetic_request.<locals>.receive
OpenAIServer.anthropic_cancel_batch
OpenAIServer.anthropic_delete_batch
OpenAIServer.data_transceiver_state
OpenAIServer.openai_responses_delete_response
OpenAIServer.rerank
OpenAIServer.rerank_v2
_preprocess_in_conversation
_warn_unresolvable_thinking_once
```

### `tensorrt_llm/serve/output_length_predictor.py`

库存函数 69；已覆盖 64；未覆盖 5。

```text
GbmFitter.busy
GbmFitter.poll
GbmFitter.start
OnlineLengthPredictor.stats_line
SubprocessGbmFitter.busy
```

### `tensorrt_llm/serve/reranker.py`

库存函数 7；已覆盖 6；未覆盖 1。

```text
_Tokenizer.encode
```

### `tensorrt_llm/serve/resource_governor.py`

库存函数 11；已覆盖 8；未覆盖 3。

```text
ResourceGovernor._truncate_kv_cache_harmony
ResourceGovernor._truncate_kv_cache_tokens
ResourceGovernor.close
```

### `tensorrt_llm/serve/responses_utils.py`

库存函数 147；已覆盖 133；未覆盖 14。

```text
ConversationHistoryStore._pop_conversation
ConversationHistoryStore._pop_conversation_by_conversation_id.<locals>.get_first_conversation_range_harmony
ConversationHistoryStore._pop_response
ConversationHistoryStore.pop_response
PeriodicLatencyLogger.record.<locals>.percentile
ResponseHooks.on_ctx_dispatch
ResponseHooks.on_ctx_resp
ResponseHooks.on_disagg_request_id
ResponseHooks.on_first_token
ResponseHooks.on_perf_metrics
ResponseHooks.on_req_begin
ResponseHooks.on_resp_done
ResponsesStreamingEventsHelper.content_index_increment
_decode_tokens
```

### `tensorrt_llm/serve/responses_web_search.py`

库存函数 12；已覆盖 6；未覆盖 6。

```text
history_items_for_search
log_search
query_from_call
search_call_item
server_executes_web_search
web_search_function_definition
```

### `tensorrt_llm/serve/router.py`

库存函数 180；已覆盖 160；未覆盖 20。

```text
ConversationRouter._evict_oldest_session
CoordinatorDelegatingRouter._on_servers_updated
KvCacheAwareRouter.finish_request_by_id
KvCacheAwareRouter.get_next_server_by_key
KvCacheAwareServerState.get_hash_algo
KvCacheAwareServerState.num_active_tokens
LoadBalancingMixin._unstage_server
PooledLoadBalancingRouter._redispatch
Router._check_server_health
Router._filter_servers_by_role
Router._get_server_type
Router._is_matching_role
Router._monitor_servers
Router._on_servers_updated
Router.finish_request
Router.get_next_server
Router.renew_request
Router.set_unpredicted_priority
Router.start_server_monitoring
Router.stop_server_monitoring
```

### `tensorrt_llm/serve/router_utils.py`

库存函数 28；已覆盖 26；未覆盖 2。

```text
BlockHashMixin._tokenize_and_compute_block_hashes
BlockHashMixin._tokenize_and_compute_block_hashes_with_salt
```

### `tensorrt_llm/serve/scripts/backend_request_func.py`

库存函数 5；已覆盖 3；未覆盖 2。

```text
async_request_openai_chat_completions
async_request_trt_llm
```

### `tensorrt_llm/serve/scripts/benchmark_dataset.py`

库存函数 40；已覆盖 11；未覆盖 29。

```text
AIMODataset.sample
ASRDataset.sample
BenchmarkDataset.apply_multimodal_chat_transformation
BenchmarkDataset.load_data
BenchmarkDataset.maybe_oversample_requests
BenchmarkDataset.sample
BurstGPTDataset.__init__
BurstGPTDataset._sample_loaded_data
BurstGPTDataset.load_data
BurstGPTDataset.sample
ConversationDataset.sample
CustomDataset.__init__
CustomDataset.load_data
CustomDataset.sample
HuggingFaceDataset.__init__
HuggingFaceDataset.load_data
InstructCoderDataset.sample
MTBenchDataset.sample
RandomImageDataset.__init__
RandomImageDataset.sample
ShareGPTDataset.__init__
ShareGPTDataset.load_data
ShareGPTDataset.sample
SonnetDataset.__init__
SonnetDataset.load_data
SonnetDataset.sample
VisionArenaDataset.sample
is_valid_sequence
process_image
```

### `tensorrt_llm/serve/scripts/benchmark_serving.py`

库存函数 15；已覆盖 13；未覆盖 2。

```text
parse_goodput
save_to_pytorch_benchmark_format
```

### `tensorrt_llm/serve/scripts/benchmark_visual_gen.py`

库存函数 50；已覆盖 40；未覆盖 10。

```text
_cli_json
_dispatch_image
_document_from_args
_merge_extra_params
_record_json
_resize_requests
_resolve_action_file
_resolve_prompt_file
_resolve_reference
_write_media
```

### `tensorrt_llm/serve/scripts/time_breakdown/time_breakdown.py`

库存函数 19；已覆盖 15；未覆盖 4。

```text
RequestTimeBreakdown._generate_stats_html
RequestTimeBreakdown.compute_statistics
RequestTimeBreakdown.export_statistics_json
main
```

### `tensorrt_llm/serve/seat_pool.py`

库存函数 24；已覆盖 23；未覆盖 1。

```text
_pct
```

### `tensorrt_llm/serve/tool_parser/base_tool_parser.py`

库存函数 17；已覆盖 13；未覆盖 4。

```text
BaseToolParser.detect_and_parse
BaseToolParser.has_tool_call
BaseToolParser.parse_partial
BaseToolParser.structure_info
```

### `tensorrt_llm/serve/tool_parser/deepseekv31_parser.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
DeepSeekV31Parser.structure_info
```

### `tensorrt_llm/serve/tool_parser/glm4_parser.py`

库存函数 20；已覆盖 19；未覆盖 1。

```text
Glm4ToolParser.structure_info
```

### `tensorrt_llm/serve/tool_parser/minimax_m2_parser.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
MiniMaxM2ToolParser.structure_info
```

### `tensorrt_llm/serve/tool_parser/minimax_m3_parser.py`

库存函数 8；已覆盖 6；未覆盖 2。

```text
MiniMaxM3ToolParser.structure_info
MiniMaxM3ToolParser.supports_structural_tag
```

### `tensorrt_llm/serve/tool_parser/poolside_v1_parser.py`

库存函数 13；已覆盖 11；未覆盖 2。

```text
PoolsideV1ToolParser.structure_info
_convert_number
```

### `tensorrt_llm/serve/tool_parser/qwen3_coder_parser.py`

库存函数 22；已覆盖 20；未覆盖 2。

```text
Qwen3CoderToolParser.parse_partial
Qwen3CoderToolParser.structure_info
```

### `tensorrt_llm/serve/visual_gen_utils.py`

库存函数 25；已覆盖 24；未覆盖 1。

```text
AsyncDictStore.update_fields
```

### `tensorrt_llm/serve/web_search.py`

库存函数 25；已覆盖 13；未覆盖 12。

```text
WebSearchSession.budget_notice
_domain_matches
_domain_of
_fetch
_search_brave
_search_mojeek
_search_searxng
_search_tavily
_search_wikipedia
_strip_html
filter_results
run_web_search
```

### `tensorrt_llm/tokenizer/deepseek_v32/encoding.py`

库存函数 15；已覆盖 9；未覆盖 6。

```text
_read_until_stop
decode_dsml_to_arguments
decode_dsml_to_arguments.<locals>._decode_value
parse_message_from_completion_text
parse_tool_calls
tool_calls_to_openai_format
```

### `tensorrt_llm/tokenizer/deepseek_v32/tokenizer.py`

库存函数 21；已覆盖 2；未覆盖 19。

```text
DeepseekV32Tokenizer.__call__
DeepseekV32Tokenizer.__hash__
DeepseekV32Tokenizer.__len__
DeepseekV32Tokenizer.all_special_ids
DeepseekV32Tokenizer.all_special_tokens
DeepseekV32Tokenizer.bos_token_id
DeepseekV32Tokenizer.convert_ids_to_tokens
DeepseekV32Tokenizer.convert_tokens_to_string
DeepseekV32Tokenizer.decode
DeepseekV32Tokenizer.encode
DeepseekV32Tokenizer.eos_token_id
DeepseekV32Tokenizer.from_pretrained
DeepseekV32Tokenizer.get_added_vocab
DeepseekV32Tokenizer.get_vocab
DeepseekV32Tokenizer.is_fast
DeepseekV32Tokenizer.max_token_id
DeepseekV32Tokenizer.pad_token_id
DeepseekV32Tokenizer.truncation_side
DeepseekV32Tokenizer.vocab_size
```

### `tensorrt_llm/tokenizer/tokenizer.py`

库存函数 43；已覆盖 38；未覆盖 5。

```text
TokenizerBase.llguidance_tokenizer
TokenizerBase.xgrammar_tokenizer_info
TransformersTokenizer.batch_encode_plus
TransformersTokenizer.get_added_vocab
TransformersTokenizer.save_pretrained
```

### `tensorrt_llm/tools/dspark_profile_costs.py`

库存函数 13；已覆盖 11；未覆盖 2。

```text
_profile_k
load_prompts
```

### `tensorrt_llm/tools/importlib_utils.py`

库存函数 3；已覆盖 1；未覆盖 2。

```text
import_custom_module
import_custom_module_from_file
```

### `tensorrt_llm/tools/layer_wise_benchmarks/calibrator.py`

库存函数 27；已覆盖 11；未覆盖 16。

```text
Calibrator._collect_slots
Calibrator._init_collect_mode
Calibrator._init_replay_mode
Calibrator._replay_slots
Calibrator._save_collected_data
Calibrator._verify_replay_metadata
Calibrator._wrap_layer_forward
Calibrator._wrap_layer_forward.<locals>.make_forward
Calibrator._wrap_layer_forward.<locals>.make_forward.<locals>.forward
Calibrator._wrap_model_forward_for_collect
Calibrator._wrap_model_forward_for_collect.<locals>.make_forward
Calibrator._wrap_model_forward_for_collect.<locals>.make_forward.<locals>.forward
Calibrator._wrap_model_forward_for_replay
Calibrator._wrap_model_forward_for_replay.<locals>.make_forward
Calibrator._wrap_model_forward_for_replay.<locals>.make_forward.<locals>.forward
Calibrator.get_replay_iteration_range
```

### `tensorrt_llm/tools/layer_wise_benchmarks/runner.py`

库存函数 33；已覆盖 23；未覆盖 10。

```text
Runner.scaled_from_ctx.<locals>.make_load_pretrained_config
Runner.scaled_from_ctx.<locals>.make_load_pretrained_config.<locals>.load_pretrained_config
get_all_to_one_selection
get_balanced_rank_imbalanced_expert_selection
get_balanced_selection_impl_random
get_num_balanced_tokens
get_token_final_scales
make_balanced_run_moe
make_balanced_run_moe.<locals>.balanced_run_moe
test_get_balanced_selection
```

### `tensorrt_llm/tools/profiler/host_profile_tools/host_profiler.py`

库存函数 27；已覆盖 22；未覆盖 5。

```text
HostProfiler.get_stats_string
HostProfiler.profile
_get_all_classes_from_module
_get_all_functions_from_module
_get_all_methods_from_class
```

### `tensorrt_llm/visual_gen/__init__.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
__dir__
```

### `tensorrt_llm/visual_gen/args.py`

库存函数 22；已覆盖 20；未覆盖 2。

```text
VisualGenArgs.from_dict
VisualGenArgs.to_dict
```

### `tensorrt_llm/visual_gen/sparse_attention.py`

库存函数 11；已覆盖 9；未覆盖 2。

```text
BaseSparseAttentionConfig.to_sparse_metadata_params
BaseSparseAttentionConfig.to_sparse_params
```

### `tensorrt_llm/visual_gen/visual_gen.py`

库存函数 21；已覆盖 19；未覆盖 2。

```text
VisualGen.ref_slot_specs
VisualGenResult.cancel
```

### `tensorrt_llm/watermark/sampling.py`

库存函数 7；已覆盖 4；未覆盖 3。

```text
_gumbel_reduce_kernel
_gumbel_scan_kernel
_philox_uniform
```

### `tensorrt_llm/watermark/state_kernels.py`

库存函数 6；已覆盖 2；未覆盖 4。

```text
_commit_kernel
_context_seed
_mix64
_plan_kernel
```

### `tensorrt_llm/watermark/step_kernel.py`

库存函数 3；已覆盖 2；未覆盖 1。

```text
_step_kernel
```

### `tests/integration/defs/conftest.py`

库存函数 83；已覆盖 44；未覆盖 39。

```text
_get_s3_output
bert_example_root
cached_in_llm_models_root.<locals>.wrapper.<locals>.decorated
cmodel_base_dir
cmodel_dir
deselect_by_regex
deselect_by_test_model_suites
eval_venv
gpt_example_root
integration_path
internlm2_example_root
llm_backend_root
llm_datasets_root
llm_gpt2_santacoder_model_root
llm_rouge_root
llm_session_data_writer
medusa_example_root
mmlu_dataset_root
parametrize_with_ids
pytest_addoption
pytest_collection_modifyitems
pytest_configure
pytest_generate_tests
qcache_dir
qcache_dir_without_install_package
remove_file
serve_test_root
skip_fp8_pre_ada
test_case
test_root
timeout_from_command_line
timeout_from_marker
timeout_manager
tritonserver_test_root
trt_gpu_clock_lock
trt_performance_cache_fpath
trt_performance_cache_name
wget
whisper_example_root
```

### `tests/test_common/_session_utils.py`

库存函数 6；已覆盖 5；未覆盖 1。

```text
_isinstance_transparent_shim.<locals>._SeamMeta.__repr__
```

### `tests/test_common/magic_import.py`

库存函数 22；已覆盖 20；未覆盖 2。

```text
MagicFinder.reset
MagicFinder.trace
```

### `tests/test_common/magic_import_hooks.py`

库存函数 13；已覆盖 12；未覆盖 1。

```text
_write_banner
```

### `tests/test_common/s3_output.py`

库存函数 38；已覆盖 37；未覆盖 1。

```text
_main
```

### `tests/test_common/serve/cpu_tests.py`

库存函数 5；已覆盖 4；未覆盖 1。

```text
main
```

### `tests/test_common/serve/fakes.py`

库存函数 35；已覆盖 27；未覆盖 8。

```text
FakeEncodeOnlyLLM.encode
FakeExecutor.fatal_error
ScriptedRequestOutput.abort
UserProbeMiddleware.__call__
UserProbeMiddleware.__init__
_FakeGeneratorMixin.get_data_transceiver_state
_FakeGeneratorMixin.get_stats_async
_FakeGeneratorMixin.llm_id
```

### `tests/test_common/serve/fakes_visual_gen.py`

库存函数 2；已覆盖 1；未覆盖 1。

```text
FakeVisualGen.generate_async
```

### `tests/test_common/serve/routes.py`

库存函数 16；已覆盖 12；未覆盖 4。

```text
build_reference_data
format_failure
main
write_reference
```

### `tests/test_common/session_reuse.py`

库存函数 26；已覆盖 22；未覆盖 4。

```text
SessionReuseCache.install_pool_factory_if_loaded.<locals>.rpc_factory
_PoolSession.release_exit_joins
_PoolSession.shutdown
_worker_start_time
```

### `tests/unittest/_torch/pyexecutor/kv_cache/sharing/conftest.py`

库存函数 75；已覆盖 73；未覆盖 2。

```text
held_stream.<locals>.trip
reclaim
```

### `tests/unittest/_torch/ray_orchestrator/conftest.py`

库存函数 9；已覆盖 4；未覆盖 5。

```text
_install_finegrained_fp8_stub.<locals>._patched_load_deepgemm_kernel
_install_finegrained_fp8_stub.<locals>.fp8_act_quant
_install_finegrained_fp8_stub.<locals>.w8a8_fp8_matmul
_install_finegrained_fp8_stub.<locals>.w8a8_fp8_matmul_batched
_install_finegrained_fp8_stub.<locals>.w8a8_fp8_matmul_grouped
```

### `tests/unittest/conftest.py`

库存函数 26；已覆盖 21；未覆盖 5。

```text
_maybe_force_ray.<locals>.wrap_llm
_maybe_force_ray.<locals>.wrap_llm.<locals>.LLMProxy.__init__
_warn_rerunfailures_cache_moved
dump_threads
setup_ray_cluster
```

### `tests/unittest/grpc/openengine/conftest.py`

库存函数 8；已覆盖 7；未覆盖 1。

```text
FakeServicerContext.add_done_callback
```

## 数据来源与校验

- 最终全量包：[cbts_pystart_report.tar.gz](https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local/llm-artifacts/LLM/main/L0_PostMerge/3092/cbts-coverage/cbts_pystart_report.tar.gz)；其中使用 `cbts_touchmap.sqlite`。
- 压缩包字节数：244,829,513；SHA-256：`71a7cd2ced5e4c7d8ca42f27f63fd4d94b64099af3270a7c95d8e59232ba7d07`，已与 Artifactory 服务端校验值核对一致。
- SQLite 格式：`storage_schema=compact-v1`，`schema_version=5`；以只读方式查询。
- 最终覆盖分子交叉校验：3,320 个文件、42,845 个函数，与构建日志和最终 HTML 报告一致。
- 独立复核使用 `touch_rows` 视图与同 commit AST 库存执行 SQL `EXCEPT`；并核对附录中文件和函数唯一键的数量，确保清单完整且不重复。

复核逻辑：

```sql
-- inventory(file, qualname) 来自同 commit 的官方 AST 枚举。
SELECT file, qualname FROM inventory
EXCEPT
SELECT DISTINCT file, qualname FROM touch_rows WHERE test != '';

SELECT DISTINCT file FROM inventory
EXCEPT
SELECT DISTINCT file FROM touch_rows WHERE test != '';
```
