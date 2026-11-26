<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# L0_PostMerge #3092：单 GPU 并集与最终全量 coverage 对比

- 分析日期：2026-11-26（Asia/Shanghai）。
- 构建：[L0_PostMerge #3092](https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/main/job/L0_PostMerge/3092/)，结果 `FAILURE`；coverage 最终生成及上传成功。
- 构建源码 commit：`38b3273d289ea24ee2e4d2d47f28817ca1cc61a1`。
- 本报告比较同一 build 的实际 SQLite 产物，不使用其他 build 或当前 main 作为基线。

## 结论

以 x86_64 与 SBSA 单 GPU checkpoint 的去重并集为基线，最终全量 coverage **新增 17 个原始触达文件、5,485 个原始函数符号**；减少 0 个文件、0 个函数符号。

其中，进入官方 coverage rate 统计范围的新增量为 **17 个文件、2,523 个函数**。文件覆盖率由 **88.009592% → 88.462563%（+0.452971 个百分点）**；函数覆盖率由 **64.631019% → 68.675066%（+4.044047 个百分点）**。

归因校验：最终包中所有非 multi-GPU stage 的函数集合，与两个单 GPU checkpoint 的并集完全相等。因此，本报告中的新增函数可以归因于 multi-GPU，未混入后补收的单 GPU 函数。

这说明文件覆盖率增量较小，但函数覆盖率并非没有明显提升。新增库存函数分布在 351 个文件中，其中 **2,364 个位于单 GPU 已触达的文件，159 个位于净新增文件**；大部分函数增量不会反映成文件覆盖率增量。

## 数据来源与比较口径

| 输入 | 含义 | 压缩包字节数 |
|---|---|---:|
| [x86_64 单 GPU checkpoint](https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local/llm-artifacts/LLM/main/L0_PostMerge/3092/cbts-coverage/cbts_pystart_report_x86_64.tar.gz) | 单 GPU 阶段结束后的快照 | 47,269,371 |
| [SBSA 单 GPU checkpoint](https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local/llm-artifacts/LLM/main/L0_PostMerge/3092/cbts-coverage/cbts_pystart_report_SBSA.tar.gz) | 单 GPU 阶段结束后的快照 | 8,240,494 |
| [最终全量包](https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-generic-local/llm-artifacts/LLM/main/L0_PostMerge/3092/cbts-coverage/cbts_pystart_report.tar.gz) | 最终收集到的全部单 GPU + 单节点多 GPU 数据 | 244,829,513 |

设 `S = touch(x86_64) ∪ touch(SBSA)`，`F = touch(final)`。仅保留 `test != ''` 的记录，函数按 `(file, qualname)` 去重，文件按 `file` 去重；新增为 `F − S`，减少为 `S − F`。两个架构的覆盖率不相加、不平均，同一函数跨测试、stage、rank 重复执行也不重复计数。

官方 rate 使用该 commit 的 `pystart_report.py` 中原样提取的 `canon`、`_collect_defs`、`enumerate_defs`，配合该 commit 的路径规范化实现和 `coveragerc.template`，重新枚举源码定义。构建实际参数是 `--source-root llm`（仓库根目录），include 为：

```text
tensorrt_llm/*.py
examples/*.py
scripts/*.py
tests/test_common/*.py
conftest.py
tests/conftest.py
tests/*/conftest.py
```

以上模式使用 `fnmatch`，`*` 可匹配 `/`。文件分母是上述范围内至少包含一个可枚举函数的 Python 文件；函数分母包括普通函数、方法及嵌套函数。源码库存复现结果为 **3,753 个文件、62,388 个函数**，与官方报告完全一致。

`covered_files = observed_files ∩ inventory_files`；`covered_functions = observed_(file,qualname) ∩ inventory_(file,qualname)`。原始符号还可能包含 `<module>`、`<lambda>`、推导式、动态生成符号及源码库存无法匹配的项，因此“raw functions touched”不能直接除以函数分母。

## 覆盖率与规模

| 指标 | 单 GPU 并集 S | 最终全量 F | 净变化 |
|---|---:|---:|---:|
| 有触达记录的 stage/test key | 2,661 | 3,538 | +877 |
| 原始触达文件 | 3,951 | 3,968 | +17 |
| 原始触达函数符号 | 104,313 | 109,798 | +5,485 |
| 官方库存内已覆盖文件 | 3,303 | 3,320 | +17 |
| 官方库存内已覆盖函数 | 40,322 | 42,845 | +2,523 |
| 文件覆盖率（分母 3,753） | 88.009592% | 88.462563% | +0.452971 个百分点 |
| 函数覆盖率（分母 62,388） | 64.631019% | 68.675066% | +4.044047 个百分点 |

最终已覆盖计数交叉校验：**3,320 / 3,753 个文件，42,845 / 62,388 个函数**，与最终 HTML 报告一致。百分点按未舍入计数计算，不能用官方展示的一位小数相减代替。

仅 `tensorrt_llm/` 的补充口径（与官方全仓口径分开）：

| 指标 | 单 GPU 并集 S | 最终全量 F | 净变化 |
|---|---:|---:|---:|
| 文件（库存 3,507） | 3,192（91.017964%） | 3,208（91.474194%） | +16；+0.456230 个百分点 |
| 函数（库存 60,155） | 39,229（65.213199%） | 41,748（69.400715%） | +2,519；+4.187516 个百分点 |

## multi-GPU 归因检查

两个 checkpoint 均不含 `_GPUs` / `_Nodes` stage；最终 DB 中含 43 个 `_GPUs` stage。归因依据实际 touch 记录，不只依据压缩包或文件名。

| 对新增函数符号的检查 | 数量 |
|---|---:|
| 至少在 multi-GPU stage 出现 | 5,485 |
| 至少在最终非 multi-GPU stage 出现 | 0 |
| 仅在 multi-GPU stage 出现 | 5,485 |
| 仅在非 multi-GPU stage 出现 | 0 |
| 两类 stage 均出现 | 0 |
| 无可归属 stage | 0 |
| checkpoint 中存在、最终非 multi-GPU 集合缺失 | 0 |
| 最终非 multi-GPU 集合新增、checkpoint 中不存在 | 0 |

## 净新增文件（完整清单）

| 文件 | 新增原始函数符号 | 其中计入官方函数 rate | 该文件计入官方文件 rate |
|---|---:|---:|---|
| `examples/llm-api/llm_inference_distributed.py` | 2 | 1 | 是 |
| `tensorrt_llm/_torch/attention/kernels/blackwell/dspark/markov_chain.py` | 120 | 13 | 是 |
| `tensorrt_llm/_torch/distributed/tp2_allreduce.py` | 6 | 4 | 是 |
| `tensorrt_llm/_torch/kernels/blackwell/gemm_allreduce_norm.py` | 62 | 1 | 是 |
| `tensorrt_llm/_torch/kernels/blackwell/gemm_twoshot_allreduce_norm.py` | 124 | 17 | 是 |
| `tensorrt_llm/_torch/kernels/blackwell/tp2_allreduce_norm.py` | 47 | 12 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/k3_latent_reduce.py` | 21 | 8 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/kernels/k3_markov/k3_markov_kernel.py` | 85 | 18 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/k3_lamport_push.py` | 3 | 1 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_moe/m3_moe16_kernel.py` | 529 | 58 | 是 |
| `tensorrt_llm/_torch/modules/engram/_evict_page_cache.py` | 5 | 4 | 是 |
| `tensorrt_llm/_torch/modules/q35_mtp_glue.py` | 2 | 1 | 是 |
| `tensorrt_llm/_torch/speculative/dspark_markov_chain.py` | 10 | 8 | 是 |
| `tensorrt_llm/_torch/speculative/group_sharded_sampling.py` | 3 | 0 | 是 |
| `tensorrt_llm/_torch/speculative/sharded_sampling.py` | 6 | 4 | 是 |
| `tensorrt_llm/_torch/visual_gen/models/wan/parallel_wino_vae.py` | 8 | 6 | 是 |
| `tensorrt_llm/evaluate/lm_eval_tasks/gpqa/cot_zeroshot_aa/utils.py` | 4 | 3 | 是 |

文件覆盖率按“文件是否有任何非空测试上下文触达”计算，包括 `<module>` 等符号；因此，某文件新增触达但其“计入官方函数 rate”的新增函数数为 0，并不矛盾。

## 新增官方库存函数集中在哪些文件

以下为新增函数最多的前 30 个文件；完整函数名见附录 A。已有文件中新覆盖的函数同样计入函数增量，但不会增加文件覆盖率。

| 文件 | 新增库存函数 | 文件此前是否已触达 |
|---|---:|---|
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/modeling.py` | 92 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/modeling.py` | 86 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_100__default/modeling.py` | 62 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_100__default_mtp_fused/modeling.py` | 62 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_moe/m3_moe16_kernel.py` | 58 | 否 |
| `tensorrt_llm/_torch/moe/fused_moe/moe_warmup_placement.py` | 46 | 是 |
| `tensorrt_llm/_torch/models/musespark_open_finalize_ar.py` | 44 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/weights.py` | 43 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/modeling.py` | 42 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/modeling.py` | 41 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_comm/k3_lamport_ar.py` | 40 | 是 |
| `tensorrt_llm/_torch/distributed/ops.py` | 36 | 是 |
| `tensorrt_llm/_torch/models/modeling_deepseekv41.py` | 35 | 是 |
| `tensorrt_llm/_torch/distributed/ll_exchange.py` | 34 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/comm/tp4_allreduce_fused_rms.py` | 34 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/modeling.py` | 34 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/weights.py` | 31 | 是 |
| `tensorrt_llm/_torch/distributed/communicator.py` | 29 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_10x__default/modeling.py` | 29 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_sandwich/k3_sandwich_kernel.py` | 29 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__dep16/modeling.py` | 29 | 是 |
| `tensorrt_llm/_torch/kernels/blackwell/mnnvl_allreduce_norm.py` | 28 | 是 |
| `tensorrt_llm/_torch/models/modeling_dspark.py` | 28 | 是 |
| `tensorrt_llm/_torch/models/modeling_step3p7.py` | 27 | 是 |
| `tensorrt_llm/_torch/visual_gen/models/wan/wino_vae_shard.py` | 26 | 是 |
| `tensorrt_llm/_torch/alltoall_watchdog.py` | 24 | 是 |
| `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/weights.py` | 24 | 是 |
| `tensorrt_llm/_torch/moe/fused_moe/communication/nvlink_one_sided.py` | 24 | 是 |
| `tensorrt_llm/_torch/visual_gen/attention/backends/parallel.py` | 22 | 是 |
| `tensorrt_llm/_torch/compilation/patterns/ar_residual_norm.py` | 21 | 是 |

## 如何解释增量

- 最终全量新增 5,485 个原始函数符号，其中 2,523 个匹配官方源码函数库存；另外 2,962 个不进入官方函数覆盖率分子，完整列于附录 B。
- 函数覆盖率实际提升 **4.044047 个百分点**，不能仅凭文件数只增加 17 个，就判断新增覆盖很少。2,523 个新增库存函数中，2,364 个（93.6980%）位于已触达文件，只有 159 个来自新文件。
- 单 GPU 已覆盖最终已覆盖库存函数的 94.1113%；multi-GPU 的很多执行与已有 Python 函数集合重叠。
- PY_START 是函数入口覆盖，不能展示已覆盖函数内部的新分支，也不统计 C++/CUDA kernel 覆盖；不同 rank、world size、TP/EP 配置可能增加了行为验证和测试到函数的映射，而不增加文件/函数覆盖率。
- 原始数据文件数是进程采集文件数，不能当作测试数、独立函数数或覆盖率增量。

## 完整性与边界

- “最终全量”表示该构建最终成功收集到的数据全集，不表示所有预期 stage 都成功回传，也不表示所有测试通过。该构建整体结果为 FAILURE。
- 最终 `test_case_meta` 状态汇总：{"nonpassed_or_tainted": 350, "passed_untainted": 3212}。本报告复现官方 observed coverage 口径，未额外剔除 non-passed / tainted 记录；这些记录不应直接视为可安全跳过测试的证据。
- 上述统计不能证明未观测函数完全没有执行；它只能说明采集数据中是否有该函数在非空测试上下文下的记录。
- 本报告基于 DB 集合差和同 commit 源码，不使用旧 forest 页面或不同日期的 rate 作为比较依据。

## 附录 A：净新增且计入官方函数 rate 的函数（完整清单）

共 **2,523** 个 `(file, qualname)`。

### `examples/llm-api/llm_inference_distributed.py`（1）

```text
main
```

### `tensorrt_llm/_jit_share.py`（1）

```text
_default_compiler
```

### `tensorrt_llm/_torch/alltoall_watchdog.py`（24）

```text
AlltoAllWatchdog.__init__
AlltoAllWatchdog._active_ranks
AlltoAllWatchdog._phase_complete
AlltoAllWatchdog._run
AlltoAllWatchdog.from_workspace
AlltoAllWatchdog.start
AlltoAllWatchdog.stop
AlltoAllWatchdog.watch
AlltoAllWatchdogCoordinator._find_shared_state
AlltoAllWatchdogCoordinator._get_shared_state
AlltoAllWatchdogCoordinator.acquire_watchdog
AlltoAllWatchdogCoordinator.active_mask_int
AlltoAllWatchdogCoordinator.active_rank_mask_for_combine
AlltoAllWatchdogCoordinator.capture_active_rank_mask
AlltoAllWatchdogCoordinator.read_current_flag_val
AlltoAllWatchdogCoordinator.release_watchdog
AlltoAllWatchdogCoordinator.watch_collective
_TorchCompletionFlagReader.__init__
_TorchCompletionFlagReader._prune_retired_copies
_TorchCompletionFlagReader._read_cuda_flags
_TorchCompletionFlagReader.read_completion_flags
_completion_flag_reached
_normalize_completion_flag
reject_rank_mask_cuda_graph_capture
```

### `tensorrt_llm/_torch/attention/attention.py`（2）

```text
_helix_post_process.<locals>.reshape_o
_helix_post_process.<locals>.reshape_s
```

### `tensorrt_llm/_torch/attention/backends/fmha/gqa_kv_replica.py`（4）

```text
_allgather_q
_pushed_partial_placeholders
kv_replica_decode
kv_replica_fused_decode
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/gather.py`（1）

```text
gather4_cached_warp
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/parallel_reduction.py`（1）

```text
_run_parallel_gmem_reduction_shared_stats
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/tasks.py`（1）

```text
create_keeps_mma_ab_correction_task.<locals>.correction_schedule.<locals>.init_work_tile_state
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/cache_manager.py`（1）

```text
CSA2CacheManager.take_page_table_stats
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/indexer.py`（2）

```text
prewarm_prefill_topk
prewarm_sparse_candidate_kernels
```

### `tensorrt_llm/_torch/attention/backends/sparse/csa2/module.py`（3）

```text
DeepseekV41Attention.attend_rows
DeepseekV41Attention.project_rows
DeepseekV41Attention.supports_row_split
```

### `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/backend.py`（1）

```text
_alloc_nvfp4_compress_fp8_scratch
```

### `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/cache_manager.py`（1）

```text
DeepseekV4CacheManager.get_num_free_blocks
```

### `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/indexer.py`（1）

```text
DeepseekV4Indexer.post_load_weights
```

### `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/metadata.py`（1）

```text
DeepseekV4TrtllmAttentionMetadata.restore_after_draft_forward
```

### `tensorrt_llm/_torch/attention/backends/sparse/deepseek_v4/module.py`（4）

```text
_deepseek_v4_q_b_layernorm_fused_fp8
_is_dsv4_q_b_gemm_norm_rope_fusion_enabled
_launch_fused_kv_norm_gen
forward_sparse_attn.<locals>._q_b_layernorm_fused_fp8
```

### `tensorrt_llm/_torch/attention/backends/sparse/dsa/backend.py`（1）

```text
_get_nvfp4_mla_kv_cache_amax
```

### `tensorrt_llm/_torch/attention/backends/sparse/dsa/custom_ops.py`（3）

```text
_extract_extra_attrs
mla_dsa_attn_inplace
mla_dsa_proj
```

### `tensorrt_llm/_torch/attention/backends/sparse/dsa/metadata.py`（3）

```text
DSAtrtllmAttentionMetadata.ensure_nvfp4_mla_generation_buffers
DSAtrtllmAttentionMetadata.set_mtp_num_accepted
DSAtrtllmAttentionMetadata.set_skip_topk
```

### `tensorrt_llm/_torch/attention/backends/sparse/dsa/module.py`（2）

```text
DSAHooks.forward_custom_op
forward_dsa_proj
```

### `tensorrt_llm/_torch/attention/backends/sparse/glm_kpool/cache_manager.py`（4）

```text
Glm5NextCacheManager._update_model_state
Glm5NextCacheManager.get_batch_slot_tables
Glm5NextCacheManager.get_index_state_buffer
Glm5NextCacheManager.get_latent_state_buffer
```

### `tensorrt_llm/_torch/attention/backends/sparse/glm_kpool/metadata.py`（2）

```text
Glm5NextMamba2Metadata.__init__
Glm5NextMamba2Metadata._glm_ensure_tables
```

### `tensorrt_llm/_torch/attention/backends/sparse/hooks.py`（3）

```text
MLASparseHooks.prepare_outputs
MLASparseHooks.project_output
MLASparseHooks.transform_weights
```

### `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/cache_manager.py`（1）

```text
MiniMaxM3DraftKVCacheView.free_resources
```

### `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_backend.py`（1）

```text
_host_prefix_sum
```

### `tensorrt_llm/_torch/attention/backends/sparse/minimax_m3/msa_indexer.py`（1）

```text
prewarm_split_proxy_variants
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/dspark/markov_chain.py`（13）

```text
DSparkMarkovChainKernel.__init__
DSparkMarkovChainKernel.smem_bytes
_add_rn
_await_entry
_bits_f32
_entry
_entry_tag
_f32_bits
_is_nan
_ld_relaxed_b64
_prefer
_st_relaxed_b64
_warp_prefer
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/gqa_kv_replica_partial_attn.py`（5）

```text
_f32_bits
_griddep_launch
_griddep_wait
_lamport_word
_st_global_u32
```

### `tensorrt_llm/_torch/attention/kernels/fused_qk_norm_rope_gate.py`（2）

```text
_fused_qkv_rope_gate_fp8_append_gather
fused_qkv_rope_gate_fp8_append_gather
```

### `tensorrt_llm/_torch/attention/mla.py`（2）

```text
MLA._attention_scaling
MLA._attention_scaling.<locals>._get_attn_scale
```

### `tensorrt_llm/_torch/autotuner.py`（2）

```text
AutoTuner._broadcast_cache_data
_majority_entries
```

### `tensorrt_llm/_torch/compilation/patterns/ar_residual_norm.py`（21）

```text
_check_getitem_only_users
_has_getitem_user
_make_fp8_quant_extra_check.<locals>.extra_check
check_f16_bf16_input
check_non_ub_strategy
register_ar_residual_norm.<locals>.extra_check
register_ar_residual_norm.<locals>.target_pattern
register_ar_residual_norm_fp8_quant.<locals>.target_pattern
register_ar_residual_norm_fp8_quant.<locals>.target_pattern_without_scale
register_ar_residual_norm_out_fp4_quant.<locals>.extra_check
register_ar_residual_norm_out_fp4_quant.<locals>.target_pattern
register_ub_patterns.<locals>.insert_copy_for_graph_output.<locals>.extra_check
register_ub_patterns.<locals>.insert_copy_for_graph_output.<locals>.target_copy_for_graph_output_pattern
register_ub_patterns.<locals>.register_convert_supported_ar_to_ub.<locals>.extra_check_convert_supported_ar_to_ub
register_ub_patterns.<locals>.register_convert_supported_ar_to_ub.<locals>.target_convert_supported_ar_to_ub
register_ub_patterns.<locals>.register_ub_finalize_patterns.<locals>.target_finalize_pattern
register_ub_patterns.<locals>.register_ub_prologue_patterns.<locals>.register_add_prologue.<locals>.target_add_prologue_pattern
register_ub_patterns.<locals>.register_ub_prologue_patterns.<locals>.register_mm_prologue.<locals>.target_mm_prologue_pattern
register_ub_patterns.<locals>.register_ub_prologue_patterns.<locals>.register_nvfp4_gemm_prologue.<locals>.extra_check
register_ub_patterns.<locals>.register_ub_prologue_patterns.<locals>.register_nvfp4_gemm_prologue.<locals>.target_nvfp4_gemm_prologue_pattern
register_ub_patterns.<locals>.register_ub_prologue_patterns.<locals>.register_scaled_mm_prologue.<locals>.target_scaled_mm_prologue_pattern
```

### `tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py`（4）

```text
_register_fake.<locals>._.<locals>._mk
_register_fake.<locals>._.<locals>.pad_up
_register_fake.<locals>.allgather
_register_fake.<locals>.allreduce
```

### `tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py`（3）

```text
_prepare_cutedsl_indexer_q_tuning_inputs
cute_dsl_bf16_bmm_blackwell
cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_zero_blackwell
```

### `tensorrt_llm/_torch/custom_ops/router_logit_partials.py`（5）

```text
RouterLogitPartials.__getitem__
RouterLogitPartials.__init__
applies
num_splits
router_logit_partials
```

### `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py`（11）

```text
AllReduceRunner.forward
AllReduceRunner.get_valid_tactics
MXFP8GemmRunner.get_valid_tactics
MXFP8GemmRunner.tune_exact_rows
MXFP8GemmRunner.unique_id
_mxfp8_scale_infer_shape
_tune_mxfp8_exact_rows
mxfp8_mxfp8_gemm_autotuned
tunable_allreduce
tunable_allreduce.<locals>._inputs_pre_hook_register_nccl_symmetric_memory_window
tunable_allreduce.<locals>._uses_nccl_symmetric_memory_window
```

### `tensorrt_llm/_torch/custom_ops/userbuffers_custom_ops.py`（4）

```text
_
add_to_ub
copy_to_userbuffers
matmul_to_ub
```

### `tensorrt_llm/_torch/device_mesh.py`（2）

```text
DeviceMeshTopologyImpl.pp_rank
DeviceMeshTopologyImpl.tp_group_name
```

### `tensorrt_llm/_torch/disaggregation/native/transfer.py`（5）

```text
Sender._get_or_connect_dealer
Sender._has_all_peer_req_infos
Sender.has_all_peer_req_infos
TransferWorker.has_all_peer_req_infos_for_send
TransferWorker.prepare_rank_info
```

### `tensorrt_llm/_torch/disaggregation/orchestration/pp_termination.py`（3）

```text
DisaggPPTerminationHandler.__init__
DisaggPPTerminationHandler.terminate
DisaggPPTerminationHandler.terminate_pending_requests
```

### `tensorrt_llm/_torch/disaggregation/transceiver.py`（1）

```text
KvCacheTransceiverV2._apply_aux
```

### `tensorrt_llm/_torch/distributed/allreduce_helper.py`（3）

```text
CustomAllReduceHelper.allocate_lowprecision_workspace
CustomAllReduceHelper.initialize_lowprecision_buffers
CustomAllReduceHelper.max_workspace_size_lowprecision
```

### `tensorrt_llm/_torch/distributed/communicator.py`（29）

```text
Distributed.is_first_pp_rank
Distributed.is_last_pp_rank
Distributed.is_second_last_pp_rank
Distributed.next_pp_rank
Distributed.pp_rank
Distributed.prev_pp_rank
MPIDist.cp_allgather
MPIDist.cp_allgather_int64
MPIDist.cp_broadcast
MPIDist.cp_comm
MPIDist.isend_object
MPIDist.pp_broadcast
MPIDist.recv_object
MPIDist.send_object
MPIDist.tp_allreduce
MPIDist.tp_barrier
MPIDist.tp_broadcast
MPIDist.tp_cp_broadcast_int64
MPIDist.tp_gather
PPCommNCCL.__init__
PPCommNCCL.recv
PPCommNCCL.send
TorchDist.log_op.<locals>.wrapper
init_pp_comm
pp_recv
pp_recv_tensors
pp_send
pp_send_tensors
safe_gather
```

### `tensorrt_llm/_torch/distributed/initialization.py`（2）

```text
ensure_deepgemm_torch_distributed
ensure_deepgemm_torch_distributed.<locals>._pick_rendezvous
```

### `tensorrt_llm/_torch/distributed/ll_allgather.py`（3）

```text
LowLatencyAllGather.create
LowLatencyAllGather.pair_programs
LowLatencyAllGather.transport
```

### `tensorrt_llm/_torch/distributed/ll_exchange.py`（34）

```text
TPGroupExchange.__init__
TPGroupExchange._launch
TPGroupExchange._trigger_mode
TPGroupExchange._use_channel
TPGroupExchange.allgather
TPGroupExchange.alltoall_merge
TPGroupExchange.buffer_bytes
TPGroupExchange.close
TPGroupExchange.create
TPGroupExchange.is_supported
TPGroupExchange.merge_push_args
TPGroupExchange.rope_gather_args
TPKeyExchange.__init__
TPKeyExchange._check
TPKeyExchange._launch
TPKeyExchange.argmax_logits
TPKeyExchange.argmax_tokens
TPKeyExchange.buffer_bytes
TPKeyExchange.channel_protocol_args
TPKeyExchange.close
TPKeyExchange.create
TPKeyExchange.exchange
TPKeyExchange.fast_gumbel_launch
_PeerBuffers.__init__
_PeerBuffers._all_filled
_PeerBuffers.close
_PeerBuffers.create
_PeerBuffers.failures
_agree_on_key_exchange
_block_m
_fast_gumbel_split
create_tp_key_exchange
get_tp_group_exchange
unpack_argmax_tokens
```

### `tensorrt_llm/_torch/distributed/mnnvl_fused_allreduce.py`（21）

```text
GemmAllReduceNorm.__call__
GemmAllReduceNorm.__init__
GemmAllReduceNorm.is_supported
GemmAllReduceNorm.tactic
GemmTwoShotAllReduceNorm.__call__
GemmTwoShotAllReduceNorm.__init__
GemmTwoShotAllReduceNorm.is_supported
GemmTwoShotAllReduceNorm.supports
GemmTwoShotAllReduceNorm.takes
MnnvlAllReduceNorm.__call__
MnnvlAllReduceNorm.__init__
MnnvlAllReduceNorm.is_supported
_LamportTable.__call__
_LamportTable.__init__
_SlotWorkspace.__init__
_compiled_gemm_norm
_compiled_slot
_compiled_two_shot
_example_operands
_l2pf_args
_rows
```

### `tensorrt_llm/_torch/distributed/mnnvl_memory.py`（8）

```text
HelixCpMnnvlMemory.comm_split_color_key
MnnvlMemory.__init__
MnnvlMemory.get_allocation_granularity
MnnvlMemory.get_allocation_prop
MnnvlMemory.initialize
MnnvlMemory.open_mnnvl_memory
TpMnnvlMemory.comm_split_color_key
init_helix_cp_comm
```

### `tensorrt_llm/_torch/distributed/ops.py`（36）

```text
AllReduce.output_buffer_kind
AllReduce.supports_moe_fold
HelixAllToAllNative.__init__
HelixAllToAllNative.alltoall_native
HelixAllToAllNative.get
MNNVLAllReduce.__init__
MNNVLAllReduce._fused_norm_call
MNNVLAllReduce._get_allreduce_mnnvl_workspace_lock
MNNVLAllReduce._moe_fold_call
MNNVLAllReduce.checkpoint_prepare
MNNVLAllReduce.checkpoint_restore
MNNVLAllReduce.forward
MNNVLAllReduce.gemm_allreduce_norm
MNNVLAllReduce.gemm_twoshot_allreduce_norm
MNNVLAllReduce.max_buffer_size_bytes
MNNVLAllReduce.supports_fp8_payload
MNNVLAllReduce.supports_moe_fold
MiniMaxAllReduceRMS.__init__
MiniMaxAllReduceRMS.forward
MoEAllReduce.forward
_allgather.<locals>.convert_output
_get_or_scale_allreduce_mnnvl_workspace
_initialize_allreduce_mnnvl_protocol
_make_mnnvl_mcast_buffer
_mark_if_capturing
_mnnvl_device_index
all_to_all_5d
alltoall_helix
cp_allgather
filter_valid_input
get_or_scale_allreduce_mnnvl_workspace
get_output_info
reducescatter
reducescatter.<locals>.convert_input
restore_full_output
userbuffers_allreduce_finalize
```

### `tensorrt_llm/_torch/distributed/peer_buffers.py`（2）

```text
PeerBuffers._create_mnnvl
PeerBuffers._group_transport
```

### `tensorrt_llm/_torch/distributed/tp2_allreduce.py`（4）

```text
Tp2AllReduceNorm.__call__
Tp2AllReduceNorm.__init__
Tp2AllReduceNorm.is_supported
Tp2AllReduceNorm.supports
```

### `tensorrt_llm/_torch/kernels/blackwell/engram/lookup.py`（2）

```text
_e2m1x4_to_e4m3x4
packed_row_bytes
```

### `tensorrt_llm/_torch/kernels/blackwell/engram/packed_projection.py`（6）

```text
EngramPackedProjectionKernel.__init__
_PackedProjectionPipeline.create
_PackedProjectionPipeline.producer_acquire_tma
_PackedProjectionPipeline.producer_acquire_tma.<locals>._leader_arrive_and_expect_tx
_PackedProjectionPipeline.producer_arrive_remote
engram_packed_projection
```

### `tensorrt_llm/_torch/kernels/blackwell/gemm_allreduce_norm.py`（1）

```text
GemmAllReduceNormKernel.__init__
```

### `tensorrt_llm/_torch/kernels/blackwell/gemm_twoshot_allreduce_norm.py`（17）

```text
_clear_dirty
_compile
_compiled
_flag_u32
_kernel
_launch
_onesm_bytes
_replayable
_shfl_idx
_sm_count
_vote_any
gemm_twoshot_allreduce_norm
supports
supports_hidden
supports_problem
supports_rows
workspace_bytes
```

### `tensorrt_llm/_torch/kernels/blackwell/mhc/mhc_boundary.py`（6）

```text
_compile_coefficients
coefficient_ctas_per_slice
coefficients_compiled
mhc_coefficients
nvfp4_fp8_quantize_compiled
nvfp4_quantize_compiled
```

### `tensorrt_llm/_torch/kernels/blackwell/mhc/mhc_split_boundary.py`（12）

```text
MhcAllReduceBoundaryKernel.__init__
_arrive_barrier_1
_atomic_add_s32
_bf16x2_neg_zero_to_zero
_ld_acquire_s32
_ld_volatile_s32
_ld_volatile_s64
_ld_volatile_v4
_red_async_release_add_u32
_st_dsmem_f32
_st_s64
_st_v4
```

### `tensorrt_llm/_torch/kernels/blackwell/mnnvl_allreduce_norm.py`（28）

```text
MnnvlAllReduceNormKernel.__init__
MnnvlAllReduceNormKernel._mt_load
MnnvlAllReduceNormKernel._mt_x
_add
_bf16r
_cluster_sync
_div_rn
_ld_acquire_s32
_ld_s32
_ld_s64
_ld_shared_f32
_ld_shared_v4
_ld_v4
_ld_volatile_v4
_max0
_min_s32
_mul
_red_release_add_s32
_rsqrt
_sanitize
_sel_ge0
_shfl_bfly
_sink4
_smem_addr
_st_s32
_st_shared_f32
_st_shared_v4
_st_v4
```

### `tensorrt_llm/_torch/kernels/blackwell/mnnvl_slot_allreduce_norm.py`（9）

```text
MnnvlSlotAllReduceNormKernel.__init__
MnnvlSlotAllReduceNormKernel.smem_bytes
MnnvlSlotAllReduceNormKernel.smem_bytes.<locals>.up16
_ld_cg_s32
_mbar_expect_tx
_mbar_init
_mbar_try_wait
_st_async_f32
_tree_sum_smem
```

### `tensorrt_llm/_torch/kernels/blackwell/tp2_allreduce_norm.py`（12）

```text
Tp2WarpAllReduceNormKernel.__init__
Tp2WarpAllReduceNormKernel.supports_hidden
_add_bf16x2
_asm
_atom_add_acq_rel_gpu
_fence_sys
_ld_acquire_sys
_ld_cg_v4
_ld_relaxed_gpu
_st_relaxed_gpu
_st_release_sys
_v
```

### `tensorrt_llm/_torch/kernels/blackwell/utils.py`（1）

```text
st_global_zero_v4
```

### `tensorrt_llm/_torch/mmap_utils.py`（2）

```text
advise_tensor_pageout
pageout_file_backed_regions
```

### `tensorrt_llm/_torch/mnnvl_alltoall_workspace.py`（3）

```text
_MnnvlAlltoAllWorkspaceLifecycle._validate_shared_context
_MnnvlAlltoAllWorkspaceLifecycle.coordinator
_MnnvlAlltoAllWorkspaceLifecycle.metainfo
```

### `tensorrt_llm/_torch/model_config.py`（2）

```text
ModelConfig._has_deepseek_v4_layer_only_modelopt_quant_config
ModelConfig._infer_nemotron_ffn_mult
```

### `tensorrt_llm/_torch/modeling_v2/catalog/comm/allgather.py`（3）

```text
Allgather.__call__
Allgather.is_valid
Allgather.reference
```

### `tensorrt_llm/_torch/modeling_v2/catalog/comm/allreduce.py`（1）

```text
Allreduce.__call__
```

### `tensorrt_llm/_torch/modeling_v2/catalog/comm/mnnvl_allreduce.py`（1）

```text
mnnvl_workspace_args
```

### `tensorrt_llm/_torch/modeling_v2/catalog/comm/reducescatter.py`（3）

```text
Reducescatter.__call__
Reducescatter.is_valid
Reducescatter.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_10x__default/modeling.py`（29）

```text
DecodeTarget.__init__
DecodeTarget._check_step_contract
DecodeTarget.forward
DecodeTarget.step_args
DeepseekV3DefaultModelingV2.__init__
DeepseekV3DefaultModelingV2._dp_pad_rows
DeepseekV3DefaultModelingV2._rope_tables
DeepseekV3DefaultModelingV2._select_target
DeepseekV3DefaultModelingV2.forward
DeepseekV3DefaultModelingV2.post_load_weights
ModelingV2DeepseekR10528Nvfp4Sm10xDefault.__init__
ModelingV2DeepseekR10528Nvfp4Sm10xDefault.forward
ModelingV2DeepseekR10528Nvfp4Sm10xDefault.load_weights
ModelingV2DeepseekR10528Nvfp4Sm10xDefault.post_load_weights
PrefillTarget.__init__
PrefillTarget._check_step_contract
PrefillTarget.forward
PrefillTarget.step_args
_build_step_args
_dense_mlp
_experts
_layer_views
_residual_norm
_residual_norm.<locals>.in_tray
_residual_norm.<locals>.local
_routed_attention_dp
_routed_block
_routed_on_every_token
_yarn_mscale
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/checkpoint.py`（7）

```text
Checkpoint.__init__
Checkpoint.check_coverage
Checkpoint.read
cols
dequantize_block_fp8
rows
share
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/dims.py`（9）

```text
Dims.from_config
Dims.kda_channels_local
Dims.kda_heads_local
Dims.layer_prefix
Dims.mla_heads_local
Dims.moe_intermediate_local
Dims.mtp_layer
Dims.select_k
Dims.vocab_local
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/layers/ffn.py`（12）

```text
DenseMlp.__init__
DenseMlp.partial
DenseMlp.partial_from_fp8
MoeLayer.__init__
MoeLayer._runner
MoeLayer.partial
MoeLayer.prefill_partial
MoeLayer.router_logits
_gated_from_fp8
_load_gated
load_dense_mlp
load_moe
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/layers/kda.py`（11）

```text
KdaLayer.__init__
KdaLayer._decode
KdaLayer._gates
KdaLayer._heads
KdaLayer._prefill
KdaLayer._project
KdaLayer._verify
KdaLayer.partial
_commit_conv_window
load_kda
load_kda.<locals>.read
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/layers/mhc.py`（5）

```text
Mhc.__init__
Mhc.fused_hc
Mhc.post_mapping
Mhc.pre_mapping
Mhc.scalars
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/layers/mla.py`（16）

```text
MlaLayer.__init__
MlaLayer.context_rows
MlaLayer.generation_partial
MlaLayer.generation_partial.<locals>.query
MlaLayer.generation_partial.<locals>.select
SparseAttentionScratch.__init__
SparseAttentionScratch.counter
_cache_pools
_fp8_gemm
_linear_fp8
_score_calls
_score_calls.<locals>.cost
_score_cost
load_mla
load_mla.<locals>.read_bf16
load_mla.<locals>.read_fp8
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/layers/streams.py`（2）

```text
AuxStream.__init__
AuxStream.fork_join
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/modeling.py`（42）

```text
DecodeTarget.__init__
DecodeTarget._fused_boundary
DecodeTarget.forward
DecodeTarget.mtp_forward
DraftModel.__init__
DraftModel.embed_tokens
Glm53FlashFp8Sm10xTp4Core.__init__
Glm53FlashFp8Sm10xTp4Core._check_step_contract
Glm53FlashFp8Sm10xTp4Core.forward
Glm53FlashFp8Sm10xTp4Core.load
Glm53FlashFp8Sm10xTp4Core.mtp_forward
Glm53FlashFp8Sm10xTp4Core.post_load_weights
Glm53FlashFp8Sm10xTp4Core.select_target
MTPLayer.__call__
MTPLayer.__init__
MTPLayer.shared_head
ModelingV2Glm53FlashFp8Sm10xTp4.forward
ModelingV2Glm53FlashFp8Sm10xTp4.infer_max_seq_len
ModelingV2Glm53FlashFp8Sm10xTp4.load_weights
ModelingV2Glm53FlashFp8Sm10xTp4.mamba_metadata_cls
ModelingV2Glm53FlashFp8Sm10xTp4.post_load_weights
PrefillTarget.__init__
PrefillTarget._ffn_partial
PrefillTarget._mla_partial
PrefillTarget.forward
PrefillTarget.mtp_forward
_Comm.__init__
_Comm.allreduce
_Comm.gather
_Comm.gather_vocab
_Comm.scatter
_Layer.__init__
_Layer.is_kda
_Mtp.__init__
_VocabParallelHead.__call__
_VocabParallelHead.__init__
_collapse_streams
_expand_streams
_linear
_mtp_own_columns
_normalize_top_config
_text_config
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/comm/mhc_ar_boundary.py`（1）

```text
MhcArBoundary.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/schedule.py`（7）

```text
ContextRows.build
GenerationRows.build
LayerSchedule.from_text_config
StepSchedule.build
StepSchedule.context_rows
StepSchedule.generation_rows
StepSchedule.num_generation_tokens
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/weights.py`（4）

```text
_decoder_layer
_hc
_mtp_layer
load_model_weights
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_100__default/modeling.py`（62）

```text
Glm5NextAllReduce.__init__
Glm5NextAllReduce.forward
Glm5NextContextRows.build
Glm5NextDecoderLayer.__init__
Glm5NextDecoderLayer.run_mlp
Glm5NextForCausalLM.__init__
Glm5NextForCausalLM.apply_quant_config_exclude_modules
Glm5NextForCausalLM.infer_max_seq_len
Glm5NextForCausalLM.mamba_metadata_cls
Glm5NextGate.__init__
Glm5NextGenerationRows.build
Glm5NextIndexer.__init__
Glm5NextIndexer.project_state
Glm5NextLinearAttention.__init__
Glm5NextLinearAttention.finalize_weights
Glm5NextLinearAttention.kda_channel_range
Glm5NextLinearAttention.kda_head_range
Glm5NextLinearAttention.shard_checkpoint_tensor
Glm5NextMTP.__init__
Glm5NextMTP.forward
Glm5NextMTPHead.__init__
Glm5NextMTPHead.forward
Glm5NextMTPHead.get_last_token_states
Glm5NextMoE.__init__
Glm5NextMoE.forward
Glm5NextModel.__init__
Glm5NextModel._forward_single_phase_fused_hc
Glm5NextModel._fused_hc_ok
Glm5NextModel.collapse_streams
Glm5NextModel.expand_streams
Glm5NextModel.forward
Glm5NextModel.mtp_layer_cls
Glm5NextRuntimeContext.context_rows
Glm5NextRuntimeContext.generation_rows
Glm5NextSchedule.attention_indices
Glm5NextSchedule.mlp_indices
Glm5NextSchedule.num_layers
Glm5NextSparseAttention.__init__
Glm5NextSparseAttention._project_attention_and_indexer
Glm5NextSparseAttention._select_and_attend_paged
Glm5NextSparseAttention.absorb_query
Glm5NextSparseAttention.absorbed_kv_b
Glm5NextSparseAttention.forward
Glm5NextSparseAttention.forward_decode
Glm5NextSparseAttention.forward_prefill
Glm5NextSparseAttention.forward_verify
Glm5NextSparseAttention.project_inputs
Glm5NextSparseAttention.project_output_latent
ModelingV2Glm53FlashNvfp4Sm100Default.__init__
ModelingV2Glm53FlashNvfp4Sm100Default.load_weights
_check_construction
_glm5_gen_tokens_per_request
_glm5_linear_kwargs
_normalize_glm5_next_top_config
build_glm5_next_runtime_context
glm5_next_attention_mapping
glm5_next_expand_streams
glm5_next_hyper_connection
glm5_next_hyper_head
glm5_next_text_quant_config
glm5_next_tp_reduces
resolve_glm5_next_schedule
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_100__default/weights.py`（5）

```text
_fill_module
_fill_module.<locals>.load_linear
_fill_module.<locals>.mark_linear
_fill_module.<locals>.materialize
load
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_100__default_mtp_fused/modeling.py`（62）

```text
Glm5NextAllReduce.__init__
Glm5NextAllReduce.forward
Glm5NextContextRows.build
Glm5NextDecoderLayer.__init__
Glm5NextDecoderLayer.run_mlp
Glm5NextDraftModel.__init__
Glm5NextForCausalLM.__init__
Glm5NextForCausalLM.apply_quant_config_exclude_modules
Glm5NextForCausalLM.forward
Glm5NextForCausalLM.infer_max_seq_len
Glm5NextForCausalLM.mamba_metadata_cls
Glm5NextGate.__init__
Glm5NextGenerationRows.build
Glm5NextIndexer.__init__
Glm5NextIndexer.project_state
Glm5NextLinearAttention.__init__
Glm5NextLinearAttention.finalize_weights
Glm5NextLinearAttention.kda_channel_range
Glm5NextLinearAttention.kda_head_range
Glm5NextLinearAttention.shard_checkpoint_tensor
Glm5NextMTP.__init__
Glm5NextMTP.forward
Glm5NextMTPHead.__init__
Glm5NextMTPHead.forward
Glm5NextMoE.__init__
Glm5NextMoE.forward
Glm5NextModel.__init__
Glm5NextModel._forward_single_phase_fused_hc
Glm5NextModel._fused_hc_ok
Glm5NextModel.collapse_streams
Glm5NextModel.expand_streams
Glm5NextModel.forward
Glm5NextRuntimeContext.context_rows
Glm5NextRuntimeContext.generation_rows
Glm5NextSchedule.attention_indices
Glm5NextSchedule.mlp_indices
Glm5NextSchedule.num_layers
Glm5NextSparseAttention.__init__
Glm5NextSparseAttention._project_attention_and_indexer
Glm5NextSparseAttention._select_and_attend_paged
Glm5NextSparseAttention.absorb_query
Glm5NextSparseAttention.absorbed_kv_b
Glm5NextSparseAttention.forward
Glm5NextSparseAttention.forward_decode
Glm5NextSparseAttention.forward_prefill
Glm5NextSparseAttention.forward_verify
Glm5NextSparseAttention.project_inputs
Glm5NextSparseAttention.project_output_latent
ModelingV2Glm53FlashNvfp4Sm100DefaultMtpFused.__init__
ModelingV2Glm53FlashNvfp4Sm100DefaultMtpFused.load_weights
_check_construction
_glm5_gen_tokens_per_request
_glm5_linear_kwargs
_normalize_glm5_next_top_config
build_glm5_next_runtime_context
glm5_next_attention_mapping
glm5_next_expand_streams
glm5_next_hyper_connection
glm5_next_hyper_head
glm5_next_text_quant_config
glm5_next_tp_reduces
resolve_glm5_next_schedule
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_100__default_mtp_fused/weights.py`（5）

```text
_fill_module
_fill_module.<locals>.load_linear
_fill_module.<locals>.mark_linear
_fill_module.<locals>.materialize
load
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/comm.py`（2）

```text
Collectives.__init__
_barrier
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/modeling.py`（41）

```text
DecodeTarget.__init__
DecodeTarget._check_step_contract
DecodeTarget.spec_in
DecodeTarget.step_args
DecodeTarget.step_in
Glm5NextTp4ModelingV2.__init__
Glm5NextTp4ModelingV2._bind_weights
Glm5NextTp4ModelingV2._bind_weights.<locals>.bf
Glm5NextTp4ModelingV2._bind_weights.<locals>.group
Glm5NextTp4ModelingV2._bind_weights.<locals>.hc
Glm5NextTp4ModelingV2._bind_weights.<locals>.mla
Glm5NextTp4ModelingV2._bind_weights.<locals>.p
Glm5NextTp4ModelingV2._bind_weights.<locals>.shared
Glm5NextTp4ModelingV2._decode_lists
Glm5NextTp4ModelingV2._host_lists
Glm5NextTp4ModelingV2._host_lists.<locals>.up
Glm5NextTp4ModelingV2._pools_of
Glm5NextTp4ModelingV2._select_target
Glm5NextTp4ModelingV2._step_in
Glm5NextTp4ModelingV2.derive_after_load
Glm5NextTp4ModelingV2.forward
Glm5NextTp4ModelingV2.kda_draft_records
Glm5NextTp4ModelingV2.spec_step
ModelingV2Glm53FlashNvfp4Sm10xTp4.__init__
ModelingV2Glm53FlashNvfp4Sm10xTp4.forward
ModelingV2Glm53FlashNvfp4Sm10xTp4.forward.<locals>.sample
ModelingV2Glm53FlashNvfp4Sm10xTp4.load_weights
ModelingV2Glm53FlashNvfp4Sm10xTp4.post_load_weights
PrefillTarget.__init__
PrefillTarget._check_step_contract
PrefillTarget.spec_in
PrefillTarget.step_args
PrefillTarget.step_in
_Pools.__init__
_SlotTablesMetadata.__init__
_SlotTablesMetadata._slot_group_of
_SlotTablesMetadata.prepare
_ptr
_runtime_model_config
_sample_rows
_view
```

### `tensorrt_llm/_torch/modeling_v2/models/glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/weights.py`（24）

```text
Layout.experts
Layout.of
_Reader.__call__
_Reader.__init__
_Reader._get
_Reader.cols
_Reader.rows
_Reader.scalar
interleave64
kda_layers
load
load.<locals>.cols
load.<locals>.fp4
load.<locals>.load_mla
load.<locals>.load_shared
load.<locals>.put
load.<locals>.rows
load_lm_head
mla_layers
param_specs
param_specs.<locals>.mla
param_specs.<locals>.shared
sf_bytes
swizzle_sf
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/comm/tp4_allgather_lamport.py`（5）

```text
_glm_tp_allgather_bf16_oneshot_lamport
_glm_tp_allgather_bf16_oneshot_lamport_kernel
_sanitize_bf16x8
_store_output_vector
compile_for_rank
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/comm/tp4_allreduce_fused_rms.py`（34）

```text
_advance_signal_if_last_cta
_all_bf16x8_lanes_ready
_cluster_inv_rms_from_partials
_compile_twoshot_common
_cta_head_load
_decode_positive_e4m3
_glm_tp_allreduce_bf16_oneshot_lamport_serving
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel
_glm_tp_allreduce_residual_rms_norm_bf16_twoshot_nvls
_glm_tp_allreduce_residual_rms_norm_out_mxf8_twoshot_push
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel
_multicast_publish_bf16x8
_multimem_st_v4_b32
_publish_cta_partial_to_cluster
_reset_bf16x8
_sanitize_bf16x8
_twoshot_block_epilogue
_twoshot_push_partial
_twoshot_reduce_slice
_validate_ctas_per_row
_validate_fused_pdl_release_policy
_validate_multicast_publish
_validate_pdl_release_policy
_validate_signal_advance_flags
_warp_sum_f32
compile_fused_for_rank
compile_fused_out_mxf8_twoshot_for_rank
compile_fused_out_quant_for_rank
compile_fused_twoshot_for_rank
compile_pure_for_rank
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/moe/mtp_moe_gather_grouped_gemm_act_fusion.py`（1）

```text
interleave_w13
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/norm/self_attn_qkv_a_rmsnorm.py`（8）

```text
_current_cuda_stream
_require_bf16_cuda_inner_contiguous
_require_bf16_cuda_vector
_require_vector_row_layout
_same_exact_view
_storage_identity
_validate_output_aliases
qkv_a_rmsnorm_run
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/modeling.py`（86）

```text
DecodeTarget._check_step_contract
DecodeTarget._mtp_step
DecodeTarget._mtp_step.<locals>.slots_of
DecodeTarget.forward
DecodeTarget.forward.<locals>.slots_of
DecodeTarget.mtp_forward
DecodeTarget.step_args
DraftModel.__init__
DraftModel.embed_tokens
GlmMoeDsaTep4ModelingV2.__init__
GlmMoeDsaTep4ModelingV2._add_rmsnorm
GlmMoeDsaTep4ModelingV2._attn_gen
GlmMoeDsaTep4ModelingV2._build_context_mha
GlmMoeDsaTep4ModelingV2._build_context_mha_fp8
GlmMoeDsaTep4ModelingV2._build_context_mha_fp8.<locals>.lg
GlmMoeDsaTep4ModelingV2._build_dec_mxf8
GlmMoeDsaTep4ModelingV2._build_kvb8
GlmMoeDsaTep4ModelingV2._build_mlp_mxf8
GlmMoeDsaTep4ModelingV2._build_moe
GlmMoeDsaTep4ModelingV2._build_moe.<locals>.build_sig
GlmMoeDsaTep4ModelingV2._build_moe.<locals>.optional
GlmMoeDsaTep4ModelingV2._build_moe.<locals>.slot
GlmMoeDsaTep4ModelingV2._build_proj_mxf8
GlmMoeDsaTep4ModelingV2._build_qbix
GlmMoeDsaTep4ModelingV2._build_qkva_ix_fold
GlmMoeDsaTep4ModelingV2._dense_mlp
GlmMoeDsaTep4ModelingV2._derive_attention
GlmMoeDsaTep4ModelingV2._derive_first_step
GlmMoeDsaTep4ModelingV2._derive_indexer
GlmMoeDsaTep4ModelingV2._derive_moe
GlmMoeDsaTep4ModelingV2._derive_mtp
GlmMoeDsaTep4ModelingV2._embed
GlmMoeDsaTep4ModelingV2._fmha_dec_handle
GlmMoeDsaTep4ModelingV2._mha_fp8_scales
GlmMoeDsaTep4ModelingV2._mla_rope_handle
GlmMoeDsaTep4ModelingV2._mtp_moe_fp4
GlmMoeDsaTep4ModelingV2._mtp_moe_fp4.<locals>.quant
GlmMoeDsaTep4ModelingV2._pf_fork
GlmMoeDsaTep4ModelingV2._project_tail
GlmMoeDsaTep4ModelingV2._q_b_e4m3
GlmMoeDsaTep4ModelingV2._qb_ixqb
GlmMoeDsaTep4ModelingV2._qkv
GlmMoeDsaTep4ModelingV2._reduce
GlmMoeDsaTep4ModelingV2._reduce_norm_x8
GlmMoeDsaTep4ModelingV2._rmsnorm
GlmMoeDsaTep4ModelingV2._rope_tables
GlmMoeDsaTep4ModelingV2._stack_ix_fold_weights
GlmMoeDsaTep4ModelingV2._stack_qbix_weights
GlmMoeDsaTep4ModelingV2._topk_buf
GlmMoeDsaTep4ModelingV2._warm_draft_rows
GlmMoeDsaTep4ModelingV2.build_logits_head
GlmMoeDsaTep4ModelingV2.forward
GlmMoeDsaTep4ModelingV2.gather_rows
GlmMoeDsaTep4ModelingV2.post_load_weights
GlmMoeDsaTep4ModelingV2.route_report
MTPLayer.__call__
MTPLayer.__init__
MTPLayer.forward
MTPLayer.shared_head
ModelingV2Glm52Nvfp4Sm103Tep4.__init__
ModelingV2Glm52Nvfp4Sm103Tep4.forward
ModelingV2Glm52Nvfp4Sm103Tep4.load_weights
ModelingV2Glm52Nvfp4Sm103Tep4.post_load_weights
PrefillTarget.__init__
PrefillTarget._attn_ctx
PrefillTarget._attn_ctx_selected
PrefillTarget._attn_mha
PrefillTarget._check_step_contract
PrefillTarget._ctx_rope_step
PrefillTarget._dense_causal_step
PrefillTarget._diet_block_offsets
PrefillTarget._diet_tables
PrefillTarget._expand_mxf8
PrefillTarget._layer_attention
PrefillTarget._mha_plan
PrefillTarget._mha_takes
PrefillTarget._mtp_last_plan
PrefillTarget.forward
PrefillTarget.forward.<locals>.slots_of
PrefillTarget.mtp_forward
PrefillTarget.mtp_forward.<locals>.slots_of
PrefillTarget.step_args
_build_step_args
_fold_pow2
_pow2_ceil
_tile_bounds
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/comm/_lamport.py`（4）

```text
build_workspace
ensure_process_group
tail
wide_ar_levers
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/comm/cute_dsl_allgather_widen.py`（1）

```text
CuteDslAllgatherWiden.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/comm/cute_dsl_allreduce.py`（1）

```text
CuteDslAllreduce.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/comm/cute_dsl_allreduce_add_rmsnorm.py`（1）

```text
CuteDslAllreduceAddRmsnorm.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/comm/cute_dsl_allreduce_add_rmsnorm_quant_nvfp4.py`（1）

```text
CuteDslAllreduceAddRmsnormQuantNvfp4.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/gemm/cute_dsl_indexer_pre_proj_split.py`（1）

```text
compile_all
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/gemm/cute_dsl_projection.py`（4）

```text
_EagerTwin.__call__
_EagerTwin.__init__
build_prefill_all
with_eager_twin
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/gemm/cute_dsl_shared_gate_up_swiglu.py`（1）

```text
build_all
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/ops/moe/cute_dsl_moe_router.py`（1）

```text
build_all
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/spec_worker.py`（10）

```text
Glm52MTPEagleWorker.__init__
Glm52MTPEagleWorker._forward_impl
Glm52MTPEagleWorker._forward_linear_draft_loop
Glm52MTPEagleWorker._glm_draft_step
Glm52MTPEagleWorker._prepare_next_new_tokens
Glm52MTPEagleWorker._sample_tokens_for_batch
Glm52MTPEagleWorker.prepare_1st_drafter_inputs
Glm52MTPEagleWorker.sample_and_accept_draft_tokens
enabled
ineligibility
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/weights.py`（31）

```text
GlmMoeDsaTep4Weights._targets
GlmMoeDsaTep4Weights.declare
GlmMoeDsaTep4Weights.dims
GlmMoeDsaTep4Weights.expected_unconsumed
GlmMoeDsaTep4Weights.load
GlmMoeDsaTep4Weights.load.<locals>.fill
GlmMoeDsaTep4Weights.manifest
W.layer_indices
W.sources
_cat_rows
_dense_layers
_expert_window.<locals>.build
_fc1_s
_fc1_w
_fc2_s
_fc2_w
_indexer_fused
_indexer_fused_bf16
_indexer_layers
_materialize
_moe_layers
_mtp_fc1_w
_mtp_only
_pair_perm
_reorder_kv_b
_routed_layers
_sf_atoms
_to_fp32
_trunk_layers
dec_hot
dec_replicas
```

### `tensorrt_llm/_torch/modeling_v2/models/gpt_oss/gpt_oss_120b__sm_10x__default/modeling.py`（3）

```text
DecodeTarget._moe_attention_dp
PrefillTarget._moe_attention_dp
_residual_norm.<locals>.in_tray
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/attn_res.py`（8）

```text
_apply_attn_res
_apply_attn_res_add_and_rmsnorm
_apply_attn_res_add_rmsnorm_fused
_apply_attn_res_and_rmsnorm
_apply_attn_res_fused
_apply_attn_res_rmsnorm_fused
_note_attn_res_fusion
_rms_norm_eps
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/front_op.py`（8）

```text
_kernel
_max_clusters_of
k3_moe_front
k3_moe_front.<locals>.arg
k3_moe_front.<locals>.arg2
max_clusters
supports
weight_supported
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/k3_latent_reduce.py`（8）

```text
_hi
_ld_acquire_gpu
_lo
_pack_bf16x2
_red_add_release
buffer_words
k3_latent_reduce
k3_latent_reduce_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/k3_moe_front.py`（18）

```text
_activation_tensor_map
_add_opaque
_atom_add_shared
_bf16_rn
_flip
_load_acquire
_mapa_u32
_poll_logit_parts
_red_add_release
_situ
_situ.<locals>.tanh_f32
_st_async_v4_f32
_test_wait_cluster
_top16_cta
_tree_sum
_weight_tensor_map
k3_moe_front
k3_moe_front_kernel
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/k3_moe_kernel.py`（2）

```text
_ar_word
_pack_bf16x2
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/k3_route_quant_ag.py`（5）

```text
_pack_bf16x2
_poll4
_sanitize_bf16x2
_sanitize_f32
_store_release
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/latent_op.py`（8）

```text
K3LatentExchange.create
K3LatentExchange.create.<locals>.build
K3LatentExchange.push_args
_kernel
default_ctas
k3_latent_reduce
k3_latent_reduce.<locals>.arg
supports
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/op.py`（2）

```text
K3MoeHeadWorkspace.create.<locals>.build
K3MoeLayer.push
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_sandwich/k3_sandwich_kernel.py`（29）

```text
_carve
_fence_sys
_fmax
_hi
_lo
_mapa_u32
_pack_bf16x2
_rcp_rn
_role_layout
_sanitize
_st_async_f32
_st_async_v4
_test_wait_cluster
_view
_warp_allsum
_warp_sum_scatter
act_word_index
activation_tensor_map
buffer_words
k3_sandwich_kernel
k3_sandwich_oproj
k3_sandwich_plain
k3_sandwich_tail
lat_buffer_words
sandwich_role
smem_bytes
swiglu_share
weight_stages
weight_tensor_map
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_sandwich/op.py`（16）

```text
K3SandwichLatentExchange.create
K3SandwichLatentExchange.create.<locals>.arm
K3SandwichWorkspace.create
_arg
_compile_and_run
_create_buffer
_slab_args
_src_args
_tap_words
_use_pdl
_words
k3_sandwich_oproj
k3_sandwich_plain
k3_sandwich_tail
supports
supports_tail
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_comm/k3_lamport_ar.py`（40）

```text
K3LamportFinishKernel.__init__
K3LamportPushKernel.__init__
LamportWorkspace.from_mnnvl
_asm
_ceil_div
_check
_compiled_finish
_compiled_push
_fv
_iv
_lv
_stream
_unpack
acc_bf16x2
add_f32
ag_add_bf16x2
ag_row_addr
clean4
clear_stage
griddepcontrol_launch_dependents
griddepcontrol_wait
lamport_state
ld_acquire_u32
ld_nc_u64
ld_v4
ld_volatile_v4
multimem_st_v4
pack_bf16x2
poll_v4
push_slot
push_v4
red_release_add
required_bytes
run_lamport_finish
run_lamport_push
sanitize_bf16x2
st_sentinel_v4
st_u32
st_v4
stage_addr
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_gemm/k3_fp8w_lowm_gemm.py`（8）

```text
_add_bf16x2
_compiled_ar_push_kernel
_compiled_mid_ar_push_kernel
ar_push_tactic
mid_ar_push_tactic
pack_ue8m0_exponents
run_k3_fp8w_lowm_gemm_ar_push
run_mid_prequantized_ar_push
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_gemm/k3_fp8w_lowm_mxfp8.py`（5）

```text
_compiled_ar_push_kernel
run_k3_fp8w_lowm_gemm_mxfp8_ag_push
run_k3_fp8w_lowm_gemm_mxfp8_ar_push
run_k3_fp8w_lowm_gemm_mxfp8_slice_ar_push
run_k3_fp8w_lowm_gemm_mxfp8_slice_ar_push.<locals>.ptr
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/decode_comm.py`（10）

```text
K3DecodeComm.allreduce_attn_res
K3DecodeComm.compile_tail
K3DecodeComm.create
K3DecodeComm.sandwich_oproj
K3DecodeComm.sandwich_tail
K3DecodeComm.takes_oproj
K3DecodeComm.takes_post_attention
_eps
_res_args
use_decode_one_shot
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/decode_moe.py`（7）

```text
K3DecodeMoe.create
K3DecodeMoeLayer._front
K3DecodeMoeLayer.create
K3DecodeMoeLayer.warm_up
_experts
_gemv
fold_latent_norm
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/kernels/k3_moe_engines/op.py`（1）

```text
_K3MoeEngineLayer.push
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/modeling.py`（3）

```text
KimiLinearDecoderLayer._o_proj
KimiLinearDecoderLayer.forward
KimiLinearDecoderLayer.forward_generic
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/ops/moe/k3_moe_m1.py`（1）

```text
k3_moe_m1_push
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/ops/moe/k3_moe_m2.py`（1）

```text
k3_moe_m2_push
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/decode_comm.py`（10）

```text
K3DecodeComm.allreduce_attn_res
K3DecodeComm.compile_tail
K3DecodeComm.create
K3DecodeComm.sandwich_oproj
K3DecodeComm.sandwich_tail
K3DecodeComm.takes_oproj
K3DecodeComm.takes_post_attention
_res_args
use_decode_one_shot
wide_all_reduce
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/decode_moe.py`（11）

```text
K3DecodeMoe.create
K3DecodeMoeLayer._front
K3DecodeMoeLayer._wide
K3DecodeMoeLayer._wide.<locals>._routed_partial
K3DecodeMoeLayer._wide.<locals>._shared_activation
K3DecodeMoeLayer.create
K3DecodeMoeLayer.forward
K3DecodeMoeLayer.warm_up
_experts
_gemv
fold_latent_norm
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/kernels/k3_markov/k3_markov_kernel.py`（18）

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
k3_markov
k3_markov_kernel
rows_per_cta
smem_bytes
supports
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/kernels/k3_markov/op.py`（7）

```text
_arg
_kernel_module
_sm_count
k3_markov
markov_workspace
pick_grid
supports
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/kernels/k3_spec_accept/op.py`（2）

```text
supports_columns
workspace
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/modeling.py`（4）

```text
KimiLinearDecoderLayer._o_allreduce
KimiLinearDecoderLayer._o_proj
KimiLinearDecoderLayer.forward
KimiLinearDecoderLayer.forward_generic
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/ops/comm/k3_sandwich_plain.py`（2）

```text
K3SandwichPlain.__call__
K3SandwichPlain.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/ops/spec/k3_markov.py`（3）

```text
K3Markov.__call__
K3Markov.is_valid
K3Markov.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_nvfp4__sm_107__dep16/modeling.py`（29）

```text
K3MegaMoE.__init__
K3MegaMoE._acquire_buffers
K3MegaMoE._alloc_symm_providers
K3MegaMoE._launch
K3MegaMoE._prime_ladder
K3MegaMoE._run
K3MegaMoE._select_bucket
K3MegaMoE._stage_inputs
K3MegaMoE.buckets
K3MegaMoE.create_weights
K3MegaMoE.forward
K3MegaMoE.quantize_input
K3MegaMoeNvfp4Experts.__init__
K3MegaMoeNvfp4Experts.create_weights
K3MegaMoeNvfp4Experts.create_weights.<locals>.param
K3MegaMoeNvfp4Experts.finalize
K3MegaMoeNvfp4Experts.load_expert
K3MegaMoeNvfp4Experts.load_expert.<locals>.take
K3MegaMoeNvfp4Experts.prepare_streaming_load
_align_up
_as_fp8_sf
_as_nvfp4
_leading_rows
_resolve_ep_process_group
blocked_block_scales
fc1_sf_elements
fc2_sf_elements
interleave_gate_up
resolve_buckets
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/mnnvl_workspace.py`（3）

```text
MnnvlWorkspace.comm_buffer
MnnvlWorkspace.create
MnnvlWorkspace.max_one_shot_tokens
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/k3_lamport_finish.py`（1）

```text
K3LamportFinish.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/k3_lamport_push.py`（1）

```text
K3LamportPush.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/k3_latent_reduce.py`（2）

```text
K3LatentReduce.__call__
K3LatentReduce.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/k3_sandwich_oproj.py`（6）

```text
K3SandwichOproj.__call__
K3SandwichOproj.is_valid
K3SandwichOproj.reference
_reduce_reference
_residual_reference
_valid_residual_inputs
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/k3_sandwich_tail.py`（3）

```text
K3SandwichTail.__call__
K3SandwichTail.is_valid
K3SandwichTail.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/mnnvl_allgather_split.py`（3）

```text
MnnvlAllgatherSplit.__call__
MnnvlAllgatherSplit.reference
required_buffer_bytes
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/mnnvl_allreduce.py`（2）

```text
K3MnnvlAllReduce.__init__
K3MnnvlAllReduce.forward
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/mnnvl_allreduce_attn_res.py`（2）

```text
MnnvlAllreduceAttnRes.__call__
MnnvlAllreduceAttnRes.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/mnnvl_fusion_allreduce.py`（4）

```text
MnnvlFusionAllreduce.__call__
MnnvlFusionAllreduce.is_valid
MnnvlFusionAllreduce.reference
_fusion
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/sm107_allreduce.py`（13）

```text
LamportPushTarget.__init__
TpAllReduce.__call__
TpAllReduce.__init__
TpAllReduce._mnnvl
TpAllReduce.mnnvl_allreduce
_lamport_view
_mnnvl_finish
finish_lamport_push
lamport_push_target
lamport_push_tensor
mnnvl_lamport_push_target
mnnvl_serves
mnnvl_workspace_bytes
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/gemm/_fp8_reference.py`（1）

```text
all_reduce_bf16
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/gemm/k3_fp8w_lowm_gemm_push.py`（3）

```text
K3Fp8wLowmGemmPush.__call__
K3Fp8wLowmGemmPush.is_valid
K3Fp8wLowmGemmPush.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/gemm/k3_fp8w_lowm_mxfp8_push.py`（3）

```text
K3Fp8wLowmMxfp8Push.__call__
K3Fp8wLowmMxfp8Push.is_valid
K3Fp8wLowmMxfp8Push.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/gemm/k3_fp8w_mid_gemm_push.py`（3）

```text
K3Fp8wMidGemmPush.__call__
K3Fp8wMidGemmPush.is_valid
K3Fp8wMidGemmPush.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/moe/k3_moe.py`（1）

```text
k3_moe_push
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/moe/k3_moe_front.py`（2）

```text
K3MoeFront.__call__
K3MoeFront.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/modeling.py`（92）

```text
DecodeTarget._attention
DecodeTarget._attention_core
DecodeTarget._boundary
DecodeTarget._decode_sparse_core
DecodeTarget._fused_attn_args
DecodeTarget._generic_boundary
DecodeTarget._layer
DecodeTarget._moe
DecodeTarget._prepushed
DecodeTarget._routed
DecodeTarget._routed_chunks
DecodeTarget._swiglu_mlp
M3Eagle3Draft.__init__
M3Eagle3Draft.apply_eagle3_fc
M3Eagle3Draft.ar_norm
M3Eagle3Draft.attention
M3Eagle3Draft.embed_table
M3Eagle3Draft.fc
M3Eagle3Draft.forward_layer
M3Eagle3Draft.mlp
M3Eagle3Worker.__init__
M3Eagle3Worker._forward_draft_loop
M3Eagle3Worker._forward_impl
M3Eagle3Worker._m3_fc_fork
M3Eagle3Worker._m3_first_drafter_inputs
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
MiniMaxM3Tp4ModelingV2.__init__
MiniMaxM3Tp4ModelingV2._build_embed_owners
MiniMaxM3Tp4ModelingV2._build_engine
MiniMaxM3Tp4ModelingV2._build_moe16_engine
MiniMaxM3Tp4ModelingV2._derive_moe
MiniMaxM3Tp4ModelingV2._expert_operands
MiniMaxM3Tp4ModelingV2._shared_weights
MiniMaxM3Tp4ModelingV2.forward
MiniMaxM3Tp4ModelingV2.post_load_weights
MiniMaxM3Tp4ModelingV2.projections
MiniMaxM3Tp4ModelingV2.sm_scale
ModelingV2MiniMaxM3Nvfp4Sm103Tp4.__init__
ModelingV2MiniMaxM3Nvfp4Sm103Tp4.cache_derived_state
ModelingV2MiniMaxM3Nvfp4Sm103Tp4.forward
ModelingV2MiniMaxM3Nvfp4Sm103Tp4.load_draft_weights
ModelingV2MiniMaxM3Nvfp4Sm103Tp4.load_weights
ModelingV2MiniMaxM3Nvfp4Sm103Tp4.post_load_weights
PrefillTarget._attention
PrefillTarget._attention_core
PrefillTarget._check_step_contract
PrefillTarget._generic_boundary
PrefillTarget._moe
PrefillTarget._routed
PrefillTarget._routed_chunks
PrefillTarget._swiglu_mlp
PrefillTarget.forward
PrefillTarget.step_args
_DraftForward.__call__
_DraftForward.__init__
_Mxfp8GemmState.__init__
_attn_c1
_attn_mega
_build_speculation
_can_mc
_fused_attn
_is_vl_config
_linear
_mc_allreduce_norm
_moe_layer_ids
_mx_mode
_mxfp8_linear
_native_allreduce_norm
_nvfp4_fused_write_layout
_nvfp4_layer
_qk_norm_rope
_routed_experts
_router_fp4
_router_logits
_runtime_hooks
_sparse_layer_ids
_text_model_config
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/comm/m3_allreduce.py`（1）

```text
M3AllReduce.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/comm/m3_mc_allreduce_norm.py`（1）

```text
M3McAllreduceNorm.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/comm/m3_mc_finalize_allreduce_norm.py`（2）

```text
M3McFinalizeAllreduceNorm.__call__
finalize_partial
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/embedding_firstnorm/op.py`（1）

```text
DecodeEmbedding.__init__
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/long_select.py`（1）

```text
long_select_blocks
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_draft_step.py`（2）

```text
_num_sms
draft_step
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_fused_attn.py`（4）

```text
_check
_load_module
m3_fused_decode_attn
m3_fused_decode_attn_checks
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_mc_allreduce.py`（8）

```text
McAllReduceWorkspace.flags_for
McBuckets.__init__
McBuckets.for_tokens
_bucket
_tp_comm
get_workspace
m3_mc_allreduce_norm
m3_mc_finalize_allreduce_norm
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_moe/m3_moe16_kernel.py`（58）

```text
_add_rn
_adv
_ar_push
_asm
_b1_addr
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
_fence_proxy_async_global
_fma_rn
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
_quant_words
_rcp
_red_release_add
_redux_max_u32
_seg
_sfb1_to_tmem
_sh_issue
_shared_cta
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

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_moe/op.py`（1）

```text
ar_push_desc
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_moe/op16.py`（5）

```text
M3Moe16State.layer
M3Moe16State.run
_group_slots
make_phase_plan16
make_phase_plan16.<locals>.counts_for
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_prequant.py`（1）

```text
PrequantSlot.take
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_router_gemv.py`（1）

```text
m3_router_tc
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_shard_accept.py`（3）

```text
_num_sms
_workspace
shard_accept
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_spec_tail.py`（2）

```text
slot_scatter
step_gather
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/moe/m3_moe16_engine.py`（2）

```text
M3Moe16Engine.__call__
M3Moe16Engine.reference
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/weights.py`（43）

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

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/routing.py`（1）

```text
_quant_algo
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/ops/gemm/nvfp4_gemv.py`（1）

```text
precompile
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/modeling.py`（34）

```text
DecodeTarget._capture_selection
DecodeTarget._gdn_epilogue
DecodeTarget._verify_qsl
DecodeTarget.mtp_forward
DecodeTarget.mtp_forward.<locals>.sel
DraftModel.__init__
DraftModel.embed_tokens
MTPLayer.__call__
PrefillTarget._attn_graph_key
PrefillTarget._capture_selection
PrefillTarget._draft_extend_query_rows
PrefillTarget._gdn_graph_key
PrefillTarget._qsa_padded_rows
PrefillTarget._selection_pipeline
PrefillTarget._sp_split.<locals>.finish
PrefillTarget._sp_split.<locals>.start
PrefillTarget.mtp_forward
Qwen4ExpDefaultModelingV2._bind_draft_cache_manager
Qwen4ExpDefaultModelingV2._bind_step_buffers
Qwen4ExpDefaultModelingV2._draft_step_bind
Qwen4ExpDefaultModelingV2._precompile_draft_hc_rows
Qwen4ExpDefaultModelingV2.split_spec_output
SpeculativeRollback.fused_pending
_combine_inject_fits
_draft_continue_ok
_draft_embedding
_draft_positions
_fp8_context_pages
_gdn_in_proj_conv_tactic
_moe_block_sp
_mtp_front
_mtp_tail
_mtp_views
_pad_window
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/attention/qsa_spec_step_metadata__decode.py`（1）

```text
SpecStepMetadataPrologue.__call__
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/comm/ce_exchange__sp_pair.py`（10）

```text
CeExchange.__call__
CopyEngineExchange.__del__
CopyEngineExchange.__init__
CopyEngineExchange.exchange
_CudaArray.__init__
_checked
_open_pair
_release
_wait
_write
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/comm/hc_allreduce_inject_norm__tp_sum.py`（3）

```text
HcAllreduceInjectNorm.__call__
_aligned
fits
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/comm/lamport_allgather__index_split.py`（3）

```text
LamportAllgather.__call__
SelectionLamport.fits
build
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/comm/mnnvl_allreduce__tp_sum.py`（2）

```text
MnnvlTpSum.__call__
MnnvlTpSum.is_valid
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/comm/moe_a2a__dp_dispatch_combine.py`（4）

```text
MoeA2aCombine.reference
MoeA2aDispatch.is_valid
MoeA2aDispatch.reference
build_strategy
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/ops/ssm/gdn_cached_replay_cute__verify.py`（1）

```text
precompile
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default/pool.py`（2）

```text
check_replay_width
check_shared_selection
```

### `tensorrt_llm/_torch/modeling_v2/models/qwen4_exp/qwen3_8_flash_next_nvfp4__sm_10x__default/modeling.py`（5）

```text
Qwen4ExpDefaultModelingV2._tp_sum_op.<locals>.tp_sum
_check_replay_ring
_decode_router
_mid_rows
_precompile_mid
```

### `tensorrt_llm/_torch/models/checkpoints/hf/glm5_next_weight_mapper.py`（2）

```text
Glm5NextHfWeightMapper.audit
glm5_next_weight_owner
```

### `tensorrt_llm/_torch/models/checkpoints/hf/nemotron_nas_weight_mapper.py`（3）

```text
NemotronNasHfWeightMapper._duplicate_kv_weights_for_layer
NemotronNasHfWeightMapper._layer_idx_from_breakdown
NemotronNasHfWeightMapper.apply_callbacks
```

### `tensorrt_llm/_torch/models/checkpoints/memory_init.py`（3）

```text
_allocate_weight_on_cuda
_allocate_weights_in_tagged_scope
_apply_to_buffers_only
```

### `tensorrt_llm/_torch/models/checkpoints/mistral/checkpoint_loader.py`（1）

```text
MistralLarge3CheckpointLoader.__init__
```

### `tensorrt_llm/_torch/models/checkpoints/mistral/tokenizer.py`（1）

```text
MistralTokenizer.name_or_path
```

### `tensorrt_llm/_torch/models/checkpoints/mistral/weight_mapper.py`（3）

```text
MistralLarge3WeightMapper.__init__
MistralWeightMapper.permute_qk
MistralWeightMapper.permute_qk.<locals>.permute
```

### `tensorrt_llm/_torch/models/deepseek_v41_processor.py`（3）

```text
DeepseekV41InputProcessor._dummy_grid
DeepseekV41InputProcessor.get_dummy_mm_data
DeepseekV41InputProcessor.get_mm_max_tokens_per_item
```

### `tensorrt_llm/_torch/models/deepseekv41_decoder_graph.py`（21）

```text
DecoderPassGraph.__init__
DecoderPassGraph._static_inputs
DecoderPassGraph.run
DecoderPassGraph.select
EncoderPassGraph.__init__
EncoderPassGraph._static_inputs
EncoderPassGraph.run
EncoderPassGraph.select
PassGraphPool.__init__
PassGraphPool.acquire
PassGraphPool.release
PassGraphPool.stream
PassRows.clear
PassRows.live_counts
_PassGraph.__init__
_PassGraph._capture_all
_PassGraph._replay
_PassGraph._run_bucket
_PassGraph._tuning
_PassGraph.reset
decoder_pass_buckets
```

### `tensorrt_llm/_torch/models/modeling_deepseekv4.py`（17）

```text
DeepseekV4ForCausalLM._post_load_decoder_layers
DeepseekV4ForCausalLM.load_weights
DeepseekV4ForCausalLM.post_load_weights
DeepseekV4ForCausalLM.prepare_disagg_generation_request
DeepseekV4ForCausalLM.register_cuda_graph_pre_replay_hooks
DeepseekV4ForCausalLM.runtime_hooks
DeepseekV4MTP.forward
DeepseekV4MTP.forward.<locals>.norm_embeds
DeepseekV4MTP.forward.<locals>.norm_hidden
DeepseekV4MTPHead.forward
DeepseekV4MoE._forward_shared_into_combine
DeepseekV4MoE._forward_shared_into_combine.<locals>.issue
DeepseekV4WeightLoader._load_weights_impl.<locals>.load_flat_hc_weights
DeepseekV4WeightLoader._load_weights_impl.<locals>.load_o_a_proj
DeepseekV4WeightLoader._load_weights_impl.<locals>.rename_moe_weight
_block128_codes
full_tp_shared_expert_weights
```

### `tensorrt_llm/_torch/models/modeling_deepseekv41.py`（35）

```text
DeepseekV41Attention._forward_live_rows
DeepseekV41Attention._forward_live_rows.<locals>.forward_live_rows
DeepseekV41Attention.forward.<locals>.attend_live_rows
DeepseekV41DecoderLayer._engram_live_rows
DeepseekV41DecoderLayer._engram_live_rows.<locals>.apply
DeepseekV41Engram._exchange_ranks
DeepseekV41Engram._kv_projection_for
DeepseekV41Engram._local_token_embeddings
DeepseekV41Engram.precompute_lookup
DeepseekV41Engram.project_combined
DeepseekV41Engram.project_lookup
DeepseekV41Engram.set_ll_allgather
DeepseekV41Engram.set_wkv_shard
DeepseekV41ForCausalLM.get_model_defaults
DeepseekV41ForCausalLM.prepare_adp_inputs
DeepseekV41ForCausalLM.register_cuda_graph_pre_replay_hooks
DeepseekV41ForCausalLM.register_cuda_graph_pre_replay_hooks.<locals>._refresh
DeepseekV41Model._engram_after_attention_callback.<locals>.project
DeepseekV41Model._engram_precompute_eligible
DeepseekV41Model._precompute_engram_kv_pair
DeepseekV41Model._run_decoder_pass
DeepseekV41Model._run_encoder_pass
DeepseekV41Model._run_encoder_pass.<locals>.project
DeepseekV41Model._supports_engram_context_overlap.<locals>.packs_rows
DeepseekV41Model.mark_prologue_start
_EngramKvPair.__init__
_EngramKvPair.launch_on
_EngramKvPair.take
_EngramKvPairShard.__init__
_EngramKvShard.__init__
_ExchangedEngramLookup.local_num_tokens
_ExchangedEngramLookup.receive
_PrologueFork.__init__
_PrologueFork.join
_shared_expert_takes_fp8_1x128
```

### `tensorrt_llm/_torch/models/modeling_deepseekv41_multimodal.py`（11）

```text
DeepseekV41ForConditionalGeneration.draft_config
DeepseekV41ForConditionalGeneration.draft_model
DeepseekV41ForConditionalGeneration.get_model_defaults
DeepseekV41ForConditionalGeneration.load_draft_weights
DeepseekV41ForConditionalGeneration.load_weights
DeepseekV41ForConditionalGeneration.load_weights.<locals>.load_image_weight
DeepseekV41ForConditionalGeneration.multimodal_data_device_paths
DeepseekV41ForConditionalGeneration.post_load_weights
DeepseekV41ForConditionalGeneration.prepare_adp_inputs
DeepseekV41ForConditionalGeneration.register_cuda_graph_pre_replay_hooks
DeepseekV41ForConditionalGeneration.samples_from_vocab_shards
```

### `tensorrt_llm/_torch/models/modeling_dspark.py`（28）

```text
CompactTPArgmax.prepare
DSv4DSparkBlock._capture_quant_config
DSv4DSparkDraftModel.__init__
DSv4DSparkDraftModel._tp_attention_step
DSv4DSparkDraftModel.forks_backfill
DSv4DSparkDraftModel.prepare_greedy_proposals
DSv4DSparkDraftModel.project_backfill_rows
DSv4DSparkDraftModel.project_captured_rows
DSv4DSparkDraftModel.run_moe_lockstep_noop
DSv4DSparkForCausalLM.__init__
DSv4DSparkForCausalLM.block_size
DSv4DSparkForCausalLM.forks_backfill
DSv4DSparkForCausalLM.forward_batched
DSv4DSparkForCausalLM.load_weights_from_target_model
DSv4DSparkForCausalLM.prepare_greedy_proposals
DSv4DSparkForCausalLM.project_backfill_rows
DSv4DSparkForCausalLM.project_captured_rows
DSv4DSparkForCausalLM.run_moe_lockstep_noop
DSv4DSparkForCausalLM.write_context_windows
DSv4DSparkForCausalLM.write_context_windows_batched
_PrivateMnnvlAllReduce.__init__
_PrivateMnnvlAllReduce.forward
_active_moe_load_balancer
_all_reduce_add_norm
_private_allreduce
_rename_dspark_stage_subkey
count_dspark_stages
remap_dspark_draft_keys
```

### `tensorrt_llm/_torch/models/modeling_dspark_v41.py`（9）

```text
DSv41DSparkBlock._capture_quant_config
_draft_activation_amax
_exponents
_materialize
_mxfp4_to_nvfp4
_nonzero_blocks
_operand_exponent_range
_pow2_to_e4m3
convert_draft_experts_to_nvfp4
```

### `tensorrt_llm/_torch/models/modeling_glm.py`（3）

```text
Glm4Attention.__init__
Glm4DecoderLayer._compute_mlp_tp_size
Glm4MTP.__init__
```

### `tensorrt_llm/_torch/models/modeling_glm5_next.py`（12）

```text
Glm5NextForCausalLM.infer_max_seq_len
Glm5NextForCausalLM.load_weights
Glm5NextForCausalLM.mamba_metadata_cls
Glm5NextGenerationRows.build
Glm5NextLinearAttention.finalize_weights
Glm5NextMTP.__init__
Glm5NextMTPHead.__init__
Glm5NextMTPHead.forward
Glm5NextRuntimeContext.generation_rows
Glm5NextSchedule.attention_indices
Glm5NextSparseAttention.forward_decode
Glm5NextSparseAttention.forward_verify
```

### `tensorrt_llm/_torch/models/modeling_glm5_next_vision.py`（5）

```text
Glm5NextInputProcessor.tokenizer
Glm5NextVLM.draft_config
Glm5NextVLM.draft_model
Glm5NextVLM.load_weights
Glm5NextVLM.mamba_metadata_cls
```

### `tensorrt_llm/_torch/models/modeling_gpt_oss.py`（2）

```text
MLPBlock.forward_attn_dp
TransformerBlock.forward_tp
```

### `tensorrt_llm/_torch/models/modeling_kimi_k25.py`（11）

```text
KimiK25ForConditionalGeneration.draft_config
KimiK25ForConditionalGeneration.draft_model
KimiK25ForConditionalGeneration.infer_max_seq_len
KimiK25ForConditionalGeneration.lm_head
KimiK25ForConditionalGeneration.load_weights
KimiK25ForConditionalGeneration.model
KimiK25InputProcessor.__init__
KimiK25InputProcessor.call_with_text_prompt
KimiK25InputProcessor.tokenizer
KimiK25VisionModel.load_weights
_detect_prefix
```

### `tensorrt_llm/_torch/models/modeling_llama.py`（7）

```text
Llama4ForConditionalGeneration.infer_max_seq_len
Llama4InputProcessor.__init__
Llama4InputProcessor.call_with_text_prompt
Llama4InputProcessor.config
Llama4InputProcessor.processor
Llama4InputProcessor.tokenizer
_load_checkpoint_into_module
```

### `tensorrt_llm/_torch/models/modeling_minimaxm3.py`（7）

```text
MiniMaxM3Attention._msa_backend_active
MiniMaxM3Attention._split_main_qkv
MiniMaxM3Gate.routing_method
MiniMaxM3MoE.__init__
_extract_minimax_m3_attention_extra_attrs
_fold_gemma_boundary_norm_weights
get_text_model_config
```

### `tensorrt_llm/_torch/models/modeling_mistral.py`（5）

```text
Mistral3VLM.load_draft_weights
MistralCommonImageProcessor._get_num_multimodal_tokens
MistralCommonImageProcessor.get_num_tokens_per_image
MistralNativeInputProcessor.get_mm_special_token_ids
MistralNativeInputProcessor.get_vocab_size
```

### `tensorrt_llm/_torch/models/modeling_mistral_large3.py`（5）

```text
Mistral3Gate.__init__
Mistral3Gate.forward
Mistral3Gate.load_weights
MistralLarge3ForCausalLM.forward
MistralLarge3ForCausalLM.load_weights
```

### `tensorrt_llm/_torch/models/modeling_musespark_open.py`（4）

```text
MuseSparkOpenMoE._apply
MuseSparkOpenMoE.cache_derived_state
MuseSparkOpenMoE.post_load_weights
MuseSparkOpenMoE.prepare_expert_input
```

### `tensorrt_llm/_torch/models/modeling_qwen3_next.py`（11）

```text
Qwen3NextFullAttentionDecoderLayer._attach_expert_prefetch
Qwen3NextLinearDecoderLayer._attach_expert_prefetch
Qwen3NextModel._cache_post_moe_norm_folds
Qwen3NextModel._cache_tp_decode_gates
Qwen3NextModel._drop_expert_prefetch_without_record
Qwen3NextModel._init_expert_prefetch
Qwen3NextModel.share_lamport_fc2
Qwen3NextSparseMoeBlock._bf16_experts_defer_finalize
_Qwen3NextDecoderLayerBase.set_expert_prefetch
_fp8_input_projection
_takes_resident_band
```

### `tensorrt_llm/_torch/models/modeling_speculative.py`（8）

```text
DSparkConfidenceHead._host_sts_temperatures
MistralLarge3DraftModel.__init__
MistralLarge3DraftModel.forward
MistralLarge3DraftModel.post_load_weights
MistralLarge3EagleForCausalLM.__init__
MistralLarge3EagleForCausalLM.apply_eagle3_fc
MistralLarge3EagleForCausalLM.load_weights
MistralLarge3EagleForCausalLM.load_weights_from_target_model
```

### `tensorrt_llm/_torch/models/modeling_step3p7.py`（27）

```text
ClampedGatedMLP.__init__
Step3p7Attention.__init__
Step3p7Attention.apply_qk_norm
Step3p7Attention.apply_rope
Step3p7DecoderLayer.__init__
Step3p7ForCausalLM.__init__
Step3p7ForCausalLM._capture_bf16_clamp_weights
Step3p7ForCausalLM.forward
Step3p7MTP.__init__
Step3p7MTP.forward
Step3p7MTPHead._get_last_token_states
Step3p7MoE._allocate_clamp_buffers
Step3p7MoE._copy_clamp_weights
Step3p7MoE._python_clamped_moe_forward
Step3p7MoE.load_clamp_weights_from_fp8_experts
Step3p7RouterBiasHolder.load_weights
Step3p7TextModel.__init__
Step3p7TextModel.forward
_build_per_layer_config
_ceil_div
_copy_model_config_with_quant
_fp8_block_dequant_3d
_mirror_step3p7_text_aliases
_model_config_keep_kv_quant
_model_config_without_quant
_parse_python_fallback_threshold
_select_python_expert_path
```

### `tensorrt_llm/_torch/models/modeling_step3p7vl.py`（8）

```text
Step3p7VLForConditionalGeneration.__init__
Step3p7VLForConditionalGeneration.draft_model
Step3p7VLForConditionalGeneration.forward
Step3p7VLForConditionalGeneration.infer_max_seq_len
Step3p7VLForConditionalGeneration.lm_head
Step3p7VLForConditionalGeneration.load_weights
Step3p7VLForConditionalGeneration.model
Step3p7VLInputProcessor.tokenizer
```

### `tensorrt_llm/_torch/models/modeling_utils.py`（2）

```text
forward_after_recv.<locals>.forward_after_recv_fn
forward_before_send.<locals>.forward_before_send_fn
```

### `tensorrt_llm/_torch/models/musespark_open_boundary.py`（5）

```text
_ClusterBoundaryKernel.__init__
_cluster_size
_compile_boundary
_float_bits
cluster_boundary
```

### `tensorrt_llm/_torch/models/musespark_open_finalize_ar.py`（44）

```text
FinalizeIpcWorkspace.__del__
FinalizeIpcWorkspace.__init__
FinalizeIpcWorkspace._abandon
FinalizeIpcWorkspace._agree
FinalizeIpcWorkspace._allocate_ipc
FinalizeIpcWorkspace._close_imports
FinalizeIpcWorkspace._failed_ranks
FinalizeIpcWorkspace._free_export
FinalizeIpcWorkspace._guarded
FinalizeIpcWorkspace._initialize
FinalizeIpcWorkspace._launch
FinalizeIpcWorkspace._prepare
FinalizeIpcWorkspace._release_together
FinalizeIpcWorkspace._retain
FinalizeIpcWorkspace._sync
FinalizeIpcWorkspace._validate
FinalizeIpcWorkspace.close
FinalizeIpcWorkspace.finalize
_FinalizeAllReduceKernel.__init__
_arrive
_asm
_compiled_kernel
_compiled_kernel.<locals>.placeholder
_describe
_device_id
_f32
_fingerprint
_host_id
_i64
_load_control
_load_input_f32
_load_input_s32
_load_input_x4
_load_peer_x4
_pack_bf16x2
_pack_publishable_bf16x2
_pdl_enabled
_split_words
_store_control
_store_peer_x4
_store_x4
_u32
_unpack_bf16x2
_zero_unless
```

### `tensorrt_llm/_torch/modules/context_logits.py`（1）

```text
ContextLogitsWorkspace._allgather
```

### `tensorrt_llm/_torch/modules/decoder_layer.py`（1）

```text
DecoderLayer.skip_forward
```

### `tensorrt_llm/_torch/modules/dwdp/page_pool.py`（1）

```text
PagePool.map_pages
```

### `tensorrt_llm/_torch/modules/dwdp/setup.py`（1）

```text
fill_edge_bytes
```

### `tensorrt_llm/_torch/modules/dwdp/specs.py`（5）

```text
EdgeInfo.__post_init__
MnnvlHandleSet.get_handle
PageAlignedLayout.get_edge_info
PageAlignedLayout.post_pages
PageAlignedLayout.pre_pages
```

### `tensorrt_llm/_torch/modules/dwdp/transport.py`（2）

```text
_export_handle
_import_handle
```

### `tensorrt_llm/_torch/modules/dwdp/vmm.py`（5）

```text
create_fabric_handle
free_va
map_handle
reserve_va
unmap_va
```

### `tensorrt_llm/_torch/modules/dwdp/weight_buffer.py`（11）

```text
WeightBuffer._compute_remote_slices
WeightBuffer._setup_layer
WeightBuffer.device_id
WeightBuffer.get_edge_info
WeightBuffer.get_full_tensor
WeightBuffer.get_layout
WeightBuffer.get_remote_slices
WeightBuffer.layer_indices
WeightBuffer.local_end
WeightBuffer.local_start
WeightBuffer.weight_names
```

### `tensorrt_llm/_torch/modules/dwdp/weight_manager.py`（5）

```text
DWDPWeightManager._build_gather_plan
DWDPWeightManager._prefetch_layer_batched
DWDPWeightManager._prefetch_layer_gather
DWDPWeightManager.use_gather
DWDPWeightManager.use_tma
```

### `tensorrt_llm/_torch/modules/embedding.py`（2）

```text
LMHead.local_logits
_
```

### `tensorrt_llm/_torch/modules/engram/_evict_page_cache.py`（4）

```text
_cgroup_headroom
_meminfo
_reclaimable_cache
main
```

### `tensorrt_llm/_torch/modules/engram/_host_memory.py`（5）

```text
HostNumaMem.export_fd
HostRegisteredMem.export_fd
ImportedHostMem.__init__
evict_numa_node_page_cache
node_gpu_numa_nodes
```

### `tensorrt_llm/_torch/modules/engram/_node_share.py`（11）

```text
NodeShare.__init__
NodeShare._claim
NodeShare._wait_for_owner
NodeShare.join
NodeShare.mark_ready
NodeShare.publish
NodeShare.wait_ready
_Served.__init__
_close_listener
_peer_uid
_serve
```

### `tensorrt_llm/_torch/modules/engram/adp_exchange.py`（10）

```text
EngramAdpExchange.__init__
EngramAdpExchange._hash_slot_packets
EngramAdpExchange._rows_base
EngramAdpExchange.buffer_bytes
EngramAdpExchange.create
EngramAdpExchange.exchange_hashes
EngramAdpExchange.push
EngramAdpExchange.receive
EngramAdpExchange.takes
engine_capacity
```

### `tensorrt_llm/_torch/modules/engram/engram.py`（11）

```text
ShardedFp8MultiHeadEmbedding._load_node_shared
ShardedFp8MultiHeadEmbedding._rows_match
ShardedFp8MultiHeadEmbedding._table_bytes
ShardedFp8MultiHeadEmbedding._table_views
ShardedFp8MultiHeadEmbedding.empty_packed
ShardedFp8MultiHeadEmbedding.forward_packed
ShardedFp8MultiHeadEmbedding.post_load_weights
_import_host_mem
_join_off_home
_shareable_host_mem
_shared_table_home_node
```

### `tensorrt_llm/_torch/modules/engram/projection.py`（2）

```text
EngramFp8Projection.forward_packed
EngramFp8Projection.supports_packed_input
```

### `tensorrt_llm/_torch/modules/expert_l2_prefetch.py`（2）

```text
ExpertL2Prefetch.registered
ExpertL2Prefetch.unregister
```

### `tensorrt_llm/_torch/modules/linear.py`（4）

```text
Linear.pre_reload_weights
LinearMethodBase.apply_linear_allreduce_norm
LinearMethodBase.pre_reload_weights
flashinfer_mxfp8_autotune_active
```

### `tensorrt_llm/_torch/modules/megamoe_shared_mlp.py`（9）

```text
MegaMoESharedMLP.__init__
MegaMoESharedMLP._validate_fp8_quantization
MegaMoESharedMLP.cache_derived_state
MegaMoESharedMLP.forward
MegaMoESharedMLP.post_load_weights
MegaMoESharedMLP.prepare_fc12_weights
MegaMoESharedMLP.sm_count
_interleave_gate_up
_pack_r128c4
```

### `tensorrt_llm/_torch/modules/mhc/mhc_cuda.py`（1）

```text
MhcPreMappingRunner.get_valid_tactics
```

### `tensorrt_llm/_torch/modules/mhc/mhc_lowm.py`（8）

```text
LamportAllReduce.__init__
LamportAllReduce.prepare
MnnvlLamportAllReduce.__init__
MnnvlLamportAllReduce.prepare
MoeUnfinalized.finalize
_compiled_allreduce_boundary
_placeholder_moe_operands
mhc_allreduce_post_collapse_norm
```

### `tensorrt_llm/_torch/modules/multi_stream_utils.py`（1）

```text
eager_forks_in_capture
```

### `tensorrt_llm/_torch/modules/q35_mtp_glue.py`（1）

```text
mtp_fused_input
```

### `tensorrt_llm/_torch/modules/rms_norm.py`（1）

```text
RMSNorm.skip_forward
```

### `tensorrt_llm/_torch/moe/custom_ops/cute_dsl_megamoe_custom_op.py`（4）

```text
MegaMoeSymmMemProvider.__init__
MegaMoeSymmMemProvider._build_regions
MegaMoeSymmMemProvider._region_view
get_megamoe_symm_provider
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/allgather_reducescatter.py`（2）

```text
AllGatherReduceScatter.combine
AllGatherReduceScatter.dispatch
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/base.py`（4）

```text
Communication.destroy
Communication.prepare_dispatch
Communication.supports_post_quant_dispatch
Communication.uses_internal_dispatch_quantization
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/deep_ep.py`（10）

```text
DeepEP.__init__
DeepEP._is_deepep_feasible
DeepEP._pad_empty_recv_tensors
DeepEP._unpad_tensors
DeepEP.combine
DeepEP.destroy
DeepEP.dispatch
DeepEP.is_platform_supported
DeepEP.is_workload_feasible
DeepEP.supports_post_quant_dispatch
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/deep_ep_low_latency.py`（7）

```text
DeepEPLowLatency.__init__
DeepEPLowLatency._has_fp8_qdq
DeepEPLowLatency._has_nvfp4
DeepEPLowLatency._has_w4afp8
DeepEPLowLatency.combine
DeepEPLowLatency.dispatch
DeepEPLowLatency.is_workload_feasible
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/nccl_ep.py`（2）

```text
NcclEP.combine
NcclEP.destroy
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/nvlink_one_sided.py`（24）

```text
NVLinkOneSided._alltoall_watchdog
NVLinkOneSided._combine_pull
NVLinkOneSided._init_constants
NVLinkOneSided._mnnvl_checkpoint_is_idle
NVLinkOneSided._mnnvl_checkpoint_reset
NVLinkOneSided._vote_sentinel
NVLinkOneSided._watchdog_coordinator
NVLinkOneSided.carries_unrouted_rows
NVLinkOneSided.checkpoint_resource_key
NVLinkOneSided.checkpoint_restore.<locals>.reinitialize_frontend
NVLinkOneSided.combine
NVLinkOneSided.combine_fuses_addend
NVLinkOneSided.dispatch
NVLinkOneSided.enable_pull_combine
NVLinkOneSided.get_combine_payload_tensor_in_workspace
NVLinkOneSided.get_eplb_gathered_statistics
NVLinkOneSided.is_platform_supported
NVLinkOneSided.is_workload_feasible
NVLinkOneSided.moe_a2a_metainfo
NVLinkOneSided.pull_combine_applies
NVLinkOneSided.reset_state
NVLinkOneSided.supports_post_quant_dispatch
_NVLinkOneSidedTimeoutProxy.handshake_timeout_seconds
_sentinel_unsupported_reason
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/nvlink_two_sided.py`（12）

```text
MnnvlMoe.checkpoint_prepare
MnnvlMoe.get_moe_prepare_workspace
MnnvlMoe.get_moe_workspaces
MnnvlMoe.mnnvl_moe_alltoallv
MnnvlMoe.mnnvl_moe_alltoallv_combine
MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather
MnnvlMoe.require_mapped
NVLinkTwoSided.checkpoint_resource_key
NVLinkTwoSided.is_platform_supported
NVLinkTwoSided.is_workload_feasible
NVLinkTwoSided.prepare_dispatch
NVLinkTwoSided.supports_post_quant_dispatch
```

### `tensorrt_llm/_torch/moe/fused_moe/communication/pull_combine.py`（4）

```text
PullCombineSync.__init__
PullCombineSync._flag
PullCombineSync.signal_wait
pull_reduce
```

### `tensorrt_llm/_torch/moe/fused_moe/configurable_moe.py`（3）

```text
ConfigurableMoE._dp_padded_num_rows
ConfigurableMoE.fuse_shared_expert
ConfigurableMoE.has_nvfp4
```

### `tensorrt_llm/_torch/moe/fused_moe/cutedsl/grouped_gemm/base.py`（1）

```text
CuteDslFusedMoEBase.supports_moe_output_in_alltoall_workspace
```

### `tensorrt_llm/_torch/moe/fused_moe/cutlass/base.py`（1）

```text
CutlassFusedMoEBase.supports_moe_output_in_alltoall_workspace
```

### `tensorrt_llm/_torch/moe/fused_moe/deep_ep_utils.py`（12）

```text
BufferPool.get_buffer
BufferPool.get_low_latency_buffer
VariableLengthBuffer.__del__
VariableLengthBuffer.__init__
VariableLengthBuffer.combine
VariableLengthBuffer.dispatch
VariableLengthBuffer.reserve
VariableLengthLowLatencyBuffer.__del__
VariableLengthLowLatencyBuffer.__init__
VariableLengthLowLatencyBuffer.low_latency_combine
VariableLengthLowLatencyBuffer.low_latency_dispatch
VariableLengthLowLatencyBuffer.reserve
```

### `tensorrt_llm/_torch/moe/fused_moe/fused_moe_deepgemm.py`（1）

```text
DeepgemmCudaFp8BlockScalesImpl.supports_moe_output_in_alltoall_workspace
```

### `tensorrt_llm/_torch/moe/fused_moe/fused_moe_triton.py`（1）

```text
TritonEPRouter.prune_routing_ep
```

### `tensorrt_llm/_torch/moe/fused_moe/impl_base.py`（1）

```text
MoEImplBase.takes_router_logit_partials
```

### `tensorrt_llm/_torch/moe/fused_moe/impl_blocks.py`（3）

```text
MoEEplbWeightLayoutMixin.register_all_parameter_slot_and_to_fix_weight_fns
MoEEplbWeightLayoutMixin.register_parameter_weight_slot_fn
MoEEplbWeightLayoutMixin.register_to_fix_weight_fn
```

### `tensorrt_llm/_torch/moe/fused_moe/interface.py`（7）

```text
MoE._get_perfect_router_dtype
MoE._load_balancer_count_warmup
MoE._load_balancer_done_wait_gpu_stage
MoE._load_balancer_get_local_statistic_tensor
MoE._load_balancer_update_statistic
MoE._load_balancer_update_statistic_with_gathered_statistic
MoE.enable_alltoall
```

### `tensorrt_llm/_torch/moe/fused_moe/marlin/base.py`（1）

```text
MarlinFusedMoEBase.supports_moe_output_in_alltoall_workspace
```

### `tensorrt_llm/_torch/moe/fused_moe/mega_moe/mega_moe_cute_dsl.py`（10）

```text
TrtllmCutedslMegaMoeNvfp4Impl._prime_ladder
TrtllmCutedslMegaMoeNvfp4Impl._prime_ladder_buckets
TrtllmCutedslMegaMoeNvfp4Impl._routing_table
TrtllmCutedslMegaMoeNvfp4Impl._supports_load_balancer
TrtllmCutedslMegaMoeNvfp4Impl._symm_provider
TrtllmCutedslMegaMoeNvfp4Impl.fuse_shared_expert
TrtllmCutedslMegaMoeNvfp4Impl.on_expert_placement_changed
TrtllmCutedslMegaMoeNvfp4Impl.takes_router_logit_partials
_slot_table
replica_slot_table
```

### `tensorrt_llm/_torch/moe/fused_moe/mega_moe/mega_moe_deepgemm.py`（3）

```text
DeepgemmCudaW4a8Mxfp4Mxfp8Impl._maybe_init_dist_from_mpi
DeepgemmCudaW4a8Mxfp4Mxfp8Impl._supports_load_balancer
_free_symm_buffer
```

### `tensorrt_llm/_torch/moe/fused_moe/mega_moe/shared_expert.py`（6）

```text
_blocked_scale_factors
_interleave_gate_up
_mega_projection
_ue8m0_row_scales
build_shared_expert_weights
shared_activation_buffers
```

### `tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py`（18）

```text
HostMoeTensorSharer.add_raw_host_weight_for_unmap
HostMoeTensorSharer.pre_register_host_tensor_with_shape
MoeLoadBalancer.register_weight_slots_after_to_cuda
SingleLayerMoeLoadBalancer._add_host_weight_from_tensor
SingleLayerMoeLoadBalancer._add_raw_host_weight_for_unmap
SingleLayerMoeLoadBalancer.add_register_weight_fn
SingleLayerMoeLoadBalancer.add_to_migrate_weight_fn
SingleLayerMoeLoadBalancer.get_load_expert_ids
SingleLayerMoeLoadBalancer.get_old_rank_expert_ids
SingleLayerMoeLoadBalancer.is_static_routing
SingleLayerMoeLoadBalancer.make_tensor_host_accessible
SingleLayerMoeLoadBalancer.need_load_shared_weights
SingleLayerMoeLoadBalancer.register_weight_slot
SingleLayerMoeLoadBalancer.register_weight_slots_after_to_cuda
SingleLayerMoeLoadBalancer.update_statistic_with_gathered_statistic
SingleLayerMoeLoadBalancer.update_statistic_with_gathered_statistic.<locals>._update_statistic
SingleLayerMoeLoadBalancer.update_statistic_with_local_ids
SingleLayerMoeLoadBalancer.update_statistic_with_local_ids.<locals>._update_statistic
```

### `tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py`（3）

```text
FusedCommMoEScheduler.routes_from_logit_partials
_record_backend_slots
routes_padding_nowhere
```

### `tensorrt_llm/_torch/moe/fused_moe/moe_warmup_placement.py`（46）

```text
WarmupPlacementController.__init__
WarmupPlacementController._check_movable
WarmupPlacementController._commit_tables
WarmupPlacementController._counting
WarmupPlacementController._decay_counters
WarmupPlacementController._freeze
WarmupPlacementController._gather_checksums
WarmupPlacementController._layer_groups
WarmupPlacementController._make_ep_comm
WarmupPlacementController._make_exchange_group
WarmupPlacementController._max_swaps
WarmupPlacementController._migrate
WarmupPlacementController._migrate_group
WarmupPlacementController._prepare_local
WarmupPlacementController._round
WarmupPlacementController._verify
WarmupPlacementController.apply_plan
WarmupPlacementController.finalize
WarmupPlacementController.on_step
WarmupPlacementController.register_layer
WarmupPlacementController.route_counters
WarmupPlacementController.set_live_rows
_excess
_follows_placement
_generated_tokens
_moe_and_backend
_pack_slot
_replica_excess
_replica_inputs
_resolve_slot_tensors
_row
_slot_checksums
_slot_runs
_slot_words
_unpack_slot
_weights_owner
build_transfers
copy_transfers
is_replicated_cover
plan_layer_copy_moves
plan_layer_swaps
rank_loads
real_rows_of
schedule_action
target_copies
transfer_groups
```

### `tensorrt_llm/_torch/moe/fused_moe/nccl_ep_utils.py`（6）

```text
NcclEpContext.__init__
NcclEpContext.destroy
NcclEpContext.get_stream
_nccl_ep_supports_int32_topk_idx
get_nccl_ep_context
release_nccl_ep_context
```

### `tensorrt_llm/_torch/moe/fused_moe/quantization.py`（5）

```text
NVFP4MegaMoECuteDslMethod._build_mega_shared_staging
NVFP4MegaMoECuteDslMethod._register_mega_shared_staging
NVFP4MegaMoECuteDslMethod.finalize_streamed_expert
NVFP4MegaMoECuteDslMethod.prepare_streaming_expert_load
W4A8MXFP4MXFP8MegaMoEDeepGemmMethod._setup_shared_weights_for_eplb.<locals>.refresh_deepgemm_scale_views
```

### `tensorrt_llm/_torch/moe/fused_moe/routing.py`（3）

```text
BasePerfectRouterPlanner.get_cache_signature
get_perfect_router_cache_stats
precompute_common_perfect_router_logits
```

### `tensorrt_llm/_torch/moe/fused_moe/trtllm_gen/decode_front.py`（2）

```text
DecodeFront._front_tactic
DecodeFront.set_expert_prefetch
```

### `tensorrt_llm/_torch/moe/fused_moe/trtllm_gen/trtllm_nvfp4.py`（1）

```text
TrtllmTrtllmGenNvfp4Impl.try_fused_route
```

### `tensorrt_llm/_torch/moe/fused_moe/trtllm_gen/trtllm_w4a8_mxfp4_mxfp8.py`（1）

```text
TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl._try_kimi_k3_route_quant
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain/one_round_router.py`（15）

```text
_OneRoundRouter.__init__
_OneRoundRouter._build_smem_workspace
_OneRoundRouter._validate_configuration
_OneRoundRouter.impl_desc_require
_OneRoundRouter.problem_desc_require
_OneRoundRouter.register_device_workspace
_OneRoundRouter.worst_case_padded_tokens
_f32_as_s32
_ld_acquire_sys_s32
_ld_b64
_ldg_v4_b32
_s32_as_f32
_st_release_sys_s32
_stg_v4_b32
fence_proxy_async_global
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain/token_comm.py`（4）

```text
TokenCommNonDeterministic._register_token_back_staging
TokenCommNonDeterministic.combine_sf_hidden_padded
TokenCommNonDeterministic.combine_stage_depth
TokenCommNonDeterministic.fc2_done_buffer_space
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/ptx_helpers.py`（3）

```text
movmatrix_b16
read_clock64
stg_f32
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/smem_workspace.py`（2）

```text
SmemWorkspace.byte_alignment
SmemWorkspace.offset
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/topk_reduce.py`（3）

```text
_mixed_add_bf16_asm
add_bf16_into_f32_ptx
cvt_e2m1_to_fp32_cvt_ptx
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_epilogue.py`（18）

```text
Fc2OutputRouter.sf_mapping
Fc2OutputRouter.sf_output
Fc2OutputRouter.sf_token_base
QuantImpl.scale_dtype
SwapABGatedActEpilogue._init_shared_expert
SwapABGatedActEpilogue.fc2_tma_staged_smem_layout
SwapABGatedActEpilogue.fc2_tma_staged_smem_spec
SwapABGatedActEpilogue.prepare_shared_tma_store_params
TmemTranspose32x64B16Movm._rmem_copy_view
TmemTranspose32x64B16Movm._tmem_layout
fc2_stg_post_f2fp_reorder
fc2_stg_store_function
fc2_stg_tmem_acc_load
fc2_tma_store_function
make_fc2_stg_cta_store_out_mapping
make_fc2_stg_cta_store_out_mapping.<locals>.stg_sf_mapping
make_fc2_stg_process_pipeline
make_fc2_tma_process_pipeline
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_kernel.py`（8）

```text
BlockScaledSwapAbFc12Kernel.__init__
BlockScaledSwapAbFc12Kernel._build_device_workspace
BlockScaledSwapAbFc12Kernel._build_mainloop_and_smem
BlockScaledSwapAbFc12Kernel._validate_geometry
BlockScaledSwapAbFc12Kernel.get_workspace_sizes
BlockScaledSwapAbFc12Kernel.impl_desc_require
BlockScaledSwapAbFc12Kernel.problem_desc_require
BlockScaledSwapAbFc12Kernel.require_zero_workspace_leading_bytes
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_mainloop.py`（6）

```text
BlockScaledSwapAbFc12Mainloop._alias_smem_tensor
BlockScaledSwapAbFc12Mainloop._make_tma_maybe_pair
BlockScaledSwapAbFc12Mainloop._materialize_shared_codegen_members
BlockScaledSwapAbFc12Mainloop._restage_sf_layout
BlockScaledSwapAbFc12Mainloop._select_shared_stage_plan
BlockScaledSwapAbFc12Mainloop.prepare_shared_tma_load_params
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py`（1）

```text
BlockScaledSwapAbMegaMoeKernel._init_shared_expert
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/fc12_scheduler.py`（3）

```text
PhaseInterleavedFc12Scheduler._map_shared_fc2_work_id
PhaseInterleavedFc12Scheduler.claim_shared_fc2_tiles
PhaseInterleavedFc12Scheduler.set_shared_fc2_tokens
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/non_clc_mixed_cga.py`（1）

```text
NonClcMixedCgaSchedulerWorker.max_fallback_cluster_count
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/shared_fc12_scheduler.py`（2）

```text
SharedExpertTileSchedule.__init__
SharedExpertTileSchedule.register_device_workspace
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/work_id_claim.py`（7）

```text
FixedGroupMixedCgaAtomicCounterWorkIdState.__extract_mlir_values__
FixedGroupMixedCgaAtomicCounterWorkIdState.__init__
FixedGroupMixedCgaAtomicCounterWorkIdState.__new_from_mlir_values__
FixedGroupMixedCgaAtomicCounterWorkIdState.__new_from_mlir_values__.<locals>.rebuild
_claim_fixed_group_fallback_work_id
_claim_fixed_group_mixed_cga_work_id
initialize_fixed_group_mixed_cga_work_id_state
```

### `tensorrt_llm/_torch/moe/kernels/megamoe_shared_fc12.py`（4）

```text
_SharedFc12Runner.__call__
_SharedFc12Runner.__init__
_get_runner
_make_kernel
```

### `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py`（2）

```text
DSparkCUDAGraphRunner._padding_dummy_cache
DSparkCUDAGraphRunner._preallocate_extra_padding_dummy
```

### `tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/dummy_requests.py`（1）

```text
DummyRequestFactory._apply_chunk_alignment
```

### `tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/graphs.py`（1）

```text
DecoderCudaGraphs._set_warmup_ragged_windows
```

### `tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/inputs.py`（2）

```text
DecoderInputPreparer.ragged_verify_token_buckets
DecoderInputPreparer.replays_final_windows_only
```

### `tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/warmup.py`（1）

```text
DecoderWarmup._reset_moe_alltoall_state
```

### `tensorrt_llm/_torch/pyexecutor/guided_decoder.py`（1）

```text
GuidedDecoder.init_disagg_gen_requests
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/host_memory.py`（3）

```text
bind_to_numa_node
compact_node
host_numa_node_of
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`（2）

```text
KVCacheManagerV2.get_storage_statistics
KVCacheManagerV2.has_remote_cache_level
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/mamba_cache_manager/common.py`（1）

```text
BaseMambaCacheManager.on_state_transfer_complete
```

### `tensorrt_llm/_torch/pyexecutor/kv_cache/remote_donor.py`（5）

```text
FabricDonorTransport._fabric_prop
FabricDonorTransport._release_fabric
FabricDonorTransport._try_create
FabricDonorTransport.start
FabricDonorTransport.start.<locals>.create
```

### `tensorrt_llm/_torch/pyexecutor/llm_request.py`（1）

```text
PyResult.reset_diff
```

### `tensorrt_llm/_torch/pyexecutor/py_executor.py`（13）

```text
PyExecutor._broadcast_sample_state_loop
PyExecutor._forward_step_inter_pp
PyExecutor._revert_ctx_alloc
PyExecutor._ring_broadcast_sample_state
PyExecutor.begin_sleep_transition
PyExecutor.begin_wakeup_transition
PyExecutor.can_shutdown
PyExecutor.complete_sleep_transition
PyExecutor.complete_wakeup_transition
PyExecutor.control_action
PyExecutor.get_memory_status
PyExecutor.wait_for_intake_window
PyExecutor.wait_shutdown
```

### `tensorrt_llm/_torch/pyexecutor/sampler/sampler_features.py`（3）

```text
AsyncWorkerMixin.async_worker_start
AsyncWorkerMixin.async_worker_start.<locals>._async_worker_initializer
AsyncWorkerMixin.async_worker_stop
```

### `tensorrt_llm/_torch/pyexecutor/scheduled_encoder_executor.py`（1）

```text
ScheduledEncoderExecutor.wait_shutdown
```

### `tensorrt_llm/_torch/speculative/dspark.py`（8）

```text
DSv4DSparkWorker.cache_derived_state
DSv4DSparkWorker.group_verify
DSv4DSparkWorker.prepare_vocab_sharded_sampling
DSv4DSparkWorker.samples_from_vocab_shards
DSv4DSparkWorker.set_draft_model
DSv4DSparkWorker.set_target_model
DSv4DSparkWorker.takes_vocab_sharded_target_logits
DSv4DSparkWorker.vocab_sharded_head
```

### `tensorrt_llm/_torch/speculative/dspark_markov_chain.py`（8）

```text
MarkovWorkspace.__init__
_compiled
_device_supported
_fabric_lamport_table
_num_ctas
allocate_markov_workspace
markov_chain_fused
markov_chain_supported
```

### `tensorrt_llm/_torch/speculative/dspark_profile.py`（1）

```text
DSparkCostProfiler.from_env
```

### `tensorrt_llm/_torch/speculative/eagle3.py`（1）

```text
Eagle3OneModelWorker._draft_kept_rows
```

### `tensorrt_llm/_torch/speculative/fused_accept.py`（1）

```text
sample_and_strict_accept
```

### `tensorrt_llm/_torch/speculative/interface.py`（3）

```text
SpecMetadata.dp_num_tokens
SpecWorkerBase.skip_forward
SpecWorkerBase.vocab_sharded_head
```

### `tensorrt_llm/_torch/speculative/sharded_sampling.py`（4）

```text
VocabShardedSampler.__init__
VocabShardedSampler.buffer_bytes
VocabShardedSampler.create
VocabShardedSampler.sample
```

### `tensorrt_llm/_torch/speculative/spec_sampler_base.py`（1）

```text
SpecSampler.bind_spec_step_fold
```

### `tensorrt_llm/_torch/speculative/vocab_parallel.py`（10）

```text
VocabParallelHead.__init__
VocabParallelHead.argmax
VocabParallelHead.gather_logits
VocabParallelHead.local_logits
VocabParallelHead.markov_tokens
VocabParallelHead.sample
VocabShardedLogits.__init__
VocabShardedLogits.num_rows
VocabShardedLogits.select_tokens
tp_pair_exchange
```

### `tensorrt_llm/_torch/utils.py`（2）

```text
create_lm_head_tp_mapping
nearest_in_buckets
```

### `tensorrt_llm/_torch/visual_gen/attention/attention.py`（4）

```text
Attention.forward_async
Attention.forward_async.<locals>.compute_k
Attention.forward_async.<locals>.compute_q
Attention.forward_async.<locals>.compute_v
```

### `tensorrt_llm/_torch/visual_gen/attention/backends/flash_attn4.py`（2）

```text
FlashAttn4Attention.support_fused_qkv
FlashAttn4Attention.support_lse
```

### `tensorrt_llm/_torch/visual_gen/attention/backends/interface.py`（1）

```text
AttentionBackend.support_fused_qkv
```

### `tensorrt_llm/_torch/visual_gen/attention/backends/parallel.py`（22）

```text
Attention2DAttention._combine
Attention2DAttention.forward
Attention2DAttention.preferred_layout
Attention2DAttention.support_fused_qkv
RingAttention.__init__
RingAttention._ensure_buffers
RingAttention._ring_send_recv
RingAttention._ring_wait
RingAttention._update_out_and_lse
RingAttention.forward
RingAttention.preferred_layout
RingAttention.support_fused_qkv
UlyssesAttention._forward_fused
UlyssesAttention._forward_packed_self_attention
UlyssesAttention._issue_async
UlyssesAttention._join_async
UlyssesAttention.forward_async
UlyssesAttention.preferred_layout
UlyssesAttention.support_kv_cache
_all_to_all_5d_raw
_ulysses_post_unscatter
post_permute_5d_to_4d
```

### `tensorrt_llm/_torch/visual_gen/mapping.py`（12）

```text
VisualGenMapping._attach_attn2d_groups_from_device_mesh
VisualGenMapping._build_vae_group
VisualGenMapping._get_host_id
VisualGenMapping._validate_vae_ranks_share_cfg_group
VisualGenMapping.attn2d_mesh_group
VisualGenMapping.build_mesh
VisualGenMapping.cp_col_rank
VisualGenMapping.cp_row_rank
VisualGenMapping.ring_group
VisualGenMapping.ring_rank
VisualGenMapping.setup_communicators
VisualGenMapping.to_autotuner_mapping
```

### `tensorrt_llm/_torch/visual_gen/models/flux/attention.py`（1）

```text
Flux2ParallelSelfAttention._apply_norm_rope_unfused
```

### `tensorrt_llm/_torch/visual_gen/models/flux/pipeline_flux.py`（1）

```text
FluxPipeline._compute_flux_timestep_embedding
```

### `tensorrt_llm/_torch/visual_gen/models/flux/transformer_flux.py`（4）

```text
CombinedTimestepGuidanceTextProjEmbeddings.forward
CombinedTimestepTextProjEmbeddings.__init__
CombinedTimestepTextProjEmbeddings.combine
CombinedTimestepTextProjEmbeddings.embed_conditions
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/parallel_vae.py`（3）

```text
_tile_in_volume
assign_tiles_lpt
tile_parallel_decode
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/pipeline_ltx2.py`（1）

```text
LTX2Pipeline.setup_parallel_vae
```

### `tensorrt_llm/_torch/visual_gen/models/ltx2/transformer_ltx2.py`（12）

```text
BasicAVTransformerBlock._sp_all_gather
LTX2Attention.forward_async
LTX2Attention.forward_async.<locals>.compute_k
LTX2Attention.forward_async.<locals>.compute_q
LTX2Attention.forward_async.<locals>.compute_v
LTX2Attention.set_active_attn
LTXModel._gather_sequence
LTXModel._pad_modality_audio
LTXModel._pad_pe
LTXModel._pad_pe.<locals>._ext
LTXModel._shard_transformer_args
LTXModel.set_ulysses_topology
```

### `tensorrt_llm/_torch/visual_gen/models/ltx23/transformer_ltx23.py`（1）

```text
LTX23Model._audio_shard_mode
```

### `tensorrt_llm/_torch/visual_gen/models/minimax_h3/pipeline_minimax_h3.py`（2）

```text
_MiniMaxH3TextEncoderMapping.tp_group_name
_MiniMaxH3TextEncoderMapping.tp_group_pg
```

### `tensorrt_llm/_torch/visual_gen/models/minimax_h3/text_encoder_tp.py`（5）

```text
MiniMaxH3TensorParallelTextEncoder.__init__
MiniMaxH3TensorParallelTextEncoder._map_weight_key
MiniMaxH3TensorParallelTextEncoder.encode
MiniMaxH3TensorParallelTextEncoder.from_pretrained_config
MiniMaxH3TensorParallelTextEncoder.load_weights
```

### `tensorrt_llm/_torch/visual_gen/models/wan/parallel_vae.py`（9）

```text
ParallelVAE_TrtllmWan._decode_impl
ParallelVAE_Wan._decode_impl
ParallelVAE_Wan._encode_impl
ParallelVAE_Wan._parallelize_modules
ParallelVAE_Wan._replace_attention
ParallelVAE_Wan._replace_conv3d
ParallelVAE_Wan._replace_resample_conv2d
ParallelVAE_Wan._replace_resample_conv2d_stride2
ParallelVAE_Wan.make_spec
```

### `tensorrt_llm/_torch/visual_gen/models/wan/parallel_wino_vae.py`（6）

```text
ParallelWinoVae._decode_impl
ParallelWinoVae._encode_impl
ParallelWinoVae._gather_tensor
ParallelWinoVae._parallelize_modules
ParallelWinoVae._split_uneven
shard_extents
```

### `tensorrt_llm/_torch/visual_gen/models/wan/wino_vae.py`（5）

```text
WinoVae.set_shard
WinoVaeAttentionBlock._qkv_sharded
WinoVaeCausalConv3d.winograd_inputs_align
WinogradInputs.physical
_frames_physical
```

### `tensorrt_llm/_torch/visual_gen/models/wan/wino_vae_shard.py`（26）

```text
DirectFrames.barrier
Shard.__init__
Shard._buffer
Shard._enable_direct
Shard._line
Shard._staged
Shard._swap
Shard.begin_call
Shard.conv_padding
Shard.direct_frames
Shard.direct_frames.<locals>.frames
Shard.direct_frames.<locals>.peer
Shard.direct_inputs
Shard.exchange
Shard.fill_pad_line
Shard.frame_key
Shard.gather_rows
Shard.geometry
Shard.output_extents
Shard.pad_frames
Shard.peer_geometry
Shard.row_counts
Shard.set_extents
Shard.uneven
Shard.zero_outside
_align_up
```

### `tensorrt_llm/_torch/visual_gen/modules/vae/attention.py`（2）

```text
ParallelVaeAttentionBlock.__init__
ParallelVaeAttentionBlock.forward
```

### `tensorrt_llm/_torch/visual_gen/modules/vae/conv.py`（11）

```text
HaloExchangeConv._exchange_halos
HaloExchangeConv._strip_halo
HaloExchangeConv.forward
HaloExchangeConv2dStride2.__init__
HaloExchangeConv2dStride2._recv_from_right
HaloExchangeConv2dStride2.forward
_cat_spatial_halos
_halo_exchange_buffer
_logical_to_physical_channels_last
_physical_to_logical_channels_last
_spatial_channels_last_format
```

### `tensorrt_llm/_torch/visual_gen/modules/vae/norm.py`（2）

```text
GroupNormParallel.__init__
GroupNormParallel.forward
```

### `tensorrt_llm/_torch/visual_gen/modules/vae/parallel_vae_interface.py`（10）

```text
ParallelVAEBase.__getattr__
ParallelVAEBase.__init__
ParallelVAEBase._gather_tensor
ParallelVAEBase._replace_module
ParallelVAEBase._split_tensor
ParallelVAEBase.decode
ParallelVAEBase.encode
ParallelVAEFactory._resolve
ParallelVAEFactory.from_vae
ParallelVAEFactory.supports
```

### `tensorrt_llm/_torch/visual_gen/parallel/token_sharded_tp.py`（1）

```text
quantize_fp8_block
```

### `tensorrt_llm/_torch/visual_gen/pipeline.py`（2）

```text
BasePipeline._denoise_step_cfg_parallel
BasePipeline.setup_parallel_vae
```

### `tensorrt_llm/_torch/visual_gen/utils.py`（1）

```text
as_tuple
```

### `tensorrt_llm/_utils.py`（4）

```text
mpi_isend_object
mpi_recv_object
mpi_send_object
torch_pybind11_abi
```

### `tensorrt_llm/commands/serve.py`（1）

```text
set_cuda_device
```

### `tensorrt_llm/evaluate/lm_eval.py`（3）

```text
GPQADiamond.__init__
GSM8KInferenceX.__init__
LmEvalEvaluator.save_results
```

### `tensorrt_llm/evaluate/lm_eval_tasks/gpqa/cot_zeroshot_aa/utils.py`（3）

```text
preprocess
process_docs
process_docs.<locals>._process_doc
```

### `tensorrt_llm/evaluate/longbench_v2.py`（6）

```text
LongBenchV2.__init__
LongBenchV2._extract_answer
LongBenchV2._format_prompt
LongBenchV2._load_and_filter_dataset
LongBenchV2._load_templates
LongBenchV2._post_process
```

### `tensorrt_llm/executor/base_worker.py`（1）

```text
BaseWorker.get_memory_status
```

### `tensorrt_llm/executor/executor.py`（1）

```text
GenerationExecutor._create_rpc_executor
```

### `tensorrt_llm/executor/prompt_tail.py`（2）

```text
PromptTailPolicy.__call__
PromptTailPolicy.__init__
```

### `tensorrt_llm/executor/request.py`（1）

```text
GenerationRequest.prompt_len
```

### `tensorrt_llm/executor/rpc_proxy.py`（6）

```text
GenerationExecutorRpcProxy.__init__
GenerationExecutorRpcProxy._create_mpi_session
GenerationExecutorRpcProxy._setup_mainloop_with_tasks
GenerationExecutorRpcProxy._shutdown_rank_local
GenerationExecutorRpcProxy.launch_workers
GenerationExecutorRpcProxy.setup_engine_remote
```

### `tensorrt_llm/executor/rpc_proxy_mixin.py`（6）

```text
RpcExecutorMixin._fetch_responses_loop_async
RpcExecutorMixin._generic_fetch_loop_async
RpcExecutorMixin.init_rpc_executor
RpcExecutorMixin.setup_mainloop
RpcExecutorMixin.setup_mainloop.<locals>._run_main_loop_task
RpcExecutorMixin.setup_mainloop.<locals>.main_loop_task
```

### `tensorrt_llm/executor/rpc_worker.py`（4）

```text
RpcWorker.__init__
RpcWorker.main_task
RpcWorker.setup_engine
RpcWorker.start
```

### `tensorrt_llm/executor/rpc_worker_mixin.py`（2）

```text
RpcWorkerMixin.fetch_responses_async
RpcWorkerMixin.fetch_responses_loop_async
```

### `tensorrt_llm/executor/utils.py`（1）

```text
create_mpi_comm_session
```

### `tensorrt_llm/functional.py`（1）

```text
MoEAllReduceParams.is_valid
```

### `tensorrt_llm/llmapi/disagg_utils.py`（1）

```text
split_world_comm
```

### `tensorrt_llm/llmapi/llm_args.py`（1）

```text
DeepSeekV4SparseAttentionConfig.needs_separate_short_long_cuda_graphs
```

### `tensorrt_llm/mapping.py`（2）

```text
MappingBase.is_second_last_pp_rank
MpiTopology.moe_ep_group
```

### `tensorrt_llm/runtime/kv_cache_manager_v2/_introspection.py`（1）

```text
probe_host_mem_placement
```

### `tensorrt_llm/serve/scripts/benchmark_dataset.py`（2）

```text
RandomDataset.load_data
batch_tokenize_prompts
```

### `tensorrt_llm/tokenizer/deepseek_v4/tokenizer.py`（1）

```text
DeepseekV4Tokenizer.from_pretrained
```

### `tensorrt_llm/tokenizer/tokenizer.py`（3）

```text
TransformersTokenizer.convert_tokens_to_ids
_fallback_to_fast_tokenizer
_load_tokenizer_config_inherits
```

### `tests/integration/defs/conftest.py`（2）

```text
deepseek_v3_model_root
get_mpi_world_size
```

### `tests/test_common/perf_metrics_utils.py`（1）

```text
get_prometheus_metrics
```

## 附录 B：其他净新增原始函数符号（完整清单）

共 **2,962** 个符号；这些是观测集合差，但未匹配官方静态函数库存，不贡献官方函数 rate。

### `examples/llm-api/llm_inference_distributed.py`（1）

```text
<module>
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/helpers/sparse_indices.py`（4）

```text
__getitem__.<locals>.ifexp_then_block_1
bind_query_row.<locals>.ifexp_else_block_1
bind_query_row.<locals>.ifexp_else_block_3
cache_sparse_source_lengths.<locals>.ifexp_then_block_4
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_2cta/kernel.py`（1）

```text
split_kv_kernel.<locals>.ifexp_then_block_158
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_2cta/resources.py`（6）

```text
init_read_state.<locals>.ifexp_else_block_2
init_read_state.<locals>.ifexp_else_block_4
read_page_offset_window.<locals>.ifexp_else_block_45
read_page_offset_window.<locals>.ifexp_then_block_45
tma_load.<locals>.ifexp_else_block_10
tma_load.<locals>.ifexp_then_block_10
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/parallel_reduction.py`（45）

```text
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.else_block_192
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113.<locals>.else_block_154
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113.<locals>.then_block_154
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113.<locals>.then_block_154.<locals>.if_region_133
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113.<locals>.then_block_154.<locals>.if_region_133.<locals>.else_block_136
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113.<locals>.then_block_154.<locals>.if_region_133.<locals>.then_block_136
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113.<locals>.then_block_154.<locals>.if_region_146
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113.<locals>.then_block_154.<locals>.if_region_146.<locals>.else_block_148
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113.<locals>.then_block_154.<locals>.if_region_146.<locals>.then_block_148
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113.<locals>.then_block_154.<locals>.if_region_149
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113.<locals>.then_block_154.<locals>.if_region_149.<locals>.else_block_153
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113.<locals>.then_block_154.<locals>.if_region_149.<locals>.then_block_153
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113.<locals>.then_block_154.<locals>.ifexp_else_block_138
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_113.<locals>.then_block_154.<locals>.ifexp_then_block_138
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_158
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_158.<locals>.else_block_191
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_158.<locals>.then_block_191
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_158.<locals>.then_block_191.<locals>.if_region_172
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_158.<locals>.then_block_191.<locals>.if_region_172.<locals>.else_block_178
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_112.<locals>.then_block_192.<locals>.if_region_158.<locals>.then_block_191.<locals>.if_region_172.<locals>.then_block_178
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.else_block_71
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_35
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_35.<locals>.else_block_60
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_35.<locals>.then_block_60
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_35.<locals>.then_block_60.<locals>.if_region_46
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_35.<locals>.then_block_60.<locals>.if_region_46.<locals>.else_block_50
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_35.<locals>.then_block_60.<locals>.if_region_46.<locals>.then_block_50
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_35.<locals>.then_block_60.<locals>.ifexp_else_block_53
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_35.<locals>.then_block_60.<locals>.ifexp_then_block_53
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_61
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_61.<locals>.else_block_63
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_61.<locals>.then_block_63
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_66
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_66.<locals>.else_block_70
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_8.<locals>.then_block_71.<locals>.if_region_66.<locals>.then_block_70
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_80
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_80.<locals>.else_block_100
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_80.<locals>.then_block_100
_run_parallel_gmem_reduction_shared_stats.<locals>.if_region_80.<locals>.then_block_100.<locals>.loop_body_87
_run_parallel_gmem_reduction_shared_stats.<locals>.ifexp_else_block_0
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/smem_resources.py`（10）

```text
_producer_load_page_offsets.<locals>.if_region_100.<locals>.then_block_160.<locals>.if_region_149
_producer_load_page_offsets.<locals>.if_region_100.<locals>.then_block_160.<locals>.if_region_149.<locals>.else_block_151
_producer_load_page_offsets.<locals>.if_region_100.<locals>.then_block_160.<locals>.if_region_149.<locals>.then_block_151
read_offsets.<locals>.if_region_68
read_offsets.<locals>.if_region_68.<locals>.else_block_80
read_offsets.<locals>.if_region_68.<locals>.then_block_80
read_offsets.<locals>.if_region_68.<locals>.then_block_80.<locals>.ifexp_else_block_76
read_offsets.<locals>.if_region_68.<locals>.then_block_80.<locals>.ifexp_then_block_76
v_desc_reused
v_desc_reused_0
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_corr.py`（2）

```text
_compute_tail_softmax_scales.<locals>.ifexp_then_block_24
_compute_tail_softmax_scales.<locals>.ifexp_then_block_26
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_o.py`（1）

```text
init_stage_state.<locals>.ifexp_then_block_0
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_p.py`（1）

```text
init_stage_work_tile_state
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_s.py`（7）

```text
qk_mma.<locals>.if_region_20.<locals>.then_block_60.<locals>.ifexp_then_block_50
update_softmax.<locals>.if_region_266
update_softmax.<locals>.if_region_266.<locals>.else_block_274
update_softmax.<locals>.if_region_266.<locals>.then_block_274
update_softmax.<locals>.if_region_266.<locals>.then_block_274.<locals>.if_region_271
update_softmax.<locals>.if_region_266.<locals>.then_block_274.<locals>.if_region_271.<locals>.else_block_273
update_softmax.<locals>.if_region_266.<locals>.then_block_274.<locals>.if_region_271.<locals>.then_block_273
```

### `tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/resources/tmem_softmax_stats.py`（2）

```text
_load_stats_payload.<locals>.ifexp_then_block_0
store_loop_stats.<locals>.if_region_9.<locals>.then_block_15.<locals>.ifexp_then_block_11
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/dspark/markov_chain.py`（107）

```text
<module>
DSparkMarkovChainKernel
__call__
_await_entry.<locals>.while_region_0
_await_entry.<locals>.while_region_0.<locals>.while_after_block_1
_await_entry.<locals>.while_region_0.<locals>.while_before_block_1
_exchange
_exchange.<locals>.if_region_0
_exchange.<locals>.if_region_0.<locals>.then_block_1
_exchange.<locals>.if_region_10
_exchange.<locals>.if_region_10.<locals>.else_block_11
_exchange.<locals>.if_region_10.<locals>.then_block_11
_exchange.<locals>.if_region_12
_exchange.<locals>.if_region_12.<locals>.else_block_17
_exchange.<locals>.if_region_12.<locals>.then_block_17
_exchange.<locals>.if_region_12.<locals>.then_block_17.<locals>.if_region_13
_exchange.<locals>.if_region_12.<locals>.then_block_17.<locals>.if_region_13.<locals>.else_block_14
_exchange.<locals>.if_region_12.<locals>.then_block_17.<locals>.if_region_13.<locals>.then_block_14
_exchange.<locals>.if_region_12.<locals>.then_block_17.<locals>.if_region_15
_exchange.<locals>.if_region_12.<locals>.then_block_17.<locals>.if_region_15.<locals>.else_block_16
_exchange.<locals>.if_region_12.<locals>.then_block_17.<locals>.if_region_15.<locals>.then_block_16
_exchange.<locals>.if_region_2
_exchange.<locals>.if_region_2.<locals>.else_block_9
_exchange.<locals>.if_region_2.<locals>.then_block_9
_exchange.<locals>.if_region_2.<locals>.then_block_9.<locals>.if_region_3
_exchange.<locals>.if_region_2.<locals>.then_block_9.<locals>.if_region_3.<locals>.else_block_4
_exchange.<locals>.if_region_2.<locals>.then_block_9.<locals>.if_region_3.<locals>.then_block_4
_exchange.<locals>.if_region_2.<locals>.then_block_9.<locals>.if_region_5
_exchange.<locals>.if_region_2.<locals>.then_block_9.<locals>.if_region_5.<locals>.else_block_6
_exchange.<locals>.if_region_2.<locals>.then_block_9.<locals>.if_region_5.<locals>.then_block_6
_exchange.<locals>.if_region_2.<locals>.then_block_9.<locals>.if_region_7
_exchange.<locals>.if_region_2.<locals>.then_block_9.<locals>.if_region_7.<locals>.then_block_8
_prefer.<locals>.if_region_0
_prefer.<locals>.if_region_0.<locals>.else_block_3
_prefer.<locals>.if_region_0.<locals>.then_block_3
_prefer.<locals>.if_region_0.<locals>.then_block_3.<locals>.if_region_1
_prefer.<locals>.if_region_0.<locals>.then_block_3.<locals>.if_region_1.<locals>.else_block_2
_prefer.<locals>.if_region_0.<locals>.then_block_3.<locals>.if_region_1.<locals>.then_block_2
_prefer.<locals>.if_region_12
_prefer.<locals>.if_region_12.<locals>.else_block_13
_prefer.<locals>.if_region_12.<locals>.then_block_13
_prefer.<locals>.if_region_4
_prefer.<locals>.if_region_4.<locals>.else_block_11
_prefer.<locals>.if_region_4.<locals>.then_block_11
_prefer.<locals>.if_region_4.<locals>.then_block_11.<locals>.if_region_5
_prefer.<locals>.if_region_4.<locals>.then_block_11.<locals>.if_region_5.<locals>.else_block_10
_prefer.<locals>.if_region_4.<locals>.then_block_11.<locals>.if_region_5.<locals>.then_block_10
_prefer.<locals>.if_region_4.<locals>.then_block_11.<locals>.if_region_5.<locals>.then_block_10.<locals>.if_region_6
_prefer.<locals>.if_region_4.<locals>.then_block_11.<locals>.if_region_5.<locals>.then_block_10.<locals>.if_region_6.<locals>.else_block_7
_prefer.<locals>.if_region_4.<locals>.then_block_11.<locals>.if_region_5.<locals>.then_block_10.<locals>.if_region_6.<locals>.then_block_7
_prefer.<locals>.if_region_4.<locals>.then_block_11.<locals>.if_region_5.<locals>.then_block_10.<locals>.if_region_8
_prefer.<locals>.if_region_4.<locals>.then_block_11.<locals>.if_region_5.<locals>.then_block_10.<locals>.if_region_8.<locals>.else_block_9
_prefer.<locals>.if_region_4.<locals>.then_block_11.<locals>.if_region_5.<locals>.then_block_10.<locals>.if_region_8.<locals>.then_block_9
kernel
kernel.<locals>.if_region_0
kernel.<locals>.if_region_0.<locals>.else_block_1
kernel.<locals>.if_region_0.<locals>.then_block_1
kernel.<locals>.if_region_10
kernel.<locals>.if_region_10.<locals>.else_block_11
kernel.<locals>.if_region_10.<locals>.then_block_11
kernel.<locals>.if_region_12
kernel.<locals>.if_region_12.<locals>.else_block_13
kernel.<locals>.if_region_12.<locals>.then_block_13
kernel.<locals>.if_region_14
kernel.<locals>.if_region_14.<locals>.then_block_15
kernel.<locals>.if_region_18
kernel.<locals>.if_region_18.<locals>.else_block_25
kernel.<locals>.if_region_18.<locals>.then_block_25
kernel.<locals>.if_region_18.<locals>.then_block_25.<locals>.if_region_19
kernel.<locals>.if_region_18.<locals>.then_block_25.<locals>.if_region_19.<locals>.else_block_20
kernel.<locals>.if_region_18.<locals>.then_block_25.<locals>.if_region_19.<locals>.then_block_20
kernel.<locals>.if_region_18.<locals>.then_block_25.<locals>.if_region_21
kernel.<locals>.if_region_18.<locals>.then_block_25.<locals>.if_region_21.<locals>.then_block_24
kernel.<locals>.if_region_18.<locals>.then_block_25.<locals>.if_region_21.<locals>.then_block_24.<locals>.if_region_22
kernel.<locals>.if_region_18.<locals>.then_block_25.<locals>.if_region_21.<locals>.then_block_24.<locals>.if_region_22.<locals>.then_block_23
kernel.<locals>.if_region_2
kernel.<locals>.if_region_2.<locals>.then_block_5
kernel.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3
kernel.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3.<locals>.else_block_4
kernel.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3.<locals>.then_block_4
kernel.<locals>.if_region_26
kernel.<locals>.if_region_26.<locals>.else_block_29
kernel.<locals>.if_region_26.<locals>.then_block_29
kernel.<locals>.if_region_26.<locals>.then_block_29.<locals>.if_region_27
kernel.<locals>.if_region_26.<locals>.then_block_29.<locals>.if_region_27.<locals>.else_block_28
kernel.<locals>.if_region_26.<locals>.then_block_29.<locals>.if_region_27.<locals>.then_block_28
kernel.<locals>.if_region_30
kernel.<locals>.if_region_30.<locals>.else_block_32
kernel.<locals>.if_region_30.<locals>.then_block_32
kernel.<locals>.if_region_30.<locals>.then_block_32.<locals>.loop_body_31
kernel.<locals>.if_region_33
kernel.<locals>.if_region_33.<locals>.else_block_34
kernel.<locals>.if_region_33.<locals>.then_block_34
kernel.<locals>.if_region_35
kernel.<locals>.if_region_35.<locals>.then_block_40
kernel.<locals>.if_region_35.<locals>.then_block_40.<locals>.if_region_36
kernel.<locals>.if_region_35.<locals>.then_block_40.<locals>.if_region_36.<locals>.then_block_39
kernel.<locals>.if_region_35.<locals>.then_block_40.<locals>.if_region_36.<locals>.then_block_39.<locals>.while_region_37
kernel.<locals>.if_region_35.<locals>.then_block_40.<locals>.if_region_36.<locals>.then_block_39.<locals>.while_region_37.<locals>.while_after_block_38
kernel.<locals>.if_region_35.<locals>.then_block_40.<locals>.if_region_36.<locals>.then_block_39.<locals>.while_region_37.<locals>.while_before_block_38
kernel.<locals>.if_region_6
kernel.<locals>.if_region_6.<locals>.then_block_9
kernel.<locals>.if_region_6.<locals>.then_block_9.<locals>.if_region_7
kernel.<locals>.if_region_6.<locals>.then_block_9.<locals>.if_region_7.<locals>.then_block_8
kernel.<locals>.while_region_16
kernel.<locals>.while_region_16.<locals>.while_after_block_17
kernel.<locals>.while_region_16.<locals>.while_before_block_17
```

### `tensorrt_llm/_torch/attention/kernels/blackwell/gqa_kv_replica_partial_attn.py`（7）

```text
_ll_store
_ll_store.<locals>.if_region_0
_ll_store.<locals>.if_region_0.<locals>.then_block_1
kernel.<locals>.if_region_73
kernel.<locals>.if_region_73.<locals>.then_block_74
kernel_two.<locals>.if_region_105
kernel_two.<locals>.if_region_105.<locals>.then_block_106
```

### `tensorrt_llm/_torch/distributed/tp2_allreduce.py`（2）

```text
<module>
Tp2AllReduceNorm
```

### `tensorrt_llm/_torch/kernels/blackwell/engram/lookup.py`（4）

```text
store_packed
store_packed.<locals>.if_region_0
store_packed.<locals>.if_region_0.<locals>.else_block_1
store_packed.<locals>.if_region_0.<locals>.then_block_1
```

### `tensorrt_llm/_torch/kernels/blackwell/engram/packed_projection.py`（74）

```text
__call__
__call__.<locals>.SharedStorage
kernel
kernel.<locals>.if_region_0
kernel.<locals>.if_region_0.<locals>.then_block_1
kernel.<locals>.if_region_15
kernel.<locals>.if_region_15.<locals>.else_block_32
kernel.<locals>.if_region_15.<locals>.then_block_32
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_18
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_18.<locals>.else_block_19
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_18.<locals>.then_block_19
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_20
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_20.<locals>.else_block_21
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_20.<locals>.then_block_21
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_30
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_30.<locals>.else_block_31
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_30.<locals>.then_block_31
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.loop_body_22
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.loop_body_22.<locals>.if_region_23
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.loop_body_22.<locals>.if_region_23.<locals>.else_block_25
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.loop_body_22.<locals>.if_region_23.<locals>.then_block_25
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.loop_body_22.<locals>.if_region_23.<locals>.then_block_25.<locals>.loop_body_24
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.loop_body_22.<locals>.if_region_26
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.loop_body_22.<locals>.if_region_26.<locals>.else_block_29
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.loop_body_22.<locals>.if_region_26.<locals>.then_block_29
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.loop_body_22.<locals>.if_region_26.<locals>.then_block_29.<locals>.if_region_27
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.loop_body_22.<locals>.if_region_26.<locals>.then_block_29.<locals>.if_region_27.<locals>.else_block_28
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.loop_body_22.<locals>.if_region_26.<locals>.then_block_29.<locals>.if_region_27.<locals>.then_block_28
kernel.<locals>.if_region_15.<locals>.then_block_32.<locals>.while_region_16.<locals>.while_before_block_17
kernel.<locals>.if_region_2
kernel.<locals>.if_region_2.<locals>.then_block_3
kernel.<locals>.if_region_33
kernel.<locals>.if_region_33.<locals>.else_block_50
kernel.<locals>.if_region_33.<locals>.then_block_50
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.if_region_34
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.if_region_34.<locals>.then_block_35
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.if_region_46
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.if_region_46.<locals>.then_block_47
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.if_region_48
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.if_region_48.<locals>.then_block_49
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_after_block_37
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_after_block_37.<locals>.ifexp_else_block_38
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_after_block_37.<locals>.ifexp_then_block_38
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_after_block_37.<locals>.loop_body_39
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_after_block_37.<locals>.loop_body_39.<locals>.if_region_40
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_after_block_37.<locals>.loop_body_39.<locals>.if_region_40.<locals>.else_block_41
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_after_block_37.<locals>.loop_body_39.<locals>.if_region_40.<locals>.then_block_41
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_after_block_37.<locals>.loop_body_39.<locals>.if_region_42
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_after_block_37.<locals>.loop_body_39.<locals>.if_region_42.<locals>.else_block_43
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_after_block_37.<locals>.loop_body_39.<locals>.if_region_42.<locals>.then_block_43
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_after_block_37.<locals>.loop_body_39.<locals>.if_region_44
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_after_block_37.<locals>.loop_body_39.<locals>.if_region_44.<locals>.else_block_45
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_after_block_37.<locals>.loop_body_39.<locals>.if_region_44.<locals>.then_block_45
kernel.<locals>.if_region_33.<locals>.then_block_50.<locals>.while_region_36.<locals>.while_before_block_37
kernel.<locals>.if_region_4
kernel.<locals>.if_region_4.<locals>.else_block_14
kernel.<locals>.if_region_4.<locals>.then_block_14
kernel.<locals>.if_region_4.<locals>.then_block_14.<locals>.while_region_5
kernel.<locals>.if_region_4.<locals>.then_block_14.<locals>.while_region_5.<locals>.while_after_block_6
kernel.<locals>.if_region_4.<locals>.then_block_14.<locals>.while_region_5.<locals>.while_after_block_6.<locals>.if_region_7
kernel.<locals>.if_region_4.<locals>.then_block_14.<locals>.while_region_5.<locals>.while_after_block_6.<locals>.if_region_7.<locals>.else_block_8
kernel.<locals>.if_region_4.<locals>.then_block_14.<locals>.while_region_5.<locals>.while_after_block_6.<locals>.if_region_7.<locals>.then_block_8
kernel.<locals>.if_region_4.<locals>.then_block_14.<locals>.while_region_5.<locals>.while_after_block_6.<locals>.loop_body_9
kernel.<locals>.if_region_4.<locals>.then_block_14.<locals>.while_region_5.<locals>.while_after_block_6.<locals>.loop_body_9.<locals>.if_region_10
kernel.<locals>.if_region_4.<locals>.then_block_14.<locals>.while_region_5.<locals>.while_after_block_6.<locals>.loop_body_9.<locals>.if_region_10.<locals>.else_block_11
kernel.<locals>.if_region_4.<locals>.then_block_14.<locals>.while_region_5.<locals>.while_after_block_6.<locals>.loop_body_9.<locals>.if_region_10.<locals>.then_block_11
kernel.<locals>.if_region_4.<locals>.then_block_14.<locals>.while_region_5.<locals>.while_after_block_6.<locals>.loop_body_9.<locals>.if_region_12
kernel.<locals>.if_region_4.<locals>.then_block_14.<locals>.while_region_5.<locals>.while_after_block_6.<locals>.loop_body_9.<locals>.if_region_12.<locals>.else_block_13
kernel.<locals>.if_region_4.<locals>.then_block_14.<locals>.while_region_5.<locals>.while_after_block_6.<locals>.loop_body_9.<locals>.if_region_12.<locals>.then_block_13
kernel.<locals>.if_region_4.<locals>.then_block_14.<locals>.while_region_5.<locals>.while_before_block_6
wrapper
```

### `tensorrt_llm/_torch/kernels/blackwell/fp8_block_scale_low_m_gemm.py`（3）

```text
mma_warp.<locals>.loop_body_2.<locals>.if_region_3
mma_warp.<locals>.loop_body_2.<locals>.if_region_3.<locals>.else_block_4
mma_warp.<locals>.loop_body_2.<locals>.if_region_3.<locals>.then_block_4
```

### `tensorrt_llm/_torch/kernels/blackwell/gemm_allreduce_norm.py`（61）

```text
<module>
GemmAllReduceNormKernel
__call__
kernel
kernel.<locals>.if_region_0
kernel.<locals>.if_region_0.<locals>.else_block_1
kernel.<locals>.if_region_0.<locals>.then_block_1
kernel.<locals>.if_region_12
kernel.<locals>.if_region_12.<locals>.else_block_13
kernel.<locals>.if_region_12.<locals>.then_block_13
kernel.<locals>.if_region_14
kernel.<locals>.if_region_14.<locals>.then_block_20
kernel.<locals>.if_region_14.<locals>.then_block_20.<locals>.ifexp_else_block_15
kernel.<locals>.if_region_14.<locals>.then_block_20.<locals>.ifexp_else_block_16
kernel.<locals>.if_region_14.<locals>.then_block_20.<locals>.ifexp_else_block_17
kernel.<locals>.if_region_14.<locals>.then_block_20.<locals>.ifexp_then_block_15
kernel.<locals>.if_region_14.<locals>.then_block_20.<locals>.ifexp_then_block_16
kernel.<locals>.if_region_14.<locals>.then_block_20.<locals>.ifexp_then_block_17
kernel.<locals>.if_region_14.<locals>.then_block_20.<locals>.while_region_18
kernel.<locals>.if_region_14.<locals>.then_block_20.<locals>.while_region_18.<locals>.while_after_block_19
kernel.<locals>.if_region_14.<locals>.then_block_20.<locals>.while_region_18.<locals>.while_before_block_19
kernel.<locals>.if_region_2
kernel.<locals>.if_region_2.<locals>.else_block_3
kernel.<locals>.if_region_2.<locals>.then_block_3
kernel.<locals>.if_region_21
kernel.<locals>.if_region_21.<locals>.else_block_36
kernel.<locals>.if_region_21.<locals>.then_block_36
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.if_region_32
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.if_region_32.<locals>.else_block_33
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.if_region_32.<locals>.then_block_33
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.if_region_34
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.if_region_34.<locals>.else_block_35
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.if_region_34.<locals>.then_block_35
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_after_block_23
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_after_block_23.<locals>.if_region_24
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_after_block_23.<locals>.if_region_24.<locals>.else_block_25
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_after_block_23.<locals>.if_region_24.<locals>.then_block_25
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_after_block_23.<locals>.if_region_26
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_after_block_23.<locals>.if_region_26.<locals>.else_block_27
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_after_block_23.<locals>.if_region_26.<locals>.then_block_27
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_after_block_23.<locals>.if_region_28
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_after_block_23.<locals>.if_region_28.<locals>.else_block_29
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_after_block_23.<locals>.if_region_28.<locals>.then_block_29
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_after_block_23.<locals>.if_region_30
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_after_block_23.<locals>.if_region_30.<locals>.else_block_31
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_after_block_23.<locals>.if_region_30.<locals>.then_block_31
kernel.<locals>.if_region_21.<locals>.then_block_36.<locals>.while_region_22.<locals>.while_before_block_23
kernel.<locals>.if_region_37
kernel.<locals>.if_region_37.<locals>.then_block_40
kernel.<locals>.if_region_37.<locals>.then_block_40.<locals>.if_region_38
kernel.<locals>.if_region_37.<locals>.then_block_40.<locals>.if_region_38.<locals>.then_block_39
kernel.<locals>.if_region_4
kernel.<locals>.if_region_4.<locals>.else_block_5
kernel.<locals>.if_region_4.<locals>.then_block_5
kernel.<locals>.ifexp_else_block_6
kernel.<locals>.ifexp_else_block_7
kernel.<locals>.ifexp_else_block_8
kernel.<locals>.ifexp_else_block_9
kernel.<locals>.loop_body_10
kernel.<locals>.loop_body_10.<locals>.ifexp_else_block_11
```

### `tensorrt_llm/_torch/kernels/blackwell/gemm_twoshot_allreduce_norm.py`（107）

```text
<module>
_clear_dirty.<locals>.if_region_0
_clear_dirty.<locals>.if_region_0.<locals>.else_block_3
_clear_dirty.<locals>.if_region_0.<locals>.then_block_3
_clear_dirty.<locals>.if_region_0.<locals>.then_block_3.<locals>.while_region_1
_clear_dirty.<locals>.if_region_0.<locals>.then_block_3.<locals>.while_region_1.<locals>.while_after_block_2
_clear_dirty.<locals>.if_region_0.<locals>.then_block_3.<locals>.while_region_1.<locals>.while_before_block_2
_kernel.<locals>.if_region_11
_kernel.<locals>.if_region_11.<locals>.else_block_14
_kernel.<locals>.if_region_11.<locals>.then_block_14
_kernel.<locals>.if_region_11.<locals>.then_block_14.<locals>.if_region_12
_kernel.<locals>.if_region_11.<locals>.then_block_14.<locals>.if_region_12.<locals>.else_block_13
_kernel.<locals>.if_region_11.<locals>.then_block_14.<locals>.if_region_12.<locals>.then_block_13
_kernel.<locals>.if_region_15
_kernel.<locals>.if_region_15.<locals>.else_block_22
_kernel.<locals>.if_region_15.<locals>.then_block_22
_kernel.<locals>.if_region_15.<locals>.then_block_22.<locals>.if_region_18
_kernel.<locals>.if_region_15.<locals>.then_block_22.<locals>.if_region_18.<locals>.else_block_19
_kernel.<locals>.if_region_15.<locals>.then_block_22.<locals>.if_region_18.<locals>.then_block_19
_kernel.<locals>.if_region_15.<locals>.then_block_22.<locals>.if_region_20
_kernel.<locals>.if_region_15.<locals>.then_block_22.<locals>.if_region_20.<locals>.then_block_21
_kernel.<locals>.if_region_15.<locals>.then_block_22.<locals>.while_region_16
_kernel.<locals>.if_region_15.<locals>.then_block_22.<locals>.while_region_16.<locals>.while_after_block_17
_kernel.<locals>.if_region_15.<locals>.then_block_22.<locals>.while_region_16.<locals>.while_before_block_17
_kernel.<locals>.if_region_23
_kernel.<locals>.if_region_23.<locals>.else_block_30
_kernel.<locals>.if_region_23.<locals>.then_block_30
_kernel.<locals>.if_region_23.<locals>.then_block_30.<locals>.if_region_24
_kernel.<locals>.if_region_23.<locals>.then_block_30.<locals>.if_region_24.<locals>.else_block_25
_kernel.<locals>.if_region_23.<locals>.then_block_30.<locals>.if_region_24.<locals>.then_block_25
_kernel.<locals>.if_region_23.<locals>.then_block_30.<locals>.if_region_26
_kernel.<locals>.if_region_23.<locals>.then_block_30.<locals>.if_region_26.<locals>.else_block_27
_kernel.<locals>.if_region_23.<locals>.then_block_30.<locals>.if_region_26.<locals>.then_block_27
_kernel.<locals>.if_region_23.<locals>.then_block_30.<locals>.while_region_28
_kernel.<locals>.if_region_23.<locals>.then_block_30.<locals>.while_region_28.<locals>.while_after_block_29
_kernel.<locals>.if_region_23.<locals>.then_block_30.<locals>.while_region_28.<locals>.while_before_block_29
_kernel.<locals>.if_region_3
_kernel.<locals>.if_region_3.<locals>.else_block_6
_kernel.<locals>.if_region_3.<locals>.then_block_6
_kernel.<locals>.if_region_3.<locals>.then_block_6.<locals>.if_region_4
_kernel.<locals>.if_region_3.<locals>.then_block_6.<locals>.if_region_4.<locals>.else_block_5
_kernel.<locals>.if_region_3.<locals>.then_block_6.<locals>.if_region_4.<locals>.then_block_5
_kernel.<locals>.if_region_31
_kernel.<locals>.if_region_31.<locals>.then_block_32
_kernel.<locals>.if_region_33
_kernel.<locals>.if_region_33.<locals>.then_block_34
_kernel.<locals>.if_region_35
_kernel.<locals>.if_region_35.<locals>.else_block_38
_kernel.<locals>.if_region_35.<locals>.then_block_38
_kernel.<locals>.if_region_35.<locals>.then_block_38.<locals>.if_region_36
_kernel.<locals>.if_region_35.<locals>.then_block_38.<locals>.if_region_36.<locals>.else_block_37
_kernel.<locals>.if_region_35.<locals>.then_block_38.<locals>.if_region_36.<locals>.then_block_37
_kernel.<locals>.if_region_39
_kernel.<locals>.if_region_39.<locals>.then_block_40
_kernel.<locals>.if_region_41
_kernel.<locals>.if_region_41.<locals>.then_block_56
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_after_block_44
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_after_block_44.<locals>.if_region_45
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_after_block_44.<locals>.if_region_45.<locals>.else_block_46
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_after_block_44.<locals>.if_region_45.<locals>.then_block_46
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_after_block_44.<locals>.if_region_47
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_after_block_44.<locals>.if_region_47.<locals>.else_block_48
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_after_block_44.<locals>.if_region_47.<locals>.then_block_48
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_after_block_44.<locals>.if_region_49
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_after_block_44.<locals>.if_region_49.<locals>.else_block_50
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_after_block_44.<locals>.if_region_49.<locals>.then_block_50
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_after_block_44.<locals>.if_region_51
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_after_block_44.<locals>.if_region_51.<locals>.else_block_52
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_after_block_44.<locals>.if_region_51.<locals>.then_block_52
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_42.<locals>.then_block_53.<locals>.while_region_43.<locals>.while_before_block_44
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_54
_kernel.<locals>.if_region_41.<locals>.then_block_56.<locals>.if_region_54.<locals>.then_block_55
_kernel.<locals>.if_region_57
_kernel.<locals>.if_region_57.<locals>.else_block_74
_kernel.<locals>.if_region_57.<locals>.then_block_74
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.if_region_68
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.if_region_68.<locals>.else_block_69
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.if_region_68.<locals>.then_block_69
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.if_region_70
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.if_region_70.<locals>.then_block_73
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.if_region_70.<locals>.then_block_73.<locals>.if_region_71
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.if_region_70.<locals>.then_block_73.<locals>.if_region_71.<locals>.then_block_72
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_after_block_59
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_after_block_59.<locals>.if_region_60
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_after_block_59.<locals>.if_region_60.<locals>.else_block_61
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_after_block_59.<locals>.if_region_60.<locals>.then_block_61
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_after_block_59.<locals>.if_region_62
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_after_block_59.<locals>.if_region_62.<locals>.else_block_63
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_after_block_59.<locals>.if_region_62.<locals>.then_block_63
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_after_block_59.<locals>.if_region_64
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_after_block_59.<locals>.if_region_64.<locals>.else_block_65
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_after_block_59.<locals>.if_region_64.<locals>.then_block_65
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_after_block_59.<locals>.if_region_66
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_after_block_59.<locals>.if_region_66.<locals>.else_block_67
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_after_block_59.<locals>.if_region_66.<locals>.then_block_67
_kernel.<locals>.if_region_57.<locals>.then_block_74.<locals>.while_region_58.<locals>.while_before_block_59
_kernel.<locals>.if_region_7
_kernel.<locals>.if_region_7.<locals>.then_block_8
_kernel.<locals>.if_region_9
_kernel.<locals>.if_region_9.<locals>.else_block_10
_kernel.<locals>.if_region_9.<locals>.then_block_10
_kernel.<locals>.ifexp_else_block_0
_launch.<locals>.ifexp_then_block_0
```

### `tensorrt_llm/_torch/kernels/blackwell/mhc/mhc_boundary.py`（34）

```text
kernel.<locals>.if_region_0.<locals>.else_block_1.<locals>.elif_region_1
kernel.<locals>.if_region_0.<locals>.else_block_1.<locals>.elif_region_1.<locals>.else_block_4
kernel.<locals>.if_region_0.<locals>.else_block_1.<locals>.elif_region_1.<locals>.then_block_4
kernel.<locals>.if_region_0.<locals>.else_block_1.<locals>.elif_region_1.<locals>.then_block_4.<locals>.if_region_2
kernel.<locals>.if_region_0.<locals>.else_block_1.<locals>.elif_region_1.<locals>.then_block_4.<locals>.if_region_2.<locals>.else_block_3
kernel.<locals>.if_region_0.<locals>.else_block_1.<locals>.elif_region_1.<locals>.then_block_4.<locals>.if_region_2.<locals>.then_block_3
kernel.<locals>.if_region_5
kernel.<locals>.if_region_5.<locals>.else_block_6
kernel.<locals>.if_region_5.<locals>.then_block_6
kernel.<locals>.ifexp_else_block_10
kernel.<locals>.ifexp_else_block_8
kernel.<locals>.ifexp_else_block_9
kernel.<locals>.loop_body_11
kernel.<locals>.loop_body_11.<locals>.if_region_13
kernel.<locals>.loop_body_11.<locals>.if_region_13.<locals>.else_block_14
kernel.<locals>.loop_body_11.<locals>.if_region_13.<locals>.then_block_14
kernel.<locals>.loop_body_11.<locals>.if_region_15
kernel.<locals>.loop_body_11.<locals>.if_region_15.<locals>.else_block_16
kernel.<locals>.loop_body_11.<locals>.if_region_15.<locals>.then_block_16
kernel.<locals>.loop_body_11.<locals>.if_region_17
kernel.<locals>.loop_body_11.<locals>.if_region_17.<locals>.else_block_18
kernel.<locals>.loop_body_11.<locals>.if_region_17.<locals>.then_block_18
kernel.<locals>.loop_body_11.<locals>.if_region_19
kernel.<locals>.loop_body_11.<locals>.if_region_19.<locals>.else_block_20
kernel.<locals>.loop_body_11.<locals>.if_region_19.<locals>.then_block_20
kernel.<locals>.loop_body_11.<locals>.if_region_21
kernel.<locals>.loop_body_11.<locals>.if_region_21.<locals>.else_block_26
kernel.<locals>.loop_body_11.<locals>.if_region_21.<locals>.then_block_26
kernel.<locals>.loop_body_11.<locals>.if_region_21.<locals>.then_block_26.<locals>.if_region_22
kernel.<locals>.loop_body_11.<locals>.if_region_21.<locals>.then_block_26.<locals>.if_region_22.<locals>.then_block_23
kernel.<locals>.loop_body_11.<locals>.if_region_21.<locals>.then_block_26.<locals>.if_region_24
kernel.<locals>.loop_body_11.<locals>.if_region_21.<locals>.then_block_26.<locals>.if_region_24.<locals>.else_block_25
kernel.<locals>.loop_body_11.<locals>.if_region_21.<locals>.then_block_26.<locals>.if_region_24.<locals>.then_block_25
kernel.<locals>.loop_body_11.<locals>.ifexp_else_block_12
```

### `tensorrt_llm/_torch/kernels/blackwell/mhc/mhc_split_boundary.py`（67）

```text
__call__.<locals>.ifexp_else_block_0
__call__.<locals>.ifexp_then_block_0
_collapse_norm.<locals>.if_region_2
_collapse_norm.<locals>.if_region_2.<locals>.else_block_3
_collapse_norm.<locals>.if_region_2.<locals>.then_block_3
ar_kernel
ar_kernel.<locals>.if_region_0
ar_kernel.<locals>.if_region_0.<locals>.else_block_3
ar_kernel.<locals>.if_region_0.<locals>.then_block_3
ar_kernel.<locals>.if_region_0.<locals>.then_block_3.<locals>.if_region_1
ar_kernel.<locals>.if_region_0.<locals>.then_block_3.<locals>.if_region_1.<locals>.then_block_2
ar_kernel.<locals>.if_region_10
ar_kernel.<locals>.if_region_10.<locals>.else_block_11
ar_kernel.<locals>.if_region_10.<locals>.then_block_11
ar_kernel.<locals>.if_region_12
ar_kernel.<locals>.if_region_12.<locals>.else_block_13
ar_kernel.<locals>.if_region_12.<locals>.then_block_13
ar_kernel.<locals>.if_region_14
ar_kernel.<locals>.if_region_14.<locals>.else_block_15
ar_kernel.<locals>.if_region_14.<locals>.then_block_15
ar_kernel.<locals>.if_region_16
ar_kernel.<locals>.if_region_16.<locals>.else_block_17
ar_kernel.<locals>.if_region_16.<locals>.then_block_17
ar_kernel.<locals>.if_region_18
ar_kernel.<locals>.if_region_18.<locals>.else_block_19
ar_kernel.<locals>.if_region_18.<locals>.then_block_19
ar_kernel.<locals>.if_region_20
ar_kernel.<locals>.if_region_20.<locals>.then_block_23
ar_kernel.<locals>.if_region_20.<locals>.then_block_23.<locals>.while_region_21
ar_kernel.<locals>.if_region_20.<locals>.then_block_23.<locals>.while_region_21.<locals>.while_after_block_22
ar_kernel.<locals>.if_region_20.<locals>.then_block_23.<locals>.while_region_21.<locals>.while_before_block_22
ar_kernel.<locals>.if_region_36
ar_kernel.<locals>.if_region_36.<locals>.then_block_43
ar_kernel.<locals>.if_region_36.<locals>.then_block_43.<locals>.if_region_37
ar_kernel.<locals>.if_region_36.<locals>.then_block_43.<locals>.if_region_37.<locals>.then_block_42
ar_kernel.<locals>.if_region_36.<locals>.then_block_43.<locals>.if_region_37.<locals>.then_block_42.<locals>.while_region_38
ar_kernel.<locals>.if_region_36.<locals>.then_block_43.<locals>.if_region_37.<locals>.then_block_42.<locals>.while_region_38.<locals>.while_after_block_39
ar_kernel.<locals>.if_region_36.<locals>.then_block_43.<locals>.if_region_37.<locals>.then_block_42.<locals>.while_region_38.<locals>.while_before_block_39
ar_kernel.<locals>.if_region_36.<locals>.then_block_43.<locals>.if_region_37.<locals>.then_block_42.<locals>.while_region_40
ar_kernel.<locals>.if_region_36.<locals>.then_block_43.<locals>.if_region_37.<locals>.then_block_42.<locals>.while_region_40.<locals>.while_after_block_41
ar_kernel.<locals>.if_region_36.<locals>.then_block_43.<locals>.if_region_37.<locals>.then_block_42.<locals>.while_region_40.<locals>.while_before_block_41
ar_kernel.<locals>.if_region_4
ar_kernel.<locals>.if_region_4.<locals>.else_block_7
ar_kernel.<locals>.if_region_4.<locals>.then_block_7
ar_kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.if_region_5
ar_kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.if_region_5.<locals>.then_block_6
ar_kernel.<locals>.if_region_8
ar_kernel.<locals>.if_region_8.<locals>.else_block_9
ar_kernel.<locals>.if_region_8.<locals>.then_block_9
ar_kernel.<locals>.while_region_24
ar_kernel.<locals>.while_region_24.<locals>.while_after_block_25
ar_kernel.<locals>.while_region_24.<locals>.while_before_block_25
ar_kernel.<locals>.while_region_26
ar_kernel.<locals>.while_region_26.<locals>.while_after_block_27
ar_kernel.<locals>.while_region_26.<locals>.while_after_block_27.<locals>.if_region_28
ar_kernel.<locals>.while_region_26.<locals>.while_after_block_27.<locals>.if_region_28.<locals>.else_block_29
ar_kernel.<locals>.while_region_26.<locals>.while_after_block_27.<locals>.if_region_28.<locals>.then_block_29
ar_kernel.<locals>.while_region_26.<locals>.while_after_block_27.<locals>.if_region_30
ar_kernel.<locals>.while_region_26.<locals>.while_after_block_27.<locals>.if_region_30.<locals>.else_block_31
ar_kernel.<locals>.while_region_26.<locals>.while_after_block_27.<locals>.if_region_30.<locals>.then_block_31
ar_kernel.<locals>.while_region_26.<locals>.while_after_block_27.<locals>.if_region_32
ar_kernel.<locals>.while_region_26.<locals>.while_after_block_27.<locals>.if_region_32.<locals>.else_block_33
ar_kernel.<locals>.while_region_26.<locals>.while_after_block_27.<locals>.if_region_32.<locals>.then_block_33
ar_kernel.<locals>.while_region_26.<locals>.while_after_block_27.<locals>.if_region_34
ar_kernel.<locals>.while_region_26.<locals>.while_after_block_27.<locals>.if_region_34.<locals>.else_block_35
ar_kernel.<locals>.while_region_26.<locals>.while_after_block_27.<locals>.if_region_34.<locals>.then_block_35
ar_kernel.<locals>.while_region_26.<locals>.while_before_block_27
```

### `tensorrt_llm/_torch/kernels/blackwell/mnnvl_allreduce_norm.py`（129）

```text
__call__
kernel
kernel.<locals>.if_region_0
kernel.<locals>.if_region_0.<locals>.else_block_1
kernel.<locals>.if_region_0.<locals>.then_block_1
kernel.<locals>.if_region_13
kernel.<locals>.if_region_13.<locals>.else_block_24
kernel.<locals>.if_region_13.<locals>.then_block_24
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_after_block_15
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_after_block_15.<locals>.if_region_16
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_after_block_15.<locals>.if_region_16.<locals>.else_block_17
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_after_block_15.<locals>.if_region_16.<locals>.then_block_17
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_after_block_15.<locals>.if_region_18
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_after_block_15.<locals>.if_region_18.<locals>.else_block_19
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_after_block_15.<locals>.if_region_18.<locals>.then_block_19
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_after_block_15.<locals>.if_region_20
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_after_block_15.<locals>.if_region_20.<locals>.else_block_21
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_after_block_15.<locals>.if_region_20.<locals>.then_block_21
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_after_block_15.<locals>.if_region_22
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_after_block_15.<locals>.if_region_22.<locals>.else_block_23
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_after_block_15.<locals>.if_region_22.<locals>.then_block_23
kernel.<locals>.if_region_13.<locals>.then_block_24.<locals>.while_region_14.<locals>.while_before_block_15
kernel.<locals>.if_region_2
kernel.<locals>.if_region_2.<locals>.then_block_3
kernel.<locals>.if_region_35
kernel.<locals>.if_region_35.<locals>.then_block_36
kernel.<locals>.if_region_37
kernel.<locals>.if_region_37.<locals>.else_block_38
kernel.<locals>.if_region_37.<locals>.then_block_38
kernel.<locals>.if_region_41
kernel.<locals>.if_region_41.<locals>.then_block_46
kernel.<locals>.if_region_41.<locals>.then_block_46.<locals>.if_region_42
kernel.<locals>.if_region_41.<locals>.then_block_46.<locals>.if_region_42.<locals>.then_block_45
kernel.<locals>.if_region_41.<locals>.then_block_46.<locals>.if_region_42.<locals>.then_block_45.<locals>.while_region_43
kernel.<locals>.if_region_41.<locals>.then_block_46.<locals>.if_region_42.<locals>.then_block_45.<locals>.while_region_43.<locals>.while_after_block_44
kernel.<locals>.if_region_41.<locals>.then_block_46.<locals>.if_region_42.<locals>.then_block_45.<locals>.while_region_43.<locals>.while_before_block_44
kernel.<locals>.if_region_6
kernel.<locals>.if_region_6.<locals>.then_block_12
kernel.<locals>.if_region_6.<locals>.then_block_12.<locals>.ifexp_else_block_7
kernel.<locals>.if_region_6.<locals>.then_block_12.<locals>.ifexp_else_block_8
kernel.<locals>.if_region_6.<locals>.then_block_12.<locals>.ifexp_else_block_9
kernel.<locals>.if_region_6.<locals>.then_block_12.<locals>.ifexp_then_block_7
kernel.<locals>.if_region_6.<locals>.then_block_12.<locals>.ifexp_then_block_8
kernel.<locals>.if_region_6.<locals>.then_block_12.<locals>.ifexp_then_block_9
kernel.<locals>.if_region_6.<locals>.then_block_12.<locals>.while_region_10
kernel.<locals>.if_region_6.<locals>.then_block_12.<locals>.while_region_10.<locals>.while_after_block_11
kernel.<locals>.if_region_6.<locals>.then_block_12.<locals>.while_region_10.<locals>.while_before_block_11
kernel.<locals>.while_region_25
kernel.<locals>.while_region_25.<locals>.while_after_block_26
kernel.<locals>.while_region_25.<locals>.while_after_block_26.<locals>.if_region_27
kernel.<locals>.while_region_25.<locals>.while_after_block_26.<locals>.if_region_27.<locals>.else_block_28
kernel.<locals>.while_region_25.<locals>.while_after_block_26.<locals>.if_region_27.<locals>.then_block_28
kernel.<locals>.while_region_25.<locals>.while_after_block_26.<locals>.if_region_29
kernel.<locals>.while_region_25.<locals>.while_after_block_26.<locals>.if_region_29.<locals>.else_block_30
kernel.<locals>.while_region_25.<locals>.while_after_block_26.<locals>.if_region_29.<locals>.then_block_30
kernel.<locals>.while_region_25.<locals>.while_after_block_26.<locals>.if_region_31
kernel.<locals>.while_region_25.<locals>.while_after_block_26.<locals>.if_region_31.<locals>.else_block_32
kernel.<locals>.while_region_25.<locals>.while_after_block_26.<locals>.if_region_31.<locals>.then_block_32
kernel.<locals>.while_region_25.<locals>.while_after_block_26.<locals>.if_region_33
kernel.<locals>.while_region_25.<locals>.while_after_block_26.<locals>.if_region_33.<locals>.else_block_34
kernel.<locals>.while_region_25.<locals>.while_after_block_26.<locals>.if_region_33.<locals>.then_block_34
kernel.<locals>.while_region_25.<locals>.while_before_block_26
kernel_mt
kernel_mt.<locals>.if_region_0
kernel_mt.<locals>.if_region_0.<locals>.else_block_1
kernel_mt.<locals>.if_region_0.<locals>.then_block_1
kernel_mt.<locals>.if_region_15
kernel_mt.<locals>.if_region_15.<locals>.else_block_26
kernel_mt.<locals>.if_region_15.<locals>.then_block_26
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_after_block_17
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_18
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_18.<locals>.else_block_19
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_18.<locals>.then_block_19
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_20
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_20.<locals>.else_block_21
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_20.<locals>.then_block_21
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_22
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_22.<locals>.else_block_23
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_22.<locals>.then_block_23
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_24
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_24.<locals>.else_block_25
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_after_block_17.<locals>.if_region_24.<locals>.then_block_25
kernel_mt.<locals>.if_region_15.<locals>.then_block_26.<locals>.while_region_16.<locals>.while_before_block_17
kernel_mt.<locals>.if_region_2
kernel_mt.<locals>.if_region_2.<locals>.then_block_3
kernel_mt.<locals>.if_region_37
kernel_mt.<locals>.if_region_37.<locals>.then_block_38
kernel_mt.<locals>.if_region_39
kernel_mt.<locals>.if_region_39.<locals>.else_block_40
kernel_mt.<locals>.if_region_39.<locals>.then_block_40
kernel_mt.<locals>.if_region_41
kernel_mt.<locals>.if_region_41.<locals>.then_block_46
kernel_mt.<locals>.if_region_41.<locals>.then_block_46.<locals>.if_region_42
kernel_mt.<locals>.if_region_41.<locals>.then_block_46.<locals>.if_region_42.<locals>.then_block_45
kernel_mt.<locals>.if_region_41.<locals>.then_block_46.<locals>.if_region_42.<locals>.then_block_45.<locals>.while_region_43
kernel_mt.<locals>.if_region_41.<locals>.then_block_46.<locals>.if_region_42.<locals>.then_block_45.<locals>.while_region_43.<locals>.while_after_block_44
kernel_mt.<locals>.if_region_41.<locals>.then_block_46.<locals>.if_region_42.<locals>.then_block_45.<locals>.while_region_43.<locals>.while_before_block_44
kernel_mt.<locals>.if_region_8
kernel_mt.<locals>.if_region_8.<locals>.then_block_14
kernel_mt.<locals>.if_region_8.<locals>.then_block_14.<locals>.ifexp_else_block_10
kernel_mt.<locals>.if_region_8.<locals>.then_block_14.<locals>.ifexp_else_block_11
kernel_mt.<locals>.if_region_8.<locals>.then_block_14.<locals>.ifexp_else_block_9
kernel_mt.<locals>.if_region_8.<locals>.then_block_14.<locals>.ifexp_then_block_10
kernel_mt.<locals>.if_region_8.<locals>.then_block_14.<locals>.ifexp_then_block_11
kernel_mt.<locals>.if_region_8.<locals>.then_block_14.<locals>.ifexp_then_block_9
kernel_mt.<locals>.if_region_8.<locals>.then_block_14.<locals>.while_region_12
kernel_mt.<locals>.if_region_8.<locals>.then_block_14.<locals>.while_region_12.<locals>.while_after_block_13
kernel_mt.<locals>.if_region_8.<locals>.then_block_14.<locals>.while_region_12.<locals>.while_before_block_13
kernel_mt.<locals>.ifexp_else_block_4
kernel_mt.<locals>.ifexp_else_block_5
kernel_mt.<locals>.ifexp_then_block_4
kernel_mt.<locals>.ifexp_then_block_5
kernel_mt.<locals>.while_region_27
kernel_mt.<locals>.while_region_27.<locals>.while_after_block_28
kernel_mt.<locals>.while_region_27.<locals>.while_after_block_28.<locals>.if_region_29
kernel_mt.<locals>.while_region_27.<locals>.while_after_block_28.<locals>.if_region_29.<locals>.else_block_30
kernel_mt.<locals>.while_region_27.<locals>.while_after_block_28.<locals>.if_region_29.<locals>.then_block_30
kernel_mt.<locals>.while_region_27.<locals>.while_after_block_28.<locals>.if_region_31
kernel_mt.<locals>.while_region_27.<locals>.while_after_block_28.<locals>.if_region_31.<locals>.else_block_32
kernel_mt.<locals>.while_region_27.<locals>.while_after_block_28.<locals>.if_region_31.<locals>.then_block_32
kernel_mt.<locals>.while_region_27.<locals>.while_after_block_28.<locals>.if_region_33
kernel_mt.<locals>.while_region_27.<locals>.while_after_block_28.<locals>.if_region_33.<locals>.else_block_34
kernel_mt.<locals>.while_region_27.<locals>.while_after_block_28.<locals>.if_region_33.<locals>.then_block_34
kernel_mt.<locals>.while_region_27.<locals>.while_after_block_28.<locals>.if_region_35
kernel_mt.<locals>.while_region_27.<locals>.while_after_block_28.<locals>.if_region_35.<locals>.else_block_36
kernel_mt.<locals>.while_region_27.<locals>.while_after_block_28.<locals>.if_region_35.<locals>.then_block_36
kernel_mt.<locals>.while_region_27.<locals>.while_before_block_28
```

### `tensorrt_llm/_torch/kernels/blackwell/mnnvl_slot_allreduce_norm.py`（57）

```text
__call__
kernel
kernel.<locals>.if_region_1
kernel.<locals>.if_region_1.<locals>.then_block_2
kernel.<locals>.if_region_17
kernel.<locals>.if_region_17.<locals>.then_block_18
kernel.<locals>.if_region_21
kernel.<locals>.if_region_21.<locals>.then_block_22
kernel.<locals>.if_region_3
kernel.<locals>.if_region_3.<locals>.then_block_4
kernel.<locals>.if_region_33
kernel.<locals>.if_region_33.<locals>.then_block_34
kernel.<locals>.if_region_35
kernel.<locals>.if_region_35.<locals>.then_block_36
kernel.<locals>.if_region_39
kernel.<locals>.if_region_39.<locals>.then_block_40
kernel.<locals>.if_region_5
kernel.<locals>.if_region_5.<locals>.else_block_6
kernel.<locals>.if_region_5.<locals>.then_block_6
kernel.<locals>.if_region_7
kernel.<locals>.if_region_7.<locals>.else_block_8
kernel.<locals>.if_region_7.<locals>.then_block_8
kernel.<locals>.ifexp_else_block_0
kernel.<locals>.ifexp_then_block_0
kernel.<locals>.while_region_23
kernel.<locals>.while_region_23.<locals>.while_after_block_24
kernel.<locals>.while_region_23.<locals>.while_after_block_24.<locals>.if_region_25
kernel.<locals>.while_region_23.<locals>.while_after_block_24.<locals>.if_region_25.<locals>.else_block_26
kernel.<locals>.while_region_23.<locals>.while_after_block_24.<locals>.if_region_25.<locals>.then_block_26
kernel.<locals>.while_region_23.<locals>.while_after_block_24.<locals>.if_region_27
kernel.<locals>.while_region_23.<locals>.while_after_block_24.<locals>.if_region_27.<locals>.else_block_28
kernel.<locals>.while_region_23.<locals>.while_after_block_24.<locals>.if_region_27.<locals>.then_block_28
kernel.<locals>.while_region_23.<locals>.while_after_block_24.<locals>.if_region_29
kernel.<locals>.while_region_23.<locals>.while_after_block_24.<locals>.if_region_29.<locals>.else_block_30
kernel.<locals>.while_region_23.<locals>.while_after_block_24.<locals>.if_region_29.<locals>.then_block_30
kernel.<locals>.while_region_23.<locals>.while_after_block_24.<locals>.if_region_31
kernel.<locals>.while_region_23.<locals>.while_after_block_24.<locals>.if_region_31.<locals>.else_block_32
kernel.<locals>.while_region_23.<locals>.while_after_block_24.<locals>.if_region_31.<locals>.then_block_32
kernel.<locals>.while_region_23.<locals>.while_before_block_24
kernel.<locals>.while_region_37
kernel.<locals>.while_region_37.<locals>.while_after_block_38
kernel.<locals>.while_region_37.<locals>.while_before_block_38
kernel.<locals>.while_region_41
kernel.<locals>.while_region_41.<locals>.while_after_block_42
kernel.<locals>.while_region_41.<locals>.while_before_block_42
kernel.<locals>.while_region_9
kernel.<locals>.while_region_9.<locals>.while_after_block_10
kernel.<locals>.while_region_9.<locals>.while_after_block_10.<locals>.if_region_11
kernel.<locals>.while_region_9.<locals>.while_after_block_10.<locals>.if_region_11.<locals>.else_block_16
kernel.<locals>.while_region_9.<locals>.while_after_block_10.<locals>.if_region_11.<locals>.then_block_16
kernel.<locals>.while_region_9.<locals>.while_after_block_10.<locals>.if_region_11.<locals>.then_block_16.<locals>.if_region_12
kernel.<locals>.while_region_9.<locals>.while_after_block_10.<locals>.if_region_11.<locals>.then_block_16.<locals>.if_region_12.<locals>.else_block_13
kernel.<locals>.while_region_9.<locals>.while_after_block_10.<locals>.if_region_11.<locals>.then_block_16.<locals>.if_region_12.<locals>.then_block_13
kernel.<locals>.while_region_9.<locals>.while_after_block_10.<locals>.if_region_11.<locals>.then_block_16.<locals>.if_region_14
kernel.<locals>.while_region_9.<locals>.while_after_block_10.<locals>.if_region_11.<locals>.then_block_16.<locals>.if_region_14.<locals>.else_block_15
kernel.<locals>.while_region_9.<locals>.while_after_block_10.<locals>.if_region_11.<locals>.then_block_16.<locals>.if_region_14.<locals>.then_block_15
kernel.<locals>.while_region_9.<locals>.while_before_block_10
```

### `tensorrt_llm/_torch/kernels/blackwell/moe_lowm_fc2.py`（8）

```text
load_b.<locals>.loop_body_4.<locals>.if_region_12
load_b.<locals>.loop_body_4.<locals>.if_region_12.<locals>.else_block_15
load_b.<locals>.loop_body_4.<locals>.if_region_12.<locals>.then_block_15
load_b.<locals>.loop_body_4.<locals>.if_region_12.<locals>.then_block_15.<locals>.if_region_13
load_b.<locals>.loop_body_4.<locals>.if_region_12.<locals>.then_block_15.<locals>.if_region_13.<locals>.else_block_14
load_b.<locals>.loop_body_4.<locals>.if_region_12.<locals>.then_block_15.<locals>.if_region_13.<locals>.then_block_14
load_b.<locals>.loop_body_4.<locals>.if_region_16
load_b.<locals>.loop_body_4.<locals>.if_region_16.<locals>.then_block_17
```

### `tensorrt_llm/_torch/kernels/blackwell/tp2_allreduce_norm.py`（35）

```text
<module>
Tp2WarpAllReduceNormKernel
__call__
_finalize
_finalize.<locals>.if_region_0
_finalize.<locals>.if_region_0.<locals>.else_block_1
_finalize.<locals>.if_region_0.<locals>.then_block_1
kernel
kernel.<locals>.if_region_10
kernel.<locals>.if_region_10.<locals>.then_block_11
kernel.<locals>.if_region_12
kernel.<locals>.if_region_12.<locals>.then_block_15
kernel.<locals>.if_region_12.<locals>.then_block_15.<locals>.while_region_13
kernel.<locals>.if_region_12.<locals>.then_block_15.<locals>.while_region_13.<locals>.while_after_block_14
kernel.<locals>.if_region_12.<locals>.then_block_15.<locals>.while_region_13.<locals>.while_before_block_14
kernel.<locals>.if_region_18
kernel.<locals>.if_region_18.<locals>.then_block_21
kernel.<locals>.if_region_18.<locals>.then_block_21.<locals>.if_region_19
kernel.<locals>.if_region_18.<locals>.then_block_21.<locals>.if_region_19.<locals>.then_block_20
kernel.<locals>.if_region_2
kernel.<locals>.if_region_2.<locals>.then_block_3
kernel.<locals>.if_region_4
kernel.<locals>.if_region_4.<locals>.then_block_7
kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.while_region_5
kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.while_region_5.<locals>.while_after_block_6
kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.while_region_5.<locals>.while_before_block_6
kernel.<locals>.while_region_0
kernel.<locals>.while_region_0.<locals>.while_after_block_1
kernel.<locals>.while_region_0.<locals>.while_before_block_1
kernel.<locals>.while_region_16
kernel.<locals>.while_region_16.<locals>.while_after_block_17
kernel.<locals>.while_region_16.<locals>.while_before_block_17
kernel.<locals>.while_region_8
kernel.<locals>.while_region_8.<locals>.while_after_block_9
kernel.<locals>.while_region_8.<locals>.while_before_block_9
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/comm/tp4_allgather_lamport.py`（24）

```text
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_14
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_14.<locals>.then_block_15
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_16
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_16.<locals>.then_block_17
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_6
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_6.<locals>.else_block_13
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_6.<locals>.then_block_13
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_6.<locals>.then_block_13.<locals>.if_region_7
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_6.<locals>.then_block_13.<locals>.if_region_7.<locals>.then_block_8
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_6.<locals>.then_block_13.<locals>.if_region_9
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_6.<locals>.then_block_13.<locals>.if_region_9.<locals>.else_block_12
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_6.<locals>.then_block_13.<locals>.if_region_9.<locals>.then_block_12
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_6.<locals>.then_block_13.<locals>.if_region_9.<locals>.then_block_12.<locals>.while_region_10
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_6.<locals>.then_block_13.<locals>.if_region_9.<locals>.then_block_12.<locals>.while_region_10.<locals>.while_after_block_11
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.if_region_6.<locals>.then_block_13.<locals>.if_region_9.<locals>.then_block_12.<locals>.while_region_10.<locals>.while_before_block_11
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.while_region_2
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.while_region_2.<locals>.while_after_block_3
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_4
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_4.<locals>.else_block_5
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_4.<locals>.then_block_5
_glm_tp_allgather_bf16_oneshot_lamport_kernel.<locals>.while_region_2.<locals>.while_before_block_3
_sanitize_bf16x8.<locals>.if_region_0
_sanitize_bf16x8.<locals>.if_region_0.<locals>.else_block_1
_sanitize_bf16x8.<locals>.if_region_0.<locals>.then_block_1
```

### `tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tep4/kernels/comm/tp4_allreduce_fused_rms.py`（574）

```text
_advance_signal_if_last_cta.<locals>.if_region_0
_advance_signal_if_last_cta.<locals>.if_region_0.<locals>.then_block_1
_cta_head_load.<locals>.if_region_0
_cta_head_load.<locals>.if_region_0.<locals>.else_block_1
_cta_head_load.<locals>.if_region_0.<locals>.then_block_1
_decode_positive_e4m3.<locals>.if_region_0
_decode_positive_e4m3.<locals>.if_region_0.<locals>.else_block_1
_decode_positive_e4m3.<locals>.if_region_0.<locals>.then_block_1
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_0
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_0.<locals>.else_block_1
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_0.<locals>.then_block_1
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.else_block_63
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_13
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_13.<locals>.else_block_20
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_13.<locals>.then_block_20
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_13.<locals>.then_block_20.<locals>.if_region_14
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_13.<locals>.then_block_20.<locals>.if_region_14.<locals>.else_block_17
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_13.<locals>.then_block_20.<locals>.if_region_14.<locals>.else_block_17.<locals>.if_region_18
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_13.<locals>.then_block_20.<locals>.if_region_14.<locals>.else_block_17.<locals>.if_region_18.<locals>.else_block_19
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_13.<locals>.then_block_20.<locals>.if_region_14.<locals>.else_block_17.<locals>.if_region_18.<locals>.then_block_19
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_13.<locals>.then_block_20.<locals>.if_region_14.<locals>.then_block_17
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_13.<locals>.then_block_20.<locals>.if_region_14.<locals>.then_block_17.<locals>.if_region_15
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_13.<locals>.then_block_20.<locals>.if_region_14.<locals>.then_block_17.<locals>.if_region_15.<locals>.else_block_16
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_13.<locals>.then_block_20.<locals>.if_region_14.<locals>.then_block_17.<locals>.if_region_15.<locals>.then_block_16
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_21
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_21.<locals>.else_block_24
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_21.<locals>.then_block_24
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_21.<locals>.then_block_24.<locals>.if_region_22
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_21.<locals>.then_block_24.<locals>.if_region_22.<locals>.else_block_23
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_21.<locals>.then_block_24.<locals>.if_region_22.<locals>.then_block_23
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_25
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_25.<locals>.then_block_28
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_25.<locals>.then_block_28.<locals>.if_region_26
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_25.<locals>.then_block_28.<locals>.if_region_26.<locals>.else_block_27
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_25.<locals>.then_block_28.<locals>.if_region_26.<locals>.then_block_27
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_29
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_29.<locals>.then_block_32
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_29.<locals>.then_block_32.<locals>.if_region_30
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_29.<locals>.then_block_32.<locals>.if_region_30.<locals>.then_block_31
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.else_block_56
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.then_block_56
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.then_block_56.<locals>.if_region_48
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.then_block_56.<locals>.if_region_48.<locals>.else_block_51
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.then_block_56.<locals>.if_region_48.<locals>.else_block_51.<locals>.while_region_52
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.then_block_56.<locals>.if_region_48.<locals>.else_block_51.<locals>.while_region_52.<locals>.while_after_block_53
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.then_block_56.<locals>.if_region_48.<locals>.else_block_51.<locals>.while_region_52.<locals>.while_before_block_53
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.then_block_56.<locals>.if_region_48.<locals>.then_block_51
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.then_block_56.<locals>.if_region_48.<locals>.then_block_51.<locals>.while_region_49
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.then_block_56.<locals>.if_region_48.<locals>.then_block_51.<locals>.while_region_49.<locals>.while_after_block_50
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.then_block_56.<locals>.if_region_48.<locals>.then_block_51.<locals>.while_region_49.<locals>.while_before_block_50
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.then_block_56.<locals>.if_region_54
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.then_block_56.<locals>.if_region_54.<locals>.else_block_55
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_47.<locals>.then_block_56.<locals>.if_region_54.<locals>.then_block_55
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_57
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_57.<locals>.else_block_60
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_57.<locals>.then_block_60
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_57.<locals>.then_block_60.<locals>.if_region_58
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_57.<locals>.then_block_60.<locals>.if_region_58.<locals>.else_block_59
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.else_block_46.<locals>.if_region_57.<locals>.then_block_60.<locals>.if_region_58.<locals>.then_block_59
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.if_region_38
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.if_region_38.<locals>.else_block_41
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.if_region_38.<locals>.then_block_41
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.if_region_38.<locals>.then_block_41.<locals>.if_region_39
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.if_region_38.<locals>.then_block_41.<locals>.if_region_39.<locals>.else_block_40
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.if_region_38.<locals>.then_block_41.<locals>.if_region_39.<locals>.then_block_40
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.if_region_42
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.if_region_42.<locals>.else_block_45
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.if_region_42.<locals>.then_block_45
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.if_region_42.<locals>.then_block_45.<locals>.if_region_43
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.if_region_42.<locals>.then_block_45.<locals>.if_region_43.<locals>.else_block_44
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.if_region_42.<locals>.then_block_45.<locals>.if_region_43.<locals>.then_block_44
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.while_region_34
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.while_region_34.<locals>.while_after_block_35
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.while_region_34.<locals>.while_after_block_35.<locals>.if_region_36
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.while_region_34.<locals>.while_after_block_35.<locals>.if_region_36.<locals>.else_block_37
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.while_region_34.<locals>.while_after_block_35.<locals>.if_region_36.<locals>.then_block_37
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_33.<locals>.then_block_46.<locals>.while_region_34.<locals>.while_before_block_35
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_61
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_61.<locals>.else_block_62
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_12.<locals>.then_block_63.<locals>.if_region_61.<locals>.then_block_62
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.else_block_11
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.then_block_11
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.then_block_11.<locals>.while_region_3
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.then_block_11.<locals>.while_region_3.<locals>.while_after_block_4
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.then_block_11.<locals>.while_region_3.<locals>.while_after_block_4.<locals>.if_region_5
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.then_block_11.<locals>.while_region_3.<locals>.while_after_block_4.<locals>.if_region_5.<locals>.else_block_8
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.then_block_11.<locals>.while_region_3.<locals>.while_after_block_4.<locals>.if_region_5.<locals>.else_block_8.<locals>.if_region_9
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.then_block_11.<locals>.while_region_3.<locals>.while_after_block_4.<locals>.if_region_5.<locals>.else_block_8.<locals>.if_region_9.<locals>.else_block_10
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.then_block_11.<locals>.while_region_3.<locals>.while_after_block_4.<locals>.if_region_5.<locals>.else_block_8.<locals>.if_region_9.<locals>.then_block_10
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.then_block_11.<locals>.while_region_3.<locals>.while_after_block_4.<locals>.if_region_5.<locals>.then_block_8
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.then_block_11.<locals>.while_region_3.<locals>.while_after_block_4.<locals>.if_region_5.<locals>.then_block_8.<locals>.if_region_6
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.then_block_11.<locals>.while_region_3.<locals>.while_after_block_4.<locals>.if_region_5.<locals>.then_block_8.<locals>.if_region_6.<locals>.else_block_7
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.then_block_11.<locals>.while_region_3.<locals>.while_after_block_4.<locals>.if_region_5.<locals>.then_block_8.<locals>.if_region_6.<locals>.then_block_7
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_2.<locals>.then_block_11.<locals>.while_region_3.<locals>.while_before_block_4
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_64
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_64.<locals>.then_block_65
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_66
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_66.<locals>.then_block_73
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_66.<locals>.then_block_73.<locals>.if_region_67
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_66.<locals>.then_block_73.<locals>.if_region_67.<locals>.then_block_68
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_66.<locals>.then_block_73.<locals>.if_region_69
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_66.<locals>.then_block_73.<locals>.if_region_69.<locals>.then_block_72
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_66.<locals>.then_block_73.<locals>.if_region_69.<locals>.then_block_72.<locals>.while_region_70
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_66.<locals>.then_block_73.<locals>.if_region_69.<locals>.then_block_72.<locals>.while_region_70.<locals>.while_after_block_71
_glm_tp_allreduce_bf16_oneshot_lamport_serving_kernel.<locals>.if_region_66.<locals>.then_block_73.<locals>.if_region_69.<locals>.then_block_72.<locals>.while_region_70.<locals>.while_before_block_71
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_0
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_0.<locals>.then_block_1
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_10
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_10.<locals>.then_block_11
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_12
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_12.<locals>.then_block_13
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_125
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_125.<locals>.then_block_126
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_127
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_127.<locals>.else_block_128
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_127.<locals>.then_block_128
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_129
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_129.<locals>.else_block_134
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_129.<locals>.then_block_134
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_129.<locals>.then_block_134.<locals>.if_region_130
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_129.<locals>.then_block_134.<locals>.if_region_130.<locals>.else_block_131
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_129.<locals>.then_block_134.<locals>.if_region_130.<locals>.else_block_131.<locals>.elif_region_131
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_129.<locals>.then_block_134.<locals>.if_region_130.<locals>.else_block_131.<locals>.elif_region_131.<locals>.else_block_132
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_129.<locals>.then_block_134.<locals>.if_region_130.<locals>.else_block_131.<locals>.elif_region_131.<locals>.else_block_132.<locals>.elif_region_132
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_129.<locals>.then_block_134.<locals>.if_region_130.<locals>.else_block_131.<locals>.elif_region_131.<locals>.else_block_132.<locals>.elif_region_132.<locals>.else_block_133
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_129.<locals>.then_block_134.<locals>.if_region_130.<locals>.else_block_131.<locals>.elif_region_131.<locals>.else_block_132.<locals>.elif_region_132.<locals>.then_block_133
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_129.<locals>.then_block_134.<locals>.if_region_130.<locals>.else_block_131.<locals>.elif_region_131.<locals>.then_block_132
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_129.<locals>.then_block_134.<locals>.if_region_130.<locals>.then_block_131
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.else_block_27
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.if_region_15
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.if_region_15.<locals>.else_block_16
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.if_region_15.<locals>.then_block_16
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.if_region_17
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.if_region_17.<locals>.else_block_18
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.if_region_17.<locals>.then_block_18
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.while_region_19
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.while_region_19.<locals>.while_after_block_20
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.while_region_19.<locals>.while_after_block_20.<locals>.if_region_21
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.while_region_19.<locals>.while_after_block_20.<locals>.if_region_21.<locals>.else_block_24
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.while_region_19.<locals>.while_after_block_20.<locals>.if_region_21.<locals>.else_block_24.<locals>.if_region_25
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.while_region_19.<locals>.while_after_block_20.<locals>.if_region_21.<locals>.else_block_24.<locals>.if_region_25.<locals>.else_block_26
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.while_region_19.<locals>.while_after_block_20.<locals>.if_region_21.<locals>.else_block_24.<locals>.if_region_25.<locals>.then_block_26
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.while_region_19.<locals>.while_after_block_20.<locals>.if_region_21.<locals>.then_block_24
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.while_region_19.<locals>.while_after_block_20.<locals>.if_region_21.<locals>.then_block_24.<locals>.if_region_22
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.while_region_19.<locals>.while_after_block_20.<locals>.if_region_21.<locals>.then_block_24.<locals>.if_region_22.<locals>.else_block_23
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.while_region_19.<locals>.while_after_block_20.<locals>.if_region_21.<locals>.then_block_24.<locals>.if_region_22.<locals>.then_block_23
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_27.<locals>.while_region_19.<locals>.while_before_block_20
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_145
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_145.<locals>.then_block_146
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_147
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_147.<locals>.else_block_148
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_147.<locals>.then_block_148
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_149
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_149.<locals>.then_block_150
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_151
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_151.<locals>.then_block_152
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_156
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_156.<locals>.then_block_159
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_156.<locals>.then_block_159.<locals>.if_region_157
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_156.<locals>.then_block_159.<locals>.if_region_157.<locals>.then_block_158
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_160
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_160.<locals>.then_block_167
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_160.<locals>.then_block_167.<locals>.if_region_161
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_160.<locals>.then_block_167.<locals>.if_region_161.<locals>.else_block_164
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_160.<locals>.then_block_167.<locals>.if_region_161.<locals>.else_block_164.<locals>.while_region_165
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_160.<locals>.then_block_167.<locals>.if_region_161.<locals>.else_block_164.<locals>.while_region_165.<locals>.while_after_block_166
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_160.<locals>.then_block_167.<locals>.if_region_161.<locals>.else_block_164.<locals>.while_region_165.<locals>.while_before_block_166
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_160.<locals>.then_block_167.<locals>.if_region_161.<locals>.then_block_164
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_160.<locals>.then_block_167.<locals>.if_region_161.<locals>.then_block_164.<locals>.while_region_162
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_160.<locals>.then_block_167.<locals>.if_region_161.<locals>.then_block_164.<locals>.while_region_162.<locals>.while_after_block_163
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_155.<locals>.then_block_168.<locals>.if_region_160.<locals>.then_block_167.<locals>.if_region_161.<locals>.then_block_164.<locals>.while_region_162.<locals>.while_before_block_163
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_2
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_2.<locals>.else_block_5
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_2.<locals>.then_block_5
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3.<locals>.else_block_4
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3.<locals>.then_block_4
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_28
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_28.<locals>.then_block_29
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.else_block_124
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_118
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_118.<locals>.then_block_119
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_120
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_120.<locals>.else_block_121
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_120.<locals>.then_block_121
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_122
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_122.<locals>.else_block_123
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_122.<locals>.then_block_123
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_31
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_31.<locals>.else_block_38
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_31.<locals>.then_block_38
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_31.<locals>.then_block_38.<locals>.if_region_32
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_31.<locals>.then_block_38.<locals>.if_region_32.<locals>.else_block_35
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_31.<locals>.then_block_38.<locals>.if_region_32.<locals>.else_block_35.<locals>.if_region_36
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_31.<locals>.then_block_38.<locals>.if_region_32.<locals>.else_block_35.<locals>.if_region_36.<locals>.else_block_37
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_31.<locals>.then_block_38.<locals>.if_region_32.<locals>.else_block_35.<locals>.if_region_36.<locals>.then_block_37
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_31.<locals>.then_block_38.<locals>.if_region_32.<locals>.then_block_35
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_31.<locals>.then_block_38.<locals>.if_region_32.<locals>.then_block_35.<locals>.if_region_33
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_31.<locals>.then_block_38.<locals>.if_region_32.<locals>.then_block_35.<locals>.if_region_33.<locals>.else_block_34
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_31.<locals>.then_block_38.<locals>.if_region_32.<locals>.then_block_35.<locals>.if_region_33.<locals>.then_block_34
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_39
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_39.<locals>.else_block_40
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_39.<locals>.then_block_40
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_41
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_41.<locals>.else_block_44
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_41.<locals>.then_block_44
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_41.<locals>.then_block_44.<locals>.if_region_42
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_41.<locals>.then_block_44.<locals>.if_region_42.<locals>.else_block_43
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_41.<locals>.then_block_44.<locals>.if_region_42.<locals>.then_block_43
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_45
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_45.<locals>.then_block_46
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_47
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_47.<locals>.then_block_50
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_47.<locals>.then_block_50.<locals>.if_region_48
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_47.<locals>.then_block_50.<locals>.if_region_48.<locals>.else_block_49
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_47.<locals>.then_block_50.<locals>.if_region_48.<locals>.then_block_49
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_51
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_51.<locals>.then_block_52
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_53
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_53.<locals>.then_block_56
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_53.<locals>.then_block_56.<locals>.if_region_54
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_53.<locals>.then_block_56.<locals>.if_region_54.<locals>.then_block_55
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_57
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_57.<locals>.else_block_58
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_57.<locals>.then_block_58
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_104
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_104.<locals>.else_block_107
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_104.<locals>.then_block_107
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_104.<locals>.then_block_107.<locals>.if_region_105
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_104.<locals>.then_block_107.<locals>.if_region_105.<locals>.else_block_106
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_104.<locals>.then_block_107.<locals>.if_region_105.<locals>.then_block_106
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.else_block_117
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.then_block_117
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_109
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_109.<locals>.else_block_112
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_109.<locals>.then_block_112
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_109.<locals>.then_block_112.<locals>.if_region_110
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_109.<locals>.then_block_112.<locals>.if_region_110.<locals>.else_block_111
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_109.<locals>.then_block_112.<locals>.if_region_110.<locals>.then_block_111
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_113
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_113.<locals>.else_block_116
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_113.<locals>.then_block_116
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_113.<locals>.then_block_116.<locals>.if_region_114
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_113.<locals>.then_block_116.<locals>.if_region_114.<locals>.else_block_115
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_113.<locals>.then_block_116.<locals>.if_region_114.<locals>.then_block_115
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.else_block_103
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_100
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_100.<locals>.else_block_101
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_100.<locals>.else_block_101.<locals>.elif_region_101
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_100.<locals>.else_block_101.<locals>.elif_region_101.<locals>.else_block_102
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_100.<locals>.else_block_101.<locals>.elif_region_101.<locals>.then_block_102
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_100.<locals>.then_block_101
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_94
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_94.<locals>.else_block_97
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_94.<locals>.else_block_97.<locals>.while_region_98
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_94.<locals>.else_block_97.<locals>.while_region_98.<locals>.while_after_block_99
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_94.<locals>.else_block_97.<locals>.while_region_98.<locals>.while_before_block_99
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_94.<locals>.then_block_97
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_94.<locals>.then_block_97.<locals>.while_region_95
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_94.<locals>.then_block_97.<locals>.while_region_95.<locals>.while_after_block_96
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.if_region_93.<locals>.then_block_103.<locals>.if_region_94.<locals>.then_block_97.<locals>.while_region_95.<locals>.while_before_block_96
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.else_block_91.<locals>.ifexp_then_block_92
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.if_region_83
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.if_region_83.<locals>.else_block_86
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.if_region_83.<locals>.then_block_86
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.if_region_83.<locals>.then_block_86.<locals>.if_region_84
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.if_region_83.<locals>.then_block_86.<locals>.if_region_84.<locals>.else_block_85
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.if_region_83.<locals>.then_block_86.<locals>.if_region_84.<locals>.then_block_85
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.if_region_87
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.if_region_87.<locals>.else_block_90
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.if_region_87.<locals>.then_block_90
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.if_region_87.<locals>.then_block_90.<locals>.if_region_88
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.if_region_87.<locals>.then_block_90.<locals>.if_region_88.<locals>.else_block_89
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.if_region_87.<locals>.then_block_90.<locals>.if_region_88.<locals>.then_block_89
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.while_region_79
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.while_region_79.<locals>.while_after_block_80
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.while_region_79.<locals>.while_after_block_80.<locals>.if_region_81
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.while_region_79.<locals>.while_after_block_80.<locals>.if_region_81.<locals>.else_block_82
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.while_region_79.<locals>.while_after_block_80.<locals>.if_region_81.<locals>.then_block_82
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.else_block_78.<locals>.elif_region_78.<locals>.then_block_91.<locals>.while_region_79.<locals>.while_before_block_80
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_60
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_60.<locals>.else_block_61
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_60.<locals>.then_block_61
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_70
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_70.<locals>.else_block_73
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_70.<locals>.then_block_73
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_70.<locals>.then_block_73.<locals>.if_region_71
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_70.<locals>.then_block_73.<locals>.if_region_71.<locals>.else_block_72
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_70.<locals>.then_block_73.<locals>.if_region_71.<locals>.then_block_72
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_74
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_74.<locals>.else_block_77
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_74.<locals>.then_block_77
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_74.<locals>.then_block_77.<locals>.if_region_75
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_74.<locals>.then_block_77.<locals>.if_region_75.<locals>.else_block_76
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.if_region_74.<locals>.then_block_77.<locals>.if_region_75.<locals>.then_block_76
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.while_region_62
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.while_region_62.<locals>.while_after_block_63
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.while_region_62.<locals>.while_after_block_63.<locals>.if_region_64
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.while_region_62.<locals>.while_after_block_63.<locals>.if_region_64.<locals>.else_block_69
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.while_region_62.<locals>.while_after_block_63.<locals>.if_region_64.<locals>.then_block_69
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.while_region_62.<locals>.while_after_block_63.<locals>.if_region_64.<locals>.then_block_69.<locals>.if_region_65
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.while_region_62.<locals>.while_after_block_63.<locals>.if_region_64.<locals>.then_block_69.<locals>.if_region_65.<locals>.else_block_68
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.while_region_62.<locals>.while_after_block_63.<locals>.if_region_64.<locals>.then_block_69.<locals>.if_region_65.<locals>.then_block_68
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.while_region_62.<locals>.while_after_block_63.<locals>.if_region_64.<locals>.then_block_69.<locals>.if_region_65.<locals>.then_block_68.<locals>.if_region_66
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.while_region_62.<locals>.while_after_block_63.<locals>.if_region_64.<locals>.then_block_69.<locals>.if_region_65.<locals>.then_block_68.<locals>.if_region_66.<locals>.else_block_67
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.while_region_62.<locals>.while_after_block_63.<locals>.if_region_64.<locals>.then_block_69.<locals>.if_region_65.<locals>.then_block_68.<locals>.if_region_66.<locals>.then_block_67
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_124.<locals>.if_region_59.<locals>.then_block_78.<locals>.while_region_62.<locals>.while_before_block_63
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_6
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_6.<locals>.else_block_9
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_6.<locals>.then_block_9
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_6.<locals>.then_block_9.<locals>.if_region_7
_glm_tp_allreduce_residual_rms_norm_bf16_oneshot_lamport_split_row_kernel.<locals>.if_region_6.<locals>.then_block_9.<locals>.if_region_7.<locals>.then_block_8
_glm_tp_allreduce_residual_rms_norm_bf16_twoshot_nvls.<locals>.ifexp_then_block_0
_glm_tp_allreduce_residual_rms_norm_out_mxf8_twoshot_push.<locals>.ifexp_then_block_0
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_0
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_0.<locals>.then_block_1
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_104
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_104.<locals>.then_block_105
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_106
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_106.<locals>.else_block_107
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_106.<locals>.then_block_107
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108.<locals>.else_block_117
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108.<locals>.then_block_117
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_109
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_109.<locals>.else_block_110
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_109.<locals>.then_block_110
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_111
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_111.<locals>.else_block_112
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_111.<locals>.then_block_112
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_113
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_113.<locals>.then_block_114
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_115
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_115.<locals>.else_block_116
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_108.<locals>.then_block_117.<locals>.if_region_115.<locals>.then_block_116
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_118
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_118.<locals>.then_block_119
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_12
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_12.<locals>.then_block_13
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_122
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_122.<locals>.then_block_129
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_122.<locals>.then_block_129.<locals>.if_region_123
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_122.<locals>.then_block_129.<locals>.if_region_123.<locals>.then_block_124
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_122.<locals>.then_block_129.<locals>.if_region_125
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_122.<locals>.then_block_129.<locals>.if_region_125.<locals>.then_block_128
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_122.<locals>.then_block_129.<locals>.if_region_125.<locals>.then_block_128.<locals>.while_region_126
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_122.<locals>.then_block_129.<locals>.if_region_125.<locals>.then_block_128.<locals>.while_region_126.<locals>.while_after_block_127
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_122.<locals>.then_block_129.<locals>.if_region_125.<locals>.then_block_128.<locals>.while_region_126.<locals>.while_before_block_127
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_14
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_14.<locals>.then_block_15
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.else_block_29
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.if_region_17
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.if_region_17.<locals>.else_block_18
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.if_region_17.<locals>.then_block_18
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.if_region_19
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.if_region_19.<locals>.else_block_20
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.if_region_19.<locals>.then_block_20
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.while_region_21
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.while_region_21.<locals>.while_after_block_22
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.while_region_21.<locals>.while_after_block_22.<locals>.if_region_23
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.while_region_21.<locals>.while_after_block_22.<locals>.if_region_23.<locals>.else_block_26
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.while_region_21.<locals>.while_after_block_22.<locals>.if_region_23.<locals>.else_block_26.<locals>.if_region_27
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.while_region_21.<locals>.while_after_block_22.<locals>.if_region_23.<locals>.else_block_26.<locals>.if_region_27.<locals>.else_block_28
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.while_region_21.<locals>.while_after_block_22.<locals>.if_region_23.<locals>.else_block_26.<locals>.if_region_27.<locals>.then_block_28
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.while_region_21.<locals>.while_after_block_22.<locals>.if_region_23.<locals>.then_block_26
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.while_region_21.<locals>.while_after_block_22.<locals>.if_region_23.<locals>.then_block_26.<locals>.if_region_24
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.while_region_21.<locals>.while_after_block_22.<locals>.if_region_23.<locals>.then_block_26.<locals>.if_region_24.<locals>.else_block_25
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.while_region_21.<locals>.while_after_block_22.<locals>.if_region_23.<locals>.then_block_26.<locals>.if_region_24.<locals>.then_block_25
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_16.<locals>.then_block_29.<locals>.while_region_21.<locals>.while_before_block_22
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_2
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_2.<locals>.else_block_3
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_2.<locals>.then_block_3
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_30
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_30.<locals>.then_block_31
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.else_block_86
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_33
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_33.<locals>.else_block_40
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_33.<locals>.then_block_40
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_33.<locals>.then_block_40.<locals>.if_region_34
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_33.<locals>.then_block_40.<locals>.if_region_34.<locals>.else_block_37
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_33.<locals>.then_block_40.<locals>.if_region_34.<locals>.else_block_37.<locals>.if_region_38
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_33.<locals>.then_block_40.<locals>.if_region_34.<locals>.else_block_37.<locals>.if_region_38.<locals>.else_block_39
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_33.<locals>.then_block_40.<locals>.if_region_34.<locals>.else_block_37.<locals>.if_region_38.<locals>.then_block_39
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_33.<locals>.then_block_40.<locals>.if_region_34.<locals>.then_block_37
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_33.<locals>.then_block_40.<locals>.if_region_34.<locals>.then_block_37.<locals>.if_region_35
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_33.<locals>.then_block_40.<locals>.if_region_34.<locals>.then_block_37.<locals>.if_region_35.<locals>.else_block_36
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_33.<locals>.then_block_40.<locals>.if_region_34.<locals>.then_block_37.<locals>.if_region_35.<locals>.then_block_36
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_41
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_41.<locals>.else_block_42
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_41.<locals>.then_block_42
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_43
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_43.<locals>.else_block_46
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_43.<locals>.then_block_46
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_43.<locals>.then_block_46.<locals>.if_region_44
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_43.<locals>.then_block_46.<locals>.if_region_44.<locals>.else_block_45
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_43.<locals>.then_block_46.<locals>.if_region_44.<locals>.then_block_45
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_47
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_47.<locals>.then_block_48
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_49
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_49.<locals>.then_block_52
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_49.<locals>.then_block_52.<locals>.if_region_50
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_49.<locals>.then_block_52.<locals>.if_region_50.<locals>.else_block_51
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_49.<locals>.then_block_52.<locals>.if_region_50.<locals>.then_block_51
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_53
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_53.<locals>.else_block_54
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_53.<locals>.then_block_54
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_55
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_55.<locals>.then_block_56
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_57
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_57.<locals>.then_block_60
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_57.<locals>.then_block_60.<locals>.if_region_58
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_57.<locals>.then_block_60.<locals>.if_region_58.<locals>.then_block_59
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_61
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_61.<locals>.then_block_62
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.else_block_73
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.then_block_73
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.then_block_73.<locals>.if_region_65
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.then_block_73.<locals>.if_region_65.<locals>.else_block_68
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.then_block_73.<locals>.if_region_65.<locals>.else_block_68.<locals>.while_region_69
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.then_block_73.<locals>.if_region_65.<locals>.else_block_68.<locals>.while_region_69.<locals>.while_after_block_70
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.then_block_73.<locals>.if_region_65.<locals>.else_block_68.<locals>.while_region_69.<locals>.while_before_block_70
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.then_block_73.<locals>.if_region_65.<locals>.then_block_68
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.then_block_73.<locals>.if_region_65.<locals>.then_block_68.<locals>.while_region_66
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.then_block_73.<locals>.if_region_65.<locals>.then_block_68.<locals>.while_region_66.<locals>.while_after_block_67
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.then_block_73.<locals>.if_region_65.<locals>.then_block_68.<locals>.while_region_66.<locals>.while_before_block_67
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.then_block_73.<locals>.if_region_71
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.then_block_73.<locals>.if_region_71.<locals>.else_block_72
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_64.<locals>.then_block_73.<locals>.if_region_71.<locals>.then_block_72
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_74
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_74.<locals>.else_block_75
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_74.<locals>.then_block_75
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_76
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_76.<locals>.else_block_81
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_76.<locals>.then_block_81
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_76.<locals>.then_block_81.<locals>.if_region_77
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_76.<locals>.then_block_81.<locals>.if_region_77.<locals>.else_block_78
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_76.<locals>.then_block_81.<locals>.if_region_77.<locals>.then_block_78
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_76.<locals>.then_block_81.<locals>.if_region_79
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_76.<locals>.then_block_81.<locals>.if_region_79.<locals>.else_block_80
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_76.<locals>.then_block_81.<locals>.if_region_79.<locals>.then_block_80
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_82
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_82.<locals>.then_block_83
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_84
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_84.<locals>.else_block_85
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.if_region_84.<locals>.then_block_85
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_32.<locals>.then_block_86.<locals>.ifexp_then_block_63
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_4
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_4.<locals>.else_block_7
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_4.<locals>.then_block_7
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.if_region_5
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.if_region_5.<locals>.else_block_6
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.if_region_5.<locals>.then_block_6
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_8
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_8.<locals>.else_block_11
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_8.<locals>.then_block_11
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_8.<locals>.then_block_11.<locals>.if_region_9
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_8.<locals>.then_block_11.<locals>.if_region_9.<locals>.then_block_10
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_87
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_87.<locals>.then_block_88
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_89
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_89.<locals>.else_block_90
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_89.<locals>.then_block_90
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_91
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_91.<locals>.else_block_95
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_91.<locals>.then_block_95
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_91.<locals>.then_block_95.<locals>.if_region_92
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_91.<locals>.then_block_95.<locals>.if_region_92.<locals>.else_block_93
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_91.<locals>.then_block_95.<locals>.if_region_92.<locals>.else_block_93.<locals>.elif_region_93
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_91.<locals>.then_block_95.<locals>.if_region_92.<locals>.else_block_93.<locals>.elif_region_93.<locals>.else_block_94
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_91.<locals>.then_block_95.<locals>.if_region_92.<locals>.else_block_93.<locals>.elif_region_93.<locals>.then_block_94
_glm_tp_allreduce_residual_rms_norm_out_quant_nvfp4_oneshot_lamport_split_row_kernel.<locals>.if_region_91.<locals>.then_block_95.<locals>.if_region_92.<locals>.then_block_93
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_0
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_0.<locals>.else_block_1
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_0.<locals>.then_block_1
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_2
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_2.<locals>.then_block_5
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3.<locals>.then_block_4
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_32
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_32.<locals>.then_block_33
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_8
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_8.<locals>.else_block_11
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_8.<locals>.then_block_11
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_8.<locals>.then_block_11.<locals>.while_region_9
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_8.<locals>.then_block_11.<locals>.while_region_9.<locals>.while_after_block_10
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.if_region_8.<locals>.then_block_11.<locals>.while_region_9.<locals>.while_before_block_10
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_14
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_14.<locals>.else_block_15
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_14.<locals>.then_block_15
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_16
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_16.<locals>.then_block_17
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_18
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_18.<locals>.else_block_23
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_18.<locals>.then_block_23
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_18.<locals>.then_block_23.<locals>.if_region_19
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_18.<locals>.then_block_23.<locals>.if_region_19.<locals>.else_block_22
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_18.<locals>.then_block_23.<locals>.if_region_19.<locals>.then_block_22
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_18.<locals>.then_block_23.<locals>.if_region_19.<locals>.then_block_22.<locals>.while_region_20
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_18.<locals>.then_block_23.<locals>.if_region_19.<locals>.then_block_22.<locals>.while_region_20.<locals>.while_after_block_21
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_18.<locals>.then_block_23.<locals>.if_region_19.<locals>.then_block_22.<locals>.while_region_20.<locals>.while_before_block_21
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_24
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_24.<locals>.else_block_31
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_24.<locals>.then_block_31
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_24.<locals>.then_block_31.<locals>.if_region_25
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_24.<locals>.then_block_31.<locals>.if_region_25.<locals>.else_block_26
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_24.<locals>.then_block_31.<locals>.if_region_25.<locals>.then_block_26
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_24.<locals>.then_block_31.<locals>.if_region_27
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_24.<locals>.then_block_31.<locals>.if_region_27.<locals>.else_block_30
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_24.<locals>.then_block_31.<locals>.if_region_27.<locals>.then_block_30
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_24.<locals>.then_block_31.<locals>.if_region_27.<locals>.then_block_30.<locals>.while_region_28
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_24.<locals>.then_block_31.<locals>.if_region_27.<locals>.then_block_30.<locals>.while_region_28.<locals>.while_after_block_29
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_after_block_13.<locals>.if_region_24.<locals>.then_block_31.<locals>.if_region_27.<locals>.then_block_30.<locals>.while_region_28.<locals>.while_before_block_29
_glm_tp_allreduce_residual_rms_norm_twoshot_push_kernel.<locals>.while_region_12.<locals>.while_before_block_13
_publish_cta_partial_to_cluster.<locals>.if_region_0
_publish_cta_partial_to_cluster.<locals>.if_region_0.<locals>.else_block_1
_publish_cta_partial_to_cluster.<locals>.if_region_0.<locals>.then_block_1
_sanitize_bf16x8.<locals>.if_region_0
_sanitize_bf16x8.<locals>.if_region_0.<locals>.else_block_1
_sanitize_bf16x8.<locals>.if_region_0.<locals>.then_block_1
_twoshot_block_epilogue.<locals>.if_region_0
_twoshot_block_epilogue.<locals>.if_region_0.<locals>.else_block_1
_twoshot_block_epilogue.<locals>.if_region_0.<locals>.then_block_1
_twoshot_block_epilogue.<locals>.if_region_2
_twoshot_block_epilogue.<locals>.if_region_2.<locals>.else_block_5
_twoshot_block_epilogue.<locals>.if_region_2.<locals>.then_block_5
_twoshot_block_epilogue.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3
_twoshot_block_epilogue.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3.<locals>.else_block_4
_twoshot_block_epilogue.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3.<locals>.then_block_4
_twoshot_block_epilogue.<locals>.if_region_6
_twoshot_block_epilogue.<locals>.if_region_6.<locals>.else_block_15
_twoshot_block_epilogue.<locals>.if_region_6.<locals>.then_block_15
_twoshot_push_partial.<locals>.if_region_0
_twoshot_push_partial.<locals>.if_region_0.<locals>.then_block_1
_twoshot_reduce_slice.<locals>.if_region_0
_twoshot_reduce_slice.<locals>.if_region_0.<locals>.else_block_5
_twoshot_reduce_slice.<locals>.if_region_0.<locals>.else_block_5.<locals>.while_region_6
_twoshot_reduce_slice.<locals>.if_region_0.<locals>.else_block_5.<locals>.while_region_6.<locals>.while_after_block_7
_twoshot_reduce_slice.<locals>.if_region_0.<locals>.else_block_5.<locals>.while_region_6.<locals>.while_before_block_7
_twoshot_reduce_slice.<locals>.if_region_0.<locals>.then_block_5
_twoshot_reduce_slice.<locals>.if_region_0.<locals>.then_block_5.<locals>.if_region_3
_twoshot_reduce_slice.<locals>.if_region_0.<locals>.then_block_5.<locals>.if_region_3.<locals>.else_block_4
_twoshot_reduce_slice.<locals>.if_region_0.<locals>.then_block_5.<locals>.if_region_3.<locals>.then_block_4
_twoshot_reduce_slice.<locals>.if_region_0.<locals>.then_block_5.<locals>.while_region_1
_twoshot_reduce_slice.<locals>.if_region_0.<locals>.then_block_5.<locals>.while_region_1.<locals>.while_after_block_2
_twoshot_reduce_slice.<locals>.if_region_0.<locals>.then_block_5.<locals>.while_region_1.<locals>.while_before_block_2
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/k3_latent_reduce.py`（13）

```text
<module>
k3_latent_reduce_kernel.<locals>.if_region_0
k3_latent_reduce_kernel.<locals>.if_region_0.<locals>.else_block_1
k3_latent_reduce_kernel.<locals>.if_region_0.<locals>.then_block_1
k3_latent_reduce_kernel.<locals>.if_region_4
k3_latent_reduce_kernel.<locals>.if_region_4.<locals>.else_block_7
k3_latent_reduce_kernel.<locals>.if_region_4.<locals>.then_block_7
k3_latent_reduce_kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.while_region_5
k3_latent_reduce_kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.while_region_5.<locals>.while_after_block_6
k3_latent_reduce_kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.while_region_5.<locals>.while_before_block_6
k3_latent_reduce_kernel.<locals>.while_region_2
k3_latent_reduce_kernel.<locals>.while_region_2.<locals>.while_after_block_3
k3_latent_reduce_kernel.<locals>.while_region_2.<locals>.while_before_block_3
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/k3_moe_front.py`（233）

```text
_flip.<locals>.while_region_0
_flip.<locals>.while_region_0.<locals>.while_after_block_1
_flip.<locals>.while_region_0.<locals>.while_before_block_1
_poll_logit_parts.<locals>.while_region_0
_poll_logit_parts.<locals>.while_region_0.<locals>.while_after_block_1
_poll_logit_parts.<locals>.while_region_0.<locals>.while_before_block_1
_top16_cta.<locals>.if_region_0
_top16_cta.<locals>.if_region_0.<locals>.else_block_1
_top16_cta.<locals>.if_region_0.<locals>.then_block_1
_top16_cta.<locals>.if_region_10
_top16_cta.<locals>.if_region_10.<locals>.else_block_13
_top16_cta.<locals>.if_region_10.<locals>.then_block_13
_top16_cta.<locals>.if_region_10.<locals>.then_block_13.<locals>.if_region_11
_top16_cta.<locals>.if_region_10.<locals>.then_block_13.<locals>.if_region_11.<locals>.else_block_12
_top16_cta.<locals>.if_region_10.<locals>.then_block_13.<locals>.if_region_11.<locals>.then_block_12
_top16_cta.<locals>.if_region_14
_top16_cta.<locals>.if_region_14.<locals>.else_block_17
_top16_cta.<locals>.if_region_14.<locals>.then_block_17
_top16_cta.<locals>.if_region_14.<locals>.then_block_17.<locals>.if_region_15
_top16_cta.<locals>.if_region_14.<locals>.then_block_17.<locals>.if_region_15.<locals>.else_block_16
_top16_cta.<locals>.if_region_14.<locals>.then_block_17.<locals>.if_region_15.<locals>.then_block_16
_top16_cta.<locals>.if_region_18
_top16_cta.<locals>.if_region_18.<locals>.else_block_23
_top16_cta.<locals>.if_region_18.<locals>.else_block_23.<locals>.elif_region_23
_top16_cta.<locals>.if_region_18.<locals>.else_block_23.<locals>.elif_region_23.<locals>.else_block_24
_top16_cta.<locals>.if_region_18.<locals>.else_block_23.<locals>.elif_region_23.<locals>.then_block_24
_top16_cta.<locals>.if_region_18.<locals>.then_block_23
_top16_cta.<locals>.if_region_18.<locals>.then_block_23.<locals>.if_region_19
_top16_cta.<locals>.if_region_18.<locals>.then_block_23.<locals>.if_region_19.<locals>.else_block_22
_top16_cta.<locals>.if_region_18.<locals>.then_block_23.<locals>.if_region_19.<locals>.then_block_22
_top16_cta.<locals>.if_region_18.<locals>.then_block_23.<locals>.if_region_19.<locals>.then_block_22.<locals>.if_region_20
_top16_cta.<locals>.if_region_18.<locals>.then_block_23.<locals>.if_region_19.<locals>.then_block_22.<locals>.if_region_20.<locals>.else_block_21
_top16_cta.<locals>.if_region_18.<locals>.then_block_23.<locals>.if_region_19.<locals>.then_block_22.<locals>.if_region_20.<locals>.then_block_21
_top16_cta.<locals>.if_region_2
_top16_cta.<locals>.if_region_2.<locals>.else_block_3
_top16_cta.<locals>.if_region_2.<locals>.then_block_3
_top16_cta.<locals>.if_region_4
_top16_cta.<locals>.if_region_4.<locals>.else_block_5
_top16_cta.<locals>.if_region_4.<locals>.then_block_5
_top16_cta.<locals>.if_region_6
_top16_cta.<locals>.if_region_6.<locals>.else_block_9
_top16_cta.<locals>.if_region_6.<locals>.then_block_9
_top16_cta.<locals>.if_region_6.<locals>.then_block_9.<locals>.if_region_7
_top16_cta.<locals>.if_region_6.<locals>.then_block_9.<locals>.if_region_7.<locals>.else_block_8
_top16_cta.<locals>.if_region_6.<locals>.then_block_9.<locals>.if_region_7.<locals>.then_block_8
k3_moe_front_kernel.<locals>.if_region_0
k3_moe_front_kernel.<locals>.if_region_0.<locals>.else_block_8
k3_moe_front_kernel.<locals>.if_region_0.<locals>.then_block_8
k3_moe_front_kernel.<locals>.if_region_0.<locals>.then_block_8.<locals>.if_region_1
k3_moe_front_kernel.<locals>.if_region_0.<locals>.then_block_8.<locals>.if_region_1.<locals>.else_block_4
k3_moe_front_kernel.<locals>.if_region_0.<locals>.then_block_8.<locals>.if_region_1.<locals>.then_block_4
k3_moe_front_kernel.<locals>.if_region_0.<locals>.then_block_8.<locals>.if_region_1.<locals>.then_block_4.<locals>.if_region_2
k3_moe_front_kernel.<locals>.if_region_0.<locals>.then_block_8.<locals>.if_region_1.<locals>.then_block_4.<locals>.if_region_2.<locals>.else_block_3
k3_moe_front_kernel.<locals>.if_region_0.<locals>.then_block_8.<locals>.if_region_1.<locals>.then_block_4.<locals>.if_region_2.<locals>.then_block_3
k3_moe_front_kernel.<locals>.if_region_0.<locals>.then_block_8.<locals>.if_region_5
k3_moe_front_kernel.<locals>.if_region_0.<locals>.then_block_8.<locals>.if_region_5.<locals>.then_block_7
k3_moe_front_kernel.<locals>.if_region_0.<locals>.then_block_8.<locals>.if_region_5.<locals>.then_block_7.<locals>.ifexp_else_block_6
k3_moe_front_kernel.<locals>.if_region_9
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.else_block_127
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_102
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_102.<locals>.else_block_103
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_102.<locals>.then_block_103
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_104
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_104.<locals>.else_block_107
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_104.<locals>.then_block_107
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_104.<locals>.then_block_107.<locals>.if_region_105
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_104.<locals>.then_block_107.<locals>.if_region_105.<locals>.then_block_106
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.else_block_122
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.else_block_122.<locals>.if_region_123
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.else_block_122.<locals>.if_region_123.<locals>.else_block_124
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.else_block_122.<locals>.if_region_123.<locals>.then_block_124
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.else_block_122.<locals>.if_region_125
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.else_block_122.<locals>.if_region_125.<locals>.else_block_126
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.else_block_122.<locals>.if_region_125.<locals>.then_block_126
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_110
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_110.<locals>.else_block_113
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_110.<locals>.then_block_113
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_110.<locals>.then_block_113.<locals>.if_region_111
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_110.<locals>.then_block_113.<locals>.if_region_111.<locals>.else_block_112
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_110.<locals>.then_block_113.<locals>.if_region_111.<locals>.then_block_112
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_114
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_114.<locals>.else_block_121
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_114.<locals>.then_block_121
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_114.<locals>.then_block_121.<locals>.if_region_115
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_114.<locals>.then_block_121.<locals>.if_region_115.<locals>.else_block_120
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_114.<locals>.then_block_121.<locals>.if_region_115.<locals>.then_block_120
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_114.<locals>.then_block_121.<locals>.if_region_115.<locals>.then_block_120.<locals>.if_region_116
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_114.<locals>.then_block_121.<locals>.if_region_115.<locals>.then_block_120.<locals>.if_region_116.<locals>.else_block_117
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_114.<locals>.then_block_121.<locals>.if_region_115.<locals>.then_block_120.<locals>.if_region_116.<locals>.then_block_117
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_114.<locals>.then_block_121.<locals>.if_region_115.<locals>.then_block_120.<locals>.if_region_118
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_114.<locals>.then_block_121.<locals>.if_region_115.<locals>.then_block_120.<locals>.if_region_118.<locals>.else_block_119
k3_moe_front_kernel.<locals>.if_region_9.<locals>.else_block_100.<locals>.if_region_101.<locals>.then_block_127.<locals>.if_region_108.<locals>.then_block_122.<locals>.loop_body_109.<locals>.if_region_114.<locals>.then_block_121.<locals>.if_region_115.<locals>.then_block_120.<locals>.if_region_118.<locals>.then_block_119
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.else_block_99
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.else_block_95
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.if_region_87
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.if_region_87.<locals>.else_block_90
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.if_region_87.<locals>.else_block_90.<locals>.if_region_91
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.if_region_87.<locals>.else_block_90.<locals>.if_region_91.<locals>.else_block_94
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.if_region_87.<locals>.else_block_90.<locals>.if_region_91.<locals>.then_block_94
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.if_region_87.<locals>.else_block_90.<locals>.if_region_91.<locals>.then_block_94.<locals>.if_region_92
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.if_region_87.<locals>.else_block_90.<locals>.if_region_91.<locals>.then_block_94.<locals>.if_region_92.<locals>.else_block_93
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.if_region_87.<locals>.else_block_90.<locals>.if_region_91.<locals>.then_block_94.<locals>.if_region_92.<locals>.then_block_93
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.if_region_87.<locals>.then_block_90
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.if_region_87.<locals>.then_block_90.<locals>.if_region_88
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.if_region_87.<locals>.then_block_90.<locals>.if_region_88.<locals>.else_block_89
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.if_region_87.<locals>.then_block_90.<locals>.if_region_88.<locals>.then_block_89
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.while_region_85
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.while_region_85.<locals>.while_after_block_86
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.else_block_84.<locals>.while_region_85.<locals>.while_before_block_86
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.else_block_83.<locals>.elif_region_83.<locals>.then_block_84
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.then_block_83
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.then_block_83.<locals>.if_region_77
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.then_block_83.<locals>.if_region_77.<locals>.else_block_82
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.then_block_83.<locals>.if_region_77.<locals>.then_block_82
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.then_block_83.<locals>.if_region_77.<locals>.then_block_82.<locals>.if_region_78
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.then_block_83.<locals>.if_region_77.<locals>.then_block_82.<locals>.if_region_78.<locals>.else_block_81
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.then_block_83.<locals>.if_region_77.<locals>.then_block_82.<locals>.if_region_78.<locals>.then_block_81
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.then_block_83.<locals>.if_region_77.<locals>.then_block_82.<locals>.if_region_78.<locals>.then_block_81.<locals>.if_region_79
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.then_block_83.<locals>.if_region_77.<locals>.then_block_82.<locals>.if_region_78.<locals>.then_block_81.<locals>.if_region_79.<locals>.else_block_80
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.if_region_76.<locals>.then_block_83.<locals>.if_region_77.<locals>.then_block_82.<locals>.if_region_78.<locals>.then_block_81.<locals>.if_region_79.<locals>.then_block_80
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.while_region_74
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.while_region_74.<locals>.while_after_block_75
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_73.<locals>.then_block_95.<locals>.while_region_74.<locals>.while_before_block_75
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_96
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_96.<locals>.then_block_98
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.if_region_96.<locals>.then_block_98.<locals>.ifexp_else_block_97
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.while_region_69
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.while_region_69.<locals>.while_after_block_70
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.while_region_69.<locals>.while_before_block_70
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.while_region_71
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.while_region_71.<locals>.while_after_block_72
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.else_block_68.<locals>.elif_region_68.<locals>.then_block_99.<locals>.while_region_71.<locals>.while_before_block_72
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_39
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_39.<locals>.else_block_45
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_39.<locals>.then_block_45
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_39.<locals>.then_block_45.<locals>.loop_body_40
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_39.<locals>.then_block_45.<locals>.loop_body_40.<locals>.if_region_43
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_39.<locals>.then_block_45.<locals>.loop_body_40.<locals>.if_region_43.<locals>.then_block_44
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_39.<locals>.then_block_45.<locals>.loop_body_40.<locals>.while_region_41
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_39.<locals>.then_block_45.<locals>.loop_body_40.<locals>.while_region_41.<locals>.while_after_block_42
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_39.<locals>.then_block_45.<locals>.loop_body_40.<locals>.while_region_41.<locals>.while_before_block_42
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.else_block_67
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.if_region_65
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.if_region_65.<locals>.else_block_66
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.if_region_65.<locals>.then_block_66
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.loop_body_56
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.loop_body_56.<locals>.if_region_59
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.loop_body_56.<locals>.if_region_59.<locals>.then_block_60
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.loop_body_56.<locals>.if_region_61
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.loop_body_56.<locals>.if_region_61.<locals>.else_block_62
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.loop_body_56.<locals>.if_region_61.<locals>.then_block_62
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.loop_body_56.<locals>.if_region_63
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.loop_body_56.<locals>.if_region_63.<locals>.else_block_64
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.loop_body_56.<locals>.if_region_63.<locals>.then_block_64
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.loop_body_56.<locals>.while_region_57
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.loop_body_56.<locals>.while_region_57.<locals>.while_after_block_58
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.else_block_54.<locals>.if_region_55.<locals>.then_block_67.<locals>.loop_body_56.<locals>.while_region_57.<locals>.while_before_block_58
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.then_block_54
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.then_block_54.<locals>.if_region_52
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.then_block_54.<locals>.if_region_52.<locals>.else_block_53
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.then_block_54.<locals>.if_region_52.<locals>.then_block_53
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.then_block_54.<locals>.loop_body_49
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.then_block_54.<locals>.loop_body_49.<locals>.if_region_50
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.if_region_48.<locals>.then_block_54.<locals>.loop_body_49.<locals>.if_region_50.<locals>.then_block_51
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.while_region_46
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.while_region_46.<locals>.while_after_block_47
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.else_block_38.<locals>.elif_region_38.<locals>.then_block_68.<locals>.while_region_46.<locals>.while_before_block_47
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.then_block_38
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.then_block_38.<locals>.if_region_35
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.then_block_38.<locals>.if_region_35.<locals>.else_block_37
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.then_block_38.<locals>.if_region_35.<locals>.then_block_37
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.else_block_34.<locals>.elif_region_34.<locals>.then_block_38.<locals>.if_region_35.<locals>.then_block_37.<locals>.loop_body_36
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_11
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_11.<locals>.else_block_18
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_11.<locals>.then_block_18
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_11.<locals>.then_block_18.<locals>.if_region_12
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_11.<locals>.then_block_18.<locals>.if_region_12.<locals>.else_block_13
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_11.<locals>.then_block_18.<locals>.if_region_12.<locals>.else_block_13.<locals>.if_region_14
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_11.<locals>.then_block_18.<locals>.if_region_12.<locals>.else_block_13.<locals>.if_region_14.<locals>.else_block_17
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_11.<locals>.then_block_18.<locals>.if_region_12.<locals>.else_block_13.<locals>.if_region_14.<locals>.then_block_17
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_11.<locals>.then_block_18.<locals>.if_region_12.<locals>.else_block_13.<locals>.if_region_14.<locals>.then_block_17.<locals>.ifexp_else_block_15
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_11.<locals>.then_block_18.<locals>.if_region_12.<locals>.else_block_13.<locals>.if_region_14.<locals>.then_block_17.<locals>.ifexp_then_block_15
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_11.<locals>.then_block_18.<locals>.if_region_12.<locals>.else_block_13.<locals>.if_region_14.<locals>.then_block_17.<locals>.loop_body_16
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_11.<locals>.then_block_18.<locals>.if_region_12.<locals>.then_block_13
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.else_block_33
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.else_block_32
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.else_block_31
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31.<locals>.loop_body_22
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31.<locals>.loop_body_22.<locals>.if_region_23
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31.<locals>.loop_body_22.<locals>.if_region_23.<locals>.else_block_26
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31.<locals>.loop_body_22.<locals>.if_region_23.<locals>.then_block_26
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31.<locals>.loop_body_22.<locals>.if_region_23.<locals>.then_block_26.<locals>.while_region_24
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31.<locals>.loop_body_22.<locals>.if_region_23.<locals>.then_block_26.<locals>.while_region_24.<locals>.while_after_block_25
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31.<locals>.loop_body_22.<locals>.if_region_23.<locals>.then_block_26.<locals>.while_region_24.<locals>.while_before_block_25
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31.<locals>.loop_body_22.<locals>.if_region_27
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31.<locals>.loop_body_22.<locals>.if_region_27.<locals>.else_block_30
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31.<locals>.loop_body_22.<locals>.if_region_27.<locals>.then_block_30
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31.<locals>.loop_body_22.<locals>.if_region_27.<locals>.then_block_30.<locals>.if_region_28
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31.<locals>.loop_body_22.<locals>.if_region_27.<locals>.then_block_30.<locals>.if_region_28.<locals>.else_block_29
k3_moe_front_kernel.<locals>.if_region_9.<locals>.then_block_100.<locals>.if_region_10.<locals>.then_block_34.<locals>.if_region_19.<locals>.then_block_33.<locals>.if_region_20.<locals>.then_block_32.<locals>.if_region_21.<locals>.then_block_31.<locals>.loop_body_22.<locals>.if_region_27.<locals>.then_block_30.<locals>.if_region_28.<locals>.then_block_29
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/k3_moe_kernel.py`（30）

```text
_emit_row.<locals>.if_region_0
_emit_row.<locals>.if_region_0.<locals>.else_block_1
_emit_row.<locals>.if_region_0.<locals>.then_block_1
_retire_claim.<locals>.if_region_0.<locals>.then_block_5.<locals>.if_region_1.<locals>.then_block_4.<locals>.if_region_2
_retire_claim.<locals>.if_region_0.<locals>.then_block_5.<locals>.if_region_1.<locals>.then_block_4.<locals>.if_region_2.<locals>.else_block_3
_retire_claim.<locals>.if_region_0.<locals>.then_block_5.<locals>.if_region_1.<locals>.then_block_4.<locals>.if_region_2.<locals>.then_block_3
k3_moe_kernel.<locals>.if_region_122.<locals>.then_block_126.<locals>.if_region_123
k3_moe_kernel.<locals>.if_region_122.<locals>.then_block_126.<locals>.if_region_123.<locals>.else_block_124
k3_moe_kernel.<locals>.if_region_122.<locals>.then_block_126.<locals>.if_region_123.<locals>.then_block_124
k3_moe_kernel.<locals>.if_region_16
k3_moe_kernel.<locals>.if_region_16.<locals>.else_block_17
k3_moe_kernel.<locals>.if_region_16.<locals>.then_block_17
k3_moe_kernel.<locals>.if_region_162.<locals>.then_block_205.<locals>.if_region_163
k3_moe_kernel.<locals>.if_region_162.<locals>.then_block_205.<locals>.if_region_163.<locals>.else_block_166
k3_moe_kernel.<locals>.if_region_162.<locals>.then_block_205.<locals>.if_region_163.<locals>.then_block_166
k3_moe_kernel.<locals>.if_region_162.<locals>.then_block_205.<locals>.if_region_163.<locals>.then_block_166.<locals>.while_region_164
k3_moe_kernel.<locals>.if_region_162.<locals>.then_block_205.<locals>.if_region_163.<locals>.then_block_166.<locals>.while_region_164.<locals>.while_after_block_165
k3_moe_kernel.<locals>.if_region_162.<locals>.then_block_205.<locals>.if_region_163.<locals>.then_block_166.<locals>.while_region_164.<locals>.while_before_block_165
k3_moe_kernel.<locals>.if_region_26.<locals>.else_block_27.<locals>.if_region_36
k3_moe_kernel.<locals>.if_region_26.<locals>.else_block_27.<locals>.if_region_36.<locals>.else_block_41
k3_moe_kernel.<locals>.if_region_26.<locals>.else_block_27.<locals>.if_region_36.<locals>.then_block_41
k3_moe_kernel.<locals>.if_region_26.<locals>.else_block_27.<locals>.if_region_36.<locals>.then_block_41.<locals>.if_region_37
k3_moe_kernel.<locals>.if_region_26.<locals>.else_block_27.<locals>.if_region_36.<locals>.then_block_41.<locals>.if_region_37.<locals>.else_block_40
k3_moe_kernel.<locals>.if_region_26.<locals>.else_block_27.<locals>.if_region_36.<locals>.then_block_41.<locals>.if_region_37.<locals>.then_block_40
k3_moe_kernel.<locals>.if_region_26.<locals>.else_block_27.<locals>.if_region_36.<locals>.then_block_41.<locals>.if_region_37.<locals>.then_block_40.<locals>.while_region_38
k3_moe_kernel.<locals>.if_region_26.<locals>.else_block_27.<locals>.if_region_36.<locals>.then_block_41.<locals>.if_region_37.<locals>.then_block_40.<locals>.while_region_38.<locals>.while_after_block_39
k3_moe_kernel.<locals>.if_region_26.<locals>.else_block_27.<locals>.if_region_36.<locals>.then_block_41.<locals>.if_region_37.<locals>.then_block_40.<locals>.while_region_38.<locals>.while_before_block_39
k3_moe_kernel.<locals>.if_region_26.<locals>.else_block_27.<locals>.if_region_42
k3_moe_kernel.<locals>.if_region_26.<locals>.else_block_27.<locals>.if_region_42.<locals>.else_block_43
k3_moe_kernel.<locals>.if_region_26.<locals>.else_block_27.<locals>.if_region_42.<locals>.then_block_43
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_fused_moe/k3_route_quant_ag.py`（3）

```text
_poll4.<locals>.while_region_0
_poll4.<locals>.while_region_0.<locals>.while_after_block_1
_poll4.<locals>.while_region_0.<locals>.while_before_block_1
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3_sandwich/k3_sandwich_kernel.py`（195）

```text
k3_sandwich_plain.<locals>.ifexp_else_block_0
k3_sandwich_plain.<locals>.ifexp_else_block_1
k3_sandwich_plain.<locals>.ifexp_else_block_2
k3_sandwich_plain.<locals>.ifexp_then_block_0
k3_sandwich_plain.<locals>.ifexp_then_block_1
k3_sandwich_plain.<locals>.ifexp_then_block_2
sandwich_role.<locals>.if_region_0
sandwich_role.<locals>.if_region_0.<locals>.else_block_6
sandwich_role.<locals>.if_region_0.<locals>.then_block_6
sandwich_role.<locals>.if_region_0.<locals>.then_block_6.<locals>.if_region_1
sandwich_role.<locals>.if_region_0.<locals>.then_block_6.<locals>.if_region_1.<locals>.else_block_5
sandwich_role.<locals>.if_region_0.<locals>.then_block_6.<locals>.if_region_1.<locals>.then_block_5
sandwich_role.<locals>.if_region_0.<locals>.then_block_6.<locals>.if_region_1.<locals>.then_block_5.<locals>.if_region_3
sandwich_role.<locals>.if_region_0.<locals>.then_block_6.<locals>.if_region_1.<locals>.then_block_5.<locals>.if_region_3.<locals>.else_block_4
sandwich_role.<locals>.if_region_0.<locals>.then_block_6.<locals>.if_region_1.<locals>.then_block_5.<locals>.if_region_3.<locals>.then_block_4
sandwich_role.<locals>.if_region_0.<locals>.then_block_6.<locals>.if_region_1.<locals>.then_block_5.<locals>.ifexp_else_block_2
sandwich_role.<locals>.if_region_11
sandwich_role.<locals>.if_region_11.<locals>.else_block_16
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.else_block_68
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.if_region_52
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.if_region_52.<locals>.else_block_53
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.if_region_52.<locals>.then_block_53
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.if_region_63
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.if_region_63.<locals>.then_block_65
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.if_region_63.<locals>.then_block_65.<locals>.ifexp_else_block_64
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.if_region_63.<locals>.then_block_65.<locals>.ifexp_then_block_64
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.if_region_66
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.if_region_66.<locals>.then_block_67
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.ifexp_else_block_60
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.ifexp_then_block_60
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.while_region_50
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.while_region_50.<locals>.while_after_block_51
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.while_region_50.<locals>.while_before_block_51
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.while_region_54
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.while_region_54.<locals>.while_after_block_55
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.while_region_54.<locals>.while_before_block_55
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.while_region_58
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.while_region_58.<locals>.while_after_block_59
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.while_region_58.<locals>.while_before_block_59
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.while_region_61
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.while_region_61.<locals>.while_after_block_62
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.else_block_49.<locals>.elif_region_49.<locals>.then_block_68.<locals>.while_region_61.<locals>.while_before_block_62
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_40
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_40.<locals>.else_block_41
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_40.<locals>.then_block_41
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_42
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_42.<locals>.else_block_45
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_42.<locals>.then_block_45
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_42.<locals>.then_block_45.<locals>.if_region_43
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_42.<locals>.then_block_45.<locals>.if_region_43.<locals>.else_block_44
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_42.<locals>.then_block_45.<locals>.if_region_43.<locals>.then_block_44
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_46
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_46.<locals>.else_block_48
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_46.<locals>.then_block_48
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_46.<locals>.then_block_48.<locals>.ifexp_else_block_47
sandwich_role.<locals>.if_region_11.<locals>.else_block_16.<locals>.elif_region_16.<locals>.then_block_49.<locals>.if_region_46.<locals>.then_block_48.<locals>.ifexp_then_block_47
sandwich_role.<locals>.if_region_11.<locals>.then_block_16
sandwich_role.<locals>.if_region_11.<locals>.then_block_16.<locals>.if_region_12
sandwich_role.<locals>.if_region_11.<locals>.then_block_16.<locals>.if_region_12.<locals>.else_block_15
sandwich_role.<locals>.if_region_11.<locals>.then_block_16.<locals>.if_region_12.<locals>.then_block_15
sandwich_role.<locals>.if_region_11.<locals>.then_block_16.<locals>.if_region_12.<locals>.then_block_15.<locals>.while_region_13
sandwich_role.<locals>.if_region_11.<locals>.then_block_16.<locals>.if_region_12.<locals>.then_block_15.<locals>.while_region_13.<locals>.while_after_block_14
sandwich_role.<locals>.if_region_11.<locals>.then_block_16.<locals>.if_region_12.<locals>.then_block_15.<locals>.while_region_13.<locals>.while_before_block_14
sandwich_role.<locals>.if_region_69
sandwich_role.<locals>.if_region_69.<locals>.else_block_151
sandwich_role.<locals>.if_region_69.<locals>.then_block_151
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_104
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_104.<locals>.else_block_105
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_104.<locals>.then_block_105
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_106
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_106.<locals>.else_block_107
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_106.<locals>.then_block_107
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_108
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_108.<locals>.then_block_109
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_110
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_110.<locals>.then_block_111
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_112
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_112.<locals>.else_block_113
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_112.<locals>.then_block_113
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.else_block_148
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_116
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_116.<locals>.else_block_117
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_116.<locals>.then_block_117
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_118
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_118.<locals>.else_block_121
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_118.<locals>.then_block_121
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_118.<locals>.then_block_121.<locals>.if_region_119
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_118.<locals>.then_block_121.<locals>.if_region_119.<locals>.else_block_120
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_118.<locals>.then_block_121.<locals>.if_region_119.<locals>.then_block_120
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_129
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_129.<locals>.else_block_136
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_129.<locals>.then_block_136
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_129.<locals>.then_block_136.<locals>.if_region_134
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_129.<locals>.then_block_136.<locals>.if_region_134.<locals>.else_block_135
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_129.<locals>.then_block_136.<locals>.if_region_134.<locals>.then_block_135
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_129.<locals>.then_block_136.<locals>.ifexp_else_block_130
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_129.<locals>.then_block_136.<locals>.ifexp_else_block_131
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_129.<locals>.then_block_136.<locals>.ifexp_else_block_132
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_129.<locals>.then_block_136.<locals>.ifexp_else_block_133
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_138
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_138.<locals>.else_block_139
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_138.<locals>.then_block_139
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_142
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_142.<locals>.else_block_143
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_142.<locals>.then_block_143
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_144
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_144.<locals>.else_block_145
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.if_region_144.<locals>.then_block_145
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.ifexp_else_block_115
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.ifexp_else_block_126
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.ifexp_else_block_137
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.ifexp_then_block_115
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.ifexp_then_block_126
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.ifexp_then_block_137
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_122
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_122.<locals>.while_after_block_123
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_122.<locals>.while_before_block_123
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_124
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_124.<locals>.while_after_block_125
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_124.<locals>.while_before_block_125
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_127
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_127.<locals>.while_after_block_128
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_127.<locals>.while_before_block_128
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_140
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_140.<locals>.while_after_block_141
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_140.<locals>.while_before_block_141
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_146
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_146.<locals>.while_after_block_147
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_114.<locals>.then_block_148.<locals>.while_region_146.<locals>.while_before_block_147
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_149
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_149.<locals>.else_block_150
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_149.<locals>.then_block_150
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_70
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_70.<locals>.else_block_73
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_70.<locals>.then_block_73
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_70.<locals>.then_block_73.<locals>.if_region_71
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_70.<locals>.then_block_73.<locals>.if_region_71.<locals>.else_block_72
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_70.<locals>.then_block_73.<locals>.if_region_71.<locals>.then_block_72
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_76
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_76.<locals>.else_block_77
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_76.<locals>.then_block_77
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_80
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_80.<locals>.else_block_81
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_80.<locals>.then_block_81
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_82
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_82.<locals>.else_block_87
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_82.<locals>.then_block_87
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_82.<locals>.then_block_87.<locals>.if_region_85
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_82.<locals>.then_block_87.<locals>.if_region_85.<locals>.else_block_86
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_82.<locals>.then_block_87.<locals>.if_region_85.<locals>.then_block_86
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_82.<locals>.then_block_87.<locals>.while_region_83
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_82.<locals>.then_block_87.<locals>.while_region_83.<locals>.while_after_block_84
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_82.<locals>.then_block_87.<locals>.while_region_83.<locals>.while_before_block_84
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_88
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_88.<locals>.else_block_97
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_88.<locals>.then_block_97
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_88.<locals>.then_block_97.<locals>.if_region_89
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_88.<locals>.then_block_97.<locals>.if_region_89.<locals>.else_block_96
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_88.<locals>.then_block_97.<locals>.if_region_89.<locals>.then_block_96
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_88.<locals>.then_block_97.<locals>.if_region_89.<locals>.then_block_96.<locals>.if_region_94
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_88.<locals>.then_block_97.<locals>.if_region_89.<locals>.then_block_96.<locals>.if_region_94.<locals>.else_block_95
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_88.<locals>.then_block_97.<locals>.if_region_89.<locals>.then_block_96.<locals>.if_region_94.<locals>.then_block_95
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_88.<locals>.then_block_97.<locals>.if_region_89.<locals>.then_block_96.<locals>.while_region_92
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_88.<locals>.then_block_97.<locals>.if_region_89.<locals>.then_block_96.<locals>.while_region_92.<locals>.while_after_block_93
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.if_region_88.<locals>.then_block_97.<locals>.if_region_89.<locals>.then_block_96.<locals>.while_region_92.<locals>.while_before_block_93
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_100
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_100.<locals>.while_after_block_101
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_100.<locals>.while_before_block_101
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_102
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_102.<locals>.while_after_block_103
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_102.<locals>.while_before_block_103
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_74
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_74.<locals>.while_after_block_75
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_74.<locals>.while_before_block_75
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_78
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_78.<locals>.while_after_block_79
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_78.<locals>.while_before_block_79
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_98
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_98.<locals>.while_after_block_99
sandwich_role.<locals>.if_region_69.<locals>.then_block_151.<locals>.while_region_98.<locals>.while_before_block_99
sandwich_role.<locals>.if_region_7
sandwich_role.<locals>.if_region_7.<locals>.then_block_8
sandwich_role.<locals>.if_region_9
sandwich_role.<locals>.if_region_9.<locals>.else_block_10
sandwich_role.<locals>.if_region_9.<locals>.then_block_10
swiglu_share.<locals>.while_region_0
swiglu_share.<locals>.while_region_0.<locals>.while_after_block_1
swiglu_share.<locals>.while_region_0.<locals>.while_before_block_1
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_comm/k3_lamport_ar.py`（48）

```text
__call__
clear_stage.<locals>.ifexp_else_block_0
clear_stage.<locals>.ifexp_else_block_1
clear_stage.<locals>.ifexp_else_block_2
clear_stage.<locals>.ifexp_else_block_3
clear_stage.<locals>.ifexp_then_block_0
clear_stage.<locals>.ifexp_then_block_1
clear_stage.<locals>.ifexp_then_block_2
clear_stage.<locals>.ifexp_then_block_3
clear_stage.<locals>.loop_body_4
clear_stage.<locals>.loop_body_4.<locals>.ifexp_else_block_5
clear_stage.<locals>.loop_body_4.<locals>.ifexp_then_block_5
kernel
kernel.<locals>.if_region_0
kernel.<locals>.if_region_0.<locals>.else_block_7
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_10
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_10.<locals>.then_block_11
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_12
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_12.<locals>.then_block_15
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_12.<locals>.then_block_15.<locals>.if_region_13
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_12.<locals>.then_block_15.<locals>.if_region_13.<locals>.then_block_14
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_16
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_16.<locals>.then_block_17
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_18
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_18.<locals>.then_block_21
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_18.<locals>.then_block_21.<locals>.while_region_19
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_18.<locals>.then_block_21.<locals>.while_region_19.<locals>.while_after_block_20
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_18.<locals>.then_block_21.<locals>.while_region_19.<locals>.while_before_block_20
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_8
kernel.<locals>.if_region_0.<locals>.else_block_7.<locals>.if_region_8.<locals>.then_block_9
kernel.<locals>.if_region_0.<locals>.then_block_1
kernel.<locals>.if_region_0.<locals>.then_block_7
kernel.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1
kernel.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6
kernel.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6.<locals>.ifexp_else_block_4
kernel.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6.<locals>.ifexp_else_block_5
kernel.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6.<locals>.ifexp_then_block_4
kernel.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6.<locals>.ifexp_then_block_5
kernel.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6.<locals>.while_region_2
kernel.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6.<locals>.while_region_2.<locals>.while_after_block_3
kernel.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6.<locals>.while_region_2.<locals>.while_before_block_3
poll_v4.<locals>.while_region_0
poll_v4.<locals>.while_region_0.<locals>.while_after_block_1
poll_v4.<locals>.while_region_0.<locals>.while_before_block_1
reduce_slots
reduce_slots.<locals>.while_region_0
reduce_slots.<locals>.while_region_0.<locals>.while_after_block_1
reduce_slots.<locals>.while_region_0.<locals>.while_before_block_1
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_gemm/k3_fp8w_lowm_gemm.py`（14）

```text
_push_lamport
_push_lamport.<locals>.if_region_0
_push_lamport.<locals>.if_region_0.<locals>.then_block_1
_push_staged_rows
_push_staged_rows.<locals>.ifexp_else_block_0
_push_staged_rows.<locals>.ifexp_else_block_1
_push_staged_rows.<locals>.ifexp_else_block_2
_push_staged_rows.<locals>.ifexp_else_block_3
_push_staged_rows.<locals>.loop_body_4
_push_staged_rows.<locals>.loop_body_4.<locals>.if_region_6
_push_staged_rows.<locals>.loop_body_4.<locals>.if_region_6.<locals>.then_block_7
_push_staged_rows.<locals>.loop_body_4.<locals>.ifexp_else_block_5
call_ar_push
call_ar_push_gated
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kernels/k3r_gemm/k3_fp8w_lowm_mxfp8.py`（10）

```text
_push
call_ar_push
epilogue.<locals>.if_region_4
epilogue.<locals>.if_region_4.<locals>.then_block_5
epilogue.<locals>.ifexp_else_block_11
epilogue.<locals>.ifexp_else_block_12
epilogue.<locals>.ifexp_else_block_13
epilogue.<locals>.ifexp_else_block_14
epilogue.<locals>.loop_body_15
epilogue.<locals>.loop_body_15.<locals>.ifexp_else_block_16
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/kernels/k3_markov/k3_markov_kernel.py`（67）

```text
<module>
k3_markov_kernel.<locals>.if_region_0
k3_markov_kernel.<locals>.if_region_0.<locals>.then_block_3
k3_markov_kernel.<locals>.if_region_0.<locals>.then_block_3.<locals>.if_region_1
k3_markov_kernel.<locals>.if_region_0.<locals>.then_block_3.<locals>.if_region_1.<locals>.then_block_2
k3_markov_kernel.<locals>.if_region_10
k3_markov_kernel.<locals>.if_region_10.<locals>.else_block_11
k3_markov_kernel.<locals>.if_region_10.<locals>.then_block_11
k3_markov_kernel.<locals>.if_region_14
k3_markov_kernel.<locals>.if_region_14.<locals>.else_block_19
k3_markov_kernel.<locals>.if_region_14.<locals>.then_block_19
k3_markov_kernel.<locals>.if_region_14.<locals>.then_block_19.<locals>.if_region_15
k3_markov_kernel.<locals>.if_region_14.<locals>.then_block_19.<locals>.if_region_15.<locals>.else_block_16
k3_markov_kernel.<locals>.if_region_14.<locals>.then_block_19.<locals>.if_region_15.<locals>.then_block_16
k3_markov_kernel.<locals>.if_region_14.<locals>.then_block_19.<locals>.if_region_17
k3_markov_kernel.<locals>.if_region_14.<locals>.then_block_19.<locals>.if_region_17.<locals>.else_block_18
k3_markov_kernel.<locals>.if_region_14.<locals>.then_block_19.<locals>.if_region_17.<locals>.then_block_18
k3_markov_kernel.<locals>.if_region_4
k3_markov_kernel.<locals>.if_region_4.<locals>.else_block_7
k3_markov_kernel.<locals>.if_region_4.<locals>.then_block_7
k3_markov_kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.if_region_5
k3_markov_kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.if_region_5.<locals>.else_block_6
k3_markov_kernel.<locals>.if_region_4.<locals>.then_block_7.<locals>.if_region_5.<locals>.then_block_6
k3_markov_kernel.<locals>.if_region_43
k3_markov_kernel.<locals>.if_region_43.<locals>.else_block_46
k3_markov_kernel.<locals>.if_region_43.<locals>.then_block_46
k3_markov_kernel.<locals>.if_region_43.<locals>.then_block_46.<locals>.if_region_44
k3_markov_kernel.<locals>.if_region_43.<locals>.then_block_46.<locals>.if_region_44.<locals>.else_block_45
k3_markov_kernel.<locals>.if_region_43.<locals>.then_block_46.<locals>.if_region_44.<locals>.then_block_45
k3_markov_kernel.<locals>.if_region_8
k3_markov_kernel.<locals>.if_region_8.<locals>.else_block_9
k3_markov_kernel.<locals>.if_region_8.<locals>.then_block_9
k3_markov_kernel.<locals>.loop_body_22
k3_markov_kernel.<locals>.loop_body_22.<locals>.if_region_30
k3_markov_kernel.<locals>.loop_body_22.<locals>.if_region_30.<locals>.else_block_31
k3_markov_kernel.<locals>.loop_body_22.<locals>.if_region_30.<locals>.then_block_31
k3_markov_kernel.<locals>.loop_body_22.<locals>.if_region_37
k3_markov_kernel.<locals>.loop_body_22.<locals>.if_region_37.<locals>.else_block_42
k3_markov_kernel.<locals>.loop_body_22.<locals>.if_region_37.<locals>.then_block_42
k3_markov_kernel.<locals>.loop_body_22.<locals>.if_region_37.<locals>.then_block_42.<locals>.if_region_38
k3_markov_kernel.<locals>.loop_body_22.<locals>.if_region_37.<locals>.then_block_42.<locals>.if_region_38.<locals>.then_block_39
k3_markov_kernel.<locals>.loop_body_22.<locals>.if_region_37.<locals>.then_block_42.<locals>.if_region_40
k3_markov_kernel.<locals>.loop_body_22.<locals>.if_region_37.<locals>.then_block_42.<locals>.if_region_40.<locals>.else_block_41
k3_markov_kernel.<locals>.loop_body_22.<locals>.if_region_37.<locals>.then_block_42.<locals>.if_region_40.<locals>.then_block_41
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_23
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_23.<locals>.if_region_26
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_23.<locals>.if_region_26.<locals>.else_block_27
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_23.<locals>.if_region_26.<locals>.then_block_27
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_23.<locals>.if_region_28
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_23.<locals>.if_region_28.<locals>.else_block_29
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_23.<locals>.if_region_28.<locals>.then_block_29
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_23.<locals>.while_region_24
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_23.<locals>.while_region_24.<locals>.while_after_block_25
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_23.<locals>.while_region_24.<locals>.while_before_block_25
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_32
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_32.<locals>.if_region_35
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_32.<locals>.if_region_35.<locals>.else_block_36
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_32.<locals>.if_region_35.<locals>.then_block_36
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_32.<locals>.while_region_33
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_32.<locals>.while_region_33.<locals>.while_after_block_34
k3_markov_kernel.<locals>.loop_body_22.<locals>.loop_body_32.<locals>.while_region_33.<locals>.while_before_block_34
k3_markov_kernel.<locals>.while_region_12
k3_markov_kernel.<locals>.while_region_12.<locals>.while_after_block_13
k3_markov_kernel.<locals>.while_region_12.<locals>.while_before_block_13
k3_markov_kernel.<locals>.while_region_20
k3_markov_kernel.<locals>.while_region_20.<locals>.while_after_block_21
k3_markov_kernel.<locals>.while_region_20.<locals>.while_before_block_21
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp4ep4/kernels/k3_spec_accept/k3_spec_accept_kernel.py`（12）

```text
k3_spec_accept_kernel.<locals>.if_region_18.<locals>.then_block_53.<locals>.if_region_29
k3_spec_accept_kernel.<locals>.if_region_18.<locals>.then_block_53.<locals>.if_region_29.<locals>.else_block_32
k3_spec_accept_kernel.<locals>.if_region_18.<locals>.then_block_53.<locals>.if_region_29.<locals>.then_block_32
k3_spec_accept_kernel.<locals>.if_region_18.<locals>.then_block_53.<locals>.if_region_29.<locals>.then_block_32.<locals>.while_region_30
k3_spec_accept_kernel.<locals>.if_region_18.<locals>.then_block_53.<locals>.if_region_29.<locals>.then_block_32.<locals>.while_region_30.<locals>.while_after_block_31
k3_spec_accept_kernel.<locals>.if_region_18.<locals>.then_block_53.<locals>.if_region_29.<locals>.then_block_32.<locals>.while_region_30.<locals>.while_before_block_31
k3_spec_accept_kernel.<locals>.if_region_18.<locals>.then_block_53.<locals>.if_region_33
k3_spec_accept_kernel.<locals>.if_region_18.<locals>.then_block_53.<locals>.if_region_33.<locals>.else_block_34
k3_spec_accept_kernel.<locals>.if_region_18.<locals>.then_block_53.<locals>.if_region_33.<locals>.then_block_34
k3_spec_accept_kernel.<locals>.if_region_18.<locals>.then_block_53.<locals>.while_region_27
k3_spec_accept_kernel.<locals>.if_region_18.<locals>.then_block_53.<locals>.while_region_27.<locals>.while_after_block_28
k3_spec_accept_kernel.<locals>.if_region_18.<locals>.then_block_53.<locals>.while_region_27.<locals>.while_before_block_28
```

### `tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl/ops/comm/k3_lamport_push.py`（2）

```text
<module>
K3LamportPush
```

### `tensorrt_llm/_torch/modeling_v2/models/minimax_m3/minimax_m3_nvfp4__sm_103__tp4/ops/m3_moe/m3_moe16_kernel.py`（471）

```text
<module>
_issue_fc1.<locals>.if_region_0
_issue_fc1.<locals>.if_region_0.<locals>.else_block_2
_issue_fc1.<locals>.if_region_0.<locals>.then_block_2
_issue_fc1.<locals>.if_region_0.<locals>.then_block_2.<locals>.ifexp_then_block_1
_issue_fc2.<locals>.if_region_0
_issue_fc2.<locals>.if_region_0.<locals>.else_block_4
_issue_fc2.<locals>.if_region_0.<locals>.then_block_4
_issue_fc2.<locals>.if_region_0.<locals>.then_block_4.<locals>.if_region_2
_issue_fc2.<locals>.if_region_0.<locals>.then_block_4.<locals>.if_region_2.<locals>.else_block_3
_issue_fc2.<locals>.if_region_0.<locals>.then_block_4.<locals>.if_region_2.<locals>.then_block_3
_issue_fc2.<locals>.if_region_0.<locals>.then_block_4.<locals>.ifexp_then_block_1
_issue_fc2.<locals>.if_region_5
_issue_fc2.<locals>.if_region_5.<locals>.else_block_6
_issue_fc2.<locals>.if_region_5.<locals>.then_block_6
_sh_issue.<locals>.if_region_2
_sh_issue.<locals>.if_region_2.<locals>.else_block_7
_sh_issue.<locals>.if_region_2.<locals>.else_block_7.<locals>.if_region_8
_sh_issue.<locals>.if_region_2.<locals>.else_block_7.<locals>.if_region_8.<locals>.else_block_9
_sh_issue.<locals>.if_region_2.<locals>.else_block_7.<locals>.if_region_8.<locals>.then_block_9
_sh_issue.<locals>.if_region_2.<locals>.then_block_7
_sh_issue.<locals>.if_region_2.<locals>.then_block_7.<locals>.if_region_3
_sh_issue.<locals>.if_region_2.<locals>.then_block_7.<locals>.if_region_3.<locals>.then_block_4
_sh_issue.<locals>.if_region_2.<locals>.then_block_7.<locals>.if_region_5
_sh_issue.<locals>.if_region_2.<locals>.then_block_7.<locals>.if_region_5.<locals>.else_block_6
_sh_issue.<locals>.if_region_2.<locals>.then_block_7.<locals>.if_region_5.<locals>.then_block_6
_shared_cta.<locals>.if_region_0
_shared_cta.<locals>.if_region_0.<locals>.else_block_5
_shared_cta.<locals>.if_region_0.<locals>.then_block_5
_shared_cta.<locals>.if_region_0.<locals>.then_block_5.<locals>.if_region_1
_shared_cta.<locals>.if_region_0.<locals>.then_block_5.<locals>.if_region_1.<locals>.else_block_2
_shared_cta.<locals>.if_region_0.<locals>.then_block_5.<locals>.if_region_1.<locals>.then_block_2
_shared_cta.<locals>.if_region_0.<locals>.then_block_5.<locals>.if_region_3
_shared_cta.<locals>.if_region_0.<locals>.then_block_5.<locals>.if_region_3.<locals>.then_block_4
_shared_cta.<locals>.if_region_14
_shared_cta.<locals>.if_region_14.<locals>.else_block_17
_shared_cta.<locals>.if_region_14.<locals>.then_block_17
_shared_cta.<locals>.if_region_14.<locals>.then_block_17.<locals>.if_region_15
_shared_cta.<locals>.if_region_14.<locals>.then_block_17.<locals>.if_region_15.<locals>.else_block_16
_shared_cta.<locals>.if_region_14.<locals>.then_block_17.<locals>.if_region_15.<locals>.then_block_16
_shared_cta.<locals>.if_region_20
_shared_cta.<locals>.if_region_20.<locals>.then_block_21
_shared_cta.<locals>.if_region_22
_shared_cta.<locals>.if_region_22.<locals>.else_block_25
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.else_block_41
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_27
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_27.<locals>.then_block_28
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_29
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_29.<locals>.else_block_30
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_29.<locals>.then_block_30
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_31
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_31.<locals>.else_block_40
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_31.<locals>.then_block_40
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_31.<locals>.then_block_40.<locals>.if_region_32
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_31.<locals>.then_block_40.<locals>.if_region_32.<locals>.else_block_35
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_31.<locals>.then_block_40.<locals>.if_region_32.<locals>.else_block_35.<locals>.if_region_38
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_31.<locals>.then_block_40.<locals>.if_region_32.<locals>.else_block_35.<locals>.if_region_38.<locals>.then_block_39
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_31.<locals>.then_block_40.<locals>.if_region_32.<locals>.else_block_35.<locals>.ifexp_else_block_36
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_31.<locals>.then_block_40.<locals>.if_region_32.<locals>.else_block_35.<locals>.ifexp_else_block_37
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_31.<locals>.then_block_40.<locals>.if_region_32.<locals>.then_block_35
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_31.<locals>.then_block_40.<locals>.if_region_32.<locals>.then_block_35.<locals>.if_region_33
_shared_cta.<locals>.if_region_22.<locals>.else_block_25.<locals>.elif_region_25.<locals>.then_block_41.<locals>.loop_body_26.<locals>.if_region_31.<locals>.then_block_40.<locals>.if_region_32.<locals>.then_block_35.<locals>.if_region_33.<locals>.then_block_34
_shared_cta.<locals>.if_region_22.<locals>.then_block_25
_shared_cta.<locals>.if_region_22.<locals>.then_block_25.<locals>.ifexp_then_block_23
_shared_cta.<locals>.if_region_22.<locals>.then_block_25.<locals>.loop_body_24
_shared_cta.<locals>.if_region_42
_shared_cta.<locals>.if_region_42.<locals>.else_block_45
_shared_cta.<locals>.if_region_42.<locals>.then_block_45
_shared_cta.<locals>.if_region_42.<locals>.then_block_45.<locals>.if_region_43
_shared_cta.<locals>.if_region_42.<locals>.then_block_45.<locals>.if_region_43.<locals>.else_block_44
_shared_cta.<locals>.if_region_42.<locals>.then_block_45.<locals>.if_region_43.<locals>.then_block_44
_shared_cta.<locals>.if_region_46
_shared_cta.<locals>.if_region_46.<locals>.else_block_67
_shared_cta.<locals>.if_region_46.<locals>.then_block_67
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.else_block_66
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_48
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_48.<locals>.then_block_49
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_50
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_50.<locals>.else_block_51
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_50.<locals>.then_block_51
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_52
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_52.<locals>.else_block_53
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_52.<locals>.then_block_53
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_54
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_54.<locals>.else_block_55
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_54.<locals>.then_block_55
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_56
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_56.<locals>.then_block_57
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_58
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_58.<locals>.else_block_61
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_58.<locals>.then_block_61
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_58.<locals>.then_block_61.<locals>.while_region_59
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_58.<locals>.then_block_61.<locals>.while_region_59.<locals>.while_after_block_60
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_58.<locals>.then_block_61.<locals>.while_region_59.<locals>.while_before_block_60
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_62
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_62.<locals>.then_block_63
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_64
_shared_cta.<locals>.if_region_46.<locals>.then_block_67.<locals>.if_region_47.<locals>.then_block_66.<locals>.if_region_64.<locals>.then_block_65
_shared_cta.<locals>.if_region_6
_shared_cta.<locals>.if_region_6.<locals>.then_block_7
_shared_cta.<locals>.if_region_68
_shared_cta.<locals>.if_region_68.<locals>.then_block_69
_shared_cta.<locals>.if_region_70
_shared_cta.<locals>.if_region_70.<locals>.else_block_71
_shared_cta.<locals>.if_region_70.<locals>.then_block_71
_shared_cta.<locals>.if_region_8
_shared_cta.<locals>.if_region_8.<locals>.else_block_13
_shared_cta.<locals>.if_region_8.<locals>.then_block_13
_wait.<locals>.while_region_0
_wait.<locals>.while_region_0.<locals>.while_after_block_1
_wait.<locals>.while_region_0.<locals>.while_before_block_1
_waitd.<locals>.while_region_4
_waitd.<locals>.while_region_4.<locals>.while_after_block_5
_waitd.<locals>.while_region_4.<locals>.while_before_block_5
_waitt.<locals>.ifexp_else_block_0
_waitt.<locals>.while_region_1
_waitt.<locals>.while_region_1.<locals>.while_after_block_2
_waitt.<locals>.while_region_1.<locals>.while_before_block_2
m3_moe_kernel.<locals>.if_region_2
m3_moe_kernel.<locals>.if_region_2.<locals>.else_block_3
m3_moe_kernel.<locals>.if_region_2.<locals>.then_block_3
m3_moe_kernel.<locals>.if_region_5
m3_moe_kernel.<locals>.if_region_5.<locals>.else_block_6
m3_moe_kernel.<locals>.if_region_5.<locals>.then_block_6
m3_moe_kernel.<locals>.if_region_7
m3_moe_kernel.<locals>.if_region_7.<locals>.else_block_328
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_108
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_108.<locals>.else_block_115
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_108.<locals>.then_block_115
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_108.<locals>.then_block_115.<locals>.if_region_113
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_108.<locals>.then_block_115.<locals>.if_region_113.<locals>.else_block_114
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_108.<locals>.then_block_115.<locals>.if_region_113.<locals>.then_block_114
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_108.<locals>.then_block_115.<locals>.loop_body_109
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_108.<locals>.then_block_115.<locals>.loop_body_109.<locals>.loop_body_110
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_108.<locals>.then_block_115.<locals>.loop_body_109.<locals>.loop_body_110.<locals>.if_region_111
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_108.<locals>.then_block_115.<locals>.loop_body_109.<locals>.loop_body_110.<locals>.if_region_111.<locals>.else_block_112
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_108.<locals>.then_block_115.<locals>.loop_body_109.<locals>.loop_body_110.<locals>.if_region_111.<locals>.then_block_112
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.else_block_142
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_117
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_117.<locals>.else_block_118
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_117.<locals>.then_block_118
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_119
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_119.<locals>.else_block_122
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_119.<locals>.then_block_122
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_119.<locals>.then_block_122.<locals>.if_region_120
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_119.<locals>.then_block_122.<locals>.if_region_120.<locals>.else_block_121
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_119.<locals>.then_block_122.<locals>.if_region_120.<locals>.then_block_121
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_123
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_123.<locals>.else_block_124
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_123.<locals>.then_block_124
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_126
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_126.<locals>.else_block_127
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_126.<locals>.then_block_127
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_128
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_128.<locals>.else_block_129
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_128.<locals>.then_block_129
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_130
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_130.<locals>.else_block_131
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_130.<locals>.then_block_131
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_132
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_132.<locals>.else_block_133
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_132.<locals>.then_block_133
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_134
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_134.<locals>.else_block_135
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_134.<locals>.then_block_135
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_136
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_136.<locals>.then_block_137
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_138
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_138.<locals>.else_block_141
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_138.<locals>.then_block_141
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_138.<locals>.then_block_141.<locals>.if_region_139
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_138.<locals>.then_block_141.<locals>.if_region_139.<locals>.else_block_140
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.if_region_138.<locals>.then_block_141.<locals>.if_region_139.<locals>.then_block_140
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.ifexp_else_block_125
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_116.<locals>.then_block_142.<locals>.ifexp_then_block_125
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_143
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_143.<locals>.then_block_144
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_151
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_151.<locals>.else_block_152
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_151.<locals>.then_block_152
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.else_block_170
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.if_region_168
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.if_region_168.<locals>.then_block_169
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.loop_body_154
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.loop_body_154.<locals>.if_region_155
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.loop_body_154.<locals>.if_region_155.<locals>.else_block_162
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.loop_body_154.<locals>.if_region_155.<locals>.then_block_162
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.loop_body_154.<locals>.if_region_155.<locals>.then_block_162.<locals>.loop_body_156
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.loop_body_154.<locals>.if_region_155.<locals>.then_block_162.<locals>.loop_body_156.<locals>.loop_body_157
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.loop_body_154.<locals>.if_region_155.<locals>.then_block_162.<locals>.loop_body_156.<locals>.loop_body_157.<locals>.if_region_158
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.loop_body_154.<locals>.if_region_155.<locals>.then_block_162.<locals>.loop_body_156.<locals>.loop_body_157.<locals>.if_region_158.<locals>.else_block_159
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.loop_body_154.<locals>.if_region_155.<locals>.then_block_162.<locals>.loop_body_156.<locals>.loop_body_157.<locals>.if_region_158.<locals>.then_block_159
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.loop_body_154.<locals>.if_region_163
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.loop_body_154.<locals>.if_region_163.<locals>.else_block_167
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.loop_body_154.<locals>.if_region_163.<locals>.then_block_167
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_153.<locals>.then_block_170.<locals>.loop_body_154.<locals>.if_region_163.<locals>.then_block_167.<locals>.loop_body_164
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.else_block_208
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_184
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_184.<locals>.else_block_185
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_184.<locals>.then_block_185
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_186
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_186.<locals>.then_block_187
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_188
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_188.<locals>.then_block_191
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_188.<locals>.then_block_191.<locals>.while_region_189
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_188.<locals>.then_block_191.<locals>.while_region_189.<locals>.while_after_block_190
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_188.<locals>.then_block_191.<locals>.while_region_189.<locals>.while_before_block_190
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_192
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_192.<locals>.then_block_193
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_194
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_194.<locals>.else_block_195
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_194.<locals>.then_block_195
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_196
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_196.<locals>.else_block_197
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_196.<locals>.then_block_197
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_198
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_198.<locals>.then_block_199
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_200
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_200.<locals>.else_block_201
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_200.<locals>.then_block_201
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_202
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_202.<locals>.then_block_203
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_204
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_204.<locals>.else_block_205
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_204.<locals>.then_block_205
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_206
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_182.<locals>.then_block_208.<locals>.loop_body_183.<locals>.if_region_206.<locals>.then_block_207
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_23
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_23.<locals>.else_block_24
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_23.<locals>.then_block_24
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_233
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_233.<locals>.else_block_240
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_233.<locals>.then_block_240
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_233.<locals>.then_block_240.<locals>.loop_body_234
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_233.<locals>.then_block_240.<locals>.loop_body_234.<locals>.if_region_236
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_233.<locals>.then_block_240.<locals>.loop_body_234.<locals>.if_region_236.<locals>.else_block_237
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_233.<locals>.then_block_240.<locals>.loop_body_234.<locals>.if_region_236.<locals>.then_block_237
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_233.<locals>.then_block_240.<locals>.loop_body_234.<locals>.if_region_238
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_233.<locals>.then_block_240.<locals>.loop_body_234.<locals>.if_region_238.<locals>.then_block_239
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_233.<locals>.then_block_240.<locals>.loop_body_234.<locals>.loop_body_235
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.else_block_266
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.if_region_264
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.if_region_264.<locals>.then_block_265
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_243
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_243.<locals>.else_block_250
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_243.<locals>.then_block_250
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_243.<locals>.then_block_250.<locals>.loop_body_244
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_243.<locals>.then_block_250.<locals>.loop_body_244.<locals>.if_region_248
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_243.<locals>.then_block_250.<locals>.loop_body_244.<locals>.if_region_248.<locals>.else_block_249
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_243.<locals>.then_block_250.<locals>.loop_body_244.<locals>.if_region_248.<locals>.then_block_249
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_243.<locals>.then_block_250.<locals>.loop_body_244.<locals>.loop_body_245
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_243.<locals>.then_block_250.<locals>.loop_body_244.<locals>.loop_body_245.<locals>.if_region_246
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_243.<locals>.then_block_250.<locals>.loop_body_244.<locals>.loop_body_245.<locals>.if_region_246.<locals>.else_block_247
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_243.<locals>.then_block_250.<locals>.loop_body_244.<locals>.loop_body_245.<locals>.if_region_246.<locals>.then_block_247
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.else_block_263
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.then_block_263
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.then_block_263.<locals>.if_region_254
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.then_block_263.<locals>.if_region_254.<locals>.then_block_255
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.then_block_263.<locals>.if_region_261
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.then_block_263.<locals>.if_region_261.<locals>.else_block_262
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.then_block_263.<locals>.if_region_261.<locals>.then_block_262
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.then_block_263.<locals>.loop_body_256
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.then_block_263.<locals>.loop_body_256.<locals>.if_region_257
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.then_block_263.<locals>.loop_body_256.<locals>.if_region_257.<locals>.else_block_258
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.then_block_263.<locals>.loop_body_256.<locals>.if_region_257.<locals>.then_block_258
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.then_block_263.<locals>.loop_body_256.<locals>.if_region_259
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.then_block_263.<locals>.loop_body_256.<locals>.if_region_259.<locals>.else_block_260
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_241.<locals>.then_block_266.<locals>.loop_body_242.<locals>.if_region_251.<locals>.then_block_263.<locals>.loop_body_256.<locals>.if_region_259.<locals>.then_block_260
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_25
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_25.<locals>.else_block_28
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_25.<locals>.then_block_28
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_25.<locals>.then_block_28.<locals>.if_region_26
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_25.<locals>.then_block_28.<locals>.if_region_26.<locals>.else_block_27
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_25.<locals>.then_block_28.<locals>.if_region_26.<locals>.then_block_27
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.else_block_312
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.if_region_268
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.if_region_268.<locals>.else_block_271
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.if_region_268.<locals>.then_block_271
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.if_region_268.<locals>.then_block_271.<locals>.if_region_269
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.if_region_268.<locals>.then_block_271.<locals>.if_region_269.<locals>.else_block_270
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.if_region_268.<locals>.then_block_271.<locals>.if_region_269.<locals>.then_block_270
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.if_region_310
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.if_region_310.<locals>.then_block_311
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.else_block_300
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.if_region_298
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.if_region_298.<locals>.then_block_299
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_281
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_281.<locals>.else_block_286
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_281.<locals>.then_block_286
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_281.<locals>.then_block_286.<locals>.if_region_282
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_281.<locals>.then_block_286.<locals>.if_region_282.<locals>.else_block_285
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_281.<locals>.then_block_286.<locals>.if_region_282.<locals>.then_block_285
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_281.<locals>.then_block_286.<locals>.if_region_282.<locals>.then_block_285.<locals>.while_region_283
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_281.<locals>.then_block_286.<locals>.if_region_282.<locals>.then_block_285.<locals>.while_region_283.<locals>.while_after_block_284
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_281.<locals>.then_block_286.<locals>.if_region_282.<locals>.then_block_285.<locals>.while_region_283.<locals>.while_before_block_284
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_287
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_287.<locals>.else_block_290
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_287.<locals>.then_block_290
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_287.<locals>.then_block_290.<locals>.if_region_288
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_287.<locals>.then_block_290.<locals>.if_region_288.<locals>.else_block_289
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.else_block_280.<locals>.if_region_287.<locals>.then_block_290.<locals>.if_region_288.<locals>.then_block_289
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.then_block_280
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.then_block_280.<locals>.if_region_278
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.then_block_280.<locals>.if_region_278.<locals>.else_block_279
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.if_region_277.<locals>.then_block_280.<locals>.if_region_278.<locals>.then_block_279
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.ifexp_else_block_275
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.ifexp_else_block_276
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_273.<locals>.then_block_300.<locals>.loop_body_274.<locals>.ifexp_else_block_291
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_301
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_301.<locals>.else_block_309
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_301.<locals>.then_block_309
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_301.<locals>.then_block_309.<locals>.loop_body_302
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_301.<locals>.then_block_309.<locals>.loop_body_302.<locals>.if_region_305
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_301.<locals>.then_block_309.<locals>.loop_body_302.<locals>.if_region_305.<locals>.else_block_308
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_301.<locals>.then_block_309.<locals>.loop_body_302.<locals>.if_region_305.<locals>.then_block_308
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_301.<locals>.then_block_309.<locals>.loop_body_302.<locals>.if_region_305.<locals>.then_block_308.<locals>.if_region_306
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_301.<locals>.then_block_309.<locals>.loop_body_302.<locals>.if_region_305.<locals>.then_block_308.<locals>.if_region_306.<locals>.else_block_307
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_267.<locals>.then_block_312.<locals>.loop_body_272.<locals>.if_region_301.<locals>.then_block_309.<locals>.loop_body_302.<locals>.if_region_305.<locals>.then_block_308.<locals>.if_region_306.<locals>.then_block_307
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_29
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_29.<locals>.then_block_30
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_31
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_31.<locals>.else_block_34
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_31.<locals>.then_block_34
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_31.<locals>.then_block_34.<locals>.if_region_32
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_31.<locals>.then_block_34.<locals>.if_region_32.<locals>.else_block_33
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_31.<locals>.then_block_34.<locals>.if_region_32.<locals>.then_block_33
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_314
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_314.<locals>.then_block_315
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_316
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_316.<locals>.else_block_321
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_316.<locals>.then_block_321
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_316.<locals>.then_block_321.<locals>.if_region_319
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_316.<locals>.then_block_321.<locals>.if_region_319.<locals>.then_block_320
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_316.<locals>.then_block_321.<locals>.while_region_317
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_316.<locals>.then_block_321.<locals>.while_region_317.<locals>.while_after_block_318
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_316.<locals>.then_block_321.<locals>.while_region_317.<locals>.while_before_block_318
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_322
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_322.<locals>.else_block_323
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_322.<locals>.then_block_323
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_324
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_324.<locals>.then_block_325
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_326
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_326.<locals>.else_block_327
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_326.<locals>.then_block_327
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_35
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_35.<locals>.else_block_44
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_35.<locals>.then_block_44
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_35.<locals>.then_block_44.<locals>.if_region_40
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_35.<locals>.then_block_44.<locals>.if_region_40.<locals>.else_block_41
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_35.<locals>.then_block_44.<locals>.if_region_40.<locals>.then_block_41
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_35.<locals>.then_block_44.<locals>.if_region_42
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_35.<locals>.then_block_44.<locals>.if_region_42.<locals>.else_block_43
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_35.<locals>.then_block_44.<locals>.if_region_42.<locals>.then_block_43
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_47
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_47.<locals>.then_block_48
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_66
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_66.<locals>.then_block_67
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_68
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_68.<locals>.else_block_69
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_68.<locals>.then_block_69
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_70
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_70.<locals>.else_block_71
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_70.<locals>.then_block_71
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_72
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_72.<locals>.then_block_73
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_74
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_74.<locals>.else_block_75
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_74.<locals>.then_block_75
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_76
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_76.<locals>.then_block_77
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_78
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_78.<locals>.else_block_79
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_78.<locals>.then_block_79
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.else_block_22
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_12
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_12.<locals>.else_block_13
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_12.<locals>.then_block_13
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_14
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_14.<locals>.else_block_15
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_14.<locals>.then_block_15
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_16
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_16.<locals>.else_block_17
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_16.<locals>.then_block_17
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_18
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_18.<locals>.else_block_19
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_18.<locals>.then_block_19
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_20
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_20.<locals>.else_block_21
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_20.<locals>.then_block_21
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_9
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_9.<locals>.else_block_11
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_9.<locals>.then_block_11
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_8.<locals>.then_block_22.<locals>.if_region_9.<locals>.then_block_11.<locals>.ifexp_then_block_10
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_80
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_80.<locals>.else_block_85
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_80.<locals>.then_block_85
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_80.<locals>.then_block_85.<locals>.if_region_81
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_80.<locals>.then_block_85.<locals>.if_region_81.<locals>.else_block_84
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_80.<locals>.then_block_85.<locals>.if_region_81.<locals>.then_block_84
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_80.<locals>.then_block_85.<locals>.if_region_81.<locals>.then_block_84.<locals>.while_region_82
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_80.<locals>.then_block_85.<locals>.if_region_81.<locals>.then_block_84.<locals>.while_region_82.<locals>.while_after_block_83
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_80.<locals>.then_block_85.<locals>.if_region_81.<locals>.then_block_84.<locals>.while_region_82.<locals>.while_before_block_83
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.else_block_107
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_103
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_103.<locals>.else_block_106
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_103.<locals>.then_block_106
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_103.<locals>.then_block_106.<locals>.if_region_104
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_103.<locals>.then_block_106.<locals>.if_region_104.<locals>.else_block_105
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_103.<locals>.then_block_106.<locals>.if_region_104.<locals>.then_block_105
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_91
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_91.<locals>.then_block_92
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_93
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_93.<locals>.else_block_96
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_93.<locals>.then_block_96
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_93.<locals>.then_block_96.<locals>.if_region_94
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_93.<locals>.then_block_96.<locals>.if_region_94.<locals>.else_block_95
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_93.<locals>.then_block_96.<locals>.if_region_94.<locals>.then_block_95
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_97
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.if_region_97.<locals>.then_block_98
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.while_region_87
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.while_region_87.<locals>.while_after_block_88
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.while_region_87.<locals>.while_after_block_88.<locals>.if_region_89
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.while_region_87.<locals>.while_after_block_88.<locals>.if_region_89.<locals>.else_block_90
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.while_region_87.<locals>.while_after_block_88.<locals>.if_region_89.<locals>.then_block_90
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.if_region_86.<locals>.then_block_107.<locals>.while_region_87.<locals>.while_before_block_88
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.ifexp_then_block_313
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_56
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_56.<locals>.then_block_57
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_58
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_58.<locals>.else_block_65
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_58.<locals>.then_block_65
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_58.<locals>.then_block_65.<locals>.if_region_59
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_58.<locals>.then_block_65.<locals>.if_region_59.<locals>.else_block_64
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_58.<locals>.then_block_65.<locals>.if_region_59.<locals>.then_block_64
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_58.<locals>.then_block_65.<locals>.if_region_59.<locals>.then_block_64.<locals>.if_region_60
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_58.<locals>.then_block_65.<locals>.if_region_59.<locals>.then_block_64.<locals>.if_region_60.<locals>.else_block_61
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_58.<locals>.then_block_65.<locals>.if_region_59.<locals>.then_block_64.<locals>.if_region_60.<locals>.then_block_61
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_58.<locals>.then_block_65.<locals>.if_region_59.<locals>.then_block_64.<locals>.if_region_62
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_58.<locals>.then_block_65.<locals>.if_region_59.<locals>.then_block_64.<locals>.if_region_62.<locals>.else_block_63
m3_moe_kernel.<locals>.if_region_7.<locals>.then_block_328.<locals>.loop_body_55.<locals>.if_region_58.<locals>.then_block_65.<locals>.if_region_59.<locals>.then_block_64.<locals>.if_region_62.<locals>.then_block_63
m3_moe_kernel.<locals>.ifexp_then_block_4
```

### `tensorrt_llm/_torch/models/musespark_open_boundary.py`（9）

```text
__call__
kernel
kernel.<locals>.if_region_0
kernel.<locals>.if_region_0.<locals>.then_block_1
load
reduce
reduce.<locals>.if_region_0
reduce.<locals>.if_region_0.<locals>.then_block_1
reduce.<locals>.loop_body_4
```

### `tensorrt_llm/_torch/models/musespark_open_finalize_ar.py`（24）

```text
__call__
kernel
kernel.<locals>.if_region_0
kernel.<locals>.if_region_0.<locals>.else_block_15
kernel.<locals>.if_region_0.<locals>.then_block_15
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_1
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_1.<locals>.else_block_4
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_1.<locals>.then_block_4
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_1.<locals>.then_block_4.<locals>.if_region_2
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_1.<locals>.then_block_4.<locals>.if_region_2.<locals>.then_block_3
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_11
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_11.<locals>.else_block_14
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_11.<locals>.then_block_14
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_11.<locals>.then_block_14.<locals>.while_region_12
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_11.<locals>.then_block_14.<locals>.while_region_12.<locals>.while_after_block_13
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_11.<locals>.then_block_14.<locals>.while_region_12.<locals>.while_before_block_13
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_5
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_5.<locals>.else_block_6
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_5.<locals>.then_block_6
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_7
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_7.<locals>.else_block_8
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_7.<locals>.then_block_8
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_9
kernel.<locals>.if_region_0.<locals>.then_block_15.<locals>.if_region_9.<locals>.then_block_10
```

### `tensorrt_llm/_torch/modules/engram/_evict_page_cache.py`（1）

```text
<module>
```

### `tensorrt_llm/_torch/modules/gated_mlp.py`（1）

```text
GatedMLP.torch_dynamo_resume_in_forward_at_583
```

### `tensorrt_llm/_torch/modules/linear.py`（1）

```text
Linear.torch_dynamo_resume_in_forward_at_5073
```

### `tensorrt_llm/_torch/modules/mlp.py`（1）

```text
MLP.torch_dynamo_resume_in_forward_at_257
```

### `tensorrt_llm/_torch/modules/q35_mtp_glue.py`（1）

```text
<module>
```

### `tensorrt_llm/_torch/moe/kernels/blackwell/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py`（7）

```text
kernel.<locals>.if_region_7.<locals>.then_block_25.<locals>.loop_body_22
kernel.<locals>.if_region_7.<locals>.then_block_25.<locals>.loop_body_22.<locals>.if_region_23
kernel.<locals>.if_region_7.<locals>.then_block_25.<locals>.loop_body_22.<locals>.if_region_23.<locals>.then_block_24
kernel.<locals>.if_region_7.<locals>.then_block_25.<locals>.while_region_15.<locals>.while_after_block_16.<locals>.if_region_17.<locals>.then_block_21.<locals>.loop_body_18
kernel.<locals>.if_region_7.<locals>.then_block_25.<locals>.while_region_15.<locals>.while_after_block_16.<locals>.if_region_17.<locals>.then_block_21.<locals>.loop_body_18.<locals>.if_region_19
kernel.<locals>.if_region_7.<locals>.then_block_25.<locals>.while_region_15.<locals>.while_after_block_16.<locals>.if_region_17.<locals>.then_block_21.<locals>.loop_body_18.<locals>.if_region_19.<locals>.then_block_20
kernel.<locals>.if_region_91.<locals>.then_block_111.<locals>.while_region_92.<locals>.while_after_block_93.<locals>.loop_body_100.<locals>.ifexp_else_block_104
```

### `tensorrt_llm/_torch/moe/kernels/blackwell/dsv4_route_quant.py`（80）

```text
clear_tail
clear_tail.<locals>.if_region_0
clear_tail.<locals>.if_region_0.<locals>.else_block_3
clear_tail.<locals>.if_region_0.<locals>.then_block_3
clear_tail.<locals>.if_region_0.<locals>.then_block_3.<locals>.if_region_1
clear_tail.<locals>.if_region_0.<locals>.then_block_3.<locals>.if_region_1.<locals>.else_block_2
clear_tail.<locals>.if_region_0.<locals>.then_block_3.<locals>.if_region_1.<locals>.then_block_2
counts_row
counts_row.<locals>.if_region_0
counts_row.<locals>.if_region_0.<locals>.else_block_1
counts_row.<locals>.if_region_0.<locals>.then_block_1
counts_row.<locals>.if_region_2
counts_row.<locals>.if_region_2.<locals>.else_block_5
counts_row.<locals>.if_region_2.<locals>.then_block_5
counts_row.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3
counts_row.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3.<locals>.else_block_4
counts_row.<locals>.if_region_2.<locals>.then_block_5.<locals>.if_region_3.<locals>.then_block_4
counts_row.<locals>.if_region_6
counts_row.<locals>.if_region_6.<locals>.else_block_9
counts_row.<locals>.if_region_6.<locals>.then_block_9
counts_row.<locals>.if_region_6.<locals>.then_block_9.<locals>.if_region_7
counts_row.<locals>.if_region_6.<locals>.then_block_9.<locals>.if_region_7.<locals>.else_block_8
counts_row.<locals>.if_region_6.<locals>.then_block_9.<locals>.if_region_7.<locals>.then_block_8
home_first_owner
home_first_owner.<locals>.if_region_0
home_first_owner.<locals>.if_region_0.<locals>.else_block_1
home_first_owner.<locals>.if_region_0.<locals>.then_block_1
home_first_owner.<locals>.if_region_2
home_first_owner.<locals>.if_region_2.<locals>.else_block_3
home_first_owner.<locals>.if_region_2.<locals>.then_block_3
quantize.<locals>.if_region_2
quantize.<locals>.if_region_2.<locals>.else_block_3
quantize.<locals>.if_region_2.<locals>.then_block_3
quantize.<locals>.if_region_4
quantize.<locals>.if_region_4.<locals>.else_block_5
quantize.<locals>.if_region_4.<locals>.then_block_5
quantize.<locals>.if_region_6
quantize.<locals>.if_region_6.<locals>.else_block_7
quantize.<locals>.if_region_6.<locals>.then_block_7
route.<locals>.if_region_6
route.<locals>.if_region_6.<locals>.else_block_7
route.<locals>.if_region_6.<locals>.then_block_7
route.<locals>.if_region_8.<locals>.then_block_11.<locals>.if_region_9
route.<locals>.if_region_8.<locals>.then_block_11.<locals>.if_region_9.<locals>.else_block_10
route.<locals>.if_region_8.<locals>.then_block_11.<locals>.if_region_9.<locals>.then_block_10
route_with_tables.<locals>.if_region_20
route_with_tables.<locals>.if_region_20.<locals>.else_block_21
route_with_tables.<locals>.if_region_20.<locals>.then_block_21
route_with_tables.<locals>.if_region_22
route_with_tables.<locals>.if_region_22.<locals>.else_block_23
route_with_tables.<locals>.if_region_22.<locals>.then_block_23
route_with_tables.<locals>.if_region_24
route_with_tables.<locals>.if_region_24.<locals>.else_block_25
route_with_tables.<locals>.if_region_24.<locals>.then_block_25
route_with_tables.<locals>.if_region_26
route_with_tables.<locals>.if_region_26.<locals>.else_block_27
route_with_tables.<locals>.if_region_26.<locals>.then_block_27
route_with_tables.<locals>.if_region_28
route_with_tables.<locals>.if_region_28.<locals>.else_block_29
route_with_tables.<locals>.if_region_28.<locals>.then_block_29
route_with_tables.<locals>.if_region_30
route_with_tables.<locals>.if_region_30.<locals>.else_block_33
route_with_tables.<locals>.if_region_30.<locals>.then_block_33
route_with_tables.<locals>.if_region_30.<locals>.then_block_33.<locals>.if_region_31
route_with_tables.<locals>.if_region_30.<locals>.then_block_33.<locals>.if_region_31.<locals>.else_block_32
route_with_tables.<locals>.if_region_30.<locals>.then_block_33.<locals>.if_region_31.<locals>.then_block_32
route_with_tables.<locals>.if_region_34
route_with_tables.<locals>.if_region_34.<locals>.else_block_35
route_with_tables.<locals>.if_region_34.<locals>.then_block_35
route_with_tables.<locals>.if_region_36
route_with_tables.<locals>.if_region_36.<locals>.else_block_37
route_with_tables.<locals>.if_region_36.<locals>.then_block_37
route_with_tables.<locals>.if_region_38
route_with_tables.<locals>.if_region_38.<locals>.else_block_41
route_with_tables.<locals>.if_region_38.<locals>.then_block_41
route_with_tables.<locals>.if_region_38.<locals>.then_block_41.<locals>.if_region_39
route_with_tables.<locals>.if_region_38.<locals>.then_block_41.<locals>.if_region_39.<locals>.else_block_40
route_with_tables.<locals>.if_region_38.<locals>.then_block_41.<locals>.if_region_39.<locals>.then_block_40
store_picks.<locals>.if_region_0.<locals>.then_block_3.<locals>.if_region_1
store_picks.<locals>.if_region_0.<locals>.then_block_3.<locals>.if_region_1.<locals>.then_block_2
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain/one_round_router.py`（96）

```text
_kernel
_kernel.<locals>.if_region_0
_kernel.<locals>.if_region_0.<locals>.then_block_1
_push
_push.<locals>.if_region_0
_push.<locals>.if_region_0.<locals>.else_block_1
_push.<locals>.if_region_0.<locals>.then_block_1
_push.<locals>.if_region_14
_push.<locals>.if_region_14.<locals>.else_block_15
_push.<locals>.if_region_14.<locals>.then_block_15
_push.<locals>.if_region_2
_push.<locals>.if_region_2.<locals>.else_block_13
_push.<locals>.if_region_2.<locals>.then_block_13
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_11
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_11.<locals>.then_block_12
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_3
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_3.<locals>.else_block_4
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_3.<locals>.then_block_4
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_5
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_5.<locals>.else_block_8
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_5.<locals>.then_block_8
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_5.<locals>.then_block_8.<locals>.if_region_6
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_5.<locals>.then_block_8.<locals>.if_region_6.<locals>.else_block_7
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_5.<locals>.then_block_8.<locals>.if_region_6.<locals>.then_block_7
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_9
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_9.<locals>.else_block_10
_push.<locals>.if_region_2.<locals>.then_block_13.<locals>.if_region_9.<locals>.then_block_10
_push_row
_push_row.<locals>.if_region_0
_push_row.<locals>.if_region_0.<locals>.else_block_1
_push_row.<locals>.if_region_0.<locals>.then_block_1
_push_row.<locals>.if_region_2
_push_row.<locals>.if_region_2.<locals>.else_block_3
_push_row.<locals>.if_region_2.<locals>.then_block_3
_push_row.<locals>.ifexp_else_block_4
_push_row.<locals>.ifexp_then_block_5
_push_row.<locals>.while_region_6
_push_row.<locals>.while_region_6.<locals>.while_after_block_7
_push_row.<locals>.while_region_6.<locals>.while_after_block_7.<locals>.if_region_10
_push_row.<locals>.while_region_6.<locals>.while_after_block_7.<locals>.if_region_10.<locals>.then_block_11
_push_row.<locals>.while_region_6.<locals>.while_after_block_7.<locals>.if_region_8
_push_row.<locals>.while_region_6.<locals>.while_after_block_7.<locals>.if_region_8.<locals>.then_block_9
_push_row.<locals>.while_region_6.<locals>.while_before_block_7
_sort
_sort.<locals>.if_region_0
_sort.<locals>.if_region_0.<locals>.else_block_1
_sort.<locals>.if_region_0.<locals>.then_block_1
_sort.<locals>.if_region_11
_sort.<locals>.if_region_11.<locals>.else_block_12
_sort.<locals>.if_region_11.<locals>.then_block_12
_sort.<locals>.if_region_13
_sort.<locals>.if_region_13.<locals>.else_block_14
_sort.<locals>.if_region_13.<locals>.then_block_14
_sort.<locals>.if_region_15
_sort.<locals>.if_region_15.<locals>.else_block_16
_sort.<locals>.if_region_15.<locals>.then_block_16
_sort.<locals>.if_region_17
_sort.<locals>.if_region_17.<locals>.then_block_23
_sort.<locals>.if_region_17.<locals>.then_block_23.<locals>.if_region_18
_sort.<locals>.if_region_17.<locals>.then_block_23.<locals>.if_region_18.<locals>.then_block_20
_sort.<locals>.if_region_17.<locals>.then_block_23.<locals>.if_region_18.<locals>.then_block_20.<locals>.ifexp_then_block_19
_sort.<locals>.if_region_17.<locals>.then_block_23.<locals>.loop_body_21
_sort.<locals>.if_region_17.<locals>.then_block_23.<locals>.loop_body_21.<locals>.ifexp_then_block_22
_sort.<locals>.if_region_2
_sort.<locals>.if_region_2.<locals>.else_block_10
_sort.<locals>.if_region_2.<locals>.then_block_10
_sort.<locals>.if_region_2.<locals>.then_block_10.<locals>.if_region_5
_sort.<locals>.if_region_2.<locals>.then_block_10.<locals>.if_region_5.<locals>.else_block_6
_sort.<locals>.if_region_2.<locals>.then_block_10.<locals>.if_region_5.<locals>.then_block_6
_sort.<locals>.if_region_2.<locals>.then_block_10.<locals>.if_region_7
_sort.<locals>.if_region_2.<locals>.then_block_10.<locals>.if_region_7.<locals>.else_block_8
_sort.<locals>.if_region_2.<locals>.then_block_10.<locals>.if_region_7.<locals>.then_block_8
_sort.<locals>.if_region_2.<locals>.then_block_10.<locals>.loop_body_9
_sort.<locals>.if_region_2.<locals>.then_block_10.<locals>.while_region_3
_sort.<locals>.if_region_2.<locals>.then_block_10.<locals>.while_region_3.<locals>.while_after_block_4
_sort.<locals>.if_region_2.<locals>.then_block_10.<locals>.while_region_3.<locals>.while_before_block_4
_sort.<locals>.if_region_24
_sort.<locals>.if_region_24.<locals>.else_block_25
_sort.<locals>.if_region_24.<locals>.then_block_25
_wait_flag
_wait_flag.<locals>.if_region_0
_wait_flag.<locals>.if_region_0.<locals>.then_block_3
_wait_flag.<locals>.if_region_0.<locals>.then_block_3.<locals>.while_region_1
_wait_flag.<locals>.if_region_0.<locals>.then_block_3.<locals>.while_region_1.<locals>.while_after_block_2
_wait_flag.<locals>.if_region_0.<locals>.then_block_3.<locals>.while_region_1.<locals>.while_before_block_2
_write_pool_row
fc1_topk_scores_tensor
launch_router
local_expert_sizes
plane_activation_pointer
plane_activation_sf_pointer
pool_expert_base_tensor
token_src_metadata_pointer
token_src_metadata_tensor
wait_for_metadata_ready
wait_for_sizes_ready
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain/token_comm.py`（36）

```text
_load_router_inputs.<locals>.if_region_4
_load_router_inputs.<locals>.if_region_4.<locals>.then_block_5
_stateless_pace
_stateless_pace.<locals>.if_region_0
_stateless_pace.<locals>.if_region_0.<locals>.else_block_1
_stateless_pace.<locals>.if_region_0.<locals>.else_block_1.<locals>.elif_region_1
_stateless_pace.<locals>.if_region_0.<locals>.else_block_1.<locals>.elif_region_1.<locals>.else_block_2
_stateless_pace.<locals>.if_region_0.<locals>.else_block_1.<locals>.elif_region_1.<locals>.then_block_2
_stateless_pace.<locals>.if_region_0.<locals>.then_block_1
_stateless_pace.<locals>.if_region_3
_stateless_pace.<locals>.if_region_3.<locals>.then_block_4
fc2_activation_tensor
next_token
token_back
token_back.<locals>.if_region_0
token_back.<locals>.if_region_0.<locals>.then_block_1
token_back.<locals>.while_region_2
token_back.<locals>.while_region_2.<locals>.while_after_block_3
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_4
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_4.<locals>.then_block_7
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_4.<locals>.then_block_7.<locals>.while_region_5
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_4.<locals>.then_block_7.<locals>.while_region_5.<locals>.while_after_block_6
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_4.<locals>.then_block_7.<locals>.while_region_5.<locals>.while_before_block_6
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.while_region_8
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.while_region_8.<locals>.while_after_block_9
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.while_region_8.<locals>.while_after_block_9.<locals>.if_region_14
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.while_region_8.<locals>.while_after_block_9.<locals>.if_region_14.<locals>.else_block_27
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.while_region_8.<locals>.while_after_block_9.<locals>.if_region_14.<locals>.then_block_27
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.while_region_8.<locals>.while_after_block_9.<locals>.if_region_14.<locals>.then_block_27.<locals>.if_region_15
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.while_region_8.<locals>.while_after_block_9.<locals>.if_region_14.<locals>.then_block_27.<locals>.if_region_15.<locals>.else_block_16
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.while_region_8.<locals>.while_after_block_9.<locals>.if_region_14.<locals>.then_block_27.<locals>.if_region_15.<locals>.then_block_16
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.while_region_8.<locals>.while_after_block_9.<locals>.if_region_14.<locals>.then_block_27.<locals>.if_region_19
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.while_region_8.<locals>.while_after_block_9.<locals>.if_region_14.<locals>.then_block_27.<locals>.if_region_19.<locals>.then_block_20
token_back.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.while_region_8.<locals>.while_before_block_9
token_back.<locals>.while_region_2.<locals>.while_before_block_3
token_back_pushes
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/dsl_helpers.py`（2）

```text
tma_multicast_mask.<locals>.ifexp_else_block_3
tma_multicast_mask.<locals>.ifexp_then_block_3
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/helpers/topk_reduce.py`（8）

```text
_fmul
_reduce_cell.<locals>.if_region_1.<locals>.then_block_11.<locals>.if_region_7
_reduce_cell.<locals>.if_region_1.<locals>.then_block_11.<locals>.if_region_7.<locals>.then_block_8
_reduce_cell.<locals>.if_region_1.<locals>.then_block_11.<locals>.if_region_9
_reduce_cell.<locals>.if_region_1.<locals>.then_block_11.<locals>.if_region_9.<locals>.else_block_10
_reduce_cell.<locals>.if_region_1.<locals>.then_block_11.<locals>.if_region_9.<locals>.then_block_10
_reduce_cell.<locals>.if_region_1.<locals>.then_block_11.<locals>.ifexp_else_block_6
_reduce_cell.<locals>.ifexp_else_block_0
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/custom_mix_cga_helpers.py`（5）

```text
create.<locals>.if_region_2
create.<locals>.if_region_2.<locals>.else_block_3
create.<locals>.if_region_2.<locals>.then_block_3
create.<locals>.ifexp_else_block_1
producer_acquire.<locals>.ifexp_else_block_2
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_epilogue.py`（35）

```text
__call__.<locals>.ifexp_then_block_0
__init__
_amax_thread_bf16
_amax_thread_bf16.<locals>.max_abs
_e8m0
_enc_mxfp8
_enc_nvfp4
fc2_stg_store_function.<locals>.if_region_0
fc2_stg_store_function.<locals>.if_region_0.<locals>.then_block_1
fc2_stg_store_function.<locals>.if_region_2
fc2_stg_store_function.<locals>.if_region_2.<locals>.then_block_3
fc2_tma_store_function.<locals>.if_region_0
fc2_tma_store_function.<locals>.if_region_0.<locals>.then_block_1
fc2_tma_store_function.<locals>.if_region_2
fc2_tma_store_function.<locals>.if_region_2.<locals>.then_block_3
get_data_dst.<locals>.if_region_0
get_data_dst.<locals>.if_region_0.<locals>.else_block_1
get_data_dst.<locals>.if_region_0.<locals>.then_block_1
get_sf_dst
get_sf_dst.<locals>.if_region_0
get_sf_dst.<locals>.if_region_0.<locals>.else_block_1
get_sf_dst.<locals>.if_region_0.<locals>.then_block_1
mx_quant_regs_in_pair_threads_impl
nvfp4_combine_quant_regs_in_thread_impl
run.<locals>.ifexp_else_block_0
run.<locals>.while_region_1.<locals>.while_after_block_2.<locals>.if_region_14
run.<locals>.while_region_1.<locals>.while_after_block_2.<locals>.if_region_14.<locals>.else_block_15
run.<locals>.while_region_1.<locals>.while_after_block_2.<locals>.if_region_14.<locals>.then_block_15
run.<locals>.while_region_1.<locals>.while_after_block_2.<locals>.if_region_16.<locals>.else_block_17.<locals>.elif_region_17.<locals>.then_block_20.<locals>.if_region_18
run.<locals>.while_region_1.<locals>.while_after_block_2.<locals>.if_region_16.<locals>.else_block_17.<locals>.elif_region_17.<locals>.then_block_20.<locals>.if_region_18.<locals>.else_block_19
run.<locals>.while_region_1.<locals>.while_after_block_2.<locals>.if_region_16.<locals>.else_block_17.<locals>.elif_region_17.<locals>.then_block_20.<locals>.if_region_18.<locals>.then_block_19
run.<locals>.while_region_1.<locals>.while_after_block_2.<locals>.if_region_8.<locals>.else_block_10.<locals>.elif_region_10.<locals>.then_block_13.<locals>.if_region_11
run.<locals>.while_region_1.<locals>.while_after_block_2.<locals>.if_region_8.<locals>.else_block_10.<locals>.elif_region_10.<locals>.then_block_13.<locals>.if_region_11.<locals>.else_block_12
run.<locals>.while_region_1.<locals>.while_after_block_2.<locals>.if_region_8.<locals>.else_block_10.<locals>.elif_region_10.<locals>.then_block_13.<locals>.if_region_11.<locals>.then_block_12
signal_fc1_done_now
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_kernel.py`（21）

```text
__call__
__call__.<locals>.rewrite_tensor_shape
_kernel
_kernel.<locals>.if_region_0
_kernel.<locals>.if_region_0.<locals>.else_block_3
_kernel.<locals>.if_region_0.<locals>.then_block_3
_kernel.<locals>.if_region_0.<locals>.then_block_3.<locals>.while_region_1
_kernel.<locals>.if_region_0.<locals>.then_block_3.<locals>.while_region_1.<locals>.while_after_block_2
_kernel.<locals>.if_region_0.<locals>.then_block_3.<locals>.while_region_1.<locals>.while_before_block_2
_kernel.<locals>.if_region_10
_kernel.<locals>.if_region_10.<locals>.else_block_11
_kernel.<locals>.if_region_10.<locals>.then_block_11
_kernel.<locals>.if_region_4
_kernel.<locals>.if_region_4.<locals>.else_block_5
_kernel.<locals>.if_region_4.<locals>.then_block_5
_kernel.<locals>.if_region_6
_kernel.<locals>.if_region_6.<locals>.else_block_7
_kernel.<locals>.if_region_6.<locals>.then_block_7
_kernel.<locals>.if_region_8
_kernel.<locals>.if_region_8.<locals>.else_block_9
_kernel.<locals>.if_region_8.<locals>.then_block_9
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_mainloop.py`（41）

```text
_load_shared_token_tile
_load_shared_token_tile.<locals>.if_region_0
_load_shared_token_tile.<locals>.if_region_0.<locals>.else_block_1
_load_shared_token_tile.<locals>.if_region_0.<locals>.then_block_1
_load_shared_token_tile.<locals>.loop_body_2
_load_shared_token_tile.<locals>.loop_body_2.<locals>.if_region_3
_load_shared_token_tile.<locals>.loop_body_2.<locals>.if_region_3.<locals>.else_block_4
_load_shared_token_tile.<locals>.loop_body_2.<locals>.if_region_3.<locals>.then_block_4
_load_shared_weight_tile
_load_shared_weight_tile.<locals>.loop_body_0
_load_shared_weight_tile.<locals>.loop_body_0.<locals>.if_region_1
_load_shared_weight_tile.<locals>.loop_body_0.<locals>.if_region_1.<locals>.else_block_2
_load_shared_weight_tile.<locals>.loop_body_0.<locals>.if_region_1.<locals>.then_block_2
_make_executable_tma_atom_for_cluster.<locals>.if_region_0
_make_executable_tma_atom_for_cluster.<locals>.if_region_0.<locals>.else_block_1
_make_executable_tma_atom_for_cluster.<locals>.if_region_0.<locals>.then_block_1
run_mma.<locals>.while_region_0.<locals>.while_after_block_1.<locals>.if_region_4
run_mma.<locals>.while_region_0.<locals>.while_after_block_1.<locals>.if_region_4.<locals>.else_block_5
run_mma.<locals>.while_region_0.<locals>.while_after_block_1.<locals>.if_region_4.<locals>.else_block_5.<locals>.elif_region_5
run_mma.<locals>.while_region_0.<locals>.while_after_block_1.<locals>.if_region_4.<locals>.else_block_5.<locals>.elif_region_5.<locals>.else_block_6
run_mma.<locals>.while_region_0.<locals>.while_after_block_1.<locals>.if_region_4.<locals>.else_block_5.<locals>.elif_region_5.<locals>.then_block_6
run_mma.<locals>.while_region_0.<locals>.while_after_block_1.<locals>.if_region_4.<locals>.then_block_5
run_mma.<locals>.while_region_0.<locals>.while_after_block_1.<locals>.if_region_7.<locals>.then_block_16.<locals>.if_region_8.<locals>.then_block_12.<locals>.loop_body_9
run_mma.<locals>.while_region_0.<locals>.while_after_block_1.<locals>.if_region_7.<locals>.then_block_16.<locals>.if_region_8.<locals>.then_block_12.<locals>.loop_body_9.<locals>.if_region_10
run_mma.<locals>.while_region_0.<locals>.while_after_block_1.<locals>.if_region_7.<locals>.then_block_16.<locals>.if_region_8.<locals>.then_block_12.<locals>.loop_body_9.<locals>.if_region_10.<locals>.else_block_11
run_mma.<locals>.while_region_0.<locals>.while_after_block_1.<locals>.if_region_7.<locals>.then_block_16.<locals>.if_region_8.<locals>.then_block_12.<locals>.loop_body_9.<locals>.if_region_10.<locals>.then_block_11
run_tma_a.<locals>.while_region_1.<locals>.while_after_block_2.<locals>.if_region_3.<locals>.else_block_7.<locals>.elif_region_7.<locals>.then_block_10.<locals>.if_region_8
run_tma_a.<locals>.while_region_1.<locals>.while_after_block_2.<locals>.if_region_3.<locals>.else_block_7.<locals>.elif_region_7.<locals>.then_block_10.<locals>.if_region_8.<locals>.else_block_9
run_tma_a.<locals>.while_region_1.<locals>.while_after_block_2.<locals>.if_region_3.<locals>.else_block_7.<locals>.elif_region_7.<locals>.then_block_10.<locals>.if_region_8.<locals>.then_block_9
run_tma_b.<locals>.if_region_4
run_tma_b.<locals>.if_region_4.<locals>.else_block_5
run_tma_b.<locals>.if_region_4.<locals>.then_block_5
run_tma_b.<locals>.ifexp_else_block_6
run_tma_b.<locals>.ifexp_then_block_0
run_tma_b.<locals>.ifexp_then_block_1
run_tma_b.<locals>.ifexp_then_block_2
run_tma_b.<locals>.ifexp_then_block_3
run_tma_b.<locals>.ifexp_then_block_6
run_tma_b.<locals>.while_region_7.<locals>.while_after_block_8.<locals>.if_region_9.<locals>.else_block_15.<locals>.elif_region_15.<locals>.then_block_18.<locals>.if_region_16
run_tma_b.<locals>.while_region_7.<locals>.while_after_block_8.<locals>.if_region_9.<locals>.else_block_15.<locals>.elif_region_15.<locals>.then_block_18.<locals>.if_region_16.<locals>.else_block_17
run_tma_b.<locals>.while_region_7.<locals>.while_after_block_8.<locals>.if_region_9.<locals>.else_block_15.<locals>.elif_region_15.<locals>.then_block_18.<locals>.if_region_16.<locals>.then_block_17
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py`（11）

```text
__call__.<locals>.ifexp_then_block_0
__call__.<locals>.ifexp_then_block_1
__call__.<locals>.ifexp_then_block_2
__call__.<locals>.ifexp_then_block_4
_kernel.<locals>.if_region_13.<locals>.then_block_15.<locals>.ifexp_then_block_14
_kernel.<locals>.if_region_7.<locals>.then_block_10.<locals>.ifexp_then_block_8
_kernel.<locals>.if_region_7.<locals>.then_block_10.<locals>.ifexp_then_block_9
_kernel.<locals>.ifexp_then_block_0
_shared_expert_tensors
_shared_expert_tensors.<locals>.k_major
_shared_expert_tensors.<locals>.scale_factors
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/fc12_scheduler.py`（3）

```text
gen_next_work.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_6.<locals>.then_block_26.<locals>.if_region_17
gen_next_work.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_6.<locals>.then_block_26.<locals>.if_region_17.<locals>.else_block_18
gen_next_work.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_6.<locals>.then_block_26.<locals>.if_region_17.<locals>.then_block_18
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/non_clc_mixed_cga.py`（10）

```text
_preferred_cluster_cta_coord
assign_device_members.<locals>.if_region_0
assign_device_members.<locals>.if_region_0.<locals>.else_block_1
assign_device_members.<locals>.if_region_0.<locals>.then_block_1
claim_next_work.<locals>.if_region_0
claim_next_work.<locals>.if_region_0.<locals>.else_block_1
claim_next_work.<locals>.if_region_0.<locals>.then_block_1
initialize_fallback_group.<locals>.if_region_0
initialize_fallback_group.<locals>.if_region_0.<locals>.else_block_1
initialize_fallback_group.<locals>.if_region_0.<locals>.then_block_1
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/shared_fc12_scheduler.py`（11）

```text
publish
publish.<locals>.if_region_2
publish.<locals>.if_region_2.<locals>.else_block_3
publish.<locals>.if_region_2.<locals>.then_block_3
publish.<locals>.ifexp_else_block_0
publish.<locals>.ifexp_else_block_1
publish.<locals>.ifexp_then_block_0
publish.<locals>.ifexp_then_block_1
publish.<locals>.while_region_4
publish.<locals>.while_region_4.<locals>.while_after_block_5
publish.<locals>.while_region_4.<locals>.while_before_block_5
```

### `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/schedulers/work_id_claim.py`（45）

```text
_claim_fixed_group_fallback_work_id.<locals>.if_region_0
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.else_block_23
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.else_block_12
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.else_block_12.<locals>.if_region_19
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.else_block_12.<locals>.if_region_19.<locals>.then_block_20
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.else_block_12.<locals>.while_region_13
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.else_block_12.<locals>.while_region_13.<locals>.while_after_block_14
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.else_block_12.<locals>.while_region_13.<locals>.while_after_block_14.<locals>.if_region_15
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.else_block_12.<locals>.while_region_13.<locals>.while_after_block_14.<locals>.if_region_15.<locals>.else_block_16
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.else_block_12.<locals>.while_region_13.<locals>.while_after_block_14.<locals>.if_region_15.<locals>.then_block_16
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.else_block_12.<locals>.while_region_13.<locals>.while_after_block_14.<locals>.if_region_17
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.else_block_12.<locals>.while_region_13.<locals>.while_after_block_14.<locals>.if_region_17.<locals>.then_block_18
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.else_block_12.<locals>.while_region_13.<locals>.while_before_block_14
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12.<locals>.if_region_10
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12.<locals>.if_region_10.<locals>.then_block_11
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12.<locals>.if_region_8
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12.<locals>.if_region_8.<locals>.else_block_9
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12.<locals>.if_region_8.<locals>.then_block_9
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12.<locals>.while_region_2
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12.<locals>.while_region_2.<locals>.while_after_block_3
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_4
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_4.<locals>.else_block_5
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_4.<locals>.then_block_5
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_6
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12.<locals>.while_region_2.<locals>.while_after_block_3.<locals>.if_region_6.<locals>.then_block_7
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_1.<locals>.then_block_12.<locals>.while_region_2.<locals>.while_before_block_3
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_21
_claim_fixed_group_fallback_work_id.<locals>.if_region_0.<locals>.then_block_23.<locals>.if_region_21.<locals>.then_block_22
_claim_fixed_group_mixed_cga_work_id.<locals>.if_region_0
_claim_fixed_group_mixed_cga_work_id.<locals>.if_region_0.<locals>.else_block_1
_claim_fixed_group_mixed_cga_work_id.<locals>.if_region_0.<locals>.then_block_1
initialize_fixed_group_mixed_cga_work_id_state.<locals>.if_region_0
initialize_fixed_group_mixed_cga_work_id_state.<locals>.if_region_0.<locals>.else_block_7
initialize_fixed_group_mixed_cga_work_id_state.<locals>.if_region_0.<locals>.then_block_7
initialize_fixed_group_mixed_cga_work_id_state.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1
initialize_fixed_group_mixed_cga_work_id_state.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.else_block_6
initialize_fixed_group_mixed_cga_work_id_state.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6
initialize_fixed_group_mixed_cga_work_id_state.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6.<locals>.if_region_2
initialize_fixed_group_mixed_cga_work_id_state.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6.<locals>.if_region_2.<locals>.else_block_3
initialize_fixed_group_mixed_cga_work_id_state.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6.<locals>.if_region_2.<locals>.then_block_3
initialize_fixed_group_mixed_cga_work_id_state.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6.<locals>.if_region_4
initialize_fixed_group_mixed_cga_work_id_state.<locals>.if_region_0.<locals>.then_block_7.<locals>.if_region_1.<locals>.then_block_6.<locals>.if_region_4.<locals>.then_block_5
```

### `tensorrt_llm/_torch/speculative/dspark_markov_chain.py`（2）

```text
<module>
MarkovWorkspace
```

### `tensorrt_llm/_torch/speculative/group_sharded_sampling.py`（3）

```text
<module>
GroupShardedVerify
_DeviceArray
```

### `tensorrt_llm/_torch/speculative/sharded_sampling.py`（2）

```text
<module>
VocabShardedSampler
```

### `tensorrt_llm/_torch/visual_gen/attention/attention.py`（4）

```text
Attention.torch_dynamo_resume_in_apply_qk_norm_at_530
Attention.torch_dynamo_resume_in_apply_qk_norm_at_531
Attention.torch_dynamo_resume_in_forward_at_785
Attention.torch_dynamo_resume_in_forward_at_803
```

### `tensorrt_llm/_torch/visual_gen/attention/backends/parallel.py`（4）

```text
UlyssesAttention.torch_dynamo_resume_in__forward_fused_at_559
UlyssesAttention.torch_dynamo_resume_in__forward_packed_self_attention_at_581
UlyssesAttention.torch_dynamo_resume_in_forward_at_487
UlyssesAttention.torch_dynamo_resume_in_forward_at_489
```

### `tensorrt_llm/_torch/visual_gen/models/minimax_h3/transformer_minimax_h3.py`（3）

```text
MiniMaxH3Attention.torch_dynamo_resume_in_forward_at_297
MiniMaxH3TransformerBlock.torch_dynamo_resume_in__forward_fused_adaln_at_721
MiniMaxH3TransformerBlock.torch_dynamo_resume_in__forward_fused_adaln_at_735
```

### `tensorrt_llm/_torch/visual_gen/models/wan/parallel_wino_vae.py`（2）

```text
<module>
ParallelWinoVae
```

### `tensorrt_llm/_torch/visual_gen/models/wan/transformer_wan.py`（4）

```text
WanBlock.torch_dynamo_resume_in_forward_at_613
WanBlock.torch_dynamo_resume_in_forward_at_657
WanBlock.torch_dynamo_resume_in_forward_at_685
WanBlock.torch_dynamo_resume_in_forward_at_711
```

### `tensorrt_llm/_torch/visual_gen/modules/rms_norm.py`（1）

```text
RMSNormTPAware.torch_dynamo_resume_in_forward_at_96
```

### `tensorrt_llm/deep_ep/buffer.py`（13）

```text
Buffer.__init__
Buffer.__init__.<locals>.all_gather_object
Buffer._unpack_bias
Buffer.combine
Buffer.dequantize_nvfp4_to_bf16
Buffer.dispatch
Buffer.get_combine_config
Buffer.get_dispatch_config
Buffer.get_dispatch_layout
Buffer.get_low_latency_rdma_size_hint
Buffer.low_latency_combine
Buffer.low_latency_dispatch
Buffer.quantize_bf16_to_nvfp4
```

### `tensorrt_llm/deep_ep/utils.py`（2）

```text
EventOverlap.__init__
check_nvlink_connections
```

### `tensorrt_llm/deep_gemm/mega/__init__.py`（1）

```text
SymmBuffer.destroy
```

### `tensorrt_llm/evaluate/lm_eval_tasks/gpqa/cot_zeroshot_aa/utils.py`（1）

```text
<module>
```

## 产物校验值与复核 SQL

- `cbts_pystart_report_x86_64.tar.gz` SHA-256：`f13e150cf287f4eccc388a12e1f929dff12f18b9cac541c633e706334ce20a8b`。
- `cbts_pystart_report_SBSA.tar.gz` SHA-256：`f8c7ef2364b5f5c6529910f2c8f0515d81a0a926a17ec8f3983761ea9e784550`。
- `cbts_pystart_report.tar.gz` SHA-256：`71a7cd2ced5e4c7d8ca42f27f63fd4d94b64099af3270a7c95d8e59232ba7d07`。

三个压缩包的本地 SHA-256 均已与 Artifactory 返回的 `X-Checksum-Sha256` 一致性校验。三个 SQLite 均为 `storage_schema=compact-v1`、`schema_version=5`。

另外使用三个数据库的 `touch_rows` 视图，独立执行 SQL `UNION` / `EXCEPT` 并与 AST 库存连接复核；原始函数集合大小、新增/减少数以及官方覆盖率分子均与主分析一致。附录 A、B 的完整函数清单也已分别核对为 2,523 与 2,962 项。

集合读取等价 SQL：

```sql
SELECT DISTINCT file, qualname FROM touch_rows WHERE test != '';
SELECT DISTINCT file FROM touch_rows WHERE test != '';
-- 对两个 checkpoint 的结果先做集合并集，再与 final 求差。
-- 覆盖率分子再分别与同 commit 的 AST 文件/函数库存求交。
```
