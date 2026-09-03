<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CBTS forward synthetic rebase 的 case selection 对比

## 1. 目标

本实验比较同一个 PR 在两种 coverage 使用方式下的选例结果：

1. **原方案**：在原始 `pr_base` 上应用 PR patch，使用当次 `/bot run` 原本可用的 coverage DB；
2. **forward synthetic rebase**：把相同 PR patch 干净地应用到当时已经存在的最新 forward coverage commit，使用该 commit 对应的 coverage DB。

为了判断 rebase 本身是否值得，不能把 coverage DB 自身不完整造成的强制运行误算成
rebase 的收益或代价。因此 case-level 比较采用以下严格口径：

- case 必须在两个 DB 中都 known；
- case 在任一 DB 中被标记为 untrusted 时，从两边同时剔除；
- case 必须同时存在于两边当时的 pre-merge single-GPU test-db/stage；
- 排除 CPU always-run、Post-Merge、multi-GPU、coarse 和 no-data test entries；
- 排除已经由 Tier 1 强制保留的 case；
- 比较主键为 `stage family / normalized nodeid`，避免 pytest-split shard 变化造成假差异。

## 2. PR #17817 实验

### 2.1 输入

| 项目 | 原方案 | Forward synthetic rebase |
|---|---|---|
| PR | [#17817](https://github.com/NVIDIA/TensorRT-LLM/pull/17817) | 同一 PR patch |
| Coverage build | 2925 | 2926 |
| Coverage commit | `4baa757f97f0` | `55548ee861c8` |
| Synthetic commit | `42d100da887a` | `76e4474573b9` |
| Changed files | `tensorrt_llm/executor/proxy.py`、`tests/unittest/executor/test_proxy_fast_death.py` | 相同 |

Build 2926 的 x86_64 和 SBSA raw coverage artifact 都在该次 `/bot run` 前存在。
用于回放的 merged DB 已验证为这两份 raw DB 的精确 union：

- touch rows：1,740,635；
- test metadata rows：1,033；
- 相对 raw union：0 missing、0 extra。

两个 synthetic tree 使用的 `selector.py` 和 `qualname_map.py` 内容完全一致。实验只在临时
worktree 中同时取消了 `import-executed class change` 的拒绝逻辑，让 class qualname 像普通
qualname 一样继续进入 coverage mapping；实验完成后临时修改已恢复。

### 2.2 严格可比结果

| 指标 | 原方案：DB 2925 | Forward rebase：DB 2926 |
|---|---:|---:|
| 共同 trusted、可运行的 stage-cases | 221 | 221 |
| Coverage 选中 | 204 | 208 |
| 可 skip | 17 | 13 |
| Skip rate | 7.69% | 5.88% |

集合差异：

- 两边共同选中：204；
- 仅原方案选中：0；
- 仅 forward rebase 选中：4；
- Jaccard similarity：98.08%；
- forward rebase 的 skip rate 下降 1.81 个百分点。

Forward-only 的 4 个 stage-cases 为：

1. `DGX_B200-PyTorch/perf/host_perf/test_module_resource_manager.py::test_kv_cache_prepare_context`
2. `DGX_B200-PyTorch/perf/host_perf/test_module_sampler.py::test_sampler_update_greedy[greedy_bs8]`
3. `DGX_B200-PyTorch/perf/host_perf/test_module_scheduler.py::test_scheduler_production[production_mixed_32gen_4ctx]`
4. `L40S-PyTorch/accuracy/test_llm_api_pytorch_encode.py::TestEncoderEncode::test_qwen3_text_embedding_matches_huggingface[qwen3-embedding-0.6b]`

### 2.3 为什么两边都是 no-data，2926 仍会新增 case

这里的 no-data 是指本 PR 新增的三个方法没有 function-level coverage rows：

- `GenerationExecutorProxy._abort_owned_session`；
- `GenerationExecutorProxy._detect_worker_death_during_init`；
- `GenerationExecutorProxy._fail_initialization`。

它并不表示整个 `proxy.py` 没有 coverage 数据。两个 DB 上的 selector 都执行相同逻辑：

```text
changed qualname no-data
        -> no_data_policy=file
        -> 选择该 DB 中所有曾进入 proxy.py 的 tests
```

所以最终集合取决于每个 DB 各自记录的 file-level touch set。上述 4 个 case 在两个 DB 中
都存在且属于共同 trusted universe，但它们在 DB 2925 中对 `proxy.py` 的 touch rows 为 0，
在 DB 2926 中各有 5 行，因而只在 forward 方案中被选中。

### 2.4 三个 DGX_B200 host-perf case 的异常 footprint

三个 DGX_B200 case 在两边 DB 中均存在且 capture outcome 为 `passed`：

| Case | DB 2925 coverage rows | DB 2926 coverage rows | DB 2925 `proxy.py` rows | DB 2926 `proxy.py` rows |
|---|---:|---:|---:|---:|
| `test_kv_cache_prepare_context` | 41 | 60 | 0 | 5 |
| `test_sampler_update_greedy[greedy_bs8]` | 37 | 64 | 0 | 5 |
| `test_scheduler_production[production_mixed_32gen_4ctx]` | 39 | 59 | 0 | 5 |

DB 2926 中，它们共同新增的 5 个 `proxy.py` qualnames 是：

- `GenerationExecutorProxy.__del__`；
- `GenerationExecutorProxy._abort_all_requests`；
- `GenerationExecutorProxy._cleanup_multi_frontend_ipc_dir`；
- `GenerationExecutorProxy.pre_shutdown`；
- `GenerationExecutorProxy.shutdown`。

三个 case 还共同新增了 `BaseLLM.shutdown`、`MpiPoolSession.shutdown`、
`MpiSession.shutdown_abort`、`FusedIpcQueue.close`、`WorkerProcessMonitor.close` 等 teardown
调用链。与此同时：

- 三个 host-perf test 文件在 coverage commits 之间没有修改；
- 它们直接测试 scheduler、sampler 或 resource manager，本身不创建
  `GenerationExecutorProxy`；
- 三个 case 的 pytest-split shard 均发生变化；
- gap 中 `ecd69d56f0b9`（#17644）修改了全局 session-reuse 后台 teardown/reap；
- gap 中 `9d396def3c1b`（#17642）修改了 session-prefetch dead-pool cleanup；
- `46fa03d533a7`（#18063）修改了 `l0_b200.yml`，可能触发重新分片。

这些证据更符合以下解释：前一个测试留下的 cached/prefetched executor 在后台延迟 teardown，
而 teardown 执行时 pytest coverage context 已经切换到当前 host-perf case，导致生命周期函数
被归到不相关 case。由于目前只有 2925 和 2926 两个观测点，尚不能把根因确定到单个 commit；
因此应称为**高度疑似 cross-test/background teardown attribution 污染**，而不是已证实的根因。

### 2.5 初步结论

在 PR #17817 上，取消 import-executed class gate 后，原方案和 forward rebase 都能完成
coverage narrowing。Forward rebase 没有移除任何共同 trusted case，反而额外选择 4 个，
使严格可比 skip rate 从 7.69% 降至 5.88%。

这个样本没有显示 rebase 能降低 CI case 成本。它同时暴露出现有 `untrusted_tests()` 无法识别
跨测试的后台 teardown attribution：这些 capture outcome 为 passed、footprint 也不小，因此
仍被视为 trusted。后续评估可以增加 coverage-edge stability 过滤，例如只信任在连续多份 DB
中重复出现的 `case -> lifecycle qualname` 映射，或单独标记只出现一次的 teardown-only edge。

## 3. 其他 PR 候选

候选必须同时满足以下条件，才能用于比较 rebase 前后的 case 集合：

1. 对应 `/bot run` 发生前已经存在 raw coverage artifact；
2. coverage commit 位于 `pr_base` 的 forward 方向；
3. PR patch 可以干净地应用到该 coverage commit；
4. 在只取消 import-executed **class** gate、仍保留 `<module>` gate 的实验策略下，原方案和
   forward rebase 都能得到 coverage selection，而不是在进入选例前就 decline。

九个代表性 PR 的筛选结果如下。`Forward gap` 按 main first-parent commit 数计算。

| PR | 当时的 forward raw DB | Forward gap | Synthetic merge | Forward selector 结果 | 可用于本实验 |
|---:|---|---:|---|---|---|
| #17817 | 2926 / `55548ee861c8` | 24 | clean | `scope=coverage` | 是，已完成 |
| #17831 | 2923 / `f51e32335aef` | 4 | clean | zero-touch：`dynamic_mainloop.py` | 否 |
| #18182 | 2942 / `6e6f506077cb` | 44 | clean | `<module>`：`mega_moe_deepgemm.py` | 否，除非另行放宽 module gate |
| #18236 | 无 | — | 未测试 | 当时没有 forward artifact | 否 |
| #18239 | 无 | — | 未测试 | 当时没有 forward artifact | 否 |
| #18256 | 2931 / `767af6f285` | 18 | clean | `<module>`：`fused_moe_triton.py` | 否，除非另行放宽 module gate |
| #18257 | 2932 / `34847b86be36` | 25 | clean | `scope=coverage` | 是，已完成 |
| #18284 | 2943 / `8c0d0cbfef5c` | 2 | clean | `<module>`：`modeling_utils.py` | 否，除非另行放宽 module gate |
| #18352 | 无 | — | 未测试 | 当时没有 forward artifact | 否 |

这里的 module decline 与已经取消的 class gate 不同。例如 #18256 不只是修改
`TritonFusedMoE.can_implement`，还新增了 module-level import；因此即使 class change 继续走
coverage mapping，`<module>` 仍会使 selector decline。#18256 已使用当次实际可用的 DB 2931
精确复核，并非根据更晚的 DB 2932 推断。

### 3.1 PR #18257：同 DB 的干净对照样本

PR #18257 是九个 PR 中最适合隔离 rebase 影响的样本，因为历史 `/bot run` 已经选中了位于
`pr_base` 前方的 DB 2932。原方案和 synthetic rebase 可以使用**完全相同的 coverage DB**，
所以 DB capture completeness、untrusted metadata 和 coverage footprint 都保持不变。

| 项目 | 原方案 | Forward synthetic rebase |
|---|---|---|
| PR CI | 57008 | 同一历史时点 |
| PR base | `0cb928b72e6c` | — |
| Coverage build / commit | 2932 / `34847b86be36` | 相同 |
| PR head | `026fe820ff57` | 相同 PR patch |
| Synthetic commit | `fb9c2b321c1c` | `4c57c66ae7b4` |
| Selector | 相同内容 | 相同内容 |
| Coverage DB | 同一个 SQLite 文件 | 同一个 SQLite 文件 |

PR diff 包含：

- `tensorrt_llm/_torch/modules/fused_moe/MOE_DEVELOPER_GUIDE.md`；
- `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py`；
- `tests/integration/test_lists/waives.txt`。

默认 selector 会因为 `CutlassFusedMoE` 是 import-executed class 而 decline。按照本实验已经
确定的策略，两边只取消 class gate、保留 `<module>` gate 后，均得到 `scope=coverage`。

### 3.2 PR #18257 的选例结果

| 指标 | 原方案 | Forward synthetic rebase |
|---|---:|---:|
| Raw impacted stage-cases | 446 | 446 |
| 当前 DB 中的 untrusted stage-cases | 131 | 131 |
| Impacted 且非 untrusted | 382 | 382 |
| 严格共同 trusted universe | 691 | 691 |
| 严格共同 trusted selected | 382 | 382 |
| Pre-merge CI 可比 universe | 331 | 331 |
| Coverage 选中 | 213 | 213 |
| 可 skip | 118 | 118 |
| Skip rate | 35.65% | 35.65% |

集合差异为 0：

- 两边共同选中：213 个 pre-merge CI stage-cases；
- 仅原方案选中：0；
- 仅 forward rebase 选中：0；
- Jaccard similarity：100%。

因此，在 #18257 上，synthetic rebase 可以把“coverage commit 比 `pr_base` 新 25 个 commit”
这一 freshness 问题转化成合法的 coverage-parent 关系，但**没有改变 coverage 选例集合，也没有
改变 skip rate**。它的收益是避免 freshness fallback，而不是减少或增加测试成本。

## 4. 当前结论

九个代表性 PR 中，按“当时已有 raw artifact”以及“只取消 class gate”的约束，只有 #17817
和 #18257 能完成严格的 case-set A/B：

- #17817 使用不同 DB，forward 方案多选 4 个 case，skip rate 下降 1.81 个百分点；其中至少
  3 个新增 edge 高度疑似后台 teardown attribution 污染；
- #18257 使用同一 DB，是更干净的因果对照；两边集合完全一致，rebase 只消除了 freshness
  fallback，没有改变选例成本；
- #17831 是 forward DB zero-touch；#18182、#18256、#18284 被保留的 `<module>` gate 拒绝；
- #18236、#18239、#18352 在对应历史时点没有 forward raw coverage，无法忠实回放。

因此现有数据能证明 forward synthetic rebase **有机会救回 freshness fallback**，但还不足以
证明它总体上值得作为默认策略：一个干净样本的 case 集合不变，另一个样本还受到 DB 间
coverage-edge 漂移的影响。若要扩大样本，应优先长期保留每次 raw coverage artifact，并把
“rebase 机制”与“是否允许 module-level change 进入 coverage mapping”拆成两个独立实验变量。
