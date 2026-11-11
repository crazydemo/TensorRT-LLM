<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Checkpoints 导入命中审计：最近一周与两周 PR

生成日期：2026-11-11。所有窗口与下表日期均使用 Asia/Shanghai（UTC+08:00）。

## 结论

**coverage 支持“导入了 checkpoints，但未进入具体函数”这一现象；两周窗口中找到 3 个有独占导入候选的 PR。最近一周没有找到同口径候选。**

这里的“独占”表示：候选记录在某个被修改的 checkpoint 文件中只执行了模块/类体，既未命中本 PR 修改的 checkpoint 函数体，也未命中其他核心 Python 文件修改的符号；同时已扣除本 PR 直接修改的测试及覆盖它们的目录包装 entry。它是 coverage 关联候选，尚未验证历史 CI 是否确实因此多跑，也不是安全跳过许可。

| 窗口 | 全部 first-parent 提交 | 改动 checkpoints 的提交/PR | 有独占导入候选的 PR |
|---|---:|---:|---:|
| 最近一周：11-04 14:18:49 至 11-11 14:18:49 | 223 | 6 | 0 |
| 最近两周：10-28 14:18:49 至 11-11 14:18:49 | 465 | 11 | 3 |

| PR | 合入时间（UTC+08） | 候选 stage/test | 跨 stage 去重后的 test 名 | 完整证据 |
|---|---|---:|---:|---|
| [#21339](https://github.com/NVIDIA/TensorRT-LLM/pull/21339) | 2026-11-02 23:57 | 552 | 366 | [完整测试清单](PR-21339.md) |
| [#21094](https://github.com/NVIDIA/TensorRT-LLM/pull/21094) | 2026-10-30 06:19 | 436 | 262 | [完整测试清单](PR-21094.md) |
| [#20694](https://github.com/NVIDIA/TensorRT-LLM/pull/20694) | 2026-10-29 02:07 | 344 | 213 | [完整测试清单](PR-20694.md) |

**上述计数不能相加成独立测试数**：同一测试可出现在多个 stage，也可被多个 PR 命中。test 名也可能是包装入口、目录或带 `-k` 的测试组，不等于内层 pytest case。

## 数据来源与固定输入

- 样本 tip：[`3cc27151f475b049d15e2c8f4119ca6eb30a11f1`](https://github.com/NVIDIA/TensorRT-LLM/commit/3cc27151f475b049d15e2c8f4119ca6eb30a11f1)。固定该 tip 扫描全部类型的 first-parent 提交，没有只挑 Python PR。465 是提交分母，不表述为 465 个唯一 PR。
- 截止此次取数，Jenkins 最新编号为 3067；向前探测到的最新完整 x86/SBSA coverage 对为 build 3063。后续扩大时间窗沿用该对，没有刷新 coverage。
- [Jenkins build 3063](https://prod.blsm.nvidia.com/sw-tensorrt-top-1/job/LLM/job/main/job/L0_PostMerge/3063/)，整体结果为 FAILURE。两份 DB 都是 schema v5。
- `build_info.txt` 未解析出采集源码 commit，报告保留 `coverage_commit: null`；不能假定 coverage 源码与每个 PR 的源码完全一致。
- 原始 DB 仅提取并合并 `(stage, test, file, qualname)` 关系做符号审计，未合并为可供生产 selector 使用的完整 DB。没有把 taint/incomplete 记录当作可安全删除测试的证据。

| Coverage 文件 | 下载地址 | SHA256 |
|---|---|---|
| `cbts_pystart_report_x86_64.tar.gz` | [Artifactory](https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/3063/cbts-coverage/cbts_pystart_report_x86_64.tar.gz) | `71546733cbc2da3b81967a43dc198e41581a81fe9412d8c26265820944f3c282` |
| `cbts_pystart_report_SBSA.tar.gz` | [Artifactory](https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/3063/cbts-coverage/cbts_pystart_report_SBSA.tar.gz) | `7f1a1dd1b6e61c902315937722f8e5178b0c28cafcadb60d20613cfe1626c241` |

## 筛选方法

1. 对固定 main tip 取最近 14 天全部 first-parent 提交，按提交的 committer timestamp 判断窗口；一周统计是同一固定 cohort 的子集。
2. 对每个提交比较第一父提交与该提交，找到 checkpoints 下修改的 Python 文件。先用 `strip_noop_diff_lines` 排除注释/空白修改，再用 `iter_diff_post_line_numbers` 和 post-image AST 映射变更符号。删除行使用现有 helper 的 post-image 锚点；删除整段代码等边界没有单独重建 pre-image 语义。
3. AST 中 `<module>` 与类体视为 import-executed 符号；函数/方法属于具体函数体。类体被执行不表示该类被实例化。coverage 里的符号是否记录为类体，按同一源码 AST 判断。
4. coverage 的空 test 名记录是未归属到具体测试的桶，全部排除。本轮计数与此前含这些桶的初筛值不同。
5. 对每个修改文件，取“命中被修改的 import-executed 符号”的测试，减去“进入该文件任何具体函数”的测试，再将各文件候选取并集。
6. 从候选并集中，减去本 PR 所有改变的 checkpoint 函数体、其他核心 Python 文件改变的任意符号，以及直接修改的测试/目录包装入口的触达集合。
7. 只对步骤 6 剩余的记录计数；按 `(stage, test)` 去重。未把非核心路径规则、Tier1、完整 YAML 匹配、信任状态与实际 CI 门禁加入回放。

可以写成集合关系（所有集合元素均为 stage/test）：

```text
candidate = union_per_changed_checkpoint_file(
    tests_touching_changed_import_symbols - tests_entering_any_function_in_that_file
) - tests_touching_changed_checkpoint_functions
  - tests_touching_any_changed_symbol_in_other_core_python_files
  - directly_changed_tests_or_containing_test_wrappers
```

## 两周内全部 11 个 checkpoints PR

“导入匹配”和“函数体匹配”只统计 checkpoints 中被修改符号，二者可能有交集；“最终候选”还要求上述文件内无任何具体函数触达及其他扣除条件。因此最终候选不等于前两列简单相减。

| PR | 时间（UTC+08） | 描述 | 导入匹配 | 函数体匹配 | 最终候选 |
|---|---|---|---:|---:|---:|
| [#20238](https://github.com/NVIDIA/TensorRT-LLM/pull/20238) | 2026-11-11 04:17 | [None][feat] Enable PPLX decider support | 711 | 0 | 0 |
| [#20853](https://github.com/NVIDIA/TensorRT-LLM/pull/20853) | 2026-11-10 17:53 | [None][feat] Classify startup failures, surface their root cause and report them in telemetry | 711 | 176 | 0 |
| [#21287](https://github.com/NVIDIA/TensorRT-LLM/pull/21287) | 2026-11-10 07:25 | [None][perf] Support FP8 rowwise GDN projections in Qwen3.5 and fuse their norm + per-token FP8 quantize | 0 | 2 | 0 |
| [#20703](https://github.com/NVIDIA/TensorRT-LLM/pull/20703) | 2026-11-07 07:20 | [TRTLLM-16725][refactor] BREAKING: Use a string enum for checkpoint formats and simplify checkpoint I/O helpers | 711 | 406 | 0 |
| [#20498](https://github.com/NVIDIA/TensorRT-LLM/pull/20498) | 2026-11-06 06:26 | [TRTLLM-16725][refactor] BREAKING: Use a string enum for checkpoint I/O policy | 711 | 405 | 0 |
| [#20843](https://github.com/NVIDIA/TensorRT-LLM/pull/20843) | 2026-11-05 07:06 | [None][chore] Mass merge release/1.3 branch back to main (batch 2) | 711 | 13 | 0 |
| [#21339](https://github.com/NVIDIA/TensorRT-LLM/pull/21339) | 2026-11-02 23:57 | [None][chore] Reject per-layer KV-head configs in the generic HF weight mapper | 711 | 235 | 552 |
| [#21225](https://github.com/NVIDIA/TensorRT-LLM/pull/21225) | 2026-11-02 21:42 | [https://nvbugs/6997728][fix] Fix DeepSeek-V4-Pro DEP4 MegaMoE CuTe DSL test: skip HF weight cache for oversized checkpoints, drop hard GSM8K floor, fix bootstrap hang | 711 | 176 | 0 |
| [#21042](https://github.com/NVIDIA/TensorRT-LLM/pull/21042) | 2026-10-31 01:02 | [TRTLLM-17145][feat] Add the Muse Spark Open model | 711 | 390 | 0 |
| [#21094](https://github.com/NVIDIA/TensorRT-LLM/pull/21094) | 2026-10-30 06:19 | [https://nvbugs/6970940][fix] Re-read rank-striped read-ahead extents after the page cache drops them | 711 | 388 | 436 |
| [#20694](https://github.com/NVIDIA/TensorRT-LLM/pull/20694) | 2026-10-29 02:07 | [None][fix] Load compressed-tensors NVFP4 weights in MiniMax-M3 | 711 | 6 | 344 |

## 重点候选与解释

### PR #21339

修改 generic HF mapper 的 KV-head 检查，同时涉及 base mapper 与 Qwen3-VL mapper 的导入执行范围。扣除修改函数体和其他改动后，仍有 552 条记录仅因这些 checkpoint 文件的导入执行符号产生关联。适合作为首个精度分析样本。

[完整 552 条 stage/test 清单及文件内 recorded qualname](PR-21339.md)。

| 修改的 checkpoint 文件 | 改变的导入符号 | 改变的函数体 |
|---|---|---|
| `tensorrt_llm/_torch/models/checkpoints/base_weight_mapper.py` | `<module>` | — |
| `tensorrt_llm/_torch/models/checkpoints/hf/gemma4_weight_mapper.py` | — | `Gemma4HfWeightMapper.apply_callbacks` |
| `tensorrt_llm/_torch/models/checkpoints/hf/qwen3vl_weight_mapper.py` | `<module>` | — |
| `tensorrt_llm/_torch/models/checkpoints/hf/weight_mapper.py` | `HfWeightMapper` | `HfWeightMapper._num_kv_heads` |

示例候选：

- `A10-PyTorch-1` / `test_unittests.py::test_unittests_v2[unittest/_torch/modeling/test_modeling_nemotron_nas.py]`
- `A10-PyTorch-1` / `test_unittests.py::test_unittests_v2[unittest/kv_cache_manager_v2_tests]`
- `A10-PyTorch-1` / `test_unittests.py::test_unittests_v2[unittest/grpc/openengine/test_multimodal_conformance.py]`

### PR #21094

修改 rank-striped read-ahead 的 import、模块常量及构造函数签名。构造函数签名在现有 AST 映射中归到类执行体，覆盖范围会远大于调用该方法的测试集合；同时存在进入 read-ahead 与 weight-loader 修改函数体的测试，它们已被扣除。

[完整 436 条 stage/test 清单及文件内 recorded qualname](PR-21094.md)。

| 修改的 checkpoint 文件 | 改变的导入符号 | 改变的函数体 |
|---|---|---|
| `tensorrt_llm/_torch/models/checkpoints/hf/rank_striped_read_ahead.py` | `<module>`, `RankStripedReadAheadSession` | `RankStripedReadAheadSession.__init__`, `RankStripedReadAheadSession._read_pass`, `RankStripedReadAheadSession._refresh_dropped_pages`, `RankStripedReadAheadSession._run`, `node_major_page_faults` |
| `tensorrt_llm/_torch/models/checkpoints/hf/weight_loader.py` | — | `HfWeightLoader._start_rank_striped_read_ahead` |

示例候选：

- `A10-PyTorch-1` / `test_unittests.py::test_unittests_v2[unittest/_torch/modeling/test_modeling_nemotron_nas.py]`
- `A10-PyTorch-1` / `test_unittests.py::test_unittests_v2[unittest/kv_cache_manager_v2_tests]`
- `A10-PyTorch-1` / `test_unittests.py::test_unittests_v2[unittest/_torch/modeling/test_modeling_apertus.py -k "not TestApertus8BInstruct and not TestApertus1p5_8B"]`

### PR #20694

修改 HF weight mapper 中 compressed-tensors NVFP4 转换函数及模块级范围，同时修改模型实现。扣除这些转换函数和模型实现的符号触达后，还剩 344 条导入候选。

[完整 344 条 stage/test 清单及文件内 recorded qualname](PR-20694.md)。

| 修改的 checkpoint 文件 | 改变的导入符号 | 改变的函数体 |
|---|---|---|
| `tensorrt_llm/_torch/models/checkpoints/hf/weight_mapper.py` | `<module>` | `_invert_compressed_tensors_global_scale`, `_read_compressed_tensors_global_scale`, `_warn_uninvertible_global_scales`, `convert_compressed_tensors_nvfp4_weights` |

示例候选：

- `A10-PyTorch-1` / `test_unittests.py::test_unittests_v2[unittest/_torch/modeling/test_modeling_nemotron_nas.py]`
- `A10-PyTorch-1` / `test_unittests.py::test_unittests_v2[unittest/kv_cache_manager_v2_tests]`
- `A10-PyTorch-1` / `test_unittests.py::test_unittests_v2[unittest/_torch/modeling/test_modeling_apertus.py -k "not TestApertus8BInstruct and not TestApertus1p5_8B"]`

## 如何理解结果与后续验证

- 数据已经支持“导入执行导致符号触达范围很宽”这一判断；可优先审核 #21339、#21094、#20694。
- 当前审计所用 checkout `22ebe475fa` 的 selector 遇到 import-executed 改动会直接 fallback。它不会把本文的候选计数直接作为成功 Tier2 选择的额外运行数量；其他部署版本必须按实际策略再核对。
- 模块级注册表、继承、decorator、默认参数和类属性变化可能影响仅导入的测试，因此“未进入函数”本身不足以证明无影响。尤其方法签名与新增函数定义也会在类体/模块体执行时生效。
- 此次缺少新版 backtest runner，未做完整 Tier1/Tier2、残余 patch compatibility、作者资格、多卡批准门禁、YAML 展开与历史 CI 最终调度回放。case 列表含 post-merge stage，不能直接换算为 pre-merge 节省。
- coverage 采集版本未知且 build 整体 FAILURE，无函数触达可能与采集缺失有关。下一步应固定实际部署策略、检查候选 taint/完整性，再对三个 PR 做 before/after 决策回放并与真实 CI 决策 artifact 比较。
- 本报告没有证明任何测试可安全跳过，也没有测量 GPU 时间或 escaped tests。

## 复核与附件

- [固定 cohort manifest：全部 465 个提交](evidence/manifest.json)
- [输入版本、计数口径与 coverage 文件指纹](evidence/inputs.json)
- [全部 11 个 PR 的文件、变更符号、覆盖计数和候选 ID](evidence/audit.json)
- [三个 PR 的全部候选及逐文件 recorded qualname 证据](evidence/candidates.json)

复核 cohort 可使用以下命令；tip 固定，避免未来 main 更新改变分母：

```bash
git log 3cc27151f475b049d15e2c8f4119ca6eb30a11f1 --first-parent \
  --since='2026-10-28T14:18:49+08:00' \
  --until='2026-11-11T14:18:49+08:00' --format='%H %cI %s'
```

符号归属使用 `22ebe475fa` 中的 `qualname_map.py` 与 `rules/_helpers.py`。具体实现引用：

- [qualname_map.py](https://github.com/crazydemo/TensorRT-LLM/blob/22ebe475fa4f5809c5e5a384a3512193b059ab2e/jenkins/scripts/cbts/coverage_selection/qualname_map.py)
- [selector.py](https://github.com/crazydemo/TensorRT-LLM/blob/22ebe475fa4f5809c5e5a384a3512193b059ab2e/jenkins/scripts/cbts/coverage_selection/selector.py)
- [_helpers.py](https://github.com/crazydemo/TensorRT-LLM/blob/22ebe475fa4f5809c5e5a384a3512193b059ab2e/jenkins/scripts/cbts/rules/_helpers.py)
