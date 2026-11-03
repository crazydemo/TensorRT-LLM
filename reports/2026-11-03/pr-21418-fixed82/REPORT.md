<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PR #21418 · 固定 82 PR 最新 CBTS dry-run

- Policy: `b50258c92c8ec621a52f117e1251598f2810e124`，已提交、工作树干净；报告生成 2026-11-03T13:41:55+08:00。
- 重新执行 82/82 个固定历史样本；DB build 3049，pre-merge，多卡批准 gate 关闭。
- 候选命中 34/82（41.5%）；Tier 1 23 / Tier 2 11。
- 兼容后命中 30/82（36.6%）；有效裁减命中 30，零裁减命中 0。
- 候选跳过 25,686/102,803；兼容后跳过 25,229/102,803（24.54%）。
- 整跑 52；执行错误 0；较上一轮 0 个决策、指标或过滤 YAML 变化。
- 核心 Python 子集兼容后命中 13/46。

固定样本与 DB 未刷新。每个样本用其历史源码/YAML/Groovy运行冻结的当前 policy。兼容性用 git merge-tree 三方树检查；不模拟作者/pilot eligibility 或全部 CI 调度 gate。结果是回溯候选决定，不是生产应用率、展开后的 pytest case 数、GPU 卡时收益或零漏检证明。

[HTML](REPORT.html) · [汇总](summary.json) · [Manifest](manifest.json) · [对比](comparison.json)

DB commit: `0fd58c90ede974e01ecc53627ad5a9bc76846703`
DB SHA256: `e60b0816dfdd4c32344392c74445c6f92f631b641f0b8ab12474d832757f686a`

## 与最初 26/82 历史报告的主对比

| 指标 | 最初报告 | 本次 | 按记录变化 |
|---|---:|---:|---:|
| 历史记录 / 当前兼容后命中 | 26/82（31.7%） | 30/82（36.6%） | +4 个，+4.9 个百分点 |
| Tier 1 | 23 | 23 | 不变 |
| Tier 2 | 3 个 shadow | 7 个兼容后命中 | 按记录 +4 |
| 原有命中丢失 | — | 0 | 26 个均保留 |

新增：

- #19585：移除过时 TensorRT telemetry 兼容路径；旧记录「preflight 挡住」，当前 coverage + patch clean，跳过 312/1256 个 YAML 条目。
- #18636：多模态 fusion 使用 index copy；旧记录「import 期代码挡住」，当前 coverage + patch clean，跳过 295/1253 个 YAML 条目。
- #19901：缓存 MegaMoE kernel descriptor；旧记录「import 期代码挡住」，当前 coverage + patch clean，跳过 21/1253 个 YAML 条目。
- #19791：跳过未使用的多模态 encoder IPC handle；旧记录「import 期代码挡住」，当前 coverage + patch clean，跳过 142/1254 个 YAML 条目。

比较边界：原记录包含 23 个 Tier 1 和 3 个 Tier 2 shadow。旧 policy/DB 未按本次口径重跑，+4.9 个百分点包含 policy、coverage 和统计口径变化，不能全部归因于 PR #21418。当前候选 34/82（41.5%）不与旧 26/82 直接混算。

[历史逐项对比](historical-comparison.json) · [历史源记录](historical-source-records.json)

## 尚未命中的 52 个 PR：按当前首个阻塞原因分类

| 类别 | 数量 | PR |
|---|---:|---|
| Preflight · 非 Python residual | 22 | #17067, #17306, #18485, #18677, #18802, #19183, #19359, #19431, #19445, #19484, #19552, #19589, #19600, #19603, #19679, #19709, #19821, #19825, #19829, #19830, #19873, #19957 |
| Import execution · 导入执行无法界定 | 17 | #17476, #18813, #19213, #19306, #19553, #19583, #19598, #19643, #19696, #19731, #19746, #19748, #19801, #19811, #19885, #19899, #19935 |
| Coverage data · 缺少可归属记录 | 4 | #19178, #19539, #19892, #19896 |
| Patch compatibility · DB 版本冲突 | 4 | #19613, #19744, #19749, #19768 |
| Shared pytest state · 共享测试状态 | 3 | #19622, #19770, #19872 |
| Rule fallback · 规则主动回退 | 1 | #18496 |
| Stage resolution · 未解析出阶段 | 1 | #19810 |

Import execution 细分：

- 类定义 / 签名变更：6 个；#18813, #19306, #19748, #19811, #19899, #19935。
- 有副作用的模块语句：5 个；#19553, #19643, #19696, #19731, #19885。
- 导入时 binding 消费者：2 个；#19746, #19801。
- import 替换无法解析：2 个；#19583, #19598。
- import 目标不是可静态分析函数：1 个；#19213。
- 类 / 函数声明有歧义：1 个；#17476。

按当前首个阻塞原因互斥归类。解决首个原因后可能暴露其他阻塞；分类数量不代表可直接新增的命中数。#19810 的最终根因是 no_stages_resolved；no_residual 仅表示 coverage 被跳过。[完整分类](fallback-classification.json)。
