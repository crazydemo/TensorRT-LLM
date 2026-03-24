# CBTS 三方案整合会议大纲

**日期**: 2026-03-24
**目标**: 对齐三套 Change-Based Test Selection 方案的整合策略

---

## 1. 目标对齐（10 min）

### 1.1 目标场景

**Nightly CI**：每晚对当天合入的 commit range 做测试选择，替代全量运行。

### 1.2 确定性要求

**强确定性**：相同 commit range 必须产出相同 test list。排除 LLM 推理等非确定性方法进入 CI 关键路径。

### 1.3 核心 KPI

| KPI | 定义 | 目标 | 度量方法 |
|-----|------|------|---------|
| **选择率** | CBTS 选出的测试数 / 全量测试数 | < 15% | 每次 nightly CI 自动统计 |
| **漏选率** | 真实 regression 中未被选出的比例 | < 5% | 每周对比 weekly full run 结果（排除 flaky） |
| **已知 Regression 捕获率** | 已知 NVBugs regression 被 CBTS 覆盖的比例 | **100%** | 上线前用历史 regression 数据回测 |
| **CBTS 自身延迟** | 输入 changed files → 输出 test list 的耗时 | < 30s | 每次 CI 自动记录 |

### 1.4 度量落地方案

| 层次 | 频率 | 方法 | 额外成本 |
|------|------|------|---------|
| 持续监控 | 每次 nightly CI | 自动统计选择率、覆盖 module 数 | 零 |
| 周级验证 | 每周 | 对比 weekly full run fail 集合，计算 escape rate | 零（复用已有 weekly full run） |
| 上线前回测 | 一次性 | 回放过去 3 个月 nightly commit range + 全量结果 | 中（需搭建回测框架） |

---

## 2. 三方案技术 Review（20 min）

### 2.1 方案概览

| 维度 | A — 静态规则 (crazydemo) | B — LLM 推理 (stsun) | C — 依赖图分析 (dlswqa) |
|------|---|---|---|
| **核心方法** | 200+ 条 fnmatch 规则，人工映射文件路径到 6 级 tier | 两阶段 Claude pipeline，LLM 语义推理选测试 | AST import 分析 + nanobind 解析，自动构建三层反向依赖图 |
| **映射粒度** | method 级（可选到参数化变体） | test ID 级 | 目录/模块级 |
| **确定性** | 确定 | 非确定 | 确定 |
| **延迟** | 秒级 | 30-60s+ | 秒级 |
| **维护成本** | 高（规则需持续人工更新） | 低（prompt 驱动） | 低（自动推导） |
| **C++ 支持** | 通过规则覆盖 | LLM 语义理解 | nanobind #include 解析 |
| **去重/时间预算** | 有（贪心去重 + GPU 时间预算裁剪） | 无 | 无 |
| **安全网** | Representative Coverage Set (~17 测试) | 无 | 高影响模块 RUN_ALL |
| **代码量** | ~2000+ 行 | ~600 行 | ~800 行 |

### 2.2 Pros & Cons

#### 方案 A — 静态规则

| Pros | Cons |
|------|------|
| 语义级特性识别：区分 CORE / DEFAULT_ON / OPT_IN，精确控制选择范围 | 200+ 条规则需持续人工维护，新模块/特性必须手动添加 |
| 三级 AST parser 解析到参数化变体级别（如 `test_bfloat16[FLASHINFER-True]`） | 路径重构后 fnmatch 规则 stale，需人工修复 |
| 成熟的去重算法 + GPU 时间预算裁剪（默认 8h） | 人脑难以穷举所有间接依赖，新增模块默认漏选 |
| Representative Coverage Set 保证全架构最低覆盖 | 规则质量完全依赖维护者对代码的理解 |
| Additive-only 优化：纯加新函数时跳过广泛测试 | |
| `--explain` 模式强可解释性 | |
| `--suspect` 反向分析辅助 regression 定位 | |

#### 方案 B — LLM 语义推理

| Pros | Cons |
|------|------|
| 语义理解能力强，无需维护映射规则 | **非确定性：相同 diff 可能产出不同结果，不满足 nightly CI 要求** |
| 零历史数据要求，开箱即用 | 30-60s+ 延迟 + LLM API 费用 |
| 三级优先级（Must Run / Should Run / Regression Sweep） | API 可用性风险：限流/宕机会阻塞 CI |
| 代码简洁，易于理解和修改 | 无反馈闭环，无法从测试结果中学习改进 |
| | diff 大小限制（80K 字符截断） |
| | 无去重、无时间预算控制 |

#### 方案 C — 依赖图分析

| Pros | Cons |
|------|------|
| 零人工维护：依赖图完全自动构建 | **结构性缺陷：不追踪 source → source 内部 import 链，导致对 `_torch/*` 模块退化为 RUN_ALL** |
| 追踪真实代码引用关系（ground truth） | 目录级粒度，无法精选参数化变体 |
| 跨语言桥接（nanobind C++ → Python） | C++ 追踪仅限 nanobind 路径，断链于 binding → 内部模块 |
| 新增模块/重构路径自动生效 | 不处理 build/config 文件（CMakeLists.txt, setup.py） |
| 轻量：纯 Python，单依赖（PyYAML） | 无去重、无时间预算、无安全网 |
| | 只追踪静态 import，动态加载不可见 |

### 2.3 以 Commit aac66bd3 为例的实际对比

**Commit**: Cache FlashMLA tile-scheduler metadata (#12161)，涉及 C++ kernel、nanobind 绑定、Python attention backend、executor、speculative decoding 共 11 个文件。

| 维度 | A | C |
|------|---|---|
| 选择范围 | ~30-50 个精选测试 | **RUN_ALL（600+ 测试）** |
| attention 变更 | 选 attention feature 相关变体 | 被 RUN_ALL 淹没 |
| mtp 变更 | 仅选 MTP spec decode 测试 | 被 RUN_ALL 淹没 |
| mlaKernels | 精确选 deepseek 架构测试 | 被 RUN_ALL 淹没 |
| C++ 追踪 | 通过规则直接匹配 | nanobind → bindings → **断链** |
| GPU 时间 | ~8h | 100h+ |

**根因**：集成测试通过高层 API（`from tensorrt_llm.llmapi import LLM`）间接使用底层模块，C 只追踪直接 import，中间的内部依赖链不可见，最终退化到 `tensorrt_llm` 根模块触发 RUN_ALL。

---

## 3. 整合策略讨论（20 min）

### 3.1 推荐架构

以 C 的自动依赖分析为基座，增强后融合 A 的精选和预算能力：

```
git diff → changed files
  │
  ├─ cpp/**           → Layer 1 (C): C++ → Python bindings via nanobind
  │                        │
  ├─ tensorrt_llm/**  → Layer 1.5 (新): 内部模块传递依赖 → API 边界模块
  │                        │
  │                        ▼
  │                    Layer 2 (C): API 边界模块 → test 文件
  │                        │
  │                        ▼
  │                    Layer 3 (C): test 文件 → test list
  │
  ├─ build/config     → A 的 impact_rules 补充层（少量规则）
  ├─ defs/**          → 直接 Layer 3
  └─ test_lists/**    → 直接包含

  → 自动 tier 推断（传递扇出度）
  → A 的 parser 做 method/参数级精选
  → A 的去重 + 时间预算裁剪
  → A 的 Representative Coverage Set 兜底
```

### 3.2 C 的关键增强项

| 增强 | 内容 | 优先级 | 工作量 |
|------|------|--------|--------|
| Layer 1.5 内部依赖图 | 扫描 tensorrt_llm/ 所有 .py 的 import，建 source → source 依赖图，算反向传递闭包 | P0 | 2-3 天 |
| API 边界层 | 定义 llmapi, executor, models 等为边界模块，控制传递爆炸 | P0 | 半天 |
| 自动 tier 推断 | 根据模块传递扇出度自动分级（>50% → CORE, >15% → DEFAULT_ON, >0 → OPT_IN） | P1 | 1 天 |
| 接入 A 的去重 + 时间预算 | 贪心参数去重 + GPU 时间预算裁剪 | P1 | 2 天 |
| 接入 A 的 parser | method/参数化变体级精选 | P1 | 2 天 |
| 少量语义规则 | model arch 映射、build/config 文件覆盖 | P2 | 持续 |
| CI 健康检查 | 解析覆盖率、孤儿检测、空映射检测、nanobind 完整性 | P2 | 1 天 |

### 3.3 B 的定位

由于 **非确定性不满足 nightly CI 要求**，B 不进入 CI 关键路径。可选作离线验证层：对相同 diff 运行 B，对比 C+A 的选择结果，差异报告用于发现盲区。需评估 ROI 后决定是否投入。

### 3.4 分工与时间线

#### 角色分工

| 角色 | Owner | 职责范围 |
|------|-------|---------|
| **核心引擎** | A 的 owner (Yan) | 增强 C 方案的依赖图（Layer 1.5、API 边界层、自动 tier），移植去重和时间预算裁剪 |
| **Functional Test** | A 的 owner (Yan) | functional test 相关的 test pool parser、impact rules、Representative Coverage Set |
| **Performance Test** | C 的 owner | perf test 相关的 test pool parser、perf 特有的选择逻辑和预算策略 |
| **离线验证层** | B 的 owner | 改造 B 为离线验证工具，增强 prompt 中的人类先验知识（test definition 理解、AST 语法树解析结果注入） |

#### Phase 时间线

| Phase | 内容 | 时间 |
|-------|------|------|
| Phase 1 | 统一仓库 + 核心引擎增强（Layer 1.5 + API 边界层） | Week 1-2 |
| Phase 2 | Functional / Perf 分别接入 + 去重/时间预算 | Week 3-4 |
| Phase 3 | 整合联调 + 历史回测验证 + CI pipeline 集成 | Week 5-6 |

---

### 3.5 Jira Checklist

#### Epic 1: CBTS 三方案整合

---

#### 3.5.1 核心引擎 — Owner: @Ivy Zhang

ETA: before next Wednesday

**Story: 增强 C 方案依赖图**

- [ ] 统一仓库选址：放 Jenkins repo（`trt_jenkins/scripts/cbts/`），核心引擎运行时扫描 TRT-LLM checkout，少量静态配置（API 边界、补充规则、Representative Set）放 `config.yaml`，合并 A/C 代码到统一目录结构
- [ ] Layer 1.5 — 内部模块依赖图：扫描 `tensorrt_llm/**/*.py`，用 AST 提取内部 import，建 source → source 依赖图
- [ ] Layer 1.5 — 反向传递闭包：BFS 计算每个模块的所有传递上游模块
- [ ] API 边界层定义：确定 `llmapi`, `executor`, `models` 等边界模块列表，限制传递爆炸
- [ ] 修复 C++ 断链：Layer 1 的 nanobind binding 模块通过 Layer 1.5 连接到内部 Python 模块
- [ ] 自动 tier 推断：根据传递扇出度自动分级（CORE / DEFAULT_ON / OPT_IN）
- [ ] 集成测试：用 commit aac66bd3 等已知 case 验证增强后不再退化为 RUN_ALL

**Story: 移植去重与时间预算**

- [ ] 移植 A 的贪心参数去重算法（两轮：参数化变体去重 + intra-class method 去重）
- [ ] 移植 A 的时间预算裁剪（outlier 剔除 + 价值排序裁剪 + 保护 Representative Coverage Set）
- [ ] 统一 `.test_durations` 数据源，确保 functional 和 perf test 都有时长数据
- [ ] 移植 A 的 `--explain` 可解释性输出

---

#### 3.5.2 Functional Test — Owner: @Ivy Zhang

Estimated Time: 2 weeks

**Story: Functional test 选择与适配**

- [ ] 合并 functional test pool parser：统一 A/C 的 `.txt` + `.yml` test list 解析逻辑
- [ ] 移植 A 的三级 AST parser（class 属性 → decorator → method body）用于 functional test 精选
- [ ] 适配 functional test 的 impact rules：从 A 的 200+ 条规则中提取 functional 相关子集，作为非 import 依赖的补充层
- [ ] 定义 functional test 的 Representative Coverage Set
- [ ] Functional test 端到端验证：对历史 nightly commit range 回测，确认选择率和漏选率达标

---

#### 3.5.3 Performance Test — Owner: @Ruodi Lu

Estimated Time: 2 weeks

**Story: Performance test 选择与适配**

- [ ] Perf test pool parser：解析 perf 相关的 test list（`llm_perf_core.yml` 等），处理 perf 特有的格式和标记
- [ ] Perf test 依赖映射：确认 perf test 文件的 import 关系是否被 Layer 2 正确追踪（perf test 可能有不同的 import 模式）
- [ ] Perf test 时间预算策略：perf test 单个耗时通常远大于 functional test，需要独立的预算参数和裁剪策略
- [ ] Perf test impact rules：perf 特有的非 import 依赖规则（如 benchmark config 变更、perf reference 数据变更）
- [ ] Perf test 端到端验证：对历史数据回测，确认 perf test 的选择率和漏选率

---

#### 3.5.4 离线验证层 — Owner: @Stanley Sun

> **注意**：此部分涉及 LLM skills/agent，与确定性引擎有本质区别。需额外讨论 output 约束、中间层约束、agent 波动性和代码漂移问题。验证周期预计比核心引擎更长，但一旦 workflow 走通，形成的 LLM-in-the-loop pattern 对后续项目（如其他 repo 的 CBTS、自动化 bug triage 等）具有广泛复用价值。

**Story: Output 约束设计**

LLM 输出是自由文本，如果不做 schema 约束，差异报告格式每次不同，无法建立自动化分析。

- [ ] 定义 Stage 1 输出 JSON schema：明确 `affected_components`, `affected_models`, `change_type`, `risk_level` 等字段的枚举值和类型约束
- [ ] 定义 Stage 2 输出 schema：test selection 结果必须是结构化列表（test_id + tier + reason），而非自由 Markdown
- [ ] 定义差异报告 schema：每条差异包含 `test_id`, `selected_by` (unified/llm/both), `root_cause_category` (枚举)
- [ ] 输出校验层：LLM 输出后做 schema validation，不合规则重试或降级处理

**Story: 中间层约束设计**

Stage 1 的 JSON 是 Stage 2 的输入，如果 Stage 1 的分类标准漂移（如 affected_components 的语义变了），Stage 2 会级联放大错误。

- [ ] Stage 1 → Stage 2 接口契约：定义 component 分类的固定枚举表（从依赖图的模块列表自动生成，而非让 LLM 自由发挥）
- [ ] Stage 1 输出校验：检查 affected_components 是否都在已知模块列表中，未知值触发 warning
- [ ] 中间结果可视化：Stage 1 输出独立存储，方便 debug Stage 2 的选择逻辑

**Story: Agent 波动性与代码漂移控制**

同一个 prompt 在不同时间运行可能产出不同结果（模型版本更新、采样随机性）；TRT-LLM 代码持续演进但 LLM 训练数据是 frozen 的。

- [ ] 波动性基线测试：同一 diff 运行 N 次（如 10 次），统计输出一致性比例，建立波动性 baseline
- [ ] 漂移检测机制：每周对同一组 golden diff 运行验证，如果输出偏离 baseline 超过阈值则告警
- [ ] 知识注入策略：将最新的模块列表、test definition 结构、架构映射等作为 prompt context 注入，减少对 LLM 训练数据中过期知识的依赖
- [ ] Prompt 版本管理：prompt 模板纳入版本控制，变更需 review，关联 changelog

**Story: 功能实现**

- [ ] 增强 Stage 1 prompt：注入 test definition 的结构化知识（AST 解析出的特性标签、参数化维度），而非仅依赖 test ID 字面语义
- [ ] 增强 Stage 2 prompt：注入依赖图拓扑信息（哪些模块受影响、影响的传递路径），让 LLM 在有结构化上下文的基础上推理
- [ ] 差异报告生成：对比 B 的选择 vs C+A 整合方案的选择，按 schema 输出结构化差异
- [ ] 差异分析分类：对每个差异标注原因（依赖图缺边 / 规则遗漏 / LLM 过选 / LLM 漏选）
- [ ] 定期运行机制：接入 CI 作为 nightly 的异步后处理步骤（不阻塞 CI）
- [ ] ROI 评估报告：统计 B 发现了多少 C+A 的真实盲区 vs 多少是 B 自身的误判，决定是否持续投入

---

#### 3.5.5 整合与验证 — Owner: All

**Story: 整合联调**

- [ ] Functional + Perf 合并输出：统一 CLI 接口，支持 `--scope functional` / `--scope perf` / `--scope all`
- [ ] 统一 JSON 输出格式：兼容 CI pipeline 消费
- [ ] CI pipeline 集成：接入 nightly CI，替代全量测试选择

**Story: 回测与度量**

- [ ] 搭建回测框架：收集过去 3 个月的 nightly commit range + 全量测试结果
- [ ] 历史回测运行：对每个 commit range 回放 CBTS，计算 precision / recall / escape rate
- [ ] 已知 regression 验证：从 NVBugs 拉取已知 regression，确认 catch rate = 100%
- [ ] KPI 持续监控接入：每次 nightly CI 自动统计选择率，每周对比 weekly full run 计算漏选率

**Story: CI 健康检查**

- [ ] test list 解析覆盖率检查：所有行都能被成功解析
- [ ] 孤儿 test 检测：有无 test 文件不在任何 test list 中
- [ ] 空映射检测：有无 source 模块没被任何 test import
- [ ] nanobind 完整性检查：C++ 目录是否都能映射到 Python 模块

---

## 4. Test Architecture Standardization（10 min）

> **基础性问题**：functional test 的架构和命名不统一，直接影响 CBTS parser 复杂度、数据库历史追踪的连贯性、LLM 验证层的准确性。需要尽快定义标准，新 case 按标准写，老 case 逐步迁移。

### 4.1 当前问题

| 不一致项 | 示例 | 对 CBTS 生态的影响 |
|---------|------|-------------------|
| class 命名 | `TestLlama3_8B_BF16` vs `TestLlama_3_1_8b_Instruct_BF16` vs `TestBfloat16` | TESTCLASS_TO_ARCH 需 80+ 手写映射 |
| 参数化方式 | `@parametrize_with_ids` vs `@pytest.mark.parametrize` vs 硬编码独立方法 | parser 需三种提取路径 |
| 特性编码 | 方法名里 (`test_fp8_kv_cache`) vs decorator 参数 vs method body 里的 Config | 特性标签提取脆弱且不完整 |
| 模型引用 | `MODEL_NAME` class 属性 vs inline 传参 vs fixture | 架构检测需多种 fallback |
| test ID 稳定性 | 重构 = test ID 变化 | 数据库历史断裂（flaky 趋势、regression 追踪全断） |

### 4.2 策略：渐进式标准化

- **Phase 1（现在）**：定义标准 + CI lint 强制新 case 合规
- **Phase 2（持续）**：按架构批次迁移老 case（每次一个架构一个 PR）
- **Phase 3（目标）**：淘汰非标准 case

### 4.3 Jira Checklist

#### Epic 2: Test Architecture Standardization

**Story: 定义标准**

- [ ] 起草 test 命名和架构规范文档：class 命名（`Test{Arch}_{Size}_{Precision}`）、参数化方式（统一 `@parametrize_with_ids`）、模型引用（`ARCHITECTURE` class 属性，受控枚举）
- [ ] 定义受控词汇表：architecture name、feature parameter name、precision name 的标准枚举值
- [ ] 提供 `AccuracyTestBase` / `PerfTestBase` 基类：强制 `ARCHITECTURE`, `MODEL_NAME`, `PRECISION` 必填属性
- [ ] 提供 test 模板 / cookiecutter：从模板生成新 test 文件

**Story: CI 强制**

- [ ] CI lint 规则：新加到 `defs/` 下的 test 文件必须通过命名规范检查，不合规则 PR fail
- [ ] PR review checklist：增加 test 命名规范检查项

**Story: 渐进迁移**

- [ ] 现有 test 合规度审计：按合规/部分合规/不合规分类
- [ ] Test ID alias mapping 文件：维护 `{old_test_id: new_test_id}` 映射表，数据库用此保持历史连贯性
- [ ] 按架构批次迁移计划：llama → deepseek → qwen → ... 每架构一个 PR
- [ ] 非标准 test 淘汰时间线

---

## 5. Open Items：扩展应用与平台整合（15 min）

依赖图不仅服务于测试选择，其拓扑结构和因果分析能力可以扩展到 QA workflow 的多个环节。**最重要的议题**：这些扩展的输出不应该是散落的 JSON 文件，需要统一的数据平台和交互界面。

### 5.1 扩展应用优先级与 Jira Checklist

| 扩展 | 优先级 | 理由 | 历史数据依赖 |
|------|--------|------|-------------|
| **Flaky Confidence** | **P0** | V1 不需要历史数据，当前 pipeline 立即可用 | V1: 不需要 / V2: 增强项 |
| **Module Health Dashboard** | **P1** | 高诊断价值，在 bug 发生前识别薄弱环节 | 需要（flaky 聚合 + git 变更频率） |
| **Dead Test Detection** | **P1** | 零成本质量提升，节省 CI 资源 | 不需要 |
| **Regression Commit Proposal** | **P1** | functional test 直接可用，perf test 需额外数据 | 需要（git log + 测试结果） |
| **Unified QA Platform** | **P2** | 所有扩展的集成层 | 是数据存储本身 |

---

#### 5.1.1 Flaky Confidence — P0

**立即可用**：V1 只需要单次运行的依赖图可达性判断，不依赖历史数据。

```
test_X fail 了 → 查依赖图 → changed files 是否在依赖链上？
  不可达 → flaky_score = HIGH
  距离 > 3 → flaky_score = MODERATE
  距离 ≤ 2 → flaky_score = LOW（很可能真实 regression）
```

**输出持久化 — 需讨论**：V1 写 JSON 文件（零依赖），V2 迁移到数据库（支持历史聚合）。

- [ ] V1 — 单次 flaky 评分：测试结束后查询依赖图，输出 `{test_id, flaky_score, distance, reachable_modules}` JSON
- [ ] 接入 CBTS escape rate 计算：`flaky_score > 0.7` 自动排除
- [ ] 输出格式：`flaky_report.json` 写在测试结果旁，同时写入 Jenkins 报告摘要
- [ ] V2 — 历史增强：积累多周 flaky report 到数据库，跨周对比提升评分准确性
- [ ] 自动建议 UNSTABLE 候选：持续高 flaky_score 的 test → 建议加入 waives.txt

---

#### 5.1.2 Module Health Dashboard — P1

- [ ] 每模块 flaky 率：聚合下游 test 的 flaky_score → flaky 率高的模块需要加固测试
- [ ] 每模块测试覆盖密度：依赖图中每个 source module 的 test 数量 → 覆盖不足的模块是风险区
- [ ] 每模块变更频率：从 git log 统计模块变更频次 → 高频变更 + 低覆盖 = 最高风险
- [ ] 输出格式：JSON report，可接入前端看板（见 5.2）

---

#### 5.1.3 Dead Test Detection — P1

- [ ] 孤儿 import 检测：test 文件 import `tensorrt_llm.foo` 但 `tensorrt_llm/foo/` 已不存在 → test 在测试死代码
- [ ] Stale test list 条目：test list 引用 `accuracy/test_old_model.py` 但文件不存在于 `defs/`
- [ ] 冗余 test 检测：多个 test 的依赖集合完全相同（相同 import、相同 feature）→ 合并候选
- [ ] 输出：dead/stale/redundant test 报告，附推荐操作（删除、更新、合并）

---

#### 5.1.4 Regression Commit Proposal — P1

**注意**：对 functional test 直接可用；对 performance regression，依赖图仅能缩小范围，确认仍需性能分析工具（nsys, NCU）数据。

- [ ] 核心算法：given failed test + commit range，反向查依赖图找到哪些 commit 改了 test 依赖链上的模块，按依赖距离排序
- [ ] 集成 `git log`：每个 commit 的 changed files 通过依赖图计算与 failed test 的 overlap score
- [ ] Functional test 输出：`{commit_sha, author, changed_modules, distance, overlap_score}` 排序列表
- [ ] Performance test 输出：同 functional + 标注"建议对 top N 嫌疑 commit 运行 perf profiler"
- [ ] Jenkins 集成：测试失败时作为 post-test step 输出 suspect report
- [ ] 整合 A 方案的 `--suspect` 功能

---

### 5.2 平台整合 — 最重要的架构议题

> 所有扩展应用（CBTS 选择结果、flaky 评分、regression suspect、模块健康、dead test）的输出需要汇聚到一个统一平台，否则每个扩展都是孤岛。

#### 数据层

| 数据源 | 量级 | 典型查询 |
|--------|------|---------|
| CBTS 选择结果 | 每次 nightly | "上周二选了哪些测试？" |
| Flaky 评分 | 每 test 每次运行 | "test_X 过去 4 周的 flaky 趋势" |
| Regression suspects | 每个 failed test | "谁最可能 break 了 test_Y？" |
| 模块健康 | 每模块，每周更新 | "哪些模块的测试健康度在恶化？" |
| Dead test 报告 | 每周更新 | "多少 CI 时间浪费在 dead test 上？" |

**数据库建议**：V1 用 SQLite（零运维、文件级、单团队规模足够），多团队访问时迁移 PostgreSQL。

#### 前端方案

| 阶段 | 方案 | 优势 | 适用场景 |
|------|------|------|---------|
| V1 | **Streamlit** | 几天内出原型，Python 原生，交互式图表 | 团队内部快速验证 |
| V2 | **AI Plugin（网页端）** | 对话式交互（类似 Cursor），可嵌入现有 web portal，自然语言查询 QA 数据 | 生产级团队工具 |

AI Plugin 的价值在于可以这样交互：
- "test_X 昨晚为什么 fail 了？是 flaky 还是真实 regression？"
- "这个月哪些模块的测试健康度最差？"
- "过去 6 周 CBTS 的 escape 趋势怎么样？"
- "attention backend 的 regression 应该找谁？"

**这把依赖图 + 历史数据变成了一个对话式 QA 知识库**。

#### Story: 统一 QA 分析平台

**V1 — Streamlit + SQLite**

- [ ] 数据库 schema 设计：覆盖所有扩展输出（selections, flaky, suspects, health, dead tests）
- [ ] 数据写入 pipeline：每个扩展执行后将结果写入 DB（Jenkins post-step）
- [ ] Streamlit 看板：总览页（KPI 趋势）、flaky 探索器、regression 调查器、模块健康热力图、dead test 报告
- [ ] 历史趋势图：选择率趋势、escape rate 趋势、各模块 flaky 率趋势

**V2 — AI Plugin 集成**

- [ ] API backend：REST/GraphQL over QA database
- [ ] AI Plugin 适配：连接 QA database，支持自然语言查询
- [ ] 对话式 workflow："调查 test_X 的失败" → 自动拉取 flaky score + regression suspect + 模块健康 + 依赖链
- [ ] 嵌入团队 web portal：全员可访问，无需本地部署
