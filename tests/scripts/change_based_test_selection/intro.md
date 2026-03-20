# Change-Based Test Selection (CBTS) — 模块介绍

> 详细设计文档见 [DESIGN.md](./DESIGN.md)

---

## 要解决什么问题

TensorRT-LLM 的集成测试套件有 **600+ 个测试用例**，覆盖 15+ 模型架构和数十种可选特性，完整跑一次需要 **100+ GPU-小时**。

但在 nightly CI 中，两次运行之间通常只有一小部分代码发生了变化。每晚全量跑所有测试，大部分时间和GPU都花在了没有被改动过的代码路径上。

**核心问题：能不能只跑那些可能被今天的代码变更影响到的测试？**

---

## 怎么做的

一句话概括：**把代码变更映射到受影响的测试，然后去重和裁剪，在有限的GPU时间内最大化覆盖。**

```
306 个变更文件  →  487 个候选测试  →  44 个选中测试 (7.1%, ~8h)
    (git diff)       (影响分析)        (去重 + 预算裁剪)
```

### 三个核心组件

整个模块由三个核心组件协作完成：

```
                    ┌──────────────────┐
                    │   Impact Rules   │
                    │  (impact_rules.py)│
                    │                  │
                    │ 静态知识：        │
                    │ • 源码路径 → Tier │
                    │ • TestClass → 架构│
                    │ • 代表性测试集    │
                    └────────┬─────────┘
                             │ 提供映射规则
                             ▼
┌──────────────┐    ┌──────────────────┐
│    Parser    │    │    Selector      │
│  (parser.py) │───▶│  (selector.py)   │
│              │    │                  │
│ 构建测试数据库：│    │ 执行选择决策：    │
│ • 解析 test list│  │ • 匹配规则 → 候选 │
│ • AST 分析源码 │    │ • 去重 + 裁剪    │
│ • 提取 feature │    │ • 输出最终列表    │
└──────────────┘    └──────────────────┘
   提供"测试长什么样"     提供"选哪些测试"
```

**三者的关系可以这样理解：**

- **Impact Rules** 是"知识层"——由人静态维护，定义了源码和测试之间的影响关系。它回答：*改了这个文件，可能影响什么？*
- **Parser** 是"感知层"——自动解析测试源码（AST 分析 class 属性、装饰器、方法体），为每个测试构建一张丰富的"画像"（架构、feature、参数、GPU 要求等）。它回答：*每个测试长什么样、测了什么？*
- **Selector** 是"决策层"——拿着 Impact Rules 的映射规则和 Parser 构建的测试数据库，执行具体的选择、去重和裁剪逻辑。它回答：*在预算内，最终跑哪些？*

**一个具体的例子：** 假设有人改了 `tensorrt_llm/_torch/speculative/eagle3.py`

1. **Impact Rules** 告诉我们：这个文件匹配 `OPT_IN` 层级，feature 是 `eagle3`，对应 Config 类是 `Eagle3DecodingConfig`
2. **Parser** 已经知道：哪些测试的方法体里实例化了 `Eagle3DecodingConfig`，哪些测试的方法名里包含 `eagle3`
3. **Selector** 做决策：从 Parser 的数据库中，选出所有 feature 包含 `eagle3` 或 config_class 包含 `Eagle3DecodingConfig` 的测试，然后去重裁剪

---

整体流程分五步：

### 1. 影响分析：哪些测试可能被影响？

通过一套**分层 Impact Rules** 把变更文件映射到受影响的测试。规则按影响范围分为 6 个层级：

| Tier | 含义 | 选测试策略 |
|------|------|-----------|
| **CORE** | 核心基础设施（executor、model base class 等） | 跑一组精选的代表性测试集（~20条，覆盖所有架构×特性） |
| **DEFAULT_ON** | 默认开启的功能（attention、KV cache 等） | 跑显式配置了该功能的测试 + 代表性测试集 |
| **OPT_IN** | 需要显式开启的功能（Eagle3、量化、MoE 等） | 仅跑使用了该功能的测试 |
| **MODEL** | 模型特定代码（modeling_llama.py 等） | 跑该架构的所有测试 |
| **TEST** | 测试文件本身被改了 | 精确到被改动的 class/method |
| **IGNORE** | 文档、CI配置等 | 不跑 |

### 2. 补充选择

- **新增测试**：test list 中新增的行自动选中——新测试必须验证能通过
- **兜底**：源码文件没匹配到任何规则时，跑代表性测试集作为安全网

### 3. 排除已知失败

waives.txt 中的已知 bug 测试直接排除，不浪费 GPU 时间。

### 4. 两轮去重

很多测试在验证内容上有重叠：

- **第一轮**：同方法的参数化变体去重（8个变体 → 3-4个覆盖所有参数维度）
- **第二轮**：同 class 内的方法级去重（按 feature 覆盖度贪心选择）

### 5. 时间预算裁剪

默认 8 小时 GPU 预算。先删单条超长测试（>2h），再贪心删最容易被替代的测试。代表性测试集和新增测试受保护，不会被删。

---

## 三个关键设计决策

### 为什么放在 TensorRT-LLM 仓库内部？

- Impact rules 直接引用源码路径——文件重命名时可以在同一个 PR 中更新规则
- Parser 读取当前 HEAD 的测试定义——同一个 checkout，不会版本不一致
- 开发者添加新模型/特性时可以在同一个 PR 中维护规则，不需要访问外部 QA 仓库

### 为什么 Impact Rules 是静态维护的？

Impact rules 是整个模块的核心——它定义了"哪些代码影响哪些测试"。这块内容**有意由人来确认**，而不是让 AI 动态生成。原因是：

- **它是最值得人思考的部分：维护这些规则的过程，迫使我们理解项目的架构边界和测试覆盖的薄弱点**
- 自动检测机制会在每次运行时检查规则是否过时（新模型没有规则、架构没有代表性测试等），确保不会悄悄漂移

### Nightly + Weekly 的互补策略

| 频率 | 策略 | 目的 |
|------|------|------|
| **Nightly** | CBTS + 8h 预算 | 低成本、快反馈的早期回归检测 |
| **Weekly** | 全量测试套件 | 安全网——捕获 CBTS 可能遗漏的问题 |

---

## 自动维护告警

每次运行时自动检查配置是否需要更新：

```
⚠ Maintenance warnings (4 issues):
  [RULE] 12 source file(s) matched no impact rule
  [ARCH] 3 test class(es) have no architecture mapping
  [REP]  2 architecture(s) have no representative test
  [FEAT] 87 tests have no extracted features
```

这些告警确保随着代码库演进，CBTS 的规则不会悄悄过时。

---

## 用法

```bash
# Nightly CI 一行调用
python -m tests.scripts.change_based_test_selection.cli \
    --base-ref $LAST_GOOD_SHA --test-list llm_function_core \
    --time-budget 8h -o selected_tests.txt

# 看每条测试为什么被选中
python -m tests.scripts.change_based_test_selection.cli \
    --base-ref main --explain

# 手动指定变更文件（不用 git diff）
python -m tests.scripts.change_based_test_selection.cli \
    --files tensorrt_llm/models/llama/model.py
```
