# Change-Based Test Selection (CBTS)

## 零、问题背景

### 现状

TensorRT-LLM 的 QA 集成测试套件包含 **600+ 测试用例**，覆盖 15+ 模型架构、数十种可选特性（推测解码、量化、MoE、分离式推理等）和多种 GPU 配置。完整运行需要 **100+ GPU-小时**。

在典型的 nightly CI 周期中，两次运行之间只有一小部分代码发生变化。每晚跑全量测试浪费 GPU 资源并延迟反馈——工程师等待数小时的结果，其中大部分是针对未被修改的代码路径。

### 核心问题

**如何选出最小的测试子集，同时仍能捕获当天变更引入的回归？**

这分解为几个子问题：

1. **影响分析** — 给定一组变更的源文件，哪些测试*可能*受影响？修改 `modeling_llama.py` 不应触发 DeepSeek 测试；修改核心 executor 则应触发广泛覆盖。

2. **边际递减** — 很多测试在验证内容上有重叠。如果已经选了 5 个 Llama 测试覆盖了 FP8、beam search 和推测解码，第 6 个参数略有不同的 Llama 测试增加的价值很小。

3. **资源预算** — 即使经过智能选择，候选集可能仍超出可用 GPU-小时预算。需要有原则的方式裁剪而不丢失关键覆盖。

4. **已知失败** — 已在 `waives.txt` 中跟踪的测试（正在调查的已知 bug）不应消耗可以用于测试新/未知回归的预算。

5. **维护漂移** — 随着代码库演进（新模型、新特性、文件重命名），选择规则可能悄悄过时。需要自动检测配置漂移。

### 解决方案：Change-Based Test Selection

CBTS 通过分层影响规则系统将源码变更映射到受影响的测试，然后应用去重和预算裁剪生成精简的测试列表：

```
306 个变更文件  →  487 个候选测试  →  44 个选中测试 (7.1%, ~8h)
    (完整 diff)      (影响分析)        (去重 + 预算裁剪)
```

### 部署模型

CBTS **有意放在 TensorRT-LLM 仓库内部**（而非外部 CI 仓库），因为：

- **影响规则直接引用源码路径** — `tensorrt_llm/_torch/models/modeling_llama*.py` 必须与实际文件位置保持同步。同仓库意味着重命名文件的 PR 可以在同一个 commit 中更新规则。
- **Parser 读取当前 HEAD 的测试定义** — 测试类属性、parametrize 装饰器和 config class 都从同一个 checkout 中解析。
- **维护告警立即生效** — 当新增模型没有对应的影响规则时，下一次 CBTS 运行就会标记出来。如果在独立仓库中，这只能事后发现。
- **开发者自助维护** — 贡献者添加新模型或特性时可以在同一个 PR 中更新 `impact_rules.py`，无需访问单独的 QA 仓库。

Jenkins CI 流水线只需要一行调用：

```bash
python -m tests.scripts.change_based_test_selection.cli \
    --base-ref $LAST_GOOD_SHA --test-list llm_function_core \
    --time-budget 8h -o selected_tests.txt
```

### 预期使用方式

| 频率 | 策略 | 目的 |
|------|------|------|
| **Nightly** | CBTS + 8h 预算 | 低 GPU 成本的早期回归检测 |
| **Weekly** | 全量测试套件 | 安全网——捕获 CBTS 遗漏的问题 |

---

## 一、整体流程

```
git diff --name-only <base-ref>...HEAD
              ↓
         changed_files (变更文件列表)
              ↓
    ┌─────────────────────────┐
    │  impact_rules 匹配       │  每个文件 fnmatch 所有规则
    │  → (file, rule) pairs   │
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  additive-only 过滤      │  高影响 tier 的文件如果 diff 只是
    │                         │  扩展/纯新增定义 → 跳过该规则
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  selector 按 tier 选测试  │  CORE / DEFAULT_ON / OPT_IN /
    │                         │  MODEL / TEST 各有策略
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  补充选择                 │  NEW_TEST: test list 新增行
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  排除已知失败              │  过滤掉当前 waives.txt 中的测试
    │                         │  释放资源给更多有效测试
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  两轮去重                 │  第一轮: 同方法的参数化变体去重
    │                         │  第二轮: 同 class 的方法级去重
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  Time Budget 裁剪        │  Phase 0: 删超长单条测试
    │  (默认 8h，可选)          │  Phase 1: 贪心删最低价值测试
    └─────────────────────────┘
              ↓
         按 reason 分组，输出 test ID 列表
```

---

## 二、Parser — 构建测试数据库

### 2.1 数据源

| 数据源 | 路径 | 作用 |
|--------|------|------|
| Test list 文件 | `tests/integration/test_lists/qa/*.txt` | 全量 pytest node ID，如 `llm_function_core.txt` |
| Test definition 源码 | `tests/integration/defs/**/*.py` | 测试类/方法的 Python 源文件 |
| TESTCLASS_TO_ARCH | `impact_rules.py` 中的 dict | TestClass → 模型架构映射 |

默认加载的 test list：
- `llm_function_core`
- `llm_function_core_sanity`
- `llm_function_rtx6k`
- `llm_function_multinode`
- `llm_function_stress`

### 2.2 解析过程

**Step 1: 解析 test list (.txt)**

每行是一个 pytest node ID，格式：
```
accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_nvfp4[moe_backend=CUTLASS-...]
```

解析出：
- `test_file`: 相对路径
- `test_class`: 类名
- `test_method`: 方法名
- `params`: key=value 参数 (从 `[...]` 中解析)
- `raw_params`: 原始括号内容
- `test_lists`: 属于哪些 test list

**Step 2: AST 解析 test definition (.py)**

三级分析：

| 级别 | 范围 | 提取内容 |
|------|------|----------|
| **L1 (类级)** | class 体 | `MODEL_NAME`, `MODEL_PATH` 类属性 |
| **L2 (装饰器级)** | method 装饰器 | `@skip_pre_hopper` → min_sm=90；`@skip_less_device(4)` → min_gpu=4；`@parametrize(...)` → 参数维度和值 |
| **L3 (方法体)** | method body | 扫描 Config 类实例化调用，如 `KvCacheConfig(...)`, `Eagle3DecodingConfig(...)` |

**Step 3: 参数解析 (parametrize)**

两种 parametrize 风格：

1. **自定义 ID** — `@pytest.mark.parametrize("a,b", [...], ids=["name1", "name2"])`
   - 通过 `ids` 列表建立 `id → {param: value}` 映射
   - 匹配 raw_params 中的子串来反查参数值

2. **自动生成 ID** — `@pytest.mark.parametrize("tp_size, ep_size, ...", [(4,4,True,...)])`
   - pytest 自动拼接 `str(val)` 用 `-` 连接，如 `4-4-True-True-True`
   - 遍历 all_values，生成 auto_id 与 raw_params 匹配

笛卡尔积情况（多个 `@parametrize` 叠加）：test ID 是各装饰器子 ID 用 `-` 拼接，逐个匹配消耗 remaining 字符串。

**Step 4: 特征提取 (feature extraction)**

从三个来源提取 feature 标签：

| 来源 | 示例 |
|------|------|
| 方法名模式匹配 | `test_eagle3_vswa_reuse_4gpus` → `{eagle3, vswa, reuse, 4gpu}` |
| 参数 key/value | `cuda_graph=True` → `{cuda_graph}`；`moe_backend=CUTLASS` → `{moe_backend, moe_backend:cutlass}` |
| Config 类 (L3) | `Eagle3DecodingConfig` → `{eagle3}`；`CudaGraphConfig` → `{cuda_graph}` |
| GPU 规模 | `_4gpu` / `tp4` in method name → `{4gpu}` / `{tp4}` |

### 2.3 最终数据结构 — TestEntry

```python
@dataclass
class TestEntry:
    test_id: str          # "accuracy/test_llm_api_pytorch.py::TestX::test_y[params]"
    test_file: str        # "accuracy/test_llm_api_pytorch.py"
    test_class: str       # "TestDeepSeekV3Lite"
    test_method: str      # "test_nvfp4"

    # L1
    model_name: str       # "deepseek-ai/DeepSeek-V3-Lite"
    arch: str             # "deepseek_v2" (来自 TESTCLASS_TO_ARCH)

    # L2
    min_sm: int           # 90 (Hopper+)
    max_sm: int           # 999
    min_gpu_count: int    # 4
    min_gpu_memory: int   # 0 (MB)
    param_dimensions: list[str]   # ["tp_size", "ep_size", "moe_backend"]
    resolved_params: dict         # {"tp_size": "4", "moe_backend": "CUTLASS"}

    # L3
    config_classes: set[str]      # {"MoeConfig", "KvCacheConfig"}

    # 综合
    features: set[str]            # {"nvfp4", "moe", "moe_backend:cutlass"}
    test_lists: set[str]          # {"llm_function_core"}
```

整个数据库是 `dict[test_id, TestEntry]`，约 600+ 条。

---

## 三、Impact Rules — 变更文件到影响层级的映射

### 3.1 Tier 定义

| Tier | 名称 | 含义 | 选测试策略 |
|------|------|------|-----------|
| **0** | CORE | 核心基础设施，改动可能影响所有测试 | 跑 **Representative Coverage Set** (约20条覆盖所有架构×特性×GPU规模) |
| **1** | DEFAULT_ON | 默认开启但可配置的功能 | 跑**显式配置了该功能**的测试 + Representative Set |
| **2** | OPT_IN | 需要显式开启的功能 | **仅**跑使用了该功能的测试 (通过 config_class 或 feature tag 匹配) |
| **3** | MODEL | 模型特定代码 | 跑该**架构的所有测试** |
| **4** | TEST | 测试文件本身被改 | 跑该**文件中被改动的 class** 的测试 (通过 git diff hunk 定位) |
| **5** | IGNORE | 无影响 (文档、CI 配置等) | 不选任何测试 |

### 3.2 规则格式

```python
ImpactRule(
    pattern="tensorrt_llm/_torch/speculative/eagle3.py",  # glob 模式
    tier=Tier.OPT_IN,
    feature="eagle3",                    # 功能名 (Tier 1/2 用)
    config_class="Eagle3DecodingConfig", # Config 类名 (Tier 2 用)
    arch=None,                           # 架构名 (MODEL tier 用)
    description="Eagle3 speculative decoding",
)
```

匹配逻辑：`fnmatch(changed_file, rule.pattern)`，一个文件可以匹配多条规则。

### 3.3 Additive-Only 优化

对于 CORE / DEFAULT_ON / OPT_IN / MODEL 这些高影响 tier，如果文件的 diff **只是扩展性修改**，则跳过该规则。判定标准：

- **扩展模式**: 每行删除内容都是某行新增内容的子串 (`"a"|"b"` → `"a"|"b"|"c"`)
- **安全新增**: 纯新增的 `def`/`class` 定义、`import`、数据结构条目 (以 `,` 结尾)
- **注释/空行**: 纯注释变更

**不安全**: 在已有函数体内新增代码 (if/else, 赋值, 函数调用等)

### 3.4 如何维护 Impact Rules

**新增源码路径时：**
1. 在 `IMPACT_RULES` 列表中添加一条 `ImpactRule`
2. 选择正确的 Tier:
   - 所有测试都会用到 → CORE
   - 默认开启、部分测试显式配置 → DEFAULT_ON (需指定 feature + config_class)
   - 需要显式开启 → OPT_IN (需指定 feature + config_class)
   - 模型实现代码 → MODEL (需指定 arch)
   - 测试定义文件 → TEST
   - 文档/CI → IGNORE

**新增模型架构时：**
1. 在 `IMPACT_RULES` 中添加 MODEL 规则 (源码路径 → arch)
2. 在 `TESTCLASS_TO_ARCH` 中添加 TestClass → arch 映射
3. 在 `MODEL_NAME_TO_ARCH` 中添加模型名关键词 → arch 映射 (给 test_e2e.py 这类模块级测试用)
4. 在 `REPRESENTATIVE_COVERAGE_SET` 中至少添加 1 条该架构的测试

**新增 Opt-In 功能时：**
1. 添加 OPT_IN 规则 (源码路径 → feature + config_class)
2. 如果有对应的 Config 类，在 `parser.py` 的 `_KNOWN_CONFIG_CLASSES` 中添加
3. 在 `FEATURE_TO_CONFIG` 中添加 feature → config class 映射
4. 在 `REPRESENTATIVE_COVERAGE_SET` 中添加 1 条覆盖测试
5. 如果需要，在 `_METHOD_FEATURE_PATTERNS` 中添加方法名模式

---

## 四、Selector — 测试选择逻辑

### 4.1 基于规则的选择

每条 `(changed_file, rule)` 匹配按 tier 分发到不同 handler：

| Tier | Handler | 选择逻辑 |
|------|---------|---------|
| CORE | `_select_by_core` | 选 `REPRESENTATIVE_COVERAGE_SET` 中的所有测试 |
| DEFAULT_ON | `_select_by_default_on` | 选 config_class 或 feature 匹配的测试 **+** Representative Set |
| OPT_IN | `_select_by_opt_in` | **仅**选 config_class 或 feature 匹配的测试。`speculative` 特殊处理：匹配所有 spec decoding 测试；`quantization` 匹配所有量化测试 |
| MODEL | `_select_by_model` | 选 `entry.arch == rule.arch` 的所有测试。同时支持模块级函数通过 `entry.model_names` 匹配 |
| TEST | `_select_by_test` | **方法级精度**：通过 git diff hunk 定位到具体被改动的 class 和 method，三级匹配 (见下文) |

#### TEST Tier 方法级精度

TEST tier 通过 `_get_changed_classes_in_test_file` 分析 git diff hunk，返回三级变更信息 (`_TestFileChanges`)：

| 级别 | 条件 | 选择范围 |
|------|------|---------|
| `class_methods` | 变更行落在某个 class 的某个 `def test_*` 方法内 | **仅选该方法**的参数化变体 |
| `class_wide` | 变更行在 class 内但不在任何 test 方法中（如 class 属性、setUp） | **选该 class 所有方法** |
| `select_all` | 无法确定（如 diff parse 失败） | **选该文件所有测试** |

模块级代码（在任何 class 外部）额外分析：如果只是新增 function/class 定义则安全跳过，否则触发 `select_all`。

**原因标签格式**:
- 方法级: `TEST method=TestClass::test_method modified in file.py`
- 类级: `TEST class=TestClass modified in file.py`
- 文件级: `TEST file=file.py (all classes)`

### 4.2 补充选择

在规则匹配之后，还有两类特殊选择：

**FALLBACK (未匹配的源码文件):**
如果有 `.py/.cpp/.h/.cu` 等源码文件没有匹配到任何规则，安全兜底：选 Representative Coverage Set。

**NEW_TEST (test list 新增行):**
解析 `tests/integration/test_lists/qa/*.txt` 的 git diff，`+` 开头的行即为新增测试，直接选中。新测试必须跑一遍验证能通过。

> **关于 waives.txt 变更**: un-waived 测试（从 waives.txt 中移除的行）不再被特殊选中。bug 关闭时已在 main 上验证过该测试能通过，不需要在 nightly 中再次强制运行。这些测试如果被其他规则（MODEL、OPT_IN 等）命中仍会正常参与选择，否则由 weekly 全量跑来兜底。

### 4.3 排除已知失败 (Waive Exclusion)

在去重**之前**，过滤掉当前 `waives.txt` 中的测试。

**原因**: waived 测试是已知会 fail 的（有对应 NVBug），跑了也是浪费时间。排除后可以把资源留给其他有效测试，提高覆盖面。

**匹配方式**:
- **精确匹配**: waive 行带参数 `[...]` 时，只排除该特定变体
- **前缀匹配**: waive 行不带参数时（如 `TestClass::test_method`），排除该方法的所有参数化变体

**为什么在去重之前**: 先排除 waived 测试，去重时 greedy coverage 算法就能在更大的有效测试池中选择，产生更好的覆盖。如果在去重之后排除，waived 测试可能在去重阶段占了名额，把其他好测试挤掉了。

| waives.txt 状态 | 含义 | 处理 |
|---|---|---|
| 当前在 waives.txt 中 | 已知 bug，跑了会 fail | **排除**，不占资源 |
| 从 waives.txt 中移除 (diff `-`) | bug 已修复，main 已验证 | 不特殊处理，走常规规则；weekly 兜底 |
| 新加入 waives.txt (diff `+`) | 新发现的 fail | 不选 |

### 4.4 去重逻辑

排除已知失败后，通过两轮去重降低测试数量：

#### 第一轮: 参数化变体去重 (`_deduplicate_parametrized`)

**范围**: 同一个 `(test_class, test_method)` 下的不同参数化变体。

**算法**: Greedy Tag Coverage
1. 对每个变体，提取 parameter tags (如 `moe_backend=cutlass`, `cuda_graph`, `tp_size`)
2. 贪心选择：每次选能贡献最多新 tag 的变体
3. 当没有变体能贡献新 tag 时停止

**效果**: 如 `test_bfloat16` 有 8 个变体（不同的 cuda_graph/overlap/torch_compile 组合），保留 3-4 个能覆盖所有参数维度的。

**豁免**: CORE、FALLBACK、NEW_TEST 标签的测试不参与去重。Representative Coverage Set 中的测试永远不被移除。

#### 第二轮: 类内方法去重 (`_deduplicate_per_class`)

**范围**: 同一个 `test_class` 下的不同 `test_method`。

**统一策略**: 所有 tier（MODEL、OPT_IN、DEFAULT_ON）均使用 **Greedy Feature Coverage** 算法。

**Greedy Feature Coverage 算法:**
1. 计算每个方法的 feature 并集（合并该方法所有参数化变体的 features）
2. 贪心选：每次选能贡献最多新 feature 的方法
3. 当没有方法能贡献新 feature 时停止
4. 被保留方法的所有参数化变体都保留

**示例**: `TestGPTOSS` 有 `test_w4_1gpu`, `test_eagle3_4gpus`, `test_eagle3_vswa_reuse_4gpus` 等方法。`vswa` 和 `reuse` 是独特 feature，所以不会被 `test_eagle3_4gpus` 覆盖。

**豁免**: CORE、FALLBACK、NEW_TEST 标签的测试不参与去重。TEST 标签（测试文件本身被改动的方法）也不参与去重——它们是精确匹配改动方法，无需去重。Representative Coverage Set 中的测试永远不被移除。

---

## 五、Time Budget — 时间预算裁剪

去重完成后，如果指定了 `--time-budget`（默认 8h），会进一步裁剪测试数量，使总预估运行时间尽量控制在预算内。

### 5.1 时长数据源

测试时长来自 `tests/integration/defs/.test_durations` 文件（JSON 格式，key 为 test ID，value 为秒数）。没有时长记录的测试使用**当前选中测试的中位数时长**作为默认值。

### 5.2 作用范围

当指定了 `--test-list`（如 `llm_function_core`）时，budget 裁剪**只作用于该 test list 内的测试**。不在目标 list 中的测试不计入总时长，也不会被删除。

### 5.3 受保护测试

以下测试**永远不会被 budget 删除**：

| 保护类型 | 原因 |
|----------|------|
| **REPRESENTATIVE_COVERAGE_SET** | 核心架构×特性覆盖，不可缺失 |
| **NEW_TEST** | 新增测试必须验证能通过 |

注意：`TEST method=`（测试文件被直接修改的方法）**不受保护**。虽然它们优先级较高，但如果单条测试耗时过长会被裁剪——测试代码的变更可以在后续 CI 中验证。

### 5.4 裁剪算法

分两个阶段，先删异常值再贪心削减：

#### Phase 0: 异常值剔除

删除**单条耗时超过 budget × 25%** 的测试（受保护测试除外）。

**目的**: 一个 18 小时的测试在 8 小时预算下毫无意义，无论它覆盖多少 feature 都应删除。

**阈值计算**: `max_single = budget_seconds × 0.25`
- 8h budget → 单条上限 2h
- 4h budget → 单条上限 1h

Phase 0 完成后检查总时长，如果已在预算内则结束。

#### Phase 1: 贪心最低价值删除

如果 Phase 0 后总时长仍超预算，进入贪心循环：

```
while total > budget:
    1. 构建 feature → {test_ids} 映射
    2. 对每个可删测试，计算 unique_feature_count:
       = 该测试覆盖的 feature 中，只有它一个测试覆盖的 feature 数量
    3. 选出 unique_feature_count 最小的测试
       (平局时选耗时最长的，省下最多时间)
    4. 删除该测试
```

**策略本质**: 每轮删除"最容易被其他测试替代"的测试。如果一个测试覆盖的所有 feature 都有其他测试也能覆盖（unique = 0），它就是最佳候选。在同等可替代性下，优先删耗时最长的以最大化节省。

**终止条件**: 总时长 ≤ budget，或所有非保护测试已删完。

### 5.5 Budget 可能超标

当受保护测试的总时长已超过预算时，budget 是**尽力而为**的——不会为了凑预算而删除 NEW_TEST 或 REPRESENTATIVE 测试。CLI 输出会明确提示：

```
Time budget: 8.0h, estimated: 10.9h (OVER by 2.9h (protected tests)), dropped 157 tests (84.8h saved)
```

### 5.6 CLI 参数

```bash
# 默认 8 小时预算
python -m tests.scripts.change_based_test_selection.cli --base-ref main --test-list llm_function_core

# 指定 4 小时预算
python -m tests.scripts.change_based_test_selection.cli --base-ref main --test-list llm_function_core --time-budget 4h

# 纯数字视为小时 (4 = 4h)
python -m tests.scripts.change_based_test_selection.cli --time-budget 4

# 支持分钟/秒
python -m tests.scripts.change_based_test_selection.cli --time-budget 480m
python -m tests.scripts.change_based_test_selection.cli --time-budget 28800s

# 禁用 budget 裁剪
python -m tests.scripts.change_based_test_selection.cli --time-budget 0
```

---

## 六、输出

### 6.1 分组优先级

输出按 reason 分组，优先级从高到低：

| 优先级 | 分组 | 含义 |
|--------|------|------|
| 0 | NEW_TEST | 新增测试 |
| 1 | MODEL: {arch} | 模型代码变更 |
| 2 | TEST: {class} modified | 测试文件变更 |
| 3 | OPT_IN: {feature} | Opt-in 功能变更 |
| 4 | DEFAULT_ON: {feature} | Default-on 功能变更 |
| 5 | Representative coverage set | CORE 基础设施变更 |
| 6 | FALLBACK | 未匹配兜底 |

同优先级内按测试数量降序，再按名称字母序。

### 6.2 输出格式

```
# NEW_TEST: newly added to test list (36 tests)
accuracy/test_llm_api_pytorch.py::TestNewClass::test_foo
...

# MODEL: llama (52 tests)
accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_eagle3[...]
...
```

### 6.3 CLI 用法

```bash
# 基本用法：对比 base-ref，选指定 test list 的测试
python -m tests.scripts.change_based_test_selection.cli \
  --base-ref 460889fa --test-list llm_function_core -o selected.txt

# 查看详细 reason（每条测试为什么被选）
python -m tests.scripts.change_based_test_selection.cli \
  --base-ref main --explain

# 指定文件（不用 git diff）
python -m tests.scripts.change_based_test_selection.cli \
  --files tensorrt_llm/models/llama/model.py

# 导出/加载数据库缓存
python -m tests.scripts.change_based_test_selection.cli --dump-db db.json
python -m tests.scripts.change_based_test_selection.cli --load-db db.json --files ...

# 反向嫌疑分析：给定失败测试和 commit 范围，排查嫌疑 commit
python -m tests.scripts.change_based_test_selection.cli \
  --suspect --good-ref abc123 --bad-ref def456 \
  --test-id "TestLlama3_1_8BInstruct::test_ngram"
```

---

## 七、维护点自动检测

每次运行时，CLI 会自动检查配置是否需要更新，并输出到 stderr：

```
⚠ Maintenance warnings (4 issues):
  [RULE] 12 source file(s) matched no impact rule ...
  [ARCH] 3 test class(es) have no architecture mapping ...
  [REP] 2 architecture(s) have no test in REPRESENTATIVE_COVERAGE_SET ...
  [FEAT] 87 tests have no extracted features ...
```

| 标签 | 检测内容 | 维护动作 |
|------|---------|---------|
| **[RULE]** | 变更的源码文件未匹配任何 impact rule（触发了 FALLBACK） | 在 `IMPACT_RULES` 中添加对应路径的规则 |
| **[ARCH]** | 数据库中的 test class 在 `TESTCLASS_TO_ARCH` 中没有架构映射 | 添加 TestClass → arch 映射 |
| **[REP]** | 某架构在数据库中有测试，但 `REPRESENTATIVE_COVERAGE_SET` 中没有该架构的代表测试 | 添加至少 1 条该架构的测试 |
| **[FEAT]** | 大量测试没有提取到任何 feature（影响去重和 budget 决策质量） | 检查 `_METHOD_FEATURE_PATTERNS`、`_PARAM_FEATURES`、`_KNOWN_CONFIG_CLASSES` |

---

## 八、关键维护点速查

| 场景 | 要改的地方 |
|------|-----------|
| 新增模型架构 | `IMPACT_RULES` + `TESTCLASS_TO_ARCH` + `MODEL_NAME_TO_ARCH` + `REPRESENTATIVE_COVERAGE_SET` |
| 新增 opt-in 功能 | `IMPACT_RULES` + `_KNOWN_CONFIG_CLASSES` + `FEATURE_TO_CONFIG` + `REPRESENTATIVE_COVERAGE_SET` |
| 新增核心源码路径 | `IMPACT_RULES` 添加 CORE 规则 |
| 新的方法名 feature 关键词 | `_METHOD_FEATURE_PATTERNS` |
| 新的参数 feature 关键词 | `_PARAM_FEATURES` |
| 新增 test definition 文件 | `IMPACT_RULES` 添加 TEST 规则 |
| 新增 Config 类 | `_KNOWN_CONFIG_CLASSES` + `_CONFIG_FEATURES` |
