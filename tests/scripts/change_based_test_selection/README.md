# Change-Based Test Selection (CBTS)

## 0. Problem Statement

### Background

TensorRT-LLM's QA integration test suite contains **600+ test cases** covering 15+ model architectures, dozens of opt-in features (speculative decoding, quantization, MoE, disaggregated serving, etc.), and multiple GPU configurations. A full run takes **100+ GPU-hours**.

In a typical nightly CI cycle, only a fraction of the codebase changes between runs. Running every test every night wastes GPU resources and delays feedback — engineers wait hours for results, most of which are for code paths that weren't touched.

### The Core Problem

**How do we select the smallest subset of tests that still catches regressions introduced by today's changes?**

This breaks down into several sub-problems:

1. **Impact analysis** — Given a set of changed source files, which tests *could* be affected? A change to `modeling_llama.py` shouldn't trigger DeepSeek tests; a change to the core executor should trigger broad coverage.

2. **Diminishing returns** — Many tests overlap in what they exercise. If we already selected 5 Llama tests covering FP8, beam search, and speculative decoding, the 6th Llama test with slightly different parameters adds marginal value.

3. **Resource budgeting** — Even after smart selection, the candidate set may exceed the available GPU-hour budget. We need a principled way to trim without losing critical coverage.

4. **Known failures** — Tests already tracked in `waives.txt` (known bugs under investigation) should not consume budget that could go to testing new/unknown regressions.

5. **Maintenance drift** — As the codebase evolves (new models, new features, renamed files), the selection rules can silently go stale. We need automated detection of configuration drift.

### Solution: Change-Based Test Selection

CBTS maps source code changes to affected tests through a tiered impact rule system, then applies deduplication and budget trimming to produce a focused test list:

```
306 changed files  →  487 candidate tests  →  44 selected tests (7.1%, ~8h)
     (full diff)        (impact analysis)       (dedup + budget)
```

### Deployment Model

CBTS is intentionally kept **inside the TensorRT-LLM repository** (not in an external CI repo) because:

- **Impact rules reference source paths directly** — `tensorrt_llm/_torch/models/modeling_llama*.py` must stay in sync with actual file locations. Same-repo means a PR that renames a file can update the rule in the same commit.
- **Parser reads test definitions at repo HEAD** — test class attributes, parametrize decorators, and config classes are parsed from the same checkout.
- **Maintenance warnings catch drift immediately** — when a new model is added without an impact rule, the next CBTS run flags it. In a separate repo, this would only be discovered after the fact.
- **Developer self-service** — contributors adding new models or features can update `impact_rules.py` in the same PR, without needing access to a separate QA repo.

The Jenkins CI pipeline only needs a single invocation:

```bash
python -m tests.scripts.change_based_test_selection.cli \
    --base-ref $LAST_GOOD_SHA --test-list llm_function_core \
    --time-budget 8h -o selected_tests.txt
```

### Intended Usage

| Cadence | Strategy | Purpose |
|---------|----------|---------|
| **Nightly** | CBTS with 8h budget | Early regression detection with minimal GPU cost |
| **Weekly** | Full test suite | Safety net — catches anything CBTS missed |

---

## 1. Overall Pipeline

```
git diff --name-only <base-ref>...HEAD
              ↓
         changed_files
              ↓
    ┌─────────────────────────┐
    │  impact_rules matching   │  fnmatch each file against all rules
    │  → (file, rule) pairs   │
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  additive-only filter    │  If diff for a high-impact tier file
    │                         │  is purely additive → skip that rule
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  selector (per tier)     │  CORE / DEFAULT_ON / OPT_IN /
    │                         │  MODEL / TEST each has its strategy
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  supplementary selection  │  NEW_TEST: new lines in test lists
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  exclude known failures  │  Filter out tests currently in
    │                         │  waives.txt (free up budget)
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  two-pass deduplication  │  Pass 1: parametrized variant dedup
    │                         │  Pass 2: intra-class method dedup
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  time budget trimming    │  Phase 0: drop outlier-duration tests
    │  (default 8h, optional) │  Phase 1: greedy lowest-value removal
    └─────────────────────────┘
              ↓
         group by reason, output test ID list
```

---

## 2. Parser — Building the Test Database

### 2.1 Data Sources

| Source | Path | Purpose |
|--------|------|---------|
| Test list files | `tests/integration/test_lists/qa/*.txt` | Full pytest node IDs, e.g. `llm_function_core.txt` |
| Test definition source | `tests/integration/defs/**/*.py` | Python source files for test classes/methods |
| TESTCLASS_TO_ARCH | dict in `impact_rules.py` | TestClass → model architecture mapping |

Test lists loaded by default:
- `llm_function_core`
- `llm_function_core_sanity`
- `llm_function_rtx6k`
- `llm_function_multinode`
- `llm_function_stress`

### 2.2 Parsing Process

**Step 1: Parse test lists (.txt)**

Each line is a pytest node ID in the format:
```
accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_nvfp4[moe_backend=CUTLASS-...]
```

Fields extracted:
- `test_file`: relative path
- `test_class`: class name
- `test_method`: method name
- `params`: key=value parameters (parsed from `[...]`)
- `raw_params`: raw bracket contents
- `test_lists`: which test lists this test belongs to

**Step 2: AST analysis of test definitions (.py)**

Three-level analysis:

| Level | Scope | Extracted |
|-------|-------|-----------|
| **L1 (class)** | class body | `MODEL_NAME`, `MODEL_PATH` class attributes |
| **L2 (decorator)** | method decorators | `@skip_pre_hopper` → min_sm=90; `@skip_less_device(4)` → min_gpu=4; `@parametrize(...)` → parameter dimensions and values |
| **L3 (method body)** | method body | Scan for Config class instantiations, e.g. `KvCacheConfig(...)`, `Eagle3DecodingConfig(...)` |

**Step 3: Parametrize resolution**

Two parametrize styles:

1. **Custom IDs** — `@pytest.mark.parametrize("a,b", [...], ids=["name1", "name2"])`
   - Build `id → {param: value}` mapping via the `ids` list
   - Match substrings in raw_params to reverse-lookup parameter values

2. **Auto-generated IDs** — `@pytest.mark.parametrize("tp_size, ep_size, ...", [(4,4,True,...)])`
   - pytest auto-joins `str(val)` with `-`, e.g. `4-4-True-True-True`
   - Iterate all_values, generate auto_id and match against raw_params

Cartesian product (multiple stacked `@parametrize`): the test ID is sub-IDs joined by `-`; each decorator is matched and consumed from the remaining string.

**Step 4: Feature extraction**

Feature tags are extracted from three sources:

| Source | Example |
|--------|---------|
| Method name pattern matching | `test_eagle3_vswa_reuse_4gpus` → `{eagle3, vswa, reuse, 4gpu}` |
| Parameter key/value | `cuda_graph=True` → `{cuda_graph}`; `moe_backend=CUTLASS` → `{moe_backend, moe_backend:cutlass}` |
| Config classes (L3) | `Eagle3DecodingConfig` → `{eagle3}`; `CudaGraphConfig` → `{cuda_graph}` |
| GPU scale | `_4gpu` / `tp4` in method name → `{4gpu}` / `{tp4}` |

### 2.3 Final Data Structure — TestEntry

```python
@dataclass
class TestEntry:
    test_id: str          # "accuracy/test_llm_api_pytorch.py::TestX::test_y[params]"
    test_file: str        # "accuracy/test_llm_api_pytorch.py"
    test_class: str       # "TestDeepSeekV3Lite"
    test_method: str      # "test_nvfp4"

    # L1
    model_name: str       # "deepseek-ai/DeepSeek-V3-Lite"
    arch: str             # "deepseek_v2" (from TESTCLASS_TO_ARCH)

    # L2
    min_sm: int           # 90 (Hopper+)
    max_sm: int           # 999
    min_gpu_count: int    # 4
    min_gpu_memory: int   # 0 (MB)
    param_dimensions: list[str]   # ["tp_size", "ep_size", "moe_backend"]
    resolved_params: dict         # {"tp_size": "4", "moe_backend": "CUTLASS"}

    # L3
    config_classes: set[str]      # {"MoeConfig", "KvCacheConfig"}

    # Combined
    features: set[str]            # {"nvfp4", "moe", "moe_backend:cutlass"}
    test_lists: set[str]          # {"llm_function_core"}
```

The full database is `dict[test_id, TestEntry]`, with ~600+ entries.

---

## 3. Impact Rules — Mapping Changed Files to Impact Tiers

### 3.1 Tier Definitions

| Tier | Name | Meaning | Test Selection Strategy |
|------|------|---------|------------------------|
| **0** | CORE | Core infrastructure; changes may affect all tests | Run **Representative Coverage Set** (~20 tests covering all arch × feature × GPU scale) |
| **1** | DEFAULT_ON | On-by-default but configurable features | Run tests that **explicitly configure the feature** + Representative Set |
| **2** | OPT_IN | Features requiring explicit opt-in | Run **only** tests that use the feature (matched via config_class or feature tag) |
| **3** | MODEL | Model-specific code | Run **all tests for that architecture** |
| **4** | TEST | Test file itself was changed | Run tests for the **specifically changed classes/methods** in that file (via git diff hunk analysis) |
| **5** | IGNORE | No impact (docs, CI config, etc.) | Select no tests |

### 3.2 Rule Format

```python
ImpactRule(
    pattern="tensorrt_llm/_torch/speculative/eagle3.py",  # glob pattern
    tier=Tier.OPT_IN,
    feature="eagle3",                    # feature name (for Tier 1/2)
    config_class="Eagle3DecodingConfig", # Config class name (for Tier 2)
    arch=None,                           # architecture name (for MODEL tier)
    description="Eagle3 speculative decoding",
)
```

Matching logic: `fnmatch(changed_file, rule.pattern)` — a single file can match multiple rules.

### 3.3 Additive-Only Optimization

For high-impact tiers (CORE / DEFAULT_ON / OPT_IN / MODEL), if the file's diff is **purely additive**, the rule is skipped. Criteria:

- **Extension pattern**: every deleted line is a substring of some added line (`"a"|"b"` → `"a"|"b"|"c"`)
- **Safe additions**: purely new `def`/`class` definitions, `import` statements, data structure entries (ending with `,`)
- **Comments/blank lines**: comment-only changes

**Unsafe**: new code inside existing function bodies (if/else, assignments, function calls, etc.)

### 3.4 Maintaining Impact Rules

**Adding a new source path:**
1. Add an `ImpactRule` entry to the `IMPACT_RULES` list
2. Choose the correct tier:
   - Used by all tests → CORE
   - On by default, some tests explicitly configure it → DEFAULT_ON (specify feature + config_class)
   - Requires explicit opt-in → OPT_IN (specify feature + config_class)
   - Model implementation code → MODEL (specify arch)
   - Test definition file → TEST
   - Docs/CI → IGNORE

**Adding a new model architecture:**
1. Add MODEL rules to `IMPACT_RULES` (source path → arch)
2. Add TestClass → arch mapping to `TESTCLASS_TO_ARCH`
3. Add model name keywords → arch mapping to `MODEL_NAME_TO_ARCH` (for module-level tests like test_e2e.py)
4. Add at least 1 test for this architecture to `REPRESENTATIVE_COVERAGE_SET`

**Adding a new opt-in feature:**
1. Add OPT_IN rule (source path → feature + config_class)
2. If a Config class exists, add it to `_KNOWN_CONFIG_CLASSES` in `parser.py`
3. Add feature → config class mapping to `FEATURE_TO_CONFIG`
4. Add 1 coverage test to `REPRESENTATIVE_COVERAGE_SET`
5. If needed, add method name patterns to `_METHOD_FEATURE_PATTERNS`

---

## 4. Selector — Test Selection Logic

### 4.1 Rule-Based Selection

Each `(changed_file, rule)` match is dispatched to a tier-specific handler:

| Tier | Handler | Selection Logic |
|------|---------|----------------|
| CORE | `_select_by_core` | Select all tests in `REPRESENTATIVE_COVERAGE_SET` |
| DEFAULT_ON | `_select_by_default_on` | Select tests matching config_class or feature **+** Representative Set |
| OPT_IN | `_select_by_opt_in` | Select **only** tests matching config_class or feature. Special handling: `speculative` matches all spec decoding tests; `quantization` matches all quantization tests |
| MODEL | `_select_by_model` | Select all tests where `entry.arch == rule.arch`. Also supports module-level functions via `entry.model_names` matching |
| TEST | `_select_by_test` | **Method-level precision**: analyze git diff hunks to locate specifically changed classes and methods, three-level matching (see below) |

#### TEST Tier Method-Level Precision

The TEST tier uses `_get_changed_classes_in_test_file` to analyze git diff hunks, returning three-level change information (`_TestFileChanges`):

| Level | Condition | Selection Scope |
|-------|-----------|----------------|
| `class_methods` | Changed lines fall within a specific `def test_*` method inside a class | Select **only that method's** parametrized variants |
| `class_wide` | Changed lines are inside a class but not within any test method (e.g. class attributes, setUp) | Select **all methods of that class** |
| `select_all` | Cannot determine (e.g. diff parse failure) | Select **all tests in that file** |

Module-level code (outside any class) gets additional analysis: if it only adds new function/class definitions it is safely skipped; otherwise it triggers `select_all`.

**Reason tag formats**:
- Method-level: `TEST method=TestClass::test_method modified in file.py`
- Class-level: `TEST class=TestClass modified in file.py`
- File-level: `TEST file=file.py (all classes)`

### 4.2 Supplementary Selection

After rule matching, two special selection types are applied:

**FALLBACK (unmatched source files):**
If any `.py/.cpp/.h/.cu` source files did not match any rule, safe fallback: select the Representative Coverage Set.

**NEW_TEST (new lines in test lists):**
Parse git diff of `tests/integration/test_lists/qa/*.txt`; lines starting with `+` are newly added tests and are directly selected. New tests must be run to verify they pass.

> **About waives.txt changes**: Un-waived tests (lines removed from waives.txt) are no longer force-selected. Bug closure already requires main-branch verification that the test passes, so un-waived tests don't need special treatment in nightly. They participate via normal rules if matched, and weekly full runs provide the safety net.

### 4.3 Exclude Known Failures (Waive Exclusion)

**Before** deduplication, tests currently listed in `waives.txt` are filtered out.

**Why**: Waived tests are known failures (with corresponding NVBugs). Running them wastes time and resources. Removing them frees budget for other valid tests, improving coverage.

**Matching**:
- **Exact match**: waive entries with parameters `[...]` exclude only that specific variant
- **Prefix match**: waive entries without parameters (e.g. `TestClass::test_method`) exclude all parametrized variants of that method

**Why before dedup**: If waived tests are removed before dedup, the greedy coverage algorithms work with a larger pool of valid tests, producing better coverage. If done after dedup, waived tests may have occupied dedup slots and crowded out better tests.

| waives.txt status | Meaning | Action |
|---|---|---|
| Currently in waives.txt | Known bug, will fail | **Exclude** — don't waste resources |
| Removed from waives.txt (diff `-`) | Bug fixed, verified on main | No special treatment; normal rules + weekly safety net |
| Added to waives.txt (diff `+`) | Newly discovered failure | Not selected |

### 4.4 Deduplication Logic

After waive exclusion, two deduplication passes reduce test count:

#### Pass 1: Parametrized Variant Dedup (`_deduplicate_parametrized`)

**Scope**: different parametrized variants of the same `(test_class, test_method)`.

**Algorithm**: Greedy Tag Coverage
1. For each variant, extract parameter tags (e.g. `moe_backend=cutlass`, `cuda_graph`, `tp_size`)
2. Greedy selection: each iteration picks the variant contributing the most new tags
3. Stop when no variant can contribute new tags

**Effect**: e.g. `test_bfloat16` with 8 variants (different cuda_graph/overlap/torch_compile combos) is reduced to 3-4 variants covering all parameter dimensions.

**Exemptions**: Tests tagged CORE, FALLBACK, or NEW_TEST are exempt from dedup. Tests in the Representative Coverage Set are never removed.

#### Pass 2: Intra-Class Method Dedup (`_deduplicate_per_class`)

**Scope**: different `test_method`s within the same `test_class`.

**Unified strategy**: all tiers (MODEL, OPT_IN, DEFAULT_ON) use the **Greedy Feature Coverage** algorithm.

**Greedy Feature Coverage algorithm:**
1. Compute the feature union for each method (merge features across all its parametrized variants)
2. Greedy selection: each iteration picks the method contributing the most new features
3. Stop when no method can contribute new features
4. All parametrized variants of retained methods are kept

**Example**: `TestGPTOSS` has `test_w4_1gpu`, `test_eagle3_4gpus`, `test_eagle3_vswa_reuse_4gpus`, etc. `vswa` and `reuse` are unique features, so they won't be subsumed by `test_eagle3_4gpus`.

**Exemptions**: Tests tagged CORE, FALLBACK, or NEW_TEST are exempt. Tests tagged TEST (directly modified test methods) are also exempt — they are precise matches of changed methods and don't need dedup. Tests in the Representative Coverage Set are never removed.

---

## 5. Time Budget — Duration-Based Trimming

After deduplication, if `--time-budget` is specified (default 8h), tests are further trimmed to fit the estimated total runtime within the budget.

### 5.1 Duration Data Source

Test durations come from the `tests/integration/defs/.test_durations` file (JSON format, key = test ID, value = seconds). Tests without duration records use the **median duration of currently selected tests** as the default.

### 5.2 Scope

When `--test-list` is specified (e.g. `llm_function_core`), budget trimming **only applies to tests in that test list**. Tests not in the target list are excluded from total duration calculations and are never dropped.

### 5.3 Protected Tests

The following tests are **never dropped by the budget**:

| Protection Type | Reason |
|----------------|--------|
| **REPRESENTATIVE_COVERAGE_SET** | Core arch × feature coverage, cannot be omitted |
| **NEW_TEST** | New tests must be validated to pass |

Note: `TEST method=` (methods directly modified in test files) is **not protected**. While they have higher selection priority, they will be trimmed if a single test's duration is excessive — test code changes can be verified in subsequent CI runs.

### 5.4 Trimming Algorithm

Two phases: first remove outliers, then greedily trim.

#### Phase 0: Outlier Removal

Drop any single test whose duration exceeds **budget × 25%** (protected tests excluded).

**Purpose**: an 18-hour test is pointless under an 8-hour budget — it should be dropped regardless of feature coverage.

**Threshold**: `max_single = budget_seconds × 0.25`
- 8h budget → single test cap of 2h
- 4h budget → single test cap of 1h

After Phase 0, check total duration. If already within budget, stop.

#### Phase 1: Greedy Lowest-Value Removal

If total duration still exceeds budget after Phase 0, enter a greedy loop:

```
while total > budget:
    1. Build feature → {test_ids} mapping
    2. For each droppable test, compute unique_feature_count:
       = number of features covered by this test that NO other test covers
    3. Pick the test with the smallest unique_feature_count
       (tie-break: pick the longest duration to maximize savings)
    4. Drop that test
```

**Core strategy**: each iteration drops the test that is "most easily replaced" by others. If all features of a test are also covered by other tests (unique = 0), it is the best candidate. Among equally replaceable tests, the longest one is dropped first to maximize time savings.

**Termination**: total duration ≤ budget, or all non-protected tests have been dropped.

### 5.5 Budget May Be Exceeded

When protected tests alone exceed the budget, the budget is **best-effort** — it will not drop NEW_TEST or REPRESENTATIVE tests to meet the target. The CLI output clearly indicates this:

```
Time budget: 8.0h, estimated: 10.9h (OVER by 2.9h (protected tests)), dropped 157 tests (84.8h saved)
```

### 5.6 CLI Parameters

```bash
# Default 8-hour budget
python -m tests.scripts.change_based_test_selection.cli --base-ref main --test-list llm_function_core

# Specify 4-hour budget
python -m tests.scripts.change_based_test_selection.cli --base-ref main --test-list llm_function_core --time-budget 4h

# Plain numbers are treated as hours (4 = 4h)
python -m tests.scripts.change_based_test_selection.cli --time-budget 4

# Minutes/seconds also supported
python -m tests.scripts.change_based_test_selection.cli --time-budget 480m
python -m tests.scripts.change_based_test_selection.cli --time-budget 28800s

# Disable budget trimming
python -m tests.scripts.change_based_test_selection.cli --time-budget 0
```

---

## 6. Output

### 6.1 Group Priority

Output is grouped by reason, from highest to lowest priority:

| Priority | Group | Meaning |
|----------|-------|---------|
| 0 | NEW_TEST | Newly added tests |
| 1 | MODEL: {arch} | Model code changes |
| 2 | TEST: {class} modified | Test file changes |
| 3 | OPT_IN: {feature} | Opt-in feature changes |
| 4 | DEFAULT_ON: {feature} | Default-on feature changes |
| 5 | Representative coverage set | CORE infrastructure changes |
| 6 | FALLBACK | Unmatched fallback |

Within the same priority, groups are sorted by test count (descending), then alphabetically.

### 6.2 Output Format

```
# NEW_TEST: newly added to test list (36 tests)
accuracy/test_llm_api_pytorch.py::TestNewClass::test_foo
...

# MODEL: llama (52 tests)
accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_eagle3[...]
...
```

### 6.3 CLI Usage

```bash
# Basic: diff against base-ref, select tests for a specific test list
python -m tests.scripts.change_based_test_selection.cli \
  --base-ref 460889fa --test-list llm_function_core -o selected.txt

# Explain mode: show why each test was selected
python -m tests.scripts.change_based_test_selection.cli \
  --base-ref main --explain

# Specify files directly (no git diff)
python -m tests.scripts.change_based_test_selection.cli \
  --files tensorrt_llm/models/llama/model.py

# Export/load database cache
python -m tests.scripts.change_based_test_selection.cli --dump-db db.json
python -m tests.scripts.change_based_test_selection.cli --load-db db.json --files ...

# Reverse suspect analysis: given a failing test and commit range, rank suspect commits
python -m tests.scripts.change_based_test_selection.cli \
  --suspect --good-ref abc123 --bad-ref def456 \
  --test-id "TestLlama3_1_8BInstruct::test_ngram"
```

---

## 7. Automatic Maintenance Warnings

On every run, the CLI automatically checks whether configuration needs updating and prints warnings to stderr:

```
⚠ Maintenance warnings (4 issues):
  [RULE] 12 source file(s) matched no impact rule ...
  [ARCH] 3 test class(es) have no architecture mapping ...
  [REP] 2 architecture(s) have no test in REPRESENTATIVE_COVERAGE_SET ...
  [FEAT] 87 tests have no extracted features ...
```

| Tag | What It Detects | Maintenance Action |
|-----|----------------|-------------------|
| **[RULE]** | Changed source files matched no impact rule (FALLBACK triggered) | Add rules for those paths in `IMPACT_RULES` |
| **[ARCH]** | Test classes in the database have no architecture mapping in `TESTCLASS_TO_ARCH` | Add TestClass → arch mapping |
| **[REP]** | An architecture exists in the database but has no test in `REPRESENTATIVE_COVERAGE_SET` | Add at least 1 representative test for that architecture |
| **[FEAT]** | Many tests have no extracted features (affects dedup and budget quality) | Check `_METHOD_FEATURE_PATTERNS`, `_PARAM_FEATURES`, `_KNOWN_CONFIG_CLASSES` |

---

## 8. Maintenance Quick Reference

| Scenario | What to Change |
|----------|---------------|
| Add a new model architecture | `IMPACT_RULES` + `TESTCLASS_TO_ARCH` + `MODEL_NAME_TO_ARCH` + `REPRESENTATIVE_COVERAGE_SET` |
| Add a new opt-in feature | `IMPACT_RULES` + `_KNOWN_CONFIG_CLASSES` + `FEATURE_TO_CONFIG` + `REPRESENTATIVE_COVERAGE_SET` |
| Add a new core source path | Add a CORE rule to `IMPACT_RULES` |
| New method name feature keyword | `_METHOD_FEATURE_PATTERNS` |
| New parameter feature keyword | `_PARAM_FEATURES` |
| Add a new test definition file | Add a TEST rule to `IMPACT_RULES` |
| Add a new Config class | `_KNOWN_CONFIG_CLASSES` + `_CONFIG_FEATURES` |
