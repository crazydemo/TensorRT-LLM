# Change-Based Test Selection (CBTS)

A change-based test selection system for TensorRT-LLM QA CI.

## What Problem Does This Solve

TRT-LLM has 600+ accuracy integration tests. Running all of them requires significant GPU resources, but between nightlies only a few files typically change, leaving most tests unaffected.

**CBTS does two things:**

1. **Test selection**: Analyzes git diff and selects only the affected test subset
2. **Regression locating**: Given a failing test and a good/bad commit range, ranks commits by likelihood of causing the regression, with PR links

---

## Core Concepts

### Test Database

Each test is indexed by three dimensions:

| Dimension | Source | Example |
|-----------|--------|---------|
| **Model architecture** | `TestClass.MODEL_NAME` -> HF architecture -> source directory | `TestDeepSeekV3Lite` -> `deepseek_v2` |
| **Features** | Method name + parameters + Config class instantiation in method body (L3 AST) | `test_nvfp4[mtp_nextn=2]` -> `{nvfp4, mtp}` |
| **GPU platform** | `@skip_pre_*` decorators + `@skip_less_device(N)` | `@skip_pre_blackwell` -> `min_sm=100` |

The database is built by parsing 4 test list files + AST analysis of test definition `.py` files, yielding ~600 entries.

### Tier System: Impact Scope of Code Changes

Different code paths have different blast radii. Source file paths are classified into 6 tiers:

```
Tier 0 (CORE)        Infrastructure that all tests depend on
                      executor, sampler, model loader, mapping...
                      -> Run representative coverage set (~20 tests)

Tier 1 (DEFAULT_ON)  Features enabled by default but configurable
                      KV cache, scheduler, attention backend, linear...
                      -> Run tests that explicitly configure the feature + representative set

Tier 2 (OPT_IN)      Features that must be explicitly enabled
                      Eagle3, MTP, NVFP4, guided decoding, disagg...
                      -> Run only tests that use the feature

MODEL                 Model implementation code
                      models/llama/*, models/deepseek_v2/*...
                      -> Run all tests for that architecture (deduped to 1 richest variant)

TEST                  Test definition files themselves
                      test_llm_api_pytorch.py, test_e2e.py...
                      -> Run tests in that file

IGNORE                Docs, markdown, CI config
                      -> Skip
```

**Core principle: the broader the impact, the more conservative the strategy (use representative set as safety net); the narrower the impact, the more precise the matching.**

### Representative Coverage Set

~20 hand-picked tests that cover:
- At least 1 test per model architecture
- At least 1 test per major opt-in feature
- All single-GPU (fast to run)
- Preference for tests that cover multiple features simultaneously

When CORE or DEFAULT_ON code changes, this set runs instead of all 600+ tests.

### New Test Detection

When test list files (`.txt` in `tests/integration/test_lists/qa/`) are modified, CBTS parses the git diff to find newly added test IDs. These are:
- **Always selected** regardless of other rules
- **Protected from deduplication** (never removed by either dedup pass)
- **Grouped first** in the output under the `NEW_TEST` heading for visibility

This ensures that newly added tests always run in the next CI cycle after being added.

### Additive-Only Safety Filter

Not every code change is risky. The following diff patterns are automatically skipped:

| Pattern | Example | Verdict |
|---------|---------|---------|
| Comments / blank lines only | All added/removed lines start with `#` | Safe |
| Expansion pattern | `"a" \| "b"` -> `"a" \| "b" \| "c"` | Safe |
| New definitions | New `def`/`class`/`import`/decorator | Safe |
| Data entries | New dict/list items (ending with `,`, no control flow) | Safe |
| New code inside function body | `if`/assignment/function call | **Unsafe** |

> **Why is new code inside function bodies unsafe?** Real case: a commit added an `if` branch in `py_executor_creator.py` that changed the default value of `max_num_tokens_in_buffer`, causing OOM.

### Parametrized Variant Deduplication

The same `(TestClass, test_method)` often has many parametrized variants, e.g.:
- `test_bfloat16[mtp_nextn=0-cuda_graph=False]` (no features enabled)
- `test_bfloat16[mtp_nextn=2-cuda_graph=True]` (all features enabled)

**Two-pass dedup** reduces test count while maintaining coverage:

1. **Per (class, method)**: Among parametrized variants, keep only the **1 variant with the most features** — it exercises the most code paths
2. **Per TestClass**: Among different test methods of the same class, keep only the **1 method with the most features** — avoids redundant coverage of the same model

Both passes skip CORE, FALLBACK, and NEW_TEST tests (these are always protected).

---

## Code Structure and Relationships

4 files with clear responsibilities:

```
                      ┌─────────────────────────────────────┐
                      │          impact_rules.py             │
                      │         (static config)              │
                      │                                      │
                      │  "which file path -> which tier"     │
                      │  "which TestClass -> which arch"     │
                      │  "which 20 tests = representative"   │
                      └──────────┬───────────────────────────┘
                                 │ referenced by
          ┌──────────────────────┼──────────────────────┐
          v                      v                      │
┌──────────────────┐   ┌──────────────────┐             │
│   parser.py       │   │   selector.py     │             │
│  (database build) │   │  (core engine)    │             │
│                   │   │                   │             │
│ Input:            │   │ Two main funcs:   │             │
│  test list .txt   │──>│                   │             │
│  test def .py     │   │ select_tests()    │             │
│                   │   │  in: database     │             │
│ Does:             │   │    + changed files │             │
│  parse test IDs   │   │  out: selected    │             │
│  AST extraction   │   │       tests       │             │
│  (arch, features, │   │                   │             │
│   config_classes) │   │ find_suspects()   │             │
│                   │   │  in: database     │             │
│ Output:           │   │    + failing test  │             │
│  test database    │   │    + good/bad ref  │             │
│  (dict/JSON)      │   │  out: suspect     │             │
│                   │   │    commits + PRs   │             │
└──────────────────┘   └──────────────────┘             │
          │                      │                      │
          └──────────┬───────────┘                      │
                     v                                  │
            ┌──────────────────┐                        │
            │     cli.py        │────────────────────────┘
            │  (CLI entry point)│
            │                   │
            │ Calls parser to   │
            │ build database,   │
            │ then dispatches   │
            │ to selector       │
            └──────────────────┘
```

**Core data flow:**

```
parser builds database ("what does each test cover")
    +
impact_rules defines rules ("what does each file change affect")
    +
git diff / git log ("what actually changed")
    ||
    vv
selector matches ("which tests are affected" or "which commit is most suspicious")
```

Key point: **the database does not contain impact rules**. The parser only answers "what model/features does this test cover", and impact rules only answer "what tier does this file path belong to". They are combined at runtime in the selector.

---

## Usage

### Test Selection

```bash
# Basic: select tests based on git diff
python -m tests.scripts.change_based_test_selection.cli --base-ref main

# Show selection reasons
python -m tests.scripts.change_based_test_selection.cli --base-ref main --explain

# Filter to a specific test list
python -m tests.scripts.change_based_test_selection.cli --base-ref main --test-list llm_function_core

# Specify files directly (no git diff)
python -m tests.scripts.change_based_test_selection.cli --files tensorrt_llm/models/llama/model.py

# Show statistics
python -m tests.scripts.change_based_test_selection.cli --files tensorrt_llm/_torch/speculative/eagle3.py --stats

# Cache database (speeds up repeated queries)
python -m tests.scripts.change_based_test_selection.cli --dump-db test_db.json
python -m tests.scripts.change_based_test_selection.cli --load-db test_db.json --files ...
```

### Output Format

Tests are grouped by selection reason. **NEW_TEST** group always appears first for visibility:

```
# NEW_TEST: newly added to test list (5 tests)
accuracy/test_llm_api_pytorch.py::TestNewModel::test_auto_dtype
...

# Representative coverage set (core infrastructure change) (17 tests)
accuracy/test_llm_api_pytorch.py::TestLlama3_1_8BInstruct::test_ngram
...

# MODEL: llama (10 tests)
...

# DEFAULT_ON: kv_cache (22 tests)
...
```

### Reverse Suspect Analysis

```bash
# Basic usage
python -m tests.scripts.change_based_test_selection.cli \
    --suspect \
    --good-ref <last known good commit SHA> \
    --bad-ref <first known bad commit SHA> \
    --test-id <failing test ID>

# Real example: OOM regression
python -m tests.scripts.change_based_test_selection.cli \
    --suspect \
    --good-ref 4adf76d8 \
    --bad-ref a9d49272 \
    --test-id "accuracy/test_disaggregated_serving.py::TestLlama3_1_8BInstruct::test_ngram"
```

Output example:

```
=== Reverse Suspect Analysis ===
Failing test: accuracy/test_disaggregated_serving.py::TestLlama3_1_8BInstruct::test_ngram
Commits analyzed: 17

--- SUSPECTS (6) ---                    <-- Non-safe changes that hit relevant code paths
  #1 0d18b2d7a4 Add priority-based KV cache offload filtering support (#10751)
      PR: https://github.com/NVIDIA/TensorRT-LLM/pull/10751
      CORE [core] <- py_executor_creator.py
      DEFAULT_ON [kv_cache] <- kv_cache_connector.py
  #4 a9d4927235 set default val of max_num_tokens_in_buffer (#11082)    <-- root cause
      PR: https://github.com/NVIDIA/TensorRT-LLM/pull/11082
      CORE [core] <- py_executor_creator.py

--- ADDITIVE-ONLY (1) ---              <-- Relevant code path but safe diff pattern
  ~ a7494a5ff4 Remove outdated comment in model_engine.py (#11240)

--- CLEAR (10) ---                      <-- Changes unrelated to this test
  . 36cb5f8c93 Fix multimodal serve test
  . 4c1d9d0c10 Pass without_comm to cutlass and deepgemm
  ...
```

**Workflow: get SUSPECTS list -> click PR links to review diffs -> manually identify root cause.**
In this example, 17 commits narrowed to 6 suspects; clicking PR #11082 immediately reveals the buffer default value change.

---

## How Reverse Suspect Analysis Works

Suspect analysis does **not** look at which source files the failing test executes. Instead, it uses impact rules as a **reverse matching** layer:

**Step 1: Extract the failing test's "profile"**

Look up the test in the database for its model architecture, features, config classes, and test file:

```
TestLlama3_1_8BInstruct::test_ngram (test_disaggregated_serving.py)
  arch:           llama
  features:       {ngram}
  config_classes: {NGramDecodingConfig}
```

**Step 2: For each commit, ask "did you change something relevant to this test?"**

For each commit in the good..bad range, get its changed files and check relevance via impact rules:

```
commit A modified speculative/eagle3.py
  -> Matches rule: OPT_IN, feature=eagle3
  -> Failing test uses ngram, not eagle3
  -> Not relevant -> CLEAR

commit B modified pyexecutor/py_executor_creator.py
  -> Matches rule: CORE
  -> CORE is relevant to all tests
  -> Check diff: changed max_num_tokens_in_buffer default (not a safe pattern)
  -> SUSPECT

commit C modified pyexecutor/model_engine.py
  -> Matches rule: CORE (relevant)
  -> But diff only removed a comment (safe pattern)
  -> ADDITIVE-ONLY
```

**Relevance rules:**

| Commit's matched tier | When is it relevant to the failing test |
|-----------------------|----------------------------------------|
| CORE / DEFAULT_ON | **Always relevant** (these code paths affect all tests) |
| OPT_IN | Only when the rule's feature/config_class matches the test's features/config_classes |
| MODEL | Only when the rule's arch matches the test's arch |
| TEST | Only when the modified file matches the test's test_file |
| IGNORE | Never relevant |

**Step 3: Classify and rank**

- **SUSPECT**: Changed relevant code path, diff is not a safe pattern (most suspicious)
- **ADDITIVE-ONLY**: Changed relevant code path, but diff is comments/definitions/data only (low risk)
- **CLEAR**: Changes are unrelated to this test

---

## Scenario Examples

| Scenario | Changed File | Tier | Result |
|----------|-------------|------|--------|
| Eagle3 code change | `speculative/eagle3.py` | OPT_IN | 59/616 (9.6%) — only Eagle3 tests |
| DeepSeek model change | `models/deepseek_v2/model.py` | MODEL | 145/616 (23.5%) — all DeepSeek/Kimi (deduped) |
| LLM API core change | `llmapi/llm.py` | CORE | 20/616 (3.2%) — representative set |
| Mixed changes | Qwen model + MTP feature | MODEL + OPT_IN | 168/616 (27.3%) — Qwen + MTP |

---

## CI Integration

### Nightly Pipeline

```bash
# In CI script:
python -m tests.scripts.change_based_test_selection.cli \
    --base-ref main \
    --test-list llm_function_core \
    -o /tmp/nightly_tests.txt

# Run only selected tests
pytest $(cat /tmp/nightly_tests.txt | tr '\n' ' ')
```

### Scheduling Strategy

| Frequency | What to Run |
|-----------|-------------|
| Nightly | CBTS-selected subset |
| Weekly | Full core + rtx6k + multinode |
| Monthly | Full regression (including stress) |

---

## Maintenance Guide

### Adding a New Model

1. Add `TestClass` -> arch mapping in `impact_rules.py` `TESTCLASS_TO_ARCH`
2. If it's a new architecture, add a corresponding `ImpactRule` (MODEL tier)
3. Add a representative test to `REPRESENTATIVE_COVERAGE_SET`

### Adding a New Feature

1. `parser.py`: Add feature extraction rules in `_METHOD_FEATURE_PATTERNS` or `_PARAM_FEATURES`
2. `parser.py`: Add Config class mapping in `_KNOWN_CONFIG_CLASSES` and `_CONFIG_FEATURES`
3. `impact_rules.py`: Add an `ImpactRule` (OPT_IN tier)
4. `impact_rules.py`: Add a representative test to `REPRESENTATIVE_COVERAGE_SET`

### Debugging

```bash
# Test selection: see why each test was selected
python -m tests.scripts.change_based_test_selection.cli --files <file> --explain

# Suspect analysis: if the root cause commit was missed, check:
# 1. Does the file have a corresponding ImpactRule?
# 2. Does the rule's feature/arch match the failing test?
# 3. Did the additive-only filter incorrectly classify the diff as safe?
```
