# Change-Based Test Selection (CBTS) — Module Introduction

> For detailed design documentation, see [DESIGN.md](./DESIGN.md)

---

## The Problem

TensorRT-LLM's integration test suite contains **600+ test cases** covering 15+ model architectures and dozens of optional features. A full run takes **100+ GPU-hours**.

In a typical nightly CI cycle, only a fraction of the codebase changes between runs. Running every test every night means most GPU time and resources are spent on code paths that weren't touched.

**Core question: can we run only the tests that are likely affected by today's code changes?**

---

## How It Works

In one sentence: **map code changes to affected tests, then deduplicate and trim to maximize coverage within a limited GPU budget.**

```
306 changed files  →  487 candidate tests  →  44 selected tests (7.1%, ~8h)
    (git diff)          (impact analysis)       (dedup + budget trimming)
```

### Three Core Components

The module is built around three components working together:

```
                    ┌──────────────────┐
                    │   Impact Rules   │
                    │  (impact_rules.py)│
                    │                  │
                    │ Static knowledge:│
                    │ • Source path → Tier │
                    │ • TestClass → Arch  │
                    │ • Representative set│
                    └────────┬─────────┘
                             │ provides mapping rules
                             ▼
┌──────────────┐    ┌──────────────────┐
│    Parser    │    │    Selector      │
│  (parser.py) │───▶│  (selector.py)   │
│              │    │                  │
│ Builds test DB:│  │ Makes decisions: │
│ • Parse test lists│ • Rules → candidates│
│ • AST analysis │  │ • Dedup + trim   │
│ • Extract features│ • Output final list│
└──────────────┘    └──────────────────┘
  "What does each       "Which tests
   test look like?"      to run?"
```

**How the three components relate:**

- **Impact Rules** is the "knowledge layer" — statically maintained by humans, defining the impact relationships between source code and tests. It answers: *if this file changed, what could be affected?*
- **Parser** is the "perception layer" — automatically analyzes test source code (AST analysis of class attributes, decorators, method bodies), building a rich "profile" for each test (architecture, features, parameters, GPU requirements, etc.). It answers: *what does each test look like, and what does it exercise?*
- **Selector** is the "decision layer" — takes the mapping rules from Impact Rules and the test database from Parser, then executes the selection, deduplication, and trimming logic. It answers: *given the budget, which tests do we actually run?*

**A concrete example:** Suppose someone modifies `tensorrt_llm/_torch/speculative/eagle3.py`

1. **Impact Rules** tells us: this file matches the `OPT_IN` tier, feature is `eagle3`, corresponding Config class is `Eagle3DecodingConfig`
2. **Parser** already knows: which tests instantiate `Eagle3DecodingConfig` in their method bodies, which test method names contain `eagle3`
3. **Selector** decides: from Parser's database, select all tests whose features include `eagle3` or whose config_class includes `Eagle3DecodingConfig`, then deduplicate and trim

---

The overall pipeline has five steps:

### 1. Impact Analysis: Which Tests Might Be Affected?

A set of **tiered Impact Rules** maps changed files to affected tests. Rules are organized into 6 tiers by scope of impact:

| Tier | Meaning | Selection Strategy |
|------|---------|-------------------|
| **CORE** | Core infrastructure (executor, model base classes, etc.) | Run a curated Representative Coverage Set (~20 tests covering all arch x feature combos) |
| **DEFAULT_ON** | On-by-default features (attention, KV cache, etc.) | Run tests that explicitly configure the feature + Representative Set |
| **OPT_IN** | Explicitly opted-in features (Eagle3, quantization, MoE, etc.) | Run only tests that use the feature |
| **MODEL** | Model-specific code (modeling_llama.py, etc.) | Run all tests for that architecture |
| **TEST** | The test file itself was modified | Pinpoint to the exact changed class/method |
| **IGNORE** | Docs, CI config, etc. | Skip |

### 2. Supplementary Selection

- **New tests**: newly added lines in test lists are automatically selected — new tests must be validated to pass
- **Fallback**: when source files don't match any rule, run the Representative Coverage Set as a safety net

### 3. Exclude Known Failures

Tests in waives.txt (known bugs under investigation) are excluded — don't waste GPU time on them.

### 4. Two-Pass Deduplication

Many tests overlap in what they exercise:

- **Pass 1**: deduplicate parametrized variants of the same method (8 variants → 3-4 covering all parameter dimensions)
- **Pass 2**: intra-class method deduplication (greedy selection by feature coverage)

### 5. Time Budget Trimming

Default 8-hour GPU budget. First drop single tests exceeding 2h, then greedily drop the most replaceable tests. The Representative Coverage Set and new tests are protected and never dropped.

---

## Three Key Design Decisions

### Why Keep It Inside the TensorRT-LLM Repository?

- Impact rules reference source paths directly — file renames can update rules in the same PR
- Parser reads test definitions at repo HEAD — same checkout, no version skew
- Developers adding new models/features can maintain rules in the same PR without needing access to a separate QA repo

### Why Are Impact Rules Statically Maintained?

Impact rules are the core of the entire module — they define "which code affects which tests." This content is **intentionally human-confirmed** rather than dynamically generated by AI. The reasons:

- **It is the part most worth thinking about: the process of maintaining these rules forces us to understand the project's architectural boundaries and the weak spots in test coverage**
- An automatic detection mechanism checks on every run whether rules are stale (new models without rules, architectures without representative tests, etc.), ensuring they don't silently drift

### Nightly + Weekly Complementary Strategy

| Cadence | Strategy | Purpose |
|---------|----------|---------|
| **Nightly** | CBTS + 8h budget | Low-cost, fast-feedback early regression detection |
| **Weekly** | Full test suite | Safety net — catches anything CBTS might have missed |

---

## Automatic Maintenance Warnings

Every run automatically checks whether configuration needs updating:

```
⚠ Maintenance warnings (4 issues):
  [RULE] 12 source file(s) matched no impact rule
  [ARCH] 3 test class(es) have no architecture mapping
  [REP]  2 architecture(s) have no representative test
  [FEAT] 87 tests have no extracted features
```

These warnings ensure that as the codebase evolves, CBTS rules don't silently go stale.

---

## Usage

```bash
# Nightly CI — single invocation
python -m tests.scripts.change_based_test_selection.cli \
    --base-ref $LAST_GOOD_SHA --test-list llm_function_core \
    --time-budget 8h -o selected_tests.txt

# See why each test was selected
python -m tests.scripts.change_based_test_selection.cli \
    --base-ref main --explain

# Specify changed files directly (no git diff)
python -m tests.scripts.change_based_test_selection.cli \
    --files tensorrt_llm/models/llama/model.py
```
