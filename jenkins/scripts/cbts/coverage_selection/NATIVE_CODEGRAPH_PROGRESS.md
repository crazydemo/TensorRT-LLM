<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CBTS native codegraph exploration: progress and handoff

Last updated: 2026-08-27 (Asia/Shanghai)

## Goal

Evaluate whether a separately built static codegraph can connect changed C/C++/CUDA files to
Python functions captured by PY_START, allowing the existing CBTS coverage tier to select cases
for native changes without adding work to case runtime.

The intended lookup is:

```text
changed .cpp/.cu/.h/.cuh
    -> statically connected Python functions
    -> tests touching those functions in cbts_touchmap.sqlite
    -> existing coverage_tier narrowing
```

## Workspace state

- Exploration worktree: `/Users/yanzh/workspace/TensorRT-LLM-cuda-case-mapping`
- Branch: `explore/cuda-case-mapping`
- Branch HEAD: `3035d47286ef3e232b16430abf376600ef192517`
- Latest fetched `upstream/main`: `2e70be324563e09255345829aad8d9e4bcf92d70`
- The original checkout `/Users/yanzh/workspace/TensorRT-LLM` was not modified.
- Changes are neither staged nor committed.

`upstream/main` advanced while the exploration was running. Before continuing implementation,
update/rebase this worktree carefully while preserving the uncommitted files listed below.

## Implemented prototype

### Static graph builder

`tools/build_native_codegraph.py` builds a standalone SQLite database by:

1. Parsing every Python function under `tensorrt_llm/` with the Python AST.
2. Finding direct `torch.ops` and `tensorrt_llm.bindings` calls.
3. Finding simple torch and nanobind registrations in the native tree.
4. Parsing `.cc/.cpp/.cu/.cxx` function definitions with Tree-sitter C++.
5. Following uniquely resolved native calls to a bounded depth, currently five hops.
6. Adding direct project-header include edges for reached native files.

Headers are not parsed as complete function-definition trees. An early attempt took more than
two minutes because of large/generated/template headers. The current split keeps function-level
parsing on translation units and uses include-level edges for headers.

The builder creates these main tables:

- `python_boundary`: Python function to `torch.ops`/nanobind boundary calls.
- `python_native_file`: Python function to reachable native file, minimum hops, and evidence.
- `native_file_summary`: reverse-map summary view.

The graph is deliberately stored separately from `cbts_touchmap.sqlite`: it is source-revision
metadata, while the touch database is runtime case coverage and is architecture/build specific.

### Coverage integration

`coverage_selection/native_graph.py` opens the graph read-only. `CoverageSelector` now accepts an
optional graph and handles native residuals as follows:

```text
no graph                         -> decline the coverage tier
native file has no mapping       -> decline the coverage tier
mapped native file               -> union tests touching mapped Python functions
mapped Python function has no DB rows
                                 -> existing no_data_policy fallback (file by default)
```

Selection does not use a numeric confidence score. Conditional Python boundaries are included,
which can only widen the impacted set. Native calls are followed only when the prototype resolves
one definition. Evidence is recorded in SQLite for inspection.

`main.py` exposes the graph through `--native-codegraph`. The graph and touch DB connections are
closed after the coverage decision, including exception paths.

### Dry-run integration

`tools/dryrun.py` now:

- accepts `--native-codegraph` alongside `--coverage-db`;
- forwards it to `main.py` for historical replay;
- sets `GIT_LFS_SKIP_SMUDGE=1` for temporary historical worktrees;
- writes each raw decision as `decision.json`, in addition to `summary.txt` and filtered YAMLs.

The raw JSON makes it possible to reuse `tools/report_cbts_decision.py::_case_counts()` for the
same case-level skip-rate definition used by OpenSearch.

## Files changed or created

Core prototype:

- `jenkins/scripts/cbts/coverage_selection/native_graph.py` (new)
- `jenkins/scripts/cbts/coverage_selection/selector.py`
- `jenkins/scripts/cbts/coverage_selection/SELECTION.md`
- `jenkins/scripts/cbts/coverage_tier.py`
- `jenkins/scripts/cbts/main.py`
- `jenkins/scripts/cbts/tools/dryrun.py`
- `jenkins/scripts/cbts/tools/build_native_codegraph.py` (new)
- `tests/unittest/scripts/test_cbts_native_graph.py` (new)

Exploration/demo files created earlier in the same worktree:

- `jenkins/scripts/cbts/tools/case_native_dependency_demo.py`
- `jenkins/scripts/cbts/tools/render_case_native_report.py`
- `jenkins/scripts/cbts/tools/tree_sitter_scan.py`
- `reports/cbts_accuracy_native_demo.sqlite`
- `reports/cbts_accuracy_native_demo.html`

The production-shaped builder currently imports matcher helpers from
`case_native_dependency_demo.py`. If this work is kept, those helpers should be moved to a
properly named shared module before review.

## Generated graph

Latest-upstream graph:

- Path: `reports/cbts_codegraph_upstream_main.sqlite`
- Source snapshot: `upstream/main@2e70be3245`
- Size: approximately 1.2 MiB
- Build time: 18.641 seconds
- Python boundaries: 1,088
- Mapped Python functions: 224
- Mapped native files: 287
- Python-function/native-file rows: 2,400

The earlier `3035d` graph remains at `reports/cbts_codegraph.sqlite` and has the same counts.

Tree-sitter environment used during exploration:

```text
/tmp/trtllm-treesitter-venv
tree-sitter==0.25.2
tree-sitter-python==0.25.0
tree-sitter-cpp==0.23.4
```

Example build command:

```bash
/tmp/trtllm-treesitter-venv/bin/python -B \
  jenkins/scripts/cbts/tools/build_native_codegraph.py \
  --repo-root /path/to/TensorRT-LLM \
  --output reports/cbts_codegraph.sqlite \
  --native-depth 5
```

## Dry-run results

### Full seven-day replay

Source window:

- Ref: `upstream/main@2e70be3245`
- 217 merged PR commits
- Oldest included commit time: 2026-08-20 16:31:02 +08:00
- Next excluded commit time: 2026-08-20 15:56:03 +08:00
- Runtime: 629.11 seconds (10 minutes 29 seconds)
- Touch DB: build 2930 x86_64 compact DB only, not the production x86/SBSA union
- Output: `/tmp/cbts_all_7d`

There were 216 valid decisions and one empty commit (`#18136`) reported as `no files changed`.

Scope distribution:

| Scope | Commits |
|---|---:|
| Full fallback (`None`) | 124 |
| `waiveonly` | 48 |
| `coverage` | 14 |
| `testsonly` | 10 |
| `testdefonly` | 9 |
| `noop` | 6 |
| `visualgenonly` | 2 |
| `agentflowonly` | 1 |
| `specdeconly` | 1 |
| `testlistonly` | 1 |
| Error / empty commit | 1 |

Across the 216 valid commits:

- CBTS hit rate: `92 / 216 = 42.59%`.
- Full fallback rate: `124 / 216 = 57.41%`.
- Coverage-scope rate: `14 / 216 = 6.48%`.

Using the existing `report_cbts_decision.py::_case_counts()` definition, with fallback commits
counted as running the full baseline:

- Overall weighted case skip rate: **34.06%**.
- Weighted skip rate among the 92 CBTS hits: **79.96%**.
- Weighted skip rate among the 14 coverage hits: **52.48%**.

Fallback reasons across the 124 full runs:

| Primary reason | Commits |
|---|---:|
| Unsupported/non-core residual | 63 |
| Import-executed Python change | 38 |
| Unmapped native file | 12 |
| Zero-touch Python file | 5 |
| Tier-1 forced fallback | 5 |
| No residual after rules | 1 |

Reproduction command, pinned to the evaluated tip/window:

```bash
python3 jenkins/scripts/cbts/tools/dryrun.py \
  --ref 2e70be324563e09255345829aad8d9e4bcf92d70 \
  --window 217 \
  --filter all \
  --coverage-db /path/to/build-2930-x86/cbts_touchmap.sqlite \
  --native-codegraph reports/cbts_codegraph_upstream_main.sqlite \
  --out /tmp/cbts_all_7d \
  --repo-root /Users/yanzh/workspace/TensorRT-LLM-cuda-case-mapping
```

### Native-changing subset

Nineteen of the 217 commits changed a native source/header under `cpp/`. Output is in
`/tmp/cbts_native_7d`.

Final CBTS result:

- Native coverage hits: `0 / 19`.
- Native commit hit rate: **0%**.
- Actual native case skip rate: **0%** because every commit failed closed.

The graph still found partial relationships:

- At least one changed native file mapped: `11 / 19 = 57.89%` of commits.
- Every changed native file mapped: `5 / 19 = 26.32%` of commits.
- Mapped native file-change instances: `18 / 106 = 16.98%`.
- Mapped unique native files: `16 / 94 = 17.02%`.

Primary fail-closed reasons for the 19 commits:

- 12 had an unmapped native file.
- 5 were blocked by an unsupported CMake/JSON residual.
- 2 were blocked by the existing Python import-executed safety gate.

Therefore partial native graph coverage did not translate into a single end-to-end hit.

### Known-positive sanity check

Historical PR `#13880` changes only `cpp/tensorrt_llm/common/attentionOp.cpp` and remains a useful
positive control:

- `scope=coverage`
- 570 impacted tests
- 63 removed YAML entries
- 7 narrowed stages
- Full dry-run time approximately 5.5 seconds

This proves that the integration path works when all residual gates pass; it does not prove the
static graph is complete enough for production selection.

## Tests and checks

- `tests/unittest/scripts/test_cbts_native_graph.py`: 5 tests passed.
- Tests cover reverse lookup, mapped function rows, file-level no-data fallback, graph absence,
  and unmapped-native fail-closed behavior.
- Pre-commit passed for all changed prototype files.
- Both generated SQLite files passed `PRAGMA integrity_check` during the exploration.

The tests were run in a temporary `uv` environment with repository-level GPU-heavy conftest
isolation:

```bash
uv run --no-project --with pytest --with pyyaml \
  pytest -q --confcutdir=tests/unittest/scripts \
  tests/unittest/scripts/test_cbts_native_graph.py
```

## Important safety limitation

The current selector checks whether a changed native file has at least one mapping. That is not a
proof that every Python/native entry into the file was found. A partially mapped native file could
therefore be treated as selectable even though an omitted dynamic/template path exists.

Keep this prototype in dry-run mode until it records an explicit completeness/eligibility verdict.
A production-safe rule should require every changed native component to be rooted in enumerated
registrations and should decline when unresolved or ambiguous edges could hide another entry.

## Recommendation for the next iteration

Continue with a Tree-sitter-first hybrid approach before introducing libclang.

High-value Tree-sitter/static-index work:

1. Support nanobind class and method registration (`nb::class_().def`, constructors, properties,
   nested modules, method pointers, and lambdas), not only simple `m.def`.
2. Support split torch registration patterns across `TORCH_LIBRARY`, `TORCH_LIBRARY_FRAGMENT`,
   and `TORCH_LIBRARY_IMPL`.
3. Build reverse and transitive include edges, including template-instantiation `.cu` files to
   `*_template.h`.
4. Separate CMake/JSON policy from the native codegraph so those files do not obscure matcher
   measurements.
5. Record unresolved and ambiguous native edges and derive `selection_eligible` from graph
   completeness, rather than using any-match existence.
6. Re-run the exact 19-commit native window after each change and compare all four mapping rates.

Add libclang as a second native-resolution layer only if the remaining misses are dominated by
overloads, receiver types, template specialization, macro expansion, or translation-unit/header
ownership. Tree-sitter should continue to own the fast Python-to-registration matcher; libclang
would consume a valid `compile_commands.json` to refine native calls and dependencies.

## Open decisions

- How the native graph is produced and versioned in CI, and how its source commit freshness is
  gated against the PR base.
- Whether to merge the graph into the coverage artifact tarball while retaining a separate SQLite
  file.
- How to expose graph evidence through the existing `pystart_report.py` HTML rather than a second
  report UI.
- Whether CMake/native build-system changes should always run full CI or get a dedicated Tier-1
  rule.
- What completeness contract is strong enough to permit production case skipping.
