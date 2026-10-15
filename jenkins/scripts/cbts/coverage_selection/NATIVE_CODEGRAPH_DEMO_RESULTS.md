<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CBTS native codegraph demo results

## Result

The x86_64 graph maps every changed production-native file in the three fixed historical PRs.
The corresponding PY_START coverage rows produce non-empty case sets:

| PR | Production-native mapping | Python entry functions | Unique cases | Whole-PR scope |
|---|---:|---:|---:|---|
| #13880 | 1 / 1 | 14 | 391 | `coverage` |
| #18081 | 3 / 3 | 6 | 17 | fallback |
| #18202 | 1 / 1 | 1 | 248 | fallback |

#18081 reaches `causalConv1d.cu` through `causal_conv1d_fn` and
`causal_conv1d_update`. #18202 reaches `capacityScheduler.cpp` through
`BindCapacityScheduler.__init__`. Its changed C++ unit test is classified as a test definition and
is handled by the existing C++ unit-test block instead of requiring a native reverse mapping.

The two fallbacks are not native mapping failures:

- #18081 also changes the `_register_fake` closure. Its closure row set is identical to the file row
  set, so the current coverage data has no wider bound.
- #18202 also changes a module-executed statement in `scheduler.py`. PY_START does not capture every
  import-phase consumer, so the selector intentionally declines.

This distinction is recorded in every dry-run directory: `decision.json` contains the fail-closed
whole-PR decision, while `native-impact.json` contains the native mappings and exact stage-prefixed
case union.

## Reproduction

The graph was built from source commit `f04859d04de68b255343a6eb4a7d9d0bbaf82ca8` and the x86_64
`compile_commands.json`:

```bash
uv run --no-project \
  --with tree-sitter==0.25.2 \
  --with tree-sitter-python==0.25.0 \
  --with tree-sitter-cpp==0.23.4 \
  python -B jenkins/scripts/cbts/tools/build_native_codegraph.py \
  --repo-root . \
  --compile-commands callgraph-input/x86_64-build/compile_commands.json \
  --compile-commands-source-root /code/tensorrt_llm \
  --output reports/cbts_codegraph_x86_compile_commands.sqlite \
  --native-depth 5
```

The fixed replay command was:

```bash
python3 jenkins/scripts/cbts/tools/dryrun.py \
  --sha f3c3cdc4cc183eed13101782ac3359ecb6cca547 \
  --sha 07c5f2145a4797c25b18ddb76634edaecaf612e3 \
  --sha a1ad3edf9232c18689d057d2e8a1019cd7e15da7 \
  --filter all \
  --coverage-db /private/tmp/cbts-build-2942/cbts_touchmap.sqlite \
  --native-codegraph reports/cbts_codegraph_x86_compile_commands.sqlite \
  --out ./cbts-native-demo \
  --repo-root .
```

## Graph provenance and completeness

The SQLite metadata records the source commit, compilation database SHA-256, parser versions, and
the following completeness counters:

- 2,807 compilation entries and 2,793 unique sources;
- 778 locally available sources and 2,015 missing generated/dependency sources;
- 5,713 indexed native function definitions;
- 1,086 Python native-boundary candidates, 238 mapped boundaries, and 2,803 file mappings;
- 118 unresolved registration targets and no ambiguous registration targets.

The 2,015 unavailable sources are the main reason this artifact remains a demo graph rather than a
production-complete graph.
