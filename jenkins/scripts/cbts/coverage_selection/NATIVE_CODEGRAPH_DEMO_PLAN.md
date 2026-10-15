<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CBTS native codegraph demo plan

## Objective

Demonstrate that CBTS can map a changed TensorRT-LLM C++ or CUDA source file to Python functions
observed by PY_START, then select the corresponding integration cases from `cbts_touchmap.sqlite`.
The demo must remain fail-closed when a changed production-native file has no complete static
mapping.

The first milestone deliberately targets three historical PRs instead of attempting a complete
repository-wide semantic graph:

| PR | Changed native path | Expected result |
|---|---|---|
| #13880 | `cpp/tensorrt_llm/common/attentionOp.cpp` | Preserve the existing `coverage` positive control. |
| #18081 | `cpp/tensorrt_llm/kernels/causalConv1d/causalConv1d.cu` | Resolve the Torch op through its THOP wrapper to the CUDA implementation. |
| #18202 | `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp` | Resolve a nanobind class and its bound `operator()`. |

## Required dependency paths

The CUDA path is:

```text
causal_conv1d_fn / causal_conv1d_update
    -> torch.ops.trtllm.causal_conv1d_fwd / causal_conv1d_update
    -> TORCH_LIBRARY_IMPL registration
    -> causalConv1dFwd / causalConv1dUpdate
    -> causal_conv1d_*_cuda
    -> causalConv1d.cu
```

The nanobind class path is:

```text
BindCapacityScheduler.__init__
    -> tensorrt_llm.bindings.internal.algorithms.CapacityScheduler
    -> nb::class_<CapacityScheduler>
    -> .def("__call__", &CapacityScheduler::operator())
    -> capacityScheduler.cpp
```

## Implementation sequence

1. Treat CUDA parser-recovery nodes whose recovered name begins with `namespace ` as containers,
   then continue indexing their nested function definitions.
2. Index chained `nb::class_<T>(...).def(...)` registrations, including method pointers such as
   `&T::operator()` and properties. Match Python construction by the bound C++ class name when the
   exposed Python name is not a string literal.
3. Keep compilation-database filtering and record unresolved or ambiguous edges as non-eligible
   evidence. Introduce Clang only as a fallback for translation units that remain unresolved after
   the targeted Tree-sitter fixes.
4. Treat changes under `cpp/tests/` as native test definitions. They must trigger the appropriate
   existing test-definition stages but must not require a Python-to-production-native reverse map.
5. Rebuild the graph at the same source commit as `compile_commands.json` and run the three fixed
   historical PRs against the build 2942 merged coverage database.

## Acceptance criteria

- All three fixed PRs produce a `native-impact.json` artifact with at least one mapped case.
- The artifact separates native impact from the complete PR decision: an unrelated unsafe Python
  residual may still keep `scope=None` and full fallback.
- #18081 records an evidence path reaching `causalConv1d.cu`.
- #18202 records an evidence path reaching `capacityScheduler.cpp`.
- A changed production-native file without a mapping still makes the coverage tier decline.
- `cpp/tests/**` does not independently force a native mapping failure.
- Graph metadata records the source commit, compilation-database provenance, parser versions, and
  completeness counters before the graph is used outside dry-run evaluation.

## Evaluation after the milestone

After the three-PR gate passes, replay a real 40-commit window and report:

- native-changing commits;
- commits where every changed production-native file maps;
- coverage-scope hits;
- fallback reasons;
- impacted cases and case skip rate;
- unresolved and ambiguous native edges grouped by syntax pattern.

Do not design the per-pipeline production rollout until this demo shows useful end-to-end native
selection.

## Safety finding during implementation

The original milestone expected all three historical PRs to finish with `scope=coverage`. That is
not a valid acceptance gate for mixed-code PRs: #18081 also changes an unrecorded closure and #18202
also changes an import-executed module statement. PY_START cannot currently enumerate a sound upper
bound for either change. The selector must therefore keep falling back for the complete PR even when
its native files map successfully. The demo reports native cases independently and does not weaken
those existing fail-closed gates.
