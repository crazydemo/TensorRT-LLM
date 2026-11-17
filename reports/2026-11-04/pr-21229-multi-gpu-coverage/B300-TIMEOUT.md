<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# B300 timeout evidence and proposed validation

Audited 2026-11-17. [Main report](REPORT.en.md).

## Finding

Coverage-disabled runs reproduce the same **outer attention batch timeout** seen with coverage enabled. This establishes that coverage is not required for this failure to occur. It does **not** establish that coverage cannot slow the batch or increase its probability of timing out. The inspected runs span different code revisions, test lists and environments, and were not a randomized or repeated paired experiment.

Not every red B300 stage is a timeout. In #70619, Post-Merge-2 failed a KV-cache unit test with an unexpected constructor argument; Post-Merge-1 hit the attention timeout. Counting both as timeouts would overstate the evidence.

## Direct observations

Times below are Jenkins Run Pytest wall time for the whole stage, **not** time spent in the individual attention batch. The batch has a separate 3,600 s deadline. Child stage steps are completed even though the #70619 parent pipeline was still running when inspected.

| CI | Commit | Actual child `cbts_coverage` flag | B300 Post-Merge-1 | B300 Post-Merge-2 |
|---|---|---|---|---|
| [#69221](https://nv/trt-llm-cicd/job/main/job/L0_MergeRequest_PR/69221/) | `1a0c33c4ccdd8d77f764253f44aa83b0d3e61f23` | `true` | 9,069.1 s; attention batch timeout at 3,600 s | 9,949.1 s; SUCCESS |
| [#69544](https://nv/trt-llm-cicd/job/main/job/L0_MergeRequest_PR/69544/) | `23311ffcebf5380364777272484c5211b50cd4f7` | `false` | 8,456.9 s; same attention batch timeout at 3,600 s | 10,545.4 s; SUCCESS |
| [#70619](https://nv/trt-llm-cicd/job/main/job/L0_MergeRequest_PR/70619/) | `be1f3b49940d78a20b55afb6118d6ef329df00ac` | `false` | 10,255.5 s; attention batch timeout at 3,600 s; expanded ignore list | 9,660.1 s; KV-cache TypeError, not a timeout |

The on/off setting was verified from the **executing multi-GPU child's `testFilter` log**, not inferred from the command, PR head or pipeline name. Both later off commits also have `ENABLE_CBTS_COVERAGE = false`. The #69221 on artifacts contain coverage SQLite files. Test-result reuse was false in all three child logs. CBTS selection was disabled in #70619; it was not explicitly disabled in the two older targeted runs.

#69221 and #69544 are not same-baseline pairs: their Git trees differ in 662 files. Their numeric stage difference (+612.2 s for Post-Merge-1) is not an attributable collector overhead, and both batch durations are censored at the deadline. Post-Merge-2 being faster with coverage on is likewise not proof of a speedup.

## Timeout identity and stack

The structured `TRTLLM_PYTEST_TIMEOUT` log records identify:

```text
#69221 and #69544, attempt 1, timeout 3600.0:
DGX_B300-4_GPUs-PyTorch-Post-Merge-1/test_unittests.py::test_unittests_v2[unittest/_torch/attention/multi_gpu --ignore=unittest/_torch/attention/multi_gpu/test_helix_zero_kv.py]

#70619, attempt 1, timeout 3600.0:
DGX_B300-4_GPUs-PyTorch-Post-Merge-1/test_unittests.py::test_unittests_v2[unittest/_torch/attention/multi_gpu --ignore=unittest/_torch/attention/multi_gpu/test_helix_zero_kv.py --ignore=unittest/_torch/attention/multi_gpu/test_gqa_kv_replica.py]
```

All three outer main-thread timeout dumps include `test_unittests_v2 -> run_command -> subprocess.wait`. MPI futures manager threads are also visible. This locates the outer wait boundary; it does not identify the blocked worker rank, a kernel deadlock, or a collector-induced wait. It also cannot distinguish a single hung inner test from cumulative batch work exceeding the deadline.

The inner logs reach `test_mla_helix_context.py::test_k3_mla_helix_owned_prefill`. Importantly, the two diagnostic sources do not always agree on the parameterized identity:

| CI | Last live test label adjacent to the timeout stack | Synthetic unfinished-case label in `results-timeout.xml` |
|---|---|---|
| #69221 | `tp1-cp4-gate-fp8-weights-bf16-kv` | `tp1-cp4-gate-fp8-weights-fp8-kv` |
| #69544 | `tp1-cp4-no_gate-bf16-weights-bf16-kv` | `tp1-cp4-no_gate-bf16-weights-fp8-kv` |
| #70619 | `tp1-cp4-gate-bf16-weights-bf16-kv` | `tp1-cp4-gate-bf16-weights-bf16-kv` |

These all concern the same test family and four-way context parallelism, but do not prove the same parameter case hung each time. Synthetic unfinished-case records have time `1.00` s; that is a placeholder and must not be interpreted as actual test duration. Buffered console timestamps should not be used as inner-test start/end timestamps.

## Separate Post-Merge-2 failure

#70619 records:

```text
unittest/_torch/pyexecutor/kv_cache/test_mamba_cache_manager.py::test_skip_softmax_hybrid_follows_the_dense_manager_routing[False-0-True-KVCacheManagerV2]
TypeError: KVCacheLayerSpec.__init__() got an unexpected keyword argument 'num_q_heads'
```

The inner failure took 0.003 s. Its outer `unittest/_torch/pyexecutor` wrapper failed after 1,664.0 s. Neither value is a timeout. The constructor mismatch needs its own investigation and should not be pooled with the attention timeout.

The older #68871 Post-Merge-1 archive likewise contains assertion/ValueError failures rather than a recorded 3,600 s timeout. Its Post-Merge-2 archive has synthetic `terminated_unexpectedly` records in the Helix context-heads family; those 1 s records alone do not establish which timeout, if any, killed the process. These failures must not all be labeled collector-induced timeouts.

## Recommendation

Existing data is enough to rebut the claim that B300 only times out when collection is enabled. It is not enough to establish non-aggravation. Repeating coverage-on alone would show whether the failure recurs, but would not separate collector effects from the existing problem.

For a focused investigation, keep the PR's production gate unchanged while deciding whether to run a diagnostic experiment:

1. Use one fixed product/test revision and matching build artifacts. Alternate coverage off/on; do not rebase or change waivers, timeouts or test lists between conditions. Explicitly disable CBTS selection and test-result reuse. Record actual runtime flags and machine/container identities.
2. As an initial diagnostic, use at least three attempts per condition on the B300 attention batch, preserving inner order and worker settings. If CI only supports whole stages, select just `DGX_B300-4_GPUs-PyTorch-Post-Merge-1`; verify the batch is still assigned there in that revision. Avoid running all GPU stages. Three pairs are an initial diagnostic, not statistical proof of an unchanged failure rate.
3. Capture monotonic inner start/end times, worker/rank stack dumps, GPU/NCCL state, and collector taint/process diagnostics. Compare time to enter the Helix family, completed case count, and the exact failure/wait point. If feasible, repeat the suspect family separately after the original batch-order reproduction; a standalone pass can miss order-dependent failures.
4. Do not simply raise the 3,600 s threshold: it would hide rather than explain the current failure. If both conditions repeatedly time out, their elapsed durations are censored and cannot establish equal overhead. Investigate the underlying stall or reduce the diagnostic scope to get completed comparable observations.
5. Agree on an acceptable slowdown/timeout-risk margin before a rollout decision. A few successful runs cannot prove zero risk. If further validation is undesirable, temporarily exclude the affected B300 stage from collection and hand the evidence to the attention/Helix owners; this does not fix the underlying timeout.

No new experiment was triggered for this audit. No B300 exclusion or production collector policy change was made.

## Source artifacts and reproducibility

The older archives were already preserved locally for the case-level report. The two off runs were downloaded again for this audit. `results-timeout.xml`, `results.xml`, nested JUnit reports, `unfinished_test.txt`, the child console, and Blue Ocean Run Pytest node durations were inspected together. No XML was extracted onto the filesystem; members were read directly from the tarballs.

- #69221: [multi-GPU child #3483](https://prod.blsm.nvidia.com/sw-tensorrt-llm-github-2/job/LLM/job/main/job/L0_Test-x86_64-Multi-GPU/3483/) · [console](https://prod.blsm.nvidia.com/sw-tensorrt-llm-github-2/job/LLM/job/main/job/L0_Test-x86_64-Multi-GPU/3483/consoleText).
  - [Post-Merge-1 archive](https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_MergeRequest_PR/69221/test-results/results-DGX_B300-4_GPUs-PyTorch-Post-Merge-1.tar.gz); downloaded-byte SHA-256 `d3338b7596a0e89a19245b4d88a9166148d91f247ec0d10c63f3771f08a38301`.
  - [Post-Merge-2 archive](https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_MergeRequest_PR/69221/test-results/results-DGX_B300-4_GPUs-PyTorch-Post-Merge-2.tar.gz); downloaded-byte SHA-256 `0e897bdeefa2514f0d0305c99949eb3e216e17b1fa717dff04c83dba8a2259db`.
- #69544: [multi-GPU child #3142](https://prod.blsm.nvidia.com/sw-tensorrt-llm-github-1/job/LLM/job/main/job/L0_Test-x86_64-Multi-GPU/3142/) · [console](https://prod.blsm.nvidia.com/sw-tensorrt-llm-github-1/job/LLM/job/main/job/L0_Test-x86_64-Multi-GPU/3142/consoleText).
  - [Post-Merge-1 archive](https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_MergeRequest_PR/69544/test-results/results-DGX_B300-4_GPUs-PyTorch-Post-Merge-1.tar.gz); downloaded-byte SHA-256 `081fbd505b8930f08ff0709e8e7ab13ed5e951f78585b88363213b912f0f2341`.
  - [Post-Merge-2 archive](https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_MergeRequest_PR/69544/test-results/results-DGX_B300-4_GPUs-PyTorch-Post-Merge-2.tar.gz); downloaded-byte SHA-256 `82084038a89207b4f6e8e1ad37c811f287d3bae390410ce0260be1a0fc691261`.
- #70619: [multi-GPU child #3540](https://prod.blsm.nvidia.com/sw-tensorrt-llm-github-2/job/LLM/job/main/job/L0_Test-x86_64-Multi-GPU/3540/) · [console](https://prod.blsm.nvidia.com/sw-tensorrt-llm-github-2/job/LLM/job/main/job/L0_Test-x86_64-Multi-GPU/3540/consoleText).
  - [Post-Merge-1 archive](https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_MergeRequest_PR/70619/test-results/results-DGX_B300-4_GPUs-PyTorch-Post-Merge-1.tar.gz); downloaded-byte SHA-256 `4824f905de5420a0a57937d56343704902bd898c73f1fe3509d8175649037b0e`.
  - [Post-Merge-2 archive](https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_MergeRequest_PR/70619/test-results/results-DGX_B300-4_GPUs-PyTorch-Post-Merge-2.tar.gz); downloaded-byte SHA-256 `83e9d734e0687e2190b4417587466dd5ebe933f0fa0ef81227433aa47d3d4832`.
