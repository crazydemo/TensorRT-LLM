<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Multi-GPU coverage collection: CI results and overhead

Updated 2026-11-13 · [PR #21229](https://github.com/NVIDIA/TensorRT-LLM/pull/21229)

This report evaluates which single-node multi-GPU stages can collect CBTS coverage safely, the observed runtime changes, and which coverage data needs investigation. Ray, multi-node, and CPP stages were excluded from collection in these experiments. Passing CI alone does not establish that the data is complete.

## Key findings

- **B200 paired experiment:** all four stages passed with collection off in [CI #69732](https://nv/trt-llm-cicd/job/main/job/L0_MergeRequest_PR/69732/) and on in [CI #69784](https://nv/trt-llm-cicd/job/main/job/L0_MergeRequest_PR/69784/). The commits differ only in the coverage enable switch. Observed wall-time changes were +0.2%, +4.9%, -0.1%, and -19.8%, respectively. One sample per condition cannot establish a stable overhead distribution; the faster M3 run does not show that collection improves performance.
- **Data quality:** #69784 B200 Post-Merge-2, Post-Merge-3, and M3 had 1, 6, and 1 tainted contexts, respectively; Post-Merge-4 had none. The M3 stage had only one context, which was tainted. Taint indicates an attribution or completeness risk, not necessarily a test failure; absence of taint is not proof of completeness.
- **Timeout:** #69221 B300 Post-Merge-1 failed when an outer `attention/multi_gpu` pytest batch reached the 3,600-second timeout. Its coverage DB has a stage-level `context_channel_unreachable` taint affecting all 21 nonempty contexts. A B300 stage also timed out with coverage disabled in #69544, but the exact inner test was not confirmed to be the same. The cause remains unproven.
- **Comparability:** #69221 was run after a rebase changed test definitions. In particular, the GB300 pre-merge test-db gained entries versus #67913; comparing those stage totals as collector overhead is invalid. The remaining historical differences also span commits and environments.

## How to read the table

All stage information is shown in one table. Durations are Jenkins **Run Pytest wall time**, including that step's setup and teardown; they exclude separate container preparation and any separately named isolated test step. For #69221 H200, an additional 191.8-second `Run Pytest (Isolated)` step is excluded. The #69784 paired differences use original millisecond durations before rounding, so subtracting displayed values may differ by 0.1 s. The #69221 differences are calculated from the displayed, rounded stage values.

- The format **on / off / difference** is seconds and percentage. #69784 uses the single matching #69732 run as its off baseline. For #67913 and #67077, historical x86 child jobs #3045/#3039 and SBSA child jobs #4518/#4515/#4514/#4513 form the off baseline. Post-merge #68569 and #68871 use official post-merge #3051/#3053/#3055/#3057, subject to the per-row availability shown below; GB300 #3051 failed and is excluded from the applicable baseline. Historical off values are medians where multiple observations exist.
- Contexts / passed / tainted counts refer to **nonempty outer contexts in the coverage DB**, not JUnit inner cases. Passed is the DB's aggregate outcome; tainted includes stage-level taint, counted once per affected context. A context may represent a pytest batch containing many inner tests.
- A negative difference is an observation, not proof that collecting coverage makes tests faster. Changes in commits, tests, machines, caches, load, and retries limit comparisons. Failed runs and changed test lists are marked accordingly; their arithmetic differences are shown but are not reliable estimates of collector overhead.

## Stage results

| Experiment / CI | Stage | Status | Assessment | On / off / difference | Contexts / passed / tainted | Off baseline observations (s) |
|---|---|---|---|---|---|---|
| Post-merge #69784 | DGX_B200-4_GPUs-PyTorch-Post-Merge-2 | SUCCESS | One run each; taint needs review | 5469.7 / 5460.0 / +9.8 (+0.2%) | 19 / 16 / 1 | #69732: 5460.0s |
| Post-merge #69784 | DGX_B200-4_GPUs-PyTorch-Post-Merge-3 | SUCCESS | One run each; taint needs review | 6362.0 / 6064.6 / +297.4 (+4.9%) | 18 / 16 / 6 | #69732: 6064.6s |
| Post-merge #69784 | DGX_B200-4_GPUs-PyTorch-Post-Merge-4 | SUCCESS | One run each; no taint (not proof of completeness) | 6062.0 / 6065.7 / -3.7 (-0.1%) | 19 / 17 / 0 | #69732: 6065.7s |
| Post-merge #69784 | DGX_B200-6_GPUs-PyTorch-M3-Post-Merge-1 | SUCCESS | One run each; taint needs review | 1258.2 / 1569.1 / -310.9 (-19.8%) | 1 / 1 / 1 | #69732: 1569.1s |
| Rerun #69221 | DGX_B300-4_GPUs-PyTorch-Post-Merge-1 | FAILURE | Outer test timed out at 3,600 s; duration is incomplete; [case details](case-comparisons/69221-0-01.md) | 9069.1 / 9712.0 / -642.9 (-6.6%) | 21 / 14 / 21 | #3051: 9712.0, #3053: 9674.7, #3057: 9975.7 |
| Rerun #69221 | DGX_B300-4_GPUs-PyTorch-Post-Merge-2 | SUCCESS | Rerun passed; verify test list and single baseline; [case details](case-comparisons/69221-1-01.md) | 9949.1 / 10880.2 / -931.1 (-8.6%) | 24 / 19 / 1 | #3051: 10880.2 |
| Rerun #69221 | GB300-4_GPUs-PyTorch-Post-Merge-3 | SUCCESS | Rerun passed; evaluate duration; [case details](case-comparisons/69221-2-01.md) | 11182.5 / 9715.4 / +1467.1 (+15.1%) | 31 / 27 / 2 | #3051: 9153.2, #3053: 9715.4, #3057: 10055.2 |
| Rerun #69221 | DGX_H200-4_GPUs-PyTorch-Post-Merge-1 | SUCCESS | Rerun passed; evaluate duration; [case details](case-comparisons/69221-3-01.md) | 9426.0 / 7902.5 / +1523.5 (+19.3%) | 67 / 65 / 1 | #3051: 8091.3, #3053: 7813.6, #3055: 7730.5, #3057: 7991.4 |
| Rerun #69221 | GB300-4_GPUs-PyTorch-1 | SUCCESS | Test list changed; observed difference only; [case details](case-comparisons/69221-4-01.md) | 11861.3 / 8221.6 / +3639.7 (+44.3%) | 98 / 98 / 1 | 8216.7, 8228.0, 7656.4, 8226.6 |
| Pre-merge #67913 | DGX_B200-2_GPUs-PyTorch-1 | SUCCESS | Ran; tainted contexts need review | 4586.1 / 3673.7 / +912.4 (+24.8%) | 25 / 24 / 2 | 3983.7, 3363.7 |
| Pre-merge #67913 | DGX_B200-4_GPUs-PyTorch-1 | SUCCESS | Ran; tainted contexts need review | 6654.6 / 6386.8 / +267.9 (+4.2%) | 18 / 18 / 3 | 6072.7, 6700.8 |
| Pre-merge #67913 | DGX_B200-4_GPUs-PyTorch-2 | SUCCESS | Ran; tainted contexts need review | 6355.6 / 6219.0 / +136.6 (+2.2%) | 18 / 16 / 1 | 5780.3, 6657.7 |
| Pre-merge #67913 | DGX_B200-4_GPUs-PyTorch-3 | SUCCESS | Ran; tainted contexts need review | 6054.7 / 5764.7 / +290.0 (+5.0%) | 18 / 18 / 2 | 6367.7, 5161.7 |
| Pre-merge #67913 | DGX_B200-4_GPUs-PyTorch-Ray-1 | SUCCESS | Excluded: Ray / multi-node / CPP | 6059.0 / 6218.5 / -159.5 (-2.6%) | Excluded by rule | 6364.4, 6072.6 |
| Pre-merge #67913 | DGX_B200-8_GPUs-PyTorch-1 | SUCCESS | Ran; tainted contexts need review | 4858.5 / 4712.6 / +145.9 (+3.1%) | 14 / 14 / 3 | 4867.9, 4557.2 |
| Pre-merge #67913 | DGX_B200-8_GPUs-PyTorch-2 | SUCCESS | Ran; tainted contexts need review | 4558.4 / 4872.9 / -314.5 (-6.5%) | 14 / 12 / 3 | 3985.5, 5760.4 |
| Pre-merge #67913 | DGX_B200-8_GPUs-PyTorch-3 | SUCCESS | Ran; tainted contexts need review | 4862.7 / 4717.5 / +145.2 (+3.1%) | 13 / 12 / 1 | 4573.5, 4861.4 |
| Pre-merge #67913 | DGX_B200-8_GPUs-PyTorch-4 | SUCCESS | Ran; tainted contexts need review | 5161.2 / 5010.9 / +150.3 (+3.0%) | 13 / 13 / 2 | 4860.7, 5161.1 |
| Pre-merge #67913 | DGX_B200-8_GPUs-PyTorch-Ray-1 | SUCCESS | Excluded: Ray / multi-node / CPP | 2458.4 / 1712.7 / +745.7 (+43.5%) | Excluded by rule | 1572.3, 1853.1 |
| Pre-merge #67913 | DGX_B300-4_GPUs-PyTorch-1 | SUCCESS | Ran; tainted contexts need review | 5779.5 / 5751.3 / +28.2 (+0.5%) | 34 / 34 / 3 | 5741.2, 5761.4 |
| Pre-merge #67913 | DGX_H100-2_GPUs-PyTorch-GptOss-1 | SUCCESS | Ran; tainted contexts need review | 3059.2 / 2851.3 / +207.9 (+7.3%) | 3 / 3 / 1 | 2912.9, 2789.7 |
| Pre-merge #67913 | DGX_H100-2_GPUs-PyTorch-ModelExpress-1 | SUCCESS | Ran; validate coverage and overhead | 359.5 / 354.9 / +4.7 (+1.3%) | 2 / 2 / 0 | 334.2, 375.5 |
| Pre-merge #67913 | DGX_H100-2_GPUs-PyTorch-Others-1 | SUCCESS | Ran; tainted contexts need review | 2963.5 / 2972.8 / -9.3 (-0.3%) | 13 / 13 / 5 | 3077.5, 2868.1 |
| Pre-merge #67913 | DGX_H100-2_GPUs-PyTorch-Others-2 | SUCCESS | Ran; tainted contexts need review | 2520.0 / 2597.4 / -77.4 (-3.0%) | 15 / 14 / 6 | 2540.9, 2653.9 |
| Pre-merge #67913 | DGX_H100-2_GPUs-PyTorch-Ray-1 | SUCCESS | Excluded: Ray / multi-node / CPP | 2482.5 / 2802.1 / -319.6 (-11.4%) | Excluded by rule | 2450.0, 3154.2 |
| Pre-merge #67913 | DGX_H100-4_GPUs-CPP-1 | SUCCESS | Excluded: Ray / multi-node / CPP | 756.5 / 977.6 / -221.1 (-22.6%) | Excluded by rule | 698.3, 1256.9 |
| Pre-merge #67913 | DGX_H100-4_GPUs-PyTorch-DeepSeek-1 | SUCCESS | Ran; tainted contexts need review | 4418.3 / 4628.0 / -209.7 (-4.5%) | 37 / 26 / 7 | 4873.6, 4382.4 |
| Pre-merge #67913 | DGX_H100-4_GPUs-PyTorch-GptOss-1 | SUCCESS | Ran; validate coverage and overhead | 581.7 / 679.8 / -98.1 (-14.4%) | 10 / 1 / 0 | 694.5, 665.2 |
| Pre-merge #67913 | DGX_H100-4_GPUs-PyTorch-Others-1 | SUCCESS | Ran; tainted contexts need review | 2919.4 / 3273.0 / -353.6 (-10.8%) | 20 / 18 / 10 | 3200.8, 3345.1 |
| Pre-merge #67913 | DGX_H100-4_GPUs-PyTorch-Others-2 | SUCCESS | Ran; tainted contexts need review | 3019.5 / 2926.6 / +92.9 (+3.2%) | 20 / 18 / 12 | 2913.7, 2939.5 |
| Pre-merge #67913 | DGX_H100-4_GPUs-PyTorch-Ray-1 | SUCCESS | Excluded: Ray / multi-node / CPP | 1165.7 / 1214.3 / -48.6 (-4.0%) | Excluded by rule | 1124.6, 1304.0 |
| Pre-merge #67913 | GB200-4_GPUs-PyTorch-1 | SUCCESS | Ran; tainted contexts need review | 5072.6 / 6404.4 / -1331.9 (-20.8%) | 13 / 11 / 2 | 6104.8, 6787.7, 6704.1, 5981.3 |
| Pre-merge #67913 | GB200-4_GPUs-PyTorch-2 | SUCCESS | Ran; tainted contexts need review | 3660.1 / 3559.4 / +100.6 (+2.8%) | 12 / 11 / 1 | 3510.5, 3647.6, 3608.4, 2887.6 |
| Pre-merge #67913 | GB200-4_GPUs-PyTorch-3 | SUCCESS | Ran; tainted contexts need review | 2473.3 / 2506.8 / -33.5 (-1.3%) | 12 / 11 / 1 | 2500.3, 2493.4, 2513.3, 3276.6 |
| Pre-merge #67913 | GB200-4_GPUs-PyTorch-4 | SUCCESS | Ran; tainted contexts need review | 3821.8 / 3433.9 / +387.9 (+11.3%) | 12 / 12 / 1 | 3379.9, 3542.2, 3487.9, 2922.5 |
| Pre-merge #67913 | GB200-4_GPUs-PyTorch-5 | SUCCESS | Ran; tainted contexts need review | 5322.9 / 4384.4 / +938.5 (+21.4%) | 12 / 12 / 1 | 4432.3, 4336.4, 4627.9, 4292.9 |
| Pre-merge #67913 | GB200-8_GPUs-2_Nodes-PyTorch-1 | SUCCESS | Excluded: Ray / multi-node / CPP | 3417.0 / 3409.4 / +7.6 (+0.2%) | Excluded by rule | 3419.0, 3722.9, 3381.2, 3399.8 |
| Pre-merge #67913 | GB200-8_GPUs-2_Nodes-PyTorch-2 | SUCCESS | Excluded: Ray / multi-node / CPP | 2552.8 / 2516.5 / +36.2 (+1.4%) | Excluded by rule | 2778.0, 2508.5, 2216.7, 2524.6 |
| Pre-merge #67913 | GB300-4_GPUs-PyTorch-1 | SUCCESS | Ran; tainted contexts need review | 9751.1 / 8221.6 / +1529.5 (+18.6%) | 59 / 59 / 1 | 8216.7, 8228.0, 7656.4, 8226.6 |
| Pre-merge #67077 | DGX_B200-2_GPUs-PyTorch-1 | SUCCESS | Historical comparison; matched 34 cases | 3981.9 / 3673.7 / +308.2 (+8.4%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_B200-4_GPUs-PyTorch-1 | SUCCESS | Historical comparison; matched 5 cases | 6098.4 / 6386.8 / -288.4 (-4.5%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_B200-4_GPUs-PyTorch-2 | FAILED | Historical comparison; matched 0 cases | 6997.7 / 6219.0 / +778.7 (+12.5%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_B200-4_GPUs-PyTorch-3 | SUCCESS | Historical comparison; matched 0 cases | 5791.5 / 5764.7 / +26.8 (+0.5%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_B200-4_GPUs-PyTorch-Ray-1 | SUCCESS | Historical comparison; matched 20 cases | 6399.7 / 6218.5 / +181.2 (+2.9%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_B200-8_GPUs-PyTorch-1 | SUCCESS | Historical comparison; matched 28 cases | 5525.6 / 4712.6 / +813.1 (+17.3%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_B200-8_GPUs-PyTorch-2 | SUCCESS | Historical comparison; matched 19 cases | 4284.5 / 4872.9 / -588.5 (-12.1%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_B200-8_GPUs-PyTorch-3 | SUCCESS | Historical comparison; matched 3 cases | 4580.5 / 4717.5 / -137.0 (-2.9%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_B200-8_GPUs-PyTorch-4 | SUCCESS | Historical comparison; matched 10 cases | 5184.6 / 5010.9 / +173.7 (+3.5%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_B200-8_GPUs-PyTorch-Ray-1 | SUCCESS | Historical comparison; matched 2 cases | 1882.5 / 1712.7 / +169.8 (+9.9%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_B300-4_GPUs-PyTorch-1 | SUCCESS | Historical comparison; matched 41 cases | 6046.8 / 5751.3 / +295.5 (+5.1%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_H100-2_GPUs-PyTorch-GptOss-1 | SUCCESS | Historical comparison; matched 3 cases | 2762.5 / 2851.3 / -88.8 (-3.1%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_H100-2_GPUs-PyTorch-ModelExpress-1 | SUCCESS | Historical comparison; matched 7 cases | 433.0 / 354.9 / +78.2 (+22.0%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_H100-2_GPUs-PyTorch-Others-1 | SUCCESS | Historical comparison; matched 19 cases | 3427.0 / 2972.8 / +454.2 (+15.3%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_H100-2_GPUs-PyTorch-Others-2 | SUCCESS | Historical comparison; matched 10 cases | 2572.6 / 2597.4 / -24.8 (-1.0%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_H100-2_GPUs-PyTorch-Ray-1 | SUCCESS | Historical comparison; matched 65 cases | 2556.0 / 2802.1 / -246.1 (-8.8%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_H100-4_GPUs-PyTorch-DeepSeek-1 | SUCCESS | Historical comparison; matched 28 cases | 4564.5 / 4628.0 / -63.5 (-1.4%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_H100-4_GPUs-PyTorch-GptOss-1 | SUCCESS | Historical comparison; matched 1 cases | 715.3 / 679.8 / +35.4 (+5.2%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_H100-4_GPUs-PyTorch-Others-1 | SUCCESS | Historical comparison; matched 36 cases | 2887.3 / 3273.0 / -385.7 (-11.8%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_H100-4_GPUs-PyTorch-Others-2 | SUCCESS | Historical comparison; matched 58 cases | 3311.2 / 2926.6 / +384.6 (+13.1%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | DGX_H100-4_GPUs-PyTorch-Ray-1 | SUCCESS | Historical comparison; matched 9 cases | 1342.2 / 1214.3 / +127.9 (+10.5%) | See historical details | Baseline child jobs: 3045,3039 |
| Pre-merge #67077 | GB200-4_GPUs-PyTorch-1 | SUCCESS | Historical comparison; matched 72 cases | 6581.3 / 6404.4 / +176.9 (+2.8%) | See historical details | Baseline child jobs: 4518,4515,4514,4513 |
| Pre-merge #67077 | GB200-4_GPUs-PyTorch-2 | SUCCESS | Historical comparison; matched 88 cases | 3666.0 / 3559.4 / +106.6 (+3.0%) | See historical details | Baseline child jobs: 4518,4515,4514,4513 |
| Pre-merge #67077 | GB200-4_GPUs-PyTorch-3 | SUCCESS | Historical comparison; matched 27 cases | 2663.4 / 2506.8 / +156.6 (+6.2%) | See historical details | Baseline child jobs: 4518,4515,4514,4513 |
| Pre-merge #67077 | GB200-4_GPUs-PyTorch-4 | SUCCESS | Historical comparison; matched 251 cases | 3910.5 / 3433.9 / +476.6 (+13.9%) | See historical details | Baseline child jobs: 4518,4515,4514,4513 |
| Pre-merge #67077 | GB200-4_GPUs-PyTorch-5 | SUCCESS | Historical comparison; matched 129 cases | 4679.2 / 4384.4 / +294.8 (+6.7%) | See historical details | Baseline child jobs: 4518,4515,4514,4513 |
| Pre-merge #67077 | GB300-4_GPUs-PyTorch-1 | FAILED | Historical comparison; matched 4464 cases | 9138.2 / 8221.6 / +916.6 (+11.1%) | See historical details | Baseline child jobs: 4518,4515,4514,4513 |
| Post-merge #68569 | DGX_B200-4_GPUs-PyTorch-Post-Merge-1 | SUCCESS | Ran; no taint; one sample | 6353.6 / 6094.3 / +259.3 (+4.3%) | 18 / 17 / 0 | #3051: 6093.8, #3053: 6094.8, #3055: 5775.7, #3057: 6094.9 |
| Post-merge #68569 | GB300-4_GPUs-PyTorch-Post-Merge-1 | SUCCESS | Ran; tainted contexts need review | 11203.0 / 10983.5 / +219.5 (+2.0%) | 28 / 25 / 1 | #3053: 10335.8, #3055: 10983.5, #3057: 11535.9; excluded #3051: FAILED |
| Post-merge #68871 | DGX_B200-4_GPUs-PyTorch-Post-Merge-1 | SUCCESS | Passed; no taint | 6365.1 / 6094.3 / +270.8 (+4.4%) | 18 / 17 / 0 | #3051: 6093.8, #3053: 6094.8, #3055: 5775.7, #3057: 6094.9 |
| Post-merge #68871 | DGX_B200-4_GPUs-PyTorch-Post-Merge-2 | SUCCESS | Passed; no taint | 6052.0 / 5653.4 / +398.6 (+7.1%) | 19 / 17 / 0 | #3051: 5798.1, #3053: 5791.3, #3055: 5474.9, #3057: 5515.6 |
| Post-merge #68871 | DGX_B200-4_GPUs-PyTorch-Post-Merge-3 | SUCCESS | Passed; tainted contexts | 6053.1 / 5932.6 / +120.4 (+2.0%) | 18 / 16 / 1 | #3051: 6069.8, #3053: 5795.5, #3055: 5763.3, #3057: 6399.3 |
| Post-merge #68871 | DGX_B200-4_GPUs-PyTorch-Post-Merge-4 | SUCCESS | Passed; no taint | 6368.7 / 6244.7 / +124.0 (+2.0%) | 18 / 16 / 0 | #3051: 6394.8, #3053: 6094.6, #3055: 6082.9, #3057: 6403.2 |
| Post-merge #68871 | DGX_B200-6_GPUs-PyTorch-M3-Post-Merge-1 | SUCCESS | Passed; tainted contexts | 1250.2 / 1295.4 / -45.3 (-3.5%) | 1 / 1 / 1 | #3051: 1898.2, #3053: 1263.6, #3055: 1276.6, #3057: 1314.2 |
| Post-merge #68871 | DGX_B300-4_GPUs-PyTorch-Post-Merge-1 | FAILED | Failed; durations not directly comparable; [case details](case-comparisons/68871-0-01.md) | 10253.9 / 9712.0 / +541.8 (+5.6%) | 24 / 19 / 3 | #3051: 9712.0, #3053: 9674.7, #3057: 9975.7 |
| Post-merge #68871 | DGX_B300-4_GPUs-PyTorch-Post-Merge-2 | FAILED | Failed; durations not directly comparable; [case details](case-comparisons/68871-1-01.md) | 10847.2 / 10880.2 / -33.0 (-0.3%) | 22 / 16 / 0 | #3051: 10880.2 |
| Post-merge #68871 | DGX_H100-2_GPUs-PyTorch-ModelExpress-Post-Merge-1 | SUCCESS | Passed; no taint | 2365.9 / 2670.9 / -305.0 (-11.4%) | 7 / 7 / 0 | #3051: 2701.6, #3053: 2640.2, #3055: 2527.2, #3057: 2793.0 |
| Post-merge #68871 | DGX_H100-4_GPUs-PyTorch-Post-Merge-1 | SUCCESS | Passed; no taint | 976.3 / 1036.0 / -59.8 (-5.8%) | 3 / 3 / 0 | #3051: 1061.9, #3053: 1033.3, #3055: 930.4, #3057: 1038.8 |
| Post-merge #68871 | DGX_H200-4_GPUs-PyTorch-Post-Merge-1 | SUCCESS | Passed; tainted contexts | 9539.8 / 7902.5 / +1637.2 (+20.7%) | 67 / 65 / 1 | #3051: 8091.3, #3053: 7813.6, #3055: 7730.5, #3057: 7991.4 |
| Post-merge #68871 | DGX_H200-8_GPUs-PyTorch-Post-Merge-1 | SUCCESS | Passed; tainted contexts | 8566.5 / 8634.1 / -67.6 (-0.8%) | 17 / 16 / 9 | #3051: 8451.0, #3053: 8636.3, #3055: 8632.0, #3057: 8794.7 |
| Post-merge #68871 | GB200-4_GPUs-PyTorch-Post-Merge-1 | SUCCESS | Passed; no taint | 9167.0 / 8939.3 / +227.7 (+2.5%) | 22 / 21 / 0 | #3051: 9368.6, #3053: 8751.1, #3055: 9127.4, #3057: 8241.4 |
| Post-merge #68871 | GB200-4_GPUs-PyTorch-Post-Merge-2 | SUCCESS | Passed; no taint | 7646.2 / 8232.5 / -586.3 (-7.1%) | 22 / 21 / 0 | #3051: 8730.0, #3053: 8377.5, #3055: 7958.6, #3057: 8087.5 |
| Post-merge #68871 | GB300-4_GPUs-PyTorch-Post-Merge-1 | SUCCESS | Passed; tainted contexts | 11034.1 / 10983.5 / +50.5 (+0.5%) | 28 / 25 / 1 | #3053: 10335.8, #3055: 10983.5, #3057: 11535.9 |
| Post-merge #68871 | GB300-4_GPUs-PyTorch-Post-Merge-2 | SUCCESS | Passed; no taint | 9526.2 / 9421.3 / +104.9 (+1.1%) | 29 / 27 / 0 | #3051: 9422.4, #3053: 9420.2, #3055: 8598.4, #3057: 10708.7 |
| Post-merge #68871 | GB300-4_GPUs-PyTorch-Post-Merge-3 | FAILED | Failed; durations not directly comparable; [case details](case-comparisons/68871-2-01.md) | 9828.4 / 9715.4 / +113.1 (+1.2%) | 29 / 26 / 2 | #3051: 9153.2, #3053: 9715.4, #3057: 10055.2 |

## Case-level comparisons for unmatched and failed stages

The stage totals below cannot all be interpreted as overhead: the selected cases changed, or the stage failed before completing. The [case comparison index](case-comparisons/INDEX.md) links to **all available case identities**, including matching passing cases, failed or skipped cases, and cases present only in one side. Each row gives the original outcome and duration for every available baseline. An em dash means no eligible matched observation or an incomplete result; it does not mean zero overhead.

The first three B300/GB300 #69221 stages have many unmatched identities because their selected tests differ from the successful historical baselines. #69221 H200 has 72 passing case identities that match a passing baseline. The pre-merge GB300 comparison has 4,463 matching passing identities, but the raw historical baselines are not fully available: the earlier report preserved values only for cases with at least two passing baseline observations. Consequently, its 3,502 on-side identities without an eligible match cannot be assigned a credible overhead from the available data.

Original result files can contain both an outer pytest wrapper and its inner tests. Their durations overlap, so **do not sum case rows** to calculate a stage duration. Deltas use unique passing on/off records and the median of eligible passing baselines; a single baseline is labeled as such. Failed, skipped, duplicated and timeout cases are retained but are not assigned a passing-case delta. The comparisons remain cross-commit observations rather than proof of collector causation.

| CI | Stage | Available identities | Matched passing | No eligible off match | On failed/skipped/no time | Baseline only | Details |
|---|---|---:|---:|---:|---:|---:|---|
| #69221 | DGX_B300-4_GPUs-PyTorch-Post-Merge-1 | 2396 | 7 | 156 | 8 | 2225 | [All parts](case-comparisons/INDEX.md) · [Part 1](case-comparisons/69221-0-01.md) |
| #69221 | DGX_B300-4_GPUs-PyTorch-Post-Merge-2 | 2397 | 9 | 2195 | 7 | 186 | [All parts](case-comparisons/INDEX.md) · [Part 1](case-comparisons/69221-1-01.md) |
| #69221 | GB300-4_GPUs-PyTorch-Post-Merge-3 | 4514 | 345 | 1715 | 17 | 2437 | [All parts](case-comparisons/INDEX.md) · [Part 1](case-comparisons/69221-2-01.md) |
| #69221 | DGX_H200-4_GPUs-PyTorch-Post-Merge-1 | 76 | 72 | 0 | 4 | 0 | [All parts](case-comparisons/INDEX.md) · [Part 1](case-comparisons/69221-3-01.md) |
| #69221 | GB300-4_GPUs-PyTorch-1 | 8886 | 4463 | 3502 | 894 | 27 | [All parts](case-comparisons/INDEX.md) · [Part 1](case-comparisons/69221-4-01.md) |
| #68871 | DGX_B300-4_GPUs-PyTorch-Post-Merge-1 | 2235 | 2214 | 0 | 12 | 9 | [All parts](case-comparisons/INDEX.md) · [Part 1](case-comparisons/68871-0-01.md) |
| #68871 | DGX_B300-4_GPUs-PyTorch-Post-Merge-2 | 199 | 100 | 51 | 7 | 41 | [All parts](case-comparisons/INDEX.md) · [Part 1](case-comparisons/68871-1-01.md) |
| #68871 | GB300-4_GPUs-PyTorch-Post-Merge-3 | 2786 | 2452 | 0 | 13 | 321 | [All parts](case-comparisons/INDEX.md) · [Part 1](case-comparisons/68871-2-01.md) |

## Coverage DB findings

The #69784 x86 DB and #69221 x86 and SBSA DBs were downloaded and passed SQLite `integrity_check`.

- #69784 B200-4 Post-Merge-2: one context with `subscriber_confirmed_dead`.
- #69784 B200-4 Post-Merge-3: six contexts affected by `context_not_acknowledged`, `subscriber_stopped_recording`, or `subscriber_confirmed_dead`.
- #69784 B200-4 Post-Merge-4: no tainted contexts were recorded.
- #69784 B200-6 M3: its only context has `subscriber_confirmed_dead`.
- #69221 B300-4 Post-Merge-1: an empty-context `context_channel_unreachable` taint applies to the stage, marking all 21 nonempty contexts tainted. B300 Post-Merge-2, H200 Post-Merge-1, and GB300 pre-merge each have one tainted context; GB300 Post-Merge-3 has two. Their recorded reasons include `subscriber_confirmed_dead`.

## Follow-up

1. Align JUnit cases between #69732 and #69784, check matching coverage and outcomes, and identify which cases explain the B200 Post-Merge-3 increase and M3 decrease.
2. Investigate the tainted B200 contexts, especially the single M3 context, before treating their collected data as complete.
3. Inspect #69221 and #69544 B300 timeout artifacts to establish whether the same inner tests timed out; use a controlled on/off comparison if needed.
4. Decide per stage whether the observed overhead and data quality meet the rollout requirements. Repeat paired runs where normal variance or a timeout makes a single pair inconclusive.

## Evidence and detailed data

- [HTML report and historical per-case comparisons](REPORT.html). The HTML report contains the original detailed case tables. The new [case comparison index](case-comparisons/INDEX.md) includes raw on/off case observations for the unmatched and failed stages.
- [#69732 coverage off: x86 multi-GPU child job #905](https://prod.blsm.nvidia.com/sw-tensorrt-llm-github-4/job/LLM/job/main/job/L0_Test-x86_64-Multi-GPU/905/) · [#69784 coverage on: child job #912](https://prod.blsm.nvidia.com/sw-tensorrt-llm-github-4/job/LLM/job/main/job/L0_Test-x86_64-Multi-GPU/912/).
- [#69784 x86 multi-GPU coverage archive](https://artifactory.nvidia.com/ui/native/sw-tensorrt-llm-generic-local/llm-artifacts/LLM/main/L0_MergeRequest_PR/69784/cbts-coverage/cbts_pystart_report_x86_64_multi_gpu.tar.gz) · [#69221 CI](https://nv/trt-llm-cicd/job/main/job/L0_MergeRequest_PR/69221/) · [#69221 failure analysis](https://pbss.s8k.io/v1/AUTH_svc_tensorrt/sw-tensorrt-ci-analysis/LLM/main/L0_MergeRequest_PR/69221/failure_analysis.html).
- [#69544 CI](https://nv/trt-llm-cicd/job/main/job/L0_MergeRequest_PR/69544/) · [#69544 failure analysis](https://pbss.s8k.io/v1/AUTH_svc_tensorrt/sw-tensorrt-ci-analysis/LLM/main/L0_MergeRequest_PR/69544/failure_analysis.html).
