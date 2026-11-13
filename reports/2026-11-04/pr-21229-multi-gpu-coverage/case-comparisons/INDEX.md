<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Case comparison index

[Back to the main report](../REPORT.en.md)

All available original JUnit records for the five #69221 stages and three failed #68871 stages are represented. Post-merge baselines show all retrieved #3051, #3053, #3055, and #3057 observations; only the successful stage baselines named in the main report are included in the median. The #69221 pre-merge GB300 comparison has only previously preserved historical matched case values, not complete raw baseline files; missing baselines remain explicit. Each report contains both on-only and baseline-only identities.

| CI | Stage | Unique JUnit identities | Classification counts | Parts |
|---|---|---:|---|---|
| #69221 | DGX_B300-4_GPUs-PyTorch-Post-Merge-1 | 2396 | Baseline only: 2225; Matched: 7; No eligible matched baseline: 156; On not passed / no time: 8 | [1](69221-0-01.md), [2](69221-0-02.md), [3](69221-0-03.md), [4](69221-0-04.md) |
| #69221 | DGX_B300-4_GPUs-PyTorch-Post-Merge-2 | 2397 | Baseline only: 186; Matched: 9; No eligible matched baseline: 2195; On not passed / no time: 7 | [1](69221-1-01.md), [2](69221-1-02.md), [3](69221-1-03.md), [4](69221-1-04.md) |
| #69221 | GB300-4_GPUs-PyTorch-Post-Merge-3 | 4514 | Baseline only: 2437; Matched: 345; No eligible matched baseline: 1715; On not passed / no time: 17 | [1](69221-2-01.md), [2](69221-2-02.md), [3](69221-2-03.md), [4](69221-2-04.md), [5](69221-2-05.md), [6](69221-2-06.md), [7](69221-2-07.md) |
| #69221 | DGX_H200-4_GPUs-PyTorch-Post-Merge-1 | 76 | Matched: 72; On not passed / no time: 4 | [1](69221-3-01.md) |
| #69221 | GB300-4_GPUs-PyTorch-1 | 8886 | Baseline only: 27; Matched: 4463; No eligible matched baseline: 3502; On not passed / no time: 894 | [1](69221-4-01.md), [2](69221-4-02.md), [3](69221-4-03.md), [4](69221-4-04.md), [5](69221-4-05.md), [6](69221-4-06.md), [7](69221-4-07.md), [8](69221-4-08.md), [9](69221-4-09.md), [10](69221-4-10.md), [11](69221-4-11.md), [12](69221-4-12.md), [13](69221-4-13.md) |
| #68871 | DGX_B300-4_GPUs-PyTorch-Post-Merge-1 | 2235 | Baseline only: 9; Matched: 2214; On not passed / no time: 12 | [1](68871-0-01.md), [2](68871-0-02.md), [3](68871-0-03.md), [4](68871-0-04.md) |
| #68871 | DGX_B300-4_GPUs-PyTorch-Post-Merge-2 | 199 | Baseline only: 41; Matched: 100; No eligible matched baseline: 51; On not passed / no time: 7 | [1](68871-1-01.md) |
| #68871 | GB300-4_GPUs-PyTorch-Post-Merge-3 | 2786 | Baseline only: 321; Matched: 2452; On not passed / no time: 13 | [1](68871-2-01.md), [2](68871-2-02.md), [3](68871-2-03.md), [4](68871-2-04.md) |
