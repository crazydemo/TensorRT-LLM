<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR Evidence — Failed Stage Type: MULTI-GPU

| Failed-stage details | Value |
|---|---|
| Stage | **DGX_B200-8_GPUs-PyTorch-4** |
| Type | **MULTI-GPU — 8 GPUs, conditional pre-merge** |
| C1 | test_ulysses_sage_attention.py |
| C2 | test_sage_ulysses_forward[True] |

| PR | Multi-GPU baseline | CBTS | C1 / C2 result | Escape relevance |
|---:|---|---|---|---|
| [#16632](https://github.com/NVIDIA/TensorRT-LLM/pull/16632) | ELIGIBLE | FALLBACK | PASS×1 / PASS×1 | No CBTS filtering |
| [#18064](https://github.com/NVIDIA/TensorRT-LLM/pull/18064) | ELIGIBLE | FALLBACK | PASS×1 / PASS×1 | No CBTS filtering |
| [#18236](https://github.com/NVIDIA/TensorRT-LLM/pull/18236) | ELIGIBLE | FALLBACK | PASS×2 / PASS×2 | No CBTS filtering |
| [#15727](https://github.com/NVIDIA/TensorRT-LLM/pull/15727) | ELIGIBLE | FALLBACK | PASS×1, REUSE×1 / PASS×1 | No CBTS filtering |
| [#18369](https://github.com/NVIDIA/TensorRT-LLM/pull/18369) | ELIGIBLE | FALLBACK | PASS×1, REUSE×1 / PASS×1 | No CBTS filtering |
| [#17393](https://github.com/NVIDIA/TensorRT-LLM/pull/17393) | ELIGIBLE | FALLBACK | PASS×1 / PASS×1 | No CBTS filtering |
| [#18409](https://github.com/NVIDIA/TensorRT-LLM/pull/18409) | ELIGIBLE | FALLBACK | PASS×3 / PASS×3 | No CBTS filtering |
| [#18387](https://github.com/NVIDIA/TensorRT-LLM/pull/18387) | INELIGIBLE | FALLBACK | NOT RUN / NOT RUN | Excluded by non-CBTS multi-GPU gate |
| [#18509](https://github.com/NVIDIA/TensorRT-LLM/pull/18509) | INELIGIBLE | FALLBACK | NOT RUN / NOT RUN | Excluded by non-CBTS multi-GPU gate |
| [#18425](https://github.com/NVIDIA/TensorRT-LLM/pull/18425) | INELIGIBLE | FALLBACK | NOT RUN / NOT RUN | Excluded by non-CBTS multi-GPU gate |
| [#18426](https://github.com/NVIDIA/TensorRT-LLM/pull/18426) | INELIGIBLE | FALLBACK | NOT RUN / NOT RUN | Excluded by non-CBTS multi-GPU gate |
| [#18460](https://github.com/NVIDIA/TensorRT-LLM/pull/18460) | INELIGIBLE | **HIT** | NOT RUN / NOT RUN | C1/C2 were outside this HIT’s scope |
| [#18331](https://github.com/NVIDIA/TensorRT-LLM/pull/18331) | INELIGIBLE | FALLBACK | NOT RUN / NOT RUN | Excluded by non-CBTS multi-GPU gate |

## Conclusion

**CBTS escape = FALSE.**

C1/C2 are globally reachable by CBTS, but #18460 was an unrelated WAIVE-only HIT.
The target B200 8-GPU stage was baseline-ineligible and outside that HIT’s selected scope.
