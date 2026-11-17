<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Case-level evidence summary

[Main report](REPORT.en.md) · [Original case tables](case-comparisons/INDEX.md)

Recomputed from all parts of the existing Markdown case tables on 2026-11-17. No new CI measurements are introduced. Identity means exact JUnit classname plus test name, including parameters. Historical off presence is the **union of retrieved baselines**, including records from stages not eligible for timing medians. It is not a single off run. An on-only identity can reflect changed grouping, renaming or selection; it does not prove a newly added test. Missing identities in a failed run can reflect incomplete execution.

Timing statistics use only the existing table's unique passing comparisons and eligible off medians; they retain its baseline exclusions. Outer wrappers overlap inner cases: neither case counts nor sums of case durations measure independent work or stage overhead. Values are based on rounded Markdown data; very short inner cases can have large percentages for tiny absolute changes.

## Case-set overlap

| On CI / stage | On identities | Off union identities | Shared identities | On only | Off only | Passing timing comparisons |
|---|---:|---:|---:|---:|---:|---:|
| [#69221 B300 Post-Merge-1](case-comparisons/69221-0-01.md) | 171 | 2,235 | 10 | 161 | 2,225 | 7 |
| [#69221 B300 Post-Merge-2](case-comparisons/69221-1-01.md) | 2,211 | 198 | 12 | 2,199 | 186 | 9 |
| [#69221 GB300 Post-Merge-3](case-comparisons/69221-2-01.md) | 2,077 | 2,786 | 349 | 1,728 | 2,437 | 345 |
| [#69221 H200 Post-Merge-1](case-comparisons/69221-3-01.md) | 76 | 76 | 76 | 0 | 0 | 72 |
| [#70387 GB300 pre-merge vs #70097](case-comparisons/70387-gb300-01.md) | 9,160 | 9,160 | 9,160 | 0 | 0 | 8,264 |

## Passing-case timing distribution

These are descriptive per-identity statistics, not a stage-average overhead. The ±10 s and +30 s thresholds are summaries, not acceptance criteria.

| Stage | Comparable passed | Within ±10 s | Increase >30 s | Median difference (s) |
|---|---:|---:|---:|---:|
| #69221 B300 Post-Merge-1 | 7 | 1 | 2 | +12.492 |
| #69221 B300 Post-Merge-2 | 9 | 0 | 4 | +10.734 |
| #69221 GB300 Post-Merge-3 | 345 | 336 | 6 | +0.001 |
| #69221 H200 Post-Merge-1 | 72 | 19 | 36 | +29.715 |
| #70387 GB300 pre-merge vs #70097 | 8,264 | 8,250 | 2 | +0.000 |

## B300 redistribution between stages

For this diagnostic only, remove `DGX_B300-4_GPUs-PyTorch-Post-Merge-1.` or `...-2.` from the classname, preserve the remaining classname and full parameterized test name, and deduplicate identities across the two stages. This is a normalized overlap check, not a recomputation of timing deltas. Off presence uses all displayed historical records, including unsuccessful baselines; successful timing eligibility remains unchanged.

The combined sets contain **2,382 on and 2,433 off identities**, with **2,310 shared**, **72 on only**, and **123 off only**. Much of the stage-local mismatch is therefore consistent with redistribution. This does not establish identical executed workloads: the off side is a union, one on stage timed out, and code changed.

## Largest observed passing-case changes

Full parameter values and baseline observations are available through each linked source.

### #69221 B300 Post-Merge-1

| Case | Off | On | Difference |
|---|---|---|---|
| `DGX_B300-4_GPUs-PyTorch-Post-Merge-1.accuracy.test_llm_api_pytorch.TestDeepSeekV3Lite` · `test_nvfp4_4gpus[moe_backend=TRTLLM-mtp_nextn=2-ep4-fp8kv=True-attention_dp=True-cuda_graph=True-overlap_scheduler=True-low_precision_combine=False-torch_compile=False]` | #3051: 493.561s passed<br>#3053: 516.705s passed<br>#3055: 490.647s passed [stage not in valid baseline]<br>#3057: 495.246s passed | 583.128s passed | +87.882s (+17.7%) |
| `DGX_B300-4_GPUs-PyTorch-Post-Merge-1.accuracy.test_llm_api_pytorch.TestDeepSeekV3Lite` · `test_bfloat16_4gpus[tp2pp2-mtp_nextn=2-attention_dp=False-cuda_graph=False-overlap_scheduler=False-torch_compile=True]` | #3051: 276.346s passed<br>#3053: 277.925s passed<br>#3055: 275.511s passed [stage not in valid baseline]<br>#3057: 272.926s passed | 309.039s passed | +32.693s (+11.8%) |
| `DGX_B300-4_GPUs-PyTorch-Post-Merge-1.accuracy.test_llm_api_pytorch.TestDeepSeekV3Lite` · `test_bfloat16_4gpus[tp2pp2-mtp_nextn=2-attention_dp=True-cuda_graph=True-overlap_scheduler=True-torch_compile=False]` | #3051: 264.997s passed<br>#3053: 272.990s passed<br>#3055: 271.937s passed [stage not in valid baseline]<br>#3057: 267.085s passed | 281.632s passed | +14.547s (+5.4%) |
| `DGX_B300-4_GPUs-PyTorch-Post-Merge-1.accuracy.test_llm_api_pytorch.TestDeepSeekV3Lite` · `test_nvfp4_4gpus[moe_backend=CUTLASS-mtp_nextn=0-pp4-fp8kv=True-attention_dp=True-cuda_graph=True-overlap_scheduler=True-low_precision_combine=False-torch_compile=True]` | #3051: 498.296s passed<br>#3053: 500.103s passed<br>#3055: 492.518s passed [stage not in valid baseline]<br>#3057: 495.101s passed | 366.362s passed | -131.934s (-26.5%) |

### #69221 B300 Post-Merge-2

| Case | Off | On | Difference |
|---|---|---|---|
| `DGX_B300-4_GPUs-PyTorch-Post-Merge-2.accuracy.test_llm_api_pytorch.TestQwen3_30B_A3B` · `test_nvfp4[tep4_latency_moe_trtllm-torch_compile=False]` | #3051: 230.515s passed<br>#3053: 295.094s passed [stage not in valid baseline]<br>#3055: 283.211s passed [stage not in valid baseline]<br>#3057: 282.692s passed [stage not in valid baseline] | 355.475s passed | +124.960s (+54.2%) |
| `DGX_B300-4_GPUs-PyTorch-Post-Merge-2.accuracy.test_llm_api_pytorch.TestDeepSeekV3Lite` · `test_nvfp4_4gpus[moe_backend=CUTLASS-mtp_nextn=2-pp4-fp8kv=False-attention_dp=False-cuda_graph=False-overlap_scheduler=False-low_precision_combine=False-torch_compile=False]` | #3051: 329.725s passed<br>#3053: 350.701s passed [stage not in valid baseline]<br>#3055: 385.195s passed [stage not in valid baseline]<br>#3057: 371.481s passed [stage not in valid baseline] | 425.748s passed | +96.023s (+29.1%) |
| `DGX_B300-4_GPUs-PyTorch-Post-Merge-2.accuracy.test_llm_api_pytorch.TestDeepSeekV3Lite` · `test_nvfp4_4gpus[moe_backend=CUTLASS-mtp_nextn=2-tp2pp2-fp8kv=False-attention_dp=False-cuda_graph=False-overlap_scheduler=False-low_precision_combine=False-torch_compile=False]` | #3051: 224.870s passed<br>#3053: 327.739s passed [stage not in valid baseline]<br>#3055: 312.515s passed [stage not in valid baseline]<br>#3057: 223.174s passed [stage not in valid baseline] | 289.816s passed | +64.946s (+28.9%) |
| `DGX_B300-4_GPUs-PyTorch-Post-Merge-2.accuracy.test_llm_api_pytorch.TestDeepSeekR1` · `test_nvfp4_multi_gpus[throughput_tp4]` | #3051: 2681.259s passed<br>#3053: 2674.459s passed [stage not in valid baseline]<br>#3055: 2536.776s passed [stage not in valid baseline]<br>#3057: 2297.514s passed [stage not in valid baseline] | 703.477s passed | -1977.782s (-73.8%) |

### #69221 GB300 Post-Merge-3

| Case | Off | On | Difference |
|---|---|---|---|
| `GB300-4_GPUs-PyTorch-Post-Merge-3.accuracy.test_llm_api_pytorch.TestQwen3_8_Flash_Next` · `test_fp8_adp4_mtp3_prefix_cache` | #3051: 880.217s passed<br>#3053: 888.901s passed<br>#3055: 891.265s passed [stage not in valid baseline]<br>#3057: 885.838s passed | 1028.037s passed | +142.199s (+16.1%) |
| `GB300-4_GPUs-PyTorch-Post-Merge-3.accuracy.test_llm_api_pytorch.TestQwen3_235B_A22B` · `test_nvfp4_4gpus[latency_moe_trtllm_eagle3]` | #3051: 266.936s passed<br>#3053: 270.069s passed | 374.754s passed | +106.252s (+39.6%) |
| `GB300-4_GPUs-PyTorch-Post-Merge-3.accuracy.test_llm_api_pytorch.TestDeepSeekV3Lite` · `test_nvfp4_4gpus[moe_backend=TRTLLM-mtp_nextn=2-tp4-fp8kv=True-attention_dp=True-cuda_graph=True-overlap_scheduler=True-low_precision_combine=False-torch_compile=False]` | #3051: 510.800s passed<br>#3053: 514.782s passed<br>#3055: 561.088s passed [stage not in valid baseline]<br>#3057: 599.116s passed | 611.264s passed | +96.482s (+18.7%) |
| `GB300-4_GPUs-PyTorch-Post-Merge-3.accuracy.test_llm_api_pytorch.TestQwen3_5_397B_A17B` · `test_nvfp4_mtp3_gdn_replay_tep4` | #3051: 1156.624s passed<br>#3053: 1068.733s passed<br>#3055: 1246.029s passed [stage not in valid baseline]<br>#3057: 1328.897s passed | 908.504s passed | -248.120s (-21.5%) |

### #69221 H200 Post-Merge-1

| Case | Off | On | Difference |
|---|---|---|---|
| `DGX_H200-4_GPUs-PyTorch-Post-Merge-1.accuracy.test_llm_api_pytorch.TestDeepSeekV3Lite` · `test_fp8_block_scales_4gpus[tp2pp2-mtp_nextn=2-fp8kv=False-attention_dp=False-cuda_graph=False-overlap_scheduler=True-torch_compile=False-sampler_async_worker=True]` | #3051: 63.806s passed<br>#3053: 66.152s passed<br>#3055: 64.437s passed<br>#3057: 63.512s passed | 129.199s passed | +65.078s (+101.5%) |
| `DGX_H200-4_GPUs-PyTorch-Post-Merge-1.accuracy.test_llm_api_pytorch.TestDeepSeekV3Lite` · `test_fp8_block_scales_4gpus[tp2pp2-mtp_nextn=2-fp8kv=False-attention_dp=False-cuda_graph=False-overlap_scheduler=True-torch_compile=False-sampler_async_worker=False]` | #3051: 63.188s passed<br>#3053: 65.934s passed<br>#3055: 63.972s passed<br>#3057: 62.833s passed | 128.374s passed | +64.794s (+101.9%) |
| `DGX_H200-4_GPUs-PyTorch-Post-Merge-1.accuracy.test_llm_api_pytorch.TestDeepSeekV3Lite` · `test_fp8_block_scales_4gpus[tp2pp2-mtp_nextn=2-fp8kv=False-attention_dp=False-cuda_graph=False-overlap_scheduler=False-torch_compile=False-sampler_async_worker=False]` | #3051: 65.678s passed<br>#3053: 68.178s passed<br>#3055: 66.007s passed<br>#3057: 65.423s passed | 128.110s passed | +62.268s (+94.6%) |
| `DGX_H200-4_GPUs-PyTorch-Post-Merge-1.accuracy.test_llm_api_pytorch.TestDeepSeekV3Lite` · `test_bfloat16_4gpus[pp4-mtp_nextn=0-attention_dp=True-cuda_graph=True-overlap_scheduler=True-torch_compile=False]` | #3051: 147.986s passed<br>#3053: 149.152s passed<br>#3055: 150.784s passed<br>#3057: 149.625s passed | 96.405s passed | -52.983s (-35.5%) |

### #70387 GB300 pre-merge vs #70097

| Case | Off | On | Difference |
|---|---|---|---|
| `GB300-4_GPUs-PyTorch-1.accuracy.test_disaggregated_serving.TestGLM52NVFP4` · `test_kv_cache_offload[kv_dtype=nvfp4]` | 1010.171 s / passed | 1265.256 s / passed | +255.085 s (+25.3%) |
| `GB300-4_GPUs-PyTorch-1.accuracy.test_llm_api_pytorch.TestDeepSeekV41` · `test_gsm8k_dspark[nvfp4-tp4]` | 382.837 s / passed | 456.900 s / passed | +74.063 s (+19.3%) |
| `GB300-4_GPUs-PyTorch-1.accuracy.test_llm_api_pytorch.TestQwen3_8B` · `test_fp8_block_scales[latency]` | 96.295 s / passed | 115.032 s / passed | +18.737 s (+19.5%) |
| `GB300-4_GPUs-PyTorch-1.accuracy.modeling_v2.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.test_accuracy.TestModelingV2Qwen38FlashNextNvfp4Sm103Default` · `test_mtp3_gsm8k_and_long_context` | 639.250 s / passed | 509.030 s / passed | -130.220 s (-20.4%) |
