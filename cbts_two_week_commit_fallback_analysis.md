<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CBTS two-week commit-gap fallback analysis

## Executive summary

This analysis estimates how often CBTS coverage-based selection would fall back if it used the expanded
`coverage_commit..pr_head` diff instead of only the PR-local `pr_base..pr_head` diff.

From 2026-08-20 10:38:14 to 2026-09-03 10:38:14 (UTC+08:00), `upstream/main` received 438 first-parent
commits. Of those commits, 156 (35.6%) introduced at least one file that remains an unhandled, non-core
residual after the static Tier 1 rules run. Such a file causes the current Tier 2 preflight to decline.

The blocker commits are approximately uniform across the commit sequence rather than concentrated in a
small part of the two-week period. Consequently, every one of the 419 possible consecutive 20-commit
windows contained at least one blocker; the median window contained seven blocker commits.

The resulting estimates are:

- If the coverage report is exactly 20 commits behind the PR base, the observed fallback probability is
  100%.
- If a new coverage report is published every 20 commits and `/bot run` is uniformly distributed between
  reports, the estimated average fallback probability is approximately 85%.

This probability is conditional on the original PR-local diff already being eligible for coverage-based
selection. It measures only additional fallback caused by files in the coverage-to-PR gap.

## Scope and methodology

The source history is `upstream/main` at commit
`2e522c5f30b12787473083fb4221b3415dd97148`. The sample contains 438 first-parent commits, of which 422
have a PR number in the commit subject and 16 are direct or automated commits without that suffix.

For every commit, the analysis:

1. Computes the first-parent changed-file set.
2. Runs the current CBTS static rules to identify files handled by Tier 1.
3. Treats Tier 1 force-fallback signals as audit-only, matching the relaxed experiment.
4. Treats residual `tensorrt_llm/**/*.py` files as Tier-2-eligible, not as unhandled files.
5. Classifies every remaining residual file as a Tier 2 non-core blocker.

A **fallback commit** is a commit containing at least one such blocker. This is a commit-level binary
classification: a commit with one blocker and a commit with 100 blockers each count once.

The 20-commit result was also checked against each cumulative endpoint diff. This removes files whose
changes were completely reverted within the window. All 419 endpoint diffs still contained blockers.

This analysis intentionally does not count other reasons Tier 2 may decline, such as missing coverage
data, an unbounded import-closure change, an unusable patch, or artifact freshness. It isolates fallback
caused by the expanded diff's file composition.

## Two-week commit totals

| Metric | Commits | Share |
|---|---:|---:|
| All first-parent commits | 438 | 100.0% |
| Fallback commits | 156 | 35.6% |
| Commits without a non-core blocker | 282 | 64.4% |

## Fallback reason by blocker file type

The table below reports **commit incidence**: the number and percentage of the 156 fallback commits that
contain at least one blocker of that type. A commit may contain multiple blocker types, so these rows are
not mutually exclusive and the percentages intentionally sum to more than 100%.

Commit incidence is more representative than raw blocker-file counts. For example, a single bulk artifact
commit can contain thousands of `.zst` files but still causes only one `/bot run` fallback event.

| Blocker file type | Fallback commits containing type | Share of 156 fallback commits |
|---|---:|---:|
| `.groovy` | 44 | 28.2% |
| `.cpp` / `.cc` / `.cxx` / `.c` | 33 | 21.2% |
| `.json` | 31 | 19.9% |
| `.h` / `.hpp` / `.hh` / `.hxx` | 30 | 19.2% |
| `.txt` | 27 | 17.3% |
| `.cu` | 21 | 13.5% |
| `.yml` / `.yaml` | 20 | 12.8% |
| `.toml` | 18 | 11.5% |
| `.py` under `tests/` | 17 | 10.9% |
| Shell or job script | 15 | 9.6% |
| `.lock` | 14 | 9.0% |
| No extension or repository config | 12 | 7.7% |
| `.py` under `examples/` | 9 | 5.8% |
| Other `.py` | 7 | 4.5% |
| `.cuh` | 5 | 3.2% |
| `.py` under `jenkins/` | 4 | 2.6% |
| `.py` under `scripts/` | 4 | 2.6% |
| `.zst` | 4 | 2.6% |
| `.rst` | 3 | 1.9% |
| `.pyi` | 3 | 1.9% |
| `.multi` | 3 | 1.9% |
| `.properties` | 2 | 1.3% |
| `.png` | 1 | 0.6% |
| `.bolt` | 1 | 0.6% |
| `.patch` | 1 | 0.6% |
| `.a` | 1 | 0.6% |

### Groovy and native-code composition

The following categories are mutually exclusive and therefore sum to all 156 fallback commits. Native
code includes C/C++, CUDA, and header suffixes.

| Exclusive category | Fallback commits | Share of fallback commits |
|---|---:|---:|
| Groovy blocker, no native-code blocker | 42 | 26.9% |
| Native-code blocker, no Groovy blocker | 41 | 26.3% |
| Both Groovy and native-code blockers | 2 | 1.3% |
| Neither Groovy nor native-code blocker | 71 | 45.5% |

In total:

- 44 fallback commits (28.2%) contain Groovy changes.
- 43 fallback commits (27.6%) contain native-code changes.
- 85 fallback commits (54.5%) contain Groovy or native-code changes.
- The remaining 71 fallback commits (45.5%) are caused entirely by other file families, so handling only
  Groovy and native code would not make the expanded diff broadly usable.

### Most frequently recurring blocker paths

| Blocker path | Fallback commits | Share of fallback commits |
|---|---:|---:|
| `jenkins/L0_MergeRequest.groovy` | 22 | 14.1% |
| `jenkins/L0_Test.groovy` | 18 | 11.5% |
| `security_scanning/metadata.json` | 15 | 9.6% |
| `security_scanning/poetry.lock` | 12 | 7.7% |
| `security_scanning/examples/ray_orchestrator/poetry.lock` | 12 | 7.7% |
| `tensorrt_llm/usage/llm_args_golden_manifest.json` | 12 | 7.7% |
| `security_scanning/examples/auto_deploy/poetry.lock` | 11 | 7.1% |
| `security_scanning/examples/trtllm-eval/poetry.lock` | 11 | 7.1% |
| `security_scanning/triton_backend/poetry.lock` | 9 | 5.8% |
| `.github/CODEOWNERS` | 9 | 5.8% |

These paths show that fallback is not driven only by compiled code. CI definitions, dependency metadata,
security-scanning environments, ownership configuration, and generated manifests are frequent sources.

## Distribution across the commit sequence

The 438 commits were divided into four nearly equal consecutive buckets:

| Commit positions | Total commits | Fallback commits | Fallback rate |
|---|---:|---:|---:|
| 1–109 | 109 | 44 | 40.4% |
| 110–219 | 110 | 35 | 31.8% |
| 220–328 | 109 | 41 | 37.6% |
| 329–438 | 110 | 36 | 32.7% |

The binary fallback sequence has a lag-1 autocorrelation of `0.017`, which is very close to zero. A
four-bucket chi-square test gives `p=0.50`, so this sample provides no evidence that fallback commits are
unevenly distributed across the commit sequence.

The longest run without any fallback commit is 15 commits. There is no blocker-free 20-commit interval in
the sample.

## Fallback probability by commit gap

For each gap length, the table measures the fraction of rolling windows containing at least one fallback
commit:

| Commit gap | Windows with a blocker | Total windows | Empirical fallback probability |
|---:|---:|---:|---:|
| 1 | 156 | 438 | 35.6% |
| 2 | 254 | 437 | 58.1% |
| 3 | 317 | 436 | 72.7% |
| 5 | 381 | 434 | 87.8% |
| 10 | 422 | 429 | 98.4% |
| 15 | 423 | 424 | 99.8% |
| 20 | 419 | 419 | 100.0% |

### Blocker-type breakdown for each commit gap

Each cell below is `windows containing this blocker type / all fallback windows for this gap`, followed by
the percentage. For example, 44 of the 156 fallback windows at gap 1 contain a Groovy blocker, so Groovy
appears in 28.2% of gap-1 fallbacks. At gap 20, 351 of all 419 fallback windows contain a Groovy blocker,
or 83.8%.

This is a multi-label breakdown. A window containing both Groovy and C++ is counted in both rows, because
either type is independently sufficient to make Tier 2 decline. Consequently, percentages within a column
do not sum to 100%. `native code (combined)` is a roll-up across `.cpp/.cc`, `.h/.hpp`, `.cu`, and `.cuh`;
it should not be added to its component rows.

| Blocker type | Gap 1 | Gap 2 | Gap 3 | Gap 5 | Gap 10 | Gap 15 | Gap 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `native code (combined)` | 43/156 (27.6%) | 79/254 (31.1%) | 113/317 (35.6%) | 162/381 (42.5%) | 253/422 (60.0%) | 320/423 (75.7%) | 360/419 (85.9%) |
| `.groovy` | 44/156 (28.2%) | 82/254 (32.3%) | 114/317 (36.0%) | 169/381 (44.4%) | 271/422 (64.2%) | 328/423 (77.5%) | 351/419 (83.8%) |
| `.cpp/.cc` | 33/156 (21.2%) | 61/254 (24.0%) | 88/317 (27.8%) | 132/381 (34.6%) | 227/422 (53.8%) | 298/423 (70.4%) | 345/419 (82.3%) |
| `.json` | 31/156 (19.9%) | 59/254 (23.2%) | 85/317 (26.8%) | 131/381 (34.4%) | 227/422 (53.8%) | 302/423 (71.4%) | 358/419 (85.4%) |
| `.h/.hpp` | 30/156 (19.2%) | 56/254 (22.0%) | 81/317 (25.6%) | 120/381 (31.5%) | 199/422 (47.2%) | 261/423 (61.7%) | 304/419 (72.6%) |
| `.txt` | 27/156 (17.3%) | 52/254 (20.5%) | 74/317 (23.3%) | 114/381 (29.9%) | 200/422 (47.4%) | 260/423 (61.5%) | 300/419 (71.6%) |
| `.cu` | 21/156 (13.5%) | 41/254 (16.1%) | 60/317 (18.9%) | 93/381 (24.4%) | 165/422 (39.1%) | 227/423 (53.7%) | 272/419 (64.9%) |
| `.yml/.yaml` | 20/156 (12.8%) | 38/254 (15.0%) | 55/317 (17.4%) | 88/381 (23.1%) | 151/422 (35.8%) | 202/423 (47.8%) | 244/419 (58.2%) |
| `.toml` | 18/156 (11.5%) | 35/254 (13.8%) | 51/317 (16.1%) | 82/381 (21.5%) | 151/422 (35.8%) | 208/423 (49.2%) | 256/419 (61.1%) |
| `.py (tests)` | 17/156 (10.9%) | 34/254 (13.4%) | 50/317 (15.8%) | 82/381 (21.5%) | 155/422 (36.7%) | 211/423 (49.9%) | 247/419 (58.9%) |
| `shell/job script` | 15/156 (9.6%) | 29/254 (11.4%) | 42/317 (13.2%) | 66/381 (17.3%) | 121/422 (28.7%) | 171/423 (40.4%) | 213/419 (50.8%) |
| `.lock` | 14/156 (9.0%) | 28/254 (11.0%) | 42/317 (13.2%) | 70/381 (18.4%) | 131/422 (31.0%) | 177/423 (41.8%) | 216/419 (51.6%) |
| `no extension / repo config` | 12/156 (7.7%) | 24/254 (9.4%) | 36/317 (11.4%) | 60/381 (15.7%) | 116/422 (27.5%) | 166/423 (39.2%) | 206/419 (49.2%) |
| `.py (examples)` | 9/156 (5.8%) | 17/254 (6.7%) | 25/317 (7.9%) | 41/381 (10.8%) | 75/422 (17.8%) | 105/423 (24.8%) | 135/419 (32.2%) |
| `.py (other)` | 7/156 (4.5%) | 14/254 (5.5%) | 21/317 (6.6%) | 34/381 (8.9%) | 64/422 (15.2%) | 94/423 (22.2%) | 117/419 (27.9%) |
| `.cuh` | 5/156 (3.2%) | 10/254 (3.9%) | 15/317 (4.7%) | 25/381 (6.6%) | 47/422 (11.1%) | 67/423 (15.8%) | 87/419 (20.8%) |
| `.py (jenkins)` | 4/156 (2.6%) | 8/254 (3.1%) | 12/317 (3.8%) | 20/381 (5.2%) | 37/422 (8.8%) | 52/423 (12.3%) | 67/419 (16.0%) |
| `.py (scripts)` | 4/156 (2.6%) | 8/254 (3.1%) | 12/317 (3.8%) | 20/381 (5.2%) | 40/422 (9.5%) | 60/423 (14.2%) | 80/419 (19.1%) |
| `.zst` | 4/156 (2.6%) | 8/254 (3.1%) | 12/317 (3.8%) | 20/381 (5.2%) | 40/422 (9.5%) | 60/423 (14.2%) | 80/419 (19.1%) |
| `.pyi` | 3/156 (1.9%) | 6/254 (2.4%) | 9/317 (2.8%) | 15/381 (3.9%) | 30/422 (7.1%) | 45/423 (10.6%) | 60/419 (14.3%) |
| `.rst` | 3/156 (1.9%) | 6/254 (2.4%) | 9/317 (2.8%) | 15/381 (3.9%) | 30/422 (7.1%) | 45/423 (10.6%) | 60/419 (14.3%) |
| `.multi` | 3/156 (1.9%) | 6/254 (2.4%) | 9/317 (2.8%) | 15/381 (3.9%) | 30/422 (7.1%) | 45/423 (10.6%) | 60/419 (14.3%) |
| `.properties` | 2/156 (1.3%) | 4/254 (1.6%) | 6/317 (1.9%) | 10/381 (2.6%) | 20/422 (4.7%) | 30/423 (7.1%) | 40/419 (9.5%) |
| `.png` | 1/156 (0.6%) | 2/254 (0.8%) | 3/317 (0.9%) | 5/381 (1.3%) | 10/422 (2.4%) | 15/423 (3.5%) | 15/419 (3.6%) |
| `.bolt` | 1/156 (0.6%) | 2/254 (0.8%) | 3/317 (0.9%) | 5/381 (1.3%) | 10/422 (2.4%) | 15/423 (3.5%) | 20/419 (4.8%) |
| `.patch` | 1/156 (0.6%) | 2/254 (0.8%) | 3/317 (0.9%) | 5/381 (1.3%) | 10/422 (2.4%) | 15/423 (3.5%) | 20/419 (4.8%) |
| `.a` | 1/156 (0.6%) | 2/254 (0.8%) | 3/317 (0.9%) | 5/381 (1.3%) | 10/422 (2.4%) | 15/423 (3.5%) | 20/419 (4.8%) |

The dominant types broaden as the gap grows. At gap 1, Groovy appears in 28.2% of fallback windows and
native code in 27.6%. By gap 20, native code appears in 85.9%, JSON in 85.4%, Groovy in 83.8%, and C/C++
source alone in 82.3% of all fallback windows. The 100% overall fallback probability therefore does not
come from one dominant file family; most long windows contain several independently blocking families.

For the 20-commit windows:

- Minimum fallback commits per window: 2
- Median fallback commits per window: 7
- Mean fallback commits per window: 7.02
- Maximum fallback commits per window: 12
- Cumulative endpoint diffs that still contain a blocker after accounting for reverts: 419/419

If fallback commits were independent with the observed single-commit probability
`p = 156 / 438 = 0.356`, the probability for a fixed gap of `n` commits would be:

```text
P(fallback | n commits) = 1 - (1 - p)^n
```

For `n=20`, this gives 99.985%, consistent with the empirical 100% result.

## Estimate for a coverage report every 20 commits

There are two different interpretations of a 20-commit coverage cadence:

1. **Coverage is already 20 commits old.** The relevant probability is the 20-commit window result:
   approximately 100%.
2. **Coverage refreshes every 20 commits and a PR run arrives at a random point in the cycle.** The gap age
   is approximately uniform from 0 through 19 commits. Simulating all 20 possible refresh alignments gives:
   - Mean fallback probability: 84.8%
   - Minimum across alignments: 80.6%
   - Maximum across alignments: 89.7%

The independent approximation for a uniformly random age from 0 through 19 is 86.0%, close to the
empirical 84.8% result. A practical planning estimate is therefore **about 85% fallback caused by the main
commit gap alone**.

The estimate assumes that `/bot run` arrivals are approximately uniform by commit age and that a newly
generated coverage report becomes available immediately. Artifact publication delay or a tendency for runs
to use older reports would increase the probability toward 100%.

## Scenario: expand only core Python and native-code changes

A less conservative policy can ignore gap-only changes that do not affect the code executed by the Python
coverage report. Under this proposal, the expanded portion contributes only:

- `tensorrt_llm/**/*.py`
- Native source and headers under `cpp/`: `.cpp`, `.cc`, `.cxx`, `.c`, `.h`, `.hpp`, `.hh`, `.hxx`, `.cu`,
  and `.cuh`

Gap-only Groovy, JSON, YAML, TOML, lock, test, example, CI, and repository-configuration changes are not
passed to coverage mapping and do not cause fallback. General `tensorrt_llm/**/*.py` remains Tier-2
eligible. With the current Python-only coverage mapping, native changes are therefore the only remaining
file-type fallback cause.

All 43 native-blocker commits in this two-week sample changed files under `cpp/`; no same-suffix files from
other top-level directories are included in these results.

| Commit gap | Native-blocker windows | Total windows | Filtered-gap fallback probability | Raw expanded-diff probability |
|---:|---:|---:|---:|---:|
| 1 | 43 | 438 | 9.8% | 35.6% |
| 2 | 79 | 437 | 18.1% | 58.1% |
| 3 | 113 | 436 | 25.9% | 72.7% |
| 5 | 162 | 434 | 37.3% | 87.8% |
| 10 | 253 | 429 | 59.0% | 98.4% |
| 15 | 320 | 424 | 75.5% | 99.8% |
| 20 | 360 | 419 | 85.9% | 100.0% |

These values were recomputed from each cumulative endpoint diff, rather than assuming that any native
commit inside a window necessarily remains visible at the endpoint. This accounts for files whose changes
are fully reverted within the window.

For a coverage report published every 20 commits:

- If the report is already exactly 20 commits behind, the filtered-gap fallback probability is 85.9%.
- If `/bot run` is uniformly distributed over gap ages 0 through 19, the empirical average is 51.1%.
- Across the 20 possible refresh alignments, the empirical average ranges from 40.4% to 58.4%.
- The independent approximation for a uniformly random age is 55.5%.

The practical headline estimate changes from approximately 85% under the raw expanded diff to
approximately **51% under the Python-plus-native filtered gap**.

The filter must apply only to the historical main-branch gap. The PR's own diff must remain complete:

```text
effective_files =
    all files in pr_base..pr_head
    UNION
    relevant Python/native files in coverage_commit..pr_base
```

Filtering the full `coverage_commit..pr_head` result without preserving all PR-local files would incorrectly
ignore a PR's own Groovy, test-list, dependency, or configuration changes.

If a future coverage artifact provides a reliable native-code-to-test mapping, native changes could also
participate in selection without forcing fallback. In that future model, file-type fallback for the filtered
gap would be zero by construction. The remaining probability would come from mapping-level declines such as
missing data, unusable diffs, or unbounded dependency closure; this commit-file analysis cannot estimate
those outcomes.

## Interpretation and recommendation

Using the raw `coverage_commit..pr_head` diff is safe in the conservative sense, but it is unlikely to be
useful with the current preflight contract. At a 20-commit report cadence, approximately five out of six
otherwise eligible PR runs would fall back because the main gap contains a non-core file.

Increasing coverage frequency alone has diminishing practical value: the observed fallback probability is
already 58.1% at a two-commit gap and 87.8% at a five-commit gap.

Filtering the gap to core Python and native code substantially improves those rates, but a 20-commit cadence
still produces approximately 51% average fallback while native changes remain unmapped.

A more promising design is to keep the two concepts separate:

- Use `pr_base..pr_head` to classify the PR's own change and determine whether the PR is eligible for Tier 2.
- Analyze `coverage_commit..pr_base` as a coverage-compatibility or invalidation gap, with explicit handling
  by file family.
- Invalidate coverage only when a gap change can make the stored Python-to-test mapping unsafe, instead of
  treating every non-core residual as equivalent.
- Refresh or incrementally update coverage when native code, CI topology, test discovery, or dependency
  changes genuinely invalidate the artifact.

That design still exposes every changed file for safety review, while avoiding an approximately 85% blanket
fallback rate caused by unrelated main-branch activity.

## Limitations

- The sample covers one recent two-week period. Repository activity and file-type mix may change over time.
- File-type incidence rows overlap; they answer “how many fallback commits contain this type,” not “what
  percentage of fallback is exclusively caused by this type.”
- A commit is classified with the current relaxed policy. Tier 1 force-fallback signals are not enforced.
- General `tensorrt_llm/**/*.py` files are explicitly eligible and never counted as unhandled blockers.
- The estimate is conditional on a PR whose original `pr_base..pr_head` diff can already use Tier 2.
- Other coverage-decline causes are excluded, so the result is not an upper bound on total production
  fallback. It estimates only fallback introduced by expanding the diff.
