<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CBTS two-week single-commit and rolling-window skip-rate analysis

## Executive summary

The current CBTS coverage selector can calculate case-level skip rates for a
single commit and for cumulative rolling commit windows. A reusable driver was
added in `jenkins/scripts/cbts/tools/analyze_skip_rate.py` to perform that
replay and emit both Markdown summaries and per-window JSON records.

The controlled two-week replay used 438 first-parent commits from
`upstream/main`, a fixed build-2944 coverage database, and the proposed gap
policy:

- Include `tensorrt_llm/**/*.py` changes in coverage selection.
- Treat native sources and headers under `cpp/` as fallback blockers because
  the Python touch database cannot map them.
- Ignore every other gap-only file.

With the **current conservative Python selector gates**, a single-commit gap
falls back in 164 of 438 cases (37.4%). A 20-commit gap falls back in all 419
rolling windows (100%). Among single commits that actually contain relevant
Python or native changes, 164 of 188 (87.2%) fall back.

The single-commit fallback rate is much higher than the native-only 9.8%
estimate because 84 commits are declined for import-executed Python changes
and 37 are declined for zero-touch Python files. For long windows, native code
becomes dominant: 360 of 419 20-commit windows (85.9%) contain native changes.

The 24 single commits accepted by coverage selection skip only 26.1% of the
1,014-case pre-merge universe on average. Once fallback windows are included,
relevant single-commit windows skip just 3.3% on average. Therefore, expanding
the diff through the current selector is not yet an effective case-reduction
strategy; safe bounds for import-executed and zero-touch Python changes are
needed first.

## Replay definition

The source interval is 2026-08-20 10:38:14 through 2026-09-03 10:38:14
(UTC+08:00). The repository snapshot is:

```text
2e522c5f30b12787473083fb4221b3415dd97148
```

For a gap of `n` and a rolling endpoint at commit `head`, the tool evaluates:

```text
diff(head~n, head)
```

It reads changed Python source from the actual endpoint commit rather than
from the latest checkout, so historical line numbers and qualified names are
interpreted against the correct source version.

The coverage input is `/private/tmp/cbts-relaxed-dbs/2944_pair.sqlite`, a
schema-v3 compact database associated with coverage commit:

```text
fcc84548ee6000530222600b33c4e733eaaf4de1
```

That coverage commit is position 390 of the 438-commit sample and is 48
first-parent commits behind the replay endpoint. The database itself does not
embed build or commit metadata, so the build/commit association comes from the
artifact collection record used in the preceding gap analysis.

This is a **controlled replay against one database and one test-list
snapshot**, not a historical reconstruction of the exact database and test
universe available at every commit.

## Skip-rate definitions

Each window receives one mutually exclusive status:

- `coverage`: coverage mapping accepted the Python diff; selected cases are
  counted from the narrowed stage test lists.
- `ignored`: the window contains neither relevant core Python nor native
  changes; it adds zero cases and therefore has a 100% **gap-only** skip rate.
- `fallback`: native code is present or the coverage selector declines; full
  CI runs and the skip rate is 0%.

The report presents three different averages:

- **All-window mean skip** includes ignored windows at 100% and fallback at
  0%. It measures the incremental selection caused by the gap itself.
- **Relevant mean skip** excludes ignored windows and includes fallback at 0%.
  It answers how much is skipped when the gap contains code that must be
  evaluated.
- **Coverage-hit mean/median** includes only windows accepted by coverage
  mapping. It measures selector quality after all gates pass.

An ignored window's 100% gap-only skip does not mean the complete `/bot run`
would run zero tests. PR-local Tier-1/Tier-2 selections still have to be
unioned with the expanded-gap result.

## Results by rolling commit gap

| Gap | Windows | Coverage | Ignored | Fallback | Fallback / all | Fallback / relevant | All-window mean skip | Relevant mean skip | Hit-only mean | Hit-only median |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 438 | 24 | 250 | 164 | 37.4% | 87.2% | 58.5% | 3.3% | 26.1% | 29.9% |
| 2 | 437 | 29 | 148 | 260 | 59.5% | 90.0% | 35.6% | 2.6% | 26.3% | 30.0% |
| 3 | 436 | 25 | 92 | 319 | 73.2% | 92.7% | 22.6% | 1.9% | 25.5% | 30.0% |
| 5 | 434 | 10 | 38 | 386 | 88.9% | 97.5% | 9.3% | 0.6% | 22.0% | 23.2% |
| 10 | 429 | 3 | 2 | 424 | 98.8% | 99.3% | 0.6% | 0.1% | 17.8% | 11.6% |
| 15 | 424 | 0 | 0 | 424 | 100.0% | 100.0% | 0.0% | 0.0% | N/A | N/A |
| 20 | 419 | 0 | 0 | 419 | 100.0% | 100.0% | 0.0% | 0.0% | N/A | N/A |

The decrease in all-window mean skip as the gap grows is not caused by
coverage becoming more precise. It occurs because ignored windows disappear
and fallback approaches 100%. By gap 15, no window reaches coverage selection.

## Exclusive fallback breakdown

The selector stops at the first decisive blocker. The categories below are
therefore mutually exclusive and sum to the fallback count for each row.
Native has precedence: a window containing both native and problematic Python
changes is classified as native.

Each cell is `count / all windows`, followed by the category's share of all
fallback windows in parentheses.

| Gap | Native | Import-executed Python | Zero-touch Python |
|---:|---:|---:|---:|
| 1 | 43/438 = 9.8% (26.2%) | 84/438 = 19.2% (51.2%) | 37/438 = 8.4% (22.6%) |
| 2 | 79/437 = 18.1% (30.4%) | 119/437 = 27.2% (45.8%) | 62/437 = 14.2% (23.8%) |
| 3 | 113/436 = 25.9% (35.4%) | 123/436 = 28.2% (38.6%) | 83/436 = 19.0% (26.0%) |
| 5 | 162/434 = 37.3% (42.0%) | 111/434 = 25.6% (28.8%) | 113/434 = 26.0% (29.3%) |
| 10 | 253/429 = 59.0% (59.7%) | 60/429 = 14.0% (14.2%) | 111/429 = 25.9% (26.2%) |
| 15 | 320/424 = 75.5% (75.5%) | 36/424 = 8.5% (8.5%) | 68/424 = 16.0% (16.0%) |
| 20 | 360/419 = 85.9% (85.9%) | 21/419 = 5.0% (5.0%) | 38/419 = 9.1% (9.1%) |

No other decline category appeared after endpoint-version source lookup was
enabled: there were no unusable-diff, unparsable-source, or closure-bound
declines in this sample.

## Single-commit detail

For gap 1:

| Metric | Result |
|:---|---:|
| All commits | 438 |
| No relevant gap change | 250 (57.1%) |
| Coverage accepted | 24 (5.5% of all; 12.8% of relevant) |
| Fallback | 164 (37.4% of all; 87.2% of relevant) |
| Coverage-hit minimum skip | 10.0% (913/1,014 cases selected) |
| Coverage-hit maximum skip | 31.1% (699/1,014 cases selected) |
| Coverage-hit mean skip | 26.1% (749.2 cases selected on average) |
| Coverage-hit median skip | 29.9% |
| Relevant-window mean skip | 3.3% |

The single-commit fallback composition is:

- Import-executed Python: 84 commits, 51.2% of fallbacks.
- Native code: 43 commits, 26.2% of fallbacks.
- Zero-touch Python: 37 commits, 22.6% of fallbacks.

The import-executed declines contain 73 distinct first reported reasons. The
most repeated were changes in `py_executor.py::PyExecutor` (four commits) and
`model_engine.py::<module>` (three commits). This is broad churn rather than
one anomalous path.

The zero-touch declines contain 30 distinct first reported files. Of the 37
single-commit declines, 19 reference a path absent from the coverage commit
snapshot and 18 reference a path that exists at the coverage commit but has no
touch rows. Therefore, source-version mismatch and incomplete instrumentation
both materially contribute to zero-touch. A fixed-DB replay cannot attribute
every absent path to production staleness because some early-window files may
have been moved or removed before build 2944 was collected.

## Comparison with native-only fallback

The earlier file-composition analysis intentionally ignored conservative
Python decline gates and counted only native code as blocking under this
filtered policy. Its fallback probabilities were 9.8% for gap 1 and 85.9% for
gap 20.

This replay answers a different question: what happens if the filtered diff is
fed to the **current selector unchanged**. The comparison is:

| Gap | Native-only fallback | Current-selector fallback | Increase from Python gates |
|---:|---:|---:|---:|
| 1 | 9.8% | 37.4% | +27.6 pp |
| 2 | 18.1% | 59.5% | +41.4 pp |
| 3 | 25.9% | 73.2% | +47.3 pp |
| 5 | 37.3% | 88.9% | +51.6 pp |
| 10 | 59.0% | 98.8% | +39.8 pp |
| 15 | 75.5% | 100.0% | +24.5 pp |
| 20 | 85.9% | 100.0% | +14.1 pp |

At short gaps, conservative Python gates are the main issue. At long gaps,
native changes alone already make fallback highly likely.

## Recommendation

Do not enable `coverage_commit..pr_head` selection through the current
selector unchanged. The experiment shows almost no case-saving benefit once a
window contains relevant code.

The next useful experiments are:

1. Replace the binary import-executed decline with a conservative executable
   bound, such as the file's touch set plus known importers, and remeasure the
   84 single-commit declines.
2. Split zero-touch into files created after the DB commit and files present
   but uninstrumented at collection time. The former needs fresher coverage;
   the latter needs a safe file/importer bound or explicit always-run policy.
3. Keep native changes as a separate full-CI signal until native test-impact
   data exists. A 20-commit coverage cadence still gives an 85.9% native-only
   fallback probability, so cadence improvement or native mapping is required
   for long gaps.
4. After those policies are defined, replay the union of PR-local and gap
   selections using the actual coverage DB available to each `/bot run`.

## Reproduction

```bash
python3 jenkins/scripts/cbts/tools/analyze_skip_rate.py \
  --repo-root /path/to/upstream-main-snapshot \
  --ref HEAD \
  --since '2026-08-20T10:38:14+08:00' \
  --until '2026-09-03T10:38:14+08:00' \
  --gaps 1,2,3,5,10,15,20 \
  --coverage-db /path/to/2944_pair.sqlite \
  --output-json /tmp/cbts-two-week-skip-rate.json \
  --output-markdown /tmp/cbts-two-week-skip-rate.md
```

The JSON output contains every commit/window, endpoint SHA, changed-file
counts, exact relevant Python/native paths, status, exclusive fallback
category, selected/total case counts, and skip rate. It is the appropriate
input for deeper per-commit inspection or dashboard prototyping; the replay
artifact is intentionally not checked into the repository.
