# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test selector: given a set of changed files, select which tests to run.

Core logic:
  1. Match changed files against impact rules
  2. For each matched rule, collect affected test entries based on tier:
     - Tier.CORE: Representative Coverage Set (all arches × all features)
     - Tier.DEFAULT_ON: tests with matching config class + Representative Set
     - Tier.OPT_IN: only tests that use the feature (config class or feature tag)
     - Tier.MODEL: all tests for that architecture
     - Tier.TEST: tests from that test file
     - Tier.IGNORE: skip
  3. Union all results
"""

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .impact_rules import (
    FEATURE_TO_CONFIG,
    IMPACT_RULES,
    REPRESENTATIVE_COVERAGE_SET,
    SPECULATIVE_FEATURES,
    ImpactRule,
    Tier,
    match_rule,
)

from .parser import TestEntry


@dataclass
class SelectionResult:
    """Result of test selection with explanations."""

    selected_tests: dict[str, TestEntry] = field(default_factory=dict)
    # Explanation: test_id → list of reasons why it was selected
    reasons: dict[str, list[str]] = field(default_factory=dict)
    # Which rules were triggered
    triggered_rules: list[tuple[str, ImpactRule]] = field(default_factory=list)
    # Unmatched files (no rule matched)
    unmatched_files: list[str] = field(default_factory=list)
    # Rules skipped due to additive-only detection
    skipped_rules: list[tuple[str, ImpactRule, str]] = field(
        default_factory=list)

    def add(self, test_id: str, entry: TestEntry, reason: str):
        self.selected_tests[test_id] = entry
        self.reasons.setdefault(test_id, []).append(reason)

    def add_skipped(self, changed_file: str, rule: ImpactRule, reason: str):
        self.skipped_rules.append((changed_file, rule, reason))


def get_changed_files(base_ref: str = "main",
                      repo_root: str = ".") -> list[str]:
    """Get list of changed files relative to base_ref using git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=True,
        )
        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        return files
    except subprocess.CalledProcessError:
        # Fallback: diff against HEAD~1
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=True,
        )
        return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]


def _is_additive_only(changed_file: str, base_ref: str) -> bool:
    """Check if a file's diff is safe to skip for test selection.

    A diff is safe if ALL hunks are one of:
      1. Expansion-only: every deleted line is a substring of a corresponding
         added line (e.g. `"a" | "b"` → `"a" | "b" | "c"`)
      2. Safe pure additions: new function/class definitions, imports, or
         comments — NOT new code inside an existing function body.

    Adding code inside an existing function body is NOT safe even if it's a
    pure addition, because it changes the function's runtime behavior (e.g.
    adding an `if` branch that sets a default value can cause OOM).
    """
    if not base_ref:
        return False

    try:
        result = subprocess.run(
            ["git", "diff", "-U0", f"{base_ref}...HEAD", "--", changed_file],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return False

    diff_output = result.stdout.strip()
    if not diff_output:
        return False

    # Parse hunks: collect removed (-) and added (+) lines per hunk
    # A hunk starts with @@ and contains - and + lines
    removed_lines: list[str] = []
    added_lines: list[str] = []
    has_additions = False

    for line in diff_output.split('\n'):
        if line.startswith('@@'):
            # Process previous hunk
            if not _is_safe_hunk(removed_lines, added_lines):
                return False
            if added_lines:
                has_additions = True
            removed_lines = []
            added_lines = []
        elif line.startswith('-') and not line.startswith('---'):
            removed_lines.append(line[1:])
        elif line.startswith('+') and not line.startswith('+++'):
            added_lines.append(line[1:])

    # Process last hunk
    if not _is_safe_hunk(removed_lines, added_lines):
        return False
    if added_lines:
        has_additions = True

    return has_additions


def _is_comment_only(lines: list[str]) -> bool:
    """Check if all lines are comments, blank, or docstrings."""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            continue
        # Triple-quote docstring lines
        if stripped.startswith(('"""', "'''", '\"\"\"', "'''")):
            continue
        # Continuation of a docstring (plain text inside """)
        # Heuristic: if it doesn't look like code, treat as comment.
        # But to be safe, only skip lines that are clearly comments.
        return False
    return True


def _is_safe_hunk(removed: list[str], added: list[str]) -> bool:
    """Check if a single diff hunk is safe to skip.

    - Comment-only changes (all added/removed lines are comments/blanks): safe.
    - Hunk with deletions: safe only if it's an expansion pattern.
    - Hunk with only additions: safe only if the additions are new
      definitions, imports, or comments (not code inside existing bodies).
    - Empty hunk: safe.
    """
    if not removed and not added:
        return True
    # Comment-only: both sides are all comments/blanks
    if _is_comment_only(removed) and _is_comment_only(added):
        return True
    if removed:
        return _is_expansion(removed, added)
    # Pure addition — check if it's a safe kind of addition
    return _is_safe_addition(added)


def _is_safe_addition(added: list[str]) -> bool:
    """Check if purely-added lines are safe (new definitions, imports, data entries).

    Safe patterns:
      - New function/class definition (possibly preceded by decorators)
      - Import statements
      - Data structure entries (dict/list items ending with ',')
      - Comments and blank lines only

    Unsafe: any executable code added inside an existing function body
    (if/else, assignments, function calls, etc.).
    """
    # Filter to significant lines (non-blank, non-comment)
    significant = [
        line for line in added
        if line.strip() and not line.strip().startswith('#')
    ]
    if not significant:
        return True  # only comments/blanks

    # Check if the additions are data structure entries (dict/list items).
    # A block of data entries ends with ',' on its last significant line,
    # and contains no control flow keywords. This catches:
    #   - `key="value",`
    #   - `"item",`
    #   - multi-line entries: `'key':\n    'value',`
    last_stripped = significant[-1].rstrip()
    if last_stripped.endswith(','):
        control_flow = ('if ', 'else:', 'elif ', 'for ', 'while ', 'return ',
                        'yield ', 'raise ', 'assert ', 'with ')
        if not any(s.strip().startswith(control_flow) for s in significant):
            return True

    # Walk through significant lines: decorators (@...) are OK if followed
    # by a def/class. Imports are OK. Anything else is unsafe.
    i = 0
    while i < len(significant):
        stripped = significant[i].strip()

        # Import statements are safe
        if stripped.startswith(('import ', 'from ')):
            i += 1
            continue

        # Decorators: consume consecutive decorators, then expect def/class
        if stripped.startswith('@'):
            i += 1
            while i < len(significant) and significant[i].strip().startswith('@'):
                i += 1
            # After decorators, must see def or class
            if i < len(significant):
                next_stripped = significant[i].strip()
                if next_stripped.startswith(('def ', 'class ')):
                    # Skip the entire definition body (all subsequent lines
                    # at deeper indentation belong to this new definition)
                    def_indent = len(significant[i]) - len(significant[i].lstrip())
                    i += 1
                    while i < len(significant):
                        line_indent = len(significant[i]) - len(significant[i].lstrip())
                        if line_indent <= def_indent:
                            break  # back to same or outer indentation
                        i += 1
                    continue
            # Decorators not followed by def/class → unsafe
            return False

        # New function/class definition
        if stripped.startswith(('def ', 'class ')):
            def_indent = len(significant[i]) - len(significant[i].lstrip())
            i += 1
            # Skip body lines (deeper indentation)
            while i < len(significant):
                line_indent = len(significant[i]) - len(significant[i].lstrip())
                if line_indent <= def_indent:
                    break
                i += 1
            continue

        # Anything else (if, assignment, function call, etc.) → unsafe
        return False

    return True


def _is_expansion(removed: list[str], added: list[str]) -> bool:
    """Check if removed lines are "expanded" versions found in added lines.

    For each removed line, checks if its stripped content appears as a
    substring of any added line. This detects patterns like:
      - `case "a" | "b":` → `case "a" | "b" | "c":`
      - `["x", "y"]` → `["x", "y", "z"]`
      - whitespace-only reformatting
    """
    if not removed:
        return True

    # Each removed line's content must be "contained" in some added line.
    # Use best-match strategy: for each removed line, find the added line
    # with the highest overlap to avoid ordering issues.
    used_indices: set[int] = set()
    for rm in removed:
        rm_stripped = rm.strip()
        if not rm_stripped:
            continue  # blank line removal is always safe

        rm_core = rm_stripped.rstrip(':,)]}')
        if not rm_core:
            continue

        best_idx = -1
        best_len = -1
        for i, ad in enumerate(added):
            if i in used_indices:
                continue
            ad_core = ad.strip().rstrip(':,)]}')
            if rm_core in ad_core and len(ad_core) > best_len:
                best_idx = i
                best_len = len(ad_core)

        if best_idx < 0:
            return False
        used_indices.add(best_idx)

    return True




def _match_files_to_rules(
        changed_files: list[str]) -> list[tuple[str, ImpactRule]]:
    """Match each changed file to applicable impact rules."""
    matches = []
    matched_files = set()

    for fpath in changed_files:
        file_matched = False
        for rule in IMPACT_RULES:
            if match_rule(fpath, rule):
                matches.append((fpath, rule))
                file_matched = True
        if file_matched:
            matched_files.add(fpath)

    return matches


def _select_by_core(database: dict[str, TestEntry], result: SelectionResult,
                    rule: ImpactRule, changed_file: str):
    """Tier 0: Add representative coverage set."""
    for test_id in REPRESENTATIVE_COVERAGE_SET:
        if test_id in database:
            result.add(test_id, database[test_id],
                       f"Tier0:CORE ({rule.description}) <- {changed_file}")


def _select_by_default_on(database: dict[str, TestEntry],
                           result: SelectionResult, rule: ImpactRule,
                           changed_file: str):
    """Tier 1: Tests that explicitly configure this feature + representative set."""
    feature = rule.feature
    config_class = rule.config_class

    for test_id, entry in database.items():
        matched = False

        # Match by config class (L3)
        if config_class and config_class in entry.config_classes:
            matched = True

        # Match by feature tag
        if feature and feature in entry.features:
            matched = True

        # For attention backends, also match tests that parametrize the backend
        if feature == 'attn_flashinfer' and 'attn_flashinfer' in entry.features:
            matched = True
        if feature == 'attn_trtllm' and 'attn_trtllm' in entry.features:
            matched = True

        if matched:
            result.add(
                test_id, entry,
                f"Tier1:DEFAULT_ON feature={feature} ({rule.description}) <- {changed_file}"
            )

    # Also add representative set
    _select_by_core(database, result, rule, changed_file)


def _select_by_opt_in(database: dict[str, TestEntry],
                       result: SelectionResult, rule: ImpactRule,
                       changed_file: str):
    """Tier 2: Only tests that use this specific feature."""
    feature = rule.feature
    config_class = rule.config_class

    # For "speculative" (the interface), match all speculative decoding tests
    if feature == 'speculative':
        for test_id, entry in database.items():
            if entry.features & SPECULATIVE_FEATURES:
                result.add(
                    test_id, entry,
                    f"Tier2:OPT_IN feature=speculative ({rule.description}) <- {changed_file}"
                )
            for cc in entry.config_classes:
                if cc in ('Eagle3DecodingConfig', 'MTPDecodingConfig',
                          'PARDDecodingConfig', 'NGramDecodingConfig',
                          'SADecodingConfig', 'AutoDecodingConfig'):
                    result.add(
                        test_id, entry,
                        f"Tier2:OPT_IN config={cc} ({rule.description}) <- {changed_file}"
                    )
        return

    # For quantization (broad), match any quant-related test
    if feature == 'quantization':
        quant_features = {'fp8', 'nvfp4', 'fp4', 'w4', 'w4a8', 'w4a16', 'mxfp4'}
        for test_id, entry in database.items():
            if entry.features & quant_features:
                result.add(
                    test_id, entry,
                    f"Tier2:OPT_IN feature=quantization ({rule.description}) <- {changed_file}"
                )
        return

    # For disaggregated, match disagg test files too
    if feature == 'disaggregated':
        for test_id, entry in database.items():
            if ('disaggregated' in entry.test_file
                    or 'disagg' in entry.test_file):
                result.add(
                    test_id, entry,
                    f"Tier2:OPT_IN feature=disaggregated ({rule.description}) <- {changed_file}"
                )
        return

    for test_id, entry in database.items():
        matched = False

        # Match by config class (L3 analysis)
        if config_class and config_class in entry.config_classes:
            matched = True

        # Match by feature tag (from method name / params)
        if feature and feature in entry.features:
            matched = True

        # Also check config classes mapped from feature name
        if feature and feature in FEATURE_TO_CONFIG:
            if entry.config_classes & FEATURE_TO_CONFIG[feature]:
                matched = True

        if matched:
            result.add(
                test_id, entry,
                f"Tier2:OPT_IN feature={feature} ({rule.description}) <- {changed_file}"
            )


def _select_by_model(database: dict[str, TestEntry],
                      result: SelectionResult, rule: ImpactRule,
                      changed_file: str):
    """MODEL tier: All tests for the given architecture."""
    arch = rule.arch
    for test_id, entry in database.items():
        if entry.arch == arch:
            result.add(
                test_id, entry,
                f"MODEL arch={arch} ({rule.description}) <- {changed_file}")


def _get_changed_classes_in_test_file(changed_file: str,
                                      base_ref: str = None) -> set[str]:
    """Use git diff to identify which test classes were modified in a test file.

    Parses the diff hunks to find class names near changed lines.
    Returns empty set if we can't determine (meaning: select all).
    """
    if not base_ref:
        # Try to get base_ref from the selector context; fall back to HEAD~1
        base_ref = "HEAD~1"

    try:
        result = subprocess.run(
            ["git", "diff", "-U0", f"{base_ref}...HEAD", "--", changed_file],
            capture_output=True,
            text=True,
            check=True,
        )
        diff_output = result.stdout
    except subprocess.CalledProcessError:
        return set()  # Can't get diff, select all

    if not diff_output.strip():
        return set()

    # Parse @@ hunk headers to get changed line numbers
    changed_lines = set()
    for m in re.finditer(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@',
                         diff_output):
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) else 1
        for line_num in range(start, start + count):
            changed_lines.add(line_num)

    if not changed_lines:
        return set()

    # Read the file and build a line_number → enclosing_class map
    try:
        with open(changed_file) as f:
            source = f.read()
    except FileNotFoundError:
        return set()

    # Find all class definitions and their line ranges
    class_ranges: list[tuple[str, int, int]] = []  # (name, start, end)
    class_pattern = re.compile(r'^class\s+(\w+)\s*[\(:]', re.MULTILINE)
    lines = source.split('\n')

    class_starts = []
    for m in class_pattern.finditer(source):
        line_num = source[:m.start()].count('\n') + 1
        class_starts.append((m.group(1), line_num))

    for i, (name, start) in enumerate(class_starts):
        end = class_starts[i + 1][1] - 1 if i + 1 < len(class_starts) else len(
            lines)
        class_ranges.append((name, start, end))

    # Find which classes contain changed lines
    changed_classes = set()
    for name, start, end in class_ranges:
        for line_num in changed_lines:
            if start <= line_num <= end:
                changed_classes.add(name)
                break

    # Also detect changes outside any class (module-level helpers, imports)
    min_class_start = min((s for _, s, _ in class_ranges),
                          default=len(lines) + 1)
    has_module_level_changes = any(ln < min_class_start
                                  for ln in changed_lines)

    if has_module_level_changes:
        # Module-level changes (imports, helpers) could affect any test
        # Return empty to signal "select all from this file"
        return set()

    return changed_classes


# Module-level variable to pass base_ref context to _select_by_test
_current_base_ref: str = ""


def _select_by_test(database: dict[str, TestEntry], result: SelectionResult,
                     rule: ImpactRule, changed_file: str):
    """TEST tier: Tests from the changed test file.

    For large test files (like test_llm_api_pytorch.py), uses git diff to
    identify which test classes were actually modified, and only selects
    tests from those classes. Falls back to selecting all tests from the
    file if we can't determine the changed classes.
    """
    # Try to narrow down to specific changed classes
    changed_classes = _get_changed_classes_in_test_file(changed_file,
                                                        _current_base_ref)

    for test_id, entry in database.items():
        # Check if this test belongs to the changed file
        if not (changed_file.endswith(entry.test_file)
                or entry.test_file in changed_file):
            continue

        if changed_classes:
            # Only select tests from changed classes
            if entry.test_class in changed_classes:
                result.add(
                    test_id, entry,
                    f"TEST class={entry.test_class} modified in {changed_file}"
                )
        else:
            # Can't determine changed classes → select all from this file
            result.add(test_id, entry,
                       f"TEST file={changed_file} (all classes)")


def _select_new_tests(database: dict[str, TestEntry],
                      result: SelectionResult,
                      changed_files: list[str],
                      base_ref: str):
    """Select any newly added tests from changed test list files.

    If a test list (.txt) was modified, parse the diff to find added lines
    (new test IDs) and ensure they are selected. New tests should always run
    to verify they pass.
    """
    if not base_ref:
        return

    test_list_files = [
        f for f in changed_files
        if f.startswith("tests/integration/test_lists/qa/") and f.endswith(".txt")
    ]
    if not test_list_files:
        return

    for list_file in test_list_files:
        try:
            diff_result = subprocess.run(
                ["git", "diff", "-U0", f"{base_ref}...HEAD", "--", list_file],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError:
            continue

        for line in diff_result.stdout.split('\n'):
            if not line.startswith('+') or line.startswith('+++'):
                continue
            test_id = line[1:].strip()
            if not test_id or test_id.startswith('#'):
                continue
            if test_id in database:
                result.add(test_id, database[test_id],
                           f"NEW_TEST: added to {list_file}")


def select_tests(database: dict[str, TestEntry],
                 changed_files: list[str],
                 base_ref: str = "") -> SelectionResult:
    """Main entry: select tests based on changed files.

    Args:
        database: Pre-built test database from parser.build_test_database()
        changed_files: List of changed file paths (relative to repo root)
        base_ref: Git ref that was used to compute the diff (for TEST tier
            class-level narrowing)

    Returns:
        SelectionResult with selected tests and explanations
    """
    global _current_base_ref
    _current_base_ref = base_ref

    result = SelectionResult()

    # Match files to rules
    matches = _match_files_to_rules(changed_files)
    result.triggered_rules = matches

    # Track unmatched files
    matched_files = {f for f, _ in matches}
    result.unmatched_files = [f for f in changed_files if f not in matched_files]

    # Apply selection per rule
    tier_handlers = {
        Tier.CORE: _select_by_core,
        Tier.DEFAULT_ON: _select_by_default_on,
        Tier.OPT_IN: _select_by_opt_in,
        Tier.MODEL: _select_by_model,
        Tier.TEST: _select_by_test,
    }

    # Pre-compute additive-only status for files with high-impact tiers
    _additive_tiers = {Tier.CORE, Tier.DEFAULT_ON, Tier.OPT_IN, Tier.MODEL}
    additive_cache: dict[str, bool] = {}

    for changed_file, rule in matches:
        if rule.tier not in tier_handlers:
            continue

        # Step 1: Skip if diff is purely additive / expansion-only
        if rule.tier in _additive_tiers and base_ref:
            if changed_file not in additive_cache:
                additive_cache[changed_file] = _is_additive_only(
                    changed_file, base_ref)
            if additive_cache[changed_file]:
                result.add_skipped(
                    changed_file, rule,
                    f"SKIP (additive-only): {rule.tier.name} "
                    f"{rule.feature or rule.arch or 'core'} "
                    f"<- {changed_file}")
                continue

        tier_handlers[rule.tier](database, result, rule, changed_file)

    # If there are unmatched files that look like source code, add representative set
    # as a safety fallback
    code_extensions = {'.py', '.cpp', '.h', '.cu', '.cuh', '.cc'}
    unmatched_code = [
        f for f in result.unmatched_files
        if any(f.endswith(ext) for ext in code_extensions)
    ]
    if unmatched_code:
        for test_id in REPRESENTATIVE_COVERAGE_SET:
            if test_id in database:
                result.add(
                    test_id, database[test_id],
                    f"FALLBACK: unmatched code files: {', '.join(unmatched_code[:3])}"
                )

    # Always select newly added tests: if a test list file changed,
    # extract newly added test IDs from the diff and select them.
    _select_new_tests(database, result, changed_files, base_ref)

    # Deduplicate parametrized variants: cap the number of parametrized
    # variants per (class, method) to keep the selected set small.
    _deduplicate_parametrized(result, max_variants=1)

    # Second pass: cap to 1 method per TestClass for all non-CORE tests.
    # The representative set covers the broad case; extra tests from
    # DEFAULT_ON/OPT_IN/MODEL/TEST are supplementary.
    _deduplicate_per_class(result)

    return result


def _deduplicate_parametrized(result: SelectionResult,
                              max_variants: int = 3):
    """Cap parametrized variants per (test_class, test_method).

    Deduplicates tests whose reasons are ALL from OPT_IN or MODEL tiers.
    Tests in the representative set, or selected by CORE/DEFAULT_ON/TEST
    tiers, are never removed.

    When capping, selects a spread: first, last, and evenly-spaced middle
    entries (sorted alphabetically by test_id) to maximize parameter diversity.
    """
    _dedup_tiers = {"Tier2:OPT_IN", "MODEL", "TEST", "Tier1:DEFAULT_ON"}
    # Only CORE (representative set) and FALLBACK are exempt from dedup.
    # DEFAULT_ON is dedup-eligible: the representative set already provides
    # broad coverage, DEFAULT_ON tests are supplementary.
    _no_dedup_tiers = {"Tier0:CORE", "FALLBACK", "NEW_TEST"}

    # Protect: never remove representative set entries
    protected_ids = set(REPRESENTATIVE_COVERAGE_SET)

    # Group candidates by (test_class, test_method)
    groups: dict[tuple[str, str], list[str]] = {}
    for test_id, entry in result.selected_tests.items():
        if test_id in protected_ids:
            continue
        # Skip dedup if ANY reason is from a non-dedup tier (CORE/DEFAULT_ON/FALLBACK)
        reasons = result.reasons.get(test_id, [])
        if not reasons:
            continue
        if any(any(nd in r for nd in _no_dedup_tiers) for r in reasons):
            continue
        key = (entry.test_class, entry.test_method)
        groups.setdefault(key, []).append(test_id)

    # For each group that exceeds max_variants, keep the variants with the
    # most enabled features. Feature-rich variants exercise more code paths
    # per test, making them higher-value representatives.
    for (cls, method), test_ids in groups.items():
        if len(test_ids) <= max_variants:
            continue

        n = len(test_ids)
        ranked = sorted(test_ids,
                        key=lambda tid: len(result.selected_tests[tid].features),
                        reverse=True)
        keep_ids = set(ranked[:max_variants])
        remove_ids = set(test_ids) - keep_ids

        for tid in remove_ids:
            del result.selected_tests[tid]
            # Update reasons to note dedup
            result.reasons.pop(tid, None)

        # Annotate kept tests
        for tid in keep_ids:
            result.reasons[tid].append(
                f"DEDUP: kept {max_variants}/{n} variants of "
                f"{cls}::{method}"
            )


def _deduplicate_per_class(result: SelectionResult):
    """Second-pass dedup: keep 1 method per TestClass.

    After the first pass (1 variant per method), there can still be many
    different methods per TestClass. For non-CORE tests, one method per class
    is sufficient — pick the one with the most features.

    Only CORE (representative set) and FALLBACK are exempt.
    """
    protected_ids = set(REPRESENTATIVE_COVERAGE_SET)
    _no_dedup = {"Tier0:CORE", "FALLBACK", "NEW_TEST"}

    # Group by test_class
    class_groups: dict[str, list[str]] = {}
    for test_id, entry in result.selected_tests.items():
        if test_id in protected_ids:
            continue
        reasons = result.reasons.get(test_id, [])
        if not reasons:
            continue
        if any(any(nd in r for nd in _no_dedup) for r in reasons):
            continue
        class_groups.setdefault(entry.test_class, []).append(test_id)

    for cls, test_ids in class_groups.items():
        if len(test_ids) <= 1:
            continue

        n = len(test_ids)
        ranked = sorted(
            test_ids,
            key=lambda tid: len(result.selected_tests[tid].features),
            reverse=True,
        )
        keep_id = ranked[0]
        remove_ids = set(test_ids) - {keep_id}

        for tid in remove_ids:
            del result.selected_tests[tid]
            result.reasons.pop(tid, None)

        result.reasons[keep_id].append(
            f"DEDUP-CLASS: kept 1/{n} methods of {cls}"
        )


def _primary_reason(reasons: list[str]) -> str:
    """Extract a short, groupable reason key from the first reason string.

    Reason strings look like:
      "Tier0:CORE (Main LLM API entry point) <- tensorrt_llm/llmapi/llm.py"
      "Tier2:OPT_IN feature=eagle3 (...) <- tensorrt_llm/..."
      "MODEL arch=deepseek_v2 (...) <- tensorrt_llm/..."
      "TEST class=TestGLM5FP8 modified in ..."
      "FALLBACK: unmatched code files: ..."
    """
    if not reasons:
        return "unknown"

    # New tests — check ALL reasons (NEW_TEST may not be the first)
    if any("NEW_TEST" in reason for reason in reasons):
        return "NEW_TEST: newly added to test list"

    r = reasons[0]

    # TEST tier
    if r.startswith("TEST class="):
        cls = r.split("class=")[1].split(" ")[0]
        return f"TEST: {cls} modified"
    if r.startswith("TEST file="):
        return "TEST: test file modified"

    # FALLBACK
    if r.startswith("FALLBACK"):
        return "FALLBACK: unmatched code"

    # Tier-based: extract tier and feature/arch
    if "Tier0:CORE" in r:
        return "Representative coverage set (core infrastructure change)"

    if "Tier1:DEFAULT_ON" in r:
        m = re.search(r'feature=(\S+)', r)
        feat = m.group(1) if m else "unknown"
        return f"DEFAULT_ON: {feat}"

    if "Tier2:OPT_IN" in r:
        m = re.search(r'feature=(\S+)', r)
        feat = m.group(1) if m else "unknown"
        return f"OPT_IN: {feat}"

    if r.startswith("MODEL"):
        m = re.search(r'arch=(\S+)', r)
        arch = m.group(1) if m else "unknown"
        return f"MODEL: {arch}"

    return reasons[0][:60]


def format_output(result: SelectionResult,
                  test_list_filter: str = None) -> str:
    """Format selection result as a grouped test list.

    Tests are grouped by their primary selection reason, with a comment
    header for each group. Within each group, tests are sorted alphabetically.

    Args:
        result: SelectionResult from select_tests()
        test_list_filter: If set, only include tests from this test list
            (e.g. 'llm_function_core')
    """
    # Group tests by primary reason
    groups: dict[str, list[str]] = {}
    for test_id in result.selected_tests:
        entry = result.selected_tests[test_id]
        if test_list_filter and test_list_filter not in entry.test_lists:
            continue
        reason_key = _primary_reason(result.reasons.get(test_id, []))
        groups.setdefault(reason_key, []).append(test_id)

    # Sort groups: representative set first, then by group size descending
    def group_sort_key(item):
        key, tests = item
        if key.startswith("NEW_TEST"):
            return (0, -len(tests), key)
        if "Representative" in key or "representative" in key:
            return (1, -len(tests), key)
        if key.startswith("MODEL"):
            return (2, -len(tests), key)
        if key.startswith("OPT_IN"):
            return (3, -len(tests), key)
        if key.startswith("TEST"):
            return (4, -len(tests), key)
        return (5, -len(tests), key)

    lines = []
    for reason_key, test_ids in sorted(groups.items(), key=group_sort_key):
        lines.append(f"# {reason_key} ({len(test_ids)} tests)")
        for tid in sorted(test_ids):
            lines.append(tid)
        lines.append("")  # blank line between groups

    # Remove trailing blank line
    if lines and lines[-1] == "":
        lines.pop()

    return '\n'.join(lines)


def format_explain(result: SelectionResult) -> str:
    """Format a detailed explanation of why each test was selected."""
    lines = []

    # Summary
    lines.append(f"=== Change-Based Test Selection Report ===")
    lines.append(f"Total tests selected: {len(result.selected_tests)}")
    lines.append(f"Rules triggered: {len(result.triggered_rules)}")
    lines.append(f"Rules skipped (additive-only): {len(result.skipped_rules)}")
    lines.append(f"Unmatched files: {len(result.unmatched_files)}")
    lines.append("")

    # Triggered rules summary
    if result.triggered_rules:
        lines.append("--- Triggered Rules ---")
        seen = set()
        for fpath, rule in result.triggered_rules:
            key = (fpath, rule.pattern, rule.tier.name)
            if key not in seen:
                seen.add(key)
                lines.append(
                    f"  {fpath} -> {rule.tier.name}"
                    f" [{rule.feature or rule.arch or 'core'}]"
                    f" ({rule.description})")
        lines.append("")

    # Skipped rules (additive-only)
    if result.skipped_rules:
        lines.append("--- Skipped Rules (additive-only diff) ---")
        for fpath, rule, reason in result.skipped_rules:
            lines.append(
                f"  {fpath} -> {rule.tier.name}"
                f" [{rule.feature or rule.arch or 'core'}]"
                f" SKIPPED ({rule.description})")
        lines.append("")

    # Unmatched files
    if result.unmatched_files:
        lines.append("--- Unmatched Files ---")
        for f in result.unmatched_files:
            lines.append(f"  {f}")
        lines.append("")

    # Per-test reasons
    lines.append("--- Selected Tests ---")
    for test_id in sorted(result.selected_tests.keys()):
        reasons = result.reasons.get(test_id, [])
        lines.append(f"  {test_id}")
        for r in reasons:
            lines.append(f"    <- {r}")

    return '\n'.join(lines)


@dataclass
class SuspectCommit:
    """A commit that may have caused a test regression."""

    sha: str
    subject: str
    changed_files: list[str]
    matched_rules: list[tuple[str, ImpactRule]]  # (file, rule) pairs
    is_additive_only: bool
    relevance: str  # "direct", "broad", "none"


def find_suspects(
    test_id: str,
    database: dict[str, TestEntry],
    good_ref: str,
    bad_ref: str,
    repo_root: str = ".",
) -> list[SuspectCommit]:
    """Reverse suspect analysis: rank commits by relevance to a failing test.

    Given a failing test ID and a good..bad commit range, identifies which
    commits modified code paths relevant to the failing test.

    Args:
        test_id: Full pytest node ID of the failing test
        database: Test database from parser.build_test_database()
        good_ref: Last known good commit SHA
        bad_ref: First known bad commit SHA
        repo_root: Path to repo root

    Returns:
        List of SuspectCommit, most suspicious first.
    """
    entry = database.get(test_id)
    if not entry:
        # Try partial match
        for tid, e in database.items():
            if test_id in tid:
                entry = e
                test_id = tid
                break
    if not entry:
        raise ValueError(f"Test ID not found in database: {test_id}")

    # Get commits in range (good exclusive, bad inclusive)
    try:
        result = subprocess.run(
            ["git", "log", "--reverse", "--format=%H %s",
             f"{good_ref}..{bad_ref}"],
            capture_output=True, text=True, cwd=repo_root, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"git log failed: {e.stderr}") from e

    commits = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        sha, subject = line.split(' ', 1)
        commits.append((sha, subject))

    if not commits:
        return []

    suspects: list[SuspectCommit] = []

    for sha, subject in commits:
        # Get changed files for this commit
        try:
            diff_result = subprocess.run(
                ["git", "diff", "--name-only", f"{sha}~1", sha],
                capture_output=True, text=True, cwd=repo_root, check=True,
            )
        except subprocess.CalledProcessError:
            continue
        changed_files = [
            f.strip() for f in diff_result.stdout.strip().split('\n')
            if f.strip()
        ]
        if not changed_files:
            continue

        # Match against impact rules
        matched = []
        for fpath in changed_files:
            for rule in IMPACT_RULES:
                if match_rule(fpath, rule):
                    matched.append((fpath, rule))

        if not matched:
            suspects.append(SuspectCommit(
                sha=sha, subject=subject, changed_files=changed_files,
                matched_rules=[], is_additive_only=False, relevance="none",
            ))
            continue

        # Filter for relevance to the target test
        relevant = []
        for fpath, rule in matched:
            if rule.tier == Tier.IGNORE:
                continue
            if rule.tier in (Tier.CORE, Tier.DEFAULT_ON):
                relevant.append((fpath, rule))
            elif rule.tier == Tier.OPT_IN:
                # Relevant if feature matches test's features or config classes
                feat_match = rule.feature and (
                    rule.feature in entry.features
                    or (rule.feature == 'speculative'
                        and bool(entry.features & SPECULATIVE_FEATURES))
                    or (rule.feature == 'quantization' and bool(
                        entry.features & {'fp8', 'nvfp4', 'fp4', 'w4',
                                          'w4a8', 'w4a16', 'mxfp4'}))
                    or (rule.feature == 'disaggregated' and (
                        'disaggregated' in entry.test_file
                        or 'disagg' in entry.test_file))
                )
                cfg_match = (rule.config_class
                             and rule.config_class in entry.config_classes)
                if feat_match or cfg_match:
                    relevant.append((fpath, rule))
            elif rule.tier == Tier.MODEL:
                if rule.arch and rule.arch == entry.arch:
                    relevant.append((fpath, rule))
            elif rule.tier == Tier.TEST:
                if (fpath.endswith(entry.test_file)
                        or entry.test_file in fpath):
                    relevant.append((fpath, rule))

        if not relevant:
            suspects.append(SuspectCommit(
                sha=sha, subject=subject, changed_files=changed_files,
                matched_rules=matched, is_additive_only=False,
                relevance="none",
            ))
            continue

        # Check additive-only status using commit's parent as base
        additive = True
        for fpath, rule in relevant:
            if rule.tier in (Tier.CORE, Tier.DEFAULT_ON, Tier.OPT_IN,
                             Tier.MODEL):
                try:
                    chk = subprocess.run(
                        ["git", "diff", "-U0", f"{sha}~1", sha, "--", fpath],
                        capture_output=True, text=True, cwd=repo_root,
                        check=True,
                    )
                    diff_text = chk.stdout.strip()
                except subprocess.CalledProcessError:
                    additive = False
                    break

                if not diff_text:
                    continue

                # Parse and check hunks
                removed_lines: list[str] = []
                added_lines: list[str] = []
                safe = True
                for dline in diff_text.split('\n'):
                    if dline.startswith('@@'):
                        if not _is_safe_hunk(removed_lines, added_lines):
                            safe = False
                            break
                        removed_lines = []
                        added_lines = []
                    elif dline.startswith('-') and not dline.startswith('---'):
                        removed_lines.append(dline[1:])
                    elif dline.startswith('+') and not dline.startswith('+++'):
                        added_lines.append(dline[1:])
                if safe and not _is_safe_hunk(removed_lines, added_lines):
                    safe = False

                if not safe:
                    additive = False
                    break

        relevance = "direct" if not additive else "broad"
        suspects.append(SuspectCommit(
            sha=sha, subject=subject, changed_files=changed_files,
            matched_rules=relevant, is_additive_only=additive,
            relevance=relevance,
        ))

    # Sort: direct (non-additive) first, then broad, then none
    rank = {"direct": 0, "broad": 1, "none": 2}
    suspects.sort(key=lambda s: (rank.get(s.relevance, 3), s.sha))

    return suspects


_PR_URL_BASE = "https://github.com/NVIDIA/TensorRT-LLM/pull/"


def _extract_pr_link(subject: str) -> str:
    """Extract PR number from commit subject and return a clickable link."""
    m = re.search(r'\(#(\d+)\)', subject)
    if m:
        return f"{_PR_URL_BASE}{m.group(1)}"
    return ""


def format_suspects(suspects: list[SuspectCommit],
                    test_id: str) -> str:
    """Format suspect analysis results for display."""
    lines = [
        f"=== Reverse Suspect Analysis ===",
        f"Failing test: {test_id}",
        f"Commits analyzed: {len(suspects)}",
        "",
    ]

    direct = [s for s in suspects if s.relevance == "direct"]
    broad = [s for s in suspects if s.relevance == "broad"]
    clear = [s for s in suspects if s.relevance == "none"]

    if direct:
        lines.append(f"--- SUSPECTS ({len(direct)}) ---")
        for i, s in enumerate(direct, 1):
            pr_link = _extract_pr_link(s.subject)
            lines.append(f"  #{i} {s.sha[:10]} {s.subject}")
            if pr_link:
                lines.append(f"      PR: {pr_link}")
            for fpath, rule in s.matched_rules:
                lines.append(
                    f"      {rule.tier.name} [{rule.feature or rule.arch or 'core'}]"
                    f" <- {fpath}")
        lines.append("")

    if broad:
        lines.append(f"--- ADDITIVE-ONLY ({len(broad)}) ---")
        for s in broad:
            pr_link = _extract_pr_link(s.subject)
            lines.append(f"  ~ {s.sha[:10]} {s.subject}")
            if pr_link:
                lines.append(f"      PR: {pr_link}")
            for fpath, rule in s.matched_rules:
                lines.append(
                    f"      {rule.tier.name} [{rule.feature or rule.arch or 'core'}]"
                    f" <- {fpath} (additive)")
        lines.append("")

    if clear:
        lines.append(f"--- CLEAR ({len(clear)}) ---")
        for s in clear:
            lines.append(f"  . {s.sha[:10]} {s.subject}")
        lines.append("")

    return '\n'.join(lines)


def compute_stats(result: SelectionResult,
                  database: dict[str, TestEntry]) -> dict:
    """Compute selection statistics."""
    total = len(database)
    selected = len(result.selected_tests)

    # By architecture
    arch_counts: dict[str, int] = {}
    for entry in result.selected_tests.values():
        arch = entry.arch or 'unknown'
        arch_counts[arch] = arch_counts.get(arch, 0) + 1

    # By test list
    list_counts: dict[str, int] = {}
    for entry in result.selected_tests.values():
        for tl in entry.test_lists:
            list_counts[tl] = list_counts.get(tl, 0) + 1

    # By tier (from triggered rules)
    tier_counts: dict[str, int] = {}
    for _, rule in result.triggered_rules:
        tier_counts[rule.tier.name] = tier_counts.get(rule.tier.name, 0) + 1

    return {
        'total_in_database': total,
        'total_selected': selected,
        'selection_ratio': f"{selected / total * 100:.1f}%" if total else "N/A",
        'by_architecture': arch_counts,
        'by_test_list': list_counts,
        'rules_by_tier': tier_counts,
        'unmatched_files': len(result.unmatched_files),
    }
