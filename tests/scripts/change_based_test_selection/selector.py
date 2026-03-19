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

import json
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
    """MODEL tier: All tests for the given architecture.

    Matches both:
      - Class-based tests via entry.arch (from TESTCLASS_TO_ARCH)
      - Module-level tests (e.g. test_e2e.py) via entry.model_names
        mapped to arch using MODEL_NAME_TO_ARCH
    """
    from .impact_rules import model_name_to_arch

    arch = rule.arch
    for test_id, entry in database.items():
        # Class-based match
        if entry.arch == arch:
            result.add(
                test_id, entry,
                f"MODEL arch={arch} ({rule.description}) <- {changed_file}")
            continue

        # Module-level match via model_names
        if entry.model_names:
            for mname in entry.model_names:
                if model_name_to_arch(mname) == arch:
                    result.add(
                        test_id, entry,
                        f"MODEL arch={arch} model={mname} "
                        f"({rule.description}) <- {changed_file}")
                    break


@dataclass
class _TestFileChanges:
    """Result of analyzing which parts of a test file were changed."""

    # Classes where we identified specific changed methods.
    # class_name → set of changed method names.
    class_methods: dict[str, set[str]] = field(default_factory=dict)
    # Classes where we know the class changed but can't narrow to methods
    # (e.g. class-level attribute change). Select all methods for these.
    class_wide: set[str] = field(default_factory=set)
    # True if we couldn't determine anything (select all from file).
    select_all: bool = False


def _get_changed_classes_in_test_file(changed_file: str,
                                      base_ref: str = None) -> _TestFileChanges:
    """Use git diff to identify which test classes/methods were modified.

    Parses the diff hunks to find class and method names near changed lines.
    Returns _TestFileChanges with method-level precision when possible.
    """
    if not base_ref:
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
        return _TestFileChanges(select_all=True)

    if not diff_output.strip():
        return _TestFileChanges(select_all=True)

    # Parse @@ hunk headers to get changed line numbers
    changed_lines = set()
    for m in re.finditer(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@',
                         diff_output):
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) else 1
        for line_num in range(start, start + count):
            changed_lines.add(line_num)

    if not changed_lines:
        return _TestFileChanges(select_all=True)

    # Read the file and build structural maps
    try:
        with open(changed_file) as f:
            source = f.read()
    except FileNotFoundError:
        return _TestFileChanges(select_all=True)

    lines = source.split('\n')

    # Find all class definitions and their line ranges
    class_ranges: list[tuple[str, int, int]] = []  # (name, start, end)
    class_pattern = re.compile(r'^class\s+(\w+)\s*[\(:]', re.MULTILINE)

    class_starts = []
    for m in class_pattern.finditer(source):
        line_num = source[:m.start()].count('\n') + 1
        class_starts.append((m.group(1), line_num))

    for i, (name, start) in enumerate(class_starts):
        end = class_starts[i + 1][1] - 1 if i + 1 < len(class_starts) else len(
            lines)
        class_ranges.append((name, start, end))

    # Find method definitions within each class: (class, method, start, end)
    method_ranges: list[tuple[str, str, int, int]] = []
    method_pattern = re.compile(r'^    def\s+(test_\w+)\s*\(', re.MULTILINE)
    for cls_name, cls_start, cls_end in class_ranges:
        cls_source = '\n'.join(lines[cls_start - 1:cls_end])
        method_starts = []
        for m in method_pattern.finditer(cls_source):
            mline = cls_source[:m.start()].count('\n') + cls_start
            method_starts.append((m.group(1), mline))
        for j, (mname, mstart) in enumerate(method_starts):
            mend = method_starts[j + 1][1] - 1 if j + 1 < len(method_starts) else cls_end
            method_ranges.append((cls_name, mname, mstart, mend))

    # Build result: map changed lines → methods → classes
    changes = _TestFileChanges()

    for cls_name, cls_start, cls_end in class_ranges:
        cls_changed_lines = {ln for ln in changed_lines
                             if cls_start <= ln <= cls_end}
        if not cls_changed_lines:
            continue

        # Find which methods within this class were changed
        cls_methods = [
            (mname, mstart, mend)
            for cn, mname, mstart, mend in method_ranges
            if cn == cls_name
        ]

        changed_methods = set()
        class_level_lines = set(cls_changed_lines)

        for mname, mstart, mend in cls_methods:
            method_lines = {ln for ln in cls_changed_lines
                            if mstart <= ln <= mend}
            if method_lines:
                changed_methods.add(mname)
                class_level_lines -= method_lines

        if class_level_lines:
            # Lines changed outside any method (class attrs, decorators,
            # class-level code) → select all methods in this class
            changes.class_wide.add(cls_name)
        elif changed_methods:
            changes.class_methods[cls_name] = changed_methods

    # Check for changes outside any class (module-level: imports, helpers)
    min_class_start = min((s for _, s, _ in class_ranges),
                          default=len(lines) + 1)
    module_level_changed_lines = {ln for ln in changed_lines
                                  if ln < min_class_start}

    if module_level_changed_lines:
        callers = _classify_module_level_changes(
            diff_output, module_level_changed_lines, lines, class_ranges)
        if callers is None:
            # Modification to existing shared code → select all
            return _TestFileChanges(select_all=True)
        # callers: classes that reference newly added symbols → class_wide
        changes.class_wide.update(callers)

    # If nothing was identified at all, fall back to select_all
    if not changes.class_methods and not changes.class_wide:
        return _TestFileChanges(select_all=True)

    return changes


def _classify_module_level_changes(
    diff_output: str,
    module_level_lines: set[int],
    file_lines: list[str],
    class_ranges: list[tuple[str, int, int]],
) -> set[str] | None:
    """Classify module-level diff hunks and determine affected classes.

    Returns:
        set[str]: class names that reference newly added/modified symbols.
        None: if changes modify existing shared code (caller should select all).
    """
    # Parse hunks to get (old_count, new_start, new_count) per hunk
    hunk_pattern = re.compile(
        r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
    # Collect added/removed lines per hunk from the raw diff
    hunks: list[dict] = []
    current_hunk = None
    for line in diff_output.splitlines():
        m = hunk_pattern.match(line)
        if m:
            if current_hunk is not None:
                hunks.append(current_hunk)
            old_count = int(m.group(2)) if m.group(2) else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) else 1
            current_hunk = {
                'old_count': old_count,
                'new_start': new_start,
                'new_count': new_count,
                'added': [],
                'removed': [],
            }
        elif current_hunk is not None:
            if line.startswith('+') and not line.startswith('+++'):
                current_hunk['added'].append(line[1:])
            elif line.startswith('-') and not line.startswith('---'):
                current_hunk['removed'].append(line[1:])
    if current_hunk is not None:
        hunks.append(current_hunk)

    # Filter to hunks that touch module-level lines
    module_hunks = []
    for h in hunks:
        hunk_lines = set(
            range(h['new_start'], h['new_start'] + h['new_count']))
        if hunk_lines & module_level_lines:
            module_hunks.append(h)

    if not module_hunks:
        return set()

    # Patterns for "harmless" lines: imports (including continuation lines
    # inside multi-line import parens), comments, and blanks.
    _harmless_pattern = re.compile(
        r'^\s*(?:'
        r'import |from \S+ import |'  # import statements
        r'#|'                          # comments
        r'[)\],]?\s*$|'               # closing parens, blank lines
        r'[A-Z]\w*(?:,\s*)?$|'        # continuation: identifier lines in import(...)
        r'\w+\s*,\s*$'                # "SamplingParams, SkipSoftmaxAttentionConfig,"
        r')')
    new_func_names: list[str] = []
    func_def_pattern = re.compile(r'^def\s+(\w+)\s*\(')

    for h in module_hunks:
        added = h['added']
        removed = h['removed']

        # Harmless changes: imports (incl. continuation), comments, blanks
        all_lines = added + removed
        has_meaningful = any(
            l.strip() and (
                func_def_pattern.match(l)
                or re.match(r'^class\s+', l)
                or ('=' in l and not l.strip().startswith('#'))
            )
            for l in all_lines)
        if not has_meaningful:
            continue

        # Pure addition (old_count == 0): new function/code block
        if h['old_count'] == 0:
            for l in added:
                m = func_def_pattern.match(l)
                if m:
                    new_func_names.append(m.group(1))
            continue

        # Modification of existing module-level code → can't narrow down
        return None

    if not new_func_names:
        # Only import changes or new code that doesn't define functions
        return set()

    # Find which classes reference the newly added functions
    callers: set[str] = set()
    for cls_name, cls_start, cls_end in class_ranges:
        cls_body = '\n'.join(file_lines[cls_start - 1:cls_end])
        for func_name in new_func_names:
            if func_name in cls_body:
                callers.add(cls_name)
                break

    return callers


# Module-level variable to pass base_ref context to _select_by_test
_current_base_ref: str = ""


def _select_by_test(database: dict[str, TestEntry], result: SelectionResult,
                     rule: ImpactRule, changed_file: str):
    """TEST tier: Tests from the changed test file.

    Uses git diff to identify changes at method-level precision:
      - Method body changed → select only that method's tests
      - Class-level change (attrs, decorators) → select all methods in class
      - Module-level shared code changed → select all from file
    """
    changes = _get_changed_classes_in_test_file(changed_file,
                                                _current_base_ref)

    for test_id, entry in database.items():
        # Check if this test belongs to the changed file
        if not (changed_file.endswith(entry.test_file)
                or entry.test_file in changed_file):
            continue

        if changes.select_all:
            result.add(test_id, entry,
                       f"TEST file={changed_file} (all classes)")
            continue

        # Class-wide changes: select all methods in these classes
        if entry.test_class in changes.class_wide:
            result.add(
                test_id, entry,
                f"TEST class={entry.test_class} modified in {changed_file}"
            )
            continue

        # Method-level precision: only select if this specific method changed
        if entry.test_class in changes.class_methods:
            if entry.test_method in changes.class_methods[entry.test_class]:
                result.add(
                    test_id, entry,
                    f"TEST method={entry.test_class}::{entry.test_method} "
                    f"modified in {changed_file}"
                )


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


def _load_waived_test_ids(repo_root: str) -> set[str]:
    """Load current waived test IDs from waives.txt.

    Returns a set of test IDs that are currently waived (known failures).
    These tests should be excluded from nightly selection to avoid wasting
    budget on tests that are expected to fail.
    """
    waives_path = (Path(repo_root) / "tests" / "integration" /
                   "test_lists" / "waives.txt")
    if not waives_path.exists():
        return set()

    waived = set()
    for line in waives_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        test_id = _parse_waive_test_id(line)
        if test_id:
            waived.add(test_id)
    return waived


def _exclude_waived_tests(result: SelectionResult, repo_root: str):
    """Remove currently waived tests from the selection.

    Waived tests are known failures — running them wastes time budget.
    The freed budget lets more non-failing tests run, improving coverage.

    Matching supports both exact IDs and prefix matching: a waive entry
    without parameters (e.g. `TestClass::test_method`) matches all
    parametrized variants (`TestClass::test_method[param1]`, etc.).
    """
    waived = _load_waived_test_ids(repo_root)
    if not waived:
        return

    # Split into exact IDs (with params) and prefix IDs (without params)
    exact_waived = set()
    prefix_waived = []
    for wid in waived:
        if '[' in wid:
            exact_waived.add(wid)
        else:
            prefix_waived.append(wid)

    excluded = 0
    for tid in list(result.selected_tests):
        if tid in exact_waived:
            del result.selected_tests[tid]
            result.reasons.pop(tid, None)
            excluded += 1
        elif any(tid.startswith(p + '[') or tid == p
                 for p in prefix_waived):
            del result.selected_tests[tid]
            result.reasons.pop(tid, None)
            excluded += 1
    if excluded:
        import sys
        print(f"Excluded {excluded} currently waived tests (known failures).",
              file=sys.stderr)


def _parse_waive_test_id(line: str) -> str:
    """Extract a test ID from a waives.txt line.

    Waive lines have formats like:
      accuracy/test.py::Class::method SKIP (url)
      full:L40S/accuracy/test.py::Class::method SKIP (url)
      full:sm100/unittest/... SKIP (reason)

    Returns the bare test ID (without SKIP/reason and without full:xxx/ prefix),
    or empty string if the line is not a valid waive entry.
    """
    # Strip SKIP reason
    test_id = re.sub(r'\s+SKIP\s*(\(.*\))?.*$', '', line).strip()
    if not test_id or test_id.startswith('#'):
        return ""

    # Strip "full:<platform>/" prefix
    if test_id.startswith("full:"):
        # e.g. "full:L40S/accuracy/test.py::..." → "accuracy/test.py::..."
        slash_idx = test_id.find('/')
        if slash_idx >= 0:
            test_id = test_id[slash_idx + 1:]
        else:
            return ""

    # Must look like a pytest node ID
    if '::' not in test_id:
        return ""

    return test_id


def _load_durations(repo_root: str = ".") -> dict[str, float]:
    """Load test duration data from .test_durations file."""
    path = Path(repo_root) / "tests" / "integration" / "defs" / ".test_durations"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _get_duration(test_id: str, durations: dict[str, float],
                  default: float) -> float:
    """Get duration for a test, falling back to default for unknown tests."""
    return durations.get(test_id, default)


def apply_time_budget(result: SelectionResult, budget_seconds: float,
                      durations: dict[str, float],
                      test_list_filter: str = None):
    """Drop lowest-value tests until total duration fits within budget.

    Algorithm:
      Phase 0: Drop any single test exceeding 25% of the budget.
      Phase 1: Greedy removal — repeatedly drop the test that loses fewest
         unique features, breaking ties by longest duration.

    Protected tests (representative set, NEW_TEST) are never dropped.
    Directly modified test methods (TEST method=) respect the budget.

    Args:
        test_list_filter: If set, only count duration for tests in this list.
            Tests not in the target list are ignored for budget purposes
            but kept in the result.
    """
    protected_ids = set(REPRESENTATIVE_COVERAGE_SET)
    # Tags that protect tests from both Phase 0 and Phase 1.
    # TEST method= is NOT protected — modified test methods are important
    # but should still respect the time budget.
    _no_drop_tags = {"NEW_TEST"}

    # Determine which tests are in scope for budget trimming
    def in_scope(test_id: str) -> bool:
        if not test_list_filter:
            return True
        entry = result.selected_tests.get(test_id)
        return entry is not None and test_list_filter in entry.test_lists

    def is_protected(test_id: str) -> bool:
        if test_id in protected_ids:
            return True
        reasons = result.reasons.get(test_id, [])
        return any(any(tag in r for tag in _no_drop_tags) for r in reasons)

    # Compute default duration (median of known in-scope durations)
    scope_ids = [tid for tid in result.selected_tests if in_scope(tid)]
    known = [durations[tid] for tid in scope_ids if tid in durations]
    default_dur = sorted(known)[len(known) // 2] if known else 300.0

    def get_dur(tid: str) -> float:
        return _get_duration(tid, durations, default_dur)

    # Phase 0: Drop any single test that exceeds 25% of the budget.
    # These are too expensive regardless of feature value.
    # Only protected tests (NEW_TEST, representative set) survive.
    max_single = budget_seconds * 0.25
    dropped_count = 0
    dropped_time = 0.0

    for tid in list(result.selected_tests):
        if not in_scope(tid) or is_protected(tid):
            continue
        dur = get_dur(tid)
        if dur > max_single:
            dropped_time += dur
            del result.selected_tests[tid]
            result.reasons.pop(tid, None)
            dropped_count += 1

    total = sum(get_dur(tid) for tid in result.selected_tests if in_scope(tid))
    if total <= budget_seconds:
        if dropped_count > 0:
            result._budget_info = {
                'dropped': dropped_count,
                'dropped_time': dropped_time,
                'remaining_time': total,
                'budget': budget_seconds,
            }
        return

    # Phase 1: Greedy removal — drop the test that loses fewest unique
    # features, breaking ties by longest duration (biggest savings).
    while total > budget_seconds:
        # Build feature → test_ids map for current in-scope selection
        feat_coverage: dict[str, set[str]] = {}
        for tid in result.selected_tests:
            if not in_scope(tid):
                continue
            for feat in result.selected_tests[tid].features:
                feat_coverage.setdefault(feat, set()).add(tid)

        best_drop = None
        best_unique = float('inf')
        best_dur = -1.0

        for tid in list(result.selected_tests):
            if not in_scope(tid) or is_protected(tid):
                continue
            dur = get_dur(tid)
            # Count features that ONLY this test covers
            unique_count = sum(
                1 for f in result.selected_tests[tid].features
                if len(feat_coverage.get(f, set())) <= 1
            )
            # Prefer: fewest unique features first, then longest duration
            if (unique_count < best_unique
                    or (unique_count == best_unique and dur > best_dur)):
                best_drop = tid
                best_unique = unique_count
                best_dur = dur

        if best_drop is None:
            break  # Only protected tests remain

        total -= get_dur(best_drop)
        dropped_time += get_dur(best_drop)
        del result.selected_tests[best_drop]
        result.reasons.pop(best_drop, None)
        dropped_count += 1

    if dropped_count > 0:
        # Add a note to the result for reporting
        result._budget_info = {
            'dropped': dropped_count,
            'dropped_time': dropped_time,
            'remaining_time': total,
            'budget': budget_seconds,
        }


def select_tests(database: dict[str, TestEntry],
                 changed_files: list[str],
                 base_ref: str = "",
                 repo_root: str = ".") -> SelectionResult:
    """Main entry: select tests based on changed files.

    Args:
        database: Pre-built test database from parser.build_test_database()
        changed_files: List of changed file paths (relative to repo root)
        base_ref: Git ref that was used to compute the diff (for TEST tier
            class-level narrowing)
        repo_root: Path to the repository root (for loading waives.txt)

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

    # Exclude currently waived tests — they are known failures and running
    # them wastes budget that could cover other tests. Done BEFORE dedup
    # so the greedy coverage algorithms see more available tests and
    # produce better coverage.
    # Note: un-waived tests (removed from waives.txt) are NOT force-selected.
    # Bug closure already requires main-branch verification, so un-waived
    # tests don't need special treatment — they participate via normal rules
    # and weekly full runs provide the safety net.
    _exclude_waived_tests(result, str(repo_root))

    # Deduplicate parametrized variants: keep variants that contribute
    # new parameter tags (backends, configs), drop redundant combos.
    _deduplicate_parametrized(result)

    # Second pass: greedy feature-coverage dedup per class.
    _deduplicate_per_class(result)

    return result


def _tokenize_params(entry) -> set[str]:
    """Extract a set of parameter tags from a test entry for diversity scoring.

    Uses resolved_params (from parametrize decorator parsing) when available,
    falling back to KV params and raw_params tokenization.

    For each parametrize dimension declared on the method, if no value for
    that dimension is found in the tags, a "dim:default" tag is injected so
    that variants using the implicit default are distinguishable from those
    with explicit values. Dimensions come from the actual @parametrize
    decorators in the test source, not from a hardcoded keyword list.
    """
    _TRIVIAL_VALUES = {'true', 'false', '0', '1', '2', '3'}

    tags = set()

    # Use resolved_params (most complete: includes custom-ID param values)
    # or fall back to parsed KV params.
    kv_source = entry.resolved_params or entry.params
    for k, v in kv_source.items():
        v_lower = str(v).lower()
        if v_lower in _TRIVIAL_VALUES:
            # Boolean/trivial values: just use the key as tag.
            # This avoids false diversity between on/off flag combos.
            tags.add(k.lower())
        else:
            # Meaningful values (e.g. CUTLASS, TRTLLM): use "key=value"
            tags.add(f'{k.lower()}={v_lower}')

    # From raw_params — only when resolved_params is empty, since
    # resolved_params already captures all parameter info from the
    # decorator. Raw tokens like "tp8ep8" would create false uniqueness.
    if not entry.resolved_params and entry.raw_params:
        for token in re.split(r'[-=_]', entry.raw_params):
            token = token.strip().lower()
            if token and token not in _TRIVIAL_VALUES:
                tags.add(token)

    # Inject "dim:default" for each declared parametrize dimension
    # that has no value in resolved_params or params.
    all_param_keys = {k.lower() for k in kv_source} if kv_source else set()

    for dim in entry.param_dimensions:
        dim_lower = dim.lower().strip()
        if dim_lower not in all_param_keys:
            tags.add(f'{dim_lower}:default')

    return tags


def _greedy_select_by_diversity(
    test_ids: list[str],
    result: SelectionResult,
    max_keep: int,
) -> set[str]:
    """Select up to max_keep variants that maximize parameter tag coverage.

    Algorithm:
      1. Sort by feature count descending (feature-rich variants first).
      2. Among candidates with the same feature count, greedily pick the one
         that covers the most new (uncovered) parameter tags.
      3. Repeat until max_keep reached.

    This ensures high-feature variants are prioritized, and among equals
    the selection maximizes diversity across parameter combinations.
    """
    # Pre-compute tags and feature counts
    tid_tags: dict[str, set[str]] = {}
    tid_fcnt: dict[str, int] = {}
    for tid in test_ids:
        entry = result.selected_tests[tid]
        tid_tags[tid] = _tokenize_params(entry)
        tid_fcnt[tid] = len(entry.features)

    # Group by feature count (descending)
    by_fcnt: dict[int, list[str]] = {}
    for tid in test_ids:
        by_fcnt.setdefault(tid_fcnt[tid], []).append(tid)

    keep = []
    covered_tags: set[str] = set()

    for fcnt in sorted(by_fcnt.keys(), reverse=True):
        if len(keep) >= max_keep:
            break
        candidates = list(by_fcnt[fcnt])
        while candidates and len(keep) < max_keep:
            # Pick the candidate that adds the most new tags
            best_tid = max(
                candidates,
                key=lambda tid: len(tid_tags[tid] - covered_tags),
            )
            covered_tags |= tid_tags[best_tid]
            keep.append(best_tid)
            candidates.remove(best_tid)

    return set(keep)


def _greedy_select_by_features(
    test_ids: list[str],
    result: SelectionResult,
) -> set[str]:
    """Select tests greedily until no remaining test adds new features.

    Algorithm:
      1. Sort candidates by feature count descending.
      2. Pick the one with the most features.
      3. Iteratively pick the candidate whose features add the most
         uncovered features.
      4. Stop when no candidate adds any new feature.

    Always returns at least 1 test (the one with the most features).
    """
    remaining = list(test_ids)
    keep = []
    covered: set[str] = set()

    while remaining:
        # Pick the candidate that adds the most new features
        best_tid = max(
            remaining,
            key=lambda tid: len(
                result.selected_tests[tid].features - covered),
        )
        new_features = result.selected_tests[best_tid].features - covered
        if not new_features and keep:
            # No new features can be added — stop
            break
        covered |= result.selected_tests[best_tid].features
        keep.append(best_tid)
        remaining.remove(best_tid)

    return set(keep)


def _greedy_select_by_tags(
    test_ids: list[str],
    result: SelectionResult,
) -> set[str]:
    """Select parametrized variants greedily until no new tags can be added.

    Algorithm:
      1. Sort by feature count descending (feature-rich first).
      2. Pick the best, then iteratively pick whichever adds the most
         new parameter tags.
      3. Stop when no remaining variant adds any new tag.

    This keeps variants that represent genuinely different configurations
    (e.g. different backends) while dropping those that are just different
    flag combinations of the same things.
    """
    remaining = list(test_ids)
    keep = []
    covered: set[str] = set()

    # Pre-compute tags
    tid_tags = {tid: _tokenize_params(result.selected_tests[tid])
                for tid in test_ids}

    # First pick: highest feature count, break ties by most tags
    remaining.sort(
        key=lambda tid: (
            len(result.selected_tests[tid].features),
            len(tid_tags[tid]),
        ),
        reverse=True,
    )

    while remaining:
        best_tid = max(
            remaining,
            key=lambda tid: len(tid_tags[tid] - covered),
        )
        new_tags = tid_tags[best_tid] - covered
        if not new_tags and keep:
            break
        covered |= tid_tags[best_tid]
        keep.append(best_tid)
        remaining.remove(best_tid)

    return set(keep)


def _deduplicate_parametrized(result: SelectionResult):
    """Reduce parametrized variants per (test_class, test_method).

    Deduplicates tests whose reasons are ALL from dedup-eligible tiers.
    Tests in the representative set, or selected by CORE/FALLBACK/NEW_TEST
    tiers, are never removed.

    Selection uses greedy tag coverage: keep adding variants as long as they
    contribute new parameter tags (e.g. a new backend, a new flag value).
    Stop when no remaining variant adds anything new.
    """
    # Only CORE (representative set), FALLBACK, and NEW_TEST are exempt.
    # NEW_TEST entries must all run to verify newly added tests pass.
    _no_dedup_tiers = {"Tier0:CORE", "FALLBACK", "NEW_TEST"}

    # Protect: never remove representative set entries
    protected_ids = set(REPRESENTATIVE_COVERAGE_SET)

    # Group candidates by (test_class, test_method)
    groups: dict[tuple[str, str], list[str]] = {}
    for test_id, entry in result.selected_tests.items():
        if test_id in protected_ids:
            continue
        # Skip dedup if ANY reason is from a non-dedup tier
        reasons = result.reasons.get(test_id, [])
        if not reasons:
            continue
        if any(any(nd in r for nd in _no_dedup_tiers) for r in reasons):
            continue
        key = (entry.test_class, entry.test_method)
        groups.setdefault(key, []).append(test_id)

    for (cls, method), test_ids in groups.items():
        if len(test_ids) <= 1:
            continue

        n = len(test_ids)
        keep_ids = _greedy_select_by_tags(test_ids, result)
        remove_ids = set(test_ids) - keep_ids

        for tid in remove_ids:
            del result.selected_tests[tid]
            result.reasons.pop(tid, None)

        for tid in keep_ids:
            result.reasons[tid].append(
                f"DEDUP: kept {len(keep_ids)}/{n} variants of "
                f"{cls}::{method}"
            )


def _deduplicate_per_class_feature_coverage(
    groups: dict[str, list[str]],
    result: SelectionResult,
    label: str,
):
    """Greedy feature-coverage dedup at the METHOD level within each class.

    For each class, group test IDs by method, compute the feature union per
    method, then greedily keep methods until no remaining method adds a new
    feature.  All surviving variants of a kept method are retained.
    """
    for cls, test_ids in groups.items():
        method_tids: dict[str, list[str]] = {}
        for tid in test_ids:
            method = result.selected_tests[tid].test_method
            method_tids.setdefault(method, []).append(tid)

        if len(method_tids) <= 1:
            continue

        # Compute union of features per method
        method_features: dict[str, set[str]] = {}
        for method, tids in method_tids.items():
            union = set()
            for tid in tids:
                union |= result.selected_tests[tid].features
            method_features[method] = union

        # Greedy: keep methods until no new features
        remaining = list(method_tids.keys())
        keep_methods = []
        covered: set[str] = set()

        while remaining:
            best = max(remaining,
                       key=lambda m: len(method_features[m] - covered))
            new_feats = method_features[best] - covered
            if not new_feats and keep_methods:
                break
            covered |= method_features[best]
            keep_methods.append(best)
            remaining.remove(best)

        n_methods = len(method_tids)
        remove_methods = set(method_tids.keys()) - set(keep_methods)
        for method in remove_methods:
            for tid in method_tids[method]:
                del result.selected_tests[tid]
                result.reasons.pop(tid, None)

        for method in keep_methods:
            for tid in method_tids[method]:
                result.reasons[tid].append(
                    f"DEDUP-CLASS: kept {len(keep_methods)}/{n_methods} "
                    f"methods of {cls} ({label})"
                )


def _deduplicate_per_class(result: SelectionResult):
    """Second-pass dedup: reduce methods per TestClass.

    After the first pass (1 variant per method), there can still be many
    different methods per TestClass.

    Strategy varies by tier:
      - MODEL tier: greedy feature coverage — keep methods that contribute
        new features, drop methods whose features are already covered.
      - TEST tier: no dedup — selection is already method-level precise.
      - Other tiers (DEFAULT_ON, OPT_IN): greedy feature coverage.

    Exempt from dedup: CORE, FALLBACK, NEW_TEST, TEST.
    """
    protected_ids = set(REPRESENTATIVE_COVERAGE_SET)
    _no_dedup = {"Tier0:CORE", "FALLBACK", "NEW_TEST"}

    # Collect dedup-eligible tests into one group
    dedup_groups: dict[str, list[str]] = {}

    for test_id, entry in result.selected_tests.items():
        if test_id in protected_ids:
            continue
        reasons = result.reasons.get(test_id, [])
        if not reasons:
            continue
        if any(any(nd in r for nd in _no_dedup) for r in reasons):
            continue
        # TEST tier: skip dedup (method-level selection is already precise)
        if any(r.startswith("TEST ") for r in reasons):
            continue

        dedup_groups.setdefault(entry.test_class, []).append(test_id)

    # All tiers use greedy feature coverage
    _deduplicate_per_class_feature_coverage(
        dedup_groups, result, "feature coverage")


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

    # Pick the most specific reason across ALL reasons, not just the first.
    # Priority: MODEL > TEST > OPT_IN > DEFAULT_ON > CORE > FALLBACK
    # This ensures e.g. an e2e test matched by both DEFAULT_ON (kv_cache)
    # and MODEL (arch=qwen) gets grouped under MODEL.
    for r in reasons:
        if r.startswith("MODEL"):
            m = re.search(r'arch=(\S+)', r)
            arch = m.group(1) if m else "unknown"
            return f"MODEL: {arch}"

    for r in reasons:
        if r.startswith("TEST method="):
            # "TEST method=TestGPTOSS::test_w4_1gpu modified in ..."
            method_id = r.split("method=")[1].split(" ")[0]
            cls = method_id.split("::")[0]
            return f"TEST: {cls} modified"
        if r.startswith("TEST class="):
            cls = r.split("class=")[1].split(" ")[0]
            return f"TEST: {cls} modified"
        if r.startswith("TEST file="):
            return "TEST: test file modified"

    for r in reasons:
        if "Tier2:OPT_IN" in r:
            m = re.search(r'feature=(\S+)', r)
            feat = m.group(1) if m else "unknown"
            return f"OPT_IN: {feat}"

    for r in reasons:
        if "Tier1:DEFAULT_ON" in r:
            m = re.search(r'feature=(\S+)', r)
            feat = m.group(1) if m else "unknown"
            return f"DEFAULT_ON: {feat}"

    for r in reasons:
        if "Tier0:CORE" in r:
            return "Representative coverage set (core infrastructure change)"

    for r in reasons:
        if r.startswith("FALLBACK"):
            return "FALLBACK: unmatched code"

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

    # Sort groups by priority (matching _primary_reason order), then by size
    _GROUP_PRIORITY = {
        "NEW_TEST": 0,
        "MODEL": 1,
        "TEST": 2,
        "OPT_IN": 3,
        "DEFAULT_ON": 4,
        "Representative": 5,
        "CORE": 5,
        "FALLBACK": 6,
    }

    def group_sort_key(item):
        key, tests = item
        prefix = key.split(":")[0].split(" ")[0]
        priority = _GROUP_PRIORITY.get(prefix, 7)
        return (priority, -len(tests), key)

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


def generate_maintenance_warnings(
    result: SelectionResult,
    database: dict[str, TestEntry],
) -> list[str]:
    """Check for conditions that indicate impact_rules or parser config
    needs updating.  Returns a list of human-readable warning strings.

    Intended to be printed to stderr so maintainers notice issues early.
    """
    from .impact_rules import TESTCLASS_TO_ARCH

    warnings: list[str] = []

    # 1. Unmatched source files → may need new impact rules
    code_exts = {'.py', '.cpp', '.h', '.cu', '.cuh', '.cc'}
    unmatched_code = [
        f for f in result.unmatched_files
        if any(f.endswith(ext) for ext in code_exts)
        and not f.startswith('.')       # skip dotfiles
        and 'test' not in f.lower()     # skip test files (covered by TEST tier)
    ]
    if unmatched_code:
        warnings.append(
            f"[RULE] {len(unmatched_code)} source file(s) matched no impact rule "
            f"(FALLBACK used). Consider adding rules to impact_rules.py:")
        for f in unmatched_code[:10]:
            warnings.append(f"  - {f}")
        if len(unmatched_code) > 10:
            warnings.append(f"  ... and {len(unmatched_code) - 10} more")

    # 2. Test classes not in TESTCLASS_TO_ARCH → need arch mapping
    unknown_classes: dict[str, set[str]] = {}  # class → set of test_files
    for entry in database.values():
        if entry.test_class and not entry.arch:
            if entry.test_class not in TESTCLASS_TO_ARCH:
                unknown_classes.setdefault(
                    entry.test_class, set()).add(entry.test_file)
    if unknown_classes:
        warnings.append(
            f"[ARCH] {len(unknown_classes)} test class(es) have no architecture "
            f"mapping in TESTCLASS_TO_ARCH:")
        for cls in sorted(unknown_classes)[:10]:
            files = ', '.join(sorted(unknown_classes[cls]))
            warnings.append(f"  - {cls} ({files})")
        if len(unknown_classes) > 10:
            warnings.append(f"  ... and {len(unknown_classes) - 10} more")

    # 3. Architectures in database but not covered by REPRESENTATIVE_COVERAGE_SET
    db_archs = {e.arch for e in database.values() if e.arch}
    rep_archs = set()
    for tid in REPRESENTATIVE_COVERAGE_SET:
        if tid in database:
            if database[tid].arch:
                rep_archs.add(database[tid].arch)
    missing_archs = db_archs - rep_archs
    if missing_archs:
        warnings.append(
            f"[REP] {len(missing_archs)} architecture(s) have no test in "
            f"REPRESENTATIVE_COVERAGE_SET:")
        for arch in sorted(missing_archs):
            count = sum(1 for e in database.values() if e.arch == arch)
            warnings.append(f"  - {arch} ({count} tests in database)")

    # 4. Tests with zero features (may indicate missing extraction patterns)
    no_features = [
        tid for tid, e in database.items()
        if not e.features and e.test_class  # skip module-level tests
    ]
    if len(no_features) > 20:
        warnings.append(
            f"[FEAT] {len(no_features)} tests have no extracted features. "
            f"Feature extraction patterns may need updating.")

    return warnings
