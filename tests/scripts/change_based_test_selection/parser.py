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
Test entry parser for change-based test selection.

Parses test lists and test definition source files to build a structured
database of test entries with model, feature, and GPU requirement metadata.

Three levels of analysis:
  L1 (class-level AST): MODEL_NAME, MODEL_PATH class attributes
  L2 (method-level AST): decorators (skip_pre_*, skip_less_device, parametrize)
  L3 (method body AST): Config class constructor calls (KvCacheConfig, etc.)
"""

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TestEntry:
    """A single test case with all extracted metadata."""

    test_id: str  # full pytest node ID from test list
    test_file: str  # relative test file path
    test_class: str  # class name (e.g. TestDeepSeekV3Lite)
    test_method: str  # method name without params (e.g. test_nvfp4)
    params: dict[str, str] = field(
        default_factory=dict)  # parametrize values from [...]
    raw_params: str = ""  # original bracket content for non-KV params

    # L1: class-level
    model_name: str = ""  # HF model ID (e.g. deepseek-ai/DeepSeek-V3-Lite)
    arch: str = ""  # architecture key (e.g. deepseek_v2)
    source_dirs: list[str] = field(
        default_factory=list)  # model source directories

    # L2: decorator-level
    min_sm: int = 0  # minimum SM version (0=any, 89=ada, 90=hopper, 100=blackwell)
    max_sm: int = 999  # maximum SM version (for skip_post_*)
    min_gpu_count: int = 1  # from @skip_less_device
    min_gpu_memory: int = 0  # from @skip_less_device_memory (MB)

    # L3: method body config classes
    config_classes: set[str] = field(
        default_factory=set)  # e.g. {"KvCacheConfig", "Eagle3DecodingConfig"}

    # features extracted from method name + params + config classes
    features: set[str] = field(default_factory=set)

    # Parametrize dimension names (union of all @parametrize decorators)
    param_dimensions: list[str] = field(default_factory=list)
    # Resolved parameter values from custom-ID parametrize decorators
    # Maps param_name -> value_str (e.g. {"moe_backend": "CUTLASS"})
    resolved_params: dict[str, str] = field(default_factory=dict)
    # Model names/paths extracted from parametrize values
    # (for module-level functions like test_e2e.py that don't have class attrs)
    model_names: list[str] = field(default_factory=list)

    # which test lists contain this entry
    test_lists: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        d = {
            "test_id": self.test_id,
            "test_file": self.test_file,
            "test_class": self.test_class,
            "test_method": self.test_method,
            "params": self.params,
            "raw_params": self.raw_params,
            "model_name": self.model_name,
            "arch": self.arch,
            "source_dirs": self.source_dirs,
            "min_sm": self.min_sm,
            "max_sm": self.max_sm,
            "min_gpu_count": self.min_gpu_count,
            "min_gpu_memory": self.min_gpu_memory,
            "config_classes": sorted(self.config_classes),
            "features": sorted(self.features),
            "param_dimensions": self.param_dimensions,
            "resolved_params": self.resolved_params,
            "model_names": self.model_names,
            "test_lists": sorted(self.test_lists),
        }
        return d


# --- Test List Parsing ---

# Regex to parse a pytest node ID:
#   file.py::ClassName::method_name[param1=val1-param2=val2]
_NODE_RE = re.compile(
    r'^(?P<file>[^:]+)::(?P<class>\w+)::(?P<method>\w+)(?:\[(?P<params>[^\]]*)\])?$'
)

# Regex for module-level test functions (no class):
#   file.py::function_name[param1-param2]
_NODE_FUNC_RE = re.compile(
    r'^(?P<file>[^:]+)::(?P<method>\w+)(?:\[(?P<params>[^\]]*)\])?$'
)

# Regex to parse key=value pairs from param string
_PARAM_KV_RE = re.compile(r'(?P<key>\w+)=(?P<val>[^-\]]+)')


def parse_test_list(path: Path) -> list[dict]:
    """Parse a test list .txt file into raw test entries.

    Each line is a pytest node ID. Lines starting with # are comments.
    Strips TIMEOUT/ISOLATION markers.
    """
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Strip markers like TIMEOUT (90), ISOLATION
            line = re.sub(r'\s+TIMEOUT\s*\(\d+\)', '', line)
            line = re.sub(r'\s+ISOLATION\b', '', line)
            line = line.strip()
            if not line:
                continue

            m = _NODE_RE.match(line)
            if m:
                raw_params = m.group('params') or ''
                params = {}
                if raw_params:
                    for kv in _PARAM_KV_RE.finditer(raw_params):
                        params[kv.group('key')] = kv.group('val')
                entries.append({
                    'test_id': line,
                    'test_file': m.group('file'),
                    'test_class': m.group('class'),
                    'test_method': m.group('method'),
                    'params': params,
                    'raw_params': raw_params,
                    'test_list': path.stem,
                })
            else:
                # Module-level test functions (e.g. test_e2e.py::test_foo[...])
                m2 = _NODE_FUNC_RE.match(line)
                if m2:
                    raw_params = m2.group('params') or ''
                    params = {}
                    if raw_params:
                        for kv in _PARAM_KV_RE.finditer(raw_params):
                            params[kv.group('key')] = kv.group('val')
                    entries.append({
                        'test_id': line,
                        'test_file': m2.group('file'),
                        'test_class': '',
                        'test_method': m2.group('method'),
                        'params': params,
                        'raw_params': raw_params,
                        'test_list': path.stem,
                    })
                else:
                    # Completely non-standard format
                    entries.append({
                        'test_id': line,
                        'test_file':
                        line.split('::')[0] if '::' in line else line,
                        'test_class': '',
                        'test_method':
                        line.split('::')[-1].split('[')[0]
                        if '::' in line else '',
                        'params': {},
                        'raw_params': '',
                        'test_list': path.stem,
                    })
    return entries


# --- AST Parsing of Test Definitions ---

# Decorator names that indicate SM requirements
_SM_DECORATORS = {
    'skip_pre_ada': ('min_sm', 89),
    'skip_pre_hopper': ('min_sm', 90),
    'skip_pre_blackwell': ('min_sm', 100),
    'skip_post_blackwell': ('max_sm', 99),
    'skip_post_blackwell_ultra': ('max_sm', 102),
    'skip_no_hopper': ('min_sm', 90),  # also sets max_sm=90
    'skip_no_sm120': ('min_sm', 120),
}

# Known Config classes whose presence indicates feature usage
_KNOWN_CONFIG_CLASSES = {
    'KvCacheConfig',
    'CudaGraphConfig',
    'TorchCompileConfig',
    'Eagle3DecodingConfig',
    'MTPDecodingConfig',
    'PARDDecodingConfig',
    'NGramDecodingConfig',
    'SADecodingConfig',
    'AutoDecodingConfig',
    'MoeConfig',
    'MoeLoadBalancerConfig',
    'SchedulerConfig',
    'SamplingParams',
    'QuantConfig',
    'DeepSeekSparseAttentionConfig',
    'RocketSparseAttentionConfig',
    'SkipSoftmaxAttentionConfig',
}


def _extract_decorator_name(decorator_node: ast.expr) -> tuple[str, list]:
    """Extract decorator name and arguments from AST node."""
    if isinstance(decorator_node, ast.Name):
        return decorator_node.id, []
    elif isinstance(decorator_node, ast.Attribute):
        # e.g. pytest.mark.skip_less_device
        parts = []
        node = decorator_node
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return '.'.join(reversed(parts)), []
    elif isinstance(decorator_node, ast.Call):
        name, _ = _extract_decorator_name(decorator_node.func)
        args = []
        for arg in decorator_node.args:
            if isinstance(arg, ast.Constant):
                args.append(arg.value)
            elif isinstance(arg, (ast.List, ast.Tuple)):
                args.append([
                    elt.value for elt in arg.elts
                    if isinstance(elt, ast.Constant)
                ])
        return name, args
    return '', []


def _extract_call_names(node: ast.AST) -> set[str]:
    """Walk an AST subtree and find all function/class call names."""
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                names.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                names.add(child.func.attr)
    return names


@dataclass
class ClassInfo:
    """AST-extracted info about a test class."""

    name: str
    model_name: str = ""
    model_path: str = ""
    # class-level decorators (inherited by all methods)
    min_sm: int = 0
    max_sm: int = 999
    min_gpu_count: int = 1
    min_gpu_memory: int = 0


@dataclass
class FunctionInfo:
    """AST-extracted info about a module-level test function."""

    name: str
    min_sm: int = 0
    max_sm: int = 999
    min_gpu_count: int = 1
    min_gpu_memory: int = 0
    config_classes: set[str] = field(default_factory=set)
    parametrize_info: list['ParametrizeInfo'] = field(default_factory=list)


@dataclass
class ParametrizeInfo:
    """Parsed info from a single @parametrize decorator."""

    param_names: list[str]  # e.g. ["tp_size", "pp_size", "moe_backend"]
    # For custom-ID parametrize: maps id_string -> {param_name: value_str}
    # Empty when IDs are auto-generated (parametrize_with_ids).
    id_to_values: dict[str, dict[str, str]] = field(default_factory=dict)
    # All value sets from the decorator (list of {param_name: value_str}).
    # Always populated regardless of custom IDs.
    all_values: list[dict[str, str]] = field(default_factory=list)


@dataclass
class MethodInfo:
    """AST-extracted info about a test method."""

    name: str
    min_sm: int = 0
    max_sm: int = 999
    min_gpu_count: int = 1
    min_gpu_memory: int = 0
    config_classes: set[str] = field(default_factory=set)
    # All parametrize decorators on this method
    parametrize_info: list[ParametrizeInfo] = field(default_factory=list)


def _parse_function_decorators(func_node: ast.AST) -> tuple[int, int, int, int]:
    """Extract SM and GPU requirements from a function's decorators.

    Returns (min_sm, max_sm, min_gpu_count, min_gpu_memory).
    """
    min_sm, max_sm, min_gpu_count, min_gpu_memory = 0, 999, 1, 0
    for dec in func_node.decorator_list:
        dec_name, dec_args = _extract_decorator_name(dec)
        short_name = dec_name.split('.')[-1]
        if short_name in _SM_DECORATORS:
            attr, val = _SM_DECORATORS[short_name]
            if attr == 'min_sm':
                min_sm = max(min_sm, val)
            else:
                max_sm = min(max_sm, val)
        elif short_name == 'skip_less_device' and dec_args:
            min_gpu_count = max(min_gpu_count, dec_args[0])
        elif short_name == 'skip_less_device_memory' and dec_args:
            min_gpu_memory = max(min_gpu_memory, dec_args[0])
    return min_sm, max_sm, min_gpu_count, min_gpu_memory


def _extract_parametrize_values(elt: ast.AST,
                                expected_count: int) -> list[str]:
    """Extract parameter values from a single parametrize entry AST node.

    Handles pytest.param(...), tuples, lists, and single constants.
    Returns list of string representations, or empty list on failure.
    """
    raw_vals: list[str] = []

    def _ast_to_str(node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        return '?'

    if isinstance(elt, ast.Call):
        # pytest.param(val1, val2, ...)
        for a in elt.args:
            raw_vals.append(_ast_to_str(a))
    elif isinstance(elt, (ast.Tuple, ast.List)):
        for a in elt.elts:
            raw_vals.append(_ast_to_str(a))
    elif isinstance(elt, ast.Constant):
        raw_vals.append(str(elt.value))

    return raw_vals


def _parse_parametrize_decorator(
        dec_name: str, dec_node: ast.Call) -> Optional[ParametrizeInfo]:
    """Extract ParametrizeInfo from a parametrize decorator AST node.

    Handles both:
      @pytest.mark.parametrize("a,b", [...], ids=[...])
      @parametrize_with_ids("a,b", [...])
    """
    if not isinstance(dec_node, ast.Call):
        return None

    args = dec_node.args
    if len(args) < 2:
        return None

    # First arg: param names (string)
    if not isinstance(args[0], ast.Constant) or not isinstance(
            args[0].value, str):
        return None
    param_names = [n.strip() for n in args[0].value.split(',')]

    # For parametrize_with_ids, IDs are auto-generated as "name=value"
    # so the KV parser already handles them. Still extract all_values
    # for model name extraction.
    is_with_ids = 'parametrize_with_ids' in dec_name

    # Extract values from the second arg (list of pytest.param() or tuples)
    values_node = args[1]
    all_values: list[dict[str, str]] = []
    if isinstance(values_node, (ast.List, ast.Tuple)):
        for elt in values_node.elts:
            raw_vals = _extract_parametrize_values(elt, len(param_names))
            if raw_vals and len(raw_vals) == len(param_names):
                all_values.append(dict(zip(param_names, raw_vals)))

    if is_with_ids:
        return ParametrizeInfo(param_names=param_names,
                               all_values=all_values)

    # For pytest.mark.parametrize with custom ids:
    # Extract ids kwarg and values to build id→values mapping.
    ids_node = None
    for kw in dec_node.keywords:
        if kw.arg == 'ids':
            ids_node = kw.value
            break

    if ids_node is None or not isinstance(ids_node, (ast.List, ast.Tuple)):
        # No custom ids — IDs are auto-generated by pytest
        return ParametrizeInfo(param_names=param_names,
                               all_values=all_values)

    # Extract ID strings
    id_strings = []
    for elt in ids_node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            id_strings.append(elt.value)
        else:
            id_strings.append(None)

    # Build id→values mapping from all_values + id_strings
    id_to_values: dict[str, dict[str, str]] = {}
    for idx, val_dict in enumerate(all_values):
        if idx < len(id_strings) and id_strings[idx] is not None:
            id_to_values[id_strings[idx]] = val_dict

    return ParametrizeInfo(param_names=param_names,
                           id_to_values=id_to_values,
                           all_values=all_values)


def parse_test_definition(path: Path) -> dict[str, tuple[ClassInfo, list[MethodInfo]]]:
    """Parse a test definition .py file using AST.

    Returns a dict mapping class_name -> (ClassInfo, [MethodInfo, ...]).
    """
    with open(path) as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return {}

    result = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # L1: Extract class-level attributes
        cls_info = ClassInfo(name=node.name)

        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target,
                                  ast.Name) and isinstance(item.value,
                                                           ast.Constant):
                        if target.id == 'MODEL_NAME':
                            cls_info.model_name = item.value.value
                        elif target.id == 'MODEL_PATH':
                            cls_info.model_path = str(item.value.value)
                    # Handle f-string MODEL_PATH
                    elif isinstance(target,
                                    ast.Name) and target.id == 'MODEL_PATH':
                        if isinstance(item.value, ast.JoinedStr):
                            # Extract string parts from f-string
                            parts = []
                            for v in item.value.values:
                                if isinstance(v, ast.Constant):
                                    parts.append(str(v.value))
                            cls_info.model_path = ''.join(parts)

        # Class-level decorators
        for dec in node.decorator_list:
            dec_name, dec_args = _extract_decorator_name(dec)
            short_name = dec_name.split('.')[-1]
            if short_name in _SM_DECORATORS:
                attr, val = _SM_DECORATORS[short_name]
                if attr == 'min_sm':
                    cls_info.min_sm = max(cls_info.min_sm, val)
                else:
                    cls_info.max_sm = min(cls_info.max_sm, val)
            elif short_name == 'skip_less_device' and dec_args:
                cls_info.min_gpu_count = max(cls_info.min_gpu_count,
                                             dec_args[0])
            elif short_name == 'skip_less_device_memory' and dec_args:
                cls_info.min_gpu_memory = max(cls_info.min_gpu_memory,
                                              dec_args[0])

        # L2 + L3: Extract method-level info
        methods = []
        for item in node.body:
            if not isinstance(item,
                              (ast.FunctionDef,
                               ast.AsyncFunctionDef)):
                continue
            if not item.name.startswith('test_'):
                continue

            method = MethodInfo(name=item.name)
            method.min_sm = cls_info.min_sm
            method.max_sm = cls_info.max_sm
            method.min_gpu_count = cls_info.min_gpu_count
            method.min_gpu_memory = cls_info.min_gpu_memory

            # Method decorators
            for dec in item.decorator_list:
                dec_name, dec_args = _extract_decorator_name(dec)
                short_name = dec_name.split('.')[-1]
                if short_name in _SM_DECORATORS:
                    attr, val = _SM_DECORATORS[short_name]
                    if attr == 'min_sm':
                        method.min_sm = max(method.min_sm, val)
                    else:
                        method.max_sm = min(method.max_sm, val)
                elif short_name == 'skip_less_device' and dec_args:
                    method.min_gpu_count = max(method.min_gpu_count,
                                               dec_args[0])
                elif short_name == 'skip_less_device_memory' and dec_args:
                    method.min_gpu_memory = max(method.min_gpu_memory,
                                                dec_args[0])

                # Extract parametrize info
                if short_name in ('parametrize', 'parametrize_with_ids'):
                    pinfo = _parse_parametrize_decorator(dec_name, dec)
                    if pinfo:
                        method.parametrize_info.append(pinfo)

            # L3: Scan method body for Config class instantiations
            call_names = _extract_call_names(item)
            method.config_classes = call_names & _KNOWN_CONFIG_CLASSES

            methods.append(method)

        if methods:
            result[node.name] = (cls_info, methods)

    return result


def parse_module_functions(path: Path) -> dict[str, FunctionInfo]:
    """Parse module-level test functions from a .py file using AST.

    Returns a dict mapping function_name -> FunctionInfo.
    Used for test files like test_e2e.py where tests are not inside classes.
    """
    with open(path) as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return {}

    functions = {}

    # Only look at top-level nodes (not nested inside classes)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith('test_'):
            continue

        min_sm, max_sm, min_gpu_count, min_gpu_memory = \
            _parse_function_decorators(node)

        func_info = FunctionInfo(
            name=node.name,
            min_sm=min_sm,
            max_sm=max_sm,
            min_gpu_count=min_gpu_count,
            min_gpu_memory=min_gpu_memory,
        )

        # L3: Scan function body for Config class instantiations
        call_names = _extract_call_names(node)
        func_info.config_classes = call_names & _KNOWN_CONFIG_CLASSES

        # Extract parametrize info from decorators
        for dec in node.decorator_list:
            dec_name, _ = _extract_decorator_name(dec)
            short_name = dec_name.split('.')[-1]
            if short_name in ('parametrize', 'parametrize_with_ids'):
                pinfo = _parse_parametrize_decorator(dec_name, dec)
                if pinfo:
                    func_info.parametrize_info.append(pinfo)

        functions[node.name] = func_info

    return functions


# --- Feature Extraction ---

# Features extracted from test method name patterns
_METHOD_FEATURE_PATTERNS = {
    # Quantization
    'fp8': r'(?:^|_)fp8(?:$|_|[^a-z])',
    'nvfp4': r'nvfp4',
    'fp4': r'(?:^|_)fp4(?:$|_|[^a-z])',
    'w4': r'(?:^|_)w4(?:$|_|[^a-z])',
    'w4a8': r'w4a8',
    'w4a16': r'w4a16',
    'mxfp4': r'mxfp4',
    'block_scales': r'block_scales?',
    # Speculative decoding
    'eagle3': r'eagle3',
    'ngram': r'ngram',
    'pard': r'pard',
    'mtp': r'mtp',
    'suffix_automaton': r'suffix_automaton',
    'auto_spec_decode': r'auto_spec_decode',
    '2_model': r'2_model',
    # Attention & KV cache
    'vswa': r'vswa',
    'reuse': r'(?:^|_)reuse(?:$|_)',
    'no_kv_cache_reuse': r'no_kv_cache_reuse',
    'dsa': r'(?:^|_)dsa(?:$|_)',
    'host_cache': r'host_cache',
    'skip_softmax': r'skip_softmax',
    'chunked_prefill': r'chunked_prefill',
    # Decoding & sampling
    'guided_decoding': r'guided_decoding',
    'beam_search': r'beam_search',
    'streaming': r'streaming',
    'return_logits': r'return_logits',
    # Parallelism & scheduling
    'parallelism': r'parallelism',
    'python_scheduler': r'python_scheduler',
    # Infrastructure
    'lora': r'lora',
    'eplb': r'eplb',
    'batch_waiting': r'batch_waiting',
    'dummy_load': r'dummy_load',
    'piecewise_cuda_graph': r'piecewise_cuda_graph',
    'mixed_precision': r'mixed_precision',
    'cuda_graph_padding': r'cuda_graph_padding',
    'corner_case': r'corner_case',
    'long_rope': r'long_rope',
    'stress': r'(?:^|_)stress(?:$|_)',
}

# Features extracted from param keys/values
_PARAM_FEATURES = {
    'cuda_graph': ['cuda_graph', 'enable_cuda_graph'],
    'overlap_scheduler': ['overlap_scheduler', 'disable_overlap_scheduler'],
    'torch_compile': ['torch_compile'],
    'attention_dp': ['attention_dp'],
    'mtp': ['mtp_nextn', 'mtp'],
    'fp8kv': ['fp8kv'],
    'chunked_prefill': ['enable_chunked_prefill'],
    'eagle3': ['eagle3_one_model'],
    'xgrammar': [],  # detected from param value
    'llguidance': [],  # detected from param value
    'moe_backend': ['moe_backend'],
    'sampler_async_worker': ['sampler_async_worker'],
    'v2_kv_cache': ['v2_kv_cache'],
}

# Config class → feature name mapping
_CONFIG_FEATURES = {
    'Eagle3DecodingConfig': 'eagle3',
    'MTPDecodingConfig': 'mtp',
    'PARDDecodingConfig': 'pard',
    'NGramDecodingConfig': 'ngram',
    'SADecodingConfig': 'suffix_automaton',
    'AutoDecodingConfig': 'auto_spec_decode',
    'CudaGraphConfig': 'cuda_graph',
    'TorchCompileConfig': 'torch_compile',
    'MoeConfig': 'moe',
    'MoeLoadBalancerConfig': 'eplb',
    'SchedulerConfig': 'scheduler_config',
    'DeepSeekSparseAttentionConfig': 'sparse_attention',
    'RocketSparseAttentionConfig': 'rocket_sparse_attention',
    'SkipSoftmaxAttentionConfig': 'skip_softmax_attention',
}


def extract_features(test_method: str, params: dict,
                     config_classes: set[str],
                     raw_params: str = "") -> set[str]:
    """Extract feature tags from method name, params, and config classes."""
    features = set()

    # From method name
    for feat, pattern in _METHOD_FEATURE_PATTERNS.items():
        if re.search(pattern, test_method):
            features.add(feat)

    # From param keys
    for feat, keys in _PARAM_FEATURES.items():
        for key in keys:
            if key in params:
                val = params[key]
                # Features with boolean-like values: only count if enabled
                if val.lower() in ('true', '1'):
                    features.add(feat)
                elif val.lower() not in ('false', '0'):
                    features.add(feat)
                    features.add(f'{feat}:{val}')

    # From param values (detect guided decoding backends)
    all_param_vals = ' '.join(params.values()).lower()
    if 'xgrammar' in all_param_vals:
        features.add('xgrammar')
    if 'llguidance' in all_param_vals:
        features.add('llguidance')
    if 'flashinfer' in all_param_vals or 'FLASHINFER' in ' '.join(
            params.values()):
        features.add('attn_flashinfer')
    if 'trtllm' in all_param_vals and 'attn_backend' in params:
        features.add('attn_trtllm')

    # From raw_params (non-KV param strings, e.g. stress test labels)
    if raw_params and not params:
        raw_lower = raw_params.lower()
        # Apply method feature patterns to raw_params too
        for feat, pattern in _METHOD_FEATURE_PATTERNS.items():
            if re.search(pattern, raw_lower):
                features.add(feat)
        # Detect TP scale from raw params (e.g. "DeepSeek-V3_tp8")
        tp_match = re.search(r'_tp(\d+)', raw_params)
        if tp_match:
            features.add(f'tp{tp_match.group(1)}')

    # From config classes (L3)
    for cls_name, feat in _CONFIG_FEATURES.items():
        if cls_name in config_classes:
            features.add(feat)

    # Infer GPU scale from method name
    for n in [2, 4, 8, 16]:
        if f'_{n}gpu' in test_method or f'_{n}_gpu' in test_method:
            features.add(f'{n}gpu')
            break
        if f'tp{n}' in test_method:
            features.add(f'tp{n}')
    if 'multi_gpu' in test_method or 'multi_nodes' in test_method:
        features.add('multi_gpu')

    # Baseline tests (auto_dtype, bfloat16 with no other features) — tag as
    # "baseline" so they still have at least one feature for dedup/budget
    if not features and re.match(
            r'test_(auto_dtype|bf16|bfloat16)$', test_method):
        features.add('baseline')

    return features


# --- Main Build Function ---


# Parametrize param names that contain model identifiers
_MODEL_PARAM_NAMES = {
    'model_name', 'model_path', 'model_dir', 'model_subdir',
    'model_root', 'eagle_model_path', 'draft_model_dir',
    'llama_model_root',
}


def _extract_model_names_for_variant(
        parametrize_info: list['ParametrizeInfo'],
        raw_params: str) -> list[str]:
    """Extract model name/path strings for a specific test variant.

    Matches the variant's raw_params against parametrize value sets to find
    which specific models this variant uses. Only returns model names from
    the matching value set, not all possible models across all variants.
    """
    if not raw_params:
        return []

    model_names = set()
    for pinfo in parametrize_info:
        # Check if this decorator has model-related params
        model_param_names = [
            name.strip() for name in pinfo.param_names
            if name.strip() in _MODEL_PARAM_NAMES
        ]
        if not model_param_names:
            continue

        # Try to find the value set that matches this variant's raw_params.
        # For auto-generated IDs, the raw_params contains the actual values
        # joined by '-'. We check if all model values appear in raw_params.
        for val_dict in pinfo.all_values:
            model_vals = [
                val_dict.get(n, '') for n in model_param_names
                if val_dict.get(n, '') and val_dict[n] != '?'
            ]
            if not model_vals:
                continue

            # Check if all model values appear in the raw_params
            if all(v in raw_params for v in model_vals):
                model_names.update(model_vals)
                break  # Found matching value set for this decorator

    return sorted(model_names)


def _resolve_custom_id_params(entry: TestEntry,
                              parametrize_info: list['ParametrizeInfo']):
    """Resolve a test entry's raw_params into named parameter values.

    For parametrize decorators with custom IDs (pytest.mark.parametrize with
    ids=[...]), this maps the test's param ID back to the actual param names
    and values. The result is stored in entry.resolved_params.

    For parametrize decorators WITHOUT custom IDs, pytest auto-generates the
    test ID by joining str(value) for each param with '-'. We match the
    raw_params against these auto-generated IDs to resolve param values.

    For cartesian product of multiple decorators, the test ID is formed by
    joining sub-IDs with '-'. We try to match each decorator's IDs against
    segments of the raw_params string.

    For parametrize_with_ids, the IDs are already in "name=value" KV format,
    so entry.params already has them — we just merge into resolved_params.
    """
    if not entry.raw_params:
        return

    # Start with whatever KV params were already parsed
    resolved = dict(entry.params)

    # Collect decorators that have custom id_to_values mappings
    custom_id_infos = [p for p in parametrize_info if p.id_to_values]
    # Collect decorators WITHOUT custom IDs but WITH all_values
    # (auto-generated IDs like "4-4-True-True-True")
    auto_id_infos = [
        p for p in parametrize_info
        if not p.id_to_values and p.all_values
    ]

    remaining = entry.raw_params

    if custom_id_infos:
        # For cartesian products, the test ID joins sub-IDs with '-'.
        # Try to match each custom-ID decorator's known IDs against the
        # raw_params string. Match longer IDs first to avoid partial matches
        # (e.g. "throughput" matching inside "throughput_trtllm").
        for pinfo in custom_id_infos:
            # Sort IDs by length descending so longer IDs match first
            sorted_ids = sorted(pinfo.id_to_values.keys(),
                                key=len, reverse=True)
            for id_str in sorted_ids:
                if id_str in remaining:
                    resolved.update(pinfo.id_to_values[id_str])
                    # Remove the matched ID from remaining to avoid
                    # double-matching in cartesian products
                    remaining = remaining.replace(id_str, '', 1)
                    break

    # For auto-generated IDs: match raw_params against the auto-generated
    # ID format (values joined by '-'). For cartesian products, each
    # decorator's auto-ID is a segment of the full raw_params.
    for pinfo in auto_id_infos:
        for val_dict in pinfo.all_values:
            auto_id = '-'.join(str(v) for v in val_dict.values())
            if auto_id in remaining:
                resolved.update(val_dict)
                remaining = remaining.replace(auto_id, '', 1)
                break

    entry.resolved_params = resolved


def build_test_database(
    test_list_dir: Path,
    test_def_dir: Path,
    test_list_names: Optional[list[str]] = None,
) -> dict[str, TestEntry]:
    """Build the complete test database by combining test lists and definitions.

    Args:
        test_list_dir: Directory containing test list .txt files
        test_def_dir: Directory containing test definition .py files
        test_list_names: Which test list files to include (without extension).
            Defaults to the 4 QA function lists.

    Returns:
        Dict mapping test_id -> TestEntry
    """
    from .impact_rules import TESTCLASS_TO_ARCH

    if test_list_names is None:
        test_list_names = [
            'llm_function_core',
            'llm_function_core_sanity',
            'llm_function_rtx6k',
            'llm_function_multinode',
            'llm_function_stress',
        ]

    # Step 1: Parse all test lists
    raw_entries: dict[str, dict] = {}  # test_id -> merged raw entry
    for name in test_list_names:
        path = test_list_dir / f'{name}.txt'
        if not path.exists():
            continue
        for entry in parse_test_list(path):
            tid = entry['test_id']
            if tid in raw_entries:
                raw_entries[tid].setdefault('test_lists',
                                            set()).add(entry['test_list'])
            else:
                entry['test_lists'] = {entry.pop('test_list')}
                raw_entries[tid] = entry

    # Step 2: Parse test definition files (AST)
    # Collect unique test files referenced by test lists
    test_files = set()
    # Track files that have classless entries (module-level functions)
    classless_files = set()
    for entry in raw_entries.values():
        tf = entry['test_file']
        # Resolve relative to test_def_dir
        full_path = test_def_dir / tf
        if full_path.exists():
            test_files.add((tf, full_path))
            if not entry['test_class']:
                classless_files.add((tf, full_path))

    # Parse each file for class-based tests
    class_registry: dict[str, tuple[ClassInfo, dict[str, MethodInfo]]] = {}
    for rel_path, full_path in test_files:
        parsed = parse_test_definition(full_path)
        for cls_name, (cls_info, methods) in parsed.items():
            method_map = {m.name: m for m in methods}
            class_registry[cls_name] = (cls_info, method_map)

    # Parse files with module-level test functions
    # Key: (test_file, func_name) -> FunctionInfo
    func_registry: dict[str, dict[str, FunctionInfo]] = {}
    for rel_path, full_path in classless_files:
        parsed_funcs = parse_module_functions(full_path)
        if parsed_funcs:
            func_registry[rel_path] = parsed_funcs

    # Step 3: Merge into TestEntry objects
    database: dict[str, TestEntry] = {}

    for tid, raw in raw_entries.items():
        entry = TestEntry(
            test_id=tid,
            test_file=raw['test_file'],
            test_class=raw['test_class'],
            test_method=raw['test_method'],
            params=raw['params'],
            raw_params=raw.get('raw_params', ''),
            test_lists=raw['test_lists'],
        )

        cls_name = raw['test_class']
        method_name = raw['test_method']

        if cls_name and cls_name in class_registry:
            # Enrich from class-based AST
            cls_info, method_map = class_registry[cls_name]

            # L1: model info
            entry.model_name = cls_info.model_name

            # Architecture mapping
            arch = TESTCLASS_TO_ARCH.get(cls_name, '')
            entry.arch = arch

            # L2: GPU requirements (start with class-level, override with method)
            if method_name in method_map:
                mi = method_map[method_name]
                entry.min_sm = mi.min_sm
                entry.max_sm = mi.max_sm
                entry.min_gpu_count = mi.min_gpu_count
                entry.min_gpu_memory = mi.min_gpu_memory
                entry.config_classes = mi.config_classes

                # Parametrize dimensions and resolved params
                all_dims = []
                for pinfo in mi.parametrize_info:
                    all_dims.extend(pinfo.param_names)
                entry.param_dimensions = all_dims

                # Resolve custom-ID params: match raw_params against
                # id_to_values from each parametrize decorator.
                # For cartesian products, the test ID is the join of
                # sub-IDs from each decorator (separated by '-').
                _resolve_custom_id_params(entry, mi.parametrize_info)

                # Extract model names for this specific variant
                entry.model_names = _extract_model_names_for_variant(
                    mi.parametrize_info, entry.raw_params)
            else:
                entry.min_sm = cls_info.min_sm
                entry.max_sm = cls_info.max_sm
                entry.min_gpu_count = cls_info.min_gpu_count
                entry.min_gpu_memory = cls_info.min_gpu_memory

        elif not cls_name and raw['test_file'] in func_registry:
            # Enrich from module-level function AST
            file_funcs = func_registry[raw['test_file']]
            if method_name in file_funcs:
                fi = file_funcs[method_name]
                entry.min_sm = fi.min_sm
                entry.max_sm = fi.max_sm
                entry.min_gpu_count = fi.min_gpu_count
                entry.min_gpu_memory = fi.min_gpu_memory
                entry.config_classes = fi.config_classes

                # Parametrize dimensions and resolved params
                all_dims = []
                for pinfo in fi.parametrize_info:
                    all_dims.extend(pinfo.param_names)
                entry.param_dimensions = all_dims
                _resolve_custom_id_params(entry, fi.parametrize_info)

                # Extract model names for this specific variant
                entry.model_names = _extract_model_names_for_variant(
                    fi.parametrize_info, entry.raw_params)

        # Feature extraction (uses all 3 levels + raw_params)
        # Prefer resolved_params (includes auto-ID resolution) over raw KV params
        feature_params = entry.resolved_params or entry.params
        entry.features = extract_features(entry.test_method, feature_params,
                                          entry.config_classes,
                                          entry.raw_params)

        # Add file-based features (e.g., disaggregated tests)
        if 'disaggregated' in entry.test_file:
            entry.features.add('disaggregated')

        database[tid] = entry

    return database


def save_database(database: dict[str, TestEntry], path: Path):
    """Save parsed database to JSON for caching/inspection."""
    data = {tid: entry.to_dict() for tid, entry in database.items()}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_database(path: Path) -> dict[str, TestEntry]:
    """Load database from JSON cache."""
    with open(path) as f:
        data = json.load(f)
    database = {}
    for tid, d in data.items():
        entry = TestEntry(
            test_id=d['test_id'],
            test_file=d['test_file'],
            test_class=d['test_class'],
            test_method=d['test_method'],
            params=d['params'],
            raw_params=d.get('raw_params', ''),
            model_name=d.get('model_name', ''),
            arch=d.get('arch', ''),
            source_dirs=d.get('source_dirs', []),
            min_sm=d.get('min_sm', 0),
            max_sm=d.get('max_sm', 999),
            min_gpu_count=d.get('min_gpu_count', 1),
            min_gpu_memory=d.get('min_gpu_memory', 0),
            config_classes=set(d.get('config_classes', [])),
            features=set(d.get('features', [])),
            param_dimensions=d.get('param_dimensions', []),
            resolved_params=d.get('resolved_params', {}),
            model_names=d.get('model_names', []),
            test_lists=set(d.get('test_lists', [])),
        )
        database[tid] = entry
    return database
