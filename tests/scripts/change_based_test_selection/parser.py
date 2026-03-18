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

    # which test lists contain this entry
    test_lists: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        d = {
            "test_id": self.test_id,
            "test_file": self.test_file,
            "test_class": self.test_class,
            "test_method": self.test_method,
            "params": self.params,
            "model_name": self.model_name,
            "arch": self.arch,
            "source_dirs": self.source_dirs,
            "min_sm": self.min_sm,
            "max_sm": self.max_sm,
            "min_gpu_count": self.min_gpu_count,
            "min_gpu_memory": self.min_gpu_memory,
            "config_classes": sorted(self.config_classes),
            "features": sorted(self.features),
            "test_lists": sorted(self.test_lists),
        }
        return d


# --- Test List Parsing ---

# Regex to parse a pytest node ID:
#   file.py::ClassName::method_name[param1=val1-param2=val2]
_NODE_RE = re.compile(
    r'^(?P<file>[^:]+)::(?P<class>\w+)::(?P<method>\w+)(?:\[(?P<params>[^\]]*)\])?$'
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
                params = {}
                if m.group('params'):
                    for kv in _PARAM_KV_RE.finditer(m.group('params')):
                        params[kv.group('key')] = kv.group('val')
                entries.append({
                    'test_id': line,
                    'test_file': m.group('file'),
                    'test_class': m.group('class'),
                    'test_method': m.group('method'),
                    'params': params,
                    'test_list': path.stem,
                })
            else:
                # Non-standard format (e.g. e2e tests without class)
                # Still track it but with limited metadata
                entries.append({
                    'test_id': line,
                    'test_file': line.split('::')[0] if '::' in line else line,
                    'test_class': '',
                    'test_method':
                    line.split('::')[-1].split('[')[0] if '::' in line else '',
                    'params': {},
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
class MethodInfo:
    """AST-extracted info about a test method."""

    name: str
    min_sm: int = 0
    max_sm: int = 999
    min_gpu_count: int = 1
    min_gpu_memory: int = 0
    config_classes: set[str] = field(default_factory=set)


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

            # L3: Scan method body for Config class instantiations
            call_names = _extract_call_names(item)
            method.config_classes = call_names & _KNOWN_CONFIG_CLASSES

            methods.append(method)

        if methods:
            result[node.name] = (cls_info, methods)

    return result


# --- Feature Extraction ---

# Features extracted from test method name patterns
_METHOD_FEATURE_PATTERNS = {
    'fp8': r'(?:^|_)fp8(?:$|_|[^a-z])',
    'nvfp4': r'nvfp4',
    'fp4': r'(?:^|_)fp4(?:$|_|[^a-z])',
    'w4': r'(?:^|_)w4(?:$|_|[^a-z])',
    'w4a8': r'w4a8',
    'w4a16': r'w4a16',
    'mxfp4': r'mxfp4',
    'eagle3': r'eagle3',
    'ngram': r'ngram',
    'pard': r'pard',
    'mtp': r'mtp',
    'suffix_automaton': r'suffix_automaton',
    'auto_spec_decode': r'auto_spec_decode',
    'guided_decoding': r'guided_decoding',
    'beam_search': r'beam_search',
    'chunked_prefill': r'chunked_prefill',
    'streaming': r'streaming',
    'lora': r'lora',
    'return_logits': r'return_logits',
    'eplb': r'eplb',
    'batch_waiting': r'batch_waiting',
    'dummy_load': r'dummy_load',
    'skip_softmax': r'skip_softmax',
    'piecewise_cuda_graph': r'piecewise_cuda_graph',
    'mixed_precision': r'mixed_precision',
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
                     config_classes: set[str]) -> set[str]:
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

    return features


# --- Main Build Function ---


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
    for entry in raw_entries.values():
        tf = entry['test_file']
        # Resolve relative to test_def_dir
        full_path = test_def_dir / tf
        if full_path.exists():
            test_files.add((tf, full_path))

    # Parse each file
    class_registry: dict[str, tuple[ClassInfo, dict[str, MethodInfo]]] = {}
    for rel_path, full_path in test_files:
        parsed = parse_test_definition(full_path)
        for cls_name, (cls_info, methods) in parsed.items():
            method_map = {m.name: m for m in methods}
            class_registry[cls_name] = (cls_info, method_map)

    # Step 3: Merge into TestEntry objects
    database: dict[str, TestEntry] = {}

    for tid, raw in raw_entries.items():
        entry = TestEntry(
            test_id=tid,
            test_file=raw['test_file'],
            test_class=raw['test_class'],
            test_method=raw['test_method'],
            params=raw['params'],
            test_lists=raw['test_lists'],
        )

        cls_name = raw['test_class']

        # Enrich from AST if available
        if cls_name in class_registry:
            cls_info, method_map = class_registry[cls_name]

            # L1: model info
            entry.model_name = cls_info.model_name

            # Architecture mapping
            arch = TESTCLASS_TO_ARCH.get(cls_name, '')
            entry.arch = arch

            # L2: GPU requirements (start with class-level, override with method)
            method_name = raw['test_method']
            if method_name in method_map:
                mi = method_map[method_name]
                entry.min_sm = mi.min_sm
                entry.max_sm = mi.max_sm
                entry.min_gpu_count = mi.min_gpu_count
                entry.min_gpu_memory = mi.min_gpu_memory
                entry.config_classes = mi.config_classes
            else:
                entry.min_sm = cls_info.min_sm
                entry.max_sm = cls_info.max_sm
                entry.min_gpu_count = cls_info.min_gpu_count
                entry.min_gpu_memory = cls_info.min_gpu_memory

        # Feature extraction (uses all 3 levels)
        entry.features = extract_features(entry.test_method, entry.params,
                                          entry.config_classes)

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
            model_name=d.get('model_name', ''),
            arch=d.get('arch', ''),
            source_dirs=d.get('source_dirs', []),
            min_sm=d.get('min_sm', 0),
            max_sm=d.get('max_sm', 999),
            min_gpu_count=d.get('min_gpu_count', 1),
            min_gpu_memory=d.get('min_gpu_memory', 0),
            config_classes=set(d.get('config_classes', [])),
            features=set(d.get('features', [])),
            test_lists=set(d.get('test_lists', [])),
        )
        database[tid] = entry
    return database
