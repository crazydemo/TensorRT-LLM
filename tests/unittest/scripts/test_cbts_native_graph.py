# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import sqlite3
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CBTS = _ROOT / "jenkins/scripts/cbts"
sys.path[:0] = [str(_CBTS), str(_CBTS / "coverage_selection"), str(_CBTS / "tools")]

from audit_native_boundaries import Edge, Entry, _representative_path  # noqa: E402
from blocks import Block, Stage  # noqa: E402
from build_native_codegraph import _load_compilation_database  # noqa: E402
from case_native_dependency_demo import _nanobind_class_registrations  # noqa: E402
from native_graph import NativeGraph  # noqa: E402
from native_impact import build_native_impact_report  # noqa: E402
from render_native_coverage_report import render as render_native_coverage_report  # noqa: E402
from rules.base import PRInputs  # noqa: E402
from rules.tests_def_rule import TestsDefRule as _TestsDefRule  # noqa: E402
from selector import CoverageSelector  # noqa: E402

_NATIVE_FILE = "cpp/tensorrt_llm/kernels/example.cu"
_PYTHON_FILE = "tensorrt_llm/example.py"


class _TouchDB:
    def __init__(self, function_tests: set[str], file_tests: set[str]) -> None:
        self._function_tests = function_tests
        self._file_tests = file_tests

    def tests_touching_func(self, file: str, qualname: str) -> set[str]:
        if (file, qualname) == (_PYTHON_FILE, "run"):
            return set(self._function_tests)
        return set()

    def tests_touching_file(self, file: str) -> set[str]:
        return set(self._file_tests) if file == _PYTHON_FILE else set()

    def untrusted_tests(self, *_args) -> set[str]:
        return set()

    def known_by_family(self) -> dict[str, set[str]]:
        tests = self._function_tests | self._file_tests
        return {"A100X-PyTorch": {test.partition("/")[2] for test in tests}}


class _NativeUnitYamlIndex:
    def __init__(self) -> None:
        self.blocks = [
            Block(
                yaml_stem="l0_a30",
                block_index=0,
                condition={},
                tests=[
                    "cpp/test_unit_tests.py::test_unit_tests[batch_manager-80]",
                    "cpp/test_unit_tests.py::test_unit_tests[common-80]",
                ],
            ),
            Block(
                yaml_stem="l0_h100",
                block_index=0,
                condition={},
                tests=["accuracy/test_other.py::test_other"],
            ),
        ]


class _ImportTouchDB:
    def __init__(self, importer_tests: set[str] | None = None) -> None:
        self.file_tests = {
            "A100-PyTorch-1/tests/test_example.py::test_file_user",
            "A100-PyTorch-2/tests/test_example.py::test_importer",
        }
        self.importer_tests = (
            {"A100-PyTorch-2/tests/test_example.py::test_importer"}
            if importer_tests is None
            else importer_tests
        )

    def file_has_touch_rows(self, file: str) -> bool:
        return file == _PYTHON_FILE

    def tests_touching_func(self, file: str, qualname: str) -> set[str]:
        if (file, qualname) == (_PYTHON_FILE, "<module>"):
            return set(self.importer_tests)
        return set()

    def tests_touching_file(self, file: str) -> set[str]:
        return set(self.file_tests) if file == _PYTHON_FILE else set()

    def untrusted_tests(self, *_args) -> set[str]:
        return set()

    def known_by_family(self) -> dict[str, set[str]]:
        return {
            "A100-PyTorch": {
                "tests/test_example.py::test_file_user",
                "tests/test_example.py::test_importer",
            }
        }


def _write_graph(path: Path, *, include_mapping: bool = True) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE python_native_file("
            "python_file TEXT, python_qualname TEXT, native_file TEXT, "
            "min_hops INTEGER, evidence TEXT)"
        )
        if include_mapping:
            connection.execute(
                "INSERT INTO python_native_file VALUES (?, ?, ?, 0, 'test')",
                (_PYTHON_FILE, "run", _NATIVE_FILE),
            )


def test_native_graph_reverse_lookup(tmp_path: Path) -> None:
    path = tmp_path / "graph.sqlite"
    _write_graph(path)

    with NativeGraph.open(path) as graph:
        assert graph.python_functions(_NATIVE_FILE) == {(_PYTHON_FILE, "run")}
        assert graph.python_functions("cpp/unmapped.cu") == set()


def test_native_selection_uses_function_touches(tmp_path: Path) -> None:
    path = tmp_path / "graph.sqlite"
    _write_graph(path)
    test = "A100X-PyTorch-1/accuracy/test_example.py::test_run"

    with NativeGraph.open(path) as graph:
        result = CoverageSelector(
            _TouchDB({test}, set()),
            tmp_path,
            native_graph=graph,
        ).decide([_NATIVE_FILE], {})

    assert result.ok
    assert result.impacted == {"A100X-PyTorch": {"accuracy/test_example.py::test_run"}}
    assert result.no_data_funcs == []


def test_native_impact_report_lists_mappings_and_cases(tmp_path: Path) -> None:
    path = tmp_path / "graph.sqlite"
    _write_graph(path)
    test = "A100X-PyTorch-1/accuracy/test_example.py::test_run"

    with NativeGraph.open(path) as graph:
        report = build_native_impact_report(
            [_NATIVE_FILE, "README.md"], graph, _TouchDB({test}, set())
        )

    assert report["summary"] == {
        "native_files": 1,
        "production_native_files": 1,
        "native_test_files": 0,
        "mapped_native_files": 1,
        "unmapped_native_files": 0,
        "python_function_mappings": 1,
        "unique_cases": 1,
    }
    assert report["native_files"][0]["cases"] == [test]
    assert report["native_files"][0]["python_functions"] == [
        {
            "python_file": _PYTHON_FILE,
            "python_qualname": "run",
            "min_hops": 0,
            "evidence": "test",
            "case_count": 1,
        }
    ]


def test_native_impact_report_classifies_cpp_tests_as_test_definitions(tmp_path: Path) -> None:
    path = tmp_path / "graph.sqlite"
    _write_graph(path, include_mapping=False)
    native_test = "cpp/tests/unit_tests/batch_manager/exampleTest.cpp"

    with NativeGraph.open(path) as graph:
        report = build_native_impact_report([native_test], graph, _TouchDB(set(), set()))

    assert report["summary"]["production_native_files"] == 0
    assert report["summary"]["native_test_files"] == 1
    assert report["summary"]["unmapped_native_files"] == 0
    assert report["native_files"][0]["mapping_status"] == "test_definition"


def test_render_native_coverage_report_builds_case_and_native_pages(tmp_path: Path) -> None:
    coverage = tmp_path / "coverage.sqlite"
    graph = tmp_path / "graph.sqlite"
    output = tmp_path / "html"
    mapped_test = "A100-PyTorch-1/accuracy/test_example.py::test_run"
    unmapped_test = "A100-PyTorch-1/accuracy/test_other.py::test_other"
    with sqlite3.connect(coverage) as connection:
        connection.executescript(
            "CREATE TABLE test_case_meta(test TEXT, stage TEXT, outcome TEXT, "
            "expected_workers INTEGER, saved_procs INTEGER);"
            "CREATE TABLE touch_rows(test TEXT, file TEXT, qualname TEXT, stage TEXT);"
            "CREATE TABLE meta(key TEXT, value TEXT);"
        )
        connection.executemany(
            "INSERT INTO test_case_meta VALUES (?, 'A100-PyTorch-1', 'passed', 0, 1)",
            [(mapped_test,), (unmapped_test,)],
        )
        connection.executemany(
            "INSERT INTO touch_rows VALUES (?, ?, ?, 'A100-PyTorch-1')",
            [
                (mapped_test, _PYTHON_FILE, "run"),
                (unmapped_test, _PYTHON_FILE, "other"),
            ],
        )
        connection.execute("INSERT INTO meta VALUES ('func_rate_pct', '50.0')")
    with sqlite3.connect(graph) as connection:
        connection.executescript(
            "CREATE TABLE metadata(key TEXT, value TEXT);"
            "CREATE TABLE python_native_file(python_file TEXT, python_qualname TEXT, "
            "native_file TEXT, min_hops INTEGER, evidence TEXT);"
        )
        connection.execute("INSERT INTO metadata VALUES ('source_commit', 'abc123')")
        connection.execute(
            "INSERT INTO python_native_file VALUES (?, 'run', ?, 1, 'test mapping')",
            (_PYTHON_FILE, _NATIVE_FILE),
        )

    summary = render_native_coverage_report(coverage, graph, output)

    assert summary == {
        "cases": 2,
        "cases_with_python_data": 2,
        "cases_with_inferred_native_hits": 1,
        "inferred_native_files": 1,
        "mapping_rows": 1,
        "python_hit_rows": 2,
    }
    assert "Per-case inferred C++ / CUDA coverage" in (output / "index.html").read_text()
    assert "test_run" in "".join(path.read_text() for path in (output / "cases").iterdir())
    assert "runtime hit set" in "".join(path.read_text() for path in (output / "python").iterdir())
    assert _NATIVE_FILE in "".join(path.read_text() for path in (output / "native").iterdir())


def test_native_selection_falls_back_to_python_file_touches(tmp_path: Path) -> None:
    path = tmp_path / "graph.sqlite"
    _write_graph(path)
    test = "A100X-PyTorch-1/accuracy/test_example.py::test_file_fallback"

    with NativeGraph.open(path) as graph:
        result = CoverageSelector(
            _TouchDB(set(), {test}),
            tmp_path,
            native_graph=graph,
        ).decide([_NATIVE_FILE], {})

    assert result.ok
    assert result.impacted == {"A100X-PyTorch": {"accuracy/test_example.py::test_file_fallback"}}
    assert result.no_data_funcs == [f"{_PYTHON_FILE}::run"]


@pytest.mark.parametrize("with_graph", [False, True])
def test_native_selection_declines_without_a_mapping(tmp_path: Path, with_graph: bool) -> None:
    path = tmp_path / "graph.sqlite"
    _write_graph(path, include_mapping=False)
    graph = NativeGraph.open(path) if with_graph else None
    try:
        result = CoverageSelector(
            _TouchDB(set(), set()),
            tmp_path,
            native_graph=graph,
        ).decide([_NATIVE_FILE], {})
    finally:
        if graph is not None:
            graph.close()

    assert not result.ok
    assert result.reason in {
        "native code graph is not configured",
        f"native file has no static mapping: {_NATIVE_FILE}",
    }


def test_load_compilation_database_remaps_container_sources(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "cpp/example.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("void example() {}\n")
    database_path = tmp_path / "compile_commands.json"
    database_path.write_text(
        json.dumps(
            [
                {
                    "directory": "/code/tensorrt_llm/cpp/build",
                    "command": "/usr/bin/c++ -c /code/tensorrt_llm/cpp/example.cpp",
                    "file": "/code/tensorrt_llm/cpp/example.cpp",
                },
                {
                    "directory": "/code/tensorrt_llm/cpp/build",
                    "command": "/usr/bin/c++ -DSECOND_BUILD -c /code/tensorrt_llm/cpp/example.cpp",
                    "file": "/code/tensorrt_llm/cpp/example.cpp",
                },
                {
                    "directory": "/code/tensorrt_llm/cpp/build",
                    "command": "/usr/local/cuda/bin/nvcc -c /code/tensorrt_llm/cpp/missing.cu",
                    "file": "/code/tensorrt_llm/cpp/missing.cu",
                },
                {
                    "directory": "/tmp",
                    "command": "/usr/bin/c++ -c /vendor/external.cpp",
                    "file": "/vendor/external.cpp",
                },
            ]
        )
    )

    database = _load_compilation_database(
        repo,
        database_path,
        Path("/code/tensorrt_llm"),
    )

    assert database.entry_count == 4
    assert database.unique_source_count == 3
    assert database.available_sources == (source,)
    assert database.missing_source_count == 1
    assert database.outside_repo_count == 1
    assert dict(database.compiler_counts) == {
        "/usr/bin/c++": 3,
        "/usr/local/cuda/bin/nvcc": 1,
    }


def test_nanobind_class_registration_extracts_method_pointers() -> None:
    source = """
    nb::class_<CapacityScheduler>(m, CapacityScheduler::name)
        .def(nb::init<int>())
        .def("__call__", &CapacityScheduler::operator())
        .def("set_policy", &CapacityScheduler::setPolicy)
        .def("name", [](CapacityScheduler const&) { return CapacityScheduler::name; });

    nb::class_<OtherScheduler>(m, "OtherScheduler")
        .def("run", &OtherScheduler::run);
    """

    registrations = list(_nanobind_class_registrations(source))

    assert [(class_name, target) for class_name, _offset, target in registrations] == [
        ("CapacityScheduler", "CapacityScheduler::operator()"),
        ("CapacityScheduler", "CapacityScheduler::setPolicy"),
        ("OtherScheduler", "OtherScheduler::run"),
    ]


def test_boundary_audit_representative_path_prefers_cuda() -> None:
    entry = Entry("entry", "cpp/entry.cpp", 10, "example::entry", "test entry")
    cpp_edge = Edge(
        1,
        "cpp/entry.cpp",
        11,
        "example::entry",
        "helper()",
        "cpp/helper.cpp",
        20,
        "example::helper",
        "test edge",
    )
    cuda_edge = Edge(
        2,
        "cpp/helper.cpp",
        21,
        "example::helper",
        "launch()",
        "cpp/kernel.cu",
        30,
        "example::launch",
        "test CUDA edge",
    )

    assert _representative_path([entry], [cpp_edge, cuda_edge]) == [cpp_edge, cuda_edge]


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (
            "file",
            {
                "tests/test_example.py::test_file_user",
                "tests/test_example.py::test_importer",
            },
        ),
        ("importers", {"tests/test_example.py::test_importer"}),
    ],
)
def test_import_executed_change_uses_configured_bound(
    tmp_path: Path, policy: str, expected: set[str]
) -> None:
    source = "SETTING = 2\n"
    diff = "@@ -1 +1 @@\n-SETTING = 1\n+SETTING = 2\n"
    result = CoverageSelector(
        _ImportTouchDB(),
        tmp_path,
        import_executed_policy=policy,
        read_source=lambda _path: source,
    ).decide([_PYTHON_FILE], {_PYTHON_FILE: diff})

    assert result.ok
    assert result.impacted == {"A100-PyTorch": expected}
    assert result.import_executed_bounds == [f"{_PYTHON_FILE}::<module>"]


@pytest.mark.parametrize(
    ("policy", "db"),
    [("decline", _ImportTouchDB()), ("importers", _ImportTouchDB(set()))],
)
def test_import_executed_change_declines_without_a_bound(
    tmp_path: Path, policy: str, db: _ImportTouchDB
) -> None:
    source = "SETTING = 2\n"
    diff = "@@ -1 +1 @@\n-SETTING = 1\n+SETTING = 2\n"
    result = CoverageSelector(
        db,
        tmp_path,
        import_executed_policy=policy,
        read_source=lambda _path: source,
    ).decide([_PYTHON_FILE], {_PYTHON_FILE: diff})

    assert not result.ok
    assert result.reason.startswith("import-executed change")


def test_cuda_namespace_recovery_indexes_nested_functions(tmp_path: Path) -> None:
    tree_sitter = pytest.importorskip("tree_sitter")
    tree_sitter_cpp = pytest.importorskip("tree_sitter_cpp")
    from case_native_dependency_demo import _index_cpp_functions

    repo = tmp_path / "repo"
    source = repo / "cpp/example.cu"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
        TRTLLM_NAMESPACE_BEGIN
        namespace kernels::example
        {
        template <typename T>
        void launch(T value)
        {
            consume(value);
        }
        }
        TRTLLM_NAMESPACE_END
        """
    )
    parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_cpp.language()))

    definitions = _index_cpp_functions(repo, parser, [source])

    assert any(definition.base_name == "launch" for definition in definitions)


def test_cpp_unit_test_change_maps_to_component_block(tmp_path: Path) -> None:
    path = "cpp/tests/unit_tests/batch_manager/capacitySchedulerTest.cpp"
    yaml_index = _NativeUnitYamlIndex()
    stages = {
        "A30-CPP-1": Stage(
            name="A30-CPP-1",
            yaml_stem="l0_a30",
            cpu_arch="x86",
            split_id=1,
            total_splits=1,
        )
    }

    result = _TestsDefRule(yaml_index, stages, tmp_path).apply(PRInputs([path], {}))

    assert result is not None
    assert result.handled_files == {path}
    assert result.affected_stages == {"A30-CPP-1"}
    assert result.scope == "testdefonly"
    assert result.block_filters == {
        ("l0_a30", 0): {
            "cpp/test_unit_tests.py::test_unit_tests": {
                "cpp/test_unit_tests.py::test_unit_tests[batch_manager-80]"
            }
        }
    }
