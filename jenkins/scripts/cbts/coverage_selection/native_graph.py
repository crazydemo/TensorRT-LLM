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
"""Read-only access to the static Python-to-native code graph."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

NATIVE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cu", ".cuh", ".cxx", ".h", ".hpp"})


@dataclass(frozen=True)
class NativeMapping:
    """One evidence-carrying native-file-to-Python-function mapping."""

    python_file: str
    python_qualname: str
    min_hops: int
    evidence: str


class NativeGraph:
    """Static reverse map from a native source file to Python entry functions."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def open(cls, path: Path | str) -> NativeGraph:
        resolved = Path(path).resolve()
        try:
            connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
        except sqlite3.Error as error:
            raise ValueError(f"{resolved}: unable to open native graph: {error}") from error
        try:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(python_native_file)")
            }
        except sqlite3.Error as error:
            connection.close()
            raise ValueError(f"{resolved}: unable to open native graph: {error}") from error
        expected = {"python_file", "python_qualname", "native_file", "min_hops", "evidence"}
        if not expected <= columns:
            connection.close()
            raise ValueError(f"unexpected native graph schema, columns={sorted(columns)}")
        return cls(connection)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> NativeGraph:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def python_functions(self, native_file: str) -> set[tuple[str, str]]:
        """Python functions with a statically resolved path to ``native_file``."""
        return {(row.python_file, row.python_qualname) for row in self.mappings(native_file)}

    def mappings(self, native_file: str) -> tuple[NativeMapping, ...]:
        """Evidence-carrying Python mappings for ``native_file``, in stable order."""
        return tuple(
            NativeMapping(*row)
            for row in self._connection.execute(
                "SELECT python_file, python_qualname, min_hops, evidence "
                "FROM python_native_file WHERE native_file=? "
                "ORDER BY python_file, python_qualname",
                (native_file,),
            )
        )

    def metadata(self) -> dict[str, str]:
        """Codegraph build provenance, or an empty mapping for a legacy graph."""
        try:
            return dict(self._connection.execute("SELECT key, value FROM metadata"))
        except sqlite3.OperationalError:
            return {}


def is_native_source(path: str) -> bool:
    """Whether ``path`` is a native source/header handled by the graph."""
    return Path(path).suffix.lower() in NATIVE_SUFFIXES
