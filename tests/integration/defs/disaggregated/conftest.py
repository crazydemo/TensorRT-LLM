# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

# nvbug 6114140: disable ib, gdr_copy, cuda_ipc for disaggregated tests.
DISAGG_UCX_TLS = "^ib,gdr_copy,cuda_ipc"


@pytest.fixture(autouse=True)
def _set_disagg_ucx_tls(monkeypatch, request):
    """Apply DISAGG_UCX_TLS to every disaggregated test.

    Sets UCX_TLS on os.environ and on llm_venv._new_env (when used) so that
    subprocesses launched by the test inherit the value. Tests that assign
    env["UCX_TLS"] explicitly still win for their own subprocess env dict.
    """
    monkeypatch.setenv("UCX_TLS", DISAGG_UCX_TLS)

    if "llm_venv" not in request.fixturenames:
        yield
        return

    llm_venv = request.getfixturevalue("llm_venv")
    previous = llm_venv._new_env.get("UCX_TLS")
    llm_venv._new_env["UCX_TLS"] = DISAGG_UCX_TLS
    try:
        yield
    finally:
        if previous is None:
            llm_venv._new_env.pop("UCX_TLS", None)
        else:
            llm_venv._new_env["UCX_TLS"] = previous
