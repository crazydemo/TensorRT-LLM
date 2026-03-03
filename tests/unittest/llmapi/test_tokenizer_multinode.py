# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for tokenizer serialization across processes (multi-node scenario).

Regression test for nvbugs/5823783: multi-node trtllm-serve with
--trust_remote_code and --tool_parser kimi_k2 would hang because the tokenizer
(loaded with trust_remote_code, using dynamic HF modules) could not be
deserialized on other MPI ranks. Fix: cloudpickle-based by-value serialization
in tensorrt_llm/tokenizer/tokenizer.py (TransformersTokenizer.__reduce__).
"""

import os
import pickle
import subprocess  # nosec B404
import sys
import tempfile

import pytest

from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer

# isort: off
from .test_llm import default_model_name, get_model_path
# isort: on

pytestmark = pytest.mark.threadleak(enabled=False)


def test_transformers_tokenizer_pickle_roundtrip():
    """TransformersTokenizer can be pickled and unpickled in the same process.

    Ensures the __reduce__ path does not break normal tokenizer usage.
    """
    model_path = get_model_path(default_model_name)
    if not os.path.isdir(model_path):
        pytest.skip(f"Model path not found: {model_path} (set LLM_MODELS_ROOT)")

    tokenizer = load_hf_tokenizer(model_path, trust_remote_code=False)
    assert tokenizer is not None

    text = "The future of AI is"
    ids_before = tokenizer.encode(text, add_special_tokens=False)

    data = pickle.dumps(tokenizer)
    tokenizer_restored = pickle.loads(data)

    ids_after = tokenizer_restored.encode(text, add_special_tokens=False)
    assert ids_before == ids_after
    decoded = tokenizer_restored.decode(ids_after)
    assert decoded == text


def test_transformers_tokenizer_trust_remote_code_serializable_across_processes():
    """Tokenizer loaded with trust_remote_code can be deserialized in another process.

    Simulates multi-node: rank 0 loads tokenizer (with dynamic HF modules),
    serializes it; other ranks receive bytes and deserialize. Without the fix
    (nvbugs/5823783), deserialization on other ranks fails because dynamic
    modules are node-local. This test would hang or fail before the fix.
    """
    # The fix (TransformersTokenizer.__reduce__) uses cloudpickle for by-value
    # serialization. Without cloudpickle the test would fail even with the fix.
    pytest.importorskip("cloudpickle")

    # Must use a model that actually has custom tokenizer code (dynamic HF modules).
    # TinyLlama does not; InternLM does. Without this, the test would pass even
    # without the nvbugs/5823783 fix and would not catch the user-reported hang.
    trust_remote_code_model = "Kimi-K2-Instruct"
    model_path = get_model_path(trust_remote_code_model)
    if not os.path.isdir(model_path):
        pytest.skip(
            f"Model path not found: {model_path} (set LLM_MODELS_ROOT for "
            f"trust_remote_code multi-node serialization test)"
        )

    tokenizer = load_hf_tokenizer(model_path, trust_remote_code=True)
    assert tokenizer is not None

    tokenizer_bytes = pickle.dumps(tokenizer)

    # Run deserialization in a subprocess with a clean HOME to simulate another
    # node: on multi-node, other ranks do not have the parent's
    # ~/.cache/huggingface/modules/transformers_modules/ (dynamic trust_remote_code
    # modules). Without the fix, standard pickle serializes by reference, so the
    # child would need to import those modules and fail. With the fix, tokenizer
    # is serialized by value (cloudpickle), so the child can unpickle without
    # the cache.
    # Use a result file instead of stdout so child's logs (e.g. "[INFO] ...")
    # cannot pollute the pickle stream and cause "invalid load key, '['".
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
    ) as runner:
        runner.write("""
import os
import pickle
import sys
tokenizer_bytes = sys.stdin.buffer.read()
result_path = os.environ["TLLM_TOKENIZER_TEST_RESULT_FILE"]
try:
    tokenizer = pickle.loads(tokenizer_bytes)
    ids = tokenizer.encode("hello", add_special_tokens=False)
    result = ("ok", list(ids))
except Exception as e:
    result = ("err", str(e))
with open(result_path, "wb") as f:
    pickle.dump(result, f)
""")
        runner_path = runner.name

    try:
        with tempfile.TemporaryDirectory() as clean_home:
            result_file = os.path.join(clean_home, "child_result.pkl")
            child_env = {**os.environ}
            child_env["HOME"] = clean_home
            child_env["TLLM_TOKENIZER_TEST_RESULT_FILE"] = result_file
            child_env.pop("HF_HOME", None)
            child_env.pop("HF_HUB_CACHE", None)
            child_env["XDG_CACHE_HOME"] = os.path.join(clean_home, ".cache")

            out = subprocess.run(
                [sys.executable, runner_path],
                input=tokenizer_bytes,
                capture_output=True,
                timeout=60,
                env=child_env,
            )
            if out.returncode != 0:
                pytest.fail(f"Child process failed: stdout={out.stdout!r} stderr={out.stderr!r}")
            with open(result_file, "rb") as f:
                result = pickle.load(f)
        status, value = result
        assert status == "ok", f"Child deserialization failed: {value}"
        assert isinstance(value, list) and len(value) > 0
    finally:
        if os.path.exists(runner_path):
            try:
                os.unlink(runner_path)
            except OSError:
                pass
