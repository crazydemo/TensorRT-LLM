# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""CBTS Feature Logger — Layer 1 PoC (JSONL output).

Captures every BaseLlmArgs instance constructed during a pytest run, dumps
non-default fields, resolves the HF architecture from <model>/config.json,
and appends one JSON line per construction.

Disabled by default. Enable by setting CBTS_FEATURE_LOG to a writable path:
    CBTS_FEATURE_LOG=/tmp/cbts.jsonl pytest tests/integration/defs/accuracy/...

Single-process only for PoC (xdist workers are skipped).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import uuid
from pathlib import Path

_LOG_ENV = "CBTS_FEATURE_LOG"
_ENABLED = bool(os.environ.get(_LOG_ENV))

_arch_cache: dict[str, str | None] = {}
_log_handle = None
_log_lock = threading.Lock()
_run_meta: dict | None = None
_patched = False


def _resolve_hf_arch(model_path):
    if not model_path:
        return None
    key = str(model_path)
    if key in _arch_cache:
        return _arch_cache[key]
    arch = None
    try:
        cfg = Path(key) / "config.json"
        if cfg.is_file():
            data = json.loads(cfg.read_text())
            archs = data.get("architectures")
            if isinstance(archs, list) and archs:
                arch = str(archs[0])
    except Exception:
        arch = None
    _arch_cache[key] = arch
    return arch


def _current_test_nodeid() -> str:
    raw = os.environ.get("PYTEST_CURRENT_TEST", "")
    if not raw:
        return ""
    for suffix in (" (call)", " (setup)", " (teardown)"):
        if raw.endswith(suffix):
            return raw[: -len(suffix)]
    return raw


def _write_record(rec: dict) -> None:
    if _log_handle is None:
        return
    line = json.dumps(rec, sort_keys=True, default=str) + "\n"
    with _log_lock:
        _log_handle.write(line)
        _log_handle.flush()


def _record_args_instance(args_obj) -> None:
    try:
        dump = args_obj.model_dump(exclude_defaults=True, mode="json")
    except Exception as e:
        sys.stderr.write(f"[cbts] model_dump failed: {e}\n")
        return
    model_path = dump.get("model") or getattr(args_obj, "model", None)
    arch = _resolve_hf_arch(model_path) if model_path else None
    dump_blob = json.dumps(dump, sort_keys=True, default=str)
    rec = {
        "kind": "args",
        "run_id": _run_meta["run_id"] if _run_meta else None,
        "test_nodeid": _current_test_nodeid(),
        "args_class": type(args_obj).__name__,
        "model_path": str(model_path) if model_path else None,
        "hf_arch": arch,
        "args_dump_hash": hashlib.sha1(
            (type(args_obj).__name__ + "|" + dump_blob).encode("utf-8")
        ).hexdigest(),
        "args_dump": dump,
        "captured_at": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    }
    _write_record(rec)


def _install_capture_hook() -> bool:
    global _patched
    if _patched:
        return True
    try:
        from tensorrt_llm.llmapi.llm_args import BaseLlmArgs
    except Exception as e:
        sys.stderr.write(f"[cbts] cannot import BaseLlmArgs ({e}); disabling logger\n")
        return False

    orig = BaseLlmArgs.model_post_init

    def _wrapped(self, __context):
        try:
            orig(self, __context)
        finally:
            try:
                _record_args_instance(self)
            except Exception as e:
                sys.stderr.write(f"[cbts] capture failed: {e}\n")

    BaseLlmArgs.model_post_init = _wrapped
    _patched = True
    return True


def _git(*args) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _build_run_meta() -> dict:
    started = datetime.datetime.utcnow().isoformat(timespec="seconds")
    return {
        "run_id": f"{started}-{socket.gethostname()}-{uuid.uuid4().hex[:6]}",
        "started_at": started,
        "git_sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "host": socket.gethostname(),
    }


# ----- pytest hooks -----------------------------------------------------------


def pytest_configure(config):
    global _ENABLED
    if not _ENABLED:
        return
    if os.environ.get("PYTEST_XDIST_WORKER"):
        sys.stderr.write(
            "[cbts] xdist worker detected — Layer 1 PoC is single-process only; skipping\n"
        )
        _ENABLED = False
        return
    if not _install_capture_hook():
        _ENABLED = False


def pytest_sessionstart(session):
    global _log_handle, _run_meta
    if not _ENABLED:
        return
    log_path = os.environ[_LOG_ENV]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    _log_handle = open(log_path, "a", encoding="utf-8")
    _run_meta = _build_run_meta()
    _write_record({"kind": "run_start", **_run_meta})
    sys.stderr.write(
        f"[cbts] feature logger active, log={log_path}, run_id={_run_meta['run_id']}\n"
    )


def pytest_sessionfinish(session, exitstatus):
    global _log_handle
    if not _ENABLED or _log_handle is None:
        return
    _write_record(
        {
            "kind": "run_end",
            "run_id": _run_meta["run_id"] if _run_meta else None,
            "ended_at": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "exitstatus": exitstatus,
        }
    )
    try:
        _log_handle.close()
    finally:
        _log_handle = None
