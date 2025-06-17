# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import signal
import subprocess
import time
import warnings
from typing import Optional


class GlobalTimeoutManager:
    """全局超时管理器，自动为所有测试用例提供超时管理"""

    def __init__(self):
        self.start_time = None
        self.total_timeout = None
        self._active = False

    def start_test(self, timeout_seconds: float):
        """开始测试，设置超时时间"""
        self.start_time = time.time()
        self.total_timeout = timeout_seconds
        self._active = True
        print(f"[GlobalTimeout] Test started with {timeout_seconds}s timeout")

    def get_remaining_time(self) -> Optional[float]:
        """获取剩余时间"""
        if not self._active or self.start_time is None or self.total_timeout is None:
            return None

        elapsed = time.time() - self.start_time
        remaining = self.total_timeout - elapsed

        # 保留10%缓冲时间，最少30秒
        buffer_time = max(remaining * 0.9, 30.0)

        if remaining <= 0:
            print(
                f"[GlobalTimeout] Test exceeded timeout of {self.total_timeout}s"
            )
            return 30.0  # 给最后30秒清理时间

        return buffer_time

    def is_expired(self) -> bool:
        """检查是否已超时"""
        if not self._active or self.start_time is None or self.total_timeout is None:
            return False
        return time.time() - self.start_time >= self.total_timeout

    def end_test(self):
        """结束测试"""
        if self._active:
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"[GlobalTimeout] Test completed in {elapsed:.1f}s")
        self._active = False
        self.start_time = None
        self.total_timeout = None


# 全局实例
_global_timeout_manager = GlobalTimeoutManager()


def get_global_timeout_manager() -> GlobalTimeoutManager:
    """获取全局超时管理器实例"""
    return _global_timeout_manager


def run_with_global_timeout(cmd: str,
                            timeout: Optional[float] = None,
                            shell: bool = True,
                            cwd: Optional[str] = None,
                            env: Optional[dict] = None) -> tuple[str, str, int]:
    """
    带全局超时管理的命令运行函数

    Args:
        cmd: 要执行的命令
        timeout: 超时时间（秒），如果为None则使用全局剩余时间
        shell: 是否使用shell执行
        cwd: 工作目录
        env: 环境变量

    Returns:
        (stdout, stderr, returncode)
    """
    # 获取超时时间
    if timeout is None:
        timeout = _global_timeout_manager.get_remaining_time()

    if timeout is None:
        timeout = 3600  # 默认1小时

    # 准备环境变量
    if env is None:
        env = os.environ.copy()

    print(f"[GlobalTimeout] Running command with {timeout}s timeout: {cmd}")

    with subprocess.Popen(
            cmd,
            shell=shell,
            start_new_session=True,  # 创建新进程组
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return stdout.decode('utf-8', errors='ignore'), \
                   stderr.decode('utf-8', errors='ignore'), \
                   proc.returncode
        except subprocess.TimeoutExpired:
            # 杀死整个进程组
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError) as e:
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                warnings.warn(
                    f"Failed to kill process group for PID {proc.pid}: {e}")

            return "", f"TIMEOUT after {timeout} seconds", -1
        except Exception as e:
            # 确保进程被清理
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            return "", f"ERROR: {str(e)}", -1


# pytest 钩子函数
def pytest_configure(config):
    """pytest 配置钩子"""
    # 避免线程泄漏
    import tqdm
    tqdm.tqdm.monitor_interval = 0


def pytest_runtest_setup(item):
    """测试运行前设置"""
    # 获取测试的超时设置
    timeout_seconds = 3600  # 默认1小时

    # 从 pytest mark 获取超时设置
    timeout_mark = item.get_closest_marker('timeout')
    if timeout_mark and timeout_mark.args:
        timeout_seconds = timeout_mark.args[0]

    # 从命令行参数获取全局超时设置
    if hasattr(item.config.option, 'timeout') and item.config.option.timeout:
        timeout_seconds = item.config.option.timeout

    # 启动全局超时管理器
    _global_timeout_manager.start_test(timeout_seconds)


def pytest_runtest_teardown(item, nextitem):
    """测试运行后清理"""
    _global_timeout_manager.end_test()


# 替换原始的 subprocess 函数
def _patch_subprocess_functions():
    """替换原始的 subprocess 函数以使用全局超时管理"""
    import subprocess as sp

    # 保存原始函数
    original_call = sp.call
    original_check_call = sp.check_call
    original_check_output = sp.check_output

    def patched_call(*popenargs, timeout=None, **kwargs):
        """带全局超时管理的 call 函数"""
        if timeout is None:
            timeout = _global_timeout_manager.get_remaining_time()
        return original_call(*popenargs, timeout=timeout, **kwargs)

    def patched_check_call(*popenargs, timeout=None, **kwargs):
        """带全局超时管理的 check_call 函数"""
        if timeout is None:
            timeout = _global_timeout_manager.get_remaining_time()
        return original_check_call(*popenargs, timeout=timeout, **kwargs)

    def patched_check_output(*popenargs, timeout=None, **kwargs):
        """带全局超时管理的 check_output 函数"""
        if timeout is None:
            timeout = _global_timeout_manager.get_remaining_time()
        return original_check_output(*popenargs, timeout=timeout, **kwargs)

    # 替换函数
    sp.call = patched_call
    sp.check_call = patched_check_call
    sp.check_output = patched_check_output


# 自动应用补丁
_patch_subprocess_functions()
