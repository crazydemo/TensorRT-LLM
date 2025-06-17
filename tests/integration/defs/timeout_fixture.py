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
import threading
import time
import warnings
from contextlib import contextmanager
from typing import Generator, Optional

import pytest


class TimeoutManager:
    """全局超时管理器，用于跟踪测试用例的剩余时间"""

    def __init__(self, total_timeout: float):
        self.start_time = time.time()
        self.total_timeout = total_timeout
        self._lock = threading.Lock()

    def get_remaining_time(self) -> float:
        """获取剩余时间"""
        with self._lock:
            elapsed = time.time() - self.start_time
            remaining = self.total_timeout - elapsed
            # 保留10%缓冲时间，最少30秒
            return max(remaining * 0.9, 30.0)

    def is_expired(self) -> bool:
        """检查是否已超时"""
        with self._lock:
            return time.time() - self.start_time >= self.total_timeout


class ProcessGroupTimeoutRunner:
    """进程组超时运行器"""

    def __init__(self, timeout_manager: Optional[TimeoutManager] = None):
        self.timeout_manager = timeout_manager

    def run_with_timeout(self,
                         cmd: str,
                         timeout: Optional[float] = None,
                         shell: bool = True,
                         cwd: Optional[str] = None,
                         env: Optional[dict] = None) -> tuple[str, str, int]:
        """
        带进程组管理的超时运行

        Args:
            cmd: 要执行的命令
            timeout: 超时时间（秒），如果为None则使用timeout_manager的剩余时间
            shell: 是否使用shell执行
            cwd: 工作目录
            env: 环境变量

        Returns:
            (stdout, stderr, returncode)
        """
        if timeout is None and self.timeout_manager:
            timeout = self.timeout_manager.get_remaining_time()

        if timeout is None:
            timeout = 3600  # 默认1小时

        # 准备环境变量
        if env is None:
            env = os.environ.copy()

        with subprocess.Popen(
                cmd,
                shell=shell,
                start_new_session=True,  # 关键：创建新进程组
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
                self._kill_process_group(proc.pid)
                return "", f"TIMEOUT after {timeout} seconds", -1
            except Exception as e:
                # 确保进程被清理
                self._kill_process_group(proc.pid)
                return "", f"ERROR: {str(e)}", -1

    def _kill_process_group(self, pid: int):
        """杀死进程组"""
        try:
            # 获取进程组ID
            pgid = os.getpgid(pid)
            # 杀死整个进程组
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError) as e:
            # 如果进程组杀死失败，尝试直接杀死进程
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            warnings.warn(f"Failed to kill process group for PID {pid}: {e}")


class CustomTimeoutPlugin:
    """自定义超时插件，覆盖 pytest-timeout 的行为"""

    def __init__(self):
        self.timeout_manager = None
        self.original_timeout_plugin = None

    def install(self, config):
        """安装自定义超时处理"""
        # 禁用原始的 pytest-timeout 插件
        if hasattr(config, 'option') and hasattr(config.option, 'timeout'):
            config.option.timeout = None

        # 保存原始插件引用（如果需要的话）
        if hasattr(config, '_timeout_plugin'):
            self.original_timeout_plugin = config._timeout_plugin

    def set_timeout(self, timeout: float):
        """设置超时时间"""
        self.timeout_manager = TimeoutManager(timeout)

    def get_runner(self) -> ProcessGroupTimeoutRunner:
        """获取超时运行器"""
        return ProcessGroupTimeoutRunner(self.timeout_manager)


# 全局插件实例
_custom_timeout_plugin = CustomTimeoutPlugin()


@pytest.fixture(scope="function")
def timeout_manager(request) -> TimeoutManager:
    """
    超时管理器 fixture

    使用方法:
    @pytest.mark.timeout(3600)  # 1小时超时
    def test_something(timeout_manager):
        runner = timeout_manager.get_runner()
        stdout, stderr, code = runner.run_with_timeout("some_command")
    """
    # 获取测试的超时设置
    timeout_seconds = 3600  # 默认1小时

    # 从 pytest mark 获取超时设置
    timeout_mark = request.node.get_closest_marker('timeout')
    if timeout_mark and timeout_mark.args:
        timeout_seconds = timeout_mark.args[0]

    # 从命令行参数获取全局超时设置
    if hasattr(request.config.option,
               'timeout') and request.config.option.timeout:
        timeout_seconds = request.config.option.timeout

    # 创建超时管理器
    manager = TimeoutManager(timeout_seconds)
    _custom_timeout_plugin.set_timeout(timeout_seconds)

    yield manager

    # 检查是否超时
    if manager.is_expired():
        pytest.fail(f"Test exceeded timeout of {timeout_seconds} seconds")


@pytest.fixture(scope="function")
def process_runner(timeout_manager) -> ProcessGroupTimeoutRunner:
    """
    进程运行器 fixture

    使用方法:
    def test_something(process_runner):
        stdout, stderr, code = process_runner.run_with_timeout("some_command")
    """
    return ProcessGroupTimeoutRunner(timeout_manager)


@contextmanager
def run_with_global_timeout(
        cmd: str,
        timeout: Optional[float] = None,
        shell: bool = True,
        cwd: Optional[str] = None,
        env: Optional[dict] = None
) -> Generator[tuple[str, str, int], None, None]:
    """
    带全局超时管理的上下文管理器

    使用方法:
    with run_with_global_timeout("some_command", timeout=1800) as (stdout, stderr, code):
        if code == -1:
            print("Command timed out or failed")
        else:
            print(f"Command succeeded: {stdout}")
    """
    runner = ProcessGroupTimeoutRunner()
    result = runner.run_with_timeout(cmd, timeout, shell, cwd, env)
    yield result


def pytest_configure(config):
    """pytest 配置钩子，安装自定义超时插件"""
    # 安装自定义超时插件
    _custom_timeout_plugin.install(config)


def pytest_runtest_setup(item):
    """测试运行前设置"""
    # 检查是否有超时标记
    timeout_mark = item.get_closest_marker('timeout')
    if timeout_mark and timeout_mark.args:
        timeout_seconds = timeout_mark.args[0]
        _custom_timeout_plugin.set_timeout(timeout_seconds)


def pytest_runtest_teardown(item, nextitem):
    """测试运行后清理"""
    # 清理超时管理器
    _custom_timeout_plugin.timeout_manager = None


# 便捷函数，用于在测试中直接使用
def run_command_with_timeout(
        cmd: str,
        timeout: Optional[float] = None,
        shell: bool = True,
        cwd: Optional[str] = None,
        env: Optional[dict] = None) -> tuple[str, str, int]:
    """
    便捷函数：运行命令并处理超时

    Args:
        cmd: 要执行的命令
        timeout: 超时时间（秒）
        shell: 是否使用shell执行
        cwd: 工作目录
        env: 环境变量

    Returns:
        (stdout, stderr, returncode)
    """
    runner = ProcessGroupTimeoutRunner()
    return runner.run_with_timeout(cmd, timeout, shell, cwd, env)


# 示例使用方式
"""
# 示例1: 使用 fixture
@pytest.mark.timeout(1800)  # 30分钟超时
def test_quantization_and_build(process_runner):
    # 阶段1: 量化
    stdout, stderr, code = process_runner.run_with_timeout(
        "python quantize.py --model_dir=model --output_dir=quantized",
        timeout=600  # 10分钟
    )
    assert code == 0, f"Quantization failed: {stderr}"

    # 阶段2: 构建引擎
    stdout, stderr, code = process_runner.run_with_timeout(
        "trtllm-build --checkpoint_dir=quantized --output_dir=engine",
        timeout=900  # 15分钟
    )
    assert code == 0, f"Engine build failed: {stderr}"

    # 阶段3: 运行测试
    stdout, stderr, code = process_runner.run_with_timeout(
        "python summarize.py --engine_dir=engine",
        timeout=300  # 5分钟
    )
    assert code == 0, f"Summary test failed: {stderr}"

# 示例2: 使用上下文管理器
def test_simple_command():
    with run_with_global_timeout("sleep 10", timeout=5) as (stdout, stderr, code):
        assert code == -1, "Command should have timed out"
        assert "TIMEOUT" in stderr

# 示例3: 使用便捷函数
def test_direct_command():
    stdout, stderr, code = run_command_with_timeout("echo 'hello'", timeout=10)
    assert code == 0
    assert "hello" in stdout
"""
