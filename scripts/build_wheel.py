#!/usr/bin/env python3
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
import platform
import sys
import sysconfig
import warnings
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from shutil import copy, copytree, rmtree
from subprocess import DEVNULL, CalledProcessError, check_output, run
from textwrap import dedent
from typing import List

try:
    from packaging.requirements import Requirement
except (ImportError, ModuleNotFoundError):
    from pip._vendor.packaging.requirements import Requirement

build_run = partial(run, shell=True, check=True)


@contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def get_project_dir():
    return Path(__file__).parent.resolve().parent


def get_source_dir():
    return get_project_dir() / "cpp"


def get_build_dir(build_dir, build_type):
    if build_dir is None:
        build_dir = get_source_dir() / ("build" if build_type == "Release" else
                                        f"build_{build_type}")
    else:
        build_dir = Path(build_dir).resolve()
    return build_dir


def clear_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and not os.path.islink(item_path):
            rmtree(item_path)
        else:
            try:
                os.remove(item_path)
            except (OSError, IOError) as e:
                print(f"Failed to remove {item_path}: {e}", file=sys.stderr)


def sysconfig_scheme(override_vars=None):
    # Backported 'venv' scheme from Python 3.11+
    if os.name == 'nt':
        scheme = {
            'purelib': '{base}/Lib/site-packages',
            'scripts': '{base}/Scripts',
        }
    else:
        scheme = {
            'purelib': '{base}/lib/python{py_version_short}/site-packages',
            'scripts': '{base}/bin',
        }

    vars_ = sysconfig.get_config_vars()
    if override_vars:
        vars_.update(override_vars)
    return {key: value.format(**vars_) for key, value in scheme.items()}


def create_venv(project_dir: Path):
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    venv_prefix = project_dir / f".venv-{py_major}.{py_minor}"
    print(
        f"-- Using virtual environment at: {venv_prefix} (Python {py_major}.{py_minor})"
    )

    # Ensure compatible virtualenv version is installed (>=20.29.1, <22.0)
    print("-- Ensuring virtualenv version >=20.29.1,<22.0 is installed...")
    build_run(f'"{sys.executable}" -m pip install "virtualenv>=20.29.1,<22.0"')

    # Create venv if it doesn't exist
    if not venv_prefix.exists():
        print(f"-- Creating virtual environment in {venv_prefix}...")
        build_run(
            f'"{sys.executable}" -m virtualenv --system-site-packages "{venv_prefix}"'
        )
    else:
        print("-- Virtual environment already exists.")

    return venv_prefix


def setup_venv(project_dir: Path, requirements_file: Path, no_venv: bool):
    """Creates/updates a venv and installs requirements.

    Args:
        project_dir: The root directory of the project.
        requirements_file: Path to the requirements file.
        no_venv: Use current Python environment as is.

    Returns:
        Tuple[Path, Path]: Paths to the python and conan executables in the venv.
    """
    if no_venv or sys.prefix != sys.base_prefix:
        reason = "Explicitly requested by user" if no_venv else "Already inside virtual environment"
        print(f"-- {reason}, using environment {sys.prefix} as is.")
        venv_prefix = Path(sys.prefix)
    else:
        venv_prefix = create_venv(project_dir)

    scheme = sysconfig_scheme({'base': venv_prefix})
    # Determine venv executable paths
    scripts_dir = Path(scheme["scripts"])
    venv_python = venv_prefix / sys.executable.removeprefix(sys.prefix)[1:]

    if os.environ.get("NVIDIA_PYTORCH_VERSION"):
        # Ensure PyPI PyTorch is not installed in the venv
        purelib_dir = Path(scheme["purelib"])
        pytorch_package_dir = purelib_dir / "torch"
        if str(venv_prefix) != sys.base_prefix and pytorch_package_dir.exists():
            warnings.warn(
                f"Using the NVIDIA PyTorch container with PyPI distributed PyTorch may lead to compatibility issues.\n"
                f"If you encounter any problems, please delete the environment at `{venv_prefix}` so that "
                f"`build_wheel.py` can recreate the virtual environment correctly."
            )
            print("^^^^^^^^^^ IMPORTANT WARNING ^^^^^^^^^^", file=sys.stderr)
            input("Press Ctrl+C to stop, any key to continue...\n")

        # Ensure inherited PyTorch version is compatible
        try:
            info = check_output(
                [str(venv_python), "-m", "pip", "show", "torch"])
        except CalledProcessError:
            raise RuntimeError(
                "NVIDIA PyTorch container detected, but cannot find PyTorch installation. "
                "The environment is corrupted. Please recreate your container.")
        version_installed = next(
            line.removeprefix("Version: ")
            for line in info.decode().splitlines()
            if line.startswith("Version: "))
        version_required = None
        try:
            with open(requirements_file) as fp:
                for line in fp:
                    if line.startswith("torch"):
                        version_required = Requirement(line)
                        break
        except FileNotFoundError:
            pass

        if version_required is not None:
            if version_installed not in version_required.specifier:
                raise RuntimeError(
                    f"Incompatible NVIDIA PyTorch container detected. "
                    f"The container provides PyTorch version {version_installed}, "
                    f"but current revision requires {version_required}. "
                    f"Please recreate your container using image specified in .devcontainer/docker-compose.yml. "
                    f"NOTE: Please don't try install PyTorch using pip. "
                    f"Using the NVIDIA PyTorch container with PyPI distributed PyTorch may lead to compatibility issues."
                )

    # Install/update requirements
    print(
        f"-- Installing requirements from {requirements_file} into {venv_prefix}..."
    )
    build_run(f'"{venv_python}" -m pip install -r "{requirements_file}"')

    venv_conan = setup_conan(scripts_dir, venv_python)

    return venv_python, venv_conan


def setup_conan(scripts_dir, venv_python):
    build_run(f'"{venv_python}" -m pip install conan==2.14.0')
    # Determine the path to the conan executable within the venv
    venv_conan = scripts_dir / "conan"
    if not venv_conan.exists():
        # Attempt to find it using shutil.which as a fallback, in case it's already installed in the system
        try:
            result = build_run(
                f'''{venv_python} -c "import shutil; print(shutil.which('conan'))" ''',
                capture_output=True,
                text=True)
            conan_path_str = result.stdout.strip()

            if conan_path_str:
                venv_conan = Path(conan_path_str)
                print(
                    f"-- Found conan executable via PATH search at: {venv_conan}"
                )
            else:
                raise RuntimeError(
                    f"Failed to locate conan executable in virtual environment {scripts_dir} or system PATH."
                )

        except CalledProcessError as e:
            print(f"Fallback search command output: {e.stdout}",
                  file=sys.stderr)
            print(f"Fallback search command error: {e.stderr}", file=sys.stderr)
            raise RuntimeError(
                f"Failed to locate conan executable in virtual environment {scripts_dir} or system PATH."
            )
    else:
        print(f"-- Found conan executable at: {venv_conan}")

    # Create default profile
    build_run(f'"{venv_conan}" profile detect -f')

    # Add the tensorrt-llm remote if it doesn't exist
    build_run(
        f'"{venv_conan}" remote add --force tensorrt-llm https://edge.urm.nvidia.com/artifactory/api/conan/sw-tensorrt-llm-conan',
        stdout=DEVNULL,
        stderr=DEVNULL)

    return venv_conan


def generate_fmha_cu(project_dir, venv_python):
    fmha_v2_cu_dir = project_dir / "cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmha_v2_cu"
    fmha_v2_cu_dir.mkdir(parents=True, exist_ok=True)

    fmha_v2_dir = project_dir / "cpp/kernels/fmha_v2"
    os.chdir(fmha_v2_dir)

    env = os.environ.copy()
    env.update({
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "ENABLE_SM89_QMMA": "1",
        "ENABLE_HMMA_FP32": "1",
        "GENERATE_CUBIN": "1",
        "SCHEDULING_MODE": "1",
        "ENABLE_SM100": "1",
        "ENABLE_SM120": "1",
        "GENERATE_CU_TRTLLM": "true"
    })

    build_run("rm -rf generated")
    build_run("rm -rf temp")
    build_run("rm -rf obj")
    build_run("python3 setup.py", env=env)

    # Copy generated header file when cu path is active and cubins are deleted.
    cubin_dir = project_dir / "cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin"
    build_run(f"mv generated/fmha_cubin.h {cubin_dir}")

    for cu_file in (fmha_v2_dir / "generated").glob("*sm*.cu"):
        build_run(f"mv {cu_file} {fmha_v2_cu_dir}")

    os.chdir(project_dir)


def main(*,
         build_type: str = "Release",
         generator: str = "",
         build_dir: Path = None,
         dist_dir: Path = None,
         cuda_architectures: str = None,
         job_count: int = None,
         extra_cmake_vars: List[str] = list(),
         extra_make_targets: str = "",
         trt_root: str = '/usr/local/tensorrt',
         nccl_root: str = None,
         nixl_root: str = None,
         internal_cutlass_kernels_root: str = None,
         clean: bool = False,
         clean_wheel: bool = False,
         configure_cmake: bool = False,
         use_ccache: bool = False,
         fast_build: bool = False,
         cpp_only: bool = False,
         install: bool = False,
         skip_building_wheel: bool = False,
         linking_install_binary: bool = False,
         binding_type: str = "pybind",
         benchmarks: bool = False,
         micro_benchmarks: bool = False,
         nvtx: bool = False,
         skip_stubs: bool = False,
         generate_fmha: bool = False,
         no_venv: bool = False,
         nvrtc_dynamic_linking: bool = False):

    if clean:
        clean_wheel = True

    project_dir = get_project_dir()
    os.chdir(project_dir)

    # Get all submodules and check their folder exists. If not,
    # invoke git submodule update
    with open(project_dir / ".gitmodules", "r") as submodules_f:
        submodules = [
            l.split("=")[1].strip() for l in submodules_f.readlines()
            if "path = " in l
        ]
    if any(not (project_dir / submodule / ".git").exists()
           for submodule in submodules):
        build_run('git submodule update --init --recursive')
    on_windows = platform.system() == "Windows"
    requirements_filename = "requirements-dev-windows.txt" if on_windows else "requirements-dev.txt"

    # Setup venv and install requirements
    venv_python, venv_conan = setup_venv(project_dir,
                                         project_dir / requirements_filename,
                                         no_venv)

    # Ensure base TRT is installed (check inside the venv)
    try:
        check_output([str(venv_python), "-m", "pip", "show", "tensorrt"])
    except CalledProcessError:
        error_msg = "TensorRT was not installed properly."
        if on_windows:
            error_msg += (
                " Please download the TensorRT zip file manually,"
                " install it and relaunch build_wheel.py."
                " See https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip for more details."
            )
        else:
            error_msg += f" Please install tensorrt into the venv using \"`{venv_python}` -m pip install tensorrt\" and relaunch build_wheel.py"
        raise RuntimeError(error_msg)

    if cuda_architectures is not None:
        if "70-real" in cuda_architectures:
            raise RuntimeError("Volta architecture is deprecated support.")

    cuda_architectures = cuda_architectures or 'all'
    cmake_cuda_architectures = f'"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}"'

    cmake_def_args = []
    cmake_generator = ""

    if on_windows:
        # Windows does not support multi-device currently.
        extra_cmake_vars.extend(["ENABLE_MULTI_DEVICE=0"])

        # The Ninja CMake generator is used for our Windows build
        # (Easier than MSBuild to make compatible with our Docker image)

    if generator:
        cmake_generator = "-G" + generator

    if job_count is None:
        job_count = cpu_count()

    if len(extra_cmake_vars):
        # Backwards compatibility, we also support semicolon expansion for each value.
        # However, it is best to use flag multiple-times due to issues with spaces in CLI.
        expanded_args = []
        for var in extra_cmake_vars:
            expanded_args += var.split(";")

        extra_cmake_vars = ["\"-D{}\"".format(var) for var in expanded_args]
        # Don't include duplicate conditions
        cmake_def_args.extend(set(extra_cmake_vars))

    if trt_root is not None:
        cmake_def_args.append(f"-DTensorRT_ROOT={trt_root}")

    if nccl_root is not None:
        cmake_def_args.append(f"-DNCCL_ROOT={nccl_root}")

    if nixl_root is not None:
        cmake_def_args.append(f"-DNIXL_ROOT={nixl_root}")

    build_dir = get_build_dir(build_dir, build_type)
    first_build = not Path(build_dir, "CMakeFiles").exists()

    if clean and build_dir.exists():
        clear_folder(build_dir)  # Keep the folder in case it is mounted.
    build_dir.mkdir(parents=True, exist_ok=True)

    def get_binding_type_from_cache():
        cmake_cache_file = build_dir / "CMakeCache.txt"
        if not cmake_cache_file.exists():
            return None

        with open(cmake_cache_file, 'r') as f:
            for line in f:
                if line.startswith("BINDING_TYPE:STRING="):
                    cashed_binding_type = line.split("=", 1)[1].strip()
                    if cashed_binding_type in ['pybind', 'nanobind']:
                        return cashed_binding_type
            return None

    cached_binding_type = get_binding_type_from_cache()

    if not first_build and cached_binding_type != binding_type:
        # Clean up of previous binding build artifacts
        nanobind_dir = build_dir / "tensorrt_llm" / "nanobind"
        if nanobind_dir.exists():
            rmtree(nanobind_dir)
        nanobind_stub_file = project_dir / "tensorrt_llm" / "bindings.pyi"
        if nanobind_stub_file.exists():
            nanobind_stub_file.unlink()

        pybind_dir = build_dir / "tensorrt_llm" / "pybind"
        if pybind_dir.exists():
            rmtree(pybind_dir)
        pybind_stub_dir = project_dir / "tensorrt_llm" / "bindings"
        if pybind_stub_dir.exists():
            rmtree(pybind_stub_dir)

        configure_cmake = True

    if use_ccache:
        cmake_def_args.append(
            f"-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
        )

    if fast_build:
        cmake_def_args.append(f"-DFAST_BUILD=ON")

    if nvrtc_dynamic_linking:
        cmake_def_args.append(f"-DNVRTC_DYNAMIC_LINKING=ON")

    targets = ["tensorrt_llm", "nvinfer_plugin_tensorrt_llm"]

    if cpp_only:
        build_pyt = "OFF"
        build_deep_ep = "OFF"
    else:
        targets.extend(["th_common", "bindings", "deep_ep"])
        build_pyt = "ON"
        build_deep_ep = "ON"

    if benchmarks:
        targets.append("benchmarks")

    if micro_benchmarks:
        targets.append("micro_benchmarks")
        build_micro_benchmarks = "ON"
    else:
        build_micro_benchmarks = "OFF"

    disable_nvtx = "OFF" if nvtx else "ON"

    if not on_windows:
        targets.append("executorWorker")

    source_dir = get_source_dir()

    fmha_v2_cu_dir = project_dir / "cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmha_v2_cu"
    if clean or generate_fmha:
        build_run(f"rm -rf {fmha_v2_cu_dir}")
        generate_fmha_cu(project_dir, venv_python)
    elif not fmha_v2_cu_dir.exists():
        generate_fmha_cu(project_dir, venv_python)

    with working_directory(build_dir):
        if clean or first_build or configure_cmake:
            build_run(
                f"\"{venv_conan}\" install --build=missing --remote=tensorrt-llm --output-folder={build_dir}/conan -s 'build_type={build_type}' {source_dir}"
            )
            cmake_def_args.append(
                f"-DCMAKE_TOOLCHAIN_FILE={build_dir}/conan/conan_toolchain.cmake"
            )
            if internal_cutlass_kernels_root:
                cmake_def_args.append(
                    f"-DINTERNAL_CUTLASS_KERNELS_PATH={internal_cutlass_kernels_root}"
                )
            cmake_def_args = " ".join(cmake_def_args)
            cmake_configure_command = (
                f'cmake -DCMAKE_BUILD_TYPE="{build_type}" -DBUILD_PYT="{build_pyt}" -DBINDING_TYPE="{binding_type}" -DBUILD_DEEP_EP="{build_deep_ep}"'
                f' -DNVTX_DISABLE="{disable_nvtx}" -DBUILD_MICRO_BENCHMARKS={build_micro_benchmarks}'
                f' -DBUILD_WHEEL_TARGETS="{";".join(targets)}"'
                f' -DPython_EXECUTABLE={venv_python} -DPython3_EXECUTABLE={venv_python}'
                f' {cmake_cuda_architectures} {cmake_def_args} {cmake_generator} -S "{source_dir}"'
            )
            print("CMake Configure command: ")
            print(cmake_configure_command)
            build_run(cmake_configure_command)

        cmake_build_command = (
            f'cmake --build . --config {build_type} --parallel {job_count} '
            f'--target build_wheel_targets {" ".join(extra_make_targets)}')
        print("CMake Build command: ")
        print(cmake_build_command)
        build_run(cmake_build_command)

    if cpp_only:
        assert not install, "Installing is not supported for cpp_only builds"
        return

    pkg_dir = project_dir / "tensorrt_llm"
    assert pkg_dir.is_dir(), f"{pkg_dir} is not a directory"
    lib_dir = pkg_dir / "libs"
    include_dir = pkg_dir / "include"
    if lib_dir.exists():
        clear_folder(lib_dir)
    if include_dir.exists():
        clear_folder(include_dir)

    cache_dir = os.getenv("TRTLLM_DG_CACHE_DIR")
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
    elif on_windows:
        if os.getenv("APPDATA") is not None:
            cache_dir = Path(os.getenv("APPDATA")) / "tensorrt_llm"
        else:
            cache_dir = Path(os.getenv("TEMP")) / "tensorrt_llm"
    else:
        if os.getenv("HOME") is not None:
            cache_dir = Path(os.getenv("HOME")) / ".tensorrt_llm"
        else:
            cache_dir = Path(os.getenv("TEMP"), "/tmp") / "tensorrt_llm"
    if cache_dir.exists():
        clear_folder(cache_dir)

    install_file = copy
    install_tree = copytree
    if skip_building_wheel and linking_install_binary:

        def symlink_remove_dst(src, dst):
            src = os.path.abspath(src)
            dst = os.path.abspath(dst)
            if os.path.isdir(dst):
                dst = os.path.join(dst, os.path.basename(src))
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)

        install_file = symlink_remove_dst

        def symlink_remove_dst_tree(src, dst, dirs_exist_ok=True):
            src = os.path.abspath(src)
            dst = os.path.abspath(dst)
            if dirs_exist_ok and os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)

        install_tree = symlink_remove_dst_tree

    lib_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)
    install_tree(get_source_dir() / "include" / "tensorrt_llm" / "deep_gemm",
                 include_dir / "deep_gemm",
                 dirs_exist_ok=True)
    required_cuda_headers = [
        "cuda_fp16.h", "cuda_fp16.hpp", "cuda_bf16.h", "cuda_bf16.hpp",
        "cuda_fp8.h", "cuda_fp8.hpp"
    ]
    if os.getenv("CUDA_HOME") is not None:
        cuda_include_dir = Path(os.getenv("CUDA_HOME")) / "include"
    elif os.getenv("CUDA_PATH") is not None:
        cuda_include_dir = Path(os.getenv("CUDA_PATH")) / "include"
    elif not on_windows:
        cuda_include_dir = Path("/usr/local/cuda/include")
    else:
        cuda_include_dir = None

    if cuda_include_dir is None or not cuda_include_dir.exists():
        print(
            "CUDA_HOME or CUDA_PATH should be set to enable DeepGEMM JIT compilation"
        )
    else:
        cuda_include_target_dir = include_dir / "cuda" / "include"
        cuda_include_target_dir.mkdir(parents=True, exist_ok=True)
        for header in required_cuda_headers:
            install_file(cuda_include_dir / header, include_dir / header)

    if on_windows:
        install_file(build_dir / "tensorrt_llm/tensorrt_llm.dll",
                     lib_dir / "tensorrt_llm.dll")
        install_file(build_dir / f"tensorrt_llm/thop/th_common.dll",
                     lib_dir / "th_common.dll")
        install_file(
            build_dir / f"tensorrt_llm/plugins/nvinfer_plugin_tensorrt_llm.dll",
            lib_dir / "nvinfer_plugin_tensorrt_llm.dll")
    else:
        install_file(build_dir / "tensorrt_llm/libtensorrt_llm.so",
                     lib_dir / "libtensorrt_llm.so")
        install_file(build_dir / "tensorrt_llm/thop/libth_common.so",
                     lib_dir / "libth_common.so")
        install_file(
            build_dir /
            "tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so",
            lib_dir / "libnvinfer_plugin_tensorrt_llm.so")
        if os.path.exists(
                build_dir /
                "tensorrt_llm/executor/cache_transmission/ucx_utils/libtensorrt_llm_ucx_wrapper.so"
        ):
            install_file(
                build_dir /
                "tensorrt_llm/executor/cache_transmission/ucx_utils/libtensorrt_llm_ucx_wrapper.so",
                lib_dir / "libtensorrt_llm_ucx_wrapper.so")
        if os.path.exists(
                build_dir /
                "tensorrt_llm/executor/cache_transmission/nixl_utils/libtensorrt_llm_nixl_wrapper.so"
        ):
            install_file(
                build_dir /
                "tensorrt_llm/executor/cache_transmission/nixl_utils/libtensorrt_llm_nixl_wrapper.so",
                lib_dir / "libtensorrt_llm_nixl_wrapper.so")
        install_file(
            build_dir /
            "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_0.so",
            lib_dir / "libdecoder_attention_0.so")
        install_file(
            build_dir /
            "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_1.so",
            lib_dir / "libdecoder_attention_1.so")

    deep_ep_dir = pkg_dir / "deep_ep"
    if deep_ep_dir.is_symlink():
        deep_ep_dir.unlink()
    elif deep_ep_dir.is_dir():
        clear_folder(deep_ep_dir)
        deep_ep_dir.rmdir()

    bin_dir = pkg_dir / "bin"
    if bin_dir.exists():
        clear_folder(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)

    if not on_windows:
        install_file(build_dir / "tensorrt_llm/executor_worker/executorWorker",
                     bin_dir / "executorWorker")

    if not cpp_only:

        def get_binding_lib(subdirectory, name):
            binding_build_dir = (build_dir / "tensorrt_llm" / subdirectory)
            if on_windows:
                binding_lib = list(binding_build_dir.glob(f"{name}.*.pyd"))
            else:
                binding_lib = list(binding_build_dir.glob(f"{name}.*.so"))

            assert len(
                binding_lib
            ) == 1, f"Exactly one binding library should be present: {binding_lib}"
            return binding_lib[0]

        install_file(get_binding_lib(binding_type, "bindings"), pkg_dir)

        with (build_dir / "tensorrt_llm" / "deep_ep" /
              "cuda_architectures.txt").open() as f:
            deep_ep_cuda_architectures = f.read().strip().strip(";")
        if deep_ep_cuda_architectures:
            install_file(get_binding_lib("deep_ep", "deep_ep_cpp_tllm"),
                         pkg_dir)
            install_tree(build_dir / "tensorrt_llm" / "deep_ep" / "python" /
                         "deep_ep",
                         deep_ep_dir,
                         dirs_exist_ok=True)
            (lib_dir / "nvshmem").mkdir(exist_ok=True)
            install_file(
                build_dir / "tensorrt_llm/deep_ep/nvshmem-build/License.txt",
                lib_dir / "nvshmem")
            install_file(
                build_dir /
                "tensorrt_llm/deep_ep/nvshmem-build/src/lib/nvshmem_bootstrap_uid.so.3",
                lib_dir / "nvshmem")
            install_file(
                build_dir /
                "tensorrt_llm/deep_ep/nvshmem-build/src/lib/nvshmem_transport_ibgda.so.103",
                lib_dir / "nvshmem")
        if not skip_stubs:
            with working_directory(project_dir):
                if binding_type == "nanobind":
                    build_run(f"\"{venv_python}\" -m pip install nanobind")
                else:
                    build_run(
                        f"\"{venv_python}\" -m pip install pybind11-stubgen")
            with working_directory(pkg_dir):
                if on_windows:
                    if binding_type == "nanobind":
                        print("Windows not yet supported for nanobind stubs")
                        exit(1)
                    else:
                        stubgen = "stubgen.py"
                        stubgen_contents = """
                        # Loading torch, trt before bindings is required to avoid import errors on windows.
                        # isort: off
                        import torch
                        import tensorrt as trt
                        # isort: on
                        import os
                        import platform

                        from pybind11_stubgen import main

                        if __name__ == "__main__":
                            # Load dlls from `libs` directory before launching bindings.
                            if platform.system() == "Windows":
                                os.add_dll_directory(r\"{lib_dir}\")
                            main()
                        """.format(lib_dir=lib_dir)
                        (pkg_dir / stubgen).write_text(dedent(stubgen_contents))
                        build_run(f"\"{venv_python}\" {stubgen} -o . bindings")
                        (pkg_dir / stubgen).unlink()
                else:
                    env_ld = os.environ.copy()

                    new_library_path = "/usr/local/cuda/compat:/usr/local/cuda/compat/lib:/usr/local/cuda/compat/lib.real"
                    if 'LD_LIBRARY_PATH' in env_ld:
                        new_library_path += f":{env_ld['LD_LIBRARY_PATH']}"

                    result = build_run("find /usr -name *libnvidia-ml.so*",
                                       capture_output=True,
                                       text=True)
                    assert result.returncode == 0, f"Failed to run find *libnvidia-ml.so*: {result.stderr}"

                    # Build containers only contain stub version of libnvidia-ml.so and not the real version.
                    # If real version not in system, we need to create symbolic link to stub version to prevent import errors.
                    if "libnvidia-ml.so.1" not in result.stdout:
                        if "libnvidia-ml.so" in result.stdout:
                            line = result.stdout.splitlines()[0]
                            path = os.path.dirname(line)
                            new_library_path += f":{path}"
                            build_run(f"ln -s {line} {path}/libnvidia-ml.so.1")
                        else:
                            print(
                                f"Failed to find libnvidia-ml.so: {result.stderr}",
                                file=sys.stderr)
                            exit(1)

                    env_ld["LD_LIBRARY_PATH"] = new_library_path
                    if binding_type == "nanobind":
                        build_run(
                            f"\"{venv_python}\" -m nanobind.stubgen -m bindings -O .",
                            env=env_ld)
                    else:
                        build_run(
                            f"\"{venv_python}\" -m pybind11_stubgen -o . bindings --exit-code",
                            env=env_ld)
                        if deep_ep_cuda_architectures:
                            build_run(
                                f"\"{venv_python}\" -m pybind11_stubgen -o . deep_ep_cpp_tllm --exit-code",
                                env=env_ld)

    if not skip_building_wheel:
        if dist_dir is None:
            dist_dir = project_dir / "build"
        else:
            dist_dir = Path(dist_dir)

        if not dist_dir.exists():
            dist_dir.mkdir(parents=True)

        if clean_wheel:
            # For incremental build, the python build module adds
            # the new files but does not remove the deleted files.
            #
            # This breaks the Windows CI/CD pipeline when building
            # and validating python changes in the whl.
            clear_folder(dist_dir)

        build_run(
            f'\"{venv_python}\" -m build {project_dir} --skip-dependency-check --no-isolation --wheel --outdir "{dist_dir}"'
        )

    if install:
        build_run(f"\"{sys.executable}\" -m pip install -e .[devel]")


def add_arguments(parser: ArgumentParser):
    parser.add_argument("--build_type",
                        "-b",
                        default="Release",
                        choices=["Release", "RelWithDebInfo", "Debug"])
    parser.add_argument("--generator", "-G", default="")
    parser.add_argument("--cuda_architectures", "-a")
    parser.add_argument("--install", "-i", action="store_true")
    parser.add_argument("--clean", "-c", action="store_true")
    parser.add_argument("--clean_wheel",
                        action="store_true",
                        help="Clear dist_dir folder creating wheel")
    parser.add_argument("--configure_cmake",
                        action="store_true",
                        help="Always configure cmake before building")
    parser.add_argument("--use_ccache",
                        "-ccache",
                        default=False,
                        action="store_true",
                        help="Use ccache compiler driver")
    parser.add_argument(
        "--fast_build",
        "-f",
        default=False,
        action="store_true",
        help=
        "Skip compiling some kernels to accelerate compilation -- for development only"
    )
    parser.add_argument("--job_count",
                        "-j",
                        const=cpu_count(),
                        nargs="?",
                        help="Parallel job count")
    parser.add_argument(
        "--cpp_only",
        "-l",
        action="store_true",
        help="Only build the C++ library without Python dependencies")
    parser.add_argument(
        "--extra-cmake-vars",
        "-D",
        action="append",
        help=
        "Extra cmake variable definition which can be specified multiple times, example: -D \"key1=value1\" -D \"key2=value2\"",
        default=[])
    parser.add_argument(
        "--extra-make-targets",
        help="A list of additional make targets, example: \"target_1 target_2\"",
        nargs="+",
        default=[])
    parser.add_argument("--trt_root",
                        default="/usr/local/tensorrt",
                        help="Directory to find TensorRT headers/libs")
    parser.add_argument("--nccl_root",
                        help="Directory to find NCCL headers/libs")
    parser.add_argument("--nixl_root",
                        help="Directory to find NIXL headers/libs")
    parser.add_argument(
        "--internal-cutlass-kernels-root",
        default="",
        help=
        "Directory to the internal_cutlass_kernels sources. If specified, the internal_cutlass_kernels and NVRTC wrapper libraries will be built from source."
    )
    parser.add_argument("--build_dir",
                        type=Path,
                        help="Directory where cpp sources are built")
    parser.add_argument("--dist_dir",
                        type=Path,
                        help="Directory where python wheels are built")
    parser.add_argument(
        "--skip_building_wheel",
        "-s",
        action="store_true",
        help=
        "Do not build the *.whl files (they are only needed for distribution).")
    parser.add_argument(
        "--linking_install_binary",
        action="store_true",
        help="Install the built binary by symbolic linking instead of copying.")
    parser.add_argument("--binding_type",
                        choices=["pybind", "nanobind"],
                        default="pybind",
                        help="Which binding type to build: pybind, nanobind")
    parser.add_argument("--benchmarks",
                        action="store_true",
                        help="Build the benchmarks for the C++ runtime.")
    parser.add_argument("--micro_benchmarks",
                        action="store_true",
                        help="Build the micro benchmarks for C++ components.")
    parser.add_argument("--nvtx",
                        action="store_true",
                        help="Enable NVTX features.")
    parser.add_argument("--skip-stubs",
                        action="store_true",
                        help="Skip building python stubs")
    parser.add_argument("--generate_fmha",
                        action="store_true",
                        help="Generate the FMHA cu files.")
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help=
        "Use the current Python interpreter without creating a virtual environment."
    )
    parser.add_argument(
        "--nvrtc_dynamic_linking",
        action="store_true",
        help="Link against the dynamic NVRTC libraries and not the static ones."
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(**vars(args))
