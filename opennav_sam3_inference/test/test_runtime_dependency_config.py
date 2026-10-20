# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only checks for the pinned Docker SAM3 runtime configuration."""

import ast
import importlib.util
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / 'opennav_sam3_setup'
DOCKER = SETUP / 'docker'
SCRIPT = SETUP / 'setup.sh'
BUILD = DOCKER / 'build.sh'
RUN = DOCKER / 'run.sh'
RUN_NODE = DOCKER / 'run_node.sh'
DOCKERFILE = DOCKER / 'Dockerfile'


def _shell_function(name: str) -> str:
    """Extract one top-level shell function for isolated behavior tests."""
    source = SCRIPT.read_text()
    start = source.index(f'{name}() {{')
    end = source.index('\n}\n', start) + 3
    return source[start:end]


OFFICIAL_WHEELS = {
    'torch-2.14.0+rocm10.1.0-cp312-cp312-linux_x86_64.whl':
        '3182f6f166f42d6f186fd8c4a4ea65e1ed02fba8a98c48f977b68f15cca82636',
    'amd_torch_device_gfx1151-2.14.0+rocm10.1.0-cp312-cp312-linux_x86_64.whl':
        '2e340119752d93d639d3baf63b33f92190260738dff3476de374adaf92fb30ef',
    'amd_torch_device_gfx115x-2.14.0+rocm10.1.0-cp312-cp312-linux_x86_64.whl':
        '73daa37345172b0556f8a20687197022b8f4c48dd74db5707b03780cc5e71a4d',
    'torchvision-0.29.0a0+rocm10.1.0-cp312-cp312-linux_x86_64.whl':
        'f0eed016b74d09192969a7db2f4018941f662c413d55b0eeb157b99941e96e5c',
    'amd_torchvision_device_gfx1151-0.29.0a0+rocm10.1.0-cp312-cp312-linux_x86_64.whl':
        '99650eae5ecc34cff9b8d57afd035f174ce7a14a4d66bb2a4bea4e4fc2f86d0d',
    'triton-3.8.0+git669b31ac.rocm10.1.0-cp312-cp312-linux_x86_64.whl':
        '2f745c62aebb33167bfb6d586f569fbea67f8fc5181b03798b91a6b9187a625e',
    'migraphx-2.18.0+rocm10.1.0-cp312-none-manylinux_2_28_x86_64.whl':
        '6a22d67926759f3ed02ba8d38e31b02e0f69579c5c2893a30745aa0708358c25',
    'migraphx_libs-2.18.0+rocm10.1.0-py3-none-manylinux_2_28_x86_64.whl':
        '0a625d0fc835db4727af8b6199a0a85bcaa0578cc2a873f6fbbe36b70d84713b',
    'onnxruntime_ep_migraphx-1.0.0+rocm10.1.0-cp312-cp312-manylinux_2_28_x86_64.whl':
        'f43aafe7581fc7f82f94518424c86418371cdcc31c2cb399cec14e7d5c8cd19c',
    'onnxruntime-1.29.0-cp312-cp312-manylinux_2_28_x86_64.whl':
        '2b80d8c7ec2cc7438e4da3760b88c24568cba72c9ace96d668800a6c79419acb',
}


def _supported_stack():
    """Read SUPPORTED_STACK without importing ONNX Runtime."""
    path = ROOT / 'opennav_sam3_inference/opennav_sam3_inference/tracker/ort_plugin_ep.py'
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'SUPPORTED_STACK'
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError('SUPPORTED_STACK not found')


def test_published_runtime_pins_are_exact():
    """Keep the image tied to the official ROCm 10.1 wheels and nothing else."""
    build = BUILD.read_text()
    rows = {}
    for line in build.splitlines():
        if line.startswith('"') and line.endswith('.whl"'):
            expected, url, name = line.strip('"').split()
            rows[name] = (expected, url)
    assert {name: row[0] for name, row in rows.items()} == OFFICIAL_WHEELS
    bases = ('${TORCH_BASE}/', '${MGX_BASE}/', '${EP_BASE}/', '${PYPI_BASE}/')
    for _expected, url in rows.values():
        assert url.startswith(bases), url
    assert 'TORCH_BASE="https://stable.repo.amd.com/rocm/pytorch/whl-next"' in build
    assert 'MGX_BASE="https://stable.repo.amd.com/rocm/migraphx/whl-next"' in build
    assert 'EP_BASE="https://stable.repo.amd.com/rocm/onnxruntime/whl-next"' in build
    assert 'PYPI_BASE="https://files.pythonhosted.org/packages/' in build
    for removed in ('github.com', 'fc1sink', 'MIGRAPHX_ARCHIVE', 'ORT_WHEEL_PATH'):
        assert removed not in build
    stack = _supported_stack()
    assert stack['migraphx'] in build.replace('%2B', '+')
    assert stack['ep_package'] in build.replace('%2B', '+')
    assert f"onnxruntime-{stack['onnxruntime_prefix']}" in build
    source = SCRIPT.read_text()
    assert 'OFFICIAL_MODEL_REPO="facebook/sam3"' in source
    assert (
        'OFFICIAL_MODEL_REVISION="3c879f39826c281e95690f02c7821c4de09afae7"'
        in source
    )
    assert 'MIRROR_MODEL_REPO="1038lab/sam3"' in source
    assert (
        'MIRROR_MODEL_REVISION="fe5e2ae858f82b44a0b18e80aa09fe1d5ed75a0b"'
        in source
    )
    assert (
        'MODEL_SHA256="6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a"'
        in source
    )
    assert 'MIRROR_MODEL_NAME="sam3.safetensors"' in source
    assert 'ea8e153c669a0284a496c0ec65a53b8e4f5ca7e7' not in source
    dockerfile = DOCKERFILE.read_text()
    from_lines = [line for line in dockerfile.splitlines() if line.startswith('FROM ')]
    assert from_lines == [
        'FROM ros:jazzy-ros-base@sha256:'
        '103b6c417973394659af6804d4917cd489b91e5ed22eb7bef263025a3bed2410'
    ]
    for value in (
        'ROCM_VERSION=10.1',
        'ROCM_PACKAGE_VERSION=10.1.0-3',
        'https://stable.repo.amd.com/rocm/core/packages/ubuntu2404',
        'COPY --from=wheels / /tmp/wheels/',
        '--no-deps /tmp/wheels/*.whl',
    ):
        assert value in dockerfile
    for removed in ('ubuntu:24.04', 'migraphx-develop', 'packages.ros.org', 'ros-dev-tools'):
        assert removed not in dockerfile
    for package in (
        'amdrocm-runtime-dev',
        'amdrocm-blas-dev',
        'amdrocm-blas-host',
        'amdrocm-dnn-dev',
        'amdrocm-dnn-host',
        'amdrocm-hipblas-common-dev',
        'amdrocm-blas${ROCM_VERSION}-gfx1151',
        'amdrocm-dnn${ROCM_VERSION}-gfx1151',
        'amdrocm-fft-host',
        'amdrocm-fft${ROCM_VERSION}-gfx1151',
        'amdrocm-rand-host',
        'amdrocm-rand${ROCM_VERSION}-gfx1151',
        'amdrocm-rccl-host',
        'amdrocm-rccl${ROCM_VERSION}-gfx1151',
        'amdrocm-hipfile${ROCM_VERSION}',
        'amdrocm-hipfile-dev${ROCM_VERSION}',
        'ros-${ROS_DISTRO}-rmw-cyclonedds-cpp',
        'ros-${ROS_DISTRO}-cv-bridge',
    ):
        assert package in dockerfile
    requirements = (SETUP / 'requirements.txt').read_text()
    constraints = (SETUP / 'runtime-constraints.txt').read_text()
    assert 'scipy>=1.17.1' in requirements
    assert 'scipy==1.17.1' in constraints
    assert 'onnxsim>=0.7.3' in requirements
    assert 'onnx-simplifier' not in requirements
    assert 'onnx-simplifier' not in constraints
    assert 'numpy==1.26.4' in constraints
    assert 'onnxruntime==1.29.0' in constraints
    for removed in ('setuptools==', 'colcon-core==', 'pytest=='):
        assert removed not in constraints


def test_shell_entrypoints_and_help_are_side_effect_free():
    """Parse shell files and ensure help exits before any Docker operation."""
    for path in (SCRIPT, BUILD, RUN, RUN_NODE, DOCKER / 'entrypoint.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    result = subprocess.run(
        ['bash', str(SCRIPT), '--help'],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert '--skip-image' in result.stdout
    assert '--skip-weights' in result.stdout
    assert 'ROCm 10.1' in result.stdout
    source = SCRIPT.read_text()
    assert 'if ((EUID == 0))' in source
    assert 'Do not run setup.sh as root' in source
    assert source.index('-h|--help)') < source.index('if ((EUID == 0))')
    assert source.index('-h|--help)') < source.index('"${REPO_DIR}/docker/build.sh"')

    build_help = subprocess.run(
        ['bash', str(BUILD), '--help'],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert build_help.returncode == 0
    assert '--no-smoke' in build_help.stdout

    conflict = subprocess.run(
        ['bash', str(SCRIPT), '--yes', '--skip-weights'],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert conflict.returncode != 0
    assert 'mutually exclusive' in conflict.stderr


def test_model_configs_and_checkpoint_are_verified_independently():
    """Install configs on every run and keep checkpoint handling optional."""
    source = SCRIPT.read_text()
    model_step = source.split('step "2. Model weights"', 1)[1]
    assert model_step.index('install_model_configs') < model_step.index(
        'if [[ -f "${WEIGHT_FILE}" ]]'
    )
    assert 'verify_model_weight' in model_step
    assert 'elif "${SKIP_WEIGHTS}"' in model_step
    assert 'elif [[ ! -t 0 ]]' in model_step
    assert model_step.index('verify_model_weight "${mirror_weight}"') < (
        model_step.index('mv -- "${mirror_weight}" "${WEIGHT_FILE}"')
    )
    assert '--revision "${MIRROR_MODEL_REVISION}"' in model_step
    for config in (SETUP / 'model-configs').iterdir():
        if config.is_file():
            assert config.name in source


def test_gpu_device_preflight_accepts_access_and_rejects_missing_render(
    tmp_path,
):
    """Require usable KFD and render-node device paths before installation."""
    function = _shell_function('check_gpu_device_access')
    valid = tmp_path / 'valid'
    (valid / 'dri').mkdir(parents=True)
    (valid / 'kfd').touch()
    (valid / 'dri/renderD128').touch()
    command = f"""
set -euo pipefail
die() {{ printf '%s\n' "$*" >&2; exit 1; }}
{function}
check_gpu_device_access "$1"
"""
    subprocess.run(
        ['bash', '--noprofile', '--norc', '-c', command, 'bash', str(valid)],
        check=True,
    )

    missing_render = tmp_path / 'missing-render'
    missing_render.mkdir()
    (missing_render / 'kfd').touch()
    result = subprocess.run(
        [
            'bash',
            '--noprofile',
            '--norc',
            '-c',
            command,
            'bash',
            str(missing_render),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'No GPU render node found' in result.stderr

    denied = tmp_path / 'denied'
    (denied / 'dri').mkdir(parents=True)
    (denied / 'kfd').touch(mode=0o000)
    (denied / 'dri/renderD128').touch()
    result = subprocess.run(
        ['bash', '--noprofile', '--norc', '-c', command, 'bash', str(denied)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'No read/write access' in result.stderr


def test_system_rocm_shim_uses_configured_root(tmp_path, monkeypatch):
    """Resolve libraries from ROCM_PATH rather than Python ROCm wheels."""
    library = tmp_path / 'lib/libamdhip64.so'
    library.parent.mkdir()
    library.touch()
    monkeypatch.setenv('ROCM_PATH', str(tmp_path))
    path = SETUP / 'rocm_sdk_system.py'
    spec = importlib.util.spec_from_file_location('rocm_sdk_system', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__version__ == '10.1.0'
    assert module.find_libraries('amdhip64') == [library]
