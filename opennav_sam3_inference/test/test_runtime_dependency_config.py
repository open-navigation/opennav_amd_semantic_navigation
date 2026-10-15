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


def test_published_runtime_pins_are_exact():
    """Keep the release URLs and hashes tied to the validated rc4 binaries."""
    build = BUILD.read_text()
    assert 'v2.17.0%2Bsam3-fc1sink.20260908.1' in build
    assert (
        'ed1458c632eb2f0e2cab3c457aee93e39196cbb77d2180e47525e0009563dac1'
        in build
    )
    assert 'sam3-tracker-rocm/releases/download/v0.2.0-rc4' in build
    assert (
        'ef10e3e808e8805c26cc27f47572a53e385463f29d578e1ea2fe13d00e6f5ee0'
        in build
    )
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
    for value in (
        'ROCM_PACKAGE_VERSION=7.14.1-0',
        'TORCH_VERSION=2.11.0+rocm7.13.0',
        'TORCHVISION_VERSION=0.26.0+rocm7.13.0',
        'TRITON_VERSION=3.6.0+rocm7.13.0',
        'onnxruntime_migraphx-1.24.2-cp312',
    ):
        assert value in dockerfile
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
    ):
        assert package in dockerfile
    requirements = (SETUP / 'requirements.txt').read_text()
    constraints = (SETUP / 'runtime-constraints.txt').read_text()
    assert 'scipy>=1.17.1' in requirements
    assert 'scipy==1.17.1' in constraints
    assert 'onnxsim>=0.7.3' in requirements
    assert 'onnx-simplifier' not in requirements
    assert 'onnx-simplifier' not in constraints
    assert 'setuptools==79.0.1' in constraints
    assert 'wheel==0.42.0' in constraints


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
    assert 'ROCm 7.14' in result.stdout
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
    assert module.__version__ == '7.14.0'
    assert module.find_libraries('amdhip64') == [library]
