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

"""CPU-only checks for the pinned host SAM3 runtime configuration."""

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / 'opennav_sam3_setup'
SCRIPT = SETUP / 'setup.sh'
ACTIVATE = SETUP / 'conda/activate.d/opennav_sam3_runtime.sh'
DEACTIVATE = SETUP / 'conda/deactivate.d/opennav_sam3_runtime.sh'
LAUNCH = ROOT / 'opennav_sam3_inference/launch/sam3_inference.launch.py'
RUNTIME = ROOT / (
    'opennav_sam3_inference/opennav_sam3_inference/tracker/'
    'migraphx_runtime.py'
)


def _shell_function(name: str) -> str:
    """Extract one top-level shell function for isolated behavior tests."""
    source = SCRIPT.read_text()
    start = source.index(f'{name}() {{')
    end = source.index('\n}\n', start) + 3
    return source[start:end]


def test_published_runtime_pins_are_exact():
    """Keep the release URLs and hashes tied to the validated rc4 binaries."""
    source = SCRIPT.read_text()
    assert 'v2.17.0%2Bsam3-fc1sink.20260908.1' in source
    assert (
        'ed1458c632eb2f0e2cab3c457aee93e39196cbb77d2180e47525e0009563dac1'
        in source
    )
    assert 'sam3-tracker-rocm/releases/download/v0.2.0-rc4' in source
    assert (
        'ef10e3e808e8805c26cc27f47572a53e385463f29d578e1ea2fe13d00e6f5ee0'
        in source
    )
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
    for value in (
        '2.11.0+rocm7.13.0',
        '0.26.0+rocm7.13.0',
        '3.6.0+rocm7.13.0',
        '7.14.1-0',
        'scipy1.17.1',
    ):
        assert value in source
    for package in (
        'amdrocm-runtime-dev7.14',
        'amdrocm-blas-dev7.14',
        'amdrocm-blas-host7.14',
        'amdrocm-dnn-dev7.14',
        'amdrocm-dnn-host7.14',
        'amdrocm-hipblas-common-dev7.14',
        'amdrocm-blas7.14-gfx1151',
        'amdrocm-dnn7.14-gfx1151',
        'amdrocm-fft-host7.14',
        'amdrocm-fft7.14-gfx1151',
        'amdrocm-rand-host7.14',
        'amdrocm-rand7.14-gfx1151',
        'amdrocm-rccl-host7.14',
        'amdrocm-rccl7.14-gfx1151',
    ):
        assert package in source
    requirements = (SETUP / 'requirements.txt').read_text()
    constraints = (SETUP / 'runtime-constraints.txt').read_text()
    assert 'scipy>=1.17.1' in requirements
    assert 'scipy==1.17.1' in constraints
    assert 'onnxsim>=0.7.3' in requirements
    assert 'onnx-simplifier' not in requirements
    assert 'onnx-simplifier' not in constraints
    assert 'colcon-common-extensions' not in requirements
    assert 'colcon-common-extensions==0.3.0' in source
    assert 'colcon-core==0.20.1' in constraints
    assert 'pytest==8.3.5' in constraints
    assert 'setuptools==79.0.1' in constraints
    assert 'wheel==0.42.0' in constraints
    toolchain_install = source.split('step "4. Pinned Python runtime"', 1)[1]
    assert 'setuptools wheel' in toolchain_install.split('-r "${REPO_DIR}', 1)[0]
    colcon_install = source.split("colcon's EmPy dependency", 1)[1]
    assert '--only-binary=:all:' not in colcon_install.split('site_packages=', 1)[0]


def test_runtime_scripts_have_no_legacy_or_source_build_path():
    """Reject the previous ABI, download source, and developer checkout paths."""
    sources = '\n'.join(
        path.read_text()
        for path in (
            SCRIPT,
            LAUNCH,
            RUNTIME,
            SETUP / 'export/backbone/compile_backbone_mxr.py',
            SETUP / 'export/tracker_modules/prewarm_ort_cache.py',
        )
    )
    for forbidden in (
        '/opt/rocm-7.2',
        '2016000',
        'v2.15+',
        'rocm.nightlies',
        'Looong01',
        '/home/amd/project/tools/AMDMIGraphX',
        'install_migraphx_patched.sh',
        'git clone',
    ):
        assert forbidden not in sources
    assert 'LD_PRELOAD' not in LAUNCH.read_text()


def test_shell_entrypoints_and_help_are_side_effect_free():
    """Parse shell files and ensure help exits before privileged operations."""
    for path in (SCRIPT, ACTIVATE, DEACTIVATE):
        subprocess.run(['bash', '-n', str(path)], check=True)
    result = subprocess.run(
        ['bash', str(SCRIPT), '--help'],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert '--recreate-env' in result.stdout
    assert '--skip-weights' in result.stdout
    assert 'ROCm 7.14' in result.stdout
    source = SCRIPT.read_text()
    assert '. /etc/os-release' in source
    assert 'The published host runtime requires Ubuntu 24.04' in source
    assert 'SUDO=()' in source
    assert 'if ((EUID == 0))' in source
    assert 'Do not run setup.sh as root' in source
    assert source.index('-h|--help)') < source.index('if ((EUID == 0))')

    conflict = subprocess.run(
        ['bash', str(SCRIPT), '--yes', '--skip-weights'],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert conflict.returncode != 0
    assert 'mutually exclusive' in conflict.stderr


def test_python_version_probe_is_executable():
    """Keep the setup-time Python probe valid through shell quoting."""
    probe = (
        'import sys; '
        'print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    )
    assert f"python -c '{probe}'" in SCRIPT.read_text()
    result = subprocess.run(
        [sys.executable, '-c', probe], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == f'{sys.version_info.major}.{sys.version_info.minor}'


def test_model_configs_and_checkpoint_are_verified_independently():
    """Install configs on every run and keep checkpoint handling optional."""
    source = SCRIPT.read_text()
    model_step = source.split('step "5. Model weights"', 1)[1]
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


def test_conda_hooks_persist_custom_roots_and_restore_state(tmp_path):
    """Activation consumes env-local roots and deactivation restores state."""
    conda_prefix = tmp_path / 'conda'
    config = conda_prefix / 'etc/opennav_sam3_runtime.conf'
    config.parent.mkdir(parents=True)
    config.write_text(
        'SAM3_MIGRAPHX_ROOT=/custom/migraphx\n'
        'SAM3_ROCM_PATH=/custom/rocm\n'
    )
    command = f"""
set -euo pipefail
unset PYTHONPATH LD_LIBRARY_PATH ROCM_PATH SAM3_MIGRAPHX_ROOT
unset HSA_OVERRIDE_GFX_VERSION MIGRAPHX_GPU_HIP_FLAGS TRANSFORMERS_OFFLINE
before_path="$PATH"
PATH="{conda_prefix}/bin:$PATH"
source "{ACTIVATE}"
first_path="$PATH"
source "{ACTIVATE}"
[[ "$PATH" == "$first_path" ]]
[[ "$SAM3_MIGRAPHX_ROOT" == "/custom/migraphx" ]]
[[ "$SAM3_ROCM_PATH" == "/custom/rocm" ]]
[[ "$ROCM_PATH" == "/custom/rocm" ]]
[[ -z "${{LD_PRELOAD+x}}" ]]
# Conda removes its environment bin before it sources deactivate.d hooks.
PATH="/custom/migraphx/bin:/custom/rocm/bin:$before_path"
source "{DEACTIVATE}"
[[ "$PATH" == "$before_path" ]]
[[ -z "${{PYTHONPATH+x}}" ]]
[[ -z "${{LD_LIBRARY_PATH+x}}" ]]
[[ -z "${{ROCM_PATH+x}}" ]]
[[ -z "${{SAM3_ROCM_PATH+x}}" ]]
[[ -z "${{SAM3_MIGRAPHX_ROOT+x}}" ]]
"""
    subprocess.run(
        ['bash', '--noprofile', '--norc', '-c', command],
        check=True,
        env={
            'CONDA_PREFIX': str(conda_prefix),
            'HOME': os.environ.get('HOME', '/tmp'),
            'PATH': '/usr/bin:/bin',
        },
    )
    assert '_OPENNAV_SAM3_SAVED_PATH' not in ACTIVATE.read_text()
    assert '_OPENNAV_SAM3_SAVED_PATH' not in DEACTIVATE.read_text()


def test_active_hook_refreshes_rewritten_runtime_config(tmp_path):
    """Reload changed runtime roots without retaining old PATH entries."""
    conda_prefix = tmp_path / 'conda'
    config = conda_prefix / 'etc/opennav_sam3_runtime.conf'
    config.parent.mkdir(parents=True)
    config.write_text(
        'SAM3_MIGRAPHX_ROOT=/old/migraphx\n'
        'SAM3_ROCM_PATH=/old/rocm\n'
    )
    command = f"""
set -euo pipefail
export CONDA_PREFIX="{conda_prefix}"
export PATH="{conda_prefix}/bin:/custom/bin:/usr/bin:/custom/bin"
original_path="$PATH"
source "{ACTIVATE}"
[[ "$PATH" == "/old/migraphx/bin:/old/rocm/bin:$original_path" ]]
source "{DEACTIVATE}"
printf '%s\n' \
  'SAM3_MIGRAPHX_ROOT=/new/migraphx' \
  'SAM3_ROCM_PATH=/new/rocm' > "{config}"
source "{ACTIVATE}"
[[ "$PATH" == "/new/migraphx/bin:/new/rocm/bin:$original_path" ]]
[[ "$PATH" != *"/old/migraphx/bin"* ]]
source "{DEACTIVATE}"
[[ "$PATH" == "$original_path" ]]
"""
    subprocess.run(
        ['bash', '--noprofile', '--norc', '-c', command],
        check=True,
        env={'HOME': os.environ.get('HOME', '/tmp'), 'PATH': '/usr/bin:/bin'},
    )
    source = SCRIPT.read_text()
    deactivate = 'source "${existing_deactivate}"'
    install_hook = 'install -m 0644 \\\n    "${REPO_DIR}/conda/activate.d'
    reactivate = 'source "${activate_dir}/opennav_sam3_runtime.sh"'
    assert source.index(deactivate) < source.index(install_hook)
    assert source.index(install_hook) < source.index(reactivate)


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


def test_corrupt_managed_cache_is_replaced_atomically(tmp_path):
    """Replace a bad managed cache but preserve explicit invalid inputs."""
    function = _shell_function('fetch_verified')
    good = tmp_path / 'good.bin'
    good.write_bytes(b'published runtime')
    expected = hashlib.sha256(good.read_bytes()).hexdigest()
    cached = tmp_path / 'cached.bin'
    cached.write_bytes(b'corrupt')
    command = f"""
set -euo pipefail
warn() {{ printf '%s\n' "$*" >&2; }}
die() {{ printf '%s\n' "$*" >&2; exit 1; }}
curl() {{
    local output=""
    while (($#)); do
        case "$1" in
            --output) output="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    printf 'downloaded\n' >&2
    cp -- "$GOOD_ASSET" "$output"
}}
{function}
fetch_verified "" "test://runtime" "$1" "$2"
"""
    result = subprocess.run(
        ['bash', '--noprofile', '--norc', '-c', command, 'bash',
         str(cached), expected],
        check=True,
        capture_output=True,
        text=True,
        env={'GOOD_ASSET': str(good), 'PATH': '/usr/bin:/bin'},
    )
    assert result.stdout.strip() == str(cached)
    assert cached.read_bytes() == good.read_bytes()
    assert not cached.with_suffix('.bin.part').exists()
    assert 'Discarding corrupt cached runtime asset' in result.stderr
    assert 'downloaded' in result.stderr

    reused = subprocess.run(
        ['bash', '--noprofile', '--norc', '-c', command, 'bash',
         str(cached), expected],
        check=True,
        capture_output=True,
        text=True,
        env={'GOOD_ASSET': str(good), 'PATH': '/usr/bin:/bin'},
    )
    assert reused.stdout.strip() == str(cached)
    assert 'downloaded' not in reused.stderr

    explicit = tmp_path / 'explicit.bin'
    explicit.write_bytes(b'do not delete')
    explicit_command = f"""
set -euo pipefail
warn() {{ :; }}
die() {{ printf '%s\n' "$*" >&2; exit 1; }}
{function}
fetch_verified "$1" "unused" "$2" "$3"
"""
    rejected = subprocess.run(
        [
            'bash',
            '--noprofile',
            '--norc',
            '-c',
            explicit_command,
            'bash',
            str(explicit),
            str(tmp_path / 'unused'),
            expected,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert explicit.read_bytes() == b'do not delete'


def test_system_prefix_install_normalizes_owner_and_permissions():
    """Keep privileged writes scoped to a hardened staging prefix."""
    source = SCRIPT.read_text()
    assert 'if [[ "${MIGRAPHX_ROOT}" == /opt/* ]]' in source
    assert 'RUNTIME_SUDO=("${SUDO[@]}")' in source
    assert 'chown -R root:root "${INSTALL_STAGE}"' in source
    assert 'chmod -R a-s,u=rwX,go=rX "${INSTALL_STAGE}"' in source
    assert '"${RUNTIME_SUDO[@]}" mv --' in source
    assert 'metadata.st_uid != 0 or metadata.st_gid != 0' in source
    assert 'mode & 0o022' in source
    assert 'verify_migraphx_install "${MIGRAPHX_ROOT}"' in source


def test_prefix_validation_checks_every_manifest_hash():
    """Installed prefixes must be checked beyond their release identity."""
    source = SCRIPT.read_text()
    assert 'for entry in manifest.get("files", [])' in source
    assert 'hashlib.sha256()' in source
    assert 'path.stat().st_size != entry.get("size")' in source
    assert 'os.readlink(path) != entry.get("target")' in source
    assert 'manifest.get("key_artifacts", {}).items()' in source


def test_recreate_refuses_to_remove_the_active_environment():
    """Environment recreation must be explicit and never target an active env."""
    source = SCRIPT.read_text()
    assert 'CONDA_DEFAULT_ENV:-' in source
    assert 'Cannot recreate the active Conda env' in source
    assert "Run 'conda deactivate'" in source


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
