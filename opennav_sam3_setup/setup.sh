#!/usr/bin/env bash
# setup.sh — One-command host setup for SAM3 on AMD ROCm / gfx1151
#
# This keeps ROS 2 Jazzy and colcon on the host. It downloads only published
# runtime binaries; it does not compile ROCm, MIGraphX, ONNX Runtime or PyTorch.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./setup.sh [--skip-apt] [--skip-migraphx] [--skip-weights]
             [--recreate-env] [--yes] [--env NAME]

Options:
  --skip-apt        Reuse an existing ROCm 7.14 multi-arch installation.
  --skip-migraphx   Reuse the verified MIGraphX prefix.
  --skip-weights    Install the runtime and model configs without weights.
  --recreate-env    Remove and recreate the dedicated Conda environment.
  --yes             Download the pinned, verified model mirror without prompting.
  --env NAME        Conda environment name (default: opennav-sam3-inference).
  -h, --help        Show this help without changing the system.

Overrides:
  SAM3_CONDA_ENV          Conda environment name.
  SAM3_MODEL_DIR          Parent directory for weights and generated artifacts.
  SAM3_BINARY_CACHE       Download cache directory.
  MIGRAPHX_ARCHIVE        Local copy of the pinned MIGraphX archive.
  ORT_WHEEL_PATH          Local copy of the pinned ONNX Runtime wheel.
  MIGRAPHX_URL            Alternate URL for the same pinned archive.
  ORT_URL                 Alternate URL for the same pinned wheel.
  SAM3_MIGRAPHX_ROOT      Runtime prefix override.
  SAM3_ROCM_PATH          ROCm root override.
EOF
}

CONDA_ENV="${SAM3_CONDA_ENV:-opennav-sam3-inference}"
SKIP_APT=false
SKIP_MIGRAPHX=false
SKIP_WEIGHTS=false
RECREATE_ENV=false
AUTO_YES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-apt) SKIP_APT=true ;;
        --skip-migraphx) SKIP_MIGRAPHX=true ;;
        --skip-weights) SKIP_WEIGHTS=true ;;
        --recreate-env) RECREATE_ENV=true ;;
        --yes) AUTO_YES=true ;;
        --env)
            [[ $# -ge 2 ]] || { echo "--env requires a name" >&2; exit 2; }
            CONDA_ENV="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

G='\033[0;32m'
Y='\033[1;33m'
R='\033[0;31m'
B='\033[1;34m'
NC='\033[0m'
step() { echo -e "\n${B}══ $* ${NC}"; }
info() { echo -e "  ${G}✓${NC} $*"; }
warn() { echo -e "  ${Y}⚠${NC}  $*"; }
die() { echo -e "  ${R}✗${NC}  $*" >&2; exit 1; }

if "${AUTO_YES}" && "${SKIP_WEIGHTS}"; then
    die "--yes and --skip-weights are mutually exclusive"
fi

if ((EUID == 0)); then
    die "Do not run setup.sh as root. Run it as your normal user; the script uses sudo only for system paths."
fi

for command in cmp install python3 readlink sha256sum tar; do
    command -v "${command}" >/dev/null || die "Missing command: ${command}"
done

SUDO=()
if command -v sudo >/dev/null; then
    SUDO=(sudo)
elif [[ "${SKIP_APT}" != true ]]; then
    die "sudo is required to install the ROCm system runtime"
fi

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

RUNTIME_RELEASE="0.2.0-rc4"
# 7.14.1-0 is the validated Debian package release. The compatibility
# rocm_sdk module below reports the ROCm API series as 7.14.0, matching the
# value expected by the gfx1151 PyTorch wheel.
RUNTIME_ID="opennav-sam3-${RUNTIME_RELEASE}-rocm7.14.1-mgx2.17-ort1.24.2-scipy1.17.1"
ROCM_VERSION="7.14"
ROCM_PACKAGE_VERSION="7.14.1-0"
ROCM_ROOT="${SAM3_ROCM_PATH:-/opt/rocm}"
ROCM_CORE="${ROCM_ROOT}/core-${ROCM_VERSION}"
MIGRAPHX_ROOT="${SAM3_MIGRAPHX_ROOT:-/opt/opennav-sam3/runtime/${RUNTIME_RELEASE}/migraphx}"
CACHE="${SAM3_BINARY_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/opennav-sam3/runtime/${RUNTIME_RELEASE}}"

TORCH_VERSION="2.11.0+rocm7.13.0"
TORCHVISION_VERSION="0.26.0+rocm7.13.0"
TRITON_VERSION="3.6.0+rocm7.13.0"
TORCH_INDEX="https://repo.amd.com/rocm/whl/gfx1151/"
PYPI_INDEX="https://pypi.org/simple"

MGX_NAME="migraphx-2.17.0-dev-9f1a138-sam3-fc1sink-rocm7.14-gfx1151-cp312.tar.gz"
MGX_URL="${MIGRAPHX_URL:-https://github.com/harrysocool/AMDMIGraphX/releases/download/v2.17.0%2Bsam3-fc1sink.20260908.1/${MGX_NAME}}"
MGX_SHA256="ed1458c632eb2f0e2cab3c457aee93e39196cbb77d2180e47525e0009563dac1"

ORT_NAME="onnxruntime_migraphx-1.24.2-cp312-cp312-linux_x86_64.whl"
ORT_URL="${ORT_URL:-https://github.com/harrysocool/sam3-tracker-rocm/releases/download/v0.2.0-rc4/${ORT_NAME}}"
ORT_SHA256="ef10e3e808e8805c26cc27f47572a53e385463f29d578e1ea2fe13d00e6f5ee0"

OFFICIAL_MODEL_REPO="facebook/sam3"
OFFICIAL_MODEL_REVISION="3c879f39826c281e95690f02c7821c4de09afae7"
# This historical mirror object has the same Git LFS SHA256 and byte size as
# the official checkpoint. The mirror's current main branch is a different,
# incompatible checkpoint, so this revision must remain immutable here.
MIRROR_MODEL_REPO="1038lab/sam3"
MIRROR_MODEL_REVISION="fe5e2ae858f82b44a0b18e80aa09fe1d5ed75a0b"
MIRROR_MODEL_NAME="sam3.safetensors"
MODEL_NAME="model.safetensors"
MODEL_SHA256="6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a"

MODEL_DIR_ROOT="${SAM3_MODEL_DIR:-${REPO_DIR}}"
MODEL_DIR="${MODEL_DIR_ROOT}/sam3"
WEIGHT_FILE="${MODEL_DIR}/${MODEL_NAME}"
MARKER_NAME=".opennav-sam3-runtime"

WORK_DIR=""
INSTALL_STAGE=""
RUNTIME_SUDO=()
cleanup() {
    [[ -z "${WORK_DIR}" ]] || rm -rf -- "${WORK_DIR}"
    if [[ -n "${INSTALL_STAGE}" && -e "${INSTALL_STAGE}" ]]; then
        "${RUNTIME_SUDO[@]}" rm -rf -- "${INSTALL_STAGE}"
    fi
}
trap cleanup EXIT

check_gpu_device_access() {
    local device_root="${1:-/dev}"
    local render_nodes=()
    local node

    [[ -e "${device_root}/kfd" ]] || \
        die "Missing ${device_root}/kfd; install the AMD GPU kernel driver"
    [[ -r "${device_root}/kfd" && -w "${device_root}/kfd" ]] || \
        die "No read/write access to ${device_root}/kfd; add this user to the render and video groups, then log in again"

    mapfile -t render_nodes < <(
        compgen -G "${device_root}/dri/renderD*" || true
    )
    (("${#render_nodes[@]}" > 0)) || \
        die "No GPU render node found below ${device_root}/dri"
    for node in "${render_nodes[@]}"; do
        if [[ -r "${node}" && -w "${node}" ]]; then
            return 0
        fi
    done
    die "No readable/writable GPU render node found; add this user to the render and video groups, then log in again"
}

fetch_verified() {
    local override="$1"
    local url="$2"
    local destination="$3"
    local expected="$4"
    local actual

    if [[ -n "${override}" ]]; then
        destination="$(readlink -f -- "${override}")"
        [[ -f "${destination}" ]] || \
            die "Runtime asset not found: ${destination}"
        actual="$(sha256sum -- "${destination}")"
        actual="${actual%% *}"
        [[ "${actual}" == "${expected}" ]] || \
            die "SHA256 mismatch for explicit runtime asset ${destination}"
        printf '%s\n' "${destination}"
        return
    fi

    if [[ -f "${destination}" ]]; then
        actual="$(sha256sum -- "${destination}")"
        actual="${actual%% *}"
        if [[ "${actual}" != "${expected}" ]]; then
            warn "Discarding corrupt cached runtime asset ${destination}" >&2
            rm -f -- "${destination}"
        fi
    fi
    if [[ ! -f "${destination}" ]]; then
        rm -f -- "${destination}.part"
        curl --fail --location --retry 3 \
            --output "${destination}.part" -- "${url}"
        actual="$(sha256sum -- "${destination}.part")"
        actual="${actual%% *}"
        if [[ "${actual}" != "${expected}" ]]; then
            rm -f -- "${destination}.part"
            die "SHA256 mismatch for downloaded runtime asset ${destination}"
        fi
        mv -- "${destination}.part" "${destination}"
    fi
    printf '%s\n' "${destination}"
}

verify_migraphx_archive() {
    python3 - "$1" <<'PY'
import json
from pathlib import PurePosixPath
import sys
import tarfile

archive = sys.argv[1]
required = {
    "migraphx/BUILD-MANIFEST.json",
    "migraphx/lib/migraphx.cpython-312-x86_64-linux-gnu.so",
    "migraphx/lib/migraphx/lib/libmigraphx_gpu.so.2017000.0",
}
with tarfile.open(archive, "r:gz") as package:
    names = set()
    for member in package.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name}")
        if not path.parts or path.parts[0] != "migraphx":
            raise SystemExit(f"unexpected archive root: {member.name}")
        if member.issym() or member.islnk():
            target = PurePosixPath(member.linkname)
            if target.is_absolute() or ".." in target.parts:
                raise SystemExit(f"unsafe archive link: {member.name}")
        names.add(member.name.rstrip("/"))
    missing = required - names
    if missing:
        raise SystemExit(f"missing MIGraphX archive files: {sorted(missing)}")
    manifest_file = package.extractfile("migraphx/BUILD-MANIFEST.json")
    if manifest_file is None:
        raise SystemExit("missing MIGraphX build manifest")
    manifest = json.load(manifest_file)

if manifest.get("format") != "sam3-migraphx-binary-prefix":
    raise SystemExit("unexpected MIGraphX archive format")
build = manifest.get("build", {})
source = manifest.get("source", {})
if build.get("rocm") != "7.14" or build.get("python_abi") != "cp312":
    raise SystemExit("MIGraphX archive has an incompatible ROCm/Python ABI")
if build.get("gpu_arch") != "gfx1151":
    raise SystemExit("MIGraphX archive is not built for gfx1151")
if source.get("release_tag") != "v2.17.0+sam3-fc1sink.20260908.1":
    raise SystemExit("MIGraphX archive release identity mismatch")
if source.get("pretag_test_mode") is not False:
    raise SystemExit("refusing a pre-tag MIGraphX package")
PY
}

verify_migraphx_prefix() {
    local root="$1"
    [[ -f "${root}/BUILD-MANIFEST.json" ]] || return 1
    [[ -f "${root}/lib/migraphx.cpython-312-x86_64-linux-gnu.so" ]] || return 1
    [[ -f "${root}/lib/migraphx/lib/libmigraphx_gpu.so.2017000.0" ]] || return 1
    python3 - "${root}" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

root = Path(sys.argv[1]).resolve()
with (root / "BUILD-MANIFEST.json").open(encoding="utf-8") as stream:
    manifest = json.load(stream)
build = manifest.get("build", {})
source = manifest.get("source", {})
identity_valid = (
    manifest.get("format") == "sam3-migraphx-binary-prefix"
    and build.get("rocm") == "7.14"
    and build.get("python_abi") == "cp312"
    and build.get("gpu_arch") == "gfx1151"
    and source.get("release_tag") == "v2.17.0+sam3-fc1sink.20260908.1"
    and source.get("pretag_test_mode") is False
)
if not identity_valid:
    raise SystemExit("installed MIGraphX identity mismatch")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


manifest_hashes = {}
for entry in manifest.get("files", []):
    relative = PurePosixPath(entry.get("path", ""))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SystemExit(f"unsafe manifest path: {relative}")
    path = root.joinpath(*relative.parts)
    entry_type = entry.get("type")
    if entry_type == "file":
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"missing manifest file: {relative}")
        if path.stat().st_size != entry.get("size"):
            raise SystemExit(f"size mismatch: {relative}")
        actual = digest(path)
        if actual != entry.get("sha256"):
            raise SystemExit(f"SHA256 mismatch: {relative}")
        manifest_hashes[str(relative)] = actual
    elif entry_type == "symlink":
        if not path.is_symlink() or os.readlink(path) != entry.get("target"):
            raise SystemExit(f"symlink mismatch: {relative}")
    else:
        raise SystemExit(f"unsupported manifest entry type: {entry_type}")

for relative, expected in manifest.get("key_artifacts", {}).items():
    if manifest_hashes.get(relative) != expected:
        raise SystemExit(f"key artifact mismatch: {relative}")
PY
}

install_model_configs() {
    local filename
    local source
    local destination

    install -d -m 0755 -- "${MODEL_DIR}"
    for filename in \
        config.json \
        configuration.json \
        LICENSE \
        merges.txt \
        processor_config.json \
        special_tokens_map.json \
        tokenizer_config.json \
        tokenizer.json \
        vocab.json; do
        source="${REPO_DIR}/model-configs/${filename}"
        destination="${MODEL_DIR}/${filename}"
        [[ -f "${source}" ]] || die "Missing bundled model config: ${source}"
        install -m 0644 -- "${source}" "${destination}"
        cmp --silent -- "${source}" "${destination}" || \
            die "Installed model config failed validation: ${destination}"
    done
}

verify_model_weight() {
    local path="${1:-${WEIGHT_FILE}}"
    local actual

    actual="$(sha256sum -- "${path}")"
    actual="${actual%% *}"
    [[ "${actual}" == "${MODEL_SHA256}" ]] || \
        die "SHA256 mismatch for ${path}"
}

verify_migraphx_permissions() {
    local root="$1"
    [[ "${root}" == /opt/* ]] || return 0
    python3 - "${root}" <<'PY'
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
for path in (root, *root.rglob("*")):
    metadata = path.lstat()
    if metadata.st_uid != 0 or metadata.st_gid != 0:
        raise SystemExit(f"non-root-owned runtime path: {path}")
    if path.is_symlink():
        continue
    mode = stat.S_IMODE(metadata.st_mode)
    if mode & 0o022:
        raise SystemExit(f"group/world-writable runtime path: {path}")
    if not mode & stat.S_IROTH:
        raise SystemExit(f"runtime path is not world-readable: {path}")
    if path.is_dir() and not mode & stat.S_IXOTH:
        raise SystemExit(f"runtime directory is not world-searchable: {path}")
PY
}

verify_migraphx_install() {
    verify_migraphx_prefix "$1" && verify_migraphx_permissions "$1"
}

echo -e "${G}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║  SAM3 Video Tracker — ROCm 7.14 Host Setup       ║"
echo "  ║  Target: gfx1151 (Ryzen AI Max+ 395)             ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "  Conda env      : ${CONDA_ENV}"
echo "  ROCm root      : ${ROCM_ROOT}"
echo "  MIGraphX prefix: ${MIGRAPHX_ROOT}"
echo "  Model dir      : ${MODEL_DIR}"
echo ""

step "0. Prerequisites and environment safety"

. /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]] || \
    die "The published host runtime requires Ubuntu 24.04"

check_gpu_device_access

if [[ "${ROCM_ROOT}" == "/opt/rocm" && -L "${ROCM_ROOT}" ]]; then
    die "/opt/rocm is a legacy symlink. Remove the old ROCm alternatives install before installing the ROCm 7.14 multi-arch runtime."
fi
if [[ "${ROCM_ROOT}" == "/opt/rocm" ]]; then
    for directory in bin lib libexec include share llvm amdgcn; do
        if [[ -e "${ROCM_ROOT}/${directory}" && \
              ! -L "${ROCM_ROOT}/${directory}" ]]; then
            die "${ROCM_ROOT}/${directory} uses a legacy ROCm layout; remove it before installing the 7.14 multi-arch runtime."
        fi
    done
fi

if ! command -v conda >/dev/null; then
    for candidate in \
        "${HOME}/miniforge3/bin/conda" \
        "${HOME}/miniconda3/bin/conda" \
        /opt/conda/bin/conda; do
        if [[ -f "${candidate}" ]]; then
            eval "$("${candidate}" shell.bash hook 2>/dev/null)"
            break
        fi
    done
fi
command -v conda >/dev/null || \
    die "conda not found. Install Miniforge before running setup.sh"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_PREFIX="$(conda env list | awk -v name="${CONDA_ENV}" '$1 == name {print $NF; exit}')"
if [[ -n "${ENV_PREFIX}" && "${RECREATE_ENV}" == true ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" || \
          "${CONDA_PREFIX:-}" == "${ENV_PREFIX}" ]]; then
        die "Cannot recreate the active Conda env '${CONDA_ENV}'. Run 'conda deactivate' and retry with --recreate-env."
    fi
    info "Removing dedicated Conda environment ${CONDA_ENV}"
    conda env remove -n "${CONDA_ENV}" -y
    ENV_PREFIX=""
elif [[ -n "${ENV_PREFIX}" ]]; then
    marker="${ENV_PREFIX}/${MARKER_NAME}"
    if [[ ! -f "${marker}" || "$(< "${marker}")" != "${RUNTIME_ID}" ]]; then
        die "Existing Conda env '${CONDA_ENV}' is not the pinned rc4 runtime. Re-run with --recreate-env or choose --env NAME."
    fi
fi

if command -v rocminfo >/dev/null; then
    GPU="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9]+' | head -1 || true)"
    [[ "${GPU}" == "gfx1151" ]] || \
        warn "Detected ${GPU:-unknown}; the published binaries require gfx1151"
else
    warn "rocminfo not found; GPU verification will run after installation"
fi

step "1. ROCm 7.14 system runtime"

ROCM_PACKAGES=(
    amdrocm-runtime-dev7.14
    amdrocm-blas-dev7.14
    amdrocm-blas-host7.14
    amdrocm-dnn-dev7.14
    amdrocm-dnn-host7.14
    amdrocm-hipblas-common-dev7.14
    amdrocm-blas7.14-gfx1151
    amdrocm-dnn7.14-gfx1151
    amdrocm-fft-host7.14
    amdrocm-fft7.14-gfx1151
    amdrocm-rand-host7.14
    amdrocm-rand7.14-gfx1151
    amdrocm-rccl-host7.14
    amdrocm-rccl7.14-gfx1151
)

if "${SKIP_APT}"; then
    info "Skipping APT install (--skip-apt)"
else
    [[ "${ROCM_ROOT}" == "/opt/rocm" ]] || \
        die "SAM3_ROCM_PATH requires --skip-apt; packages install under /opt/rocm"
    (("${#SUDO[@]}" > 0)) || \
        die "sudo is required to install the ROCm system runtime"
    "${SUDO[@]}" apt-get update -qq
    "${SUDO[@]}" apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg libgl1 libglib2.0-0 libnuma1 libssl3
    curl -fsSL https://repo.amd.com/rocm/packages/gpg/rocm.gpg |
        gpg --dearmor | "${SUDO[@]}" tee /etc/apt/keyrings/amdrocm.gpg >/dev/null
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main" |
        "${SUDO[@]}" tee /etc/apt/sources.list.d/amd-rocm-multiarch.list >/dev/null
    "${SUDO[@]}" apt-get update -qq
    pinned_packages=()
    for package in "${ROCM_PACKAGES[@]}"; do
        pinned_packages+=("${package}=${ROCM_PACKAGE_VERSION}")
    done
    "${SUDO[@]}" apt-get install -y --no-install-recommends "${pinned_packages[@]}"
    for directory in bin lib lib64 libexec include share llvm amdgcn; do
        "${SUDO[@]}" ln -sfnT "core-${ROCM_VERSION}/${directory}" \
            "${ROCM_ROOT}/${directory}"
    done
fi

command -v curl >/dev/null || die "Missing command: curl"

[[ -d "${ROCM_CORE}/lib" ]] || \
    die "ROCm 7.14 runtime not found below ${ROCM_CORE}"
[[ -e "${ROCM_ROOT}/lib/libamdhip64.so" ]] || \
    die "ROCm 7.14 HIP runtime is incomplete under ${ROCM_ROOT}"

step "2. Published MIGraphX 2.17 binary prefix"

mkdir -p "${CACHE}"
if "${SKIP_MIGRAPHX}"; then
    verify_migraphx_install "${MIGRAPHX_ROOT}" || \
        die "--skip-migraphx requested but ${MIGRAPHX_ROOT} is not the pinned runtime"
    info "Using verified MIGraphX prefix ${MIGRAPHX_ROOT}"
elif verify_migraphx_install "${MIGRAPHX_ROOT}"; then
    info "MIGraphX prefix already installed and verified"
else
    [[ ! -e "${MIGRAPHX_ROOT}" ]] || \
        die "Refusing to overwrite incompatible MIGraphX prefix ${MIGRAPHX_ROOT}"
    RUNTIME_SUDO=()
    if [[ "${MIGRAPHX_ROOT}" == /opt/* ]]; then
        (("${#SUDO[@]}" > 0)) || \
            die "sudo is required to install MIGraphX below /opt"
        RUNTIME_SUDO=("${SUDO[@]}")
    fi
    mgx_archive="$(fetch_verified \
        "${MIGRAPHX_ARCHIVE:-}" "${MGX_URL}" \
        "${CACHE}/${MGX_NAME}" "${MGX_SHA256}")"
    verify_migraphx_archive "${mgx_archive}"
    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/opennav-sam3-mgx.XXXXXXXX")"
    tar -xzf "${mgx_archive}" -C "${WORK_DIR}"
    runtime_parent="$(dirname -- "${MIGRAPHX_ROOT}")"
    INSTALL_STAGE="${runtime_parent}/.migraphx.tmp.$$"
    "${RUNTIME_SUDO[@]}" install -d -m 0755 "${runtime_parent}"
    "${RUNTIME_SUDO[@]}" cp -a -- \
        "${WORK_DIR}/migraphx" "${INSTALL_STAGE}"
    if (("${#RUNTIME_SUDO[@]}" > 0)); then
        "${RUNTIME_SUDO[@]}" chown -R root:root "${INSTALL_STAGE}"
    fi
    "${RUNTIME_SUDO[@]}" chmod -R a-s,u=rwX,go=rX "${INSTALL_STAGE}"
    "${RUNTIME_SUDO[@]}" mv -- "${INSTALL_STAGE}" "${MIGRAPHX_ROOT}"
    INSTALL_STAGE=""
    verify_migraphx_install "${MIGRAPHX_ROOT}" || \
        die "Installed MIGraphX prefix failed validation"
    info "Installed verified MIGraphX prefix"
fi

ort_wheel="$(fetch_verified \
    "${ORT_WHEEL_PATH:-}" "${ORT_URL}" \
    "${CACHE}/${ORT_NAME}" "${ORT_SHA256}")"

step "3. Conda Python 3.12 environment"

if [[ -z "${ENV_PREFIX}" ]]; then
    conda create -n "${CONDA_ENV}" python=3.12 pip -y -q
fi
conda activate "${CONDA_ENV}"
ENV_PREFIX="${CONDA_PREFIX}"

[[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.12" ]] || \
    die "The SAM3 Conda environment must use Python 3.12"
[[ "$(python -c 'import sys; print(sys.prefix)')" == "${ENV_PREFIX}" ]] || \
    die "Conda activation did not select the requested environment"

if python -m pip list --format=freeze |
        grep -Eiq '^(rocm|rocm[-_].*)=='; then
    die "Python-packaged ROCm libraries detected in ${CONDA_ENV}; recreate it with --recreate-env"
fi

activate_dir="${ENV_PREFIX}/etc/conda/activate.d"
deactivate_dir="${ENV_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${activate_dir}" "${deactivate_dir}"
if [[ "${_OPENNAV_SAM3_RUNTIME_ACTIVE:-0}" == "1" ]]; then
    existing_deactivate="${deactivate_dir}/opennav_sam3_runtime.sh"
    [[ -r "${existing_deactivate}" ]] || \
        die "Active SAM3 runtime has no readable deactivation hook"
    source "${existing_deactivate}"
fi
install -m 0644 \
    "${REPO_DIR}/conda/activate.d/opennav_sam3_runtime.sh" \
    "${activate_dir}/opennav_sam3_runtime.sh"
install -m 0644 \
    "${REPO_DIR}/conda/deactivate.d/opennav_sam3_runtime.sh" \
    "${deactivate_dir}/opennav_sam3_runtime.sh"
runtime_config="${ENV_PREFIX}/etc/opennav_sam3_runtime.conf"
runtime_config_tmp="${runtime_config}.tmp.$$"
{
    printf 'SAM3_MIGRAPHX_ROOT=%q\n' "${MIGRAPHX_ROOT}"
    printf 'SAM3_ROCM_PATH=%q\n' "${ROCM_ROOT}"
} > "${runtime_config_tmp}"
mv -- "${runtime_config_tmp}" "${runtime_config}"
source "${activate_dir}/opennav_sam3_runtime.sh"

step "4. Pinned Python runtime"

python -m pip install -q --only-binary=:all: \
    --index-url "${PYPI_INDEX}" \
    --constraint "${REPO_DIR}/runtime-constraints.txt" \
    setuptools wheel

python -m pip install -q --only-binary=:all: \
    --index-url "${PYPI_INDEX}" \
    --constraint "${REPO_DIR}/runtime-constraints.txt" \
    -r "${REPO_DIR}/requirements.txt"

python -m pip install -q --only-binary=:all: --no-deps \
    --force-reinstall --index-url "${TORCH_INDEX}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "triton==${TRITON_VERSION}"

python -m pip install -q --only-binary=:all: --force-reinstall \
    --index-url "${PYPI_INDEX}" \
    --constraint "${REPO_DIR}/runtime-constraints.txt" \
    "${ort_wheel}"

# colcon's EmPy dependency is distributed as an sdist. Keep build tooling out
# of the binary-only runtime resolver while retaining OpenNav's Conda/colcon
# workflow.
python -m pip install -q \
    --index-url "${PYPI_INDEX}" \
    --constraint "${REPO_DIR}/runtime-constraints.txt" \
    "colcon-common-extensions==0.3.0"

site_packages="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
mkdir -p "${site_packages}/rocm_sdk"
install -m 0644 "${REPO_DIR}/rocm_sdk_system.py" \
    "${site_packages}/rocm_sdk/__init__.py"

python - "${MIGRAPHX_ROOT}" "${TORCH_VERSION}" <<'PY'
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

prefix = Path(sys.argv[1]).resolve()
expected_torch = sys.argv[2]

import torch
import migraphx
import onnxruntime as ort
import rocm_sdk

assert sys.version_info[:2] == (3, 12)
assert version("packaging") == "23.0"
assert version("setuptools") == "79.0.1"
assert version("wheel") == "0.42.0"
assert torch.__version__ == expected_torch
assert torch.cuda.is_available()
assert str(migraphx.__version__).startswith("2.17")
assert Path(migraphx.__file__).resolve().is_relative_to(prefix)
assert ort.__version__ == "1.24.2"
assert "MIGraphXExecutionProvider" in ort.get_available_providers()
assert rocm_sdk.__version__ == "7.14.0"
pip_check = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    check=False,
    capture_output=True,
    text=True,
)
unexpected = [
    line for line in pip_check.stdout.splitlines()
    if line and not (line.startswith("torch ") and " requires rocm" in line)
]
assert not unexpected, unexpected
print("Runtime smoke passed:")
print(f"  torch:       {torch.__version__} (HIP {torch.version.hip})")
print(f"  migraphx:    {migraphx.__version__} ({migraphx.__file__})")
print(f"  onnxruntime: {ort.__version__} ({ort.get_available_providers()})")
PY

printf '%s\n' "${RUNTIME_ID}" > "${ENV_PREFIX}/${MARKER_NAME}"
info "Pinned SAM3 runtime installed"

step "5. Model weights"

install_model_configs
info "Installed and verified bundled model configs"

if [[ -f "${WEIGHT_FILE}" ]]; then
    verify_model_weight
    SIZE="$(du -sh "${WEIGHT_FILE}" | cut -f1)"
    info "Verified official weights (${WEIGHT_FILE}, ${SIZE})"
elif "${SKIP_WEIGHTS}"; then
    warn "Skipping model weights (--skip-weights); add ${WEIGHT_FILE} before building artifacts"
else
    echo "  Could not find model weights file: ${WEIGHT_FILE}"
    echo "  SAM3 model weights (~3.3 GB) are required."
    echo "  Config/tokenizer files have already been installed."
    echo ""
    if command -v hf >/dev/null 2>&1; then
        HF=hf
    else
        HF=huggingface-cli
    fi
    echo "  Option A — Official (requires accepted terms and hf auth login):"
    echo "    https://huggingface.co/facebook/sam3"
    echo "    ${HF} download ${OFFICIAL_MODEL_REPO} ${MODEL_NAME} --revision ${OFFICIAL_MODEL_REVISION} --local-dir ${MODEL_DIR}"
    echo ""
    echo "  Option B — Pinned byte-identical community mirror:"
    echo "    ${HF} download ${MIRROR_MODEL_REPO} ${MIRROR_MODEL_NAME} --revision ${MIRROR_MODEL_REVISION} --local-dir ${MODEL_DIR}"
    echo ""
    if "${AUTO_YES}"; then
        answer=Y
        echo "  Download via pinned Option B now? [y/N] y  (auto-yes)"
    elif [[ ! -t 0 ]]; then
        answer=N
        warn "Non-interactive shell detected; skipping model weights"
    else
        read -rp "  Download via pinned Option B now? [y/N] " answer || \
            answer=N
        answer="${answer:-N}"
    fi
    if [[ "${answer}" =~ ^[Yy] ]]; then
        mirror_weight="${MODEL_DIR}/${MIRROR_MODEL_NAME}"
        if [[ "${HF}" == "huggingface-cli" ]]; then
            "${HF}" download "${MIRROR_MODEL_REPO}" "${MIRROR_MODEL_NAME}" \
                --revision "${MIRROR_MODEL_REVISION}" \
                --local-dir "${MODEL_DIR}" --local-dir-use-symlinks False
        else
            "${HF}" download "${MIRROR_MODEL_REPO}" "${MIRROR_MODEL_NAME}" \
                --revision "${MIRROR_MODEL_REVISION}" --local-dir "${MODEL_DIR}"
        fi
        verify_model_weight "${mirror_weight}"
        mv -- "${mirror_weight}" "${WEIGHT_FILE}"
        verify_model_weight
        info "Verified official weights -> ${WEIGHT_FILE}"
    else
        warn "Skipping weights; place model.safetensors in ${MODEL_DIR}"
    fi
fi

echo ""
echo -e "${G}══════════════════════════════════════════════════════${NC}"
echo -e "${G}  Environment setup complete.${NC}"
echo -e "${G}══════════════════════════════════════════════════════${NC}"
echo "Reactivate the environment before building SAM3 artifacts:"
echo "  conda deactivate"
echo "  conda activate ${CONDA_ENV}"
echo "Build from a new empty onnx_files_<resolution> directory; old MIGraphX"
echo "2.15/2.16 MXR files and ORT caches are not compatible with this runtime."
