#!/usr/bin/env bash
# setup.sh — One-command setup for SAM3 on AMD ROCm / gfx1151
#
# SAM3 inference and model export run in the ROS 2 Jazzy / ROCm 10.1 /
# MIGraphX 2.18 Docker image defined in docker/, assembled from official AMD
# packages. This script builds that image and installs the
# bundled model configs and the SAM3 weights into SAM3_MODEL_DIR. ROS 2 Jazzy
# and colcon for the rest of the stack stay on the host (docker/run_node.sh
# runs the inference node inside the image).
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./setup.sh [--skip-image] [--skip-weights] [--yes]

Builds the ROCm 10.1 / MIGraphX 2.18 Docker image (docker/build.sh) and
prepares the SAM3 model directory.

Options:
  --skip-image      Do not build the Docker image.
  --skip-weights    Install the model configs without downloading weights.
  --yes             Download the pinned, verified model mirror without prompting.
  -h, --help        Show this help without changing the system.

Overrides:
  SAM3_MODEL_DIR          Parent directory for weights and generated artifacts.
  SAM3_BINARY_CACHE       Download cache for the pinned official wheels.
  SAM3_DOCKER_IMAGE       Docker image tag to build (also used by docker/run.sh).
USAGE
}

SKIP_IMAGE=false
SKIP_WEIGHTS=false
AUTO_YES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-image) SKIP_IMAGE=true ;;
        --skip-weights) SKIP_WEIGHTS=true ;;
        --yes) AUTO_YES=true ;;
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
    die "Do not run setup.sh as root. Run it as your normal user in the docker group."
fi

for command in cmp install readlink sha256sum; do
    command -v "${command}" >/dev/null || die "Missing command: ${command}"
done

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

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

echo "  Model dir      : ${MODEL_DIR}"
echo ""

step "0. Prerequisites"

check_gpu_device_access
info "GPU devices accessible"
command -v docker >/dev/null || die "docker not found. Install Docker with BuildKit first"
docker info >/dev/null 2>&1 || \
    die "Cannot talk to the Docker daemon; start it and add this user to the docker group"
info "Docker available"
if command -v rocminfo >/dev/null; then
    GPU="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9]+' | head -1 || true)"
    [[ "${GPU}" == "gfx1151" ]] || \
        warn "Detected ${GPU:-unknown}; the SAM3 image and model build require gfx1151"
fi

step "1. ROCm 10.1 / MIGraphX 2.18 Docker image"

if "${SKIP_IMAGE}"; then
    warn "Skipping image build (--skip-image)"
else
    "${REPO_DIR}/docker/build.sh"
fi

step "2. Model weights"

install_model_configs
info "Installed and verified bundled model configs"

if [[ -f "${WEIGHT_FILE}" ]]; then
    verify_model_weight
    SIZE="$(du -shL "${WEIGHT_FILE}" | cut -f1)"
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
    command -v "${HF}" >/dev/null 2>&1 || \
        warn "${HF} not found; install it with 'pip install huggingface-hub' to download"
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
echo -e "${G}  Setup complete.${NC}"
echo -e "${G}══════════════════════════════════════════════════════${NC}"
echo "Build model artifacts inside the container into a new empty"
echo "onnx_files_504 directory; MXR files and ORT caches from other MIGraphX"
echo "versions (including the earlier ROCm 7.14 image) are not compatible with"
echo "this runtime:"
echo "  ./docker/run.sh python export/build.py --pipeline text --imgsz 504"
