#!/usr/bin/env bash
# setup.sh — One-command setup for SAM3 video tracker on AMD ROCm / gfx1151
#
# Usage:
#   ./setup.sh                        # full setup
#   ./setup.sh --skip-apt             # skip ROCm 7.2 APT install (already done)
#   ./setup.sh --skip-migraphx        # skip patched MIGraphX install (already done)
#   ./setup.sh --skip-weights         # skip downloading model weights
#   ./setup.sh --env my-env           # custom conda environment name
#   ./setup.sh --yes                  # run in non-interactive mode assuming yes for all options
#
# Environment variables (alternative to flags):
#   SAM3_CONDA_ENV=opennav-sam3-inference       conda environment name
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CONDA_ENV="${SAM3_CONDA_ENV:-opennav-sam3-inference}"

SKIP_APT=false
SKIP_MIGRAPHX=false
SKIP_MODEL_WEIGHTS=false
AUTO_YES=false

# Pinned package versions — update together if you change the ROCm stack
ROCM_SDK_VER="7.13.0a20260411"
TORCH_VER="2.12.0a0+rocm7.13.0a20260411"
TORCHVISION_VER="0.27.0a0+rocm7.13.0a20260411"
NIGHTLY_INDEX="https://rocm.nightlies.amd.com/v2/gfx1151/"
ORT_WHL="https://github.com/Looong01/onnxruntime-rocm-build/releases/download/v1.24.2/onnxruntime_migraphx-1.24.2-cp312-cp312-manylinux_2_34_x86_64.whl"

MXR_TAG="v2.15+patches.20260512"
MXR_ASSET="migraphx-2.15+patches-linux-x86_64-rocm7.2-py312.tar.gz"
MXR_URL="https://github.com/harrysocool/AMDMIGraphX/releases/download/${MXR_TAG}/${MXR_ASSET}"
MXR_SHA256="18b8fc856b145972f30ccfb5f22a03ccdb718e04661538454278a28de23859dc"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-apt)       SKIP_APT=true ;;
        --skip-migraphx)  SKIP_MIGRAPHX=true ;;
        --skip-weights)   SKIP_MODEL_WEIGHTS=true ;;
        --yes)            AUTO_YES=true ;;
        --env)            CONDA_ENV="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ── Colours ───────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[1;34m'; NC='\033[0m'
step()  { echo -e "\n${B}══ $* ${NC}"; }
info()  { echo -e "  ${G}✓${NC} $*"; }
warn()  { echo -e "  ${Y}⚠${NC}  $*"; }
die()   { echo -e "  ${R}✗${NC}  $*"; exit 1; }


# Auto-detect ROCm install path (supports any 7.2.x patch release).
# Override with:  ROCM_PATH=/opt/rocm-7.2.3 ./setup.sh
_detect_rocm_path() {
    [[ -n "${ROCM_PATH:-}" && -d "$ROCM_PATH/lib" ]] && echo "$ROCM_PATH" && return
    local _p
    _p="$(dpkg -L migraphx 2>/dev/null | grep 'lib/libmigraphx\.so$' | head -1 | sed 's|/lib/libmigraphx\.so||')"
    [[ -n "$_p" && -d "$_p/lib" ]] && echo "$_p" && return
    for _p in $(ls -d /opt/rocm-7.2.* 2>/dev/null | sort -rV); do
        [[ -d "$_p/lib" ]] && echo "$_p" && return
    done
    echo "/opt/rocm-7.2.0"
}
ROCM_PATH="$(_detect_rocm_path)"

echo -e "${G}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║  SAM3 Video Tracker — ROCm Setup                 ║"
echo "  ║  Target: gfx1151 (Ryzen AI Max+ 395)             ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "  Conda env : $CONDA_ENV"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
step "0. Prerequisites"
# ─────────────────────────────────────────────────────────────────────────────
# Locate conda
if ! command -v conda &>/dev/null; then
    for p in ~/miniforge3/bin/conda ~/miniconda3/bin/conda /opt/conda/bin/conda; do
        [[ -f "$p" ]] && eval "$($p shell.bash hook 2>/dev/null)" && break
    done
fi
command -v conda &>/dev/null || die "conda not found. Install miniforge: https://github.com/conda-forge/miniforge"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
info "conda: $(conda --version)"

# ─────────────────────────────────────────────────────────────────────────────
step "0a. ROCm 7.2 APT (stock MIGraphX)"
# ─────────────────────────────────────────────────────────────────────────────
# Always install runtime libs needed by OpenCV/cv2 (missing from minimal envs)
if ! $SKIP_APT; then
    sudo apt-get update -qq 2>/dev/null || true
    sudo apt-get install -y libgl1 libglib2.0-0 2>/dev/null || true
fi

if $SKIP_APT; then
    info "Skipping APT install (--skip-apt)"
elif dpkg -s migraphx &>/dev/null; then
    info "migraphx already installed: $(dpkg -s migraphx | grep Version | awk '{print $2}')"
else
    echo "  Installing ROCm 7.2 APT packages (requires sudo)..."
    sudo apt-get update -qq
    sudo apt-get install -y wget gnupg
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
        https://repo.radeon.com/rocm/apt/7.2 noble main" | \
        sudo tee /etc/apt/sources.list.d/rocm.list
    sudo apt-get update -qq
    sudo apt-get install -y migraphx migraphx-dev
    info "migraphx installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "0b. Patched MIGraphX 2.15+patches"
# ─────────────────────────────────────────────────────────────────────────────
if $SKIP_MIGRAPHX; then
    info "Skipping patched MIGraphX install (--skip-migraphx)"
elif [[ -f "$ROCM_PATH/lib/libmigraphx_c.so.3.0.2016000" \
     && -f "$ROCM_PATH/lib/migraphx/lib/libmigraphx_ref.so.2016000.0" \
     && -f "$ROCM_PATH/lib/migraphx/lib/libmigraphx_cpu.so.2016000.0" \
     && -f "$ROCM_PATH/lib/migraphx/lib/libdnnl.so.1" \
     && -f "$ROCM_PATH/lib/migraphx/lib/libomp.so" \
     && -f /etc/ld.so.conf.d/rocm-migraphx-2016.conf ]]; then
    # Marker + libs that older releases (<=20260511) shipped without + the
    # ldconfig conf the older install script forgot to write. Reinstall if any
    # of these is missing so users with broken older installs auto-recover.
    info "Patched MIGraphX already installed (ROCm: $ROCM_PATH)"
else
    echo "  Downloading patched MIGraphX (~85 MB)..."
    TMP_MXR=$(mktemp -d)
    wget -q --show-progress -O "$TMP_MXR/$MXR_ASSET" "$MXR_URL"
    echo "  Verifying checksum..."
    echo "${MXR_SHA256}  $TMP_MXR/$MXR_ASSET" | sha256sum --check --quiet
    echo "  Installing (requires sudo)..."
    tar -xzf "$TMP_MXR/$MXR_ASSET" -C "$TMP_MXR"
    
    if [ ! -d "$ROCM_PATH/lib/migraphx/lib" ]; then
        sudo mkdir -p "$ROCM_PATH/lib/migraphx/lib"
    fi
    
    (cd "$TMP_MXR/migraphx-2.15+patches" && sudo BUILD=. ROCM="$ROCM_PATH" bash install_migraphx_patched.sh)
    rm -rf "$TMP_MXR"
    
    info "Patched MIGraphX installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "1. Conda environment: $CONDA_ENV"
# ─────────────────────────────────────────────────────────────────────────────
if conda env list | grep -q "^${CONDA_ENV} "; then
    info "Conda env '$CONDA_ENV' already exists — activating"
else
    echo "  Creating conda env '$CONDA_ENV' (Python 3.12 + pip)..."
    # pip is required explicitly: conda-forge's python=3.12 metapackage no longer
    # ships pip by default. Without this `pip` falls through to the system Python's
    # pip (which Ubuntu 24.04 protects with PEP 668 externally-managed-environment).
    conda create -n "$CONDA_ENV" python=3.12 pip -y -q
    info "Conda env created"
fi
conda activate "$CONDA_ENV"
# Belt-and-braces: `conda activate` from inside a script doesn't always
# update PATH (depends on conda init / shell type). Explicitly prepend the
# env bin so subsequent `python` / `pip` invocations resolve to the right
# binaries instead of the system Python (which on Ubuntu 24.04 PEP 668
# refuses pip installs).
ENV_BIN="$CONDA_BASE/envs/$CONDA_ENV/bin"
export PATH="$ENV_BIN:$PATH"

# Diagnostic: confirm the right python/pip resolved (1 line, low noise).
info "  python=$(command -v python)  pip=$(command -v pip)"
if [[ "$(command -v pip)" != "$ENV_BIN/pip" ]]; then
    die "conda env activation did not put $ENV_BIN/pip on PATH first; got $(command -v pip). Aborting before installing into wrong python."
fi

# ─────────────────────────────────────────────────────────────────────────────
step "2. ROCm SDK + PyTorch (pinned $ROCM_SDK_VER)"
# ─────────────────────────────────────────────────────────────────────────────
if python -c "import torch; assert torch.__version__ == '${TORCH_VER}'" 2>/dev/null; then
    info "PyTorch ${TORCH_VER} already installed"
else
    echo "  Installing ROCm SDK + PyTorch (~2-5 min)..."
    pip install -q \
        rocm \
        "rocm-sdk-core==${ROCM_SDK_VER}" \
        rocm-sdk-libraries-gfx1151 \
        rocm-sdk-devel \
        --index-url "$NIGHTLY_INDEX"
    pip install -q \
        "torch==${TORCH_VER}" \
        "torchvision==${TORCHVISION_VER}" \
        triton \
        --index-url "$NIGHTLY_INDEX"
    info "PyTorch ${TORCH_VER} installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "3. onnxruntime-migraphx"
# ─────────────────────────────────────────────────────────────────────────────
if python -c "import onnxruntime; assert '1.24.2' in onnxruntime.__version__" 2>/dev/null; then
    info "onnxruntime-migraphx 1.24.2 already installed"
else
    echo "  Installing onnxruntime-migraphx 1.24.2..."
    pip install -q "$ORT_WHL"
    info "onnxruntime-migraphx installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
step "4. Python dependencies"
# ─────────────────────────────────────────────────────────────────────────────
pip install -q -r requirements.txt
info "requirements.txt installed"

# ─────────────────────────────────────────────────────────────────────────────
step "5. Download model weights"
# ─────────────────────────────────────────────────────────────────────────────
if [ ! $SKIP_MODEL_WEIGHTS ] && [ ! $AUTO_YES ] ; then
    $PWD/download-weights.sh
elif [ ! $SKIP_MODEL_WEIGHTS ] && [ $AUTO_YES ] ; then
    $PWD/download-weights.sh --yes
else
    echo ""
    echo "Skipped downloading SAM3 model weights"
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# Environment ready
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${G}══════════════════════════════════════════════════════${NC}"
echo -e "${G}  Environment setup complete!${NC}"
echo -e "${G}══════════════════════════════════════════════════════${NC}"
