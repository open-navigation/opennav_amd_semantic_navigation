#!/usr/bin/env bash
# Assemble the OpenNav SAM3 image from published binaries. This script never
# compiles rocMLIR, MIGraphX, ONNX Runtime, or PyTorch.
set -euo pipefail

SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_RELEASE="0.2.0-rc4"
CACHE="${SAM3_BINARY_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/opennav-sam3/runtime/${RUNTIME_RELEASE}}"
RUNTIME_IMAGE="${SAM3_DOCKER_IMAGE:-opennav-sam3-gpu714-ort1242-mgx217-gfx1151:${RUNTIME_RELEASE}}"

# Keep these pins identical to setup.sh.
MGX_NAME=migraphx-2.17.0-dev-9f1a138-sam3-fc1sink-rocm7.14-gfx1151-cp312.tar.gz
MGX_URL="${MIGRAPHX_URL:-https://github.com/harrysocool/AMDMIGraphX/releases/download/v2.17.0%2Bsam3-fc1sink.20260908.1/${MGX_NAME}}"
MGX_SHA256=ed1458c632eb2f0e2cab3c457aee93e39196cbb77d2180e47525e0009563dac1
ORT_NAME=onnxruntime_migraphx-1.24.2-cp312-cp312-linux_x86_64.whl
ORT_URL="${ORT_URL:-https://github.com/harrysocool/sam3-tracker-rocm/releases/download/v0.2.0-rc4/${ORT_NAME}}"
ORT_SHA256=ef10e3e808e8805c26cc27f47572a53e385463f29d578e1ea2fe13d00e6f5ee0

usage() {
    cat <<'USAGE'
Usage: docker/build.sh [--no-cache] [--no-smoke]

Downloads checksum-pinned MIGraphX and ORT binaries, then assembles the local
Docker image. It does not clone or compile runtime sources.

Overrides:
  MIGRAPHX_ARCHIVE=/local/file.tar.gz
  ORT_WHEEL_PATH=/local/file.whl
  MIGRAPHX_URL=https://...
  ORT_URL=https://...
  SAM3_DOCKER_IMAGE=name:tag   (same variable run.sh uses)
  SAM3_BINARY_CACHE=/path
USAGE
}

NO_SMOKE=false
NO_CACHE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cache) NO_CACHE=true ;;
        --no-smoke) NO_SMOKE=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "binary runtime build: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

for command in curl docker readlink sha256sum tar; do
    command -v "${command}" >/dev/null || { echo "missing command: ${command}" >&2; exit 2; }
done

mkdir -p "${CACHE}"
fetch() {
    local override="$1" url="$2" destination="$3" expected="$4" actual
    if [[ -n "${override}" ]]; then
        destination="$(readlink -f -- "${override}")"
    elif [[ ! -f "${destination}" ]]; then
        curl --fail --location --retry 3 --output "${destination}.part" -- "${url}"
        mv -- "${destination}.part" "${destination}"
    fi
    actual="$(sha256sum -- "${destination}")"; actual="${actual%% *}"
    [[ "${actual}" == "${expected}" ]] || {
        echo "SHA256 mismatch: ${destination}" >&2; return 2;
    }
    printf '%s\n' "${destination}"
}

mgx="$(fetch "${MIGRAPHX_ARCHIVE:-}" "${MGX_URL}" "${CACHE}/${MGX_NAME}" "${MGX_SHA256}")"
ort="$(fetch "${ORT_WHEEL_PATH:-}" "${ORT_URL}" "${CACHE}/${ORT_NAME}" "${ORT_SHA256}")"

work="$(mktemp -d "${TMPDIR:-/tmp}/opennav-sam3-binary-image.XXXXXXXX")"
trap 'rm -rf -- "${work}"' EXIT
mkdir "${work}/migraphx" "${work}/ort"
tar -xzf "${mgx}" -C "${work}/migraphx"
[[ -f "${work}/migraphx/migraphx/lib/migraphx/lib/libmigraphx_gpu.so.2017000.0" ]] || {
    echo "invalid MIGraphX binary archive" >&2; exit 2;
}
cp "${ort}" "${work}/ort/${ORT_NAME}"

docker_args=()
if "${NO_CACHE}"; then
    docker_args+=(--no-cache)
fi
DOCKER_BUILDKIT=1 docker build "${docker_args[@]}" \
    --build-context "migraphx=${work}/migraphx" \
    --build-context "ort=${work}/ort" \
    -f "${SETUP_DIR}/docker/Dockerfile" \
    -t "${RUNTIME_IMAGE}" "${SETUP_DIR}"

if ! "${NO_SMOKE}"; then
    gpu=(--device=/dev/kfd --device=/dev/dri --group-add "$(stat -c '%g' /dev/kfd)")
    [[ ! -e /dev/dri/renderD128 ]] || gpu+=(--group-add "$(stat -c '%g' /dev/dri/renderD128)")
    docker run --rm --pull=never "${gpu[@]}" "${RUNTIME_IMAGE}" python -c \
        'import torch, migraphx, onnxruntime as o; assert torch.cuda.is_available(); assert str(migraphx.__version__).startswith("2.17"); assert o.__version__ == "1.24.2"; assert "MIGraphXExecutionProvider" in o.get_available_providers()'
fi

echo "Binary runtime ready: ${RUNTIME_IMAGE}"
