#!/usr/bin/env bash
# Assemble the OpenNav SAM3 image from official AMD binaries. This script never
# compiles rocMLIR, MIGraphX, ONNX Runtime, or PyTorch.
set -euo pipefail

SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd -- "${SETUP_DIR}/.." && pwd)"
RUNTIME_RELEASE="0.4.0-rc1"
CACHE="${SAM3_BINARY_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/opennav-sam3/runtime/${RUNTIME_RELEASE}}"
RUNTIME_IMAGE="${SAM3_DOCKER_IMAGE:-opennav-sam3-gpu101-ort129-mgx218-gfx1151:${RUNTIME_RELEASE}}"

TORCH_BASE="https://stable.repo.amd.com/rocm/pytorch/whl-next"
MGX_BASE="https://stable.repo.amd.com/rocm/migraphx/whl-next"
EP_BASE="https://stable.repo.amd.com/rocm/onnxruntime/whl-next"
PYPI_BASE="https://files.pythonhosted.org/packages/96/eb/e6968f5e41aac3125f2ff5708855f09cb0b70d85ed3115b625b0b58305ba"

# "<sha256> <url> <file name>". Local versions (+rocm10.1.0) are percent-encoded
# in the URL but not in the file name.
WHEELS=(
"3182f6f166f42d6f186fd8c4a4ea65e1ed02fba8a98c48f977b68f15cca82636 ${TORCH_BASE}/torch/torch-2.14.0%2Brocm10.1.0-cp312-cp312-linux_x86_64.whl torch-2.14.0+rocm10.1.0-cp312-cp312-linux_x86_64.whl"
"2e340119752d93d639d3baf63b33f92190260738dff3476de374adaf92fb30ef ${TORCH_BASE}/amd-torch-device-gfx1151/amd_torch_device_gfx1151-2.14.0%2Brocm10.1.0-cp312-cp312-linux_x86_64.whl amd_torch_device_gfx1151-2.14.0+rocm10.1.0-cp312-cp312-linux_x86_64.whl"
"73daa37345172b0556f8a20687197022b8f4c48dd74db5707b03780cc5e71a4d ${TORCH_BASE}/amd-torch-device-gfx115x/amd_torch_device_gfx115x-2.14.0%2Brocm10.1.0-cp312-cp312-linux_x86_64.whl amd_torch_device_gfx115x-2.14.0+rocm10.1.0-cp312-cp312-linux_x86_64.whl"
"f0eed016b74d09192969a7db2f4018941f662c413d55b0eeb157b99941e96e5c ${TORCH_BASE}/torchvision/torchvision-0.29.0a0%2Brocm10.1.0-cp312-cp312-linux_x86_64.whl torchvision-0.29.0a0+rocm10.1.0-cp312-cp312-linux_x86_64.whl"
"99650eae5ecc34cff9b8d57afd035f174ce7a14a4d66bb2a4bea4e4fc2f86d0d ${TORCH_BASE}/amd-torchvision-device-gfx1151/amd_torchvision_device_gfx1151-0.29.0a0%2Brocm10.1.0-cp312-cp312-linux_x86_64.whl amd_torchvision_device_gfx1151-0.29.0a0+rocm10.1.0-cp312-cp312-linux_x86_64.whl"
"2f745c62aebb33167bfb6d586f569fbea67f8fc5181b03798b91a6b9187a625e ${TORCH_BASE}/triton/triton-3.8.0%2Bgit669b31ac.rocm10.1.0-cp312-cp312-linux_x86_64.whl triton-3.8.0+git669b31ac.rocm10.1.0-cp312-cp312-linux_x86_64.whl"
"6a22d67926759f3ed02ba8d38e31b02e0f69579c5c2893a30745aa0708358c25 ${MGX_BASE}/migraphx/migraphx-2.18.0%2Brocm10.1.0-cp312-none-manylinux_2_28_x86_64.whl migraphx-2.18.0+rocm10.1.0-cp312-none-manylinux_2_28_x86_64.whl"
"0a625d0fc835db4727af8b6199a0a85bcaa0578cc2a873f6fbbe36b70d84713b ${MGX_BASE}/migraphx-libs/migraphx_libs-2.18.0%2Brocm10.1.0-py3-none-manylinux_2_28_x86_64.whl migraphx_libs-2.18.0+rocm10.1.0-py3-none-manylinux_2_28_x86_64.whl"
"f43aafe7581fc7f82f94518424c86418371cdcc31c2cb399cec14e7d5c8cd19c ${EP_BASE}/onnxruntime-ep-migraphx/onnxruntime_ep_migraphx-1.0.0%2Brocm10.1.0-cp312-cp312-manylinux_2_28_x86_64.whl onnxruntime_ep_migraphx-1.0.0+rocm10.1.0-cp312-cp312-manylinux_2_28_x86_64.whl"
"2b80d8c7ec2cc7438e4da3760b88c24568cba72c9ace96d668800a6c79419acb ${PYPI_BASE}/onnxruntime-1.29.0-cp312-cp312-manylinux_2_28_x86_64.whl onnxruntime-1.29.0-cp312-cp312-manylinux_2_28_x86_64.whl"
)

usage() {
    cat <<'USAGE'
Usage: docker/build.sh [--no-cache] [--no-smoke]

Downloads checksum-pinned official AMD wheels (Torch 2.14, MIGraphX 2.18, ONNX
Runtime MIGraphX plugin EP) and ONNX Runtime 1.29.0 from PyPI, then assembles
the local Docker image on top of the official ROS 2 Jazzy image. It does not
clone or compile runtime sources.

Overrides:
  SAM3_DOCKER_IMAGE=name:tag   (same variable run.sh uses)
  SAM3_BINARY_CACHE=/path      (wheel download cache; files are re-verified)
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

for command in curl docker sha256sum; do
    command -v "${command}" >/dev/null || { echo "missing command: ${command}" >&2; exit 2; }
done

mkdir -p "${CACHE}"
work="$(mktemp -d "${TMPDIR:-/tmp}/opennav-sam3-binary-image.XXXXXXXX")"
trap 'rm -rf -- "${work}"' EXIT
mkdir "${work}/wheels"

for row in "${WHEELS[@]}"; do
    read -r expected url name <<<"${row}"
    destination="${CACHE}/${name}"
    if [[ ! -f "${destination}" ]]; then
        curl --fail --location --retry 3 --output "${destination}.part" -- "${url}"
        mv -- "${destination}.part" "${destination}"
    fi
    actual="$(sha256sum -- "${destination}")"; actual="${actual%% *}"
    [[ "${actual}" == "${expected}" ]] || {
        # Move the bad download aside so the next run fetches it again.
        mv -f -- "${destination}" "${destination}.bad"
        echo "SHA256 mismatch: ${destination} (moved to ${destination}.bad)" >&2; exit 2;
    }
    ln -- "${destination}" "${work}/wheels/${name}" 2>/dev/null \
        || cp -- "${destination}" "${work}/wheels/${name}"
done

docker_args=()
if "${NO_CACHE}"; then
    docker_args+=(--no-cache)
fi
DOCKER_BUILDKIT=1 docker build "${docker_args[@]}" \
    --build-context "wheels=${work}/wheels" \
    -f "${SETUP_DIR}/docker/Dockerfile" \
    -t "${RUNTIME_IMAGE}" "${SETUP_DIR}"

if ! "${NO_SMOKE}"; then
    gpu=(--device=/dev/kfd --device=/dev/dri --group-add "$(stat -c '%g' /dev/kfd)")
    [[ ! -e /dev/dri/renderD128 ]] || gpu+=(--group-add "$(stat -c '%g' /dev/dri/renderD128)")
    docker run --rm --pull=never "${gpu[@]}" --ipc=host \
        --read-only --tmpfs /tmp:rw,exec,nosuid,size=2g -e HOME=/tmp \
        -v "${ROOT}:/workspace:ro" -e PYTHONPATH=/workspace/opennav_sam3_inference \
        -w /workspace "${RUNTIME_IMAGE}" \
        python opennav_sam3_setup/docker/smoke_test.py
fi

echo "Binary runtime ready: ${RUNTIME_IMAGE}"
