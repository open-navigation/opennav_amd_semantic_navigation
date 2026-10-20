#!/usr/bin/env bash
# Run a command inside the OpenNav SAM3 ROCm 10.1 container.
#
#   SAM3_MODEL_DIR   host weights + onnx_files_* directory (mounted at /models;
#                    default: opennav_sam3_setup, as in setup.sh)
#   SAM3_DOCKER_IMAGE  image tag (also read by build.sh)
#   SAM3_COLCON_WS   host colcon build/install workspace (mounted at /ws;
#                    default ~/.cache/opennav-sam3/colcon_ws)
#   CYCLONEDDS_URI  Cyclone DDS config: host file (mounted), file:// URI or inline XML
#
# The repository is mounted at /workspace; the working directory is
# opennav_sam3_setup, so `./docker/run.sh python export/build.py ...` works.
set -euo pipefail

SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd -- "${SETUP_DIR}/.." && pwd)"
IMAGE="${SAM3_DOCKER_IMAGE:-opennav-sam3-gpu101-ort129-mgx218-gfx1151:0.4.0-rc1}"
MODEL_DIR="${SAM3_MODEL_DIR:-${SETUP_DIR}}"

if [[ ! -e /dev/kfd || ! -d /dev/dri ]]; then
    echo "ROCm devices /dev/kfd and /dev/dri are required" >&2
    exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "SAM3_MODEL_DIR must be an existing weights/artifact directory" >&2
    exit 1
fi
MODEL_DIR="$(cd -- "${MODEL_DIR}" && pwd)"

COLCON_WS="${SAM3_COLCON_WS:-${XDG_CACHE_HOME:-${HOME}/.cache}/opennav-sam3/colcon_ws}"
mkdir -p "${COLCON_WS}"

gpu_args=(--device=/dev/kfd --device=/dev/dri --group-add "$(stat -c '%g' /dev/kfd)")
if [[ -e /dev/dri/renderD128 ]]; then
    gpu_args+=(--group-add "$(stat -c '%g' /dev/dri/renderD128)")
fi

mount_args=(
    -v "${ROOT}:/workspace"
    -v "${MODEL_DIR}:/models"
    -v "${COLCON_WS}:/ws"
)

# Docker does not follow absolute symlinks inside a bind mount (for example
# weights or onnx_files_* linked from another disk). Bind each link's resolved
# target at the path the link points to; files are read-only, directories not.
add_link_mount() {
    local link="$1" depth="$2" raw resolved inner mode=""
    raw="$(readlink "${link}")"
    [[ "${raw}" = /* ]] || return 0
    resolved="$(readlink -f "${link}" 2>/dev/null || true)"
    if [[ ! -e "${resolved}" ]]; then
        echo "run.sh: warning: ${link} -> ${raw} does not resolve" >&2
        return 0
    fi
    [[ -f "${resolved}" ]] && mode=":ro"
    mount_args+=(-v "${resolved}:${raw}${mode}")
    if [[ -d "${resolved}" && "${depth}" -lt 2 ]]; then
        while IFS= read -r -d '' inner; do
            add_link_mount "${inner}" $((depth + 1))
        done < <(find "${resolved}" -maxdepth 1 -type l -print0)
    fi
}
while IFS= read -r -d '' link; do
    add_link_mount "${link}" 0
done < <(find "${MODEL_DIR}" -maxdepth 3 -type l -print0)

# Forward ROS 2 discovery settings when set on the host. The image uses Cyclone
# DDS (rmw_cyclonedds_cpp) only, like the Honeybee reference platform.
if [[ -n "${RMW_IMPLEMENTATION:-}" && "${RMW_IMPLEMENTATION}" != rmw_cyclonedds_cpp ]]; then
    echo "run.sh: the image only provides rmw_cyclonedds_cpp (RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION})" >&2
    exit 2
fi
ros_env_args=()
for name in ROS_DOMAIN_ID RMW_IMPLEMENTATION ROS_LOCALHOST_ONLY \
            ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS; do
    if [[ -n "${!name:-}" ]]; then
        ros_env_args+=(-e "${name}=${!name}")
    fi
done
# CYCLONEDDS_URI is inline XML, file://<path> or a plain path. A file is a host
# file: mount it and point the variable at the in-container path.
if [[ -n "${CYCLONEDDS_URI:-}" ]]; then
    cyclone_file=""
    case "${CYCLONEDDS_URI}" in
        file://*) cyclone_file="${CYCLONEDDS_URI#file://}" ;;
        /*) cyclone_file="${CYCLONEDDS_URI}" ;;
    esac
    if [[ -n "${cyclone_file}" ]]; then
        if [[ ! -f "${cyclone_file}" ]]; then
            echo "run.sh: CYCLONEDDS_URI is not a file on the host: ${cyclone_file}" >&2
            exit 2
        fi
        mount_args+=(-v "$(readlink -f -- "${cyclone_file}"):/ros_dds/cyclonedds.xml:ro")
        ros_env_args+=(-e CYCLONEDDS_URI=file:///ros_dds/cyclonedds.xml)
    else
        ros_env_args+=(-e "CYCLONEDDS_URI=${CYCLONEDDS_URI}")
    fi
fi

tty_args=(-i)
if [[ -t 0 && -t 1 ]]; then
    tty_args+=(-t)
fi
if [[ $# -eq 0 ]]; then
    set -- bash
fi

CONTAINER_USER="$(id -un 2>/dev/null || printf 'opennav')"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE}" 2>/dev/null || true)"
exec docker run --pull=never --rm "${tty_args[@]}" \
    --network host --ipc=host \
    "${gpu_args[@]}" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e USER="${CONTAINER_USER}" \
    -e LOGNAME="${CONTAINER_USER}" \
    -e HF_HUB_OFFLINE=1 \
    -e SAM3_MODEL_DIR=/models \
    -e SAM3_DOCKER_IMAGE_ID="${IMAGE_ID}" \
    -e PYTHONPATH=/workspace/opennav_sam3_inference \
    "${ros_env_args[@]}" \
    "${mount_args[@]}" \
    -w /workspace/opennav_sam3_setup \
    "${IMAGE}" "$@"
