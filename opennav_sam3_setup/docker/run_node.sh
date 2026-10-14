#!/usr/bin/env bash
# Build the OpenNav SAM3 ROS packages inside the container (first run, or with
# --build) and launch the SAM3 inference node.
#
#   SAM3_MODEL_DIR must contain sam3/ (checkpoint) and onnx_files_504/.
#
# Usage: docker/run_node.sh [--build] [launch args, e.g. image_topic:=/cam/image]
# sensor_processing_pipeline defaults to true, as in the host launch.
#
# The node uses Cyclone DDS. Large camera images need the host UDP buffers
# raised (net.core.rmem_max); see docker/README.md.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

rmem_max="$(sysctl -n net.core.rmem_max 2>/dev/null || echo 0)"
if ((rmem_max < 10000000)); then
    echo "run_node: warning: net.core.rmem_max=${rmem_max}; camera images above ~200 KB may be dropped (see docker/README.md)" >&2
fi

FORCE_BUILD=0
launch_args=()
for arg in "$@"; do
    case "${arg}" in
        --build) FORCE_BUILD=1 ;;
        -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) launch_args+=("${arg}") ;;
    esac
done

# Build with the runtime venv's Python so installed entry points (shebangs)
# can import torch / migraphx. The params yaml is read from the source tree
# (so edits need no rebuild) with its model paths rewritten to /models.
inner='
set -euo pipefail
cd /ws
if [[ "$1" == 1 || ! -f /ws/install/setup.bash ]]; then
    python -m colcon --log-base /ws/log build \
        --base-paths /workspace \
        --packages-select opennav_sam3_msgs opennav_sam3_inference \
        --build-base /ws/build --install-base /ws/install
fi
shift
set +u; source /ws/install/setup.bash; set -u
sed -E \
    -e "s#^( *checkpoint:) *[^ #]+#\1 /models/sam3#" \
    -e "s#^( *onnx_dir:) *[^ #]+#\1 /models/onnx_files_504#" \
    /workspace/opennav_sam3_inference/config/sam3_inference.yaml \
    > /ws/sam3_inference_container.yaml
exec ros2 launch opennav_sam3_inference sam3_inference.launch.py \
    params_file:=/ws/sam3_inference_container.yaml "$@"
'
exec "${SCRIPT_DIR}/run.sh" bash -c "${inner}" _ "${FORCE_BUILD}" "${launch_args[@]}"
