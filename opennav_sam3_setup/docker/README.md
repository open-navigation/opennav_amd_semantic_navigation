# SAM3 Docker runtime (ROS 2 Jazzy / ROCm 10.1 / MIGraphX 2.18 / ORT 1.29)

Built by `../setup.sh` on the official `ros:jazzy-ros-base` image from AMD's
ROCm 10.1 apt packages and checksum-pinned official wheels (Torch 2.14,
MIGraphX 2.18, the ONNX Runtime MIGraphX plugin EP) plus ONNX Runtime 1.29 from
PyPI. Weights and model artifacts stay in `SAM3_MODEL_DIR`.

| Script | Purpose |
|---|---|
| `build.sh` | Build the image and run a GPU check (called by `setup.sh`) |
| `run.sh <cmd>` | Run a command in the image (repo at `/workspace`, `SAM3_MODEL_DIR` at `/models`) |
| `run_node.sh` | Build the ROS packages (first use, or after the image or package sources change) and launch the SAM3 node |

```bash
./docker/run_node.sh [--build] [launch args...]
./docker/run_node.sh image_topic:=/camera/color/image sensor_processing_pipeline:=false
```

`checkpoint` and `onnx_dir` in `sam3_inference.yaml` are mapped to
`/models/sam3` and `/models/onnx_files_504` (504 px only). The container uses
host networking; `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`, `ROS_LOCALHOST_ONLY`
and `CYCLONEDDS_URI` (a host file is mounted) are forwarded.

## Known issues

- **Use Cyclone DDS on the host too.** The image only provides
  `rmw_cyclonedds_cpp`, as used by the reference platform's
  [camera bringup](https://github.com/open-navigation/opennav_amd_demonstrations/blob/main/honeybee_bringup/systemd/robot_bringup.service);
  mixing RMWs is unreliable:

  ```bash
  sudo apt install ros-jazzy-rmw-cyclonedds-cpp
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  ```
- **Large camera images need larger UDP buffers.** With the default
  `net.core.rmem_max` the node starts but receives no frames above a few
  hundred KB (`run_node.sh` warns). Apply the large-topic `sysctl` settings
  from the reference platform's
  [robot computer setup](https://github.com/open-navigation/opennav_amd_demonstrations/blob/main/docs/bootstrapping_robot_computer.md).
- **Keep `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False`** (set in the
  image); `expandable_segments:True` hangs the first GPU model upload with this
  Torch / ROCm build.
- **Rebuild model artifacts after changing the image.** MXR files and ONNX
  Runtime caches only load with the MIGraphX build that produced them; build
  into a new empty `onnx_files_504` folder.
